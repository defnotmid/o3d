package o3d

import "base:runtime"
import "core:container/queue"
import "core:math"
import "core:math/linalg"
import "core:slice"
import "core:sync"
import "core:thread"

import "deps:oksl"

// The CPU renderer: the pixel reference for the GPU's and the path without
// one. Four samples per pixel (the GPU path's 4x MSAA), a depth per sample,
// shading once per pixel per triangle at its centre with perspective-correct
// values, near-plane clipping. Samples hold light in half floats,
// premultiplied by coverage, as the GPU's frame target does: blended there,
// a pixel's averaged, then tone mapped and sRGB-encoded once (three.js's
// EffectComposer with an OutputPass).
//
// A frame (and before it, the shadow map, shadow.odin: depth only, one
// sample per texel) is three phases, each spread over the caller's pool (nil: run on
// the caller): chunks of the drawing order (up to CHUNK_TRIS triangles)
// run the vertex stage over the vertices they use, clip and keep screen
// triangles with their corners' values, and count them per band of rows;
// once the counts are summed, each chunk lists its triangles in the bands,
// so a band's list keeps the drawing order; then each band clears its rows,
// draws its list and resolves its pixels.
//
// A triangle's corners carry floats interpolated across it: for the
// standard material its world position and normal, for a ShaderMaterial
// its varyings (run by oksl: the vertex program over the chunk's vertices,
// the fragment program over 2x2 quads of a run of the draw's triangles,
// then the same coverage, depth and blending as the standard material).

SAMPLES :: 4
// Sample positions in a pixel (the standard 4x MSAA pattern).
SAMPLE_OFFSETS := [SAMPLES]Vec2{{0.375, 0.125}, {0.875, 0.375}, {0.125, 0.625}, {0.625, 0.875}}
// The samples of a pixel a triangle covers.
Sample_Mask :: bit_set[0 ..< SAMPLES; u8]
BAND_ROWS :: 32
CHUNK_TRIS :: 8192
// Quads shaded by one oksl call (a run of one draw's triangles).
QUAD_BATCH :: 256
Software :: struct {
	width, height: int,
	light:         [][4]f16, // per sample, premultiplied (alpha: coverage)
	depth:         []f32, // per sample, 1 near .. 0 far (reversed)
	pixels:        []u8, // the resolved frame, RGBA8 premultiplied: the caller's `out` or `own`
	own:           []u8, // the frame when the caller gives none
	// Scratch per frame (kept, with the chunks' and workers' arrays, to
	// reuse).
	order:         [dynamic]int,
	materials:     [dynamic]Shading_Material, // per draw
	// o3d's own draws' variants (shading.odin); per draw its variant through
	// the camera and through the mirror (-1: not drawn so) and its
	// material's block; each pass's frame block.
	programs:           Programs,
	program_ids:        [dynamic]i32,
	mirror_program_ids: [dynamic]i32,
	// Every draw's in a shadow map's pass, o3d's draws' in the frame's
	// depth pass (depth_key).
	shadow_program_ids:   [dynamic]i32,
	velocity_program_ids: [dynamic]i32,
	material_blocks:    [dynamic]Material_Block,
	frame_block:        Frame_Block,
	mirror_frame_block: Frame_Block,
	// The frame's joint table (joint_table) and each draw's place in it.
	joint_table:        [dynamic]Mat4,
	// The fullscreen passes' light: the background's (and the depth image's),
	// the depth pass's far plane's velocity and nearness; and the camera the
	// last frame was drawn through.
	bg_light:           [dynamic][4]f32,
	// The post passes' last light (sRGB-encoded) and their scratch, one
	// per worker (post.odin).
	post_light:         [dynamic][4]f32,
	post_scratch:       [dynamic]oksl.Scratch,
	far_light:          [2][dynamic][4]f32,
	camera:             Camera,
	joint_bases:        [dynamic][2]u32,
	// ShaderMaterials' uniform blocks, through the camera and through the
	// mirror, and for the frame's depth pass (`depth_blocks`).
	blocks:        Shader_Blocks,
	mirror_blocks: Shader_Blocks,
	depth_blocks:  Shader_Blocks,
	// The frame's chunks are chunks[:chunk_count]; the ones past it keep
	// their lists' memory for later frames.
	chunks:        [dynamic]Chunk,
	chunk_count:   int,
	// Per chunk and band (chunk * bands + band): the chunk's triangles in
	// the band, then where it lists them in band_tris.
	chunk_bands:   [dynamic]int,
	// Band b's triangles are band_tris[band_start[b]:band_start[b + 1]].
	band_start:    [dynamic]int,
	band_tris:     [dynamic]Tri_Ref,
	// One per thread that runs a phase (the caller is 0).
	workers:       [dynamic]Worker,
	// Each shadow light's depth per texel (shadow.size squared), drawn first.
	shadow_maps:   [MAX_SHADOWS][]f32,
	// The frame's depth per pixel (`Scene.depth`), drawn after them.
	depth_map:     [dynamic]f32,
	expanded:      Expanded_Draws,
	// Per draw: its triplanar projection frame (set for `.Triplanar` ones).
	frames:        [dynamic]Projection_Frame,
	// The host's pixels for external textures (by id - 1), as the GPU's
	// `texture_set_external` binds its textures: `software_set_external`.
	externals:     [dynamic]Software_External,
	// The frame drawn before its transmissive draws, for them to see
	// through: its light resolved by each band into level 0 of
	// `snapshot_tex` (texture 1), mipmapped as the GPU's, then its levels
	// read as floats (`snapshot_light`).
	snapshot:      Textures,
	snapshot_tex:  Texture,
	snapshot_light: [dynamic][4]f32,
	// The scene mirrored in its reflector (`mirror_view`), drawn first in
	// the same samples at its size, with each sample's run (o3d.frag's
	// `mirror_run_of`) premultiplied by its coverage, and the coverage,
	// blended as the light (`mirror_runs`): its light resolved into level 0 of `reflection_tex`
	// (texture 1), its runs into the same levels of texture 2 (red and
	// green, alpha 1, as the GPU's second layer), both mipmapped as the
	// GPU's, then their levels read as floats (`reflection_light`,
	// `reflection_runs`: converted once, read a dozen texels a pixel) by the
	// draws that show it.
	reflection:    Textures,
	reflection_tex: Texture,
	reflection_light: [dynamic][4]f32,
	reflection_runs: [dynamic][4]f32,
	mirror_runs:   [dynamic][2]f16,
	// With depth of field or motion blur: the frame's light per pixel,
	// resolved by the bands, and the gather's tables (depth_of_field.odin).
	frame:         [dynamic][4]f32,
	focus:         Focus_Tables,
	bloom:         Bloom_Tables,
	// With motion blur: each pixel's velocity and nearness from the depth
	// pass, the light blurred along them, and the filter's tables
	// (motion_blur.odin).
	velocity:      [dynamic][2]f32,
	nearness:      [dynamic]f32,
	blurred:       [dynamic][4]f32,
	motion:        Motion_Tables,
}

// An external texture's pixels for the software renderer, rows from the
// top, the host's: RGBA8 (sRGB bytes, as an image's), or the same values as
// half floats (`halves`, unclamped).
Software_External :: struct {
	pixels:        [][4]u8,
	halves:        [][4]f16,
	width, height: int,
}

software_destroy :: proc(s: ^Software) {
	delete(s.light)
	delete(s.depth)
	delete(s.own)
	textures_destroy(&s.snapshot)
	delete(s.snapshot_light)
	textures_destroy(&s.reflection)
	delete(s.reflection_light)
	delete(s.reflection_runs)
	delete(s.mirror_runs)
	delete(s.frame)
	focus_tables_destroy(&s.focus)
	bloom_tables_destroy(&s.bloom)
	delete(s.velocity)
	delete(s.nearness)
	delete(s.blurred)
	motion_tables_destroy(&s.motion)
	delete(s.order)
	delete(s.materials)
	programs_destroy(&s.programs)
	delete(s.program_ids)
	delete(s.mirror_program_ids)
	delete(s.shadow_program_ids)
	delete(s.velocity_program_ids)
	delete(s.material_blocks)
	delete(s.joint_table)
	delete(s.bg_light)
	delete(s.post_light)
	for &sc in s.post_scratch {
		oksl.scratch_destroy(&sc)
	}
	delete(s.post_scratch)
	delete(s.far_light[0])
	delete(s.far_light[1])
	delete(s.joint_bases)
	shader_blocks_destroy(&s.blocks)
	shader_blocks_destroy(&s.mirror_blocks)
	shader_blocks_destroy(&s.depth_blocks)
	for &c in s.chunks {
		delete(c.tris)
		delete(c.values)
	}
	delete(s.chunks)
	delete(s.chunk_bands)
	delete(s.band_start)
	delete(s.band_tris)
	for &w in s.workers {
		worker_destroy(&w)
	}
	delete(s.workers)
	for &sm in s.shadow_maps {
		delete(sm)
	}
	delete(s.depth_map)
	delete(s.frames)
	expanded_draws_destroy(&s.expanded)
	delete(s.externals)
	s^ = {}
}

// Binds the host's pixels (`width x height` RGBA8, rows from the top, as
// `texture_add` takes them; with `halves`, RGBA16F: four little-endian
// half floats a pixel holding the same values, unclamped, as an RGBA16F GPU
// texture does) to an external texture (`texture_add_external`) for the
// software renders that follow, until bound again: what
// `o3d_sdlgpu.texture_set_external` binds on the GPU, read as it reads an
// unmipped one (bilinear at its level 0). The host keeps them alive while
// bound; nil (or too few bytes) unbinds it, then sampled white.
software_set_external :: proc(s: ^Software, id: Texture_Id, pixels: []u8, width, height: int, halves := false) {
	if id == 0 {
		return
	}
	if int(id) > len(s.externals) {
		resize(&s.externals, int(id))
	}
	size := width * height * (8 if halves else 4)
	ok := width > 0 && height > 0 && len(pixels) >= size
	switch {
	case !ok:
		s.externals[id - 1] = {}
	case halves:
		s.externals[id - 1] = {halves = slice.reinterpret([][4]f16, pixels[:size]), width = width, height = height}
	case:
		s.externals[id - 1] = {pixels = slice.reinterpret([][4]u8, pixels[:size]), width = width, height = height}
	}
}

// A triangle ready to draw: screen positions (x, y, depth) with 1/w,
// where its corners' floats are in its chunk's `values` (per corner, the
// draw's floats, divided by w unless flat), and an exact shape's quad's
// entry (`curve_quad_entry`).
@(private)
Tri :: struct {
	p:            [3]Vec3,
	inv_w:        [3]f32,
	values:       u32,
	draw:         int,
	min_y, max_y: int,
	front:        bool, // facing the camera (before back faces were turned)
}

// A run of one draw's triangles (indices[first:][:count * 3]) and the
// screen triangles made from them; a structure of arrays, since binning
// reads only the row spans.
@(private)
Chunk :: struct {
	order:        int, // the draw's place in `order`
	first, count: int,
	tris:         #soa[dynamic]Tri,
	values:       [dynamic]f32,
}

@(private)
Tri_Ref :: struct {
	chunk, tri: u32,
}

// A thread's working memory: the vertex stage's outputs over a chunk's
// vertices, and a batch of quads for a ShaderMaterial.
@(private)
Worker :: struct {
	vm:        oksl.Scratch,
	positions: [dynamic][4]f32, // clip space
	values:    [dynamic]f32,
	skinned:   [dynamic]Vertex, // posed chunk vertices for a skinned ShaderMaterial
	opened:    [dynamic]Vertex, // and posed as at the shutter's opening (the depth pass's twins)
	quads:     Quads,
}

// A batch of one draw's 2x2 quads for its fragment program: each quad's
// top-left pixel, helper lanes, facing, its lanes' covered samples and its
// triangle; per lane (structure of arrays, as run_fragment reads them) its
// varyings and depth, its outputs (a colour pass's light and the mirror's
// run; a depth pass's depth, velocity and nearness) and covered samples
// (gl_SampleMask), and the lanes each quad shaded.
@(private)
Quads :: struct {
	origin:  [dynamic][2]i32,
	helpers: [dynamic]oksl.Quad_Lanes,
	facing:  [dynamic]bool,
	covered: [dynamic][4]Sample_Mask,
	tri:     [dynamic]Tri_Ref,
	values:  [dynamic]f32,
	depths:  [dynamic]f32,
	targets: [3][dynamic][4]f32,
	masks:   [dynamic]u32,
	shaded:  [dynamic]oksl.Quad_Lanes,
}


@(private = "file")
worker_destroy :: proc(w: ^Worker) {
	oksl.scratch_destroy(&w.vm)
	delete(w.positions)
	delete(w.values)
	delete(w.skinned)
	delete(w.opened)
	q := &w.quads
	delete(q.origin)
	delete(q.helpers)
	delete(q.facing)
	delete(q.covered)
	delete(q.tri)
	delete(q.values)
	delete(q.depths)
	for &t in q.targets {
		delete(t)
	}
	delete(q.masks)
	delete(q.shaded)
}

// Draws `scene` into `width x height` premultiplied RGBA8, rows from the
// top: into `out` when it holds them (the caller's image: no copy), else
// into the renderer's own (returned, borrowed until the next render).
// `shaders` holds the ShaderMaterials' shaders (nil: none); `textures`
// holds the material textures (nil: none).
software_render :: proc(s: ^Software, scene: ^Scene, geos: ^Geometries, shaders: ^Shaders, width, height: int, pool: ^thread.Pool = nil, textures: ^Textures = nil, out: []u8 = nil) -> []u8 {
	if s.width != width || s.height != height {
		delete(s.light)
		delete(s.depth)
		s.width, s.height = width, height
		s.light = make([][4]f16, width * height * SAMPLES)
		s.depth = make([]f32, width * height * SAMPLES)
	}
	if len(out) >= width * height * 4 {
		s.pixels = out[:width * height * 4]
	} else {
		if len(s.own) != width * height * 4 {
			delete(s.own)
			s.own = make([]u8, width * height * 4)
		}
		s.pixels = s.own
	}
	view, proj := camera_matrices(scene.camera, width, height)
	active := expand_draws(&s.expanded, scene, geos, view, proj, width, height)
	lights: [MAX_LIGHTS]Shading_Light
	nl := shading_lights(active.lights[:], &lights)
	shading_materials(scene_materials(active), &s.materials, textures)
	transmission_scale(s.materials[:len(active.draws)], active.draws.model[:len(active.draws)])
	sprite_materials(active, geos, &s.expanded, s.materials[:])
	// The mirror's view first: its receivers count for the shadows' fit.
	mv, reflect := mirror_view(active, s.materials[:], geos, view, proj, width, height, &s.expanded)
	views := [2]Mat4{proj * view, proj * mv.view}
	shadows: [MAX_SHADOWS]Shadow
	ns := frame_shadows(active, geos, shaders, views[:1 + int(reflect)], &shadows, &s.expanded)
	shader_blocks(active, shaders, view, proj, shadows[:ns], &s.blocks)
	table_grow_non_zero(&s.frames, len(active.draws))
	for m, i in s.materials[:len(active.draws)] {
		if .Triplanar in m.flags {
			s.frames[i] = projection_frame(active.draws[i].material.projection, active.draws[i].model)
		}
	}
	threads := 0
	if pool != nil {
		sync.guard(&pool.mutex)
		threads = len(pool.threads)
	}
	if len(s.workers) < threads + 1 {
		table_grow(&s.workers, threads + 1)
	}
	env := shading_environment(active, textures)
	bg_tex, bg_intensity, bg_lod := shading_background(active, textures)
	s.camera = active.camera
	// The fullscreen passes the bands run, compiled here (the bands only read
	// them).
	if bg_tex != 0 {
		table_grow_non_zero(&s.bg_light, width * height)
		pass_get(&s.programs, .Background)
	}
	if motion_of(active) {
		pass_get(&s.programs, .Velocity_Far)
	}
	job := Frame_Job {
		s             = s,
		blocks        = &s.blocks,
		scene         = active,
		geos          = geos,
		shaders       = shaders,
		textures      = textures,
		lights        = lights[:nl],
		env           = env,
		bg_tex        = bg_tex,
		bg_intensity  = bg_intensity,
		bg_lod        = bg_lod,
		clear         = to_halves(background_clear_light(active)),
		eye           = active.camera.position,
		ortho         = active.camera.orthographic,
		shadows       = shadows,
		shadow_count  = ns,
		tone_mapping  = active.tone_mapping,
		tone_exposure = active.tone_mapping_exposure,
		fog           = active.fog,
		fog_color     = linear(active.fog.color),
	}
	// Each light's shadow by the frame's shadows (as the GPU's).
	for si in 0 ..< ns {
		for &l, li in lights[:nl] {
			if li == shadows[si].light {
				l.shadowed = u32(si + 1)
			}
		}
	}
	// o3d's own draws: each one's variant through the camera and the
	// mirror, its material's block, and the passes' frame blocks.
	{
		nd := len(active.draws)
		rad, _ := texture_get(textures, env.radiance)
		shape := frame_shape(lights[:nl], ns, env, rad.hdr, active.fog.kind)
		joint_table(active, motion_of(active), &s.joint_table, &s.joint_bases)
		table_grow_non_zero(&s.program_ids, nd)
		table_grow_non_zero(&s.mirror_program_ids, nd)
		table_grow_non_zero(&s.material_blocks, nd)
		table_grow_non_zero(&s.shadow_program_ids, nd)
		table_grow_non_zero(&s.velocity_program_ids, nd)
		depth := needs_depth(active) || motion_of(active)
		for &m, i in s.materials[:nd] {
			s.program_ids[i], s.mirror_program_ids[i], s.shadow_program_ids[i], s.velocity_program_ids[i] = -1, -1, -1, -1
			d := &active.draws[i]
			geo, has_geo := geometry_get_any(geos, &s.expanded, d.geometry)
			if !has_geo {
				continue
			}
			skinned := len(geometry_skin(geos, d.geometry)) > 0 && len(scene_draw_joints(active, i)) > 0
			if ns > 0 {
				s.shadow_program_ids[i] = program_get(&s.programs, depth_key(&m, skinned, false))
			}
			if m.shader != 0 {
				s.material_blocks[i] = material_block(&m, {}, geo.curve_table)
				continue
			}
			if depth {
				s.velocity_program_ids[i] = program_get(&s.programs, depth_key(&m, skinned, true))
			}
			s.program_ids[i] = program_get(&s.programs, program_key(&m, skinned, false, shape))
			if reflect {
				s.mirror_program_ids[i] = program_get(&s.programs, program_key(&m, skinned, true, shape))
			}
			s.material_blocks[i] = material_block(&m, s.frames[i], geo.sprite_table if geo.mode == .Sprites else geo.curve_table)
		}
		frame_block(&s.frame_block, active.camera, active.camera.position, proj * view, width, height, lights[:nl], shadows[:ns], env, active.fog)
		if reflect {
			s.frame_block.reflect = {f32(mv.levels - 1), mv.focal, 0, 0}
			frame_block(&s.mirror_frame_block, active.camera, mv.eye, proj * mv.view, mv.width, mv.height, lights[:nl], shadows[:ns], env, active.fog)
			s.mirror_frame_block.clip_plane = mv.plane
		}
		job.programs, job.frame_block = s.program_ids[:nd], &s.frame_block
	}
	for si in 0 ..< ns {
		sh := &job.shadows[si]
		size := sh.size
		if len(s.shadow_maps[si]) != size * size {
			delete(s.shadow_maps[si])
			s.shadow_maps[si] = make([]f32, size * size)
		}
		shadow_casters(active, geos, shaders, sh, &s.order, &s.expanded)
		job.shadow_index = si
		job.programs = s.shadow_program_ids[:len(active.draws)]
		job.view_proj, job.width, job.height, job.depth_only = sh.view_proj, size, size, true
		job.depth_target, job.own_side = s.shadow_maps[si], false
		run_pass(s, active, geos, &job, pool, draw_depth_band, 0, len(s.order))
	}
	// The frame's depth, each ShaderMaterial by its own vertex stage, with
	// motion blur its velocity and nearness too (where each pixel was at
	// the shutter's opening).
	moving := motion_of(active)
	if needs_depth(active) || moving {
		table_grow_non_zero(&s.depth_map, width * height)
		depth_draws(active, geos, shaders, proj * view, &s.order, &s.expanded)
		opened := opened_camera(active.camera) or_else active.camera
		ov, op := camera_matrices(opened, width, height)
		depth_blocks(active, shaders, view, proj, ov, op, width, height, shadows[:ns], &s.depth_blocks)
		job.view_proj, job.width, job.height, job.depth_only = proj * view, width, height, true
		job.depth_target, job.own_side, job.depth_blocks = s.depth_map[:width * height], true, &s.depth_blocks
		job.programs = s.velocity_program_ids[:len(active.draws)]
		if moving {
			table_grow_non_zero(&s.velocity, width * height)
			table_grow_non_zero(&s.nearness, width * height)
			table_grow_non_zero(&s.far_light[0], width * height)
			table_grow_non_zero(&s.far_light[1], width * height)
			job.velocity, job.nearness, job.camera = s.velocity[:width * height], s.nearness[:width * height], active.camera
			job.opened_view_proj = op * ov
			job.opened_far = job.opened_view_proj * linalg.inverse(proj * view)
		}
		run_pass(s, active, geos, &job, pool, draw_depth_band, 0, len(s.order))
		job.velocity, job.nearness = nil, nil
	} else {
		clear(&s.depth_map)
	}
	job.programs = s.program_ids[:len(active.draws)]
	if reflect {
		// The mirror's light in the frame's samples at its size: its pass
		// with the mirrored camera, front faces turned, what is behind the
		// plane clipped, cleared to nothing (the environment shows there).
		mirror_order(active, geos, shaders, s.materials[:], mv.view, proj, &s.order, &s.expanded)
		shader_blocks(active, shaders, mv.view, proj, shadows[:ns], &s.mirror_blocks, mv.eye)
		clear(&s.reflection.list)
		clear(&s.reflection.hdr_texels)
		s.reflection_tex = texture_grow(&s.reflection.hdr_texels, mv.width, mv.height)
		s.reflection_tex.hdr = true
		s.reflection_tex.levels = u32(mv.levels)
		runs_tex := texture_grow(&s.reflection.hdr_texels, mv.width, mv.height)
		runs_tex.hdr = true
		runs_tex.levels = u32(mv.levels)
		table_grow_non_zero(&s.mirror_runs, mv.width * mv.height * SAMPLES)
		mj := job
		mj.view_proj, mj.width, mj.height, mj.depth_only = proj * mv.view, mv.width, mv.height, false
		mj.eye, mj.mirrored, mj.clip, mj.blocks = mv.eye, true, mv.plane, &s.mirror_blocks
		mj.programs, mj.frame_block = s.mirror_program_ids[:len(active.draws)], &s.mirror_frame_block
		mj.bg_tex, mj.clear = 0, {}
		mj.capture_into = s.reflection.hdr_texels[s.reflection_tex.mips[0]:][:mv.width * mv.height]
		mj.runs = s.mirror_runs[:mv.width * mv.height * SAMPLES]
		mj.capture_runs = s.reflection.hdr_texels[runs_tex.mips[0]:][:mv.width * mv.height]
		run_pass(s, active, geos, &mj, pool, draw_band, 0, len(s.order))
		texture_finish(&s.reflection, &s.reflection_tex, blit = true)
		texture_finish(&s.reflection, &runs_tex, blit = true)
		job.reflection, job.reflection_tex = light_levels(&s.reflection, &s.reflection_tex, &s.reflection_light), s.reflection_tex
		job.reflection_runs = light_levels(&s.reflection, &runs_tex, &s.reflection_runs)
		job.reflection_last, job.reflection_focal = f32(mv.levels - 1), mv.focal
	}
	snapshot := draw_order(active, geos, shaders, view, proj, &s.order, &s.expanded)
	job.view_proj, job.width, job.height, job.depth_only = proj * view, width, height, false
	if bg_tex != 0 {
		job.inv_vp = linalg.inverse(job.view_proj)
	}
	if snapshot < len(s.order) {
		// The frame so far, its light resolved and mipmapped, for the
		// transmissive draws after it.
		clear(&s.snapshot.list)
		clear(&s.snapshot.hdr_texels)
		s.snapshot_tex = texture_grow(&s.snapshot.hdr_texels, width, height)
		s.snapshot_tex.hdr = true
		job.capture_into = s.snapshot.hdr_texels[s.snapshot_tex.mips[0]:][:width * height]
		run_pass(s, active, geos, &job, pool, draw_band, 0, snapshot)
		job.capture_into = nil
		// Mipmapped as the GPU's snapshot is (three.js's too).
		texture_finish(&s.snapshot, &s.snapshot_tex, blit = true)
		job.snapshot, job.snapshot_tex = light_levels(&s.snapshot, &s.snapshot_tex, &s.snapshot_light), s.snapshot_tex
		job.keep = true
	}
	// The frame's light, resolved by the bands, then its post passes as the
	// GPU's: blurred along its motion, its bloom, then through its depth of
	// field or straight to the output (tone mapped and encoded).
	table_grow_non_zero(&s.frame, width * height)
	job.light_into = s.frame[:width * height]
	run_pass(s, active, geos, &job, pool, draw_band, snapshot if job.keep else 0, len(s.order))
	frame := s.frame[:width * height]
	if moving {
		table_grow_non_zero(&s.blurred, width * height)
		motion_blur(&s.motion, frame, s.velocity[:], s.nearness[:], width, height, motion_params(active.camera, height), s.blurred[:width * height], pool)
		frame = s.blurred[:width * height]
	}
	bloom: Bloom_Output
	if chain, bloomed := bloom_of(active.bloom, width, height); bloomed {
		bloom = bloom_run(&s.bloom, chain, frame, width, height, &s.programs, &s.post_scratch, pool)
	}
	if focus, blurred := focus_of(active.camera, height); blurred {
		depth_of_field(s, frame, focus, active.camera, active.tone_mapping, active.tone_mapping_exposure, bloom, pool)
	} else {
		exposure := active.tone_mapping_exposure
		block := Post_Block {
			tone  = {f32(active.tone_mapping), exposure if exposure > 0 else 1, f32(width), f32(height)},
			frame = {f32(width), f32(height), 0, 0},
			bloom = {0, 0, bloom.gain, 0},
		}
		table_grow_non_zero(&s.post_light, width * height)
		maps := #partial Post_Images{.Light = oksl.channel_image(frame, width, height), .Bloom = oksl.channel_image(bloom.glow, bloom.width, bloom.height)}
		out := [1][][4]f32{s.post_light[:width * height]}
		post_run(&s.programs, &s.post_scratch, .Output, &block, maps, out[:], width, height, pool)
		post_pixels(s.post_light[:width * height], s.pixels)
	}
	return s.pixels
}


// A mipmapped texture's levels (from its first texel, as `tex` lays them
// out) as floats, into `out`: a bicubic or trilinear read takes 16 or 8
// texels, so their half floats are converted once.
@(private = "file")
light_levels :: proc(t: ^Textures, tex: ^Texture, out: ^[dynamic][4]f32) -> [][4]f32 {
	last := tex.levels - 1
	used := int(tex.mips[last] - tex.first_texel) + int(max(tex.width >> last, 1) * max(tex.height >> last, 1))
	table_grow_non_zero(out, used)
	for c, i in t.hdr_texels[tex.first_texel:][:used] {
		out[i] = from_halves(c)
	}
	return out[:used]
}

// The frame's depth per pixel from the last `software_render` of a scene
// with `Scene.depth` (rows from the top; empty without).
software_depth :: proc "contextless" (s: ^Software) -> []f32 {
	return s.depth_map[:]
}

// The phases over the draws `s.order[first:last]` into a `job.width x
// job.height` target, its bands drawn by `draw`.
@(private = "file")
run_pass :: proc(s: ^Software, scene: ^Scene, geos: ^Geometries, job: ^Frame_Job, pool: ^thread.Pool, draw: proc(data: rawptr, band, worker: int), first, last: int) {
	make_chunks(s, scene, geos, first, last)
	job.bands = (job.height + BAND_ROWS - 1) / BAND_ROWS
	table_grow_non_zero(&s.chunk_bands, s.chunk_count * job.bands)
	parallel(pool, s.chunk_count, job, transform_chunk)
	sum_bands(s, job.bands)
	parallel(pool, s.chunk_count, job, list_chunk)
	parallel(pool, job.bands, job, draw)
}

// Splits the drawing order into chunks of at most CHUNK_TRIS triangles.
@(private = "file")
make_chunks :: proc(s: ^Software, scene: ^Scene, geos: ^Geometries, from, to: int) {
	n := 0
	for oi in from ..< to {
		di := s.order[oi]
		geo, _ := geometry_get_any(geos, &s.expanded, scene.draws[di].geometry)
		tris := int(geo.index_count / 3)
		for first := 0; first < tris; first += CHUNK_TRIS {
			if n == len(s.chunks) {
				append(&s.chunks, Chunk{})
			}
			c := &s.chunks[n]
			c.order, c.first, c.count = oi, first, min(CHUNK_TRIS, tris - first)
			n += 1
		}
	}
	s.chunk_count = n
}

@(private)
Frame_Job :: struct {
	s:             ^Software,
	blocks:        ^Shader_Blocks, // ShaderMaterials' uniforms for the pass's camera
	depth_blocks:  ^Shader_Blocks, // and for the frame's depth pass
	scene:         ^Scene,
	geos:          ^Geometries,
	shaders:       ^Shaders,
	textures:      ^Textures,
	lights:        []Shading_Light,
	env:           Shading_Environment,
	bg_tex:        Texture_Id,
	bg_intensity:  f32,
	bg_lod:        f32,
	clear:         [4]f16, // a sample's light where nothing is: background_clear_light
	inv_vp:        Mat4,
	eye:           Vec3,
	shadows:       [MAX_SHADOWS]Shadow,
	shadow_count:  int,
	shadow_index:  int,
	tone_mapping:  Tone_Mapping,
	tone_exposure: f32,
	fog:           Fog,
	fog_color:     Vec3,
	// The pass: its camera and target, and whether it draws only depth
	// (into `depth_target`: positions only; a shadow map's casters with
	// their other side, the frame's depth with their own).
	view_proj:     Mat4,
	width, height: int,
	depth_only:    bool,
	depth_target:  []f32,
	own_side:      bool,
	// The frame's depth pass with motion blur: each pixel's velocity (its
	// place now less where it was at the shutter's opening, in pixels) and
	// nearness (motion_blur's), the camera then, and where the far plane's
	// points were through it from each pixel's now (`opened_far`, nearness
	// `MOTION_FAR`).
	velocity:      [][2]f32,
	nearness:      []f32,
	camera:        Camera,
	opened_view_proj: Mat4,
	opened_far:    Mat4,
	bands:         int,
	// Over what the pass before drew (no clear, no background).
	keep:          bool,
	// The pass's pixels kept as light, into a level 0 at its size (the
	// snapshot's, the reflection's), or the frame's for its depth of field
	// (`light_into`), not tone mapped into its pixels.
	capture_into:  [][4]f16,
	light_into:    [][4]f32,
	// The mirror's pass: its samples' runs and coverage
	// (Software.mirror_runs), resolved into `capture_runs`.
	runs:          [][2]f16,
	capture_runs:  [][4]f16,
	// The frame's snapshot for the transmissive draws (nil: none): its
	// levels' light, laid out as `snapshot_tex`.
	snapshot:      [][4]f32,
	snapshot_tex:  Texture,
	// The mirror's pass (`mirror_view`): front faces turn, transmissive
	// draws see nothing behind them, what is behind `clip` is left out (a
	// zero plane: nothing), and the frame's pixels are not written.
	mirrored:      bool,
	clip:          Vec4,
	// The frame's reflection for the draws that show it (nil: none): its
	// levels' light and runs, laid out as `reflection_tex`, its last level,
	// and its scale (Mirror_View.focal) through an orthographic camera or
	// not.
	reflection:    [][4]f32,
	reflection_runs: [][4]f32,
	reflection_tex: Texture,
	reflection_last: f32,
	reflection_focal: f32,
	ortho:         bool,
	// The pass's variant per draw (Software.program_ids or
	// mirror_program_ids) and its frame block.
	programs:      []i32,
	frame_block:   ^Frame_Block,
}

@(private = "file")
band_range :: #force_inline proc "contextless" (min_y, max_y: int) -> (first, last: int) {
	return min_y / BAND_ROWS, (max_y - 1) / BAND_ROWS
}

// A triangle corner on its way to the screen: clip position and floats.
@(private = "file")
Corner :: struct {
	c: Vec4,
	v: [MAX_VARYING_FLOATS]f32,
}

// Phase 1: a chunk's vertices through the vertex stage, its triangles
// clipped and kept as screen triangles, then counted per band.
@(private = "file")
transform_chunk :: proc(data: rawptr, ci, wi: int) {
	job := (^Frame_Job)(data)
	s := job.s
	w := &s.workers[wi]
	c := &s.chunks[ci]
	clear(&c.tris)
	clear(&c.values)
	di := s.order[c.order]
	d := &job.scene.draws[di]
	m := &s.materials[di]
	geo, _ := geometry_get_any(job.geos, &s.expanded, d.geometry)
	vertices, all, _ := geometry_mesh_any(job.geos, &s.expanded, d.geometry)
	skin := geometry_skin(job.geos, d.geometry)
	joints := scene_draw_joints(job.scene, di)
	skinned := len(skin) > 0 && len(joints) > 0

	active_buf: [MAX_MORPH_TARGETS]Active_Morph = ---
	na := active_morphs(geo, scene_draw_morphs(job.scene, di), &active_buf)
	active := active_buf[:na]
	morphed := na > 0
	deltas := job.geos.morph_deltas[:]
	indices := all[c.first * 3:][:c.count * 3]
	lo, hi := max(u32), u32(0)
	for i in indices {
		lo, hi = min(lo, i), max(hi, i)
	}
	if len(indices) == 0 || int(hi) >= len(vertices) {
		slice.zero(s.chunk_bands[ci * job.bands:][:job.bands])
		return
	}
	count := int(hi - lo) + 1
	// What the vertices go through: a ShaderMaterial's vertex stage in a
	// colour pass (`sh`), its depth pass's in the frame's depth pass
	// (`custom`), else o3d's variant for the pass (`x`).
	sh, custom: ^Shader
	x: ^Program
	switch {
	case !job.depth_only && m.shader != 0:
		sh = shader_get(job.shaders, m.shader)
	case job.depth_only && job.own_side && m.shader != 0:
		custom = shader_get(job.shaders, m.shader)
	case:
		if id := job.programs[di]; id >= 0 {
			x = &s.programs.list[id]
		}
	}
	// A masked draw its opacity leaves wholly cut draws no depth.
	masked_out := job.depth_only && .Alpha_Mask in m.flags && m.flags & {.Color_Map, .Alpha_Map} == {} && m.opacity < m.alpha_cutoff
	if sh == nil && custom == nil && x == nil || masked_out {
		slice.zero(s.chunk_bands[ci * job.bands:][:job.bands])
		return
	}
	// The vertex stage's record (nf floats a vertex) and what each corner
	// keeps of it.
	nf: int
	keep: ^Corner_Floats
	switch {
	case sh != nil:
		nf, keep = sh.varying_floats, &sh.corners
	case custom != nil:
		nf, keep = custom.depth.vertex.varying_floats, &custom.depth_corners
	case:
		nf, keep = x.pipeline.vertex.varying_floats, &x.corners
	}
	table_grow_non_zero(&w.positions, count)
	table_grow_non_zero(&w.values, count * nf)
	if custom != nil {
		// Posed now and as at the shutter's opening (its model, joints and
		// weights then, each as now when it has none).
		opened_joints := joints
		opened_buf: [MAX_MORPH_TARGETS]Active_Morph = ---
		opened_active := active
		if job.velocity != nil {
			if j := scene_draw_opened_joints(job.scene, di); j != nil {
				opened_joints = j
			}
			if ow := scene_draw_opened_morphs(job.scene, di); ow != nil {
				opened_active = opened_buf[:active_morphs(geo, ow, &opened_buf)]
			}
		}
		depth_vertices(job, w, custom, di, vertices[lo:][:count], lo, skin, joints, opened_joints, active, opened_active, deltas, count)
	} else if x != nil {
		attrs: [len(Program_Attribute)]oksl.Attribute
		for a in Program_Attribute {
			loc := x.attributes[a]
			if loc < 0 {
				continue
			}
			row := PROGRAM_ATTRIBUTES[a]
			if row.skin {
				attrs[loc] = {rawptr(uintptr(&skin[lo]) + row.offset), size_of(Skin_Vertex), row.format}
			} else {
				attrs[loc] = {rawptr(uintptr(&vertices[lo]) + row.offset), size_of(Vertex), row.format}
			}
		}
		weights := scene_draw_morphs(job.scene, di)
		db: Draw_Block = ---
		draw_block(&db, job.view_proj * d.model, d.model, geo, weights, s.joint_bases[di])
		if x.key.entry == .Velocity {
			// The frame's depth pass: the draw as at the shutter's opening,
			// through the camera then (as now without motion blur).
			if job.velocity != nil {
				model := scene_draw_opened(job.scene, di)
				opened_weights := scene_draw_opened_morphs(job.scene, di)
				draw_block_opened(&db, job.opened_view_proj * (model if model != {} else d.model), geo, opened_weights if opened_weights != nil else weights)
			} else {
				draw_block_opened(&db, job.view_proj * d.model, geo, weights)
			}
		}
		vb: [size_of(Draw_Block)]u8 = ---
		block_fill(&x.vertex, {&db, nil}, vb[:])
		buffers: [2][]u32
		for b, i in x.vertex_buffers {
			#partial switch b {
			case .Morph_Deltas:
				buffers[i] = slice.reinterpret([]u32, deltas)
			case .Skin_Joints:
				buffers[i] = slice.reinterpret([]u32, s.joint_table[:])
			}
		}
		oksl.run_vertex(&x.pipeline.vertex.program, vb[:x.vertex.size], attrs[:], count, w.positions[:], w.values[:], &w.vm, {buffers = buffers[:], first_vertex = int(lo)})
	} else {
		src_verts := vertices[lo:][:count]
		if morphed || skinned {
			table_grow_non_zero(&w.skinned, count)
			for vtx, i in src_verts {
				vi := lo + u32(i)
				v := vtx
				if morphed {
					v.position, v.normal = morph_vertex(v, vi, deltas, active)
				}
				if skinned {
					v.position, v.normal = skin_vertex(v, skin[vi], joints)
				}
				w.skinned[i] = v
			}
			src_verts = w.skinned[:count]
		}
		attrs: [16]oksl.Attribute
		for a in Vertex_Attribute {
			loc := sh.attributes[a]
			if loc < 0 {
				continue
			}
			row := VERTEX_ATTRIBUTES[a]
			attrs[loc] = {rawptr(uintptr(&src_verts[0]) + row.offset), size_of(Vertex), .Float2 if row.components == 2 else .Float3}
		}
		oksl.run_vertex(&sh.pipeline.vertex.program, block(job.blocks, di, .Vertex), attrs[:], count, w.positions[:], w.values[:], &w.vm)
	}
	side := shadow_side(d.material.side) if job.depth_only && !job.own_side else d.material.side
	values := raw_data(w.values) // nil with no floats (the shadow map): never read
	for t := 0; t + 2 < len(indices); t += 3 {
		v := [3]int{int(indices[t] - lo), int(indices[t + 1] - lo), int(indices[t + 2] - lo)}
		clip := [3]Vec4{w.positions[v[0]], w.positions[v[1]], w.positions[v[2]]}
		if clip[0].z <= clip[0].w && clip[1].z <= clip[1].w && clip[2].z <= clip[2].w {
			emit(c, clip, {values[v[0] * nf:], values[v[1] * nf:], values[v[2] * nf:]}, keep, di, job.width, job.height, side, job.mirrored)
			continue
		}
		corners: [3]Corner = ---
		for k in 0 ..< 3 {
			corners[k].c = clip[k]
			copy(corners[k].v[:nf], w.values[v[k] * nf:][:nf])
		}
		clip_near(c, &corners, nf, keep, di, job.width, job.height, side, job.mirrored)
	}
	counts := s.chunk_bands[ci * job.bands:][:job.bands]
	slice.zero(counts)
	max_ys := c.tris.max_y[:len(c.tris)]
	for min_y, i in c.tris.min_y[:len(c.tris)] {
		first, last := band_range(min_y, max_ys[i])
		for b in first ..= last {
			counts[b] += 1
		}
	}
}

// A ShaderMaterial's chunk vertices through its depth pass's vertex stage
// (Shader.depth): positions now, and where the tap put each at the
// shutter's opening repacked as the depth band reads it (OPENED_FLOATS),
// its twins the vertices posed as then.
@(private = "file")
depth_vertices :: proc(job: ^Frame_Job, w: ^Worker, sh: ^Shader, di: int, src: []Vertex, lo: u32, skin: []Skin_Vertex, joints, opened_joints: []Mat4, active, opened_active: []Active_Morph, deltas: []Morph_Delta, count: int) {
	pose :: proc(out: ^[dynamic]Vertex, src: []Vertex, lo: u32, skin: []Skin_Vertex, joints: []Mat4, active: []Active_Morph, deltas: []Morph_Delta) -> []Vertex {
		if len(active) == 0 && len(joints) == 0 {
			return src
		}
		table_grow_non_zero(out, len(src))
		for vtx, i in src {
			vi := lo + u32(i)
			v := vtx
			if len(active) > 0 {
				v.position, v.normal = morph_vertex(v, vi, deltas, active)
			}
			if len(joints) > 0 && len(skin) > 0 {
				v.position, v.normal = skin_vertex(v, skin[vi], joints)
			}
			out[i] = v
		}
		return out[:len(src)]
	}
	now := pose(&w.skinned, src, lo, skin, joints, active, deltas)
	then := pose(&w.opened, src, lo, skin, opened_joints, opened_active, deltas)
	attrs: [16]oksl.Attribute
	for a in Vertex_Attribute {
		if loc := sh.depth_attributes[a]; loc >= 0 {
			row := VERTEX_ATTRIBUTES[a]
			attrs[loc] = {rawptr(uintptr(&now[0]) + row.offset), size_of(Vertex), .Float2 if row.components == 2 else .Float3}
		}
	}
	for loc, k in sh.depth_twins {
		if loc >= 0 {
			row := VERTEX_ATTRIBUTES[.Position if k == 0 else .Normal]
			attrs[loc] = {rawptr(uintptr(&then[0]) + row.offset), size_of(Vertex), .Float3}
		}
	}
	oksl.run_vertex(&sh.depth.vertex.program, block(job.depth_blocks, di, .Vertex), attrs[:], count, w.positions[:], w.values[:], &w.vm)
}

// A draw's uniform block for a stage (nil: none).
@(private)
block :: proc(b: ^Shader_Blocks, draw: int, stage: Shader_Stage) -> []u8 {
	at := b.at[draw][stage]
	return b.bytes[at:] if at >= 0 else nil
}

// Between the phases: each band's start, and where in it each chunk lists
// its triangles (chunks in drawing order).
@(private = "file")
sum_bands :: proc(s: ^Software, bands: int) {
	table_grow_non_zero(&s.band_start, bands + 1)
	at := 0
	for b in 0 ..< bands {
		s.band_start[b] = at
		for ci in 0 ..< s.chunk_count {
			n := s.chunk_bands[ci * bands + b]
			s.chunk_bands[ci * bands + b] = at
			at += n
		}
	}
	s.band_start[bands] = at
	table_grow_non_zero(&s.band_tris, at)
}

// Phase 2: a chunk's triangles listed in the bands they touch.
@(private = "file")
list_chunk :: proc(data: rawptr, ci, _: int) {
	job := (^Frame_Job)(data)
	s := job.s
	c := &s.chunks[ci]
	at := s.chunk_bands[ci * job.bands:][:job.bands]
	max_ys := c.tris.max_y[:len(c.tris)]
	for min_y, i in c.tris.min_y[:len(c.tris)] {
		first, last := band_range(min_y, max_ys[i])
		for b in first ..= last {
			s.band_tris[at[b]] = {u32(ci), u32(i)}
			at[b] += 1
		}
	}
}

// Clips a triangle crossing the near plane (z <= w in clip space) and keeps
// what is left as one or two screen triangles; `nf` floats per corner, the
// flat ones `keep` keeps the first vertex's, carried, not interpolated.
@(private = "file")
clip_near :: proc(c: ^Chunk, tri: ^[3]Corner, nf: int, keep: ^Corner_Floats, draw: int, width, height: int, side: Side, mirrored: bool) {
	for j in keep.flat {
		f := keep.from[j]
		tri[1].v[f], tri[2].v[f] = tri[0].v[f], tri[0].v[f]
	}
	in_: [4]Corner = ---
	count := 0
	for k in 0 ..< 3 {
		a, b := &tri[k], &tri[(k + 1) % 3]
		da, db := a.c.w - a.c.z, b.c.w - b.c.z
		if da >= 0 {
			in_[count] = a^
			count += 1
		}
		if (da >= 0) != (db >= 0) {
			t := da / (da - db)
			o := &in_[count]
			o.c = linalg.lerp(a.c, b.c, t)
			for f in 0 ..< nf {
				o.v[f] = a.v[f] * (1 - t) + b.v[f] * t
			}
			for j in keep.flat {
				o.v[keep.from[j]] = a.v[keep.from[j]]
			}
			count += 1
		}
	}
	for k in 1 ..< count - 1 {
		emit(c, {in_[0].c, in_[k].c, in_[k + 1].c}, {&in_[0].v[0], &in_[k].v[0], &in_[k + 1].v[0]}, keep, draw, width, height, side, mirrored)
	}
}

// A clipped triangle to the screen, kept if it faces a drawn side and
// reaches a row: of its corners' records (`values`) the floats `keep`
// keeps, divided by w except the flat ones, the first corner's. Seen in a
// mirror, a front face turns clockwise.
@(private = "file")
emit :: proc(c: ^Chunk, clip: [3]Vec4, values: [3][^]f32, keep: ^Corner_Floats, draw: int, width, height: int, side: Side, mirrored: bool) {
	t: Tri
	t.draw = draw
	for k in 0 ..< 3 {
		clip := clip[k]
		w := clip.w if abs(clip.w) > 1e-9 else 1e-9
		iw := 1 / w
		t.inv_w[k] = iw
		t.p[k] = {(clip.x * iw * 0.5 + 0.5) * f32(width), (0.5 - clip.y * iw * 0.5) * f32(height), clip.z * iw}
	}
	// Screen area: negative is counter-clockwise on screen (y down), the
	// front face.
	area := (t.p[1].x - t.p[0].x) * (t.p[2].y - t.p[0].y) - (t.p[2].x - t.p[0].x) * (t.p[1].y - t.p[0].y)
	t.front = (area < 0) != mirrored
	if area == 0 || (side == .Front && !t.front) || (side == .Back && t.front) {
		return
	}
	order := [3]int{0, 1, 2}
	if area > 0 {
		order = {0, 2, 1}
		t.p[1], t.p[2] = t.p[2], t.p[1]
		t.inv_w[1], t.inv_w[2] = t.inv_w[2], t.inv_w[1]
	}
	lo := min(t.p[0].y, t.p[1].y, t.p[2].y)
	hi := max(t.p[0].y, t.p[1].y, t.p[2].y)
	t.min_y = clamp(int(math.floor(lo)), 0, height)
	t.max_y = clamp(int(math.ceil(hi)), 0, height)
	if t.min_y >= t.max_y {
		return
	}
	n := keep.count
	at := len(c.values)
	t.values = u32(at)
	table_grow_non_zero(&c.values, at + 3 * n)
	out := c.values[at:]
	for k, i in order {
		iw := t.inv_w[i]
		for f, j in keep.from[:n] {
			out[i * n + j] = values[k][f] * iw
		}
		for j in keep.flat {
			out[i * n + j] = values[0][keep.from[j]]
		}
	}
	append(&c.tris, t)
}

// Runs `work(data, i, worker)` for every i in 0 ..< count: on the caller
// (worker 0) and up to a pool thread per item (worker 1 + its task), each
// claiming the next item until none is left. The caller then takes the
// tasks no thread has started off the pool's queue (so it may run in a
// task on the same pool, even on all its threads at once), waits for the
// others to finish, and takes them off the pool's done list, which would
// otherwise keep every task of every frame (the caller's own done tasks
// stay).
@(private)
parallel :: proc(pool: ^thread.Pool, count: int, data: rawptr, work: proc(data: rawptr, i, worker: int)) {
	Parallel :: struct {
		count:   int,
		data:    rawptr,
		work:    proc(data: rawptr, i, worker: int),
		next:    int, // atomic: the next item
		running: int, // atomic: tasks not finished
		ctx:     runtime.Context,
	}
	claim :: proc(p: ^Parallel, worker: int) {
		temp := context.temp_allocator
		context = p.ctx
		if worker > 0 {
			context.temp_allocator = temp
		}
		for {
			i := sync.atomic_add(&p.next, 1)
			if i >= p.count {
				return
			}
			p.work(p.data, i, worker)
		}
	}
	task :: proc(t: thread.Task) {
		p := (^Parallel)(t.data)
		claim(p, 1 + t.user_index)
		sync.atomic_sub(&p.running, 1)
	}
	p := Parallel{count = count, data = data, work = work, ctx = context}
	helpers := 0
	if pool != nil {
		sync.guard(&pool.mutex)
		helpers = min(len(pool.threads), count - 1)
	}
	p.running = max(helpers, 0)
	for i in 0 ..< helpers {
		thread.pool_add_task(pool, context.allocator, task, rawptr(&p), i) // waited for below
	}
	claim(&p, 0)
	// Every item is taken: a task not started would find none. Its post
	// on the pool's semaphore stays, and the thread it wakes finds no task
	// and waits again.
	left := helpers
	if helpers > 0 {
		sync.guard(&pool.mutex)
		kept := 0
		for i in 0 ..< queue.len(pool.tasks) {
			task := queue.get(&pool.tasks, i)
			if task.data == rawptr(&p) {
				sync.atomic_sub(&pool.num_waiting, 1)
				sync.atomic_sub(&pool.num_outstanding, 1)
				sync.atomic_sub(&p.running, 1)
				left -= 1
				continue
			}
			queue.set(&pool.tasks, kept, task)
			kept += 1
		}
		queue.consume_back(&pool.tasks, queue.len(pool.tasks) - kept)
	}
	for sync.atomic_load(&p.running) > 0 {
		thread.yield()
	}
	// A task joins the done list just after it returns: the newest are ours.
	for left > 0 {
		sync.lock(&pool.mutex)
		for i := len(pool.tasks_done) - 1; i >= 0 && left > 0; i -= 1 {
			if pool.tasks_done[i].data == rawptr(&p) {
				ordered_remove(&pool.tasks_done, i)
				sync.atomic_sub(&pool.num_done, 1)
				left -= 1
			}
		}
		sync.unlock(&pool.mutex)
		if left > 0 {
			thread.yield()
		}
	}
}

