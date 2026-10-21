package o3d

import "base:runtime"
import "core:math"
import "core:math/linalg"
import "core:slice"
import "core:sync"
import "core:thread"

import "deps:oksl"

// The CPU renderer: the pixel reference for the GPU's and the path without
// one. Four samples per pixel (the GPU path's 4x MSAA), a depth per sample,
// shading once per pixel per triangle at its centre with perspective-correct
// values, near-plane clipping. Samples hold sRGB premultiplied RGBA8, so the
// resolve averages what the GPU's does.
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
// The standard material's corner floats: world position, normal, and uv + uv1 when textured
// (or an exact shape's: uv1 is its shape coordinate), then its vertex colour (linear) and alpha
// with `.Vertex_Colors`.
UNTEX_FLOATS :: 6
STANDARD_FLOATS :: 10
COLOR_FLOATS :: 4
// The shadow map's corner floats: the alpha mask's colour map uv, then an
// exact shape's coordinate, each when there is one.
SHADOW_MASK_FLOATS :: 2

@(private)
shadow_floats :: #force_inline proc "contextless" (masked, curved: bool) -> int {
	return SHADOW_MASK_FLOATS * (int(masked) + int(curved))
}

Software :: struct {
	width, height: int,
	color:         []u32, // per sample, RGBA8 sRGB premultiplied
	depth:         []f32, // per sample, 1 near .. 0 far (reversed)
	pixels:        []u8, // the resolved frame, RGBA8 premultiplied
	// Scratch per frame (kept, with the chunks' and workers' arrays, to
	// reuse).
	order:         [dynamic]int,
	materials:     [dynamic]Shading_Material, // per draw
	blocks:        Shader_Blocks,
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
}

// An external texture's pixels for the software renderer: RGBA8 (sRGB
// bytes, as an image's), rows from the top, the host's.
Software_External :: struct {
	pixels:        [][4]u8,
	width, height: int,
}

software_destroy :: proc(s: ^Software) {
	delete(s.color)
	delete(s.depth)
	delete(s.pixels)
	delete(s.order)
	delete(s.materials)
	shader_blocks_destroy(&s.blocks)
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
// `texture_add` takes them) to an external texture (`texture_add_external`)
// for the software renders that follow, until bound again: what
// `o3d_sdlgpu.texture_set_external` binds on the GPU, read as it reads an
// unmipped one (bilinear at its level 0). The host keeps them alive while
// bound; nil (or too few bytes) unbinds it, then sampled white.
software_set_external :: proc(s: ^Software, id: Texture_Id, pixels: []u8, width, height: int) {
	if id == 0 {
		return
	}
	if int(id) > len(s.externals) {
		resize(&s.externals, int(id))
	}
	ok := width > 0 && height > 0 && len(pixels) >= width * height * 4
	s.externals[id - 1] = {slice.reinterpret([][4]u8, pixels[:width * height * 4]), width, height} if ok else {}
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
	curve:        u32,
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
	quads:     struct {
		origin:  [dynamic][2]i32,
		helpers: [dynamic]oksl.Quad_Lanes,
		facing:  [dynamic]bool,
		covered: [dynamic][4]Sample_Mask, // per lane
		tri:     [dynamic]Tri_Ref,
		values:  [dynamic]f32, // structure of arrays, as run_fragment reads them
		colours: [dynamic][4]f32,
		shaded:  [dynamic]oksl.Quad_Lanes,
	},
}

@(private = "file")
worker_destroy :: proc(w: ^Worker) {
	oksl.scratch_destroy(&w.vm)
	delete(w.positions)
	delete(w.values)
	delete(w.skinned)
	delete(w.quads.origin)
	delete(w.quads.helpers)
	delete(w.quads.facing)
	delete(w.quads.covered)
	delete(w.quads.tri)
	delete(w.quads.values)
	delete(w.quads.colours)
	delete(w.quads.shaded)
}

// Draws `scene` into `width x height` premultiplied RGBA8 (returned,
// borrowed until the next render). `shaders` holds the ShaderMaterials'
// shaders (nil: none); `textures` holds the material textures (nil: none).
software_render :: proc(s: ^Software, scene: ^Scene, geos: ^Geometries, shaders: ^Shaders, width, height: int, pool: ^thread.Pool = nil, textures: ^Textures = nil) -> []u8 {
	if s.width != width || s.height != height {
		delete(s.color)
		delete(s.depth)
		delete(s.pixels)
		s.width, s.height = width, height
		s.color = make([]u32, width * height * SAMPLES)
		s.depth = make([]f32, width * height * SAMPLES)
		s.pixels = make([]u8, width * height * 4)
	}
	view, proj := camera_matrices(scene.camera, width, height)
	active := expand_draws(&s.expanded, scene, geos, view, proj, width, height)
	lights: [MAX_LIGHTS]Shading_Light
	nl := shading_lights(active.lights[:], &lights)
	shading_materials(scene_materials(active), &s.materials, textures)
	sprite_materials(active, geos, &s.expanded, s.materials[:])
	shader_blocks(active, shaders, view, proj, &s.blocks)
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
	shadows: [MAX_SHADOWS]Shadow
	ns := frame_shadows(active, &shadows)
	job := Frame_Job {
		s             = s,
		scene         = active,
		geos          = geos,
		shaders       = shaders,
		textures      = textures,
		lights        = lights[:nl],
		env           = env,
		bg_tex        = bg_tex,
		bg_intensity  = bg_intensity,
		bg_lod        = bg_lod,
		eye           = active.camera.position,
		shadows       = shadows,
		shadow_count  = ns,
		tone_mapping  = active.tone_mapping,
		tone_exposure = active.tone_mapping_exposure,
		fog           = active.fog,
		fog_color     = linear(active.fog.color),
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
		job.view_proj, job.width, job.height, job.depth_only = sh.view_proj, size, size, true
		job.depth_target, job.own_side = s.shadow_maps[si], false
		run_pass(s, active, geos, &job, pool, draw_depth_band)
	}
	if needs_depth(active) {
		table_grow_non_zero(&s.depth_map, width * height)
		depth_draws(active, geos, shaders, proj * view, &s.order, &s.expanded)
		job.view_proj, job.width, job.height, job.depth_only = proj * view, width, height, true
		job.depth_target, job.own_side = s.depth_map[:width * height], true
		run_pass(s, active, geos, &job, pool, draw_depth_band)
	} else {
		clear(&s.depth_map)
	}
	draw_order(active, geos, shaders, view, proj, &s.order, &s.expanded)
	job.view_proj, job.width, job.height, job.depth_only = proj * view, width, height, false
	if bg_tex != 0 {
		job.inv_vp = linalg.inverse(job.view_proj)
	}
	run_pass(s, active, geos, &job, pool, draw_band)
	return s.pixels
}

// The frame's depth per pixel from the last `software_render` of a scene
// with `Scene.depth` (rows from the top; empty without).
software_depth :: proc "contextless" (s: ^Software) -> []f32 {
	return s.depth_map[:]
}

// The phases over the draws in `s.order` into a `job.width x job.height`
// target, its bands drawn by `draw`.
@(private = "file")
run_pass :: proc(s: ^Software, scene: ^Scene, geos: ^Geometries, job: ^Frame_Job, pool: ^thread.Pool, draw: proc(data: rawptr, band, worker: int)) {
	make_chunks(s, scene, geos)
	job.bands = (job.height + BAND_ROWS - 1) / BAND_ROWS
	table_grow_non_zero(&s.chunk_bands, s.chunk_count * job.bands)
	parallel(pool, s.chunk_count, job, transform_chunk)
	sum_bands(s, job.bands)
	parallel(pool, s.chunk_count, job, list_chunk)
	parallel(pool, job.bands, job, draw)
}

// Splits the drawing order into chunks of at most CHUNK_TRIS triangles.
@(private = "file")
make_chunks :: proc(s: ^Software, scene: ^Scene, geos: ^Geometries) {
	n := 0
	for di, oi in s.order {
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
	scene:         ^Scene,
	geos:          ^Geometries,
	shaders:       ^Shaders,
	textures:      ^Textures,
	lights:        []Shading_Light,
	env:           Shading_Environment,
	bg_tex:        Texture_Id,
	bg_intensity:  f32,
	bg_lod:        f32,
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
	bands:         int,
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
	count := int(hi - lo) + 1
	sh := shader_get(job.shaders, m.shader) if !job.depth_only else nil
	textured := m.flags & (TEXTURE_FLAGS + {.Curves, .Sprite, .Vertex_Colors}) != {}
	coloured := .Vertex_Colors in m.flags
	nf := sh.varying_floats if sh != nil else (STANDARD_FLOATS + (COLOR_FLOATS if coloured else 0) if textured else UNTEX_FLOATS)
	masked, curved: bool
	if job.depth_only {
		if .Alpha_Mask in m.flags && m.flags & {.Color_Map, .Alpha_Map} == {} && m.opacity < m.alpha_cutoff {
			slice.zero(s.chunk_bands[ci * job.bands:][:job.bands])
			return
		}
		masked, _ = shadow_mask(m)
		curved = .Curves in m.flags
		nf = shadow_floats(masked, curved)
	}
	flat := shader_flat(sh) if sh != nil else nil
	table_grow_non_zero(&w.positions, count)
	table_grow_non_zero(&w.values, count * nf)
	if job.depth_only {
		mvp := job.view_proj * d.model
		_, use_uv1 := shadow_mask(m)
		for vtx, i in vertices[lo:][:count] {
			vi := lo + u32(i)
			v := vtx
			if morphed {
				v.position, v.normal = morph_vertex(v, vi, deltas, active)
			}
			if skinned {
				v.position, v.normal = skin_vertex(v, skin[vi], joints)
			}
			w.positions[i] = mvp * Vec4{v.position.x, v.position.y, v.position.z, 1}
			if masked {
				(^Vec2)(&w.values[i * nf])^ = v.uv1 if use_uv1 else v.uv
			}
			if curved {
				(^Vec2)(&w.values[i * nf + nf - SHADOW_MASK_FLOATS])^ = v.uv1
			}
		}
	} else if sh == nil {
		mvp := job.view_proj * d.model
		nm := normal_matrix(d.model)
		for vtx, i in vertices[lo:][:count] {
			vi := lo + u32(i)
			v := vtx
			if morphed {
				v.position, v.normal = morph_vertex(v, vi, deltas, active)
			}
			if skinned {
				v.position, v.normal = skin_vertex(v, skin[vi], joints)
			}
			p := Vec4{v.position.x, v.position.y, v.position.z, 1}
			w.positions[i] = mvp * p
			base := i * nf
			(^Vec3)(&w.values[base])^ = (d.model * p).xyz
			(^Vec3)(&w.values[base + 3])^ = (nm * Vec4{v.normal.x, v.normal.y, v.normal.z, 0}).xyz
			if textured {
				(^Vec2)(&w.values[base + 6])^ = v.uv
				(^Vec2)(&w.values[base + 8])^ = v.uv1
			}
			if coloured {
				c := v.color
				(^Vec4)(&w.values[base + STANDARD_FLOATS])^ = {srgb_to_linear(f32(c.r) / 255), srgb_to_linear(f32(c.g) / 255), srgb_to_linear(f32(c.b) / 255), f32(c.a) / 255}
			}
		}
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
		oksl.run_vertex(&sh.pipeline.vertex.program, block(&s.blocks, di, .Vertex), attrs[:], count, w.positions[:], w.values[:], &w.vm)
	}
	side := shadow_side(d.material.side) if job.depth_only && !job.own_side else d.material.side
	values := raw_data(w.values) // nil with no floats (the shadow map): never read
	curved_geo := geo.mode == .Curves && .Curves in m.flags
	for t := 0; t + 2 < len(indices); t += 3 {
		v := [3]int{int(indices[t] - lo), int(indices[t + 1] - lo), int(indices[t + 2] - lo)}
		clip := [3]Vec4{w.positions[v[0]], w.positions[v[1]], w.positions[v[2]]}
		// Its quad's entry: an exact shape's curve set, or a particle's row.
		curve := curve_quad_entry(job.geos, geo, indices[t] / 4) if curved_geo else (geo.sprite_table + indices[t] / 4 if geo.mode == .Sprites else 0)
		if clip[0].z <= clip[0].w && clip[1].z <= clip[1].w && clip[2].z <= clip[2].w {
			emit(c, clip, {values[v[0] * nf:], values[v[1] * nf:], values[v[2] * nf:]}, nf, flat, di, curve, job.width, job.height, side)
			continue
		}
		corners: [3]Corner = ---
		for k in 0 ..< 3 {
			corners[k].c = clip[k]
			copy(corners[k].v[:nf], w.values[v[k] * nf:][:nf])
		}
		clip_near(c, &corners, nf, flat, di, curve, job.width, job.height, side)
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
// what is left as one or two screen triangles; `nf` floats per corner,
// the `flat` ones the first vertex's, carried, not interpolated.
@(private = "file")
clip_near :: proc(c: ^Chunk, tri: ^[3]Corner, nf: int, flat: []u8, draw: int, curve: u32, width, height: int, side: Side) {
	for f in flat {
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
			for f in flat {
				o.v[f] = a.v[f]
			}
			count += 1
		}
	}
	for k in 1 ..< count - 1 {
		emit(c, {in_[0].c, in_[k].c, in_[k + 1].c}, {&in_[0].v[0], &in_[k].v[0], &in_[k + 1].v[0]}, nf, flat, draw, curve, width, height, side)
	}
}

// A clipped triangle to the screen, kept if it faces a drawn side and
// reaches a row: its corners' `nf` floats each (`values`), divided by w
// except the flat ones, the first corner's.
@(private = "file")
emit :: proc(c: ^Chunk, clip: [3]Vec4, values: [3][^]f32, nf: int, flat: []u8, draw: int, curve: u32, width, height: int, side: Side) {
	t: Tri
	t.draw = draw
	t.curve = curve
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
	t.front = area < 0
	if area == 0 || (side == .Front && !t.front) || (side == .Back && t.front) {
		return
	}
	order := [3]int{0, 1, 2}
	if !t.front {
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
	at := len(c.values)
	t.values = u32(at)
	table_grow_non_zero(&c.values, at + 3 * nf)
	out := c.values[at:]
	for k, i in order {
		iw := t.inv_w[i]
		for f in 0 ..< nf {
			out[i * nf + f] = values[k][f] * iw
		}
		for f in flat {
			out[i * nf + int(f)] = values[0][f]
		}
	}
	append(&c.tris, t)
}

// Runs `work(data, i, worker)` for every i in 0 ..< count: on the caller
// (worker 0) and up to a pool thread per item (worker 1 + its task), each
// claiming the next item until none is left; the caller waits for the
// tasks to finish (they may start after the items ran out).
@(private = "file")
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
		context = p.ctx
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
		thread.pool_add_task(pool, context.allocator, task, &p, i)
	}
	claim(&p, 0)
	for sync.atomic_load(&p.running) > 0 {
		thread.yield()
	}
}

