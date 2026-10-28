package o3d_sdlgpu

import "core:math"
import "core:math/linalg"
import "core:mem"
import "core:slice"
import sdl "vendor:sdl3"

import o3d ".."

// Starts the mirror of a table over when it holds other contents than the
// GPU mirrors (a new table reuses ids, positions, even its address): the
// geometry re-uploads, textures and ShaderMaterials' pipelines are made
// again (what was made is released once the GPU is done with it).
@(private = "file")
sync_mirrors :: proc(g: ^GPU, geos: ^o3d.Geometries, shaders: ^o3d.Shaders, textures: ^o3d.Textures) {
	if geos != nil && geos.generation != g.mirrored.geometry {
		g.vertices.uploaded, g.skin_vertices.uploaded, g.morph_deltas.uploaded, g.indices.uploaded = 0, 0, 0, 0
		g.curve_words.uploaded = 0
		g.mirrored.geometry = geos.generation
	}
	if textures != nil && textures.generation != g.mirrored.textures {
		for tex in g.textures {
			if tex != nil {
				sdl.ReleaseGPUTexture(g.device, tex)
			}
		}
		clear(&g.textures)
		g.mirrored.textures = textures.generation
	}
	if shaders != nil && shaders.generation != g.mirrored.shaders {
		for &c in g.custom {
			release_custom(g, &c)
		}
		clear(&g.custom)
		g.mirrored.shaders = shaders.generation
	}
}

// A render in two steps, for a host recording it at a point of its own
// command stream where nothing may be made (making a GPU resource may
// submit): `prepare` makes and uploads everything the scene's render needs,
// in a command buffer of its own that it submits; `record` then records the
// passes into the host's command buffer, making nothing. A job holds what one
// render prepared, so a frame's several renders can each be prepared ahead and
// recorded later in any order, once per prepare. Until then its scene and
// tables stay as prepared, and the host's external textures may be bound up
// to `record`. The jobs of one GPU share its mirrors of the tables (so the
// jobs prepared together draw from the same tables) and its frame targets
// (grown to the largest job); each job has its own shadow maps and per-frame
// buffers. A job is reused frame after frame (what it made only grows), and
// `job_destroy` frees it.
Job :: struct {
	// As prepared: the scene drawn (the caller's, or its expansion in
	// `expanded`), its tables and the tables' generations then.
	scene:                ^o3d.Scene,
	geos:                 ^o3d.Geometries,
	shaders:              ^o3d.Shaders,
	mirrored:             Mirrored,
	width, height:        u32,
	prepared:             bool,
	view_proj:            o3d.Mat4,
	lighting:             Lighting_Uniforms,
	env:                  o3d.Shading_Environment,
	background:           o3d.Texture_Id,
	// The light where nothing is drawn (o3d.background_clear_light).
	clear:                [4]f32,
	// The background's intensity and mip level; whether its map is linear
	// (HDR).
	bg_params:            struct {
		params: [4]f32,
		flags:  [4]u32,
	},
	shadows:              [o3d.MAX_SHADOWS]o3d.Shadow,
	shadow_count:         int,
	// Per shadow: its casters, its map (depth per texel, read by the main
	// pass) and the map's depth buffer at `shadow_sizes` squared.
	casters:              [o3d.MAX_SHADOWS][dynamic]int,
	shadow_maps:          [o3d.MAX_SHADOWS]^sdl.GPUTexture,
	shadow_depths:        [o3d.MAX_SHADOWS]^sdl.GPUTexture,
	shadow_sizes:         [o3d.MAX_SHADOWS]u32,
	// The frame's depth (`Scene.depth`): its draws, its map (R32F, sampled
	// by the host: `job_depth`) and depth buffer at `depth_size`.
	depth:                bool,
	depth_draws:          [dynamic]int,
	depth_map:            ^sdl.GPUTexture,
	depth_buffer:         ^sdl.GPUTexture,
	depth_size:           [2]u32,
	order:                [dynamic]int,
	// Where in `order` the transmissive draws start (len(order): none);
	// the draws before it are drawn first into the snapshot (their light,
	// mipmapped, at the frame's size).
	snapshot:             int,
	snapshot_tex:         ^sdl.GPUTexture,
	snapshot_size:        [2]u32,
	// The mirror's pass (`o3d.mirror_view`) when the frame has one, drawn
	// first: its draws, camera and lighting; its light and runs (in
	// `runs_target`, resolved into `runs_resolved`) into the two layers of
	// `reflection_tex` (width, height and levels in `reflection_size`),
	// mipmapped, for the draws that show it.
	reflect:              bool,
	mirror_order:         [dynamic]int,
	mirror_view_proj:     o3d.Mat4,
	mirror_lighting:      Lighting_Uniforms,
	reflection_tex:       ^sdl.GPUTexture,
	runs_target:          ^sdl.GPUTexture,
	runs_resolved:        ^sdl.GPUTexture,
	reflection_size:      [3]u32,
	// The depth of field (`o3d.focus_of`) when the frame has one: its
	// post passes' uniforms and its tables (tiles, near, half light, half
	// keys) at `focus_size`.
	focus:                bool,
	post:                 Post_Uniforms,
	focus_tables:         [4]^sdl.GPUTexture,
	focus_size:           [2]u32,
	// Motion blur (`o3d.motion_of`) when the frame has it: its depth pass
	// also writes each pixel's velocity and nearness (at `depth_size`), the
	// camera at the shutter's opening, the frame's uniforms for it, and its
	// light blurred along them (at the targets' capacity, `blurred_size`)
	// through the filter's tables.
	motion:               bool,
	velocity_map:         ^sdl.GPUTexture,
	nearness_map:         ^sdl.GPUTexture,
	velocity_size:        [2]u32,
	opened_view_proj:     o3d.Mat4,
	velocity:             Velocity_Uniforms,
	motion_params:        o3d.Motion_Blur,
	blurred:              ^sdl.GPUTexture,
	blurred_size:         [2]u32,
	motion_tables:        Motion_Tables,
	materials:            [dynamic]o3d.Shading_Material, // per draw
	// ShaderMaterials' uniform blocks, through the camera and the mirror.
	blocks:               o3d.Shader_Blocks,
	mirror_blocks:        o3d.Shader_Blocks,
	// And for the frame's depth pass (o3d.depth_blocks), with each posed
	// ShaderMaterial's vertices as at the shutter's opening in
	// `custom_skinned` too (from `custom_opened_first`; -1: as now).
	depth_blocks:         o3d.Shader_Blocks,
	custom_opened_first:  [dynamic]i32,
	expanded:             o3d.Expanded_Draws,
	custom_skinned_verts: [dynamic]o3d.Vertex,
	custom_skinned_first: [dynamic]i32,
	// The expanded line/point geometry and ShaderMaterials' posed vertices.
	dynamic_vertices:     Pool_Buffer,
	dynamic_indices:      Pool_Buffer,
	// The particle quads' rows (`Expanded_Draws.sprite_rows`).
	sprite_rows:          Pool_Buffer,
	custom_skinned:       Pool_Buffer,
}

job_destroy :: proc(g: ^GPU, job: ^Job) {
	for idx in 0 ..< o3d.MAX_SHADOWS {
		release_shadow_slot(g, job, idx)
		delete(job.casters[idx])
	}
	release_depth(g, job)
	release_snapshot(g, job)
	release_reflection(g, job)
	release_focus(g, job)
	release_motion(g, job)
	delete(job.depth_draws)
	delete(job.mirror_order)
	for b in ([]^Pool_Buffer{&job.dynamic_vertices, &job.dynamic_indices, &job.custom_skinned, &job.sprite_rows}) {
		if b.buffer != nil {
			sdl.ReleaseGPUBuffer(g.device, b.buffer)
		}
	}
	delete(job.order)
	delete(job.materials)
	o3d.shader_blocks_destroy(&job.blocks)
	o3d.shader_blocks_destroy(&job.mirror_blocks)
	o3d.shader_blocks_destroy(&job.depth_blocks)
	delete(job.custom_opened_first)
	o3d.expanded_draws_destroy(&job.expanded)
	delete(job.custom_skinned_verts)
	delete(job.custom_skinned_first)
	job^ = {}
}

// A pool's upload this prepare: its data, of which the GPU lacks what is
// past `pool.uploaded`.
@(private = "file")
Upload :: struct {
	pool:  ^Pool_Buffer,
	usage: sdl.GPUBufferUsageFlags,
	data:  []u8,
}

// What `geos` added since the last frame.
@(private = "file")
upload_geometry :: proc(g: ^GPU, geos: ^o3d.Geometries, out: ^[dynamic; 9]Upload) {
	append(out, Upload{&g.vertices, {.VERTEX}, mem.slice_to_bytes(geos.vertices[:])})
	append(out, Upload{&g.skin_vertices, {.VERTEX}, mem.slice_to_bytes(geos.skin_vertices[:])})
	append(out, Upload{&g.morph_deltas, {.GRAPHICS_STORAGE_READ}, mem.slice_to_bytes(geos.morph_deltas[:])})
	append(out, Upload{&g.indices, {.INDEX}, mem.slice_to_bytes(geos.indices[:])})
	append(out, Upload{&g.curve_words, {.GRAPHICS_STORAGE_READ}, mem.slice_to_bytes(geos.curve_words[:])})
}

// The job's expanded line/point geometry when present, whole.
@(private = "file")
upload_dynamic :: proc(job: ^Job, out: ^[dynamic; 9]Upload) {
	if len(job.expanded.vertices) == 0 || len(job.expanded.indices) == 0 {
		return
	}
	job.dynamic_vertices.uploaded = 0
	job.dynamic_indices.uploaded = 0
	append(out, Upload{&job.dynamic_vertices, {.VERTEX}, mem.slice_to_bytes(job.expanded.vertices[:])})
	append(out, Upload{&job.dynamic_indices, {.INDEX}, mem.slice_to_bytes(job.expanded.indices[:])})
	if len(job.expanded.sprite_rows) > 0 {
		job.sprite_rows.uploaded = 0
		append(out, Upload{&job.sprite_rows, {.GRAPHICS_STORAGE_READ}, mem.slice_to_bytes(job.expanded.sprite_rows[:])})
	}
}

// When a morphed or skinned draw uses a ShaderMaterial, morphs and skins its
// vertices into `job.custom_skinned` before the render pass so the user's
// vertex shader receives posed positions and normals.
@(private = "file")
upload_custom_skinned :: proc(job: ^Job, scene: ^o3d.Scene, geos: ^o3d.Geometries, out: ^[dynamic; 9]Upload) {
	clear(&job.custom_skinned_verts)
	o3d.table_grow_non_zero(&job.custom_skinned_first, len(scene.draws))
	o3d.table_grow_non_zero(&job.custom_opened_first, len(scene.draws))
	for &f in job.custom_skinned_first {
		f = -1
	}
	for &f in job.custom_opened_first {
		f = -1
	}
	// The frame's draws, the mirror's and the depth pass's (some only one
	// sees).
	for list in ([3][]int{job.order[:], job.mirror_order[:] if job.reflect else nil, job.depth_draws[:] if job.depth else nil}) do for di in list {
		if job.materials[di].shader == 0 || job.custom_skinned_first[di] >= 0 {
			continue
		}
		d := &scene.draws[di]
		if d.geometry >= o3d.DYNAMIC_GEO_BIT {
			continue
		}
		geo, ok := o3d.geometry_get(geos, d.geometry)
		if !ok {
			continue
		}
		morphs: [o3d.MAX_MORPH_TARGETS]o3d.Active_Morph
		nm := o3d.active_morphs(geo, o3d.scene_draw_morphs(scene, di), &morphs)
		morphed := nm > 0 && len(geos.morph_deltas) > 0
		skin := o3d.geometry_skin(geos, d.geometry)
		joints := o3d.scene_draw_joints(scene, di)
		skinned := len(skin) > 0 && len(joints) > 0
		reposed := job.depth && (o3d.scene_draw_opened_joints(scene, di) != nil && len(skin) > 0 || o3d.scene_draw_opened_morphs(scene, di) != nil && len(geos.morph_deltas) > 0)
		if !morphed && !skinned && !reposed {
			continue
		}
		verts, _, _ := o3d.geometry_mesh(geos, d.geometry)
		first := len(job.custom_skinned_verts)
		o3d.table_grow_non_zero(&job.custom_skinned_verts, first + len(verts))
		deltas := geos.morph_deltas[:]
		for vtx, i in verts {
			v := vtx
			if morphed {
				v.position, v.normal = o3d.morph_vertex(v, u32(i), deltas, morphs[:nm])
			}
			if skinned {
				v.position, v.normal = o3d.skin_vertex(v, skin[i], joints)
			}
			job.custom_skinned_verts[first + i] = v
		}
		job.custom_skinned_first[di] = i32(first)
		// As at the shutter's opening, for the depth pass's twins.
		opened_joints := o3d.scene_draw_opened_joints(scene, di)
		opened_weights := o3d.scene_draw_opened_morphs(scene, di)
		if !reposed {
			continue
		}
		if opened_joints == nil {
			opened_joints = joints
		}
		skinned = len(skin) > 0 && len(opened_joints) > 0
		on := o3d.active_morphs(geo, opened_weights if opened_weights != nil else o3d.scene_draw_morphs(scene, di), &morphs) if len(geos.morph_deltas) > 0 else 0
		at := len(job.custom_skinned_verts)
		o3d.table_grow_non_zero(&job.custom_skinned_verts, at + len(verts))
		for vtx, i in verts {
			v := vtx
			if on > 0 {
				v.position, v.normal = o3d.morph_vertex(v, u32(i), deltas, morphs[:on])
			}
			if skinned {
				v.position, v.normal = o3d.skin_vertex(v, skin[i], opened_joints)
			}
			job.custom_skinned_verts[at + i] = v
		}
		job.custom_opened_first[di] = i32(at)
	}
	if len(job.custom_skinned_verts) == 0 {
		return
	}
	job.custom_skinned.uploaded = 0
	append(out, Upload{&job.custom_skinned, {.VERTEX}, mem.slice_to_bytes(job.custom_skinned_verts[:])})
}

@(private = "file")
make_draw_uniforms :: #force_inline proc "contextless" (mvp, model: o3d.Mat4, geo: o3d.Geometry, weights: []f32) -> (du: Draw_Uniforms) {
	du.mvp = mvp
	du.model = model
	du.normal = o3d.normal_matrix(model)
	morphs: [o3d.MAX_MORPH_TARGETS]o3d.Active_Morph
	nm := o3d.active_morphs(geo, weights, &morphs)
	for i in 0 ..< nm {
		du.morph_bases[i >> 2][i & 3] = morphs[i].base
		du.morph_weights[i >> 2][i & 3] = morphs[i].weight
	}
	du.morph_info = {u32(nm), 0, 0, 0}
	return
}

@(private = "file")
make_material_uniforms :: #force_inline proc "contextless" (m: ^o3d.Shading_Material, geo: o3d.Geometry) -> (mu: Material_Uniforms) {
	mu = {
		color        = {m.base.r, m.base.g, m.base.b, m.opacity},
		emissive     = {m.emissive.r, m.emissive.g, m.emissive.b, 0},
		metalness    = m.metalness,
		roughness    = m.roughness,
		ao_intensity = m.ao_intensity,
		flags        = m.flags,
		alpha_cutoff = m.alpha_cutoff,
		uv_sets      = m.uv_sets,
		curve_table  = geo.sprite_table if geo.mode == .Sprites else geo.curve_table,
		softness     = m.softness,
		layers        = {m.layers.f0, m.layers.clearcoat, m.layers.clearcoat_roughness, m.layers.sheen_roughness},
		sheen         = {m.layers.sheen.r, m.layers.sheen.g, m.layers.sheen.b, m.env_intensity},
		normal_scales = {m.normal_scale.x, m.normal_scale.y, m.layers.coat_normal_scale.x, m.layers.coat_normal_scale.y},
		transmission  = {m.layers.transmission, m.layers.thickness, m.layers.ior, 0},
		scale         = {m.layers.scale.x, m.layers.scale.y, m.layers.scale.z, 0},
		attenuation   = {m.layers.attenuation.r, m.layers.attenuation.g, m.layers.attenuation.b, 0},
		layer_info    = {m.layer_slots, 0, 0, 0},
	}
	for s, i in o3d.Texture_Slot {
		t := m.uv_transforms[s]
		mu.uv_transforms[i * 2] = {t[0].x, t[0].y, t[0].z, 0}
		mu.uv_transforms[i * 2 + 1] = {t[1].x, t[1].y, t[1].z, 0}
	}
	return
}

// The pools' uploads, sized then filled: each pool's GPU buffer grown to
// hold its data (a buffer always exists once uploaded to: some are bound
// with nothing in them), what each lacks laid end to end in `g.stream`
// (grown geometrically), mapped once, then copied from there.
@(private = "file")
upload_pools :: proc(g: ^GPU, uploads: []Upload, pass: ^sdl.GPUCopyPass) -> o3d.Error {
	total: u32
	for u in uploads {
		size, b := u32(len(u.data)), u.pool
		if size > b.capacity || b.buffer == nil {
			if b.buffer != nil {
				sdl.ReleaseGPUBuffer(g.device, b.buffer)
			}
			b.capacity = max(size, b.capacity * 2, 64 << 10)
			b.uploaded = 0
			b.buffer = sdl.CreateGPUBuffer(g.device, {usage = u.usage, size = b.capacity})
			if b.buffer == nil {
				b^ = {}
				return .Out_Of_Memory
			}
		}
		total += size - b.uploaded
	}
	if total == 0 {
		return nil
	}
	if total > g.stream_capacity || g.stream == nil {
		if g.stream != nil {
			sdl.ReleaseGPUTransferBuffer(g.device, g.stream)
		}
		g.stream_capacity = max(total, g.stream_capacity * 2, 1 << 20)
		g.stream = sdl.CreateGPUTransferBuffer(g.device, {usage = .UPLOAD, size = g.stream_capacity})
		if g.stream == nil {
			g.stream_capacity = 0
			return .Out_Of_Memory
		}
	}
	mapped := ([^]u8)(sdl.MapGPUTransferBuffer(g.device, g.stream, true))
	if mapped == nil {
		return .Device
	}
	at: u32
	for u in uploads {
		new := u.data[u.pool.uploaded:]
		mem.copy(&mapped[at], raw_data(new), len(new))
		at += u32(len(new))
	}
	sdl.UnmapGPUTransferBuffer(g.device, g.stream)
	at = 0
	for u in uploads {
		b := u.pool
		n := u32(len(u.data)) - b.uploaded
		if n > 0 {
			sdl.UploadToGPUBuffer(pass, {transfer_buffer = g.stream, offset = at}, {buffer = b.buffer, offset = b.uploaded, size = n}, b.uploaded == 0)
			at += n
		}
		b.uploaded = u32(len(u.data))
	}
	return nil
}

// A shadow map's pass: its casters' depth through `view_proj` into
// `map_tex` (cleared to 0, the far plane: depth is reversed), drawn with
// their other side (the frame's depth is draw_velocity's).
@(private = "file")
draw_depth :: proc(g: ^GPU, job: ^Job, draws: []int, view_proj: o3d.Mat4, map_tex, depth_tex: ^sdl.GPUTexture, cmd: ^sdl.GPUCommandBuffer) {
	scene, geos := job.scene, job.geos
	color := sdl.GPUColorTargetInfo{texture = map_tex, clear_color = {0, 0, 0, 0}, load_op = .CLEAR, store_op = .STORE}
	depth := sdl.GPUDepthStencilTargetInfo{texture = depth_tex, clear_depth = 0, load_op = .CLEAR, store_op = .DONT_CARE, stencil_load_op = .DONT_CARE, stencil_store_op = .DONT_CARE}
	pass := sdl.BeginGPURenderPass(cmd, &color, 1, &depth)
	defer sdl.EndGPURenderPass(pass)
	if g.vertices.buffer == nil && job.dynamic_vertices.buffer == nil {
		return
	}
	sdl.BindGPUVertexStorageBuffers(pass, 0, &g.morph_deltas.buffer, 1)
	bound: ^sdl.GPUGraphicsPipeline
	bound_vb: ^sdl.GPUBuffer
	bound_ib: ^sdl.GPUBuffer
	for di in draws {
		m := &job.materials[di]
		if .Alpha_Mask in m.flags && m.flags & {.Color_Map, .Alpha_Map} == {} && m.opacity < m.alpha_cutoff {
			continue
		}
		d := &scene.draws[di]
		geo, _ := o3d.geometry_get_any(geos, &job.expanded, d.geometry) // shadow_casters kept only draws with geometry
		dyn := d.geometry >= o3d.DYNAMIC_GEO_BIT
		vbuf := job.dynamic_vertices.buffer if dyn else g.vertices.buffer
		ibuf := job.dynamic_indices.buffer if dyn else g.indices.buffer
		if vbuf == nil || ibuf == nil {
			continue
		}
		if ibuf != bound_ib {
			sdl.BindGPUIndexBuffer(pass, {buffer = ibuf}, ._32BIT)
			bound_ib = ibuf
		}
		joints := o3d.scene_draw_joints(scene, di)
		skinned := !dyn && geo.skinned && len(joints) > 0 && g.skin_vertices.buffer != nil
		curved := .Curves in m.flags
		// The shadow pipelines cull by the other side.
		side := d.material.side
		p := g.shadow_skinned_pipelines[side] if skinned else g.shadow_pipelines[side]
		if curved {
			p = g.shadow_curve_pipelines[side]
		}
		if p != bound {
			sdl.BindGPUGraphicsPipeline(pass, p)
			bound = p
		}
		if curved {
			sdl.BindGPUFragmentStorageBuffers(pass, 0, &g.curve_words.buffer, 1)
		}
		mask_samplers := [2]sdl.GPUTextureSamplerBinding{tex_binding(g, m.color_map), tex_binding(g, m.alpha_map)}
		sdl.BindGPUFragmentSamplers(pass, 0, &mask_samplers[0], len(mask_samplers))
		du := make_draw_uniforms(view_proj * d.model, d.model, geo, o3d.scene_draw_morphs(scene, di))
		sdl.PushGPUVertexUniformData(cmd, 0, &du, size_of(du))
		mu := make_material_uniforms(m, geo)
		sdl.PushGPUFragmentUniformData(cmd, 0, &mu, size_of(mu))
		if skinned {
			vbs := [2]sdl.GPUBufferBinding {
				{buffer = vbuf, offset = geo.first_vertex * size_of(o3d.Vertex)},
				{buffer = g.skin_vertices.buffer, offset = geo.first_skin * size_of(o3d.Skin_Vertex)},
			}
			sdl.BindGPUVertexBuffers(pass, 0, &vbs[0], 2)
			bound_vb = nil
			ju: [o3d.MAX_JOINTS]o3d.Mat4
			copy(ju[:], joints)
			sdl.PushGPUVertexUniformData(cmd, 1, &ju, size_of(ju))
			sdl.DrawGPUIndexedPrimitives(pass, geo.index_count, 1, geo.first_index, 0, 0)
		} else if (du.morph_info.x > 0 || curved) && geo.first_vertex > 0 {
			// Vertices counted from 0 (SV_VertexID: morph deltas, curves' quads).
			sdl.BindGPUVertexBuffers(pass, 0, &sdl.GPUBufferBinding{buffer = vbuf, offset = geo.first_vertex * size_of(o3d.Vertex)}, 1)
			bound_vb = nil
			sdl.DrawGPUIndexedPrimitives(pass, geo.index_count, 1, geo.first_index, 0, 0)
		} else {
			if bound_vb != vbuf {
				sdl.BindGPUVertexBuffers(pass, 0, &sdl.GPUBufferBinding{buffer = vbuf}, 1)
				bound_vb = vbuf
			}
			sdl.DrawGPUIndexedPrimitives(pass, geo.index_count, 1, geo.first_index, i32(geo.first_vertex), 0)
		}
	}
}

// The frame's depth pass with motion blur: its depth map, and each
// pixel's velocity and nearness, under the draws the far plane moving only
// as the camera did (o3d.slang's fs_velocity_far), each draw where it is
// and where it was at the shutter's opening (its model, joints and morph
// weights then, each as now when it has none).
@(private = "file")
draw_velocity :: proc(g: ^GPU, job: ^Job, cmd: ^sdl.GPUCommandBuffer) {
	scene, geos := job.scene, job.geos
	colors := [3]sdl.GPUColorTargetInfo {
		{texture = job.depth_map, load_op = .CLEAR, store_op = .STORE},
		{texture = job.velocity_map, load_op = .CLEAR, store_op = .STORE},
		{texture = job.nearness_map, clear_color = {o3d.MOTION_FAR, 0, 0, 0}, load_op = .CLEAR, store_op = .STORE},
	}
	depth := sdl.GPUDepthStencilTargetInfo{texture = job.depth_buffer, clear_depth = 0, load_op = .CLEAR, store_op = .DONT_CARE, stencil_load_op = .DONT_CARE, stencil_store_op = .DONT_CARE}
	pass := sdl.BeginGPURenderPass(cmd, &colors[0], len(colors), &depth)
	defer sdl.EndGPURenderPass(pass)
	sdl.PushGPUFragmentUniformData(cmd, 1, &job.velocity, size_of(job.velocity))
	sdl.BindGPUGraphicsPipeline(pass, g.velocity_far_pipeline)
	sdl.DrawGPUPrimitives(pass, 3, 1, 0, 0)
	if g.vertices.buffer == nil && job.dynamic_vertices.buffer == nil {
		return
	}
	sdl.BindGPUVertexStorageBuffers(pass, 0, &g.morph_deltas.buffer, 1)
	bound: ^sdl.GPUGraphicsPipeline
	bound_vb, bound_ib: ^sdl.GPUBuffer
	for di in job.depth_draws {
		m := &job.materials[di]
		if .Alpha_Mask in m.flags && m.flags & {.Color_Map, .Alpha_Map} == {} && m.opacity < m.alpha_cutoff {
			continue
		}
		d := &scene.draws[di]
		geo, _ := o3d.geometry_get_any(geos, &job.expanded, d.geometry)
		dyn := d.geometry >= o3d.DYNAMIC_GEO_BIT
		vbuf := job.dynamic_vertices.buffer if dyn else g.vertices.buffer
		ibuf := job.dynamic_indices.buffer if dyn else g.indices.buffer
		if vbuf == nil || ibuf == nil {
			continue
		}
		if ibuf != bound_ib {
			sdl.BindGPUIndexBuffer(pass, {buffer = ibuf}, ._32BIT)
			bound_ib = ibuf
		}
		if m.shader != 0 {
			// Its own vertex stage, with its twins posed as at the opening.
			p := custom_depth_pipeline(g, job.shaders, m.shader, d.material.side, make = false)
			if p == nil {
				continue
			}
			if p != bound {
				sdl.BindGPUGraphicsPipeline(pass, p)
				bound = p
			}
			sh := o3d.shader_get(job.shaders, m.shader)
			for stage in o3d.Shader_Stage {
				at := job.depth_blocks.at[di][stage]
				if at < 0 {
					continue
				}
				if stage == .Vertex {
					sdl.PushGPUVertexUniformData(cmd, 0, &job.depth_blocks.bytes[at], u32(sh.depth.vertex.block_size))
				} else {
					sdl.PushGPUFragmentUniformData(cmd, 0, &job.depth_blocks.bytes[at], u32(sh.depth.fragment.block_size))
				}
			}
			now := sdl.GPUBufferBinding{buffer = vbuf, offset = geo.first_vertex * size_of(o3d.Vertex)}
			if f := job.custom_skinned_first[di]; f >= 0 && job.custom_skinned.buffer != nil {
				now = {buffer = job.custom_skinned.buffer, offset = u32(f) * size_of(o3d.Vertex)}
			}
			then := now
			if f := job.custom_opened_first[di]; f >= 0 && job.custom_skinned.buffer != nil {
				then.offset = u32(f) * size_of(o3d.Vertex)
			}
			vbs := [2]sdl.GPUBufferBinding{now, then}
			sdl.BindGPUVertexBuffers(pass, 0, &vbs[0], 2)
			bound_vb = nil
			sdl.DrawGPUIndexedPrimitives(pass, geo.index_count, 1, geo.first_index, 0, 0)
			continue
		}
		joints := o3d.scene_draw_joints(scene, di)
		skinned := !dyn && geo.skinned && len(joints) > 0 && g.skin_vertices.buffer != nil
		curved := .Curves in m.flags
		kind := 2 if curved else (1 if skinned else 0)
		p := g.velocity_pipelines[kind][d.material.side]
		if p != bound {
			sdl.BindGPUGraphicsPipeline(pass, p)
			bound = p
		}
		if curved {
			sdl.BindGPUFragmentStorageBuffers(pass, 0, &g.curve_words.buffer, 1)
		}
		mask_samplers := [2]sdl.GPUTextureSamplerBinding{tex_binding(g, m.color_map), tex_binding(g, m.alpha_map)}
		sdl.BindGPUFragmentSamplers(pass, 0, &mask_samplers[0], len(mask_samplers))
		weights := o3d.scene_draw_morphs(scene, di)
		du := make_draw_uniforms(job.view_proj * d.model, d.model, geo, weights)
		sdl.PushGPUVertexUniformData(cmd, 0, &du, size_of(du))
		model := o3d.scene_draw_opened(scene, di)
		opened_weights := o3d.scene_draw_opened_morphs(scene, di)
		ou := make_opened_uniforms(job.opened_view_proj * (model if model != {} else d.model), geo, opened_weights if opened_weights != nil else weights)
		sdl.PushGPUVertexUniformData(cmd, 2 if skinned else 1, &ou, size_of(ou))
		mu := make_material_uniforms(m, geo)
		sdl.PushGPUFragmentUniformData(cmd, 0, &mu, size_of(mu))
		if skinned {
			vbs := [2]sdl.GPUBufferBinding {
				{buffer = vbuf, offset = geo.first_vertex * size_of(o3d.Vertex)},
				{buffer = g.skin_vertices.buffer, offset = geo.first_skin * size_of(o3d.Skin_Vertex)},
			}
			sdl.BindGPUVertexBuffers(pass, 0, &vbs[0], 2)
			bound_vb = nil
			ju: [o3d.MAX_JOINTS]o3d.Mat4
			copy(ju[:], joints)
			sdl.PushGPUVertexUniformData(cmd, 1, &ju, size_of(ju))
			if oj := o3d.scene_draw_opened_joints(scene, di); oj != nil {
				copy(ju[:], oj)
			}
			sdl.PushGPUVertexUniformData(cmd, 3, &ju, size_of(ju))
			sdl.DrawGPUIndexedPrimitives(pass, geo.index_count, 1, geo.first_index, 0, 0)
		} else if (du.morph_info.x > 0 || ou.morph_info.x > 0 || curved) && geo.first_vertex > 0 {
			// Vertices counted from 0 (SV_VertexID: morph deltas, curves' quads).
			sdl.BindGPUVertexBuffers(pass, 0, &sdl.GPUBufferBinding{buffer = vbuf, offset = geo.first_vertex * size_of(o3d.Vertex)}, 1)
			bound_vb = nil
			sdl.DrawGPUIndexedPrimitives(pass, geo.index_count, 1, geo.first_index, 0, 0)
		} else {
			if bound_vb != vbuf {
				sdl.BindGPUVertexBuffers(pass, 0, &sdl.GPUBufferBinding{buffer = vbuf}, 1)
				bound_vb = vbuf
			}
			sdl.DrawGPUIndexedPrimitives(pass, geo.index_count, 1, geo.first_index, i32(geo.first_vertex), 0)
		}
	}
}

// A draw at the shutter's opening: `mvp` then and its morph targets'
// weights then (as make_draw_uniforms takes them).
@(private = "file")
make_opened_uniforms :: #force_inline proc "contextless" (mvp: o3d.Mat4, geo: o3d.Geometry, weights: []f32) -> (ou: Opened_Uniforms) {
	ou.mvp = mvp
	morphs: [o3d.MAX_MORPH_TARGETS]o3d.Active_Morph
	nm := o3d.active_morphs(geo, weights, &morphs)
	for i in 0 ..< nm {
		ou.morph_bases[i >> 2][i & 3] = morphs[i].base
		ou.morph_weights[i >> 2][i & 3] = morphs[i].weight
	}
	ou.morph_info = {u32(nm), 0, 0, 0}
	return
}

@(private = "file")
hdr_texture :: proc "contextless" (textures: ^o3d.Textures, id: o3d.Texture_Id) -> bool {
	tex, ok := o3d.texture_get(textures, id)
	return ok && tex.hdr
}

// Uploads what `textures` added since the last frame (all mip levels per
// texture, an HDR one's as half floats); an external texture's slot stays
// nil (its texture is bound).
@(private = "file")
upload_textures :: proc(g: ^GPU, textures: ^o3d.Textures, pass: ^sdl.GPUCopyPass) -> o3d.Error {
	if textures == nil || len(textures.list) <= len(g.textures) {
		return nil
	}
	for tex in textures.list[len(g.textures):] {
		if tex.external {
			append(&g.textures, nil)
			continue
		}
		gpu_tex := sdl.CreateGPUTexture(g.device, {
			type                 = .D2,
			format               = .R16G16B16A16_FLOAT if tex.hdr else COLOR_FORMAT,
			usage                = {.SAMPLER},
			width                = tex.width,
			height               = tex.height,
			layer_count_or_depth = 1,
			num_levels           = tex.levels,
			sample_count         = ._1,
		})
		if gpu_tex == nil {
			return .Out_Of_Memory
		}
		texel_size := u32(size_of([4]f16) if tex.hdr else size_of([4]u8))
		byte_size := tex.texel_count * texel_size
		transfer := sdl.CreateGPUTransferBuffer(g.device, {usage = .UPLOAD, size = byte_size})
		if transfer == nil {
			sdl.ReleaseGPUTexture(g.device, gpu_tex)
			return .Out_Of_Memory
		}
		mapped := sdl.MapGPUTransferBuffer(g.device, transfer, false)
		if tex.hdr {
			mem.copy(mapped, raw_data(textures.hdr_texels[tex.first_texel:]), int(byte_size))
		} else {
			mem.copy(mapped, raw_data(textures.texels[tex.first_texel:]), int(byte_size))
		}
		sdl.UnmapGPUTransferBuffer(g.device, transfer)
		for l in 0 ..< tex.levels {
			mw, mh := max(tex.width >> l, 1), max(tex.height >> l, 1)
			offset := (tex.mips[l] - tex.first_texel) * texel_size
			sdl.UploadToGPUTexture(
				pass,
				{transfer_buffer = transfer, offset = offset, pixels_per_row = mw, rows_per_layer = mh},
				{texture = gpu_tex, mip_level = l, w = mw, h = mh, d = 1},
				false,
			)
		}
		sdl.ReleaseGPUTransferBuffer(g.device, transfer)
		append(&g.textures, gpu_tex)
	}
	return nil
}

@(private = "file")
tex_binding :: #force_inline proc "contextless" (g: ^GPU, id: o3d.Texture_Id, sampler: ^sdl.GPUSampler = nil) -> sdl.GPUTextureSamplerBinding {
	tex := g.textures[id - 1] if id > 0 && int(id) <= len(g.textures) else g.dummy_texture
	if tex == nil {
		ext := g.externals[id - 1] if int(id) <= len(g.externals) else {}
		tex = ext.texture if ext.texture != nil && ext.generation == g.mirrored.textures else g.dummy_texture
	}
	return {texture = tex, sampler = sampler if sampler != nil else g.linear_sampler}
}

// Draws `scene` into `out` (made by `target`, `width x height`): `prepare`
// and `record` with the GPU's own job. With `cmd` the passes are recorded into
// it (the caller submits); else they are submitted and waited for.
render :: proc(g: ^GPU, scene: ^o3d.Scene, geos: ^o3d.Geometries, shaders: ^o3d.Shaders, out: ^sdl.GPUTexture, width, height: u32, cmd: ^sdl.GPUCommandBuffer = nil, textures: ^o3d.Textures = nil) -> o3d.Error {
	prepare(g, &g.job, scene, geos, shaders, width, height, textures) or_return
	if cmd != nil {
		return record(g, &g.job, out, cmd)
	}
	own := sdl.AcquireGPUCommandBuffer(g.device)
	if own == nil {
		return .Device
	}
	if err := record(g, &g.job, out, own); err != nil {
		_ = sdl.CancelGPUCommandBuffer(own)
		return err
	}
	fence := sdl.SubmitGPUCommandBufferAndAcquireFence(own)
	if fence == nil {
		return .Device
	}
	_ = sdl.WaitForGPUFences(g.device, true, &fence, 1)
	sdl.ReleaseGPUFence(g.device, fence)
	return nil
}

// Prepares `job` to draw `scene` at `width x height` (see `Job`): the frame
// targets, the job's shadow maps and buffers, the tables' mirrors and
// ShaderMaterials' pipelines are made, and what is new uploaded, in a command
// buffer it submits (SDL_GPU runs submissions in order, so before the
// host's command buffer that records the job).
prepare :: proc(g: ^GPU, job: ^Job, scene: ^o3d.Scene, geos: ^o3d.Geometries, shaders: ^o3d.Shaders, width, height: u32, textures: ^o3d.Textures = nil) -> o3d.Error {
	job.prepared = false
	if !ensure_targets(g, width, height) {
		return .Out_Of_Memory
	}
	sync_mirrors(g, geos, shaders, textures)
	job.env = o3d.shading_environment(scene, textures)
	bg_intensity, bg_lod: f32
	job.background, bg_intensity, bg_lod = o3d.shading_background(scene, textures)

	view, proj := o3d.camera_matrices(scene.camera, int(width), int(height))
	active := o3d.expand_draws(&job.expanded, scene, geos, view, proj, int(width), int(height))
	lights: [o3d.MAX_LIGHTS]o3d.Shading_Light
	nl := o3d.shading_lights(active.lights[:], &lights)
	o3d.shading_materials(o3d.scene_materials(active), &job.materials, textures)
	o3d.transmission_scale(job.materials[:len(active.draws)], active.draws.model[:len(active.draws)])
	o3d.sprite_materials(active, geos, &job.expanded, job.materials[:])
	// The mirror's view first: its receivers count for the shadows' fit.
	mv: o3d.Mirror_View
	mv, job.reflect = o3d.mirror_view(active, job.materials[:], geos, view, proj, int(width), int(height), &job.expanded)
	views := [2]o3d.Mat4{proj * view, proj * mv.view}
	ns := o3d.frame_shadows(active, geos, shaders, views[:1 + int(job.reflect)], &job.shadows, &job.expanded)
	o3d.shader_blocks(active, shaders, view, proj, job.shadows[:ns], &job.blocks)
	for si in 0 ..< ns {
		if !ensure_shadow(g, job, si, u32(job.shadows[si].size)) {
			return .Out_Of_Memory
		}
		o3d.shadow_casters(active, geos, shaders, &job.shadows[si], &job.casters[si], &job.expanded)
		for &l, li in lights[:nl] {
			if li == job.shadows[si].light {
				l.shadowed = u32(si + 1)
			}
		}
	}
	// The frame's depth (with its velocity and nearness: where each pixel
	// was at the shutter's opening, none moving without motion blur), each
	// ShaderMaterial by its own vertex stage.
	job.motion = o3d.motion_of(active)
	job.depth = o3d.needs_depth(active) || job.motion
	if job.depth {
		if !ensure_depth(g, job, width, height) || !ensure_motion(g, job, width, height, job.motion) {
			return .Out_Of_Memory
		}
		o3d.depth_draws(active, geos, shaders, proj * view, &job.depth_draws, &job.expanded)
		opened := o3d.opened_camera(active.camera) or_else active.camera
		ov, op := o3d.camera_matrices(opened, int(width), int(height))
		job.opened_view_proj = op * ov
		near := active.camera.near if active.camera.near > 0 else 0.1
		far := active.camera.far if active.camera.far > near else near * 1e4
		job.velocity = {
			frame      = {f32(width), f32(height), near, far},
			flags      = {u32(active.camera.orthographic), 0, 0, 0},
			opened_far = job.opened_view_proj * linalg.inverse(proj * view),
		}
		job.motion_params = o3d.motion_params(active.camera, int(height))
		o3d.depth_blocks(active, shaders, view, proj, ov, op, int(width), int(height), job.shadows[:ns], &job.depth_blocks)
		for di in job.depth_draws {
			if m := &job.materials[di]; m.shader != 0 {
				custom_depth_pipeline(g, shaders, m.shader, active.draws[di].material.side)
			}
		}
	}
	exposure := active.tone_mapping_exposure if active.tone_mapping_exposure > 0 else 1
	job.bg_params.params = {bg_intensity, bg_lod, 0, 0}
	job.bg_params.flags = {u32(hdr_texture(textures, job.background)), 0, 0, 0}
	job.clear = o3d.background_clear_light(scene)
	job.post.tone = {f32(active.tone_mapping), exposure, 0, 0}
	lu := &job.lighting
	lu^ = {}
	lu.eye = {active.camera.position.x, active.camera.position.y, active.camera.position.z, 0}
	lu.count = {i32(nl), 0, i32(active.fog.kind), i32(ns)}
	for si in 0 ..< ns {
		lu.shadow_matrix[si] = job.shadows[si].view_proj
		lu.shadow_params[si] = {f32(job.shadows[si].size), job.shadows[si].bias, job.shadows[si].normal_bias, 0}
		lu.shadow_from[si] = job.shadows[si].from
	}
	env := job.env
	fog_near, fog_far, fog_density := o3d.fog_terms(active.fog)
	lu.env_params = {env.intensity if env.radiance != 0 else 0, env.max_lod if env.radiance != 0 else 0, 0, fog_density}
	lu.env_flags = {u32(hdr_texture(textures, env.radiance)), u32(active.camera.orthographic), 0, 0}
	near := active.camera.near if active.camera.near > 0 else 0.1
	far := active.camera.far if active.camera.far > near else near * 1e4
	lu.frame_params = {f32(width), f32(height), near, far}
	focus: o3d.Focus
	focus, job.focus = o3d.focus_of(active.camera, int(height))
	if job.focus {
		if !ensure_focus(g, job, width, height) {
			return .Out_Of_Memory
		}
		tiles := [2]f32{f32((width + o3d.FOCUS_TILE - 1) / o3d.FOCUS_TILE), f32((height + o3d.FOCUS_TILE - 1) / o3d.FOCUS_TILE)}
		job.post.lens = {focus.focus, focus.blur, focus.max_blur, math.ceil(focus.max_blur / o3d.FOCUS_TILE)}
		job.post.camera = {near, far, f32(u32(active.camera.orthographic)), 0}
		job.post.frame = {f32(width), f32(height), tiles.x, tiles.y}
	}
	lu.view_proj = proj * view
	if active.fog.kind != .None {
		fc := o3d.linear(active.fog.color)
		lu.fog_color = {fc.r, fc.g, fc.b, 0}
		lu.fog_params = {fog_near, fog_far, 0, 0}
	}
	for l, i in lights[:nl] {
		lu.lights[i] = {
			color    = {l.color.r, l.color.g, l.color.b, l.cos_outer},
			ground   = {l.ground.r, l.ground.g, l.ground.b, l.cos_inner},
			vector   = {l.vector.x, l.vector.y, l.vector.z, 0},
			kind     = u32(l.kind),
			distance = l.distance,
			decay    = l.decay,
			shadowed = u32(l.shadowed),
		}
	}
	if job.reflect {
		if !ensure_reflection(g, job, u32(mv.width), u32(mv.height), u32(mv.levels)) || !ensure_mirror_pass(g) {
			return .Out_Of_Memory
		}
		o3d.mirror_order(active, geos, shaders, job.materials[:], mv.view, proj, &job.mirror_order, &job.expanded)
		o3d.shader_blocks(active, shaders, mv.view, proj, job.shadows[:ns], &job.mirror_blocks, mv.eye)
		job.mirror_view_proj = proj * mv.view
		lu.reflect_params = {f32(mv.levels - 1), mv.focal, f32(int(active.camera.orthographic)), 0}
		// The mirror's pass: through the mirrored camera at its size, what
		// is behind the plane left out.
		ml := &job.mirror_lighting
		ml^ = job.lighting
		ml.eye = {mv.eye.x, mv.eye.y, mv.eye.z, 0}
		ml.frame_params[0], ml.frame_params[1] = f32(mv.width), f32(mv.height)
		ml.view_proj = job.mirror_view_proj
		ml.clip_plane = mv.plane
		for di in job.mirror_order {
			m := &job.materials[di]
			if m.shader != 0 {
				custom_pipeline(g, shaders, m.shader, active.draws[di].material.side, o3d.blend_of(m), mirrored = true)
			}
		}
	}
	job.snapshot = o3d.draw_order(active, geos, shaders, view, proj, &job.order, &job.expanded)
	if job.snapshot < len(job.order) && !ensure_snapshot(g, job, width, height) {
		return .Out_Of_Memory
	}
	for di in job.order {
		m := &job.materials[di]
		if m.shader != 0 {
			custom_pipeline(g, shaders, m.shader, active.draws[di].material.side, o3d.blend_of(m))
		}
	}

	cmd := sdl.AcquireGPUCommandBuffer(g.device)
	if cmd == nil {
		return .Device
	}
	pass := sdl.BeginGPUCopyPass(cmd)
	uploads: [dynamic; 9]Upload
	upload_geometry(g, geos, &uploads)
	upload_dynamic(job, &uploads)
	upload_custom_skinned(job, active, geos, &uploads)
	err := upload_pools(g, uploads[:], pass)
	if err == nil {
		err = upload_textures(g, textures, pass)
	}
	sdl.EndGPUCopyPass(pass)
	if err != nil {
		// None of the pass's copies happen: the geometry and texture
		// mirrors start over, so the next prepare uploads them again rather
		// than drawing from what was counted as uploaded.
		_ = sdl.CancelGPUCommandBuffer(cmd)
		g.mirrored.geometry, g.mirrored.textures = 0, 0
		return err
	}
	if !sdl.SubmitGPUCommandBuffer(cmd) {
		return .Device
	}
	job.scene, job.geos, job.shaders = active, geos, shaders
	job.mirrored = g.mirrored
	job.width, job.height = width, height
	job.view_proj = proj * view
	job.shadow_count = ns
	job.prepared = true
	return nil
}

// Records `job` (prepared by `prepare`) into `cmd` and copies the frame to
// `out` (`target`'s, at least the prepared size): the shadow passes, the main
// pass and the copy, making nothing. A job records once per prepare;
// `.Not_Prepared` when it was not prepared, or another job's prepare has
// since started the tables' mirrors over (other tables) or the targets are
// smaller than it.
record :: proc(g: ^GPU, job: ^Job, out: ^sdl.GPUTexture, cmd: ^sdl.GPUCommandBuffer) -> o3d.Error {
	if !job.prepared || job.mirrored != g.mirrored || job.width > g.capacity.x || job.height > g.capacity.y {
		return .Not_Prepared
	}
	job.prepared = false
	width, height := job.width, job.height
	for si in 0 ..< job.shadow_count {
		draw_depth(g, job, job.casters[si][:], job.shadows[si].view_proj, job.shadow_maps[si], job.shadow_depths[si], cmd)
	}
	if job.depth {
		draw_velocity(g, job, cmd)
	}
	shadow_maps: [o3d.MAX_SHADOWS]^sdl.GPUTexture
	for &m, si in shadow_maps {
		m = job.shadow_maps[si] if si < job.shadow_count else g.dummy_shadow
	}
	// The resolved light a pass leaves (the colour target itself without
	// multisampling).
	light := g.resolved if g.samples != ._1 else g.color
	// A pass of light into the frame's targets' top-left w x h: cleared to
	// `clear`, the background under the draws when there is one, resolved.
	light_pass :: proc(g: ^GPU, job: ^Job, cmd: ^sdl.GPUCommandBuffer, w, h: u32, clear: [4]f32, background: bool, order: []int, shadow_maps: ^[o3d.MAX_SHADOWS]^sdl.GPUTexture, mirrored: bool) {
		// The mirror's pass draws its runs too (none where nothing is).
		colors := [2]sdl.GPUColorTargetInfo {
			{texture = g.color, clear_color = sdl.FColor(clear), load_op = .CLEAR, store_op = .STORE},
			{texture = job.runs_target, load_op = .CLEAR, store_op = .STORE},
		}
		if g.samples != ._1 {
			colors[0].store_op, colors[0].resolve_texture = .RESOLVE, g.resolved
			colors[1].store_op, colors[1].resolve_texture = .RESOLVE, job.runs_resolved
		}
		depth := sdl.GPUDepthStencilTargetInfo{texture = g.depth, clear_depth = 0, load_op = .CLEAR, store_op = .DONT_CARE, stencil_load_op = .DONT_CARE, stencil_store_op = .DONT_CARE}
		pass := sdl.BeginGPURenderPass(cmd, &colors[0], 2 if mirrored else 1, &depth)
		defer sdl.EndGPURenderPass(pass)
		sdl.SetGPUViewport(pass, {x = 0, y = 0, w = f32(w), h = f32(h), min_depth = 0, max_depth = 1})
		sdl.SetGPUScissor(pass, {x = 0, y = 0, w = i32(w), h = i32(h)})
		if background {
			sdl.BindGPUGraphicsPipeline(pass, g.bg_pipeline)
			bg_binding := tex_binding(g, job.background, g.env_sampler)
			sdl.BindGPUFragmentSamplers(pass, 0, &bg_binding, 1)
			inv_vp := linalg.inverse(job.view_proj)
			sdl.PushGPUVertexUniformData(cmd, 0, &inv_vp, size_of(inv_vp))
			sdl.PushGPUFragmentUniformData(cmd, 0, &job.bg_params, size_of(job.bg_params))
			sdl.DrawGPUPrimitives(pass, 3, 1, 0, 0)
		}
		draw_list(g, job, pass, cmd, order, shadow_maps, mirrored)
	}
	if job.reflect {
		// The scene through the mirror at its size, copied into the
		// reflection and mipmapped; nothing where it shows nothing.
		w, h := job.reflection_size.x, job.reflection_size.y
		light_pass(g, job, cmd, w, h, {}, false, job.mirror_order[:], &shadow_maps, true)
		copy_pass := sdl.BeginGPUCopyPass(cmd)
		sdl.CopyGPUTextureToTexture(copy_pass, {texture = light}, {texture = job.reflection_tex}, w, h, 1, false)
		sdl.EndGPUCopyPass(copy_pass)
		// The runs into the second layer's red and green (alpha 1), exactly.
		runs := job.runs_resolved if g.samples != ._1 else job.runs_target
		sdl.BlitGPUTexture(cmd, {
			source = {texture = runs, w = w, h = h},
			destination = {texture = job.reflection_tex, layer_or_depth_plane = 1, w = w, h = h},
			load_op = .DONT_CARE,
			filter = .NEAREST,
		})
		sdl.GenerateMipmapsForGPUTexture(cmd, job.reflection_tex)
	}
	if job.snapshot < len(job.order) {
		// The opaque draws before the transmissive ones, copied into the
		// snapshot and mipmapped (three.js's transmission pass).
		light_pass(g, job, cmd, width, height, job.clear, job.background != 0, job.order[:job.snapshot], &shadow_maps, false)
		copy_pass := sdl.BeginGPUCopyPass(cmd)
		sdl.CopyGPUTextureToTexture(copy_pass, {texture = light}, {texture = job.snapshot_tex}, width, height, 1, false)
		sdl.EndGPUCopyPass(copy_pass)
		sdl.GenerateMipmapsForGPUTexture(cmd, job.snapshot_tex)
	}
	light_pass(g, job, cmd, width, height, job.clear, job.background != 0, job.order[:], &shadow_maps, false)
	// The frame's light blurred along its motion, then tone mapped and
	// encoded (through its depth of field: its tables, then the gather as
	// the output), then copied out.
	if job.motion {
		record_motion_blur(g, &job.motion_tables, cmd, light, job.velocity_map, job.nearness_map, job.blurred, width, height, job.motion_params, g.capacity) or_return
		light = job.blurred
	}
	job.post.tone.zw = {f32(g.capacity.x), f32(g.capacity.y)}
	if job.focus {
		tiles, near, half, half_keys := job.focus_tables[0], job.focus_tables[1], job.focus_tables[2], job.focus_tables[3]
		tile_size := [2]u32{u32(job.post.frame.z), u32(job.post.frame.w)}
		half_size := [2]u32{(width + 1) / 2, (height + 1) / 2}
		post_draw(g, cmd, g.focus_pipelines[.Tiles], {tiles}, tile_size, #partial Post_Bindings{.Depth = job.depth_map}, &job.post)
		post_draw(g, cmd, g.focus_pipelines[.Near], {near}, tile_size, #partial Post_Bindings{.Tiles = tiles}, &job.post)
		post_draw(g, cmd, g.focus_pipelines[.Half], {half, half_keys}, half_size, #partial Post_Bindings{.Light = light, .Depth = job.depth_map}, &job.post)
		post_draw(g, cmd, g.focus_pipelines[.Gather], {g.output}, {width, height}, #partial Post_Bindings{.Light = light, .Depth = job.depth_map, .Tiles = near, .Half = half, .Half_Keys = half_keys}, &job.post)
	} else {
		post_draw(g, cmd, g.output_pipeline, {g.output}, {width, height}, #partial Post_Bindings{.Light = light}, &job.post)
	}
	copy_pass := sdl.BeginGPUCopyPass(cmd)
	sdl.CopyGPUTextureToTexture(copy_pass, {texture = g.output}, {texture = out}, width, height, 1, false)
	sdl.EndGPUCopyPass(copy_pass)
	return nil
}

// The draws `order` (the job's) into the open pass: the frame's (the
// transmission snapshot's is its first part) or the mirror's (`mirrored`:
// its pipelines, lighting and camera).
@(private = "file")
draw_list :: proc(g: ^GPU, job: ^Job, pass: ^sdl.GPURenderPass, cmd: ^sdl.GPUCommandBuffer, order: []int, shadow_maps: ^[o3d.MAX_SHADOWS]^sdl.GPUTexture, mirrored: bool) {
	scene, geos, shaders := job.scene, job.geos, job.shaders
	bound: ^sdl.GPUGraphicsPipeline
	lighting := &job.lighting
	view_proj := job.view_proj
	blocks := &job.blocks
	if mirrored {
		lighting, view_proj, blocks = &job.mirror_lighting, job.mirror_view_proj, &job.mirror_blocks
	}
	sdl.PushGPUFragmentUniformData(cmd, 1, lighting, size_of(job.lighting))
	sdl.BindGPUVertexStorageBuffers(pass, 0, &g.morph_deltas.buffer, 1)
	bound_vb: ^sdl.GPUBuffer
	bound_ib: ^sdl.GPUBuffer
	if g.vertices.buffer != nil && g.indices.buffer != nil {
		sdl.BindGPUVertexBuffers(pass, 0, &sdl.GPUBufferBinding{buffer = g.vertices.buffer}, 1)
		sdl.BindGPUIndexBuffer(pass, {buffer = g.indices.buffer}, ._32BIT)
		bound_vb = g.vertices.buffer
		bound_ib = g.indices.buffer
	}
	for di in order {
		d := &scene.draws[di]
		geo, _ := o3d.geometry_get_any(geos, &job.expanded, d.geometry) // draw_order kept only draws with geometry
		dyn := d.geometry >= o3d.DYNAMIC_GEO_BIT
		vbuf := job.dynamic_vertices.buffer if dyn else g.vertices.buffer
		ibuf := job.dynamic_indices.buffer if dyn else g.indices.buffer
		if vbuf == nil || ibuf == nil {
			continue
		}
		if ibuf != bound_ib {
			sdl.BindGPUIndexBuffer(pass, {buffer = ibuf}, ._32BIT)
			bound_ib = ibuf
		}
		m := &job.materials[di]
		blend := o3d.blend_of(m)
		if m.shader != 0 {
			p := custom_pipeline(g, shaders, m.shader, d.material.side, blend, make = false, mirrored = mirrored)
			if p == nil {
				continue
			}
			if p != bound {
				sdl.BindGPUGraphicsPipeline(pass, p)
				bound = p
			}
			if n := o3d.shader_module(o3d.shader_get(shaders, m.shader), .Fragment).num_samplers; n > 0 {
				maps: [o3d.SHADER_MAPS]sdl.GPUTextureSamplerBinding
				for id, c in m.shader_maps {
					maps[c] = tex_binding(g, id)
				}
				// shadowAt's: the first shadow map (unshadowed without one).
				if c := o3d.shader_get(shaders, m.shader).shadow_channel; c >= 0 {
					maps[c] = {texture = shadow_maps[0], sampler = g.shadow_sampler}
				}
				sdl.BindGPUFragmentSamplers(pass, 0, &maps[0], u32(min(n, len(maps))))
			}
			for stage in o3d.Shader_Stage {
				at := blocks.at[di][stage]
				if at < 0 {
					continue
				}
				size := u32(o3d.shader_module(o3d.shader_get(shaders, m.shader), stage).block_size)
				if stage == .Vertex {
					sdl.PushGPUVertexUniformData(cmd, 0, &blocks.bytes[at], size)
				} else {
					sdl.PushGPUFragmentUniformData(cmd, 0, &blocks.bytes[at], size)
				}
			}
			if custom_first := job.custom_skinned_first[di]; custom_first >= 0 && job.custom_skinned.buffer != nil {
				if bound_vb != job.custom_skinned.buffer {
					sdl.BindGPUVertexBuffers(pass, 0, &sdl.GPUBufferBinding{buffer = job.custom_skinned.buffer}, 1)
					bound_vb = job.custom_skinned.buffer
				}
				sdl.DrawGPUIndexedPrimitives(pass, geo.index_count, 1, geo.first_index, custom_first, 0)
			} else {
				if bound_vb != vbuf {
					sdl.BindGPUVertexBuffers(pass, 0, &sdl.GPUBufferBinding{buffer = vbuf}, 1)
					bound_vb = vbuf
				}
				sdl.DrawGPUIndexedPrimitives(pass, geo.index_count, 1, geo.first_index, i32(geo.first_vertex), 0)
			}
			continue
		}
		joints := o3d.scene_draw_joints(scene, di)
		skinned := !dyn && geo.skinned && len(joints) > 0 && g.skin_vertices.buffer != nil
		curved := .Curves in m.flags
		sprite := .Sprite in m.flags && job.sprite_rows.buffer != nil
		by := Draw_Kind.Skinned if skinned else .Standard
		if curved {
			by = .Curves
		}
		if sprite {
			by = .Sprite
		}
		p := g.mirror_pipelines[by][d.material.side][blend] if mirrored else g.pipelines[by][d.material.side][blend]
		if p != bound {
			sdl.BindGPUGraphicsPipeline(pass, p)
			bound = p
		}
		if curved {
			sdl.BindGPUFragmentStorageBuffers(pass, 0, &g.curve_words.buffer, 1)
		}
		if sprite {
			sdl.BindGPUFragmentStorageBuffers(pass, 0, &job.sprite_rows.buffer, 1)
		}
		samplers := [16]sdl.GPUTextureSamplerBinding {
			tex_binding(g, m.color_map),
			{texture = shadow_maps[0], sampler = g.shadow_sampler},
			tex_binding(g, m.normal_map),
			tex_binding(g, m.roughness_map),
			tex_binding(g, m.emissive_map),
			tex_binding(g, job.env.radiance, g.env_sampler),
			tex_binding(g, job.env.irradiance, g.env_sampler),
			tex_binding(g, m.metalness_map),
			tex_binding(g, m.ao_map),
			{texture = shadow_maps[1], sampler = g.shadow_sampler},
			{texture = shadow_maps[2], sampler = g.shadow_sampler},
			{texture = shadow_maps[3], sampler = g.shadow_sampler},
			tex_binding(g, m.alpha_map),
			// Sprites: the frame's depth (soft particles); transmissive
			// draws: the frame's snapshot; reflective ones: the reflection.
			{texture = job.depth_map if job.depth && job.depth_map != nil else g.dummy_shadow, sampler = g.shadow_sampler},
			// The layers' two textures (white for none).
			tex_binding(g, m.layer_textures[0]),
			tex_binding(g, m.layer_textures[1]),
		}
		if !sprite {
			frame := g.dummy_texture
			if .Transmission in m.flags {
				// None through the mirror (no transmissive draw is in the
				// snapshot's own part).
				frame = job.snapshot_tex if !mirrored && job.snapshot_tex != nil else g.clear_texture
			} else if .Reflection in m.flags {
				frame = job.reflection_tex if job.reflect else g.clear_texture
			}
			samplers[13] = {texture = frame, sampler = g.frame_sampler}
		}
		sdl.BindGPUFragmentSamplers(pass, 0, &samplers[0], 16)
		du := make_draw_uniforms(view_proj * d.model, d.model, geo, o3d.scene_draw_morphs(scene, di))
		sdl.PushGPUVertexUniformData(cmd, 0, &du, size_of(du))
		mu := make_material_uniforms(m, geo)
		if .Triplanar in m.flags {
			f := o3d.projection_frame(d.material.projection, d.model)
			mu.triplanar = {m.projection_scale, m.projection_sharpness, 0, 0}
			for r in 0 ..< 3 {
				mu.to_local[r] = {f.to_local[r, 0], f.to_local[r, 1], f.to_local[r, 2], f.to_local[r, 3]}
				mu.normal_to_local[r] = {f.normal_to_local[r, 0], f.normal_to_local[r, 1], f.normal_to_local[r, 2], 0}
			}
		}
		sdl.PushGPUFragmentUniformData(cmd, 0, &mu, size_of(mu))
		if skinned {
			vbs := [2]sdl.GPUBufferBinding {
				{buffer = vbuf, offset = geo.first_vertex * size_of(o3d.Vertex)},
				{buffer = g.skin_vertices.buffer, offset = geo.first_skin * size_of(o3d.Skin_Vertex)},
			}
			sdl.BindGPUVertexBuffers(pass, 0, &vbs[0], 2)
			bound_vb = nil
			ju: [o3d.MAX_JOINTS]o3d.Mat4
			copy(ju[:], joints)
			sdl.PushGPUVertexUniformData(cmd, 1, &ju, size_of(ju))
			sdl.DrawGPUIndexedPrimitives(pass, geo.index_count, 1, geo.first_index, 0, 0)
		} else if (du.morph_info.x > 0 || curved || sprite) && geo.first_vertex > 0 {
			// Vertices counted from 0 (SV_VertexID: morph deltas, curves' and sprites' quads).
			sdl.BindGPUVertexBuffers(pass, 0, &sdl.GPUBufferBinding{buffer = vbuf, offset = geo.first_vertex * size_of(o3d.Vertex)}, 1)
			bound_vb = nil
			sdl.DrawGPUIndexedPrimitives(pass, geo.index_count, 1, geo.first_index, 0, 0)
		} else {
			if bound_vb != vbuf {
				sdl.BindGPUVertexBuffers(pass, 0, &sdl.GPUBufferBinding{buffer = vbuf}, 1)
				bound_vb = vbuf
			}
			sdl.DrawGPUIndexedPrimitives(pass, geo.index_count, 1, geo.first_index, i32(geo.first_vertex), 0)
		}
	}
}
// The frame's depth map from the job's last render of a scene with
// `Scene.depth` (R32F, `width x height` of that render, rows from the top;
// nil without): to bind in the host's own passes (depth of field, haze).
job_depth :: proc "contextless" (job: ^Job) -> ^sdl.GPUTexture {
	return job.depth_map if job.depth else nil
}

// Records into `cmd` (after `record` of the same job, in the same command
// buffer) the frame's depth as distance into the host's `target`: an
// RGBA16F texture (`DEPTH_IMAGE_FORMAT`, `.COLOR_TARGET` usage, the
// render's `width x height` at its top left), its rgb the distance along
// the camera's axis (`o3d.view_depth`, world units) and a 1 where something
// was drawn; rgb `far` and a 0 where nothing was. `o3d.depth_image` makes
// the software renderer's. `.Not_Prepared` for a render without
// `Scene.depth`.
record_depth_into :: proc(g: ^GPU, job: ^Job, target: ^sdl.GPUTexture, far: f32, cmd: ^sdl.GPUCommandBuffer) -> o3d.Error {
	if !job.depth || job.depth_map == nil || target == nil {
		return .Not_Prepared
	}
	color := sdl.GPUColorTargetInfo{texture = target, load_op = .DONT_CARE, store_op = .STORE}
	pass := sdl.BeginGPURenderPass(cmd, &color, 1, nil)
	defer sdl.EndGPURenderPass(pass)
	w, h := job.depth_size.x, job.depth_size.y
	sdl.SetGPUViewport(pass, {x = 0, y = 0, w = f32(w), h = f32(h), min_depth = 0, max_depth = 1})
	sdl.SetGPUScissor(pass, {x = 0, y = 0, w = i32(w), h = i32(h)})
	sdl.BindGPUGraphicsPipeline(pass, g.depth_pipeline)
	binding := sdl.GPUTextureSamplerBinding{texture = job.depth_map, sampler = g.shadow_sampler}
	sdl.BindGPUFragmentSamplers(pass, 0, &binding, 1)
	fp := job.lighting.frame_params
	params := struct {
		params: [4]f32,
		size:   [4]f32,
	}{{fp.z, fp.w, far, f32(job.lighting.env_flags.y)}, {f32(w), f32(h), 0, 0}}
	sdl.PushGPUFragmentUniformData(cmd, 0, &params, size_of(params))
	sdl.DrawGPUPrimitives(pass, 3, 1, 0, 0)
	return nil
}

// `job_depth` of the GPU's own job (`render`).
depth :: proc "contextless" (g: ^GPU) -> ^sdl.GPUTexture {
	return job_depth(&g.job)
}

// A depth map's values (`width x height` floats), as `o3d.software_depth`
// gives the software renderer's.
read_depth :: proc(g: ^GPU, tex: ^sdl.GPUTexture, width, height: u32, allocator := context.allocator) -> (depths: []f32, err: o3d.Error) {
	if tex == nil {
		return nil, .Not_Prepared
	}
	bytes := read(g, tex, width, height, allocator) or_return
	return slice.reinterpret([]f32, bytes), nil
}

// The texture's pixels (RGBA8 premultiplied), for tests and saving.
read :: proc(g: ^GPU, tex: ^sdl.GPUTexture, width, height: u32, allocator := context.allocator) -> (pixels: []u8, err: o3d.Error) {
	return download(g, tex, width, height, 4, allocator)
}

// A `record_depth_into` image's values (`width x height` RGBA halves), as
// `o3d.depth_image` gives the software renderer's.
read_depth_image :: proc(g: ^GPU, tex: ^sdl.GPUTexture, width, height: u32, allocator := context.allocator) -> (values: []f16, err: o3d.Error) {
	if tex == nil {
		return nil, .Not_Prepared
	}
	bytes := download(g, tex, width, height, size_of([4]f16), allocator) or_return
	return slice.reinterpret([]f16, bytes), nil
}

// A texture's `width x height` texels of `texel_size` bytes.
@(private = "file")
download :: proc(g: ^GPU, tex: ^sdl.GPUTexture, width, height, texel_size: u32, allocator := context.allocator) -> (pixels: []u8, err: o3d.Error) {
	size := width * height * texel_size
	transfer := sdl.CreateGPUTransferBuffer(g.device, {usage = .DOWNLOAD, size = size})
	if transfer == nil {
		return nil, .Out_Of_Memory
	}
	defer sdl.ReleaseGPUTransferBuffer(g.device, transfer)
	cmd := sdl.AcquireGPUCommandBuffer(g.device)
	pass := sdl.BeginGPUCopyPass(cmd)
	sdl.DownloadFromGPUTexture(pass, {texture = tex, w = width, h = height, d = 1}, {transfer_buffer = transfer, pixels_per_row = width, rows_per_layer = height})
	sdl.EndGPUCopyPass(pass)
	fence := sdl.SubmitGPUCommandBufferAndAcquireFence(cmd)
	if fence == nil {
		return nil, .Device
	}
	_ = sdl.WaitForGPUFences(g.device, true, &fence, 1)
	sdl.ReleaseGPUFence(g.device, fence)
	mapped := ([^]u8)(sdl.MapGPUTransferBuffer(g.device, transfer, false))
	defer sdl.UnmapGPUTransferBuffer(g.device, transfer)
	pixels = make([]u8, int(size), allocator)
	copy(pixels, mapped[:size])
	return pixels, nil
}
