package o3d_sdlgpu

import "core:math/linalg"
import "core:mem"
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
		g.mirrored.geometry = geos.generation
	}
	if textures != nil && textures.generation != g.mirrored.textures {
		for tex in g.textures {
			sdl.ReleaseGPUTexture(g.device, tex)
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

// Uploads what `geos` added since the last frame (in a copy pass of `cmd`).
@(private)
upload_geometry :: proc(g: ^GPU, geos: ^o3d.Geometries, cmd: ^sdl.GPUCommandBuffer) -> o3d.Error {
	vertices := mem.slice_to_bytes(geos.vertices[:])
	skin_vertices := mem.slice_to_bytes(geos.skin_vertices[:])
	morph_deltas := mem.slice_to_bytes(geos.morph_deltas[:])
	indices := mem.slice_to_bytes(geos.indices[:])
	if u32(len(vertices)) == g.vertices.uploaded && u32(len(skin_vertices)) == g.skin_vertices.uploaded && u32(len(morph_deltas)) == g.morph_deltas.uploaded && u32(len(indices)) == g.indices.uploaded {
		return nil
	}
	pass := sdl.BeginGPUCopyPass(cmd)
	defer sdl.EndGPUCopyPass(pass)
	upload_pool(g, &g.vertices, {.VERTEX}, vertices, pass) or_return
	upload_pool(g, &g.skin_vertices, {.VERTEX}, skin_vertices, pass) or_return
	upload_pool(g, &g.morph_deltas, {.GRAPHICS_STORAGE_READ}, morph_deltas, pass) or_return
	upload_pool(g, &g.indices, {.INDEX}, indices, pass) or_return
	return nil
}

// Uploads per-frame expanded line/point geometry when present.
@(private = "file")
upload_dynamic :: proc(g: ^GPU, cmd: ^sdl.GPUCommandBuffer) -> o3d.Error {
	if len(g.expanded.vertices) == 0 || len(g.expanded.indices) == 0 {
		return nil
	}
	g.dynamic_vertices.uploaded = 0
	g.dynamic_indices.uploaded = 0
	pass := sdl.BeginGPUCopyPass(cmd)
	defer sdl.EndGPUCopyPass(pass)
	upload_pool(g, &g.dynamic_vertices, {.VERTEX}, mem.slice_to_bytes(g.expanded.vertices[:]), pass) or_return
	upload_pool(g, &g.dynamic_indices, {.INDEX}, mem.slice_to_bytes(g.expanded.indices[:]), pass) or_return
	return nil
}

// When a morphed or skinned draw uses a ShaderMaterial, morphs and skins its
// vertices into `g.custom_skinned` before the render pass so the user's vertex
// shader receives posed positions and normals.
@(private = "file")
upload_custom_skinned :: proc(g: ^GPU, scene: ^o3d.Scene, geos: ^o3d.Geometries, cmd: ^sdl.GPUCommandBuffer) -> o3d.Error {
	clear(&g.custom_skinned_verts)
	o3d.table_grow_non_zero(&g.custom_skinned_first, len(scene.draws))
	for &f in g.custom_skinned_first {
		f = -1
	}
	for di in g.order {
		if g.materials[di].shader == 0 {
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
		if !morphed && !skinned {
			continue
		}
		verts, _, _ := o3d.geometry_mesh(geos, d.geometry)
		first := len(g.custom_skinned_verts)
		o3d.table_grow_non_zero(&g.custom_skinned_verts, first + len(verts))
		deltas := geos.morph_deltas[:]
		for vtx, i in verts {
			v := vtx
			if morphed {
				v.position, v.normal = o3d.morph_vertex(v, u32(i), deltas, morphs[:nm])
			}
			if skinned {
				v.position, v.normal = o3d.skin_vertex(v, skin[i], joints)
			}
			g.custom_skinned_verts[first + i] = v
		}
		g.custom_skinned_first[di] = i32(first)
	}
	if len(g.custom_skinned_verts) == 0 {
		return nil
	}
	g.custom_skinned.uploaded = 0
	pass := sdl.BeginGPUCopyPass(cmd)
	defer sdl.EndGPUCopyPass(pass)
	return upload_pool(g, &g.custom_skinned, {.VERTEX}, mem.slice_to_bytes(g.custom_skinned_verts[:]), pass)
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
make_material_uniforms :: #force_inline proc "contextless" (m: ^o3d.Shading_Material) -> (mu: Material_Uniforms) {
	mu = {
		color        = {m.base.r, m.base.g, m.base.b, m.opacity},
		emissive     = {m.emissive.r, m.emissive.g, m.emissive.b, 0},
		metalness    = m.metalness,
		roughness    = m.roughness,
		ao_intensity = m.ao_intensity,
		flags        = m.flags,
		alpha_cutoff = m.alpha_cutoff,
		uv_sets      = m.uv_sets,
	}
	for s, i in o3d.Texture_Slot {
		t := m.uv_transforms[s]
		mu.uv_transforms[i * 2] = {t[0].x, t[0].y, t[0].z, 0}
		mu.uv_transforms[i * 2 + 1] = {t[1].x, t[1].y, t[1].z, 0}
	}
	return
}

// Uploads `data` past what `b` holds, into a buffer twice as large when it
// no longer fits (the old one is released once the GPU is done with it).
@(private = "file")
upload_pool :: proc(g: ^GPU, b: ^Pool_Buffer, usage: sdl.GPUBufferUsageFlags, data: []u8, pass: ^sdl.GPUCopyPass) -> o3d.Error {
	size := u32(len(data))
	if size == b.uploaded {
		return nil
	}
	if size > b.capacity {
		if b.buffer != nil {
			sdl.ReleaseGPUBuffer(g.device, b.buffer)
		}
		b.capacity = max(size, b.capacity * 2, 64 << 10)
		b.uploaded = 0
		b.buffer = sdl.CreateGPUBuffer(g.device, {usage = usage, size = b.capacity})
		if b.buffer == nil {
			b^ = {}
			return .Out_Of_Memory
		}
	}
	new := data[b.uploaded:]
	transfer := sdl.CreateGPUTransferBuffer(g.device, {usage = .UPLOAD, size = u32(len(new))})
	if transfer == nil {
		return .Out_Of_Memory
	}
	defer sdl.ReleaseGPUTransferBuffer(g.device, transfer)
	mapped := sdl.MapGPUTransferBuffer(g.device, transfer, false)
	mem.copy(mapped, raw_data(new), len(new))
	sdl.UnmapGPUTransferBuffer(g.device, transfer)
	sdl.UploadToGPUBuffer(pass, {transfer_buffer = transfer}, {buffer = b.buffer, offset = b.uploaded, size = u32(len(new))}, b.uploaded == 0)
	b.uploaded = size
	return nil
}

// The shadow pass: the casters' depth from a shadow light into its map.
@(private)
draw_shadow :: proc(g: ^GPU, scene: ^o3d.Scene, geos: ^o3d.Geometries, shaders: ^o3d.Shaders, idx: int, shadow: ^o3d.Shadow, cmd: ^sdl.GPUCommandBuffer) -> o3d.Error {
	if !ensure_shadow(g, idx, u32(shadow.size)) {
		return .Out_Of_Memory
	}
	o3d.shadow_casters(scene, geos, shaders, shadow, &g.casters, &g.expanded)
	color := sdl.GPUColorTargetInfo{texture = g.shadow_maps[idx], clear_color = {1, 1, 1, 1}, load_op = .CLEAR, store_op = .STORE}
	depth := sdl.GPUDepthStencilTargetInfo{texture = g.shadow_depths[idx], clear_depth = 1, load_op = .CLEAR, store_op = .DONT_CARE, stencil_load_op = .DONT_CARE, stencil_store_op = .DONT_CARE}
	pass := sdl.BeginGPURenderPass(cmd, &color, 1, &depth)
	defer sdl.EndGPURenderPass(pass)
	if g.vertices.buffer == nil && g.dynamic_vertices.buffer == nil {
		return nil
	}
	sdl.BindGPUVertexStorageBuffers(pass, 0, &g.morph_deltas.buffer, 1)
	bound: ^sdl.GPUGraphicsPipeline
	bound_vb: ^sdl.GPUBuffer
	bound_ib: ^sdl.GPUBuffer
	for di in g.casters {
		m := &g.materials[di]
		if .Alpha_Mask in m.flags && .Color_Map not_in m.flags && m.opacity < m.alpha_cutoff {
			continue
		}
		d := &scene.draws[di]
		geo, _ := o3d.geometry_get_any(geos, &g.expanded, d.geometry) // shadow_casters kept only draws with geometry
		dyn := d.geometry >= o3d.DYNAMIC_GEO_BIT
		vbuf := g.dynamic_vertices.buffer if dyn else g.vertices.buffer
		ibuf := g.dynamic_indices.buffer if dyn else g.indices.buffer
		if vbuf == nil || ibuf == nil {
			continue
		}
		if ibuf != bound_ib {
			sdl.BindGPUIndexBuffer(pass, {buffer = ibuf}, ._32BIT)
			bound_ib = ibuf
		}
		joints := o3d.scene_draw_joints(scene, di)
		skinned := !dyn && geo.skinned && len(joints) > 0 && g.skin_vertices.buffer != nil
		p := g.shadow_skinned_pipelines[d.material.side] if skinned else g.shadow_pipelines[d.material.side]
		if p != bound {
			sdl.BindGPUGraphicsPipeline(pass, p)
			bound = p
		}
		color_sampler := tex_binding(g, m.color_map)
		sdl.BindGPUFragmentSamplers(pass, 0, &color_sampler, 1)
		du := make_draw_uniforms(shadow.view_proj * d.model, d.model, geo, o3d.scene_draw_morphs(scene, di))
		sdl.PushGPUVertexUniformData(cmd, 0, &du, size_of(du))
		mu := make_material_uniforms(m)
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
		} else if du.morph_info.x > 0 && geo.first_vertex > 0 {
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
	return nil
}

// Uploads what `textures` added since the last frame (all mip levels per texture).
@(private)
upload_textures :: proc(g: ^GPU, textures: ^o3d.Textures, cmd: ^sdl.GPUCommandBuffer) -> o3d.Error {
	if textures == nil || len(textures.list) <= len(g.textures) {
		return nil
	}
	pass := sdl.BeginGPUCopyPass(cmd)
	defer sdl.EndGPUCopyPass(pass)
	for tex in textures.list[len(g.textures):] {
		gpu_tex := sdl.CreateGPUTexture(g.device, {
			type                 = .D2,
			format               = COLOR_FORMAT,
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
		byte_size := tex.texel_count * 4
		transfer := sdl.CreateGPUTransferBuffer(g.device, {usage = .UPLOAD, size = byte_size})
		if transfer == nil {
			sdl.ReleaseGPUTexture(g.device, gpu_tex)
			return .Out_Of_Memory
		}
		mapped := sdl.MapGPUTransferBuffer(g.device, transfer, false)
		mem.copy(mapped, raw_data(textures.texels[tex.first_texel:]), int(byte_size))
		sdl.UnmapGPUTransferBuffer(g.device, transfer)
		for l in 0 ..< tex.levels {
			mw, mh := max(tex.width >> l, 1), max(tex.height >> l, 1)
			offset := (tex.mips[l] - tex.first_texel) * 4
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
	return {texture = tex, sampler = sampler if sampler != nil else g.linear_sampler}
}

// Draws `scene` into `out` (made by `target`, `width x height`). With `cmd`
// the work is recorded into it (the caller submits); else it is submitted
// and waited for.
render :: proc(g: ^GPU, scene: ^o3d.Scene, geos: ^o3d.Geometries, shaders: ^o3d.Shaders, out: ^sdl.GPUTexture, width, height: u32, cmd: ^sdl.GPUCommandBuffer = nil, textures: ^o3d.Textures = nil) -> o3d.Error {
	if !ensure_targets(g, width, height) {
		return .Out_Of_Memory
	}
	own := cmd == nil
	cmd := cmd
	if own {
		cmd = sdl.AcquireGPUCommandBuffer(g.device)
		if cmd == nil {
			return .Device
		}
	}
	sync_mirrors(g, geos, shaders, textures)
	env := o3d.shading_environment(scene, textures)
	bg_tex, bg_intensity, bg_lod := o3d.shading_background(scene, textures)
	upload_geometry(g, geos, cmd) or_return
	upload_textures(g, textures, cmd) or_return

	view, proj := o3d.camera_matrices(scene.camera, int(width), int(height))
	view_proj := proj * view
	active := o3d.expand_draws(&g.expanded, scene, geos, view, proj, int(width), int(height))
	upload_dynamic(g, cmd) or_return

	lights: [o3d.MAX_LIGHTS]o3d.Shading_Light
	nl := o3d.shading_lights(active.lights[:], &lights)
	o3d.shading_materials(o3d.scene_materials(active), &g.materials, textures)
	o3d.shader_blocks(active, shaders, view, proj, &g.blocks)
	shadows: [o3d.MAX_SHADOWS]o3d.Shadow
	ns := o3d.frame_shadows(active, &shadows)
	for si in 0 ..< ns {
		draw_shadow(g, active, geos, shaders, si, &shadows[si], cmd) or_return
		for &l, li in lights[:nl] {
			if li == shadows[si].light {
				l.shadowed = u32(si + 1)
			}
		}
	}
	exposure := active.tone_mapping_exposure if active.tone_mapping_exposure > 0 else 1
	lu: Lighting_Uniforms
	lu.eye = {active.camera.position.x, active.camera.position.y, active.camera.position.z, 0}
	lu.count = {i32(nl), i32(active.tone_mapping), i32(active.fog.kind), i32(ns)}
	for si in 0 ..< ns {
		lu.shadow_matrix[si] = shadows[si].view_proj
		lu.shadow_params[si] = {f32(shadows[si].size), shadows[si].bias, shadows[si].normal_bias, 0}
	}
	lu.env_params = {env.intensity if env.radiance != 0 else 0, env.max_lod if env.radiance != 0 else 0, exposure, active.fog.density}
	if active.fog.kind != .None {
		fc := o3d.linear(active.fog.color)
		lu.fog_color = {fc.r, fc.g, fc.b, 0}
		lu.fog_params = {active.fog.near, active.fog.far, 0, 0}
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

	o3d.draw_order(active, geos, shaders, view, proj, &g.order, &g.expanded)
	// ShaderMaterials' pipelines made before the pass, on first use.
	for di in g.order {
		m := &g.materials[di]
		if m.shader != 0 {
			custom_pipeline(g, shaders, m.shader, active.draws[di].material.side, .Blended in m.flags)
		}
	}
	upload_custom_skinned(g, active, geos, cmd) or_return

	// Into the targets' top-left width x height (they may be larger), then
	// copied to `out`.
	color := sdl.GPUColorTargetInfo{texture = g.color, clear_color = {0, 0, 0, 0}, load_op = .CLEAR, store_op = .STORE}
	frame := g.color
	if g.samples != ._1 {
		color.store_op, color.resolve_texture = .RESOLVE, g.resolved
		frame = g.resolved
	}
	depth := sdl.GPUDepthStencilTargetInfo{texture = g.depth, clear_depth = 1, load_op = .CLEAR, store_op = .DONT_CARE, stencil_load_op = .DONT_CARE, stencil_store_op = .DONT_CARE}
	pass := sdl.BeginGPURenderPass(cmd, &color, 1, &depth)
	sdl.SetGPUViewport(pass, {x = 0, y = 0, w = f32(width), h = f32(height), min_depth = 0, max_depth = 1})
	sdl.SetGPUScissor(pass, {x = 0, y = 0, w = i32(width), h = i32(height)})
	bound: ^sdl.GPUGraphicsPipeline
	if bg_tex != 0 {
		sdl.BindGPUGraphicsPipeline(pass, g.bg_pipeline)
		bound = g.bg_pipeline
		bg_binding := tex_binding(g, bg_tex, g.env_sampler)
		sdl.BindGPUFragmentSamplers(pass, 0, &bg_binding, 1)
		inv_vp := linalg.inverse(view_proj)
		sdl.PushGPUVertexUniformData(cmd, 0, &inv_vp, size_of(inv_vp))
		bg_params := [4]f32{bg_intensity, bg_lod, f32(active.tone_mapping), exposure}
		sdl.PushGPUFragmentUniformData(cmd, 0, &bg_params, size_of(bg_params))
		sdl.DrawGPUPrimitives(pass, 3, 1, 0, 0)
	}
	sdl.PushGPUFragmentUniformData(cmd, 1, &lu, size_of(lu))
	sdl.BindGPUVertexStorageBuffers(pass, 0, &g.morph_deltas.buffer, 1)
	bound_vb: ^sdl.GPUBuffer
	bound_ib: ^sdl.GPUBuffer
	if g.vertices.buffer != nil && g.indices.buffer != nil {
		sdl.BindGPUVertexBuffers(pass, 0, &sdl.GPUBufferBinding{buffer = g.vertices.buffer}, 1)
		sdl.BindGPUIndexBuffer(pass, {buffer = g.indices.buffer}, ._32BIT)
		bound_vb = g.vertices.buffer
		bound_ib = g.indices.buffer
	}
	for di in g.order {
		d := &active.draws[di]
		geo, _ := o3d.geometry_get_any(geos, &g.expanded, d.geometry) // draw_order kept only draws with geometry
		dyn := d.geometry >= o3d.DYNAMIC_GEO_BIT
		vbuf := g.dynamic_vertices.buffer if dyn else g.vertices.buffer
		ibuf := g.dynamic_indices.buffer if dyn else g.indices.buffer
		if vbuf == nil || ibuf == nil {
			continue
		}
		if ibuf != bound_ib {
			sdl.BindGPUIndexBuffer(pass, {buffer = ibuf}, ._32BIT)
			bound_ib = ibuf
		}
		m := &g.materials[di]
		blend := .Blended in m.flags
		if m.shader != 0 {
			p := custom_pipeline(g, shaders, m.shader, d.material.side, blend)
			if p == nil {
				continue
			}
			if p != bound {
				sdl.BindGPUGraphicsPipeline(pass, p)
				bound = p
			}
			for stage in o3d.Shader_Stage {
				at := g.blocks.at[di][stage]
				if at < 0 {
					continue
				}
				size := u32(o3d.shader_module(o3d.shader_get(shaders, m.shader), stage).block_size)
				if stage == .Vertex {
					sdl.PushGPUVertexUniformData(cmd, 0, &g.blocks.bytes[at], size)
				} else {
					sdl.PushGPUFragmentUniformData(cmd, 0, &g.blocks.bytes[at], size)
				}
			}
			if custom_first := g.custom_skinned_first[di]; custom_first >= 0 && g.custom_skinned.buffer != nil {
				if bound_vb != g.custom_skinned.buffer {
					sdl.BindGPUVertexBuffers(pass, 0, &sdl.GPUBufferBinding{buffer = g.custom_skinned.buffer}, 1)
					bound_vb = g.custom_skinned.buffer
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
		joints := o3d.scene_draw_joints(active, di)
		skinned := !dyn && geo.skinned && len(joints) > 0 && g.skin_vertices.buffer != nil
		p := g.skinned_pipelines[d.material.side][1 if blend else 0] if skinned else g.pipelines[d.material.side][1 if blend else 0]
		if p != bound {
			sdl.BindGPUGraphicsPipeline(pass, p)
			bound = p
		}
		samplers := [12]sdl.GPUTextureSamplerBinding {
			tex_binding(g, m.color_map),
			{texture = g.shadow_maps[0], sampler = g.shadow_sampler},
			tex_binding(g, m.normal_map),
			tex_binding(g, m.roughness_map),
			tex_binding(g, m.emissive_map),
			tex_binding(g, env.radiance, g.env_sampler),
			tex_binding(g, env.irradiance, g.env_sampler),
			tex_binding(g, m.metalness_map),
			tex_binding(g, m.ao_map),
			{texture = g.shadow_maps[1], sampler = g.shadow_sampler},
			{texture = g.shadow_maps[2], sampler = g.shadow_sampler},
			{texture = g.shadow_maps[3], sampler = g.shadow_sampler},
		}
		sdl.BindGPUFragmentSamplers(pass, 0, &samplers[0], len(samplers))
		du := make_draw_uniforms(view_proj * d.model, d.model, geo, o3d.scene_draw_morphs(active, di))
		sdl.PushGPUVertexUniformData(cmd, 0, &du, size_of(du))
		mu := make_material_uniforms(m)
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
		} else if du.morph_info.x > 0 && geo.first_vertex > 0 {
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
	sdl.EndGPURenderPass(pass)
	copy_pass := sdl.BeginGPUCopyPass(cmd)
	sdl.CopyGPUTextureToTexture(copy_pass, {texture = frame}, {texture = out}, width, height, 1, false)
	sdl.EndGPUCopyPass(copy_pass)
	if own {
		fence := sdl.SubmitGPUCommandBufferAndAcquireFence(cmd)
		if fence == nil {
			return .Device
		}
		_ = sdl.WaitForGPUFences(g.device, true, &fence, 1)
		sdl.ReleaseGPUFence(g.device, fence)
	}
	return nil
}

// The texture's pixels (RGBA8 premultiplied), for tests and saving.
read :: proc(g: ^GPU, tex: ^sdl.GPUTexture, width, height: u32, allocator := context.allocator) -> (pixels: []u8, err: o3d.Error) {
	size := width * height * 4
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
