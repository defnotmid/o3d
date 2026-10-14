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
		g.vertices.uploaded, g.indices.uploaded = 0, 0
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
	indices := mem.slice_to_bytes(geos.indices[:])
	if u32(len(vertices)) == g.vertices.uploaded && u32(len(indices)) == g.indices.uploaded {
		return nil
	}
	pass := sdl.BeginGPUCopyPass(cmd)
	defer sdl.EndGPUCopyPass(pass)
	upload_pool(g, &g.vertices, {.VERTEX}, vertices, pass) or_return
	upload_pool(g, &g.indices, {.INDEX}, indices, pass) or_return
	return nil
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
	sdl.UploadToGPUBuffer(pass, {transfer_buffer = transfer}, {buffer = b.buffer, offset = b.uploaded, size = u32(len(new))}, false)
	b.uploaded = size
	return nil
}

// The shadow pass: the casters' depth from the shadow light into the map.
@(private)
draw_shadow :: proc(g: ^GPU, scene: ^o3d.Scene, geos: ^o3d.Geometries, shaders: ^o3d.Shaders, shadow: ^o3d.Shadow, cmd: ^sdl.GPUCommandBuffer) -> o3d.Error {
	if !ensure_shadow(g, u32(shadow.size)) {
		return .Out_Of_Memory
	}
	o3d.shadow_casters(scene, geos, shaders, shadow, &g.casters)
	color := sdl.GPUColorTargetInfo{texture = g.shadow_map, clear_color = {1, 1, 1, 1}, load_op = .CLEAR, store_op = .STORE}
	depth := sdl.GPUDepthStencilTargetInfo{texture = g.shadow_depth, clear_depth = 1, load_op = .CLEAR, store_op = .DONT_CARE, stencil_load_op = .DONT_CARE, stencil_store_op = .DONT_CARE}
	pass := sdl.BeginGPURenderPass(cmd, &color, 1, &depth)
	defer sdl.EndGPURenderPass(pass)
	if g.vertices.buffer == nil {
		return nil
	}
	sdl.BindGPUVertexBuffers(pass, 0, &sdl.GPUBufferBinding{buffer = g.vertices.buffer}, 1)
	sdl.BindGPUIndexBuffer(pass, {buffer = g.indices.buffer}, ._32BIT)
	bound: ^sdl.GPUGraphicsPipeline
	for di in g.casters {
		d := &scene.draws[di]
		geo, _ := o3d.geometry_get(geos, d.geometry) // shadow_casters kept only draws with geometry
		p := g.shadow_pipelines[d.material.side]
		if p != bound {
			sdl.BindGPUGraphicsPipeline(pass, p)
			bound = p
		}
		du := Draw_Uniforms{shadow.view_proj * d.model, d.model, o3d.normal_matrix(d.model)}
		sdl.PushGPUVertexUniformData(cmd, 0, &du, size_of(du))
		sdl.DrawGPUIndexedPrimitives(pass, geo.index_count, 1, geo.first_index, i32(geo.first_vertex), 0)
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
	lights: [o3d.MAX_LIGHTS]o3d.Shading_Light
	nl := o3d.shading_lights(scene.lights[:], &lights)
	o3d.shading_materials(o3d.scene_materials(scene), &g.materials, textures)
	o3d.shader_blocks(scene, shaders, view, proj, &g.blocks)
	shadow := o3d.frame_shadow(scene)
	if shadow.light >= 0 {
		draw_shadow(g, scene, geos, shaders, &shadow, cmd) or_return
	}
	lu: Lighting_Uniforms
	lu.eye = {scene.camera.position.x, scene.camera.position.y, scene.camera.position.z, 0}
	lu.count = {i32(nl), 0, 0, 0}
	if shadow.light >= 0 {
		lu.shadow_matrix = shadow.view_proj
		lu.shadow_params = {f32(shadow.size), shadow.bias, shadow.normal_bias, 0}
	}
	if env.radiance != 0 {
		lu.env_params = {env.intensity, env.max_lod, 0, 0}
	}
	for l, i in lights[:nl] {
		lu.lights[i] = {
			color    = {l.color.r, l.color.g, l.color.b, 0},
			ground   = {l.ground.r, l.ground.g, l.ground.b, 0},
			vector   = {l.vector.x, l.vector.y, l.vector.z, 0},
			kind     = u32(l.kind),
			distance = l.distance,
			decay    = l.decay,
			shadowed = u32(l.shadowed),
		}
	}

	o3d.draw_order(scene, geos, shaders, view, proj, &g.order)
	// ShaderMaterials' pipelines made before the pass, on first use.
	for di in g.order {
		m := &g.materials[di]
		if m.shader != 0 {
			custom_pipeline(g, shaders, m.shader, scene.draws[di].material.side, .Blended in m.flags)
		}
	}

	color := sdl.GPUColorTargetInfo{clear_color = {0, 0, 0, 0}, load_op = .CLEAR}
	if g.samples == ._1 {
		color.texture, color.store_op = out, .STORE
	} else {
		color.texture, color.store_op, color.resolve_texture = g.color, .RESOLVE, out
	}
	depth := sdl.GPUDepthStencilTargetInfo{texture = g.depth, clear_depth = 1, load_op = .CLEAR, store_op = .DONT_CARE, stencil_load_op = .DONT_CARE, stencil_store_op = .DONT_CARE}
	pass := sdl.BeginGPURenderPass(cmd, &color, 1, &depth)
	bound: ^sdl.GPUGraphicsPipeline
	if bg_tex != 0 {
		sdl.BindGPUGraphicsPipeline(pass, g.bg_pipeline)
		bound = g.bg_pipeline
		bg_binding := tex_binding(g, bg_tex, g.env_sampler)
		sdl.BindGPUFragmentSamplers(pass, 0, &bg_binding, 1)
		inv_vp := linalg.inverse(view_proj)
		sdl.PushGPUVertexUniformData(cmd, 0, &inv_vp, size_of(inv_vp))
		bg_params := [4]f32{bg_intensity, bg_intensity, bg_intensity, bg_lod}
		sdl.PushGPUFragmentUniformData(cmd, 0, &bg_params, size_of(bg_params))
		sdl.DrawGPUPrimitives(pass, 3, 1, 0, 0)
	}
	sdl.PushGPUFragmentUniformData(cmd, 1, &lu, size_of(lu))
	if g.vertices.buffer != nil && g.indices.buffer != nil {
		sdl.BindGPUVertexBuffers(pass, 0, &sdl.GPUBufferBinding{buffer = g.vertices.buffer}, 1)
		sdl.BindGPUIndexBuffer(pass, {buffer = g.indices.buffer}, ._32BIT)
	}
	for di in g.order {
		d := &scene.draws[di]
		geo, _ := o3d.geometry_get(geos, d.geometry) // draw_order kept only draws with geometry
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
			sdl.DrawGPUIndexedPrimitives(pass, geo.index_count, 1, geo.first_index, i32(geo.first_vertex), 0)
			continue
		}
		p := g.pipelines[d.material.side][1 if blend else 0]
		if p != bound {
			sdl.BindGPUGraphicsPipeline(pass, p)
			bound = p
		}
		samplers := [8]sdl.GPUTextureSamplerBinding {
			{texture = g.shadow_map, sampler = g.shadow_sampler},
			tex_binding(g, m.color_map),
			tex_binding(g, m.normal_map),
			tex_binding(g, m.roughness_map),
			tex_binding(g, m.emissive_map),
			tex_binding(g, env.radiance, g.env_sampler),
			tex_binding(g, env.irradiance, g.env_sampler),
			tex_binding(g, m.metalness_map),
		}
		sdl.BindGPUFragmentSamplers(pass, 0, &samplers[0], len(samplers))
		du := Draw_Uniforms{view_proj * d.model, d.model, o3d.normal_matrix(d.model)}
		sdl.PushGPUVertexUniformData(cmd, 0, &du, size_of(du))
		mu := Material_Uniforms {
			color     = {m.base.r, m.base.g, m.base.b, m.opacity},
			emissive  = {m.emissive.r, m.emissive.g, m.emissive.b, 0},
			metalness = m.metalness,
			roughness = m.roughness,
			flags     = m.flags,
		}
		sdl.PushGPUFragmentUniformData(cmd, 0, &mu, size_of(mu))
		sdl.DrawGPUIndexedPrimitives(pass, geo.index_count, 1, geo.first_index, i32(geo.first_vertex), 0)
	}
	sdl.EndGPURenderPass(pass)
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
