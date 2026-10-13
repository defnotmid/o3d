// SDL_GPU renderer for o3d: Vulkan on Linux and Windows, Metal on macOS.
// It draws the same scene as the software renderer, with the same shading
// (shaders/o3d.slang), 4x MSAA resolved into the caller's texture.
//
// `init` creates a headless device, or borrows the host's (the one its
// window or 2D canvas uses, so the result composites without a copy).
// Geometry lives in one vertex and one index buffer, mirroring
// `Geometries`' shared arrays: what is new uploads at the start of a frame.
package o3d_sdlgpu

import "core:mem"
import sdl "vendor:sdl3"

import o3d ".."

VERT_SPV :: #load("../shaders/o3d.vert.spv")
FRAG_SPV :: #load("../shaders/o3d.frag.spv")
VERT_MSL :: #load("../shaders/o3d.vert.msl")
FRAG_MSL :: #load("../shaders/o3d.frag.msl")

COLOR_FORMAT :: sdl.GPUTextureFormat.R8G8B8A8_UNORM

// A GPU copy of one of `Geometries`' shared arrays: the first `uploaded`
// bytes are on the GPU; growing past `capacity` makes a new buffer.
Pool_Buffer :: struct {
	buffer:             ^sdl.GPUBuffer,
	capacity, uploaded: u32,
}

GPU :: struct {
	device:       ^sdl.GPUDevice,
	owns_device:  bool,
	vert, frag:   ^sdl.GPUShader,
	samples:      sdl.GPUSampleCount,
	depth_format: sdl.GPUTextureFormat,
	// By the material's side, then blended or not.
	pipelines:    [o3d.Side][2]^sdl.GPUGraphicsPipeline,
	// The multisampled colour and depth targets, at `size`.
	color, depth: ^sdl.GPUTexture,
	size:         [2]u32,
	vertices:     Pool_Buffer,
	indices:      Pool_Buffer,
	// Scratch per frame.
	order:        [dynamic]int,
	materials:    [dynamic]o3d.Shading_Material, // per draw
}

// The uniforms as the shader reads them (std140 / Metal layouts agree here).
Draw_Uniforms :: struct {
	mvp, model, normal: o3d.Mat4,
}

Material_Uniforms :: struct {
	color, emissive:     [4]f32,
	metalness, roughness: f32,
	flags:               o3d.Shading_Flags,
	_:                   u32,
}
#assert(size_of(Material_Uniforms) == 48)

Light_Uniforms :: struct {
	color, ground, vector: [4]f32,
	kind:                  u32, // o3d.Light_Kind
	distance, decay:       f32,
	_:                     f32,
}
#assert(size_of(Light_Uniforms) == 64)

Lighting_Uniforms :: struct {
	eye:    [4]f32,
	count:  [4]i32,
	lights: [o3d.MAX_LIGHTS]Light_Uniforms,
}

init :: proc(g: ^GPU, device: ^sdl.GPUDevice = nil) -> o3d.Error {
	g^ = {device = device}
	if g.device == nil {
		if .VIDEO not_in sdl.WasInit({.VIDEO}) && !sdl.Init({.VIDEO}) {
			sdl.SetHint(sdl.HINT_VIDEO_DRIVER, "offscreen")
			if !sdl.Init({.VIDEO}) {
				return .Device
			}
		}
		g.device = sdl.CreateGPUDevice({.SPIRV, .MSL}, false, nil)
		if g.device == nil {
			return .Device
		}
		g.owns_device = true
	}
	// slangc names the SPIR-V entry points "main" and keeps the names for Metal.
	formats := sdl.GetGPUShaderFormats(g.device)
	vert_code, frag_code: []u8
	format: sdl.GPUShaderFormat
	vs_entry, fs_entry: cstring
	switch {
	case .SPIRV in formats:
		vert_code, frag_code, format, vs_entry, fs_entry = VERT_SPV, FRAG_SPV, {.SPIRV}, "main", "main"
	case .MSL in formats:
		vert_code, frag_code, format, vs_entry, fs_entry = VERT_MSL, FRAG_MSL, {.MSL}, "vs_main", "fs_main"
	case:
		return .Device
	}
	g.vert = sdl.CreateGPUShader(g.device, {code_size = len(vert_code), code = raw_data(vert_code), entrypoint = vs_entry, format = format, stage = .VERTEX, num_uniform_buffers = 1})
	g.frag = sdl.CreateGPUShader(g.device, {code_size = len(frag_code), code = raw_data(frag_code), entrypoint = fs_entry, format = format, stage = .FRAGMENT, num_uniform_buffers = 2})
	if g.vert == nil || g.frag == nil {
		return .Shader
	}
	g.depth_format = .D32_FLOAT
	if !sdl.GPUTextureSupportsFormat(g.device, g.depth_format, .D2, {.DEPTH_STENCIL_TARGET}) {
		g.depth_format = .D24_UNORM
	}
	g.samples = ._4
	if !sdl.GPUTextureSupportsSampleCount(g.device, COLOR_FORMAT, ._4) || !sdl.GPUTextureSupportsSampleCount(g.device, g.depth_format, ._4) {
		g.samples = ._1
	}
	for side in o3d.Side {
		for blend in 0 ..< 2 {
			g.pipelines[side][blend] = make_pipeline(g, side, blend == 1)
			if g.pipelines[side][blend] == nil {
				return .Shader
			}
		}
	}
	return nil
}

destroy :: proc(g: ^GPU) {
	if g.device == nil {
		return
	}
	for b in ([]^Pool_Buffer{&g.vertices, &g.indices}) {
		if b.buffer != nil {
			sdl.ReleaseGPUBuffer(g.device, b.buffer)
		}
	}
	delete(g.order)
	delete(g.materials)
	for side in o3d.Side {
		for p in g.pipelines[side] {
			if p != nil {
				sdl.ReleaseGPUGraphicsPipeline(g.device, p)
			}
		}
	}
	release_targets(g)
	if g.vert != nil {
		sdl.ReleaseGPUShader(g.device, g.vert)
	}
	if g.frag != nil {
		sdl.ReleaseGPUShader(g.device, g.frag)
	}
	if g.owns_device {
		sdl.DestroyGPUDevice(g.device)
	}
	g^ = {}
}

// A texture to render into (and sample or read back).
target :: proc(g: ^GPU, width, height: u32) -> ^sdl.GPUTexture {
	return sdl.CreateGPUTexture(g.device, {type = .D2, format = COLOR_FORMAT, usage = {.SAMPLER, .COLOR_TARGET}, width = width, height = height, layer_count_or_depth = 1, num_levels = 1, sample_count = ._1})
}

@(private = "file")
make_pipeline :: proc(g: ^GPU, side: o3d.Side, blend: bool) -> ^sdl.GPUGraphicsPipeline {
	buffer := sdl.GPUVertexBufferDescription{slot = 0, pitch = size_of(o3d.Vertex), input_rate = .VERTEX}
	attrs := [3]sdl.GPUVertexAttribute {
		{location = 0, buffer_slot = 0, format = .FLOAT3, offset = 0},
		{location = 1, buffer_slot = 0, format = .FLOAT3, offset = 12},
		{location = 2, buffer_slot = 0, format = .FLOAT2, offset = 24},
	}
	color := sdl.GPUColorTargetDescription {
		format = COLOR_FORMAT,
		blend_state = {
			enable_blend = blend,
			src_color_blendfactor = .ONE,
			dst_color_blendfactor = .ONE_MINUS_SRC_ALPHA,
			color_blend_op = .ADD,
			src_alpha_blendfactor = .ONE,
			dst_alpha_blendfactor = .ONE_MINUS_SRC_ALPHA,
			alpha_blend_op = .ADD,
		},
	}
	cull: sdl.GPUCullMode
	switch side {
	case .Front:
		cull = .BACK
	case .Back:
		cull = .FRONT
	case .Double:
		cull = .NONE
	}
	info := sdl.GPUGraphicsPipelineCreateInfo {
		vertex_shader = g.vert,
		fragment_shader = g.frag,
		vertex_input_state = {vertex_buffer_descriptions = &buffer, num_vertex_buffers = 1, vertex_attributes = &attrs[0], num_vertex_attributes = len(attrs)},
		primitive_type = .TRIANGLELIST,
		rasterizer_state = {fill_mode = .FILL, cull_mode = cull, front_face = .COUNTER_CLOCKWISE},
		multisample_state = {sample_count = g.samples},
		depth_stencil_state = {compare_op = .LESS, enable_depth_test = true, enable_depth_write = !blend},
		target_info = {color_target_descriptions = &color, num_color_targets = 1, depth_stencil_format = g.depth_format, has_depth_stencil_target = true},
	}
	return sdl.CreateGPUGraphicsPipeline(g.device, info)
}

@(private = "file")
release_targets :: proc(g: ^GPU) {
	if g.color != nil {
		sdl.ReleaseGPUTexture(g.device, g.color)
	}
	if g.depth != nil {
		sdl.ReleaseGPUTexture(g.device, g.depth)
	}
	g.color, g.depth, g.size = nil, nil, {}
}

@(private = "file")
ensure_targets :: proc(g: ^GPU, w, h: u32) -> bool {
	if g.size == {w, h} && g.depth != nil {
		return true
	}
	release_targets(g)
	if g.samples != ._1 {
		g.color = sdl.CreateGPUTexture(g.device, {type = .D2, format = COLOR_FORMAT, usage = {.COLOR_TARGET}, width = w, height = h, layer_count_or_depth = 1, num_levels = 1, sample_count = g.samples})
		if g.color == nil {
			return false
		}
	}
	g.depth = sdl.CreateGPUTexture(g.device, {type = .D2, format = g.depth_format, usage = {.DEPTH_STENCIL_TARGET}, width = w, height = h, layer_count_or_depth = 1, num_levels = 1, sample_count = g.samples})
	g.size = {w, h}
	return g.depth != nil
}

// Uploads what `geos` added since the last frame (in a copy pass of `cmd`).
@(private = "file")
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

// Draws `scene` into `out` (made by `target`, `width x height`). With `cmd`
// the work is recorded into it (the caller submits); else it is submitted
// and waited for.
render :: proc(g: ^GPU, scene: ^o3d.Scene, geos: ^o3d.Geometries, out: ^sdl.GPUTexture, width, height: u32, cmd: ^sdl.GPUCommandBuffer = nil) -> o3d.Error {
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
	upload_geometry(g, geos, cmd) or_return

	view, proj := o3d.camera_matrices(scene.camera, int(width), int(height))
	view_proj := proj * view
	lights: [o3d.MAX_LIGHTS]o3d.Shading_Light
	nl := o3d.shading_lights(scene.lights[:], &lights)
	o3d.shading_materials(o3d.scene_materials(scene), &g.materials)
	lu: Lighting_Uniforms
	lu.eye = {scene.camera.position.x, scene.camera.position.y, scene.camera.position.z, 0}
	lu.count = {i32(nl), 0, 0, 0}
	for l, i in lights[:nl] {
		lu.lights[i] = {
			color    = {l.color.r, l.color.g, l.color.b, 0},
			ground   = {l.ground.r, l.ground.g, l.ground.b, 0},
			vector   = {l.vector.x, l.vector.y, l.vector.z, 0},
			kind     = u32(l.kind),
			distance = l.distance,
			decay    = l.decay,
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
	sdl.PushGPUFragmentUniformData(cmd, 1, &lu, size_of(lu))
	o3d.draw_order(scene, view, &g.order)
	if g.vertices.buffer != nil {
		sdl.BindGPUVertexBuffers(pass, 0, &sdl.GPUBufferBinding{buffer = g.vertices.buffer}, 1)
		sdl.BindGPUIndexBuffer(pass, {buffer = g.indices.buffer}, ._32BIT)
	}
	bound: ^sdl.GPUGraphicsPipeline
	for di in g.order {
		d := &scene.draws[di]
		geo, ok := o3d.geometry_get(geos, d.geometry)
		if !ok || geo.index_count == 0 {
			continue
		}
		m := &g.materials[di]
		p := g.pipelines[d.material.side][1 if .Blended in m.flags else 0]
		if p != bound {
			sdl.BindGPUGraphicsPipeline(pass, p)
			bound = p
		}
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
