// SDL_GPU renderer for o3d: Vulkan on Linux and Windows, Metal on macOS.
// It draws the same scene as the software renderer, with the same shading
// (shaders/o3d.slang, shaders/background.slang), 4x MSAA resolved into the
// caller's texture.
//
// `init` creates a headless device, or borrows the host's (the one its
// window or 2D canvas uses, so the result composites without a copy).
// Geometry lives in one vertex and one index buffer, mirroring
// `Geometries`' shared arrays: what is new uploads at the start of a frame.
package o3d_sdlgpu

import "core:sync"
import sdl "vendor:sdl3"

import o3d ".."
import "deps:oksl"

VERT_SPV :: #load("../shaders/o3d.vert.spv")
SKINNED_VERT_SPV :: #load("../shaders/o3d.skinned_vert.spv")
FRAG_SPV :: #load("../shaders/o3d.frag.spv")
VERT_MSL :: #load("../shaders/o3d.vert.msl")
SKINNED_VERT_MSL :: #load("../shaders/o3d.skinned_vert.msl")
FRAG_MSL :: #load("../shaders/o3d.frag.msl")
SHADOW_SPV :: #load("../shaders/o3d.shadow.spv")
SHADOW_MSL :: #load("../shaders/o3d.shadow.msl")
CURVES_VERT_SPV :: #load("../shaders/o3d.curves_vert.spv")
CURVES_VERT_MSL :: #load("../shaders/o3d.curves_vert.msl")
CURVES_FRAG_SPV :: #load("../shaders/o3d.curves_frag.spv")
CURVES_FRAG_MSL :: #load("../shaders/o3d.curves_frag.msl")
SHADOW_CURVES_SPV :: #load("../shaders/o3d.shadow_curves.spv")
SHADOW_CURVES_MSL :: #load("../shaders/o3d.shadow_curves.msl")
BG_VERT_SPV :: #load("../shaders/background.vert.spv")
BG_FRAG_SPV :: #load("../shaders/background.frag.spv")
BG_VERT_MSL :: #load("../shaders/background.vert.msl")
BG_FRAG_MSL :: #load("../shaders/background.frag.msl")

COLOR_FORMAT :: sdl.GPUTextureFormat.R8G8B8A8_UNORM
// The shadow map holds depth as a colour, so every backend can sample it.
SHADOW_FORMAT :: sdl.GPUTextureFormat.R32_FLOAT

// A GPU copy of one of `Geometries`' shared arrays: the first `uploaded`
// bytes are on the GPU; growing past `capacity` makes a new buffer.
Pool_Buffer :: struct {
	buffer:             ^sdl.GPUBuffer,
	capacity, uploaded: u32,
}

GPU :: struct {
	device:                   ^sdl.GPUDevice,
	owns_device:              bool,
	vert, skinned_vert, frag: ^sdl.GPUShader,
	bg_vert, bg_frag:         ^sdl.GPUShader,
	samples:                  sdl.GPUSampleCount,
	depth_format:             sdl.GPUTextureFormat,
	// By the material's side, then blended or not.
	pipelines:                [o3d.Side][2]^sdl.GPUGraphicsPipeline,
	skinned_pipelines:        [o3d.Side][2]^sdl.GPUGraphicsPipeline,
	// Exact shapes and glyph runs (`.Curves`): their vertex shader (each
	// vertex's quad), fragment shader (coverage from the quad's curve set
	// in `curve_words`: by area blended, by sample opaque) and pipelines,
	// as `pipelines`.
	curves_vert:              ^sdl.GPUShader,
	curves_frag:              ^sdl.GPUShader,
	curve_pipelines:          [o3d.Side][2]^sdl.GPUGraphicsPipeline,
	bg_pipeline:              ^sdl.GPUGraphicsPipeline,
	// The multisampled colour and depth targets, at `size`.
	// The frame's targets, at `capacity`: a frame draws into their top-left
	// width x height and is copied out, so frames of other sizes (several
	// views, a view whose size animates) reuse them; they only grow. `color`
	// is multisampled when `samples` is, resolved into `resolved`.
	color, depth:             ^sdl.GPUTexture,
	resolved:                 ^sdl.GPUTexture,
	capacity:                 [2]u32,
	vertices:                 Pool_Buffer,
	skin_vertices:            Pool_Buffer,
	morph_deltas:             Pool_Buffer,
	indices:                  Pool_Buffer,
	curve_words:              Pool_Buffer,
	// The generations of the tables mirrored (geometry, textures, shaders'
	// pipelines): a different one (new contents, ids reused) starts over.
	mirrored:                 Mirrored,
	// Per texture (its id - 1): `Textures.list` mirrored on the GPU, nil
	// for an external texture, whose GPU texture (the host's, not released
	// here) is in `externals` (by id - 1, set by `texture_set_external`, for
	// the table of that generation only).
	textures:                 [dynamic]^sdl.GPUTexture,
	externals:                [dynamic]External_Texture,
	dummy_texture:            ^sdl.GPUTexture,
	linear_sampler:           ^sdl.GPUSampler,
	env_sampler:              ^sdl.GPUSampler,
	// Per shader (its id - 1): ShaderMaterials' GPU shaders and pipelines,
	// made the first time a frame draws with them.
	custom:                   [dynamic]Custom,
	// The shadow pass (shadow.odin): its fragment shader (depth out) and its
	// pipelines by the casters' material side (the maps are a job's).
	shadow_frag:              ^sdl.GPUShader,
	shadow_pipelines:         [o3d.Side]^sdl.GPUGraphicsPipeline,
	shadow_skinned_pipelines: [o3d.Side]^sdl.GPUGraphicsPipeline,
	shadow_curves_frag:       ^sdl.GPUShader,
	shadow_curve_pipelines:   [o3d.Side]^sdl.GPUGraphicsPipeline,
	shadow_sampler:           ^sdl.GPUSampler,
	dummy_shadow:             ^sdl.GPUTexture, // bound for a job's unused shadow slots
	// `render`'s own job.
	job:                      Job,
}

Mirrored :: struct {
	geometry, textures, shaders: o3d.Generation,
}

External_Texture :: struct {
	texture:    ^sdl.GPUTexture,
	generation: o3d.Generation, // of the `Textures` it was bound for
}

// A ShaderMaterial's shaders and pipelines (as `GPU.pipelines`), or why
// they could not be made.
Custom :: struct {
	vert, frag: ^sdl.GPUShader,
	pipelines:  [o3d.Side][2]^sdl.GPUGraphicsPipeline,
	made:       bool,
	failed:     bool,
}

// The uniforms as the shader reads them (std140 / Metal layouts agree here).
Draw_Uniforms :: struct {
	mvp, model, normal: o3d.Mat4,
	morph_bases:        [2][4]u32,
	morph_weights:      [2][4]f32,
	morph_info:         [4]u32,
}
#assert(size_of(Draw_Uniforms) == 272)

Material_Uniforms :: struct {
	color, emissive:                    [4]f32,
	metalness, roughness, ao_intensity: f32,
	flags:                              o3d.Shading_Flags,
	alpha_cutoff:                       f32,
	uv_sets:                            u32,
	// An exact shape's or glyph run's table of its quads' curve sets in
	// `curve_words` (o3d.Geometry.curve_table).
	curve_table, _:                     u32,
	uv_transforms:                      [2 * len(o3d.Texture_Slot)][4]f32,
}
#assert(size_of(Material_Uniforms) == 288)

Light_Uniforms :: struct {
	color, ground, vector: [4]f32,
	kind:                  u32, // o3d.Light_Kind
	distance, decay:       f32,
	shadowed:              u32,
}
#assert(size_of(Light_Uniforms) == 64)

Lighting_Uniforms :: struct {
	eye:           [4]f32,
	count:         [4]i32,
	shadow_matrix: [o3d.MAX_SHADOWS]o3d.Mat4,
	// Map size (0: no shadow), bias, normal bias.
	shadow_params: [o3d.MAX_SHADOWS][4]f32,
	// Environment intensity (0: no environment), max radiance mip level, tone exposure, fog density.
	env_params:    [4]f32,
	fog_color:     [4]f32,
	fog_params:    [4]f32,
	lights:        [o3d.MAX_LIGHTS]Light_Uniforms,
}

// SDL_Init and making or destroying a device are not safe from several
// threads at once (tests run in parallel; a host may init from workers):
// they take turns.
@(private = "file")
device_mutex: sync.Mutex

init :: proc(g: ^GPU, device: ^sdl.GPUDevice = nil) -> o3d.Error {
	g^ = {device = device}
	if g.device == nil {
		sync.guard(&device_mutex)
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
	vert_code, skinned_vert_code, frag_code, shadow_code, bg_vert_code, bg_frag_code, curves_vert_code, curves_code, shadow_curves_code: []u8
	format: sdl.GPUShaderFormat
	vs_entry, skinned_vs_entry, fs_entry, shadow_entry, curves_vs_entry, curves_entry, shadow_curves_entry: cstring
	switch {
	case .SPIRV in formats:
		vert_code, skinned_vert_code, frag_code, shadow_code = VERT_SPV, SKINNED_VERT_SPV, FRAG_SPV, SHADOW_SPV
		bg_vert_code, bg_frag_code = BG_VERT_SPV, BG_FRAG_SPV
		curves_vert_code, curves_code, shadow_curves_code = CURVES_VERT_SPV, CURVES_FRAG_SPV, SHADOW_CURVES_SPV
		format = {.SPIRV}
		vs_entry, skinned_vs_entry, fs_entry, shadow_entry = "main", "main", "main", "main"
		curves_vs_entry, curves_entry, shadow_curves_entry = "main", "main", "main"
	case .MSL in formats:
		vert_code, skinned_vert_code, frag_code, shadow_code = VERT_MSL, SKINNED_VERT_MSL, FRAG_MSL, SHADOW_MSL
		bg_vert_code, bg_frag_code = BG_VERT_MSL, BG_FRAG_MSL
		curves_vert_code, curves_code, shadow_curves_code = CURVES_VERT_MSL, CURVES_FRAG_MSL, SHADOW_CURVES_MSL
		format = {.MSL}
		vs_entry, skinned_vs_entry, fs_entry, shadow_entry = "vs_main", "vs_skinned", "fs_main", "fs_shadow"
		curves_vs_entry, curves_entry, shadow_curves_entry = "vs_curves", "fs_curves", "fs_shadow_curves"
	case:
		return .Device
	}
	g.vert = sdl.CreateGPUShader(g.device, {code_size = len(vert_code), code = raw_data(vert_code), entrypoint = vs_entry, format = format, stage = .VERTEX, num_storage_buffers = 1, num_uniform_buffers = 1})
	g.skinned_vert = sdl.CreateGPUShader(g.device, {code_size = len(skinned_vert_code), code = raw_data(skinned_vert_code), entrypoint = skinned_vs_entry, format = format, stage = .VERTEX, num_storage_buffers = 1, num_uniform_buffers = 2})
	g.frag = sdl.CreateGPUShader(g.device, {code_size = len(frag_code), code = raw_data(frag_code), entrypoint = fs_entry, format = format, stage = .FRAGMENT, num_samplers = 13, num_uniform_buffers = 2})
	g.shadow_frag = sdl.CreateGPUShader(g.device, {code_size = len(shadow_code), code = raw_data(shadow_code), entrypoint = shadow_entry, format = format, stage = .FRAGMENT, num_samplers = 2, num_uniform_buffers = 1})
	g.curves_vert = sdl.CreateGPUShader(g.device, {code_size = len(curves_vert_code), code = raw_data(curves_vert_code), entrypoint = curves_vs_entry, format = format, stage = .VERTEX, num_uniform_buffers = 1})
	g.curves_frag = sdl.CreateGPUShader(g.device, {code_size = len(curves_code), code = raw_data(curves_code), entrypoint = curves_entry, format = format, stage = .FRAGMENT, num_samplers = 13, num_storage_buffers = 1, num_uniform_buffers = 2})
	g.shadow_curves_frag = sdl.CreateGPUShader(g.device, {code_size = len(shadow_curves_code), code = raw_data(shadow_curves_code), entrypoint = shadow_curves_entry, format = format, stage = .FRAGMENT, num_samplers = 2, num_storage_buffers = 1, num_uniform_buffers = 1})
	g.bg_vert = sdl.CreateGPUShader(g.device, {code_size = len(bg_vert_code), code = raw_data(bg_vert_code), entrypoint = vs_entry, format = format, stage = .VERTEX, num_uniform_buffers = 1})
	g.bg_frag = sdl.CreateGPUShader(g.device, {code_size = len(bg_frag_code), code = raw_data(bg_frag_code), entrypoint = fs_entry, format = format, stage = .FRAGMENT, num_samplers = 1, num_uniform_buffers = 1})
	if g.vert == nil || g.skinned_vert == nil || g.frag == nil || g.shadow_frag == nil || g.curves_vert == nil || g.curves_frag == nil || g.shadow_curves_frag == nil || g.bg_vert == nil || g.bg_frag == nil {
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
			g.pipelines[side][blend] = make_pipeline(g, g.vert, g.frag, STANDARD_ATTRIBUTES[:], side, blend == 1, false, false)
			g.skinned_pipelines[side][blend] = make_pipeline(g, g.skinned_vert, g.frag, SKINNED_ATTRIBUTES[:], side, blend == 1, false, true)
			g.curve_pipelines[side][blend] = make_pipeline(g, g.curves_vert, g.curves_frag, STANDARD_ATTRIBUTES[:], side, blend == 1, false, false)
			if g.pipelines[side][blend] == nil || g.skinned_pipelines[side][blend] == nil || g.curve_pipelines[side][blend] == nil {
				return .Shader
			}
		}
		g.shadow_pipelines[side] = make_shadow_pipeline(g, g.vert, g.shadow_frag, side, false)
		g.shadow_skinned_pipelines[side] = make_shadow_pipeline(g, g.skinned_vert, g.shadow_frag, side, true)
		g.shadow_curve_pipelines[side] = make_shadow_pipeline(g, g.curves_vert, g.shadow_curves_frag, side, false)
		if g.shadow_pipelines[side] == nil || g.shadow_skinned_pipelines[side] == nil || g.shadow_curve_pipelines[side] == nil {
			return .Shader
		}
	}
	g.bg_pipeline = make_background_pipeline(g)
	if g.bg_pipeline == nil {
		return .Shader
	}
	g.shadow_sampler = sdl.CreateGPUSampler(g.device, {min_filter = .NEAREST, mag_filter = .NEAREST, mipmap_mode = .NEAREST, address_mode_u = .CLAMP_TO_EDGE, address_mode_v = .CLAMP_TO_EDGE, address_mode_w = .CLAMP_TO_EDGE})
	g.linear_sampler = sdl.CreateGPUSampler(g.device, {min_filter = .LINEAR, mag_filter = .LINEAR, mipmap_mode = .LINEAR, address_mode_u = .REPEAT, address_mode_v = .REPEAT, address_mode_w = .REPEAT, max_lod = f32(o3d.MAX_MIPS)})
	g.env_sampler = sdl.CreateGPUSampler(g.device, {min_filter = .LINEAR, mag_filter = .LINEAR, mipmap_mode = .LINEAR, address_mode_u = .REPEAT, address_mode_v = .CLAMP_TO_EDGE, address_mode_w = .CLAMP_TO_EDGE, max_lod = f32(o3d.MAX_MIPS)})
	g.dummy_texture = sdl.CreateGPUTexture(g.device, {type = .D2, format = COLOR_FORMAT, usage = {.SAMPLER, .COLOR_TARGET}, width = 1, height = 1, layer_count_or_depth = 1, num_levels = 1, sample_count = ._1})
	g.morph_deltas.capacity = 64 << 10
	g.morph_deltas.buffer = sdl.CreateGPUBuffer(g.device, {usage = {.GRAPHICS_STORAGE_READ}, size = g.morph_deltas.capacity})
	g.curve_words.capacity = 64 << 10
	g.curve_words.buffer = sdl.CreateGPUBuffer(g.device, {usage = {.GRAPHICS_STORAGE_READ}, size = g.curve_words.capacity})
	if g.shadow_sampler == nil || g.linear_sampler == nil || g.env_sampler == nil || g.dummy_texture == nil || g.morph_deltas.buffer == nil || g.curve_words.buffer == nil {
		return .Device
	}
	// 1x1 fallback shadow and colour textures, cleared, bound for unused slots.
	cmd := sdl.AcquireGPUCommandBuffer(g.device)
	if cmd == nil {
		return .Device
	}
	g.dummy_shadow = sdl.CreateGPUTexture(g.device, {type = .D2, format = SHADOW_FORMAT, usage = {.SAMPLER, .COLOR_TARGET}, width = 1, height = 1, layer_count_or_depth = 1, num_levels = 1})
	if g.dummy_shadow == nil {
		_ = sdl.CancelGPUCommandBuffer(cmd)
		return .Device
	}
	clear_shadow := sdl.GPUColorTargetInfo{texture = g.dummy_shadow, clear_color = {1, 1, 1, 1}, load_op = .CLEAR, store_op = .STORE}
	sdl.EndGPURenderPass(sdl.BeginGPURenderPass(cmd, &clear_shadow, 1, nil))
	clear := sdl.GPUColorTargetInfo{texture = g.dummy_texture, clear_color = {1, 1, 1, 1}, load_op = .CLEAR, store_op = .STORE}
	sdl.EndGPURenderPass(sdl.BeginGPURenderPass(cmd, &clear, 1, nil))
	if !sdl.SubmitGPUCommandBuffer(cmd) {
		return .Device
	}
	return nil
}

destroy :: proc(g: ^GPU) {
	if g.device == nil {
		return
	}
	job_destroy(g, &g.job)
	for b in ([]^Pool_Buffer{&g.vertices, &g.skin_vertices, &g.morph_deltas, &g.indices, &g.curve_words}) {
		if b.buffer != nil {
			sdl.ReleaseGPUBuffer(g.device, b.buffer)
		}
	}
	for tex in g.textures {
		if tex != nil {
			sdl.ReleaseGPUTexture(g.device, tex)
		}
	}
	delete(g.textures)
	delete(g.externals)
	if g.dummy_texture != nil {
		sdl.ReleaseGPUTexture(g.device, g.dummy_texture)
	}
	if g.linear_sampler != nil {
		sdl.ReleaseGPUSampler(g.device, g.linear_sampler)
	}
	if g.env_sampler != nil {
		sdl.ReleaseGPUSampler(g.device, g.env_sampler)
	}
	for &c in g.custom {
		release_custom(g, &c)
	}
	delete(g.custom)
	for side in o3d.Side {
		for p in g.pipelines[side] {
			if p != nil {
				sdl.ReleaseGPUGraphicsPipeline(g.device, p)
			}
		}
		for p in g.skinned_pipelines[side] {
			if p != nil {
				sdl.ReleaseGPUGraphicsPipeline(g.device, p)
			}
		}
		for p in g.curve_pipelines[side] {
			if p != nil {
				sdl.ReleaseGPUGraphicsPipeline(g.device, p)
			}
		}
	}
	if g.bg_pipeline != nil {
		sdl.ReleaseGPUGraphicsPipeline(g.device, g.bg_pipeline)
	}
	release_targets(g)
	for p in g.shadow_pipelines {
		if p != nil {
			sdl.ReleaseGPUGraphicsPipeline(g.device, p)
		}
	}
	for p in g.shadow_skinned_pipelines {
		if p != nil {
			sdl.ReleaseGPUGraphicsPipeline(g.device, p)
		}
	}
	for p in g.shadow_curve_pipelines {
		if p != nil {
			sdl.ReleaseGPUGraphicsPipeline(g.device, p)
		}
	}
	if g.shadow_sampler != nil {
		sdl.ReleaseGPUSampler(g.device, g.shadow_sampler)
	}
	if g.dummy_shadow != nil {
		sdl.ReleaseGPUTexture(g.device, g.dummy_shadow)
	}
	for sh in ([]^sdl.GPUShader{g.shadow_frag, g.curves_vert, g.curves_frag, g.shadow_curves_frag}) {
		if sh != nil {
			sdl.ReleaseGPUShader(g.device, sh)
		}
	}
	if g.bg_vert != nil {
		sdl.ReleaseGPUShader(g.device, g.bg_vert)
	}
	if g.bg_frag != nil {
		sdl.ReleaseGPUShader(g.device, g.bg_frag)
	}
	if g.vert != nil {
		sdl.ReleaseGPUShader(g.device, g.vert)
	}
	if g.skinned_vert != nil {
		sdl.ReleaseGPUShader(g.device, g.skinned_vert)
	}
	if g.frag != nil {
		sdl.ReleaseGPUShader(g.device, g.frag)
	}
	if g.owns_device {
		sync.guard(&device_mutex)
		sdl.DestroyGPUDevice(g.device)
	}
	g^ = {}
}

// Binds the host's GPU texture to an external texture of `textures`
// (`o3d.texture_add_external`) for the renders that follow, until bound
// again: a video's frame, set before each render that draws it. It is
// sampled as an image is (RGBA, sRGB-encoded bytes in a UNORM format, not an
// _SRGB one), from the same device with `.SAMPLER` usage, at its own mip
// levels; the host keeps it alive while bound and unbinds it (nil, then
// sampled white) before releasing it. A binding for another table (its ids
// reused) is not used.
texture_set_external :: proc(g: ^GPU, textures: ^o3d.Textures, id: o3d.Texture_Id, tex: ^sdl.GPUTexture) {
	if textures == nil || id == 0 {
		return
	}
	if int(id) > len(g.externals) {
		resize(&g.externals, int(id))
	}
	g.externals[id - 1] = {tex, textures.generation}
}

// A texture to render into (and sample or read back).
target :: proc(g: ^GPU, width, height: u32) -> ^sdl.GPUTexture {
	return sdl.CreateGPUTexture(g.device, {type = .D2, format = COLOR_FORMAT, usage = {.SAMPLER, .COLOR_TARGET}, width = width, height = height, layer_count_or_depth = 1, num_levels = 1, sample_count = ._1})
}

// The standard material's vertex shader reads every attribute.
@(private = "file", rodata)
STANDARD_ATTRIBUTES := [4]sdl.GPUVertexAttribute {
	{location = 0, buffer_slot = 0, format = .FLOAT3, offset = 0},
	{location = 1, buffer_slot = 0, format = .FLOAT3, offset = 12},
	{location = 2, buffer_slot = 0, format = .FLOAT2, offset = 24},
	{location = 3, buffer_slot = 0, format = .FLOAT2, offset = 32},
}

@(private = "file", rodata)
SKINNED_ATTRIBUTES := [6]sdl.GPUVertexAttribute {
	{location = 0, buffer_slot = 0, format = .FLOAT3, offset = 0},
	{location = 1, buffer_slot = 0, format = .FLOAT3, offset = 12},
	{location = 2, buffer_slot = 0, format = .FLOAT2, offset = 24},
	{location = 3, buffer_slot = 0, format = .FLOAT2, offset = 32},
	{location = 4, buffer_slot = 1, format = .USHORT4, offset = 0},
	{location = 5, buffer_slot = 1, format = .FLOAT4, offset = 8},
}

// A pipeline drawing `Vertex`es (slot 0, plus `Skin_Vertex`es in slot 1 when
// `skinned`) for a material side, blended or not. The standard material's
// output is premultiplied; a ShaderMaterial's (`raw`) is as written,
// premultiplied by the blend.
@(private = "file")
make_pipeline :: proc(g: ^GPU, vert, frag: ^sdl.GPUShader, attrs: []sdl.GPUVertexAttribute, side: o3d.Side, blend, raw, skinned: bool) -> ^sdl.GPUGraphicsPipeline {
	buffers := [2]sdl.GPUVertexBufferDescription {
		{slot = 0, pitch = size_of(o3d.Vertex), input_rate = .VERTEX},
		{slot = 1, pitch = size_of(o3d.Skin_Vertex), input_rate = .VERTEX},
	}
	color := sdl.GPUColorTargetDescription {
		format = COLOR_FORMAT,
		blend_state = {
			enable_blend = blend,
			src_color_blendfactor = .SRC_ALPHA if raw else .ONE,
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
		vertex_shader = vert,
		fragment_shader = frag,
		vertex_input_state = {vertex_buffer_descriptions = &buffers[0], num_vertex_buffers = 2 if skinned else 1, vertex_attributes = raw_data(attrs), num_vertex_attributes = u32(len(attrs))},
		primitive_type = .TRIANGLELIST,
		rasterizer_state = {fill_mode = .FILL, cull_mode = cull, front_face = .COUNTER_CLOCKWISE},
		multisample_state = {sample_count = g.samples},
		depth_stencil_state = {compare_op = .LESS, enable_depth_test = true, enable_depth_write = !blend},
		target_info = {color_target_descriptions = &color, num_color_targets = 1, depth_stencil_format = g.depth_format, has_depth_stencil_target = true},
	}
	return sdl.CreateGPUGraphicsPipeline(g.device, info)
}

// The fullscreen background pipeline: 3 vertices via SV_VertexID, no vertex
// buffer, no depth test or write (drawn first into the cleared target).
@(private = "file")
make_background_pipeline :: proc(g: ^GPU) -> ^sdl.GPUGraphicsPipeline {
	color := sdl.GPUColorTargetDescription{format = COLOR_FORMAT}
	info := sdl.GPUGraphicsPipelineCreateInfo {
		vertex_shader = g.bg_vert,
		fragment_shader = g.bg_frag,
		primitive_type = .TRIANGLELIST,
		rasterizer_state = {fill_mode = .FILL, cull_mode = .NONE, front_face = .COUNTER_CLOCKWISE},
		multisample_state = {sample_count = g.samples},
		depth_stencil_state = {compare_op = .ALWAYS, enable_depth_test = false, enable_depth_write = false},
		target_info = {color_target_descriptions = &color, num_color_targets = 1, depth_stencil_format = g.depth_format, has_depth_stencil_target = true},
	}
	return sdl.CreateGPUGraphicsPipeline(g.device, info)
}

// A ShaderMaterial's pipeline, its shaders made from oksl's output the first
// time (nil: they could not be). `record` only looks it up (`make` false).
@(private)
custom_pipeline :: proc(g: ^GPU, shaders: ^o3d.Shaders, id: o3d.Shader_Id, side: o3d.Side, blend: bool, make := true) -> ^sdl.GPUGraphicsPipeline {
	sh := o3d.shader_get(shaders, id)
	if sh == nil {
		return nil
	}
	if !make {
		c := &g.custom[id - 1] if int(id) <= len(g.custom) else nil
		return c.pipelines[side][1 if blend else 0] if c != nil && !c.failed else nil
	}
	if len(g.custom) < int(id) {
		resize(&g.custom, int(id))
	}
	c := &g.custom[id - 1]
	if !c.made {
		c.made = true
		formats := sdl.GetGPUShaderFormats(g.device)
		for stage in o3d.Shader_Stage {
			m := o3d.shader_module(sh, stage)
			info := sdl.GPUShaderCreateInfo {
				stage               = .VERTEX if stage == .Vertex else .FRAGMENT,
				num_uniform_buffers = 1 if m.block_size > 0 else 0,
				num_samplers        = u32(m.num_samplers),
			}
			if .SPIRV in formats {
				info.code, info.code_size = ([^]u8)(raw_data(m.spirv)), len(m.spirv) * size_of(u32)
				info.entrypoint, info.format = oksl.SPIRV_ENTRY, {.SPIRV}
			} else {
				info.code, info.code_size = raw_data(m.msl), len(m.msl)
				info.entrypoint, info.format = oksl.MSL_ENTRY, {.MSL}
			}
			made := sdl.CreateGPUShader(g.device, info)
			if stage == .Vertex {
				c.vert = made
			} else {
				c.frag = made
			}
		}
		c.failed = c.vert == nil || c.frag == nil
	}
	if c.failed {
		return nil
	}
	p := &c.pipelines[side][1 if blend else 0]
	if p^ == nil {
		attrs: [len(o3d.Vertex_Attribute)]sdl.GPUVertexAttribute
		n := 0
		for a in o3d.Vertex_Attribute {
			if loc := sh.attributes[a]; loc >= 0 {
				row := o3d.VERTEX_ATTRIBUTES[a]
				attrs[n] = {location = u32(loc), buffer_slot = 0, format = .FLOAT2 if row.components == 2 else .FLOAT3, offset = u32(row.offset)}
				n += 1
			}
		}
		p^ = make_pipeline(g, c.vert, c.frag, attrs[:n], side, blend, true, false)
	}
	return p^
}

@(private)
release_custom :: proc(g: ^GPU, c: ^Custom) {
	for side in o3d.Side {
		for p in c.pipelines[side] {
			if p != nil {
				sdl.ReleaseGPUGraphicsPipeline(g.device, p)
			}
		}
	}
	if c.vert != nil {
		sdl.ReleaseGPUShader(g.device, c.vert)
	}
	if c.frag != nil {
		sdl.ReleaseGPUShader(g.device, c.frag)
	}
	c^ = {}
}

// The shadow pass's pipeline for casters of a material side: depth into the
// R32_FLOAT map (nearest kept by the depth test), the other side drawn, by
// `vert` and `frag` (`vert` or `skinned_vert` and `shadow_frag`, or
// `curves_vert` and `shadow_curves_frag` for exact shapes).
@(private = "file")
make_shadow_pipeline :: proc(g: ^GPU, vert, frag: ^sdl.GPUShader, side: o3d.Side, skinned: bool) -> ^sdl.GPUGraphicsPipeline {
	buffers := [2]sdl.GPUVertexBufferDescription {
		{slot = 0, pitch = size_of(o3d.Vertex), input_rate = .VERTEX},
		{slot = 1, pitch = size_of(o3d.Skin_Vertex), input_rate = .VERTEX},
	}
	attrs := SKINNED_ATTRIBUTES[:] if skinned else STANDARD_ATTRIBUTES[:]
	color := sdl.GPUColorTargetDescription{format = SHADOW_FORMAT}
	cull: sdl.GPUCullMode
	switch o3d.shadow_side(side) {
	case .Front:
		cull = .BACK
	case .Back:
		cull = .FRONT
	case .Double:
		cull = .NONE
	}
	info := sdl.GPUGraphicsPipelineCreateInfo {
		vertex_shader = vert,
		fragment_shader = frag,
		vertex_input_state = {vertex_buffer_descriptions = &buffers[0], num_vertex_buffers = 2 if skinned else 1, vertex_attributes = raw_data(attrs), num_vertex_attributes = u32(len(attrs))},
		primitive_type = .TRIANGLELIST,
		rasterizer_state = {fill_mode = .FILL, cull_mode = cull, front_face = .COUNTER_CLOCKWISE},
		multisample_state = {sample_count = ._1},
		depth_stencil_state = {compare_op = .LESS, enable_depth_test = true, enable_depth_write = true},
		target_info = {color_target_descriptions = &color, num_color_targets = 1, depth_stencil_format = g.depth_format, has_depth_stencil_target = true},
	}
	return sdl.CreateGPUGraphicsPipeline(g.device, info)
}

// A job's shadow map and its depth buffer at `size` squared.
@(private)
ensure_shadow :: proc(g: ^GPU, job: ^Job, idx: int, size: u32) -> bool {
	if job.shadow_sizes[idx] == size && job.shadow_maps[idx] != nil {
		return true
	}
	release_shadow_slot(g, job, idx)
	job.shadow_maps[idx] = sdl.CreateGPUTexture(g.device, {type = .D2, format = SHADOW_FORMAT, usage = {.SAMPLER, .COLOR_TARGET}, width = size, height = size, layer_count_or_depth = 1, num_levels = 1})
	job.shadow_depths[idx] = sdl.CreateGPUTexture(g.device, {type = .D2, format = g.depth_format, usage = {.DEPTH_STENCIL_TARGET}, width = size, height = size, layer_count_or_depth = 1, num_levels = 1})
	job.shadow_sizes[idx] = size
	return job.shadow_maps[idx] != nil && job.shadow_depths[idx] != nil
}

@(private)
release_shadow_slot :: proc(g: ^GPU, job: ^Job, idx: int) {
	if job.shadow_maps[idx] != nil {
		sdl.ReleaseGPUTexture(g.device, job.shadow_maps[idx])
	}
	if job.shadow_depths[idx] != nil {
		sdl.ReleaseGPUTexture(g.device, job.shadow_depths[idx])
	}
	job.shadow_maps[idx], job.shadow_depths[idx], job.shadow_sizes[idx] = nil, nil, 0
}

@(private = "file")
release_targets :: proc(g: ^GPU) {
	if g.color != nil {
		sdl.ReleaseGPUTexture(g.device, g.color)
	}
	if g.depth != nil {
		sdl.ReleaseGPUTexture(g.device, g.depth)
	}
	if g.resolved != nil {
		sdl.ReleaseGPUTexture(g.device, g.resolved)
	}
	g.color, g.depth, g.resolved, g.capacity = nil, nil, nil, {}
}

@(private)
ensure_targets :: proc(g: ^GPU, w, h: u32) -> bool {
	if w <= g.capacity.x && h <= g.capacity.y && g.depth != nil {
		return true
	}
	size := [2]u32{max(w, g.capacity.x), max(h, g.capacity.y)}
	release_targets(g)
	single := g.samples == ._1
	// The texture copied out is the resolve target, or the colour target
	// itself without multisampling.
	g.color = sdl.CreateGPUTexture(g.device, {type = .D2, format = COLOR_FORMAT, usage = {.COLOR_TARGET} if !single else {.COLOR_TARGET, .SAMPLER}, width = size.x, height = size.y, layer_count_or_depth = 1, num_levels = 1, sample_count = g.samples})
	if !single {
		g.resolved = sdl.CreateGPUTexture(g.device, {type = .D2, format = COLOR_FORMAT, usage = {.COLOR_TARGET, .SAMPLER}, width = size.x, height = size.y, layer_count_or_depth = 1, num_levels = 1, sample_count = ._1})
	}
	g.depth = sdl.CreateGPUTexture(g.device, {type = .D2, format = g.depth_format, usage = {.DEPTH_STENCIL_TARGET}, width = size.x, height = size.y, layer_count_or_depth = 1, num_levels = 1, sample_count = g.samples})
	g.capacity = size
	return g.color != nil && g.depth != nil && (single || g.resolved != nil)
}
