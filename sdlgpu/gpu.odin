// SDL_GPU renderer for o3d: Vulkan on Linux and Windows, Metal on macOS.
// It draws the same scene as the software renderer with the same programs
// (o3d.Programs: shaders/*.vert and *.frag compiled by oksl at load, as
// SPIR-V or MSL), into light (half floats, 4x MSAA, blended there),
// resolved, then tone mapped and encoded into the caller's texture
// (shaders/post.frag's passes).
//
// `init` creates a headless device, or borrows the host's (the one its
// window or 2D canvas uses, so the result composites without a copy).
// Geometry lives in one vertex and one index buffer, mirroring
// `Geometries`' shared arrays: what is new uploads at the start of a frame.
package o3d_sdlgpu

import "base:runtime"
import "core:math"
import "core:slice"
import "core:sync"
import sdl "vendor:sdl3"

import o3d ".."
import "deps:oksl"

// The frame as the caller gets it, and textures: sRGB-encoded bytes,
// premultiplied.
COLOR_FORMAT :: sdl.GPUTextureFormat.R8G8B8A8_UNORM
// The frame as it is drawn, the transmission snapshot and the reflection:
// light, premultiplied by coverage (three.js's HalfFloat targets).
LIGHT_FORMAT :: sdl.GPUTextureFormat.R16G16B16A16_FLOAT
// The mirror's pass's second target: each sample's run past the mirror
// (o3d.mirror_run) premultiplied by its coverage, and the coverage.
RUNS_FORMAT :: sdl.GPUTextureFormat.R16G16_FLOAT
// Motion blur's: each pixel's velocity (pixels over the shutter's first
// half) and nearness (o3d.motion_blur's).
VELOCITY_FORMAT :: sdl.GPUTextureFormat.R16G16_FLOAT
NEARNESS_FORMAT :: sdl.GPUTextureFormat.R16_FLOAT

// What a pipeline draws into: the frame's light; or the mirror's light and
// its runs, written (fs_mirror, fs_curves_mirror) or left as they are
// (ShaderMaterials, particles).
Targets :: enum u8 {
	Light,
	Mirror,
	Mirror_No_Runs,
	// A shadow map's depth (R32F, single-sampled; casters drawn with their
	// other side), and the frame's depth with velocity and nearness.
	Shadow,
	Velocity,
}


// The shadow map holds depth as a colour, so every backend can sample it.
SHADOW_FORMAT :: sdl.GPUTextureFormat.R32_FLOAT
// The host's image of the frame's depth as distance (`record_depth_into`).
DEPTH_IMAGE_FORMAT :: sdl.GPUTextureFormat.R16G16B16A16_FLOAT

// A GPU copy of one of `Geometries`' shared arrays: the first `uploaded`
// bytes are on the GPU; growing past `capacity` makes a new buffer.
Pool_Buffer :: struct {
	buffer:             ^sdl.GPUBuffer,
	capacity, uploaded: u32,
}

// GPU memory given up and not yet given back to SDL. Releasing a texture or
// buffer frees its container at once, while a presenting submit may be writing
// into that container (Vulkan's defragmenter moves what it holds), so on a
// device that one thread presents on and others use, only the presenting
// thread releases, outside its submits, and everything else pushes here. The
// caller's, one per device, zero to start with, kept until the GPUs given it
// are destroyed (their destroy pushes what they held) and what it holds is
// released. Only what the defragmenter can move comes here: pipelines, samplers
// and shaders have no memory, and a transfer buffer has its own allocation,
// which it never picks.
Retired :: struct {
	mutex:    sync.Mutex,
	textures: [dynamic]^sdl.GPUTexture,
	buffers:  [dynamic]^sdl.GPUBuffer,
}

// Gives up a texture (nil: nothing): into `r`, from any thread, or with no
// table to SDL at once. The arrays grow on the heap whatever the caller's
// allocator, since they outlive the frame and the thread that pushed.
retire_texture :: proc(device: ^sdl.GPUDevice, r: ^Retired, t: ^sdl.GPUTexture) {
	if t == nil {
		return
	}
	if r == nil {
		sdl.ReleaseGPUTexture(device, t)
		return
	}
	context.allocator = runtime.heap_allocator()
	sync.guard(&r.mutex)
	append(&r.textures, t)
}

// `retire_texture`, for a buffer.
retire_buffer :: proc(device: ^sdl.GPUDevice, r: ^Retired, b: ^sdl.GPUBuffer) {
	if b == nil {
		return
	}
	if r == nil {
		sdl.ReleaseGPUBuffer(device, b)
		return
	}
	context.allocator = runtime.heap_allocator()
	sync.guard(&r.mutex)
	append(&r.buffers, b)
}

// Gives everything in `r` back to SDL, keeping the arrays' capacity: on the
// presenting thread, outside its submits. Until then what is in it holds its
// memory, so a table that is never released only grows.
retired_release :: proc(device: ^sdl.GPUDevice, r: ^Retired) {
	sync.guard(&r.mutex)
	for t in r.textures {
		sdl.ReleaseGPUTexture(device, t)
	}
	for b in r.buffers {
		sdl.ReleaseGPUBuffer(device, b)
	}
	clear(&r.textures)
	clear(&r.buffers)
}

// Frees the arrays, after a last `retired_release`.
retired_destroy :: proc(r: ^Retired) {
	delete(r.textures)
	delete(r.buffers)
	r^ = {}
}

GPU :: struct {
	device:                   ^sdl.GPUDevice,
	owns_device:              bool,
	// Where memory given up goes (`init`'s); nil: to SDL at once.
	retired:                  ^Retired,
	// o3d's fullscreen passes' shaders (o3d.Pass: vertex, fragment).
	pass_shaders:             [o3d.Pass][2]^sdl.GPUShader,
	samples:                  sdl.GPUSampleCount,
	depth_format:             sdl.GPUTextureFormat,
	// o3d's own draws' variants (o3d shading.odin), and per variant its
	// shaders and pipelines (`program_pipeline`), made when a frame first
	// draws with them.
	programs:                 o3d.Programs,
	program_pipelines:        [dynamic]Program_Pipelines,
	bg_pipeline:              ^sdl.GPUGraphicsPipeline,
	// The passes after the frame's light (o3d.POST_PASSES: its output, bloom,
	// depth of field and motion blur), by pass; the depth pass's under its
	// draws.
	post_pipelines:           [o3d.Pass]^sdl.GPUGraphicsPipeline,
	velocity_far_pipeline:    ^sdl.GPUGraphicsPipeline,
	// The frame's depth as distance into the host's image.
	depth_pipeline:           ^sdl.GPUGraphicsPipeline,
	// The frame's targets, at `capacity`: a frame draws into their top-left
	// width x height and is copied out, so frames of other sizes (several
	// views, a view whose size animates) reuse them; they only grow. `color`
	// (light) is multisampled when `samples` is, resolved into `resolved`;
	// every pass draws there (the snapshot's, the mirror's, the frame's),
	// the frame's output into `output`.
	color, depth:             ^sdl.GPUTexture,
	resolved:                 ^sdl.GPUTexture,
	output:                   ^sdl.GPUTexture,
	capacity:                 [2]u32,
	vertices:                 Pool_Buffer,
	skin_vertices:            Pool_Buffer,
	morph_deltas:             Pool_Buffer,
	indices:                  Pool_Buffer,
	curve_words:              Pool_Buffer,
	// Every prepare's uploads to the pools (the geometry's, a job's
	// dynamic ones) through one transfer buffer, kept and grown
	// geometrically, mapped once a prepare and cycled: SDL hands a fresh
	// backing while an earlier frame's copy is in flight (ovg's streams).
	stream:                   ^sdl.GPUTransferBuffer,
	stream_capacity:          u32,
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
	// 1x1 transparent black, both layers: what a draw sees in frameMap with
	// no snapshot or reflection (glass in the mirror, a reflective draw
	// without one).
	clear_texture:            ^sdl.GPUTexture,
	linear_sampler:           ^sdl.GPUSampler,
	env_sampler:              ^sdl.GPUSampler,
	// Mipmapped, clamped: a frame's snapshot for transmissive draws.
	frame_sampler:            ^sdl.GPUSampler,
	// Per shader (its id - 1): ShaderMaterials' GPU shaders and pipelines,
	// made the first time a frame draws with them.
	custom:                   [dynamic]Custom,
	// The shadow pass (shadow.odin): its fragment shader (depth out) and its
	// pipelines by the casters' material side (the maps are a job's).
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
	pipelines:  [o3d.Side][o3d.Blend]^sdl.GPUGraphicsPipeline,
	// Through the mirror.
	mirror_pipelines:   [o3d.Side][o3d.Blend]^sdl.GPUGraphicsPipeline,
	made:       bool,
	failed:     bool,
	// The frame's depth pass's (o3d.Shader.depth): its shaders and
	// pipelines by side.
	depth_vert, depth_frag: ^sdl.GPUShader,
	depth_pipelines:        [o3d.Side]^sdl.GPUGraphicsPipeline,
	depth_made:             bool,
	depth_failed:           bool,
}

// The textures a post pass reads, by what its samplers name.
Post_Bindings :: [o3d.Pass_Map]^sdl.GPUTexture

// SDL_Init and making or destroying a device are not safe from several
// threads at once (tests run in parallel; a host may init from workers):
// they take turns.
@(private = "file")
device_mutex: sync.Mutex

init :: proc(g: ^GPU, device: ^sdl.GPUDevice = nil, retired: ^Retired = nil) -> (err: o3d.Error) {
	g^ = {device = device, retired = retired}
	// What was made goes with a failure (the host's device stays).
	defer if err != nil {
		destroy(g)
	}
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
	// o3d's shaders are oksl's, compiled at run time to SPIR-V or MSL.
	if sdl.GetGPUShaderFormats(g.device) & {.SPIRV, .MSL} == {} {
		return .Device
	}
	for pass in o3d.Pass {
		x := o3d.pass_get(&g.programs, pass)
		if x == nil {
			return .Shader
		}
		for m, stage in ([2]^oksl.Module{x.pipeline.vertex, x.pipeline.fragment}) {
			g.pass_shaders[pass][stage] = module_shader(g, m, .VERTEX if stage == 0 else .FRAGMENT)
			if g.pass_shaders[pass][stage] == nil {
				return .Shader
			}
		}
	}
	g.depth_format = .D32_FLOAT
	if !sdl.GPUTextureSupportsFormat(g.device, g.depth_format, .D2, {.DEPTH_STENCIL_TARGET}) {
		g.depth_format = .D24_UNORM
	}
	g.samples = ._4
	if !sdl.GPUTextureSupportsSampleCount(g.device, LIGHT_FORMAT, ._4) || !sdl.GPUTextureSupportsSampleCount(g.device, RUNS_FORMAT, ._4) || !sdl.GPUTextureSupportsSampleCount(g.device, g.depth_format, ._4) {
		g.samples = ._1
	}
	g.bg_pipeline = make_background_pipeline(g)
	g.depth_pipeline = make_fullscreen_pipeline(g, g.pass_shaders[.Depth_Image][0], g.pass_shaders[.Depth_Image][1], DEPTH_IMAGE_FORMAT)
	// Each post pass into its targets' formats.
	for pass in o3d.POST_PASSES {
		vert, frag := g.pass_shaders[pass][0], g.pass_shaders[pass][1]
		#partial switch pass {
		case .Output, .Focus_Gather:
			g.post_pipelines[pass] = make_fullscreen_pipeline(g, vert, frag, COLOR_FORMAT)
		case .Bloom_Bright, .Bloom_Down, .Bloom_Blur, .Motion_Gather:
			g.post_pipelines[pass] = make_fullscreen_pipeline(g, vert, frag, LIGHT_FORMAT)
		case .Focus_Rows, .Focus_Tiles, .Focus_Near:
			g.post_pipelines[pass] = make_fullscreen_pipeline(g, vert, frag, SHADOW_FORMAT)
		case .Focus_Half:
			g.post_pipelines[pass] = make_fullscreen_pipeline(g, vert, frag, LIGHT_FORMAT, SHADOW_FORMAT)
		case .Motion_Rows, .Motion_Tiles, .Motion_Around:
			g.post_pipelines[pass] = make_fullscreen_pipeline(g, vert, frag, VELOCITY_FORMAT)
		}
		if g.post_pipelines[pass] == nil {
			return .Shader
		}
	}
	g.velocity_far_pipeline = make_velocity_pipeline(g, g.pass_shaders[.Velocity_Far][0], g.pass_shaders[.Velocity_Far][1], nil, .Double, false, far = true)
	if g.velocity_far_pipeline == nil || g.bg_pipeline == nil || g.depth_pipeline == nil {
		return .Shader
	}
	g.shadow_sampler = sdl.CreateGPUSampler(g.device, {min_filter = .NEAREST, mag_filter = .NEAREST, mipmap_mode = .NEAREST, address_mode_u = .CLAMP_TO_EDGE, address_mode_v = .CLAMP_TO_EDGE, address_mode_w = .CLAMP_TO_EDGE})
	g.linear_sampler = sdl.CreateGPUSampler(g.device, {min_filter = .LINEAR, mag_filter = .LINEAR, mipmap_mode = .LINEAR, address_mode_u = .REPEAT, address_mode_v = .REPEAT, address_mode_w = .REPEAT, max_lod = f32(o3d.MAX_MIPS)})
	g.env_sampler = sdl.CreateGPUSampler(g.device, {min_filter = .LINEAR, mag_filter = .LINEAR, mipmap_mode = .LINEAR, address_mode_u = .REPEAT, address_mode_v = .CLAMP_TO_EDGE, address_mode_w = .CLAMP_TO_EDGE, max_lod = f32(o3d.MAX_MIPS)})
	g.frame_sampler = sdl.CreateGPUSampler(g.device, {min_filter = .LINEAR, mag_filter = .LINEAR, mipmap_mode = .LINEAR, address_mode_u = .CLAMP_TO_EDGE, address_mode_v = .CLAMP_TO_EDGE, address_mode_w = .CLAMP_TO_EDGE, max_lod = f32(o3d.MAX_MIPS)})
	g.dummy_texture = sdl.CreateGPUTexture(g.device, {type = .D2, format = COLOR_FORMAT, usage = {.SAMPLER, .COLOR_TARGET}, width = 1, height = 1, layer_count_or_depth = 1, num_levels = 1, sample_count = ._1})
	g.clear_texture = sdl.CreateGPUTexture(g.device, {type = .D2_ARRAY, format = COLOR_FORMAT, usage = {.SAMPLER, .COLOR_TARGET}, width = 1, height = 1, layer_count_or_depth = 2, num_levels = 1, sample_count = ._1})
	g.morph_deltas.capacity = 64 << 10
	g.morph_deltas.buffer = sdl.CreateGPUBuffer(g.device, {usage = {.GRAPHICS_STORAGE_READ}, size = g.morph_deltas.capacity})
	g.curve_words.capacity = 64 << 10
	g.curve_words.buffer = sdl.CreateGPUBuffer(g.device, {usage = {.GRAPHICS_STORAGE_READ}, size = g.curve_words.capacity})
	if g.shadow_sampler == nil || g.linear_sampler == nil || g.env_sampler == nil || g.frame_sampler == nil || g.dummy_texture == nil || g.clear_texture == nil || g.morph_deltas.buffer == nil || g.curve_words.buffer == nil {
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
	clear_shadow := sdl.GPUColorTargetInfo{texture = g.dummy_shadow, clear_color = {0, 0, 0, 0}, load_op = .CLEAR, store_op = .STORE}
	sdl.EndGPURenderPass(sdl.BeginGPURenderPass(cmd, &clear_shadow, 1, nil))
	clear := sdl.GPUColorTargetInfo{texture = g.dummy_texture, clear_color = {1, 1, 1, 1}, load_op = .CLEAR, store_op = .STORE}
	sdl.EndGPURenderPass(sdl.BeginGPURenderPass(cmd, &clear, 1, nil))
	for layer in u32(0) ..< 2 {
		transparent := sdl.GPUColorTargetInfo{texture = g.clear_texture, layer_or_depth_plane = layer, clear_color = {0, 0, 0, 0}, load_op = .CLEAR, store_op = .STORE}
		sdl.EndGPURenderPass(sdl.BeginGPURenderPass(cmd, &transparent, 1, nil))
	}
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
		retire_buffer(g.device, g.retired, b.buffer)
	}
	if g.stream != nil {
		sdl.ReleaseGPUTransferBuffer(g.device, g.stream)
	}
	for tex in g.textures {
		retire_texture(g.device, g.retired, tex)
	}
	delete(g.textures)
	delete(g.externals)
	retire_texture(g.device, g.retired, g.dummy_texture)
	retire_texture(g.device, g.retired, g.clear_texture)
	if g.linear_sampler != nil {
		sdl.ReleaseGPUSampler(g.device, g.linear_sampler)
	}
	if g.env_sampler != nil {
		sdl.ReleaseGPUSampler(g.device, g.env_sampler)
	}
	if g.frame_sampler != nil {
		sdl.ReleaseGPUSampler(g.device, g.frame_sampler)
	}
	for &c in g.custom {
		release_custom(g, &c)
	}
	delete(g.custom)
	for &pp in g.program_pipelines {
		release_program(g, &pp)
	}
	delete(g.program_pipelines)
	o3d.programs_destroy(&g.programs)
	for p in ([]^sdl.GPUGraphicsPipeline{g.bg_pipeline, g.depth_pipeline}) {
		if p != nil {
			sdl.ReleaseGPUGraphicsPipeline(g.device, p)
		}
	}
	for p in g.post_pipelines {
		if p != nil {
			sdl.ReleaseGPUGraphicsPipeline(g.device, p)
		}
	}
	if g.velocity_far_pipeline != nil {
		sdl.ReleaseGPUGraphicsPipeline(g.device, g.velocity_far_pipeline)
	}
	release_targets(g)
	if g.shadow_sampler != nil {
		sdl.ReleaseGPUSampler(g.device, g.shadow_sampler)
	}
	retire_texture(g.device, g.retired, g.dummy_shadow)
	for shaders in g.pass_shaders {
		for sh in shaders {
			if sh != nil {
				sdl.ReleaseGPUShader(g.device, sh)
			}
		}
	}
	if g.owns_device {
		// Nothing can be released into its device after this.
		if g.retired != nil {
			retired_release(g.device, g.retired)
		}
		sync.guard(&device_mutex)
		sdl.DestroyGPUDevice(g.device)
	}
	g^ = {}
}

// Binds the host's GPU texture to an external texture of `textures`
// (`o3d.texture_add_external`) for the renders that follow, until bound
// again: a video's frame, set before each render that draws it. It is
// sampled as an image is (RGBA, sRGB-encoded values in a UNORM format, not
// an _SRGB one, or the same values in a float one such as RGBA16F, past 1
// as they are), from the same device with `.SAMPLER` usage, at its own mip
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

// A texture for `record_depth_into` (RGBA16F; to sample or read back).
depth_image_target :: proc(g: ^GPU, width, height: u32) -> ^sdl.GPUTexture {
	return sdl.CreateGPUTexture(g.device, {type = .D2, format = DEPTH_IMAGE_FORMAT, usage = {.SAMPLER, .COLOR_TARGET}, width = width, height = height, layer_count_or_depth = 1, num_levels = 1, sample_count = ._1})
}


// A pipeline drawing `Vertex`es (slot 0, plus `Skin_Vertex`es in slot 1 when
// `skinned`) for a material side and blend into `targets`. The standard
// material's output is premultiplied; a ShaderMaterial's (`raw`) is as
// written, premultiplied by the blend. Through a mirror front faces are
// clockwise, and the runs blend as the light's alpha does.
@(private = "file")
make_pipeline :: proc(g: ^GPU, vert, frag: ^sdl.GPUShader, attrs: []sdl.GPUVertexAttribute, side: o3d.Side, blend: o3d.Blend, raw, skinned: bool, targets: Targets) -> ^sdl.GPUGraphicsPipeline {
	buffers := [2]sdl.GPUVertexBufferDescription {
		{slot = 0, pitch = size_of(o3d.Vertex), input_rate = .VERTEX},
		{slot = 1, pitch = size_of(o3d.Skin_Vertex), input_rate = .VERTEX},
	}
	mirrored := targets != .Light
	colors := [2]sdl.GPUColorTargetDescription {
		{
			format = LIGHT_FORMAT,
			blend_state = {
				enable_blend = !o3d.replaces(blend),
				src_color_blendfactor = .SRC_ALPHA if raw else .ONE,
				dst_color_blendfactor = .ONE if blend == .Add else .ONE_MINUS_SRC_ALPHA,
				color_blend_op = .ADD,
				src_alpha_blendfactor = .ONE,
				dst_alpha_blendfactor = .ONE_MINUS_SRC_ALPHA,
				alpha_blend_op = .ADD,
			},
		},
		{
			format = RUNS_FORMAT,
			blend_state = {
				enable_blend = !o3d.replaces(blend),
				src_color_blendfactor = .ONE,
				dst_color_blendfactor = .ONE_MINUS_SRC_ALPHA,
				color_blend_op = .ADD,
				src_alpha_blendfactor = .ONE,
				dst_alpha_blendfactor = .ONE_MINUS_SRC_ALPHA,
				alpha_blend_op = .ADD,
				enable_color_write_mask = targets == .Mirror_No_Runs,
			},
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
		rasterizer_state = {fill_mode = .FILL, cull_mode = cull, front_face = .CLOCKWISE if mirrored else .COUNTER_CLOCKWISE},
		multisample_state = {sample_count = g.samples},
		depth_stencil_state = {compare_op = .GREATER, enable_depth_test = true, enable_depth_write = o3d.replaces(blend)},
		target_info = {color_target_descriptions = &colors[0], num_color_targets = 2 if mirrored else 1, depth_stencil_format = g.depth_format, has_depth_stencil_target = true},
	}
	return sdl.CreateGPUGraphicsPipeline(g.device, info)
}


// The fullscreen background pipeline: 3 vertices via SV_VertexID, no vertex
// buffer, no depth test or write (drawn first into the cleared target).
@(private = "file")
make_background_pipeline :: proc(g: ^GPU) -> ^sdl.GPUGraphicsPipeline {
	color := sdl.GPUColorTargetDescription{format = LIGHT_FORMAT}
	info := sdl.GPUGraphicsPipelineCreateInfo {
		vertex_shader = g.pass_shaders[.Background][0],
		fragment_shader = g.pass_shaders[.Background][1],
		primitive_type = .TRIANGLELIST,
		rasterizer_state = {fill_mode = .FILL, cull_mode = .NONE, front_face = .COUNTER_CLOCKWISE},
		multisample_state = {sample_count = g.samples},
		depth_stencil_state = {compare_op = .ALWAYS, enable_depth_test = false, enable_depth_write = false},
		target_info = {color_target_descriptions = &color, num_color_targets = 1, depth_stencil_format = g.depth_format, has_depth_stencil_target = true},
	}
	return sdl.CreateGPUGraphicsPipeline(g.device, info)
}

// A fullscreen pass into single-sampled targets of `formats`, no depth: 3
// vertices via SV_VertexID (the frame's depth as distance into the host's
// image, the frame's output, its depth of field's passes).
@(private = "file")
make_fullscreen_pipeline :: proc(g: ^GPU, vert, frag: ^sdl.GPUShader, formats: ..sdl.GPUTextureFormat) -> ^sdl.GPUGraphicsPipeline {
	colors: [2]sdl.GPUColorTargetDescription
	for f, i in formats {
		colors[i] = {format = f}
	}
	info := sdl.GPUGraphicsPipelineCreateInfo {
		vertex_shader = vert,
		fragment_shader = frag,
		primitive_type = .TRIANGLELIST,
		rasterizer_state = {fill_mode = .FILL, cull_mode = .NONE, front_face = .COUNTER_CLOCKWISE},
		multisample_state = {sample_count = ._1},
		target_info = {color_target_descriptions = &colors[0], num_color_targets = u32(len(formats))},
	}
	return sdl.CreateGPUGraphicsPipeline(g.device, info)
}

// A ShaderMaterial's pipeline, into the frame's light or through the
// mirror (`mirrored`), its shaders made from oksl's output the first time
// (nil: they could not be). `record` only looks it up (`make` false).
@(private)
custom_pipeline :: proc(g: ^GPU, shaders: ^o3d.Shaders, id: o3d.Shader_Id, side: o3d.Side, blend: o3d.Blend, make := true, mirrored := false) -> ^sdl.GPUGraphicsPipeline {
	sh := o3d.shader_get(shaders, id)
	if sh == nil {
		return nil
	}
	if !make {
		c := &g.custom[id - 1] if int(id) <= len(g.custom) else nil
		if c == nil || c.failed {
			return nil
		}
		return c.mirror_pipelines[side][blend] if mirrored else c.pipelines[side][blend]
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
				code := slice.to_bytes(m.spirv)
				info.code, info.code_size = raw_data(code), len(code)
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
	p := &c.mirror_pipelines[side][blend] if mirrored else &c.pipelines[side][blend]
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
		p^ = make_pipeline(g, c.vert, c.frag, attrs[:n], side, blend, true, false, .Mirror_No_Runs if mirrored else .Light)
	}
	return p^
}

// A ShaderMaterial's pipeline for the frame's depth pass (o3d.Shader.depth:
// its vertex stage with a tap, DEPTH_FRAGMENT), as make_velocity_pipeline's
// targets: its vertices from slot 0, their twins (posed as at the shutter's
// opening) from slot 1. `make` false only looks it up.
@(private)
custom_depth_pipeline :: proc(g: ^GPU, shaders: ^o3d.Shaders, id: o3d.Shader_Id, side: o3d.Side, make := true) -> ^sdl.GPUGraphicsPipeline {
	sh := o3d.shader_get(shaders, id)
	if sh == nil || sh.depth == nil || int(id) > len(g.custom) && !make {
		return nil
	}
	if len(g.custom) < int(id) {
		resize(&g.custom, int(id))
	}
	c := &g.custom[id - 1]
	if !make {
		return nil if c.depth_failed else c.depth_pipelines[side]
	}
	if !c.depth_made {
		c.depth_made = true
		formats := sdl.GetGPUShaderFormats(g.device)
		for m, stage in ([2]^oksl.Module{sh.depth.vertex, sh.depth.fragment}) {
			info := sdl.GPUShaderCreateInfo {
				stage               = .VERTEX if stage == 0 else .FRAGMENT,
				num_uniform_buffers = 1 if m.block_size > 0 else 0,
			}
			if .SPIRV in formats {
				code := slice.to_bytes(m.spirv)
				info.code, info.code_size = raw_data(code), len(code)
				info.entrypoint, info.format = oksl.SPIRV_ENTRY, {.SPIRV}
			} else {
				info.code, info.code_size = raw_data(m.msl), len(m.msl)
				info.entrypoint, info.format = oksl.MSL_ENTRY, {.MSL}
			}
			made := sdl.CreateGPUShader(g.device, info)
			if stage == 0 {
				c.depth_vert = made
			} else {
				c.depth_frag = made
			}
		}
		c.depth_failed = c.depth_vert == nil || c.depth_frag == nil
	}
	if c.depth_failed {
		return nil
	}
	p := &c.depth_pipelines[side]
	if p^ == nil {
		attrs: [len(o3d.Vertex_Attribute) + 2]sdl.GPUVertexAttribute
		n := 0
		for a in o3d.Vertex_Attribute {
			if loc := sh.depth_attributes[a]; loc >= 0 {
				row := o3d.VERTEX_ATTRIBUTES[a]
				attrs[n] = {location = u32(loc), buffer_slot = 0, format = .FLOAT2 if row.components == 2 else .FLOAT3, offset = u32(row.offset)}
				n += 1
			}
		}
		for loc, k in sh.depth_twins {
			if loc >= 0 {
				attrs[n] = {location = u32(loc), buffer_slot = 1, format = .FLOAT3, offset = u32(o3d.VERTEX_ATTRIBUTES[.Position if k == 0 else .Normal].offset)}
				n += 1
			}
		}
		p^ = make_velocity_pipeline(g, c.depth_vert, c.depth_frag, attrs[:n], side, false, twins = true)
	}
	return p^
}

@(private)
release_custom :: proc(g: ^GPU, c: ^Custom) {
	for p in c.depth_pipelines {
		if p != nil {
			sdl.ReleaseGPUGraphicsPipeline(g.device, p)
		}
	}
	for sh in ([2]^sdl.GPUShader{c.depth_vert, c.depth_frag}) {
		if sh != nil {
			sdl.ReleaseGPUShader(g.device, sh)
		}
	}
	for side in o3d.Side {
		for p in c.pipelines[side] {
			if p != nil {
				sdl.ReleaseGPUGraphicsPipeline(g.device, p)
			}
		}
		for p in c.mirror_pipelines[side] {
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

// An o3d variant's shaders and pipelines (`GPU.program_pipelines`), by the
// targets, the material's side and its blend.
Program_Pipelines :: struct {
	made, failed: bool,
	vert, frag:   ^sdl.GPUShader,
	pipelines:    [Targets][o3d.Side][o3d.Blend]^sdl.GPUGraphicsPipeline,
}

@(private)
release_program :: proc(g: ^GPU, pp: ^Program_Pipelines) {
	for &by_side in pp.pipelines {
		for &by_blend in by_side {
			for p in by_blend {
				if p != nil {
					sdl.ReleaseGPUGraphicsPipeline(g.device, p)
				}
			}
		}
	}
	for sh in ([2]^sdl.GPUShader{pp.vert, pp.frag}) {
		if sh != nil {
			sdl.ReleaseGPUShader(g.device, sh)
		}
	}
	pp^ = {}
}

// SDL's format for each of o3d.vert's attributes.
@(private = "file", rodata)
ATTRIBUTE_FORMATS := [o3d.Program_Attribute]sdl.GPUVertexElementFormat {
	.Position = .FLOAT3,
	.Normal   = .FLOAT3,
	.Uv       = .FLOAT2,
	.Uv1      = .FLOAT2,
	.Color    = .UBYTE4_NORM,
	.Joints   = .USHORT4,
	.Weights  = .FLOAT4,
}

// An oksl stage's SDL shader: its SPIR-V, else its MSL.
@(private)
module_shader :: proc(g: ^GPU, m: ^oksl.Module, stage: sdl.GPUShaderStage) -> ^sdl.GPUShader {
	info := sdl.GPUShaderCreateInfo {
		stage               = stage,
		num_samplers        = u32(m.num_samplers),
		num_storage_buffers = u32(len(m.storage_buffers)),
		num_uniform_buffers = u32(m.num_uniform_buffers),
	}
	if .SPIRV in sdl.GetGPUShaderFormats(g.device) {
		code := slice.to_bytes(m.spirv)
		info.code, info.code_size = raw_data(code), len(code)
		info.entrypoint, info.format = oksl.SPIRV_ENTRY, {.SPIRV}
	} else {
		info.code, info.code_size = raw_data(m.msl), len(m.msl)
		info.entrypoint, info.format = oksl.MSL_ENTRY, {.MSL}
	}
	return sdl.CreateGPUShader(g.device, info)
}

// Variant `id`'s pipeline for a side and blend into `targets`, its shaders
// made from oksl's output the first time (nil: they could not be). `make`
// false only looks it up (`record` makes nothing).
@(private)
program_pipeline :: proc(g: ^GPU, id: i32, side: o3d.Side, blend: o3d.Blend, targets: Targets, make := true) -> ^sdl.GPUGraphicsPipeline {
	if int(id) >= len(g.program_pipelines) {
		if !make {
			return nil
		}
		resize(&g.program_pipelines, int(id) + 1)
	}
	pp := &g.program_pipelines[id]
	x := &g.programs.list[id]
	if !pp.made {
		if !make {
			return nil
		}
		pp.made = true
		pp.vert = module_shader(g, x.pipeline.vertex, .VERTEX)
		pp.frag = module_shader(g, x.pipeline.fragment, .FRAGMENT)
		pp.failed = pp.vert == nil || pp.frag == nil
	}
	if pp.failed {
		return nil
	}
	p := &pp.pipelines[targets][side][blend]
	if p^ == nil && make {
		attrs: [len(o3d.Program_Attribute)]sdl.GPUVertexAttribute
		n := 0
		for a in o3d.Program_Attribute {
			if loc := x.attributes[a]; loc >= 0 {
				row := o3d.PROGRAM_ATTRIBUTES[a]
				attrs[n] = {location = u32(loc), buffer_slot = 1 if row.skin else 0, format = ATTRIBUTE_FORMATS[a], offset = u32(row.offset)}
				n += 1
			}
		}
		skinned := x.key.geometry == .Skinned
		#partial switch targets {
		case .Shadow:
			p^ = make_shadow_pipeline(g, pp.vert, pp.frag, attrs[:n], side, skinned)
		case .Velocity:
			p^ = make_velocity_pipeline(g, pp.vert, pp.frag, attrs[:n], side, skinned)
		case:
			p^ = make_pipeline(g, pp.vert, pp.frag, attrs[:n], side, blend, false, skinned, targets)
		}
	}
	return p^
}

// The shadow pass's pipeline for casters of a material side: depth into the
// R32_FLOAT map (nearest, the greatest, kept by the depth test), the other side drawn, by
// `vert` and `frag` (`vert` or `skinned_vert` and `shadow_frag`, or
// `curves_vert` and `shadow_curves_frag` for exact shapes).
@(private = "file")
make_shadow_pipeline :: proc(g: ^GPU, vert, frag: ^sdl.GPUShader, attrs: []sdl.GPUVertexAttribute, side: o3d.Side, skinned: bool) -> ^sdl.GPUGraphicsPipeline {
	buffers := [2]sdl.GPUVertexBufferDescription {
		{slot = 0, pitch = size_of(o3d.Vertex), input_rate = .VERTEX},
		{slot = 1, pitch = size_of(o3d.Skin_Vertex), input_rate = .VERTEX},
	}
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
		depth_stencil_state = {compare_op = .GREATER, enable_depth_test = true, enable_depth_write = true},
		target_info = {color_target_descriptions = &color, num_color_targets = 1, depth_stencil_format = g.depth_format, has_depth_stencil_target = true},
	}
	return sdl.CreateGPUGraphicsPipeline(g.device, info)
}

// The frame's depth pass with motion blur: its depth map, velocity and
// nearness, culled by the material's own side; or what is under its draws
// (`far`: a full-screen triangle, no depth test).
@(private = "file")
make_velocity_pipeline :: proc(g: ^GPU, vert, frag: ^sdl.GPUShader, attrs: []sdl.GPUVertexAttribute, side: o3d.Side, skinned: bool, far := false, twins := false) -> ^sdl.GPUGraphicsPipeline {
	buffers := [2]sdl.GPUVertexBufferDescription {
		{slot = 0, pitch = size_of(o3d.Vertex), input_rate = .VERTEX},
		{slot = 1, pitch = size_of(o3d.Skin_Vertex), input_rate = .VERTEX},
	}
	if twins {
		// A ShaderMaterial's: its vertices and their twins, both Vertex rows.
		buffers[1].pitch = size_of(o3d.Vertex)
	}
	two := skinned || len(attrs) > 0 && slice.any_of_proc(attrs, proc(a: sdl.GPUVertexAttribute) -> bool {return a.buffer_slot == 1})
	colors := [3]sdl.GPUColorTargetDescription{{format = SHADOW_FORMAT}, {format = VELOCITY_FORMAT}, {format = NEARNESS_FORMAT}}
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
		vertex_input_state = {vertex_buffer_descriptions = &buffers[0], num_vertex_buffers = 2 if two else 1, vertex_attributes = raw_data(attrs), num_vertex_attributes = u32(len(attrs))},
		primitive_type = .TRIANGLELIST,
		rasterizer_state = {fill_mode = .FILL, cull_mode = cull, front_face = .COUNTER_CLOCKWISE},
		multisample_state = {sample_count = ._1},
		depth_stencil_state = {compare_op = .GREATER, enable_depth_test = true, enable_depth_write = true},
		target_info = {color_target_descriptions = &colors[0], num_color_targets = len(colors), depth_stencil_format = g.depth_format, has_depth_stencil_target = true},
	}
	if far {
		info.vertex_input_state = {}
		info.depth_stencil_state = {}
	}
	return sdl.CreateGPUGraphicsPipeline(g.device, info)
}

// The motion blur filter's tables (o3d.Motion_Tables' on the GPU): its
// tiles' rows' longest velocity (tiles across by the frame's rows), its
// tiles' and the longest about them, made for a frame's size and tile, the
// caller's, kept across frames.
Motion_Tables :: struct {
	rows, tiles, around: ^sdl.GPUTexture,
	size:                [3]u32,
}

motion_tables_destroy :: proc(g: ^GPU, t: ^Motion_Tables) {
	for tex in ([]^sdl.GPUTexture{t.rows, t.tiles, t.around}) {
		retire_texture(g.device, g.retired, tex)
	}
	t^ = {}
}

// The motion blur filter (o3d.motion_blur's) recorded into `cmd` on `g`'s
// device: `color` (light, premultiplied, its texture `color_size`; 0: as
// the frame) blurred along `velocity` (VELOCITY_FORMAT, pixels over the
// shutter's first half) and ordered by `nearness` (NEARNESS_FORMAT, larger
// nearer), each `width x height` from the top left, into the same of `out`
// (LIGHT_FORMAT, not `color`).
record_motion_blur :: proc(g: ^GPU, t: ^Motion_Tables, cmd: ^sdl.GPUCommandBuffer, color, velocity, nearness, out: ^sdl.GPUTexture, width, height: u32, params: o3d.Motion_Blur, color_size := [2]u32{}) -> o3d.Error {
	tile := u32(max(math.ceil(params.max_blur), 1))
	tiles := [2]u32{(width + tile - 1) / tile, (height + tile - 1) / tile}
	if t.size != {width, height, tile} || t.tiles == nil {
		motion_tables_destroy(g, t)
		for &tex, k in ([]^^sdl.GPUTexture{&t.rows, &t.tiles, &t.around}) {
			tex^ = sdl.CreateGPUTexture(g.device, {type = .D2, format = VELOCITY_FORMAT, usage = {.SAMPLER, .COLOR_TARGET}, width = tiles.x, height = height if k == 0 else tiles.y, layer_count_or_depth = 1, num_levels = 1})
			if tex^ == nil {
				return .Device
			}
		}
		t.size = {width, height, tile}
	}
	cs := color_size if color_size != {} else [2]u32{width, height}
	u := o3d.Post_Block {
		tone        = {0, 0, f32(cs.x), f32(cs.y)},
		frame       = {f32(width), f32(height), 0, 0},
		motion      = {max(params.max_blur, 0), params.depth_scale if params.depth_scale > 0 else o3d.MOTION_DEPTH_SCALE, f32(tile), 0},
		motion_size = {f32(width), f32(height), f32(tiles.x), f32(tiles.y)},
	}
	post_draw(g, cmd, .Motion_Rows, {t.rows}, {tiles.x, height}, #partial Post_Bindings{.Velocity = velocity}, &u)
	post_draw(g, cmd, .Motion_Tiles, {t.tiles}, tiles, #partial Post_Bindings{.Motion = t.rows}, &u)
	post_draw(g, cmd, .Motion_Around, {t.around}, tiles, #partial Post_Bindings{.Motion = t.tiles}, &u)
	post_draw(g, cmd, .Motion_Gather, {out}, {width, height}, #partial Post_Bindings{.Light = color, .Velocity = velocity, .Nearness = nearness, .Motion = t.around}, &u)
	return nil
}

// A job's bloom levels for a `width x height` frame, remade when it changes,
// and the Gaussian's first axis at the last level's size; a chain of fewer levels
// leaves the rest for one of more.
@(private)
ensure_bloom :: proc(g: ^GPU, job: ^Job, width, height: u32, levels: int) -> bool {
	if job.bloom_size != {width, height} {
		release_bloom(g, job)
		job.bloom_size = {width, height}
	}
	make_level :: proc(g: ^GPU, width, height: u32, k: int) -> ^sdl.GPUTexture {
		w, h := o3d.bloom_level_size(int(width), int(height), k)
		return sdl.CreateGPUTexture(g.device, {type = .D2, format = LIGHT_FORMAT, usage = {.SAMPLER, .COLOR_TARGET}, width = u32(w), height = u32(h), layer_count_or_depth = 1, num_levels = 1, sample_count = ._1})
	}
	for k in 0 ..< levels {
		if job.bloom_levels[k] == nil {
			job.bloom_levels[k] = make_level(g, width, height, k)
		}
		if job.bloom_levels[k] == nil {
			return false
		}
	}
	if job.bloom_across == nil || job.bloom_across_level != levels - 1 {
		retire_texture(g.device, g.retired, job.bloom_across)
		job.bloom_across = make_level(g, width, height, levels - 1)
		job.bloom_across_level = levels - 1
	}
	return job.bloom_across != nil
}

@(private)
release_bloom :: proc(g: ^GPU, job: ^Job) {
	for t in job.bloom_levels {
		retire_texture(g.device, g.retired, t)
	}
	retire_texture(g.device, g.retired, job.bloom_across)
	job.bloom_levels, job.bloom_across, job.bloom_size = {}, nil, {}
}

// The chain over the frame's light; the level the output adds.
@(private)
record_bloom :: proc(g: ^GPU, job: ^Job, cmd: ^sdl.GPUCommandBuffer, light: ^sdl.GPUTexture, width, height: u32) -> ^sdl.GPUTexture {
	c := job.bloom
	u := job.post
	u.frame.xy = {f32(width), f32(height)}
	u.bloom = {c.threshold, c.sigma, 0, 0}
	size :: proc(width, height: u32, k: int) -> [2]u32 {
		w, h := o3d.bloom_level_size(int(width), int(height), k)
		return {u32(w), u32(h)}
	}
	post_draw(g, cmd, .Bloom_Bright, {job.bloom_levels[0]}, size(width, height, 0), #partial Post_Bindings{.Light = light}, &u)
	for k in 1 ..< c.levels {
		to := size(width, height, k)
		u.bloom_size = {0, 0, f32(to.x), f32(to.y)}
		post_draw(g, cmd, .Bloom_Down, {job.bloom_levels[k]}, to, #partial Post_Bindings{.Bloom = job.bloom_levels[k - 1]}, &u)
	}
	last := job.bloom_levels[c.levels - 1]
	to := size(width, height, c.levels - 1)
	u.bloom_size = {1, 0, f32(to.x), f32(to.y)}
	post_draw(g, cmd, .Bloom_Blur, {job.bloom_across}, to, #partial Post_Bindings{.Bloom = last}, &u)
	u.bloom_size = {0, 1, f32(to.x), f32(to.y)}
	post_draw(g, cmd, .Bloom_Blur, {last}, to, #partial Post_Bindings{.Bloom = job.bloom_across}, &u)
	return last
}

// A post pass (o3d.POST_PASSES) into `targets` at `size`: its samplers
// bound by what they name (bloom read linearly, the rest nearest; nothing
// bound reads empty), its block from `block`.
@(private)
post_draw :: proc(g: ^GPU, cmd: ^sdl.GPUCommandBuffer, pass: o3d.Pass, targets: []^sdl.GPUTexture, size: [2]u32, maps: Post_Bindings, block: ^o3d.Post_Block) {
	colors: [2]sdl.GPUColorTargetInfo
	for t, i in targets {
		colors[i] = {texture = t, load_op = .DONT_CARE, store_op = .STORE}
	}
	rp := sdl.BeginGPURenderPass(cmd, &colors[0], u32(len(targets)), nil)
	defer sdl.EndGPURenderPass(rp)
	sdl.SetGPUViewport(rp, {x = 0, y = 0, w = f32(size.x), h = f32(size.y), min_depth = 0, max_depth = 1})
	sdl.SetGPUScissor(rp, {x = 0, y = 0, w = i32(size.x), h = i32(size.y)})
	sdl.BindGPUGraphicsPipeline(rp, g.post_pipelines[pass])
	x := &g.programs.passes[pass]
	if n := x.pipeline.fragment.num_samplers; n > 0 {
		bindings: [oksl.SAMPLER_COUNT]sdl.GPUTextureSamplerBinding
		for k, c in x.maps[:n] {
			t := maps[k]
			if t == nil {
				t = g.dummy_texture if k == .Light || k == .Half || k == .Bloom else g.dummy_shadow
			}
			bindings[c] = {texture = t, sampler = g.frame_sampler if k == .Bloom else g.shadow_sampler}
		}
		sdl.BindGPUFragmentSamplers(rp, 0, &bindings[0], u32(n))
	}
	b: [size_of(o3d.Post_Block)]u8 = ---
	o3d.block_fill(&x.block, {block, nil}, b[:])
	sdl.PushGPUFragmentUniformData(cmd, 0, &b[0], u32(x.block.size))
	sdl.DrawGPUPrimitives(rp, 3, 1, 0, 0)
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

// A job's frame depth map (R32F, sampled by the host) and its depth buffer
// at the frame's size, remade when it changes.
@(private)
ensure_depth :: proc(g: ^GPU, job: ^Job, width, height: u32) -> bool {
	if job.depth_size == {width, height} && job.depth_map != nil {
		return true
	}
	release_depth(g, job)
	job.depth_map = sdl.CreateGPUTexture(g.device, {type = .D2, format = SHADOW_FORMAT, usage = {.SAMPLER, .COLOR_TARGET}, width = width, height = height, layer_count_or_depth = 1, num_levels = 1})
	job.depth_buffer = sdl.CreateGPUTexture(g.device, {type = .D2, format = g.depth_format, usage = {.DEPTH_STENCIL_TARGET}, width = width, height = height, layer_count_or_depth = 1, num_levels = 1})
	job.depth_size = {width, height}
	return job.depth_map != nil && job.depth_buffer != nil
}

@(private)
release_depth :: proc(g: ^GPU, job: ^Job) {
	retire_texture(g.device, g.retired, job.depth_map)
	retire_texture(g.device, g.retired, job.depth_buffer)
	job.depth_map, job.depth_buffer, job.depth_size = nil, nil, {}
}

// A job's snapshot of its frame before the transmissive draws, with a
// full mip chain (o3d.texture_grow's), at the frame's size.
@(private)
ensure_snapshot :: proc(g: ^GPU, job: ^Job, width, height: u32) -> bool {
	if job.snapshot_size == {width, height} && job.snapshot_tex != nil {
		return true
	}
	release_snapshot(g, job)
	levels: u32 = 1
	for w, h := width, height; w > 1 || h > 1; w, h = max(w >> 1, 1), max(h >> 1, 1) {
		levels += 1
	}
	job.snapshot_tex = sdl.CreateGPUTexture(g.device, {type = .D2_ARRAY, format = LIGHT_FORMAT, usage = {.SAMPLER, .COLOR_TARGET}, width = width, height = height, layer_count_or_depth = 1, num_levels = levels})
	job.snapshot_size = {width, height}
	return job.snapshot_tex != nil
}

@(private)
release_snapshot :: proc(g: ^GPU, job: ^Job) {
	retire_texture(g.device, g.retired, job.snapshot_tex)
	job.snapshot_tex, job.snapshot_size = nil, {}
}

// A job's reflection: the mirror's light and runs at its size with `levels`
// mips (o3d.Mirror_View's), two layers; and its pass's runs target
// (multisampled as the light, resolved into `runs_resolved`).
@(private)
ensure_reflection :: proc(g: ^GPU, job: ^Job, width, height, levels: u32) -> bool {
	if job.reflection_size == {width, height, levels} && job.reflection_tex != nil {
		return true
	}
	release_reflection(g, job)
	single := g.samples == ._1
	job.reflection_tex = sdl.CreateGPUTexture(g.device, {type = .D2_ARRAY, format = LIGHT_FORMAT, usage = {.SAMPLER, .COLOR_TARGET}, width = width, height = height, layer_count_or_depth = 2, num_levels = levels})
	job.runs_target = sdl.CreateGPUTexture(g.device, {type = .D2, format = RUNS_FORMAT, usage = {.COLOR_TARGET} if !single else {.COLOR_TARGET, .SAMPLER}, width = width, height = height, layer_count_or_depth = 1, num_levels = 1, sample_count = g.samples})
	if !single {
		job.runs_resolved = sdl.CreateGPUTexture(g.device, {type = .D2, format = RUNS_FORMAT, usage = {.COLOR_TARGET, .SAMPLER}, width = width, height = height, layer_count_or_depth = 1, num_levels = 1})
	}
	job.reflection_size = {width, height, levels}
	return job.reflection_tex != nil && job.runs_target != nil && (single || job.runs_resolved != nil)
}

// A job's depth of field tables at a frame's size: its tiles' largest blur
// and the largest near them (R32F), the half-size copy's light and keys.
@(private)
ensure_focus :: proc(g: ^GPU, job: ^Job, width, height: u32) -> bool {
	if job.focus_size == {width, height} && job.focus_tables[0] != nil {
		return true
	}
	release_focus(g, job)
	tiles := [2]u32{(width + o3d.FOCUS_TILE - 1) / o3d.FOCUS_TILE, (height + o3d.FOCUS_TILE - 1) / o3d.FOCUS_TILE}
	half := [2]u32{(width + 1) / 2, (height + 1) / 2}
	sizes := [5][2]u32{{tiles.x, height}, tiles, tiles, half, half}
	formats := [5]sdl.GPUTextureFormat{SHADOW_FORMAT, SHADOW_FORMAT, SHADOW_FORMAT, LIGHT_FORMAT, SHADOW_FORMAT}
	for &t, i in job.focus_tables {
		t = sdl.CreateGPUTexture(g.device, {type = .D2, format = formats[i], usage = {.SAMPLER, .COLOR_TARGET}, width = sizes[i].x, height = sizes[i].y, layer_count_or_depth = 1, num_levels = 1})
		if t == nil {
			return false
		}
	}
	job.focus_size = {width, height}
	return true
}

@(private)
release_focus :: proc(g: ^GPU, job: ^Job) {
	for &t in job.focus_tables {
		retire_texture(g.device, g.retired, t)
		t = nil
	}
	job.focus_size = {}
}

// A job's depth pass's velocity and nearness at the frame's size, and with
// motion blur (`blurred`) its blurred light at the targets' capacity (read
// as the light is).
@(private)
ensure_motion :: proc(g: ^GPU, job: ^Job, width, height: u32, blurred: bool) -> bool {
	if job.velocity_size != {width, height} || job.velocity_map == nil {
		for t in ([]^sdl.GPUTexture{job.velocity_map, job.nearness_map}) {
			retire_texture(g.device, g.retired, t)
		}
		job.velocity_map = sdl.CreateGPUTexture(g.device, {type = .D2, format = VELOCITY_FORMAT, usage = {.SAMPLER, .COLOR_TARGET}, width = width, height = height, layer_count_or_depth = 1, num_levels = 1})
		job.nearness_map = sdl.CreateGPUTexture(g.device, {type = .D2, format = NEARNESS_FORMAT, usage = {.SAMPLER, .COLOR_TARGET}, width = width, height = height, layer_count_or_depth = 1, num_levels = 1})
		job.velocity_size = {width, height}
	}
	if blurred && (job.blurred_size != g.capacity || job.blurred == nil) {
		retire_texture(g.device, g.retired, job.blurred)
		job.blurred = sdl.CreateGPUTexture(g.device, {type = .D2, format = LIGHT_FORMAT, usage = {.SAMPLER, .COLOR_TARGET}, width = g.capacity.x, height = g.capacity.y, layer_count_or_depth = 1, num_levels = 1})
		job.blurred_size = g.capacity
	}
	return job.velocity_map != nil && job.nearness_map != nil && (!blurred || job.blurred != nil)
}

@(private)
release_motion :: proc(g: ^GPU, job: ^Job) {
	for t in ([]^sdl.GPUTexture{job.velocity_map, job.nearness_map, job.blurred}) {
		retire_texture(g.device, g.retired, t)
	}
	job.velocity_map, job.nearness_map, job.blurred, job.velocity_size, job.blurred_size = nil, nil, nil, {}, {}
	motion_tables_destroy(g, &job.motion_tables)
}

@(private)
release_reflection :: proc(g: ^GPU, job: ^Job) {
	for t in ([]^sdl.GPUTexture{job.reflection_tex, job.runs_target, job.runs_resolved}) {
		retire_texture(g.device, g.retired, t)
	}
	job.reflection_tex, job.runs_target, job.runs_resolved, job.reflection_size = nil, nil, nil, {}
}

@(private)
release_shadow_slot :: proc(g: ^GPU, job: ^Job, idx: int) {
	retire_texture(g.device, g.retired, job.shadow_maps[idx])
	retire_texture(g.device, g.retired, job.shadow_depths[idx])
	job.shadow_maps[idx], job.shadow_depths[idx], job.shadow_sizes[idx] = nil, nil, 0
}

@(private = "file")
release_targets :: proc(g: ^GPU) {
	for t in ([]^sdl.GPUTexture{g.color, g.depth, g.resolved, g.output}) {
		retire_texture(g.device, g.retired, t)
	}
	g.color, g.depth, g.resolved, g.output, g.capacity = nil, nil, nil, nil, {}
}

@(private)
ensure_targets :: proc(g: ^GPU, w, h: u32) -> bool {
	if w <= g.capacity.x && h <= g.capacity.y && g.depth != nil {
		return true
	}
	size := [2]u32{max(w, g.capacity.x), max(h, g.capacity.y)}
	release_targets(g)
	single := g.samples == ._1
	// The light read after a pass is the resolve target, or the colour
	// target itself without multisampling.
	g.color = sdl.CreateGPUTexture(g.device, {type = .D2, format = LIGHT_FORMAT, usage = {.COLOR_TARGET} if !single else {.COLOR_TARGET, .SAMPLER}, width = size.x, height = size.y, layer_count_or_depth = 1, num_levels = 1, sample_count = g.samples})
	if !single {
		g.resolved = sdl.CreateGPUTexture(g.device, {type = .D2, format = LIGHT_FORMAT, usage = {.COLOR_TARGET, .SAMPLER}, width = size.x, height = size.y, layer_count_or_depth = 1, num_levels = 1, sample_count = ._1})
	}
	g.depth = sdl.CreateGPUTexture(g.device, {type = .D2, format = g.depth_format, usage = {.DEPTH_STENCIL_TARGET}, width = size.x, height = size.y, layer_count_or_depth = 1, num_levels = 1, sample_count = g.samples})
	g.output = sdl.CreateGPUTexture(g.device, {type = .D2, format = COLOR_FORMAT, usage = {.COLOR_TARGET, .SAMPLER}, width = size.x, height = size.y, layer_count_or_depth = 1, num_levels = 1, sample_count = ._1})
	g.capacity = size
	return g.color != nil && g.depth != nil && g.output != nil && (single || g.resolved != nil)
}
