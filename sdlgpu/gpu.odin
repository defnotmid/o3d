// SDL_GPU renderer for o3d: Vulkan on Linux and Windows, Metal on macOS.
// It draws the same scene as the software renderer, with the same shading
// (shaders/o3d.slang, shaders/background.slang), into light (half floats,
// 4x MSAA, blended there), resolved, then tone mapped and encoded into the
// caller's texture (shaders/post.slang).
//
// `init` creates a headless device, or borrows the host's (the one its
// window or 2D canvas uses, so the result composites without a copy).
// Geometry lives in one vertex and one index buffer, mirroring
// `Geometries`' shared arrays: what is new uploads at the start of a frame.
package o3d_sdlgpu

import "core:math"
import "core:slice"
import "core:sync"
import sdl "vendor:sdl3"

import o3d ".."
import "deps:oksl"
import curves "deps:ocurves"

VERT_SPV :: #load("../shaders/o3d.vert.spv")
SKINNED_VERT_SPV :: #load("../shaders/o3d.skinned_vert.spv")
FRAG_SPV :: #load("../shaders/o3d.frag.spv")
VERT_MSL :: #load("../shaders/o3d.vert.msl")
SKINNED_VERT_MSL :: #load("../shaders/o3d.skinned_vert.msl")
FRAG_MSL :: #load("../shaders/o3d.frag.msl")
FRAG_MIRROR_SPV :: #load("../shaders/o3d.frag_mirror.spv")
FRAG_MIRROR_MSL :: #load("../shaders/o3d.frag_mirror.msl")
SHADOW_SPV :: #load("../shaders/o3d.shadow.spv")
SHADOW_MSL :: #load("../shaders/o3d.shadow.msl")
CURVES_VERT_SPV :: #load("../shaders/o3d.curves_vert.spv")
CURVES_VERT_MSL :: #load("../shaders/o3d.curves_vert.msl")
// The curve shaders read the words as the ocurves they were compiled from
// lays them out; against another layout they would draw garbage.
#assert(curves.FORMAT == SHADERS_CURVES_FORMAT, "o3d's curve shaders read another curves.FORMAT: use the ocurves this o3d's tag names, or rerun shaders/build.sh against this one")
CURVES_FRAG_SPV :: #load("../shaders/o3d.curves_frag.spv")
CURVES_FRAG_MSL :: #load("../shaders/o3d.curves_frag.msl")
CURVES_FRAG_MIRROR_SPV :: #load("../shaders/o3d.curves_frag_mirror.spv")
CURVES_FRAG_MIRROR_MSL :: #load("../shaders/o3d.curves_frag_mirror.msl")
SPRITE_FRAG_SPV :: #load("../shaders/o3d.sprite_frag.spv")
SPRITE_FRAG_MSL :: #load("../shaders/o3d.sprite_frag.msl")
SHADOW_CURVES_SPV :: #load("../shaders/o3d.shadow_curves.spv")
SHADOW_CURVES_MSL :: #load("../shaders/o3d.shadow_curves.msl")
BG_VERT_SPV :: #load("../shaders/background.vert.spv")
BG_FRAG_SPV :: #load("../shaders/background.frag.spv")
BG_VERT_MSL :: #load("../shaders/background.vert.msl")
BG_FRAG_MSL :: #load("../shaders/background.frag.msl")
DEPTH_VERT_SPV :: #load("../shaders/depth.vert.spv")
DEPTH_FRAG_SPV :: #load("../shaders/depth.frag.spv")
DEPTH_VERT_MSL :: #load("../shaders/depth.vert.msl")
DEPTH_FRAG_MSL :: #load("../shaders/depth.frag.msl")
POST_VERT_SPV :: #load("../shaders/post.vert.spv")
OUTPUT_FRAG_SPV :: #load("../shaders/post.output.spv")
POST_VERT_MSL :: #load("../shaders/post.vert.msl")
OUTPUT_FRAG_MSL :: #load("../shaders/post.output.msl")
// Depth of field: the tiles' largest blur, near them, the half-size copy,
// the gather.
@(private = "file", rodata)
FOCUS_FRAGS_SPV := [Focus_Pass][]u8 {
	.Tiles  = #load("../shaders/post.focus_tiles.spv"),
	.Near   = #load("../shaders/post.focus_near.spv"),
	.Half   = #load("../shaders/post.focus_half.spv"),
	.Gather = #load("../shaders/post.focus.spv"),
}
@(private = "file", rodata)
FOCUS_FRAGS_MSL := [Focus_Pass][]u8 {
	.Tiles  = #load("../shaders/post.focus_tiles.msl"),
	.Near   = #load("../shaders/post.focus_near.msl"),
	.Half   = #load("../shaders/post.focus_half.msl"),
	.Gather = #load("../shaders/post.focus.msl"),
}
@(private = "file", rodata)
FOCUS_ENTRIES := [Focus_Pass]cstring {
	.Tiles  = "fs_focus_tiles",
	.Near   = "fs_focus_near",
	.Half   = "fs_focus_half",
	.Gather = "fs_focus",
}

Focus_Pass :: enum u8 {
	Tiles,
	Near,
	Half,
	Gather,
}

// Motion blur: the frame's depth pass with velocity and nearness
// (o3d.slang's vs_velocity*, fs_velocity*; fs_velocity_far under the
// draws), then the filter (post.slang's fs_motion*: the tiles' longest
// velocity, about them, the gather).
Velocity_Shader :: enum u8 {
	Vert,
	Skinned_Vert,
	Curves_Vert,
	Frag,
	Curves_Frag,
	Far_Frag,
}

Motion_Pass :: enum u8 {
	Rows,
	Tiles,
	Around,
	Gather,
}

@(private = "file", rodata)
VELOCITY_SPV := [Velocity_Shader][]u8 {
	.Vert         = #load("../shaders/o3d.velocity_vert.spv"),
	.Skinned_Vert = #load("../shaders/o3d.velocity_skinned_vert.spv"),
	.Curves_Vert  = #load("../shaders/o3d.velocity_curves_vert.spv"),
	.Frag         = #load("../shaders/o3d.velocity.spv"),
	.Curves_Frag  = #load("../shaders/o3d.velocity_curves.spv"),
	.Far_Frag     = #load("../shaders/o3d.velocity_far.spv"),
}
@(private = "file", rodata)
VELOCITY_MSL := [Velocity_Shader][]u8 {
	.Vert         = #load("../shaders/o3d.velocity_vert.msl"),
	.Skinned_Vert = #load("../shaders/o3d.velocity_skinned_vert.msl"),
	.Curves_Vert  = #load("../shaders/o3d.velocity_curves_vert.msl"),
	.Frag         = #load("../shaders/o3d.velocity.msl"),
	.Curves_Frag  = #load("../shaders/o3d.velocity_curves.msl"),
	.Far_Frag     = #load("../shaders/o3d.velocity_far.msl"),
}
@(private = "file", rodata)
VELOCITY_ENTRIES := [Velocity_Shader]cstring {
	.Vert         = "vs_velocity",
	.Skinned_Vert = "vs_velocity_skinned",
	.Curves_Vert  = "vs_velocity_curves",
	.Frag         = "fs_velocity",
	.Curves_Frag  = "fs_velocity_curves",
	.Far_Frag     = "fs_velocity_far",
}
// Each one's stage, samplers, storage buffers and uniform buffers.
@(private = "file", rodata)
VELOCITY_LAYOUTS := [Velocity_Shader]struct {
	stage:                       sdl.GPUShaderStage,
	samplers, storage, uniforms: u32,
} {
	.Vert         = {.VERTEX, 0, 1, 2},
	.Skinned_Vert = {.VERTEX, 0, 1, 4},
	.Curves_Vert  = {.VERTEX, 0, 0, 2},
	.Frag         = {.FRAGMENT, 2, 0, 2},
	.Curves_Frag  = {.FRAGMENT, 2, 1, 2},
	.Far_Frag     = {.FRAGMENT, 0, 0, 2},
}
@(private = "file", rodata)
MOTION_FRAGS_SPV := [Motion_Pass][]u8 {
	.Rows   = #load("../shaders/post.motion_rows.spv"),
	.Tiles  = #load("../shaders/post.motion_tiles.spv"),
	.Around = #load("../shaders/post.motion_around.spv"),
	.Gather = #load("../shaders/post.motion.spv"),
}
@(private = "file", rodata)
MOTION_FRAGS_MSL := [Motion_Pass][]u8 {
	.Rows   = #load("../shaders/post.motion_rows.msl"),
	.Tiles  = #load("../shaders/post.motion_tiles.msl"),
	.Around = #load("../shaders/post.motion_around.msl"),
	.Gather = #load("../shaders/post.motion.msl"),
}
@(private = "file", rodata)
MOTION_ENTRIES := [Motion_Pass]cstring {
	.Rows   = "fs_motion_rows",
	.Tiles  = "fs_motion_tiles",
	.Around = "fs_motion_around",
	.Gather = "fs_motion",
}

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
}

// What a draw is drawn by: the standard material's shaders, skinned, an
// exact shape's (curves) or a particle's.
Draw_Kind :: enum u8 {
	Standard,
	Skinned,
	Curves,
	Sprite,
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

GPU :: struct {
	device:                   ^sdl.GPUDevice,
	owns_device:              bool,
	vert, skinned_vert, frag: ^sdl.GPUShader,
	bg_vert, bg_frag:         ^sdl.GPUShader,
	samples:                  sdl.GPUSampleCount,
	depth_format:             sdl.GPUTextureFormat,
	// Exact shapes and glyph runs (`.Curves`): their vertex shader (each
	// vertex's quad) and fragment shader (coverage from the quad's curve set
	// in `curve_words`: by area blended, by sample opaque); particle quads'
	// fragment shader (the curves' vertex shader gives its quad index).
	curves_vert:              ^sdl.GPUShader,
	curves_frag:              ^sdl.GPUShader,
	sprite_frag:              ^sdl.GPUShader,
	// The mirror's pass's fragment shaders (`frag`'s and `curves_frag`'s,
	// with their runs).
	frag_mirror:              ^sdl.GPUShader,
	curves_frag_mirror:       ^sdl.GPUShader,
	// The draws' pipelines into the frame's light, by kind, the material's
	// side and its blend (`o3d.blend_of`); the mirror's (front faces
	// clockwise), made when a frame first needs them.
	pipelines:                [Draw_Kind][o3d.Side][o3d.Blend]^sdl.GPUGraphicsPipeline,
	mirror_pipelines:         [Draw_Kind][o3d.Side][o3d.Blend]^sdl.GPUGraphicsPipeline,
	bg_pipeline:              ^sdl.GPUGraphicsPipeline,
	// The frame's output: its light tone mapped and sRGB-encoded; or
	// blurred by its depth of field (shaders/post.slang), its last pass
	// the output.
	post_vert, output_frag:   ^sdl.GPUShader,
	output_pipeline:          ^sdl.GPUGraphicsPipeline,
	focus_frags:              [Focus_Pass]^sdl.GPUShader,
	focus_pipelines:          [Focus_Pass]^sdl.GPUGraphicsPipeline,
	// Motion blur: the depth pass with velocity, by the draw's kind
	// (standard, skinned, curves) and its material's side, and under its
	// draws; the filter's passes.
	velocity_shaders:         [Velocity_Shader]^sdl.GPUShader,
	velocity_pipelines:       [3][o3d.Side]^sdl.GPUGraphicsPipeline,
	velocity_far_pipeline:    ^sdl.GPUGraphicsPipeline,
	motion_frags:             [Motion_Pass]^sdl.GPUShader,
	motion_pipelines:         [Motion_Pass]^sdl.GPUGraphicsPipeline,
	// The frame's depth as distance into the host's image.
	depth_vert, depth_frag:   ^sdl.GPUShader,
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
	curve_table:                        u32,
	// Particles: fade distance before the frame's depth.
	softness:                           f32,
	uv_transforms:                      [2 * len(o3d.Texture_Slot)][4]f32,
	// Triplanar: repeats per unit, the weights' power; the projection
	// frame's world-to-local rows and normals-to-local rows.
	triplanar:                          [4]f32,
	to_local, normal_to_local:          [3][4]f32,
	// o3d.Shading_Layers: f0, clear coat, its roughness, sheen roughness;
	// sheen colour, w the environment's intensity on it
	// (o3d.Shading_Material.env_intensity); the normal map's x, y scale and
	// the coat normal map's;
	// transmission, thickness (the model's units), ior; the model's axis
	// scales; absorption per unit; x: o3d.Shading_Material.layer_slots.
	layers, sheen, normal_scales:       [4]f32,
	transmission, scale, attenuation:   [4]f32,
	layer_info:                         [4]u32,
}
#assert(size_of(Material_Uniforms) == 736)

// The post passes' uniforms (shaders/post.slang's Post_FS): the tone mapping,
// exposure, the light texture's size; the focus, blur and its most
// (pixels), the tiles a blur reaches; the camera's near, far, orthographic;
// the frame's size, its tiles across and down.
Post_Uniforms :: struct {
	tone, lens, camera, frame: [4]f32,
	motion, motion_size:       [4]f32,
}

// The post passes' maps, in their slots (post.slang's).
Post_Maps :: enum u8 {
	Light,
	Depth,
	Tiles,
	Half,
	Half_Keys,
	Velocity,
	Nearness,
	Motion,
}

// The depth pass's draw at the shutter's opening (o3d.slang's Opened_VS),
// and its frame (Velocity_FS).
Opened_Uniforms :: struct {
	mvp:           o3d.Mat4,
	morph_bases:   [2][4]u32,
	morph_weights: [2][4]f32,
	morph_info:    [4]u32,
}

Velocity_Uniforms :: struct {
	frame:      [4]f32, // width, height, the camera's near and far
	flags:      [4]u32, // orthographic
	opened_far: o3d.Mat4,
}

Post_Bindings :: [Post_Maps]^sdl.GPUTexture

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
	// Where the light comes from (o3d.Shadow.from).
	shadow_from:   [o3d.MAX_SHADOWS]o3d.Vec4,
	// Environment intensity (0: no environment), max radiance mip level, 0, fog density.
	env_params:    [4]f32,
	// x: the environment's maps are linear (HDR), not sRGB-encoded; y: the camera is orthographic.
	env_flags:     [4]u32,
	// The frame's width and height, the camera's near and far.
	frame_params:  [4]f32,
	fog_color:     [4]f32,
	fog_params:    [4]f32,
	// The camera's view-projection (transmission's refraction).
	view_proj:     o3d.Mat4,
	// The mirror's pass: its plane (behind it is left out; zero in the
	// other passes). x: the reflection's last level.
	clip_plane:    [4]f32,
	reflect_params: [4]f32,
	lights:        [o3d.MAX_LIGHTS]Light_Uniforms,
}

// SDL_Init and making or destroying a device are not safe from several
// threads at once (tests run in parallel; a host may init from workers):
// they take turns.
@(private = "file")
device_mutex: sync.Mutex

init :: proc(g: ^GPU, device: ^sdl.GPUDevice = nil) -> (err: o3d.Error) {
	g^ = {device = device}
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
	// slangc names the SPIR-V entry points "main" and keeps the names for Metal.
	formats := sdl.GetGPUShaderFormats(g.device)
	vert_code, skinned_vert_code, frag_code, shadow_code, bg_vert_code, bg_frag_code, curves_vert_code, curves_code, shadow_curves_code, sprite_code, depth_vert_code, depth_frag_code, post_vert_code, output_code, mirror_code, curves_mirror_code: []u8
	format: sdl.GPUShaderFormat
	vs_entry, skinned_vs_entry, fs_entry, shadow_entry, curves_vs_entry, curves_entry, shadow_curves_entry, sprite_entry: cstring
	output_entry, mirror_entry, curves_mirror_entry: cstring = "main", "main", "main"
	switch {
	case .SPIRV in formats:
		vert_code, skinned_vert_code, frag_code, shadow_code = VERT_SPV, SKINNED_VERT_SPV, FRAG_SPV, SHADOW_SPV
		bg_vert_code, bg_frag_code = BG_VERT_SPV, BG_FRAG_SPV
		depth_vert_code, depth_frag_code = DEPTH_VERT_SPV, DEPTH_FRAG_SPV
		post_vert_code, output_code = POST_VERT_SPV, OUTPUT_FRAG_SPV
		curves_vert_code, curves_code, shadow_curves_code, sprite_code = CURVES_VERT_SPV, CURVES_FRAG_SPV, SHADOW_CURVES_SPV, SPRITE_FRAG_SPV
		mirror_code, curves_mirror_code = FRAG_MIRROR_SPV, CURVES_FRAG_MIRROR_SPV
		format = {.SPIRV}
		vs_entry, skinned_vs_entry, fs_entry, shadow_entry = "main", "main", "main", "main"
		curves_vs_entry, curves_entry, shadow_curves_entry, sprite_entry = "main", "main", "main", "main"
	case .MSL in formats:
		vert_code, skinned_vert_code, frag_code, shadow_code = VERT_MSL, SKINNED_VERT_MSL, FRAG_MSL, SHADOW_MSL
		bg_vert_code, bg_frag_code = BG_VERT_MSL, BG_FRAG_MSL
		depth_vert_code, depth_frag_code = DEPTH_VERT_MSL, DEPTH_FRAG_MSL
		post_vert_code, output_code = POST_VERT_MSL, OUTPUT_FRAG_MSL
		curves_vert_code, curves_code, shadow_curves_code, sprite_code = CURVES_VERT_MSL, CURVES_FRAG_MSL, SHADOW_CURVES_MSL, SPRITE_FRAG_MSL
		mirror_code, curves_mirror_code = FRAG_MIRROR_MSL, CURVES_FRAG_MIRROR_MSL
		mirror_entry, curves_mirror_entry = "fs_mirror", "fs_curves_mirror"
		format = {.MSL}
		vs_entry, skinned_vs_entry, fs_entry, shadow_entry = "vs_main", "vs_skinned", "fs_main", "fs_shadow"
		curves_vs_entry, curves_entry, shadow_curves_entry, sprite_entry = "vs_curves", "fs_curves", "fs_shadow_curves", "fs_sprite"
		output_entry = "fs_output"
	case:
		return .Device
	}
	g.vert = sdl.CreateGPUShader(g.device, {code_size = len(vert_code), code = raw_data(vert_code), entrypoint = vs_entry, format = format, stage = .VERTEX, num_storage_buffers = 1, num_uniform_buffers = 1})
	g.skinned_vert = sdl.CreateGPUShader(g.device, {code_size = len(skinned_vert_code), code = raw_data(skinned_vert_code), entrypoint = skinned_vs_entry, format = format, stage = .VERTEX, num_storage_buffers = 1, num_uniform_buffers = 2})
	g.frag = sdl.CreateGPUShader(g.device, {code_size = len(frag_code), code = raw_data(frag_code), entrypoint = fs_entry, format = format, stage = .FRAGMENT, num_samplers = 16, num_uniform_buffers = 2})
	g.shadow_frag = sdl.CreateGPUShader(g.device, {code_size = len(shadow_code), code = raw_data(shadow_code), entrypoint = shadow_entry, format = format, stage = .FRAGMENT, num_samplers = 2, num_uniform_buffers = 1})
	g.curves_vert = sdl.CreateGPUShader(g.device, {code_size = len(curves_vert_code), code = raw_data(curves_vert_code), entrypoint = curves_vs_entry, format = format, stage = .VERTEX, num_uniform_buffers = 1})
	g.curves_frag = sdl.CreateGPUShader(g.device, {code_size = len(curves_code), code = raw_data(curves_code), entrypoint = curves_entry, format = format, stage = .FRAGMENT, num_samplers = 16, num_storage_buffers = 1, num_uniform_buffers = 2})
	g.frag_mirror = sdl.CreateGPUShader(g.device, {code_size = len(mirror_code), code = raw_data(mirror_code), entrypoint = mirror_entry, format = format, stage = .FRAGMENT, num_samplers = 16, num_uniform_buffers = 2})
	g.curves_frag_mirror = sdl.CreateGPUShader(g.device, {code_size = len(curves_mirror_code), code = raw_data(curves_mirror_code), entrypoint = curves_mirror_entry, format = format, stage = .FRAGMENT, num_samplers = 16, num_storage_buffers = 1, num_uniform_buffers = 2})
	g.sprite_frag = sdl.CreateGPUShader(g.device, {code_size = len(sprite_code), code = raw_data(sprite_code), entrypoint = sprite_entry, format = format, stage = .FRAGMENT, num_samplers = 16, num_storage_buffers = 1, num_uniform_buffers = 2})
	g.shadow_curves_frag = sdl.CreateGPUShader(g.device, {code_size = len(shadow_curves_code), code = raw_data(shadow_curves_code), entrypoint = shadow_curves_entry, format = format, stage = .FRAGMENT, num_samplers = 2, num_storage_buffers = 1, num_uniform_buffers = 1})
	g.bg_vert = sdl.CreateGPUShader(g.device, {code_size = len(bg_vert_code), code = raw_data(bg_vert_code), entrypoint = vs_entry, format = format, stage = .VERTEX, num_uniform_buffers = 1})
	g.bg_frag = sdl.CreateGPUShader(g.device, {code_size = len(bg_frag_code), code = raw_data(bg_frag_code), entrypoint = fs_entry, format = format, stage = .FRAGMENT, num_samplers = 1, num_uniform_buffers = 1})
	g.depth_vert = sdl.CreateGPUShader(g.device, {code_size = len(depth_vert_code), code = raw_data(depth_vert_code), entrypoint = vs_entry, format = format, stage = .VERTEX})
	g.depth_frag = sdl.CreateGPUShader(g.device, {code_size = len(depth_frag_code), code = raw_data(depth_frag_code), entrypoint = fs_entry, format = format, stage = .FRAGMENT, num_samplers = 1, num_uniform_buffers = 1})
	g.post_vert = sdl.CreateGPUShader(g.device, {code_size = len(post_vert_code), code = raw_data(post_vert_code), entrypoint = vs_entry, format = format, stage = .VERTEX})
	g.output_frag = sdl.CreateGPUShader(g.device, {code_size = len(output_code), code = raw_data(output_code), entrypoint = output_entry, format = format, stage = .FRAGMENT, num_samplers = 1, num_uniform_buffers = 1})
	for sh in Velocity_Shader {
		code := VELOCITY_SPV[sh] if format == {.SPIRV} else VELOCITY_MSL[sh]
		entry := cstring("main") if format == {.SPIRV} else VELOCITY_ENTRIES[sh]
		l := VELOCITY_LAYOUTS[sh]
		g.velocity_shaders[sh] = sdl.CreateGPUShader(g.device, {code_size = len(code), code = raw_data(code), entrypoint = entry, format = format, stage = l.stage, num_samplers = l.samplers, num_storage_buffers = l.storage, num_uniform_buffers = l.uniforms})
		if g.velocity_shaders[sh] == nil {
			return .Shader
		}
	}
	for pass in Motion_Pass {
		code := MOTION_FRAGS_SPV[pass] if format == {.SPIRV} else MOTION_FRAGS_MSL[pass]
		entry := cstring("main") if format == {.SPIRV} else MOTION_ENTRIES[pass]
		g.motion_frags[pass] = sdl.CreateGPUShader(g.device, {code_size = len(code), code = raw_data(code), entrypoint = entry, format = format, stage = .FRAGMENT, num_samplers = len(Post_Maps), num_uniform_buffers = 1})
		if g.motion_frags[pass] == nil {
			return .Shader
		}
	}
	for pass in Focus_Pass {
		code := FOCUS_FRAGS_SPV[pass] if format == {.SPIRV} else FOCUS_FRAGS_MSL[pass]
		entry := cstring("main") if format == {.SPIRV} else FOCUS_ENTRIES[pass]
		g.focus_frags[pass] = sdl.CreateGPUShader(g.device, {code_size = len(code), code = raw_data(code), entrypoint = entry, format = format, stage = .FRAGMENT, num_samplers = len(Post_Maps), num_uniform_buffers = 1})
		if g.focus_frags[pass] == nil {
			return .Shader
		}
	}
	if g.vert == nil || g.skinned_vert == nil || g.frag == nil || g.frag_mirror == nil || g.curves_frag_mirror == nil || g.shadow_frag == nil || g.curves_vert == nil || g.curves_frag == nil || g.sprite_frag == nil || g.shadow_curves_frag == nil || g.bg_vert == nil || g.bg_frag == nil || g.depth_vert == nil || g.depth_frag == nil || g.post_vert == nil || g.output_frag == nil {
		return .Shader
	}
	g.depth_format = .D32_FLOAT
	if !sdl.GPUTextureSupportsFormat(g.device, g.depth_format, .D2, {.DEPTH_STENCIL_TARGET}) {
		g.depth_format = .D24_UNORM
	}
	g.samples = ._4
	if !sdl.GPUTextureSupportsSampleCount(g.device, LIGHT_FORMAT, ._4) || !sdl.GPUTextureSupportsSampleCount(g.device, RUNS_FORMAT, ._4) || !sdl.GPUTextureSupportsSampleCount(g.device, g.depth_format, ._4) {
		g.samples = ._1
	}
	for kind in Draw_Kind {
		for side in o3d.Side {
			for blend in o3d.Blend {
				g.pipelines[kind][side][blend] = make_draw_pipeline(g, kind, side, blend, .Light)
				if g.pipelines[kind][side][blend] == nil {
					return .Shader
				}
			}
		}
	}
	for side in o3d.Side {
		g.shadow_pipelines[side] = make_shadow_pipeline(g, g.vert, g.shadow_frag, side, false)
		g.shadow_skinned_pipelines[side] = make_shadow_pipeline(g, g.skinned_vert, g.shadow_frag, side, true)
		g.shadow_curve_pipelines[side] = make_shadow_pipeline(g, g.curves_vert, g.shadow_curves_frag, side, false)
		if g.shadow_pipelines[side] == nil || g.shadow_skinned_pipelines[side] == nil || g.shadow_curve_pipelines[side] == nil {
			return .Shader
		}
		vs := &g.velocity_shaders
		g.velocity_pipelines[0][side] = make_velocity_pipeline(g, vs[.Vert], vs[.Frag], side, false)
		g.velocity_pipelines[1][side] = make_velocity_pipeline(g, vs[.Skinned_Vert], vs[.Frag], side, true)
		g.velocity_pipelines[2][side] = make_velocity_pipeline(g, vs[.Curves_Vert], vs[.Curves_Frag], side, false)
		for k in 0 ..< 3 {
			if g.velocity_pipelines[k][side] == nil {
				return .Shader
			}
		}
	}
	g.bg_pipeline = make_background_pipeline(g)
	g.depth_pipeline = make_fullscreen_pipeline(g, g.depth_vert, g.depth_frag, DEPTH_IMAGE_FORMAT)
	g.output_pipeline = make_fullscreen_pipeline(g, g.post_vert, g.output_frag, COLOR_FORMAT)
	g.focus_pipelines[.Tiles] = make_fullscreen_pipeline(g, g.post_vert, g.focus_frags[.Tiles], SHADOW_FORMAT)
	g.focus_pipelines[.Near] = make_fullscreen_pipeline(g, g.post_vert, g.focus_frags[.Near], SHADOW_FORMAT)
	g.focus_pipelines[.Half] = make_fullscreen_pipeline(g, g.post_vert, g.focus_frags[.Half], LIGHT_FORMAT, SHADOW_FORMAT)
	g.focus_pipelines[.Gather] = make_fullscreen_pipeline(g, g.post_vert, g.focus_frags[.Gather], COLOR_FORMAT)
	g.motion_pipelines[.Rows] = make_fullscreen_pipeline(g, g.post_vert, g.motion_frags[.Rows], VELOCITY_FORMAT)
	g.motion_pipelines[.Tiles] = make_fullscreen_pipeline(g, g.post_vert, g.motion_frags[.Tiles], VELOCITY_FORMAT)
	g.motion_pipelines[.Around] = make_fullscreen_pipeline(g, g.post_vert, g.motion_frags[.Around], VELOCITY_FORMAT)
	g.motion_pipelines[.Gather] = make_fullscreen_pipeline(g, g.post_vert, g.motion_frags[.Gather], LIGHT_FORMAT)
	g.velocity_far_pipeline = make_velocity_pipeline(g, g.post_vert, g.velocity_shaders[.Far_Frag], .Double, false, far = true)
	for p in g.motion_pipelines {
		if p == nil {
			return .Shader
		}
	}
	if g.velocity_far_pipeline == nil {
		return .Shader
	}
	if g.bg_pipeline == nil || g.depth_pipeline == nil || g.output_pipeline == nil {
		return .Shader
	}
	for p in g.focus_pipelines {
		if p == nil {
			return .Shader
		}
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
	if g.clear_texture != nil {
		sdl.ReleaseGPUTexture(g.device, g.clear_texture)
	}
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
	for kind in Draw_Kind {
		for side in o3d.Side {
			for blend in o3d.Blend {
				for p in ([2]^sdl.GPUGraphicsPipeline{g.pipelines[kind][side][blend], g.mirror_pipelines[kind][side][blend]}) {
					if p != nil {
						sdl.ReleaseGPUGraphicsPipeline(g.device, p)
					}
				}
			}
		}
	}
	for p in ([]^sdl.GPUGraphicsPipeline{g.bg_pipeline, g.depth_pipeline, g.output_pipeline}) {
		if p != nil {
			sdl.ReleaseGPUGraphicsPipeline(g.device, p)
		}
	}
	for p in g.focus_pipelines {
		if p != nil {
			sdl.ReleaseGPUGraphicsPipeline(g.device, p)
		}
	}
	for sh in g.focus_frags {
		if sh != nil {
			sdl.ReleaseGPUShader(g.device, sh)
		}
	}
	for pipelines in g.velocity_pipelines {
		for p in pipelines {
			if p != nil {
				sdl.ReleaseGPUGraphicsPipeline(g.device, p)
			}
		}
	}
	for p in g.motion_pipelines {
		if p != nil {
			sdl.ReleaseGPUGraphicsPipeline(g.device, p)
		}
	}
	if g.velocity_far_pipeline != nil {
		sdl.ReleaseGPUGraphicsPipeline(g.device, g.velocity_far_pipeline)
	}
	for sh in g.velocity_shaders {
		if sh != nil {
			sdl.ReleaseGPUShader(g.device, sh)
		}
	}
	for sh in g.motion_frags {
		if sh != nil {
			sdl.ReleaseGPUShader(g.device, sh)
		}
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
	for sh in ([]^sdl.GPUShader{g.shadow_frag, g.curves_vert, g.curves_frag, g.frag_mirror, g.curves_frag_mirror, g.sprite_frag, g.shadow_curves_frag, g.depth_vert, g.depth_frag, g.post_vert, g.output_frag}) {
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

// A texture for `record_depth_into` (RGBA16F; to sample or read back).
depth_image_target :: proc(g: ^GPU, width, height: u32) -> ^sdl.GPUTexture {
	return sdl.CreateGPUTexture(g.device, {type = .D2, format = DEPTH_IMAGE_FORMAT, usage = {.SAMPLER, .COLOR_TARGET}, width = width, height = height, layer_count_or_depth = 1, num_levels = 1, sample_count = ._1})
}

// The standard material's vertex shader reads every attribute (locations
// in VS_In's order, as slang numbers them).
@(private = "file", rodata)
STANDARD_ATTRIBUTES := [5]sdl.GPUVertexAttribute {
	{location = 0, buffer_slot = 0, format = .FLOAT3, offset = 0},
	{location = 1, buffer_slot = 0, format = .FLOAT3, offset = 12},
	{location = 2, buffer_slot = 0, format = .FLOAT2, offset = 24},
	{location = 3, buffer_slot = 0, format = .FLOAT2, offset = 32},
	{location = 4, buffer_slot = 0, format = .UBYTE4_NORM, offset = 40},
}

@(private = "file", rodata)
SKINNED_ATTRIBUTES := [7]sdl.GPUVertexAttribute {
	{location = 0, buffer_slot = 0, format = .FLOAT3, offset = 0},
	{location = 1, buffer_slot = 0, format = .FLOAT3, offset = 12},
	{location = 2, buffer_slot = 0, format = .FLOAT2, offset = 24},
	{location = 3, buffer_slot = 0, format = .FLOAT2, offset = 32},
	{location = 4, buffer_slot = 1, format = .USHORT4, offset = 0},
	{location = 5, buffer_slot = 1, format = .FLOAT4, offset = 8},
	{location = 6, buffer_slot = 0, format = .UBYTE4_NORM, offset = 40},
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
				enable_blend = blend != .Opaque,
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
				enable_blend = blend != .Opaque,
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
		depth_stencil_state = {compare_op = .GREATER, enable_depth_test = true, enable_depth_write = blend == .Opaque},
		target_info = {color_target_descriptions = &colors[0], num_color_targets = 2 if mirrored else 1, depth_stencil_format = g.depth_format, has_depth_stencil_target = true},
	}
	return sdl.CreateGPUGraphicsPipeline(g.device, info)
}

// A kind of draw's pipeline (its shaders and vertex streams), into the
// frame (`.Light`) or through the mirror (`.Mirror`).
@(private = "file")
make_draw_pipeline :: proc(g: ^GPU, kind: Draw_Kind, side: o3d.Side, blend: o3d.Blend, targets: Targets) -> ^sdl.GPUGraphicsPipeline {
	mirrored := targets != .Light
	frag := g.frag_mirror if mirrored else g.frag
	switch kind {
	case .Skinned:
		return make_pipeline(g, g.skinned_vert, frag, SKINNED_ATTRIBUTES[:], side, blend, false, true, targets)
	case .Curves:
		return make_pipeline(g, g.curves_vert, g.curves_frag_mirror if mirrored else g.curves_frag, STANDARD_ATTRIBUTES[:], side, blend, false, false, targets)
	case .Sprite:
		return make_pipeline(g, g.curves_vert, g.sprite_frag, STANDARD_ATTRIBUTES[:], side, blend, false, false, .Mirror_No_Runs if mirrored else .Light)
	case .Standard:
	}
	return make_pipeline(g, g.vert, frag, STANDARD_ATTRIBUTES[:], side, blend, false, false, targets)
}

// The fullscreen background pipeline: 3 vertices via SV_VertexID, no vertex
// buffer, no depth test or write (drawn first into the cleared target).
@(private = "file")
make_background_pipeline :: proc(g: ^GPU) -> ^sdl.GPUGraphicsPipeline {
	color := sdl.GPUColorTargetDescription{format = LIGHT_FORMAT}
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
				info.code, info.code_size = ([^]u8)(raw_data(m.spirv)), len(m.spirv) * size_of(u32)
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
		p^ = make_velocity_pipeline(g, c.depth_vert, c.depth_frag, side, false, attrs = attrs[:n])
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

// The shadow pass's pipeline for casters of a material side: depth into the
// R32_FLOAT map (nearest, the greatest, kept by the depth test), the other side drawn, by
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
		depth_stencil_state = {compare_op = .GREATER, enable_depth_test = true, enable_depth_write = true},
		target_info = {color_target_descriptions = &color, num_color_targets = 1, depth_stencil_format = g.depth_format, has_depth_stencil_target = true},
	}
	return sdl.CreateGPUGraphicsPipeline(g.device, info)
}

// The frame's depth pass with motion blur: its depth map, velocity and
// nearness, culled by the material's own side; or what is under its draws
// (`far`: a full-screen triangle, no depth test).
@(private = "file")
make_velocity_pipeline :: proc(g: ^GPU, vert, frag: ^sdl.GPUShader, side: o3d.Side, skinned: bool, far := false, attrs: []sdl.GPUVertexAttribute = nil) -> ^sdl.GPUGraphicsPipeline {
	buffers := [2]sdl.GPUVertexBufferDescription {
		{slot = 0, pitch = size_of(o3d.Vertex), input_rate = .VERTEX},
		{slot = 1, pitch = size_of(o3d.Skin_Vertex), input_rate = .VERTEX},
	}
	attrs := attrs
	if attrs == nil {
		attrs = SKINNED_ATTRIBUTES[:] if skinned else STANDARD_ATTRIBUTES[:]
	} else {
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
		if tex != nil {
			sdl.ReleaseGPUTexture(g.device, tex)
		}
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
	u := Post_Uniforms {
		tone        = {0, 0, f32(cs.x), f32(cs.y)},
		frame       = {f32(width), f32(height), 0, 0},
		motion      = {max(params.max_blur, 0), params.depth_scale if params.depth_scale > 0 else o3d.MOTION_DEPTH_SCALE, f32(tile), 0},
		motion_size = {f32(width), f32(height), f32(tiles.x), f32(tiles.y)},
	}
	post_draw(g, cmd, g.motion_pipelines[.Rows], {t.rows}, {tiles.x, height}, #partial Post_Bindings{.Velocity = velocity}, &u)
	post_draw(g, cmd, g.motion_pipelines[.Tiles], {t.tiles}, tiles, #partial Post_Bindings{.Motion = t.rows}, &u)
	post_draw(g, cmd, g.motion_pipelines[.Around], {t.around}, tiles, #partial Post_Bindings{.Motion = t.tiles}, &u)
	post_draw(g, cmd, g.motion_pipelines[.Gather], {out}, {width, height}, #partial Post_Bindings{.Light = color, .Velocity = velocity, .Nearness = nearness, .Motion = t.around}, &u)
	return nil
}

// A full-screen pass of `pipeline` (post.slang's) into `targets` at `size`
// from the top left, reading `maps` (a placeholder for those unread).
@(private)
post_draw :: proc(g: ^GPU, cmd: ^sdl.GPUCommandBuffer, pipeline: ^sdl.GPUGraphicsPipeline, targets: []^sdl.GPUTexture, size: [2]u32, maps: Post_Bindings, uniforms: ^Post_Uniforms) {
	colors: [2]sdl.GPUColorTargetInfo
	for t, i in targets {
		colors[i] = {texture = t, load_op = .DONT_CARE, store_op = .STORE}
	}
	pass := sdl.BeginGPURenderPass(cmd, &colors[0], u32(len(targets)), nil)
	defer sdl.EndGPURenderPass(pass)
	sdl.SetGPUViewport(pass, {x = 0, y = 0, w = f32(size.x), h = f32(size.y), min_depth = 0, max_depth = 1})
	sdl.SetGPUScissor(pass, {x = 0, y = 0, w = i32(size.x), h = i32(size.y)})
	sdl.BindGPUGraphicsPipeline(pass, pipeline)
	bindings: [Post_Maps]sdl.GPUTextureSamplerBinding
	for m, k in maps {
		bindings[k] = {texture = m if m != nil else (g.dummy_texture if k == .Light || k == .Half else g.dummy_shadow), sampler = g.shadow_sampler}
	}
	sdl.BindGPUFragmentSamplers(pass, 0, &bindings[.Light], 1 if pipeline == g.output_pipeline else len(Post_Maps))
	sdl.PushGPUFragmentUniformData(cmd, 0, uniforms, size_of(uniforms^))
	sdl.DrawGPUPrimitives(pass, 3, 1, 0, 0)
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
	if job.depth_map != nil {
		sdl.ReleaseGPUTexture(g.device, job.depth_map)
	}
	if job.depth_buffer != nil {
		sdl.ReleaseGPUTexture(g.device, job.depth_buffer)
	}
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
	if job.snapshot_tex != nil {
		sdl.ReleaseGPUTexture(g.device, job.snapshot_tex)
	}
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
	sizes := [4][2]u32{tiles, tiles, half, half}
	formats := [4]sdl.GPUTextureFormat{SHADOW_FORMAT, SHADOW_FORMAT, LIGHT_FORMAT, SHADOW_FORMAT}
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
		if t != nil {
			sdl.ReleaseGPUTexture(g.device, t)
		}
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
			if t != nil {
				sdl.ReleaseGPUTexture(g.device, t)
			}
		}
		job.velocity_map = sdl.CreateGPUTexture(g.device, {type = .D2, format = VELOCITY_FORMAT, usage = {.SAMPLER, .COLOR_TARGET}, width = width, height = height, layer_count_or_depth = 1, num_levels = 1})
		job.nearness_map = sdl.CreateGPUTexture(g.device, {type = .D2, format = NEARNESS_FORMAT, usage = {.SAMPLER, .COLOR_TARGET}, width = width, height = height, layer_count_or_depth = 1, num_levels = 1})
		job.velocity_size = {width, height}
	}
	if blurred && (job.blurred_size != g.capacity || job.blurred == nil) {
		if job.blurred != nil {
			sdl.ReleaseGPUTexture(g.device, job.blurred)
		}
		job.blurred = sdl.CreateGPUTexture(g.device, {type = .D2, format = LIGHT_FORMAT, usage = {.SAMPLER, .COLOR_TARGET}, width = g.capacity.x, height = g.capacity.y, layer_count_or_depth = 1, num_levels = 1})
		job.blurred_size = g.capacity
	}
	return job.velocity_map != nil && job.nearness_map != nil && (!blurred || job.blurred != nil)
}

@(private)
release_motion :: proc(g: ^GPU, job: ^Job) {
	for t in ([]^sdl.GPUTexture{job.velocity_map, job.nearness_map, job.blurred}) {
		if t != nil {
			sdl.ReleaseGPUTexture(g.device, t)
		}
	}
	job.velocity_map, job.nearness_map, job.blurred, job.velocity_size, job.blurred_size = nil, nil, nil, {}, {}
	motion_tables_destroy(g, &job.motion_tables)
}

@(private)
release_reflection :: proc(g: ^GPU, job: ^Job) {
	for t in ([]^sdl.GPUTexture{job.reflection_tex, job.runs_target, job.runs_resolved}) {
		if t != nil {
			sdl.ReleaseGPUTexture(g.device, t)
		}
	}
	job.reflection_tex, job.runs_target, job.runs_resolved, job.reflection_size = nil, nil, nil, {}
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
	for t in ([]^sdl.GPUTexture{g.color, g.depth, g.resolved, g.output}) {
		if t != nil {
			sdl.ReleaseGPUTexture(g.device, t)
		}
	}
	g.color, g.depth, g.resolved, g.output, g.capacity = nil, nil, nil, nil, {}
}

// The mirror's pipelines, made the first time a frame has a mirror.
@(private)
ensure_mirror_pass :: proc(g: ^GPU) -> bool {
	for kind in Draw_Kind {
		for side in o3d.Side {
			for blend in o3d.Blend {
				p := &g.mirror_pipelines[kind][side][blend]
				if p^ == nil {
					p^ = make_draw_pipeline(g, kind, side, blend, .Mirror)
				}
				if p^ == nil {
					return false
				}
			}
		}
	}
	return true
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
