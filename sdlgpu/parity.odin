package o3d_sdlgpu

import "core:encoding/base64"
import "core:fmt"
import "core:math"
import "core:math/linalg"

import sdl "vendor:sdl3"

import "deps:ovg/curves"

import o3d ".."

// GPU-vs-software parity: scenes that cover every shading path (standard
// materials and lights, ShaderMaterials, shadows, textures, environment
// lighting and background, an animated glTF model), each drawn by both
// renderers and compared by PSNR. gpu_test.odin runs them under `odin test`;
// tools/gpu_check runs them on the main thread, which SDL's Metal driver
// needs (the test runner's threads cannot open it).

PARITY_MIN_PSNR :: 32

// What a case draws with, freed by `parity_assets_destroy`.
Parity_Assets :: struct {
	scene:    o3d.Scene,
	geos:     o3d.Geometries,
	shaders:  o3d.Shaders,
	textures: o3d.Textures,
	model:    o3d.Model,
}

Parity_Case :: struct {
	name:          string,
	width, height: int,
	// Fills the assets; errors (a shader's, a model's) say why it could not.
	build:         proc(a: ^Parity_Assets) -> (errors: string),
}

PARITY_CASES := [?]Parity_Case {
	{"standard", 160, 120, build_standard},
	{"custom", 240, 180, build_custom},
	{"shadow", 240, 180, build_shadow},
	{"textures", 200, 150, build_textures},
	{"environment", 200, 150, build_environment},
	{"gltf", 160, 120, build_gltf},
	{"skinned", 160, 120, build_skinned},
	{"morph", 160, 120, build_morph},
	{"manim", 200, 150, build_manim},
	{"shapes", 200, 150, build_shapes},
	{"extrude", 200, 150, build_extrude},
	{"curves", 240, 180, build_curves},
	{"curve_shadow", 200, 150, build_curve_shadow},
	{"text", 240, 180, build_text},
	{"alpha_map", 200, 150, build_alpha_map},
	{"custom_maps", 200, 150, build_custom_maps},
	{"hdr_environment", 200, 150, build_hdr_environment},
	{"shadow_only", 200, 150, build_shadow_only},
	{"particles", 240, 180, build_particles},
	{"labels", 240, 180, build_labels},
	{"additive", 240, 180, build_additive},
	{"flare", 240, 180, build_flare},
	{"vertex_colors", 240, 180, build_vertex_colors},
	{"far_layers", 240, 180, build_far_layers},
	{"triplanar", 240, 180, build_triplanar},
	{"custom_shadow", 240, 180, build_custom_shadow},
	{"custom_finish", 240, 180, build_custom_finish},
	{"sky", 240, 180, build_sky},
	{"background_color", 240, 180, build_background_color},
	{"physical", 240, 180, build_physical},
	{"transmission", 240, 180, build_transmission},
	{"layer_maps", 240, 180, build_layer_maps},
}

parity_assets_destroy :: proc(a: ^Parity_Assets) {
	o3d.scene_destroy(&a.scene)
	o3d.geometries_destroy(&a.geos)
	o3d.shaders_destroy(&a.shaders)
	o3d.textures_destroy(&a.textures)
	o3d.model_destroy(&a.model)
}

// Draws a case on both renderers: the PSNR between their frames, and the
// frames (premultiplied RGBA8, the caller deletes them); `err` when the
// case could not be built or drawn.
parity_check :: proc(g: ^GPU, c: Parity_Case) -> (psnr: f64, gpu_px, cpu_px: []u8, err: string) {
	a: Parity_Assets
	defer parity_assets_destroy(&a)
	if e := c.build(&a); e != "" {
		return 0, nil, nil, fmt.tprintf("%s: %s", c.name, e)
	}
	w, h := u32(c.width), u32(c.height)
	out := target(g, w, h)
	defer sdl.ReleaseGPUTexture(g.device, out)
	if e := render(g, &a.scene, &a.geos, &a.shaders, out, w, h, textures = &a.textures); e != nil {
		return 0, nil, nil, fmt.tprintf("%s: GPU render: %v", c.name, e)
	}
	gpu_px, _ = read(g, out, w, h)
	sw: o3d.Software
	defer o3d.software_destroy(&sw)
	cpu_px = clone_pixels(o3d.software_render(&sw, &a.scene, &a.geos, &a.shaders, c.width, c.height, nil, &a.textures))
	sum: f64
	for i in 0 ..< len(cpu_px) {
		d := f64(cpu_px[i]) - f64(gpu_px[i])
		sum += d * d
	}
	psnr = 10 * math.log10(255 * 255 / max(sum / f64(len(cpu_px)), 1e-9))
	return
}

@(private = "file")
clone_pixels :: proc(px: []u8) -> []u8 {
	out := make([]u8, len(px))
	copy(out, px)
	return out
}

// ---- the cases --------------------------------------------------------------------------

// A scene with every kind of light and material the shading covers.
test_scene :: proc(scene: ^o3d.Scene, geos: ^o3d.Geometries) {
	o3d.scene_clear(scene)
	scene.camera = {position = {2.5, 2, 5}, target = {0, 0.5, 0}, fov = 45, near = 0.1, far = 50}
	append(&scene.lights, o3d.Light{kind = .Hemisphere, color = {0.8, 0.9, 1}, ground = {0.3, 0.2, 0.25}, intensity = 1.5})
	append(&scene.lights, o3d.Light{kind = .Directional, color = {1, 0.95, 0.9}, intensity = 2.5, position = {3, 5, 2}})
	append(&scene.lights, o3d.Light{kind = .Point, color = {0.3, 0.6, 1}, intensity = 8, position = {-1.5, 1, 1.5}, distance = 6, decay = 2})
	append(&scene.lights, o3d.Light{kind = .Ambient, color = {1, 1, 1}, intensity = 0.2})
	red := o3d.MATERIAL_DEFAULT
	red.color, red.roughness = {1, 0.35, 0.3}, 0.35
	metal := o3d.MATERIAL_DEFAULT
	metal.color, metal.metalness, metal.roughness = {0.9, 0.8, 0.5}, 1, 0.3
	floor := o3d.MATERIAL_DEFAULT
	floor.color, floor.roughness = {0.4, 0.42, 0.5}, 0.9
	glass := o3d.MATERIAL_DEFAULT
	glass.color, glass.opacity, glass.flags, glass.side = {0.4, 0.8, 1}, 0.45, {.Transparent}, .Double
	glow := o3d.MATERIAL_DEFAULT
	glow.color, glow.emissive, glow.emissive_intensity, glow.flags = {0.1, 0.1, 0.1}, {1, 0.6, 0.1}, 1.5, {.Flat_Shading}
	append(&scene.draws, o3d.Draw{o3d.plane(geos, 6, 6), o3d.compose({0, 0, 0}, {-math.PI / 2, 0, 0}), floor})
	append(&scene.draws, o3d.Draw{o3d.sphere(geos, 0.7, 48, 24), o3d.compose({-0.6, 0.7, 0}), red})
	append(&scene.draws, o3d.Draw{o3d.box(geos, 0.8, 0.8, 0.8), o3d.compose({1, 0.4, -0.4}, {0, 0.6, 0}), metal})
	append(&scene.draws, o3d.Draw{o3d.torus(geos, 0.45, 0.15, 12, 32), o3d.compose({0.4, 1.4, 0.6}, {0.5, 0.3, 0}), glow})
	append(&scene.draws, o3d.Draw{o3d.plane(geos, 1.4, 1.2), o3d.compose({0.2, 0.6, 1.4}, {0, 0.3, 0}), glass})
}

@(private = "file")
build_standard :: proc(a: ^Parity_Assets) -> string {
	test_scene(&a.scene, &a.geos)
	return ""
}

// The alpha map case (three.js's alphaMap, its green channel): a tiled
// mask cutting a net (Alpha_Mask, a transformed alpha slot without a
// colour map) that casts its holes' shadow on the floor, and a blended
// panel fading by a smooth alpha map over its colour map.
@(private = "file")
build_alpha_map :: proc(a: ^Parity_Assets) -> string {
	TW, TH :: 32, 32
	mask_px: [TW * TH * 4]u8
	fade_px: [TW * TH * 4]u8
	for y in 0 ..< TH {
		for x in 0 ..< TW {
			i := (y * TW + x) * 4
			u, v := f32(x) / f32(TW - 1), f32(y) / f32(TH - 1)
			// Green carries the alpha; red and blue say otherwise, so a
			// renderer reading the wrong channel shows.
			hole := u8(255 if (x / 8 + y / 8) % 2 == 0 else 0)
			mask_px[i], mask_px[i + 1], mask_px[i + 2], mask_px[i + 3] = 255 - hole, hole, 255 - hole, 255
			fade := u8(clamp((0.15 + 0.8 * (0.5 + 0.5 * math.cos(u * math.TAU)) * v) * 255, 0, 255))
			fade_px[i], fade_px[i + 1], fade_px[i + 2], fade_px[i + 3] = 0, fade, 0, 255
		}
	}
	mask_id := o3d.texture_add(&a.textures, mask_px[:], TW, TH)
	fade_id := o3d.texture_add(&a.textures, fade_px[:], TW, TH)
	s := &a.scene
	o3d.scene_clear(s)
	s.camera = {position = {2.2, 2.4, 4.2}, target = {0, 0.4, 0}, fov = 45, near = 0.1, far = 30}
	append(&s.lights, o3d.Light{kind = .Directional, color = {1, 1, 1}, intensity = 2.5, position = {1.5, 5, 2}, cast_shadow = true, shadow = {map_size = 512, bias = -0.0005}})
	append(&s.lights, o3d.Light{kind = .Ambient, color = {1, 1, 1}, intensity = 0.35})
	floor := o3d.MATERIAL_DEFAULT
	floor.color, floor.roughness, floor.flags = {0.7, 0.72, 0.75}, 0.9, {.Receive_Shadow}
	append(&s.draws, o3d.Draw{o3d.plane(&a.geos, 6, 6), o3d.compose({0, 0, 0}, {-math.PI / 2, 0, 0}), floor})
	net := o3d.MATERIAL_DEFAULT
	net.color, net.metalness, net.roughness = {0.55, 0.57, 0.6}, 1, 0.35
	net.flags, net.side, net.alpha_cutoff = {.Alpha_Mask, .Cast_Shadow, .Receive_Shadow}, .Double, 0.5
	net.alpha_map = mask_id
	net.alpha_transform = {scale = {2, 2}}
	append(&s.draws, o3d.Draw{o3d.plane(&a.geos, 1.6, 1.2), o3d.compose({-0.35, 0.9, 0}, {-0.35, 0.4, 0}), net})
	panel := o3d.MATERIAL_DEFAULT
	panel.color, panel.flags, panel.side = {1, 0.6, 0.3}, {.Transparent}, .Double
	panel.alpha_map = fade_id
	append(&s.draws, o3d.Draw{o3d.plane(&a.geos, 1, 1), o3d.compose({0.9, 0.7, 0.6}, {0, -0.5, 0}), panel})
	return ""
}

MAPS_FLOOR_VERT :: `varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}`
// GLSL 1 texture2D at the level its derivatives pick: a floor receding
// far enough to sample several mip levels.
MAPS_FLOOR_FRAG :: `uniform sampler2D map;
varying vec2 vUv;
void main() {
  gl_FragColor = vec4(texture2D(map, vUv * 6.0).rgb, 1.0);
}`
MAPS_BOX_VERT :: `out vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}`
// GLSL 3 with two samplers: textureSize and texelFetch of one, texture of
// the other.
MAPS_BOX_FRAG :: `uniform sampler2D tint;
uniform sampler2D map;
in vec2 vUv;
out vec4 fragColor;
void main() {
  ivec2 size = textureSize(map, 0);
  vec3 texel = texelFetch(map, ivec2(vUv * vec2(size)), 0).rgb;
  fragColor = vec4(mix(texel, texture(tint, vUv).rgb, 0.4), 1.0);
}`

// ShaderMaterial textures: a smooth pattern (so hardware and software
// filtering stay close) with its full mip chain.
@(private = "file")
build_custom_maps :: proc(a: ^Parity_Assets) -> string {
	T :: 64
	px: [T * T * 4]u8
	tint_px: [16 * 16 * 4]u8
	for y in 0 ..< T {
		for x in 0 ..< T {
			i := (y * T + x) * 4
			u, v := f32(x) / T, f32(y) / T
			px[i] = u8(clamp(0.5 + 0.45 * math.cos(u * math.TAU), 0, 1) * 255)
			px[i + 1] = u8(clamp(0.5 + 0.45 * math.sin(v * math.TAU * 2), 0, 1) * 255)
			px[i + 2] = u8(clamp(0.3 + 0.6 * u * v, 0, 1) * 255)
			px[i + 3] = 255
		}
	}
	for i in 0 ..< 16 * 16 {
		tint_px[i * 4], tint_px[i * 4 + 1], tint_px[i * 4 + 2], tint_px[i * 4 + 3] = u8(i % 16 * 16), 90, u8(255 - i / 16 * 16), 255
	}
	tex := o3d.texture_add(&a.textures, px[:], T, T)
	tint := o3d.texture_add(&a.textures, tint_px[:], 16, 16)
	floor_id, e1 := o3d.shader(&a.shaders, MAPS_FLOOR_VERT, MAPS_FLOOR_FRAG, false)
	box_id, e2 := o3d.shader(&a.shaders, MAPS_BOX_VERT, MAPS_BOX_FRAG, true)
	if len(e1) > 0 || len(e2) > 0 {
		return o3d.format_shader_errors(e1 if len(e1) > 0 else e2, context.temp_allocator)
	}
	s := &a.scene
	o3d.scene_clear(s)
	s.camera = {position = {0, 1.2, 4}, target = {0, 0.2, -4}, fov = 50, near = 0.1, far = 60}
	m := o3d.MATERIAL_DEFAULT
	m.shader = floor_id
	m.shader_maps[o3d.shader_map(&a.shaders, floor_id, "map")] = tex
	append(&s.draws, o3d.Draw{o3d.plane(&a.geos, 8, 24), o3d.compose({0, 0, -8}, {-math.PI / 2, 0, 0}), m})
	m = o3d.MATERIAL_DEFAULT
	m.shader = box_id
	m.shader_maps[o3d.shader_map(&a.shaders, box_id, "map")] = tex
	m.shader_maps[o3d.shader_map(&a.shaders, box_id, "tint")] = tint
	append(&s.draws, o3d.Draw{o3d.box(&a.geos, 0.8, 0.8, 0.8), o3d.compose({0.9, 0.6, 0.8}, {0.4, 0.6, 0}), m})
	return ""
}

// The custom case: ShaderMaterials of each kind: three.js's GLSL 1 style with uniforms; GLSL 3
// with a flat int varying and screen derivatives (the 2x2 quads); a
// transparent double-sided one reading gl_FrontFacing; one reading the
// fragment built-ins.
WAVE_VERT :: `varying vec3 vNormal;
varying vec2 vUv;
uniform float uTime;
void main() {
  vNormal = normalize(normalMatrix * normal);
  vUv = uv;
  vec3 p = position + normal * 0.06 * sin(uTime + position.y * 8.0);
  gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
}`

WAVE_FRAG :: `varying vec3 vNormal;
varying vec2 vUv;
uniform vec3 uColor;
void main() {
  float l = 0.3 + 0.7 * max(dot(normalize(vNormal), normalize(vec3(0.4, 0.8, 1.0))), 0.0);
  gl_FragColor = vec4(uColor * l + vec3(vUv * 0.2, 0.0), 1.0);
}`

FACETS_VERT :: `out vec3 vView;
flat out int vFace;
void main() {
  vec4 view = modelViewMatrix * vec4(position, 1.0);
  vView = view.xyz;
  vFace = int(dot(abs(normal), vec3(0.0, 1.0, 2.0)) + 0.5);
  gl_Position = projectionMatrix * view;
}`

FACETS_FRAG :: `in vec3 vView;
flat in int vFace;
uniform vec3 uTints[3];
out vec4 colour;
void main() {
  vec3 n = normalize(cross(dFdx(vView), dFdy(vView)));
  float l = 0.4 + 0.6 * abs(n.z);
  colour = vec4(uTints[vFace] * l, 1.0);
}`

GLASS_VERT :: `varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}`

GLASS_FRAG :: `varying vec2 vUv;
uniform float uAlpha;
void main() {
  vec3 c = gl_FrontFacing ? vec3(0.2, 0.7, 1.0) : vec3(1.0, 0.4, 0.2);
  gl_FragColor = vec4(c * (0.6 + 0.4 * vUv.y), uAlpha);
}`

RIM_VERT :: `varying vec3 vWorld;
varying vec3 vNormalW;
void main() {
  vec4 w = modelMatrix * vec4(position, 1.0);
  vWorld = w.xyz;
  vNormalW = normalize(mat3(modelMatrix) * normal);
  gl_Position = projectionMatrix * viewMatrix * w;
}`

RIM_FRAG :: `varying vec3 vWorld;
varying vec3 vNormalW;
void main() {
  vec3 v = isOrthographic ? vec3(viewMatrix[0][2], viewMatrix[1][2], viewMatrix[2][2]) : normalize(cameraPosition - vWorld);
  float rim = pow(1.0 - max(dot(normalize(vNormalW), v), 0.0), 2.0);
  gl_FragColor = vec4(vec3(0.1, 0.1, 0.2) + vec3(1.0, 0.8, 0.3) * rim, 1.0);
}`

// The standard scene and a ShaderMaterial of each kind.
@(private = "file")
build_custom :: proc(a: ^Parity_Assets) -> string {
	test_scene(&a.scene, &a.geos)
	Source :: struct {
		vertex, fragment: string,
		glsl3:            bool,
	}
	sources := [4]Source{{WAVE_VERT, WAVE_FRAG, false}, {FACETS_VERT, FACETS_FRAG, true}, {RIM_VERT, RIM_FRAG, false}, {GLASS_VERT, GLASS_FRAG, false}}
	ids: [4]o3d.Shader_Id
	for src, i in sources {
		id, errors := o3d.shader(&a.shaders, src.vertex, src.fragment, src.glsl3)
		if len(errors) > 0 {
			return o3d.format_shader_errors(errors, context.temp_allocator)
		}
		ids[i] = id
	}
	m := o3d.MATERIAL_DEFAULT
	m.shader = ids[0]
	m.uniforms = o3d.scene_uniforms(&a.scene, {"uTime", {1.3}}, {"uColor", {0.3, 0.8, 0.4}})
	append(&a.scene.draws, o3d.Draw{o3d.sphere(&a.geos, 0.45, 48, 24), o3d.compose({-1.6, 0.5, 1.2}), m})
	m = o3d.MATERIAL_DEFAULT
	m.shader = ids[1]
	m.uniforms = o3d.scene_uniforms(&a.scene, {"uTints", {0.9, 0.3, 0.3, 0.3, 0.9, 0.3, 0.3, 0.3, 0.9}})
	append(&a.scene.draws, o3d.Draw{o3d.box(&a.geos, 0.6, 0.6, 0.6), o3d.compose({1.5, 1.4, 0.4}, {0.5, 0.7, 0}), m})
	m = o3d.MATERIAL_DEFAULT
	m.shader = ids[2]
	append(&a.scene.draws, o3d.Draw{o3d.torus(&a.geos, 0.35, 0.12, 16, 40), o3d.compose({-0.9, 1.6, -0.6}, {0.9, 0, 0.3}), m})
	m = o3d.MATERIAL_DEFAULT
	m.shader = ids[3]
	m.flags, m.side = {.Transparent}, .Double
	m.uniforms = o3d.scene_uniforms(&a.scene, {"uAlpha", {0.55}})
	append(&a.scene.draws, o3d.Draw{o3d.plane(&a.geos, 1.2, 1.0), o3d.compose({-0.3, 0.9, 1.9}, {0, 2.6, 0}), m})
	return ""
}

// The custom case with the key light casting a shadow, every draw casting
// and receiving (ShaderMaterials cast with their plain geometry and do not
// receive), plus an alpha-masked cutout casting a cut-out shadow.
@(private = "file")
build_shadow :: proc(a: ^Parity_Assets) -> string {
	if e := build_custom(a); e != "" {
		return e
	}
	T :: 32
	mask_px: [T * T * 4]u8
	for y in 0 ..< T {
		for x in 0 ..< T {
			i := (y * T + x) * 4
			dx, dy := f32(x) - 15.5, f32(y) - 15.5
			r := math.sqrt(dx * dx + dy * dy)
			// Annulus cutout: opaque ring, transparent hole and corners.
			alpha := u8(255) if r >= 5.5 && r <= 13.5 else u8(0)
			mask_px[i + 0], mask_px[i + 1], mask_px[i + 2], mask_px[i + 3] = 230, 140, 50, alpha
		}
	}
	mask_id := o3d.texture_add(&a.textures, mask_px[:], T, T)
	cutout := o3d.MATERIAL_DEFAULT
	cutout.color_map = mask_id
	cutout.alpha_cutoff = 0.5
	cutout.flags = {.Alpha_Mask, .Cast_Shadow, .Receive_Shadow}
	cutout.side = .Double
	append(&a.scene.draws, o3d.Draw{o3d.plane(&a.geos, 1.3, 1.3), o3d.compose({0.1, 1.1, 0.3}, {-0.6, 0.3, 0}), cutout})
	for &l in a.scene.lights {
		if l.kind == .Directional {
			l.cast_shadow = true
			l.shadow = {map_size = 1024, height = 9, bias = -0.0005, normal_bias = 0.01}
		}
	}
	append(&a.scene.lights, o3d.Light{
		kind        = .Spot,
		color       = {1, 0.7, 0.35},
		intensity   = 18,
		position    = {-2.2, 3.2, 2.2},
		target      = {0, 0.4, 0},
		distance    = 10,
		decay       = 2,
		angle       = 32 * math.RAD_PER_DEG,
		penumbra    = 0.35,
		cast_shadow = true,
		shadow      = {map_size = 512, bias = -0.001, normal_bias = 0.01},
	})
	for i in 0 ..< len(a.scene.draws) {
		a.scene.draws.material[i].flags += {.Cast_Shadow, .Receive_Shadow}
	}
	return ""
}

// Colour, normal, roughness, emissive and AO maps with mipmaps, UV1,
// KHR_texture_transform, and alpha masking on the standard scene.
@(private = "file")
build_textures :: proc(a: ^Parity_Assets) -> string {
	TW, TH :: 32, 32
	color_px: [TW * TH * 4]u8
	normal_px: [TW * TH * 4]u8
	rough_px: [TW * TH * 4]u8
	emissive_px: [TW * TH * 4]u8
	ao_px: [TW * TH * 4]u8
	for y in 0 ..< TH {
		for x in 0 ..< TW {
			i := (y * TW + x) * 4
			u := f32(x) / f32(TW - 1)
			v := f32(y) / f32(TH - 1)
			// Smooth color pattern so GPU hardware trilinear and software trilinear agree closely
			color_px[i + 0] = u8(clamp(0.25 + 0.65 * (0.5 + 0.5 * math.cos(u * math.TAU * 2)), 0, 1) * 255)
			color_px[i + 1] = u8(clamp(0.25 + 0.65 * (0.5 + 0.5 * math.sin(v * math.TAU * 2)), 0, 1) * 255)
			color_px[i + 2] = u8(clamp(0.35 + 0.50 * u * v, 0, 1) * 255)
			color_px[i + 3] = 255

			nx := 0.25 * math.sin(u * math.TAU * 2)
			ny := 0.25 * math.cos(v * math.TAU * 2)
			nz := math.sqrt(max(1 - nx * nx - ny * ny, 0))
			normal_px[i + 0] = u8(clamp((nx * 0.5 + 0.5) * 255, 0, 255))
			normal_px[i + 1] = u8(clamp((ny * 0.5 + 0.5) * 255, 0, 255))
			normal_px[i + 2] = u8(clamp((nz * 0.5 + 0.5) * 255, 0, 255))
			normal_px[i + 3] = 255

			rough := u8(clamp((0.35 + 0.55 * (0.5 + 0.5 * math.sin((u + v) * math.TAU))) * 255, 0, 255))
			rough_px[i + 0] = rough
			rough_px[i + 1] = rough
			rough_px[i + 2] = rough
			rough_px[i + 3] = 255

			emissive_px[i + 0] = u8(clamp(0.4 * u * 255, 0, 255))
			emissive_px[i + 1] = u8(clamp(0.2 * v * 255, 0, 255))
			emissive_px[i + 2] = u8(clamp(0.5 * (1 - u) * 255, 0, 255))
			emissive_px[i + 3] = 255

			ao_val := u8(clamp((0.25 + 0.75 * (0.5 + 0.5 * math.cos((u - v) * math.TAU))) * 255, 0, 255))
			ao_px[i + 0], ao_px[i + 1], ao_px[i + 2], ao_px[i + 3] = ao_val, ao_val, ao_val, 255
		}
	}
	color_id := o3d.texture_add(&a.textures, color_px[:], TW, TH)
	normal_id := o3d.texture_add(&a.textures, normal_px[:], TW, TH)
	rough_id := o3d.texture_add(&a.textures, rough_px[:], TW, TH)
	emissive_id := o3d.texture_add(&a.textures, emissive_px[:], TW, TH)
	ao_id := o3d.texture_add(&a.textures, ao_px[:], TW, TH)

	test_scene(&a.scene, &a.geos)
	a.scene.draws.material[0].color_map = color_id
	a.scene.draws.material[0].color_transform = {offset = {0.15, 0.25}, rotation = 0.35, scale = {1.5, 1.5}}
	a.scene.draws.material[0].roughness_map = rough_id
	a.scene.draws.material[0].ao_map = ao_id
	a.scene.draws.material[0].ao_map_intensity = 0.85
	a.scene.draws.material[1].color_map = color_id
	a.scene.draws.material[1].normal_map = normal_id
	a.scene.draws.material[1].roughness_map = rough_id
	a.scene.draws.material[1].ao_map = ao_id
	a.scene.draws.material[1].ao_map_intensity = 0.9
	a.scene.draws.material[2].normal_map = normal_id
	a.scene.draws.material[2].emissive = {1, 1, 1}
	a.scene.draws.material[2].emissive_intensity = 0.6
	a.scene.draws.material[2].emissive_map = emissive_id
	a.scene.draws.material[2].emissive_transform = {offset = {0.1, 0.2}, rotation = -0.25, scale = {1.2, 0.8}}

	// A quad with distinct UV0 and UV1 using AO on UV1 (tex_coord = 1).
	q_verts := [4]o3d.Vertex {
		{{-0.7, -0.5, 0}, {0, 0, 1}, {0, 0}, {0, 1}, {}},
		{{ 0.7, -0.5, 0}, {0, 0, 1}, {1, 0}, {1, 0}, {}},
		{{ 0.7,  0.5, 0}, {0, 0, 1}, {1, 1}, {0, 0}, {}},
		{{-0.7,  0.5, 0}, {0, 0, 1}, {0, 1}, {1, 1}, {}},
	}
	q_idx := [6]u32{0, 1, 2, 0, 2, 3}
	uv1_mat := o3d.MATERIAL_DEFAULT
	uv1_mat.color_map = color_id
	uv1_mat.ao_map = ao_id
	uv1_mat.ao_map_intensity = 0.95
	uv1_mat.ao_transform = {offset = {0.1, 0.05}, rotation = 0.2, scale = {1.3, 1.3}, tex_coord = 1}
	append(&a.scene.draws, o3d.Draw{o3d.geometry_add(&a.geos, q_verts[:], q_idx[:]), o3d.compose({-1.4, 1.2, 0.5}, {0, 0.4, 0}), uv1_mat})

	return ""
}

// Equirectangular environment lighting (prefiltered radiance and diffuse
// irradiance), blurred background, ACES Filmic tone mapping, and Exp2 fog.
@(private = "file")
build_environment :: proc(a: ^Parity_Assets) -> string {
	EW, EH :: 64, 32
	env_px: [EW * EH * 4]u8
	for y in 0 ..< EH {
		v := 1 - (f32(y) + 0.5) / f32(EH)
		for x in 0 ..< EW {
			u := (f32(x) + 0.5) / f32(EW)
			dir := o3d.equirect_dir({u, v})
			sky := clamp(dir.y * 0.5 + 0.5, 0, 1)
			col := o3d.Vec3{0.55, 0.3, 0.15} * (1 - sky) + o3d.Vec3{0.2, 0.5, 0.95} * sky
			sun := math.pow(max(dir.x * 0.6 + dir.y * 0.5 + dir.z * 0.6, 0), 16)
			col = {
				clamp(col.r + 0.6 * sun, 0, 1),
				clamp(col.g + 0.55 * sun, 0, 1),
				clamp(col.b + 0.4 * sun, 0, 1),
			}
			i := (y * EW + x) * 4
			env_px[i + 0] = u8(col.r * 255)
			env_px[i + 1] = u8(col.g * 255)
			env_px[i + 2] = u8(col.b * 255)
			env_px[i + 3] = 255
		}
	}
	env_id := o3d.texture_add(&a.textures, env_px[:], EW, EH)

	test_scene(&a.scene, &a.geos)
	clear(&a.scene.lights) // Pure IBL scene so environment radiance & irradiance dominate!
	a.scene.environment = env_id
	a.scene.environment_intensity = 1.2
	a.scene.background = env_id
	a.scene.background_blurriness = 0.25
	a.scene.tone_mapping = .ACES_Filmic
	a.scene.tone_mapping_exposure = 1.15
	a.scene.fog = {kind = .Exp2, color = {0.75, 0.8, 0.9}, density = 0.06}

	o3d.environment_prefilter(&a.textures, env_id)
	return ""
}

// An HDR environment (linear half floats): a sky under 1 and a sun at 40,
// whose highlights stay bright after the radiance pyramid and irradiance
// average it (in 8 bits they clip at 1), tone mapped; the background is
// the environment itself, sharp.
@(private = "file")
build_hdr_environment :: proc(a: ^Parity_Assets) -> string {
	EW, EH :: 64, 32
	env: [EW * EH * 4]f32
	for y in 0 ..< EH {
		v := 1 - (f32(y) + 0.5) / f32(EH)
		for x in 0 ..< EW {
			dir := o3d.equirect_dir({(f32(x) + 0.5) / f32(EW), v})
			sky := clamp(dir.y * 0.5 + 0.5, 0, 1)
			col := o3d.Vec3{0.35, 0.22, 0.12} * (1 - sky) + o3d.Vec3{0.15, 0.35, 0.8} * sky
			// A smooth sun: a peak of 40, not a texel-sharp disk.
			sun := math.pow(max(dir.x * 0.6 + dir.y * 0.55 + dir.z * 0.58, 0), 24) * 40
			i := (y * EW + x) * 4
			env[i], env[i + 1], env[i + 2], env[i + 3] = col.r + sun, col.g + sun * 0.9, col.b + sun * 0.75, 1
		}
	}
	env_id := o3d.texture_add_hdr(&a.textures, env[:], EW, EH)
	test_scene(&a.scene, &a.geos)
	clear(&a.scene.lights)
	a.scene.environment = env_id
	a.scene.background = env_id
	a.scene.tone_mapping = .ACES_Filmic
	o3d.environment_prefilter(&a.textures, env_id)
	return ""
}

// A shadow-only ground (three.js's ShadowMaterial) under two casters, over
// an environment background: the ground shows only their shadows, tinted
// and half opaque; elsewhere the background shows through.
@(private = "file")
build_shadow_only :: proc(a: ^Parity_Assets) -> string {
	if e := build_environment(a); e != "" {
		return e
	}
	s := &a.scene
	clear(&s.draws)
	s.fog = {}
	s.background_blurriness = 0
	append(&s.lights, o3d.Light{kind = .Directional, color = {1, 1, 1}, intensity = 2, position = {2, 6, 3}, cast_shadow = true, shadow = {map_size = 1024, height = 8, bias = -0.0005, normal_bias = 0.01}})
	catcher := o3d.MATERIAL_DEFAULT
	catcher.color, catcher.opacity, catcher.flags = {0.1, 0.05, 0.2}, 0.6, {.Shadow_Only}
	append(&s.draws, o3d.Draw{o3d.plane(&a.geos, 6, 6), o3d.compose({0, 0, 0}, {-math.PI / 2, 0, 0}), catcher})
	caster := o3d.MATERIAL_DEFAULT
	caster.color, caster.roughness, caster.flags = {0.9, 0.5, 0.3}, 0.4, {.Cast_Shadow}
	append(&s.draws, o3d.Draw{o3d.sphere(&a.geos, 0.6, 32, 16), o3d.compose({-0.5, 0.9, 0}), caster})
	append(&s.draws, o3d.Draw{o3d.box(&a.geos, 0.7, 0.7, 0.7), o3d.compose({0.9, 0.35, -0.3}, {0, 0.6, 0}), caster})
	return ""
}

// Particles: a soft smoke column (blended, rotated, farthest first, fading
// where it cuts the floor and the box) and additive sparks, both a radial
// sprite, over a lit floor and box and a bright sky.
@(private = "file")
build_particles :: proc(a: ^Parity_Assets) -> string {
	T :: 32
	puff_px: [T * T * 4]u8
	for y in 0 ..< T {
		for x in 0 ..< T {
			q := o3d.Vec2{f32(x) + 0.5, f32(y) + 0.5} / T - 0.5
			r := math.sqrt(q.x * q.x + q.y * q.y) * 2
			i := (y * T + x) * 4
			fall := clamp(1 - r, 0, 1)
			puff_px[i], puff_px[i + 1], puff_px[i + 2], puff_px[i + 3] = 255, 255, 255, u8(fall * fall * (3 - 2 * fall) * 255)
		}
	}
	puff := o3d.texture_add(&a.textures, puff_px[:], T, T)
	// A bright sky behind: additive sparks add onto it, not onto black.
	SW, SH :: 32, 16
	sky_px: [SW * SH * 4]u8
	for y in 0 ..< SH {
		for x in 0 ..< SW {
			i := (y * SW + x) * 4
			sky_px[i], sky_px[i + 1], sky_px[i + 2], sky_px[i + 3] = u8(120 + y * 6), u8(150 + y * 4), 230, 255
		}
	}
	sky := o3d.texture_add(&a.textures, sky_px[:], SW, SH)
	s := &a.scene
	o3d.scene_clear(s)
	s.background = sky
	s.camera = {position = {2.4, 1.6, 3.6}, target = {0, 0.8, 0}, fov = 45, near = 0.1, far = 30}
	append(&s.lights, o3d.Light{kind = .Hemisphere, color = {0.8, 0.85, 1}, ground = {0.3, 0.25, 0.2}, intensity = 1.2})
	append(&s.lights, o3d.Light{kind = .Directional, color = {1, 0.95, 0.9}, intensity = 2, position = {3, 5, 2}})
	floor := o3d.MATERIAL_DEFAULT
	floor.color, floor.roughness = {0.35, 0.37, 0.42}, 0.9
	append(&s.draws, o3d.Draw{o3d.plane(&a.geos, 6, 6), o3d.compose({0, 0, 0}, {-math.PI / 2, 0, 0}), floor})
	box := o3d.MATERIAL_DEFAULT
	box.color, box.roughness = {0.7, 0.3, 0.2}, 0.5
	append(&s.draws, o3d.Draw{o3d.box(&a.geos, 0.6, 0.6, 0.6), o3d.compose({0.35, 0.3, -0.2}, {0, 0.5, 0}), box})
	seed: u32 = 12345
	rnd :: proc(seed: ^u32) -> f32 {
		seed^ = seed^ * 1664525 + 1013904223
		return f32(seed^ >> 8) / f32(1 << 24)
	}
	smoke: [120]o3d.Particle
	for &p, i in smoke {
		h := f32(i) / len(smoke)
		p.position = {(rnd(&seed) - 0.5) * (0.3 + h), h * 2.2, (rnd(&seed) - 0.5) * (0.3 + h)}
		p.size = 0.35 + 0.5 * h
		p.rotation = rnd(&seed) * math.TAU
		grey := 0.6 + 0.3 * rnd(&seed)
		p.color = {grey, grey, grey * 1.05, 0.45 - 0.25 * h}
	}
	m := o3d.MATERIAL_DEFAULT
	m.color_map, m.softness = puff, 0.3
	append(&s.draws, o3d.Draw{o3d.scene_particles(s, smoke[:]), o3d.compose({-0.3, 0, 0.2}), m})
	sparks: [80]o3d.Particle
	for &p in sparks {
		p.position = {(rnd(&seed) - 0.5) * 1.6, rnd(&seed) * 1.6, (rnd(&seed) - 0.5) * 1.6}
		p.size = 0.05 + 0.06 * rnd(&seed)
		p.color = {1, 0.55 + 0.3 * rnd(&seed), 0.2, 0.9}
	}
	add := o3d.MATERIAL_DEFAULT
	add.color_map, add.flags = puff, {.Additive}
	append(&s.draws, o3d.Draw{o3d.scene_particles(s, sparks[:]), o3d.compose({0.6, 0.2, 0.6}), add})
	return ""
}

// Labels as particles (pixel-sized quads the shape of their premultiplied
// panel): one behind a cut-out net (seen through its gaps), one pinned a
// little inside a wall (`depth_offset` brings it in front), one anchored
// by its bottom edge on the wall and turned.
@(private = "file")
build_labels :: proc(a: ^Parity_Assets) -> string {
	LW, LH :: 64, 24
	label_px: [LW * LH * 4]u8
	for y in 0 ..< LH {
		for x in 0 ..< LW {
			i := (y * LW + x) * 4
			// A rounded panel filling the canvas, its edge a pixel and a
			// half soft, with lighter bars of text on it.
			q := [2]f32{abs(f32(x) + 0.5 - LW / 2) - 25, abs(f32(y) + 0.5 - LH / 2) - 5}
			d := math.sqrt(max(q.x, 0) * max(q.x, 0) + max(q.y, 0) * max(q.y, 0)) + min(max(q.x, q.y), 0) - 5
			cover := clamp(0.5 - d / 1.5, 0, 1) * 0.92
			text := (y == 7 || y == 8 || y == 15 || y == 16) && x > 12 && x < 52 && (x / 5) % 3 != 2
			c := [3]f32{0.95, 0.95, 0.9} if text else {0.12, 0.2, 0.35}
			for ch in 0 ..< 3 {
				label_px[i + ch] = u8(math.round(c[ch] * cover * 255))
			}
			label_px[i + 3] = u8(math.round(cover * 255))
		}
	}
	label := o3d.texture_add(&a.textures, label_px[:], LW, LH)
	N :: 32
	net_px: [N * N * 4]u8
	for y in 0 ..< N {
		for x in 0 ..< N {
			i := (y * N + x) * 4
			wire := u8(255 if x % 8 < 2 || y % 8 < 2 else 0)
			net_px[i], net_px[i + 1], net_px[i + 2], net_px[i + 3] = 200, 200, 210, wire
		}
	}
	net_tex := o3d.texture_add(&a.textures, net_px[:], N, N)
	s := &a.scene
	o3d.scene_clear(s)
	s.camera = {position = {0.3, 1.1, 4}, target = {0, 0.9, 0}, fov = 45, near = 0.1, far = 30}
	append(&s.lights, o3d.Light{kind = .Directional, color = {1, 1, 1}, intensity = 2.2, position = {1, 3, 4}})
	append(&s.lights, o3d.Light{kind = .Ambient, color = {1, 1, 1}, intensity = 0.4})
	wall := o3d.MATERIAL_DEFAULT
	wall.color, wall.roughness = {0.6, 0.55, 0.5}, 0.8
	append(&s.draws, o3d.Draw{o3d.box(&a.geos, 4, 2.4, 0.2), o3d.compose({0, 1, -0.6}), wall})
	net := o3d.MATERIAL_DEFAULT
	net.metalness, net.roughness = 0.8, 0.4
	net.flags, net.side, net.alpha_cutoff, net.color_map = {.Alpha_Mask}, .Double, 0.5, net_tex
	net.color_transform = {scale = {3, 2}}
	append(&s.draws, o3d.Draw{o3d.plane(&a.geos, 1.6, 1.1), o3d.compose({-0.8, 1.1, 0.7}), net})
	labels := [3]o3d.Particle{
		{position = {-0.8, 1.15, 0.2}, size = {64, 24}, color = {1, 1, 1, 1}},
		{position = {0.9, 1.5, -0.53}, size = {64, 24}, color = {1, 0.9, 0.8, 1}},
		{position = {0.9, 0.7, -0.5}, size = {64, 24}, color = {0.8, 1, 0.9, 0.9}, offset = {0, 0.5}, rotation = 0.15},
	}
	m := o3d.MATERIAL_DEFAULT
	m.color_map, m.depth_offset = label, 0.1
	m.flags = {.No_Size_Attenuation, .Premultiplied_Color_Map}
	append(&s.draws, o3d.Draw{o3d.scene_particles(s, labels[:]), o3d.compose({0, 0, 0}), m})
	return ""
}

// A halo's colour at an alpha rising to its rim (added: colour times alpha).
@(private = "file")
HALO_FRAG :: `varying vec3 vWorld;
varying vec3 vNormalW;
void main() {
  vec3 v = normalize(cameraPosition - vWorld);
  float rim = pow(1.0 - abs(dot(normalize(vNormalW), v)), 1.5);
  gl_FragColor = vec4(1.0, 0.7, 0.3, rim);
}`

// Additive materials (three.js's AdditiveBlending) on meshes, a
// ShaderMaterial and lines over a lit planet and a dark sky: a back-face
// atmosphere shell, an opaque-coloured lit ring added, a halo shader and a
// line loop crossing them, overlapping where their sums clamp.
@(private = "file")
build_additive :: proc(a: ^Parity_Assets) -> string {
	SW, SH :: 16, 8
	sky_px: [SW * SH * 4]u8
	for y in 0 ..< SH {
		for x in 0 ..< SW {
			i := (y * SW + x) * 4
			sky_px[i], sky_px[i + 1], sky_px[i + 2], sky_px[i + 3] = u8(4 + y * 2), u8(6 + y * 3), u8(20 + y * 5), 255
		}
	}
	s := &a.scene
	o3d.scene_clear(s)
	s.background = o3d.texture_add(&a.textures, sky_px[:], SW, SH)
	s.camera = {position = {0.4, 0.6, 4.2}, target = {0, 0, 0}, fov = 45, near = 0.1, far = 30}
	append(&s.lights, o3d.Light{kind = .Directional, color = {1, 0.97, 0.9}, intensity = 3, position = {-3, 1, 2}})
	append(&s.lights, o3d.Light{kind = .Ambient, color = {0.6, 0.7, 1}, intensity = 0.1})
	planet := o3d.MATERIAL_DEFAULT
	planet.color, planet.roughness = {0.2, 0.4, 0.8}, 0.7
	append(&s.draws, o3d.Draw{o3d.sphere(&a.geos, 1, 48, 24), o3d.compose({0, 0, 0}), planet})
	air := o3d.MATERIAL_DEFAULT
	air.color, air.opacity, air.side, air.flags = {0.3, 0.6, 1}, 0.7, .Back, {.Basic, .Additive}
	append(&s.draws, o3d.Draw{o3d.sphere(&a.geos, 1.15, 48, 24), o3d.compose({0, 0, 0}), air})
	ring := o3d.MATERIAL_DEFAULT
	ring.color, ring.roughness, ring.flags = {0.8, 0.5, 0.3}, 0.5, {.Additive}
	append(&s.draws, o3d.Draw{o3d.torus(&a.geos, 1.4, 0.06, 12, 64), o3d.compose({0, 0, 0}, {1.2, 0.3, 0}), ring})
	halo, errors := o3d.shader(&a.shaders, RIM_VERT, HALO_FRAG)
	if len(errors) > 0 {
		return o3d.format_shader_errors(errors, context.temp_allocator)
	}
	glow := o3d.MATERIAL_DEFAULT
	glow.shader, glow.flags, glow.side = halo, {.Additive}, .Double
	append(&s.draws, o3d.Draw{o3d.sphere(&a.geos, 0.35, 32, 16), o3d.compose({1.1, 0.7, 0.8}), glow})
	loop: [48]o3d.Vec3
	for &p, i in loop {
		t := f32(i) / len(loop) * math.TAU
		p = {1.25 * math.cos(t), 0.5 * math.sin(2 * t), 1.25 * math.sin(t)}
	}
	line := o3d.MATERIAL_DEFAULT
	line.color, line.size, line.flags = {0.4, 1, 0.5}, 3, {.Basic, .Additive, .No_Size_Attenuation}
	append(&s.draws, o3d.Draw{o3d.scene_polyline(s, loop[:], closed = true), o3d.compose({0, 0, 0}), line})
	return ""
}

// Lens flares (three.js's Lensflare) as additive particles drawn over
// everything (`depth_offset` past the camera) and faded by their probe
// (`occlusion_radius`): a sun partly behind a hull's edge (its glow and
// ghosts along the line through the frame's centre, dimmed alike), one
// wholly behind it (not drawn) and one in the open.
@(private = "file")
build_flare :: proc(a: ^Parity_Assets) -> string {
	T :: 32
	glow_px: [T * T * 4]u8
	for y in 0 ..< T {
		for x in 0 ..< T {
			q := o3d.Vec2{f32(x) + 0.5, f32(y) + 0.5} / T - 0.5
			r := math.sqrt(q.x * q.x + q.y * q.y) * 2
			i := (y * T + x) * 4
			fall := clamp(1 - r, 0, 1)
			glow_px[i], glow_px[i + 1], glow_px[i + 2], glow_px[i + 3] = 255, 255, 255, u8(fall * fall * 255)
		}
	}
	glow := o3d.texture_add(&a.textures, glow_px[:], T, T)
	SW, SH :: 16, 8
	sky_px: [SW * SH * 4]u8
	for y in 0 ..< SH {
		for x in 0 ..< SW {
			i := (y * SW + x) * 4
			sky_px[i], sky_px[i + 1], sky_px[i + 2], sky_px[i + 3] = u8(4 + y * 2), u8(6 + y * 3), u8(20 + y * 5), 255
		}
	}
	s := &a.scene
	o3d.scene_clear(s)
	s.background = o3d.texture_add(&a.textures, sky_px[:], SW, SH)
	W, H :: 240, 180
	s.camera = {position = {0, 0.4, 5}, target = {0, 0.4, 0}, fov = 50, near = 0.1, far = 60}
	append(&s.lights, o3d.Light{kind = .Directional, color = {1, 0.95, 0.85}, intensity = 2.5, position = {2, 1, 3}})
	append(&s.lights, o3d.Light{kind = .Ambient, color = {0.5, 0.6, 1}, intensity = 0.3})
	hull := o3d.MATERIAL_DEFAULT
	hull.color, hull.metalness, hull.roughness = {0.7, 0.72, 0.75}, 0.6, 0.4
	append(&s.draws, o3d.Draw{o3d.box(&a.geos, 1.2, 3, 0.6), o3d.compose({-0.9, 0.2, 0}), hull})
	append(&s.draws, o3d.Draw{o3d.box(&a.geos, 1, 0.8, 0.5), o3d.compose({1.5, -0.6, 0.5}, {0, 0.4, 0}), hull})
	// Suns far behind: half over the tall hull's right edge, wholly behind
	// the low one, and in the open (one past the far plane).
	eye := s.camera.position
	suns := [3]o3d.Vec3{eye + (o3d.Vec3{-0.3, 1.2, 0.3} - eye) * 8, eye + (o3d.Vec3{1.5, -0.6, 0.5} - eye) * 9, {2.3, 1.7, -1}}
	centre := o3d.Vec2{W / 2, H / 2}
	flare: [dynamic]o3d.Particle
	defer delete(flare)
	for sun, k in suns {
		px, _, _ := o3d.camera_project(s.camera, W, H, sun)
		append(&flare, o3d.Particle{position = sun, size = 70 - f32(k) * 20, color = {1, 0.9, 0.7, 1}})
		// Ghosts along the line from the sun through the centre.
		for t, j in ([3]f32{0.6, 1.3, 1.8}) {
			size := o3d.Vec2{18, 18} + f32(j) * 10
			d := (centre - px) * t
			append(&flare, o3d.Particle{position = sun, size = size, color = {0.5, 0.7, 1, 0.6}, offset = {d.x / size.x, -d.y / size.y}})
		}
	}
	m := o3d.MATERIAL_DEFAULT
	m.color_map, m.flags = glow, {.Additive, .No_Size_Attenuation}
	m.depth_offset, m.occlusion_radius = 1e6, 10
	append(&s.draws, o3d.Draw{o3d.scene_particles(s, flare[:]), o3d.compose({0, 0, 0}), m})
	return ""
}

// Vertex colours (three.js's vertexColors): a lit sphere banded by height,
// a panel fading by its vertices' alpha, a polyline's gradient, segments
// of their own colours and points of three brightnesses (a starfield).
@(private = "file")
build_vertex_colors :: proc(a: ^Parity_Assets) -> string {
	s := &a.scene
	o3d.scene_clear(s)
	s.camera = {position = {0.5, 1.2, 4.5}, target = {0, 0.4, 0}, fov = 45, near = 0.1, far = 30}
	append(&s.lights, o3d.Light{kind = .Directional, color = {1, 1, 1}, intensity = 2.5, position = {2, 3, 2}})
	append(&s.lights, o3d.Light{kind = .Ambient, color = {1, 1, 1}, intensity = 0.3})
	// A sphere's vertices coloured by height: red at the bottom to cyan.
	ball := o3d.sphere(&a.geos, 0.6, 32, 16)
	sv, si, _ := o3d.geometry_mesh(&a.geos, ball)
	verts := make([]o3d.Vertex, len(sv), context.temp_allocator)
	indices := make([]u32, len(si), context.temp_allocator)
	copy(verts, sv)
	copy(indices, si)
	for &v in verts {
		h := v.position.y / 1.2 + 0.5
		v.color = o3d.vertex_color({1 - h, h, h, 1})
	}
	lit := o3d.MATERIAL_DEFAULT
	lit.roughness, lit.flags = 0.5, {.Vertex_Colors}
	append(&s.draws, o3d.Draw{o3d.geometry_add(&a.geos, verts, indices), o3d.compose({-0.8, 0.6, 0}), lit})
	// A panel opaque at its left edge, clear at its right, yellow to magenta.
	pv := [4]o3d.Vertex{
		{position = {-0.6, -0.4, 0}, normal = {0, 0, 1}, color = o3d.vertex_color({1, 1, 0, 1})},
		{position = {0.6, -0.4, 0}, normal = {0, 0, 1}, color = o3d.vertex_color({1, 0, 1, 0})},
		{position = {0.6, 0.4, 0}, normal = {0, 0, 1}, color = o3d.vertex_color({1, 0, 1, 0})},
		{position = {-0.6, 0.4, 0}, normal = {0, 0, 1}, color = o3d.vertex_color({1, 1, 0, 1})},
	}
	pi := [6]u32{0, 1, 2, 0, 2, 3}
	fade := o3d.MATERIAL_DEFAULT
	fade.flags, fade.side = {.Basic, .Vertex_Colors, .Transparent}, .Double
	append(&s.draws, o3d.Draw{o3d.geometry_add(&a.geos, pv[:], pi[:]), o3d.compose({0.9, 0.9, 0.3}, {0, -0.3, 0}), fade})
	// A spiral, blue to orange along it.
	N :: 40
	spiral: [N]o3d.Vec3
	spiral_c: [N]o3d.Vec4
	for i in 0 ..< N {
		t := f32(i) / (N - 1)
		spiral[i] = {0.5 * math.cos(t * 9), t * 1.4 - 0.2, 0.5 * math.sin(t * 9)}
		spiral_c[i] = {t, 0.4 + 0.2 * t, 1 - t, 1}
	}
	line := o3d.MATERIAL_DEFAULT
	line.size, line.flags = 4, {.Basic, .Vertex_Colors, .No_Size_Attenuation}
	append(&s.draws, o3d.Draw{o3d.scene_polyline(s, spiral[:], colors = spiral_c[:]), o3d.compose({0.9, 0, -0.3}), line})
	// Segments, each its own colour at each end.
	segs := [6]o3d.Vec3{{-1.6, -0.1, 0.6}, {-0.4, -0.1, 1}, {-1.6, 0, 0.9}, {-1.2, 0.5, 0.9}, {0.2, -0.2, 1}, {1.4, -0.2, 0.8}}
	seg_c := [6]o3d.Vec4{{1, 0, 0, 1}, {1, 1, 0, 1}, {0, 1, 0, 1}, {0, 1, 0, 1}, {1, 1, 1, 1}, {0, 0, 1, 1}}
	thin := o3d.MATERIAL_DEFAULT
	thin.size, thin.flags = 0.04, {.Basic, .Vertex_Colors}
	append(&s.draws, o3d.Draw{o3d.scene_line_segments(s, segs[:], seg_c[:]), o3d.compose({0, 0, 0}), thin})
	// Stars: three brightnesses over a field behind.
	seed: u32 = 777
	rnd :: proc(seed: ^u32) -> f32 {
		seed^ = seed^ * 1664525 + 1013904223
		return f32(seed^ >> 8) / f32(1 << 24)
	}
	stars: [90]o3d.Vec3
	star_c: [90]o3d.Vec4
	for i in 0 ..< len(stars) {
		stars[i] = {(rnd(&seed) - 0.5) * 9, rnd(&seed) * 5 - 1.5, -4 - rnd(&seed) * 2}
		levels := [3]f32{0.35, 0.65, 1}
		b := levels[i % 3]
		star_c[i] = {b, b, b * 0.9 + 0.1, 1}
	}
	star := o3d.MATERIAL_DEFAULT
	star.size, star.flags = 3, {.Basic, .Vertex_Colors, .No_Size_Attenuation}
	append(&s.draws, o3d.Draw{o3d.scene_points(s, stars[:], star_c[:]), o3d.compose({0, 0, 0}), star})
	return ""
}

// Depth precision (reversed depth): striped panels 0.005 in front of
// blue ones 200 units out with near 1 and far 40000, the back drawn last;
// no blue shows through on either renderer.
@(private = "file")
build_far_layers :: proc(a: ^Parity_Assets) -> string {
	s := &a.scene
	o3d.scene_clear(s)
	dir := linalg.normalize(o3d.Vec3{0.3, 0.2, -1})
	s.camera = {position = {0, 0, 0}, target = dir, fov = 30, near = 1, far = 40000}
	append(&s.lights, o3d.Light{kind = .Directional, color = {1, 1, 1}, intensity = 2, position = {1, 2, 3}})
	append(&s.lights, o3d.Light{kind = .Ambient, color = {1, 1, 1}, intensity = 0.4})
	front, back := o3d.MATERIAL_DEFAULT, o3d.MATERIAL_DEFAULT
	front.color, front.roughness = {0.8, 0.7, 0.3}, 0.5
	back.color, back.flags = {0, 0, 1}, {.Basic}
	for k in 0 ..< 5 {
		at := dir * 200 + o3d.Vec3{f32(k - 2) * 14, f32(k % 2) * 6 - 3, 0}
		turn := o3d.Vec3{0.4, 0.3 + f32(k) * 0.1, 0}
		append(&s.draws, o3d.Draw{o3d.box(&a.geos, 12, 8, 0.05), o3d.compose(at, turn), front})
		append(&s.draws, o3d.Draw{o3d.box(&a.geos, 12, 8, 0.05), o3d.compose(at + dir * 0.005, turn), back})
	}
	// The Earth far behind.
	earth := o3d.MATERIAL_DEFAULT
	earth.color = {0.2, 0.4, 0.8}
	append(&s.draws, o3d.Draw{o3d.sphere(&a.geos, 3000, 48, 24), o3d.compose(dir * 20000 + o3d.Vec3{0, -2500, 0}), earth})
	return ""
}

// Triplanar mapping on meshes without uvs (their uvs zeroed): a turned,
// scaled torus and box with colour, roughness and normal maps projected in
// their frames, and a sphere projected in the world's, a slot transform
// turning its pattern; lit, shadowed, two-sided.
@(private = "file")
build_triplanar :: proc(a: ^Parity_Assets) -> string {
	T :: 64
	grime_px, rough_px, bump_px: [T * T * 4]u8
	for y in 0 ..< T {
		for x in 0 ..< T {
			i := (y * T + x) * 4
			check := (x / 8 + y / 8) % 2 == 0
			streak := f32((x * 7 + y * 13) % 23) / 23
			c := [3]f32{0.75, 0.72, 0.68} if check else {0.35, 0.3, 0.28}
			for k in 0 ..< 3 {
				grime_px[i + k] = u8(clamp(c[k] * (0.8 + 0.2 * streak), 0, 1) * 255)
			}
			grime_px[i + 3] = 255
			r := u8(60 + 150 * streak)
			rough_px[i], rough_px[i + 1], rough_px[i + 2], rough_px[i + 3] = r, r, r, 255
			// Round rivets every 16 texels: their slope as a normal.
			q := [2]f32{f32(x % 16) - 7.5, f32(y % 16) - 7.5}
			d := math.sqrt(q.x * q.x + q.y * q.y)
			n := [3]f32{0, 0, 1}
			if d < 5 {
				h := q / 5 * 0.8
				n = {h.x, -h.y, math.sqrt(max(1 - h.x * h.x - h.y * h.y, 0))}
			}
			for k in 0 ..< 3 {
				bump_px[i + k] = u8(math.round((n[k] * 0.5 + 0.5) * 255))
			}
			bump_px[i + 3] = 255
		}
	}
	grime := o3d.texture_add(&a.textures, grime_px[:], T, T)
	rough := o3d.texture_add(&a.textures, rough_px[:], T, T)
	bump := o3d.texture_add(&a.textures, bump_px[:], T, T)
	no_uvs :: proc(geos: ^o3d.Geometries, id: o3d.Geometry_Id) -> o3d.Geometry_Id {
		sv, si, _ := o3d.geometry_mesh(geos, id)
		verts := make([]o3d.Vertex, len(sv), context.temp_allocator)
		indices := make([]u32, len(si), context.temp_allocator)
		copy(verts, sv)
		copy(indices, si)
		for &v in verts {
			v.uv, v.uv1 = {}, {}
		}
		return o3d.geometry_add(geos, verts, indices)
	}
	s := &a.scene
	o3d.scene_clear(s)
	s.camera = {position = {2.2, 2, 4.2}, target = {0, 0.6, 0}, fov = 45, near = 0.1, far = 50}
	append(&s.lights, o3d.Light{kind = .Directional, color = {1, 0.97, 0.92}, intensity = 2.8, position = {3, 5, 2}, cast_shadow = true})
	append(&s.lights, o3d.Light{kind = .Hemisphere, color = {0.7, 0.8, 1}, ground = {0.3, 0.25, 0.2}, intensity = 0.8})
	hull := o3d.MATERIAL_DEFAULT
	hull.color, hull.metalness = {1, 1, 1}, 0.4
	hull.color_map, hull.roughness_map, hull.normal_map = grime, rough, bump
	hull.projection, hull.projection_scale, hull.projection_sharpness = .Object, 1.5, 6
	hull.flags = {.Cast_Shadow, .Receive_Shadow}
	append(&s.draws, o3d.Draw{no_uvs(&a.geos, o3d.torus(&a.geos, 0.55, 0.22, 24, 48)), o3d.compose({-0.8, 0.9, 0}, {0.9, 0.4, 0.2}, {1.2, 0.8, 1}), hull})
	box := hull
	box.side = .Double
	append(&s.draws, o3d.Draw{no_uvs(&a.geos, o3d.box(&a.geos, 0.8, 0.8, 0.8)), o3d.compose({0.9, 0.4, -0.3}, {0, 0.7, 0.3}), box})
	world := hull
	world.projection, world.projection_scale = .World, 2
	world.color_transform = {rotation = 0.5}
	append(&s.draws, o3d.Draw{no_uvs(&a.geos, o3d.sphere(&a.geos, 0.45, 32, 16)), o3d.compose({0.5, 0.45, 1.1}), world})
	floor := o3d.MATERIAL_DEFAULT
	floor.color, floor.roughness, floor.flags = {0.5, 0.5, 0.55}, 0.9, {.Receive_Shadow}
	append(&s.draws, o3d.Draw{o3d.plane(&a.geos, 6, 6), o3d.compose({0, 0, 0}, {-math.PI / 2, 0, 0}), floor})
	return ""
}

@(private = "file")
LIT_VERT :: `varying vec3 vWorld;
varying vec3 vNormal;
varying vec2 vUv;
void main() {
  vec4 w = modelMatrix * vec4(position, 1.0);
  vWorld = w.xyz;
  vNormal = normalize(mat3(modelMatrix) * normal);
  vUv = uv;
  gl_Position = projectionMatrix * viewMatrix * w;
}`

// A ShaderMaterial lit by a uniform sun and shadowed by o3d's shadowAt,
// its own texture beside the shadow map.
@(private = "file")
LIT_FRAG :: `uniform sampler2D grime;
uniform vec3 sunDir;
varying vec3 vWorld;
varying vec3 vNormal;
varying vec2 vUv;
void main() {
  vec3 n = normalize(vNormal);
  float lit = max(dot(n, normalize(sunDir)), 0.0) * shadowAt(vWorld, n);
  vec3 base = texture2D(grime, vUv * 3.0).rgb;
  gl_FragColor = vec4(base * (0.25 + 0.75 * lit), 1.0);
}`

// ShaderMaterials receiving shadows (shadowAt): a floor and a sphere under
// a standard box and torus, the sun from a directional light, the floor's
// texture read by name beside the shadow map.
@(private = "file")
build_custom_shadow :: proc(a: ^Parity_Assets) -> string {
	T :: 16
	px: [T * T * 4]u8
	for i in 0 ..< T * T {
		v := u8(200 if (i % T / 4 + i / T / 4) % 2 == 0 else 150)
		px[i * 4], px[i * 4 + 1], px[i * 4 + 2], px[i * 4 + 3] = v, v, u8(int(v) * 9 / 10), 255
	}
	grime := o3d.texture_add(&a.textures, px[:], T, T)
	id, errors := o3d.shader(&a.shaders, LIT_VERT, LIT_FRAG)
	if len(errors) > 0 {
		return o3d.format_shader_errors(errors, context.temp_allocator)
	}
	s := &a.scene
	o3d.scene_clear(s)
	s.camera = {position = {2.5, 2.4, 4}, target = {0, 0.4, 0}, fov = 45, near = 0.1, far = 50}
	sun := o3d.Vec3{2, 5, 1.5}
	append(&s.lights, o3d.Light{kind = .Directional, color = {1, 1, 1}, intensity = 2.5, position = sun, cast_shadow = true})
	append(&s.lights, o3d.Light{kind = .Ambient, color = {1, 1, 1}, intensity = 0.4})
	m := o3d.MATERIAL_DEFAULT
	m.shader, m.flags = id, {.Receive_Shadow}
	m.uniforms = o3d.scene_uniforms(s, {"sunDir", {sun.x, sun.y, sun.z}})
	ch := o3d.shader_map(&a.shaders, id, "grime")
	if ch < 0 {
		return "no grime sampler"
	}
	m.shader_maps[ch] = grime
	append(&s.draws, o3d.Draw{o3d.plane(&a.geos, 5, 5), o3d.compose({0, 0, 0}, {-math.PI / 2, 0, 0}), m})
	append(&s.draws, o3d.Draw{o3d.sphere(&a.geos, 0.45, 32, 16), o3d.compose({0.9, 0.45, 0.6}), m})
	caster := o3d.MATERIAL_DEFAULT
	caster.color, caster.flags = {0.8, 0.4, 0.3}, {.Cast_Shadow}
	append(&s.draws, o3d.Draw{o3d.box(&a.geos, 0.6, 0.6, 0.6), o3d.compose({-0.4, 1.1, 0}, {0.3, 0.5, 0}), caster})
	append(&s.draws, o3d.Draw{o3d.torus(&a.geos, 0.3, 0.08, 12, 32), o3d.compose({0.75, 1.4, 0.4}, {1.2, 0, 0}), caster})
	return ""
}

// A ShaderMaterial finishing its colour as the standard material does:
// the scene's tone mapping, fog and output encoding.
@(private = "file")
FINISH_FRAG :: `uniform vec3 sunDir;
uniform vec3 ground;
varying vec3 vWorld;
varying vec3 vNormal;
varying vec2 vUv;
void main() {
  vec3 n = normalize(vNormal);
  float stripe = step(0.5, fract(vUv.x * 8.0));
  vec3 c = ground * (0.6 + 0.4 * stripe) * (0.3 + 4.0 * max(dot(n, normalize(sunDir)), 0.0));
  gl_FragColor = linearToOutputTexel(vec4(fogged(toneMapping(c), vWorld), 1.0));
}`

// ShaderMaterials finished with toneMapping, fogged and linearToOutputTexel
// (a striped ground and a bright sphere, past 1) beside standard spheres,
// under AgX with an exposure and FogExp2.
@(private = "file")
build_custom_finish :: proc(a: ^Parity_Assets) -> string {
	id, errors := o3d.shader(&a.shaders, LIT_VERT, FINISH_FRAG)
	if len(errors) > 0 {
		return o3d.format_shader_errors(errors, context.temp_allocator)
	}
	s := &a.scene
	o3d.scene_clear(s)
	s.camera = {position = {0, 1.6, 6}, target = {0, 0.5, -4}, fov = 50, near = 0.1, far = 100}
	s.tone_mapping, s.tone_mapping_exposure = .AgX, 1.4
	s.fog = {kind = .Exp2, color = {0.75, 0.62, 0.55}, density = 0.07}
	sun := o3d.Vec3{-1, 1.2, 0.8}
	append(&s.lights, o3d.Light{kind = .Directional, color = {1, 0.9, 0.8}, intensity = 3, position = sun})
	append(&s.lights, o3d.Light{kind = .Ambient, color = {1, 1, 1}, intensity = 0.3})
	m := o3d.MATERIAL_DEFAULT
	m.shader = id
	m.uniforms = o3d.scene_uniforms(s, {"sunDir", {sun.x, sun.y, sun.z}}, {"ground", {0.35, 0.3, 0.22}})
	append(&s.draws, o3d.Draw{o3d.plane(&a.geos, 40, 40), o3d.compose({0, 0, -10}, {-math.PI / 2, 0, 0}), m})
	hot := m
	hot.uniforms = o3d.scene_uniforms(s, {"sunDir", {sun.x, sun.y, sun.z}}, {"ground", {0.9, 0.5, 0.2}})
	append(&s.draws, o3d.Draw{o3d.sphere(&a.geos, 0.7, 32, 16), o3d.compose({-1.2, 0.7, 0}), hot})
	for z, k in ([]f32{0, -6, -14}) {
		std := o3d.MATERIAL_DEFAULT
		std.color, std.roughness = {0.3, 0.55, 0.85}, 0.5
		append(&s.draws, o3d.Draw{o3d.sphere(&a.geos, 0.7, 32, 16), o3d.compose({1.2 + f32(k), 0.7, z}), std})
	}
	return ""
}

// three.js's Sky at dawn baked by o3d.sky_texture: the background and,
// prefiltered, the environment of a mirror, a rough metal and a dielectric
// sphere (ACES, exposure 0.5 as three.js's example).
@(private = "file")
build_sky :: proc(a: ^Parity_Assets) -> string {
	sky := o3d.SKY_DEFAULT
	sky.sun = {0.6, 0.08, -1}
	sky.turbidity, sky.rayleigh, sky.mie_directional_g = 10, 3, 0.7
	id := o3d.sky_texture(&a.textures, sky, 512)
	o3d.environment_prefilter(&a.textures, id)
	s := &a.scene
	o3d.scene_clear(s)
	s.camera = {position = {0, 0.4, 4}, target = {0.3, 0.6, -1}, fov = 55, near = 0.1, far = 50}
	s.background = id
	s.environment = id
	s.tone_mapping, s.tone_mapping_exposure = .ACES_Filmic, 0.5
	for roughness, k in ([]f32{0.05, 0.45, 0.6}) {
		m := o3d.MATERIAL_DEFAULT
		m.color = {0.95, 0.95, 0.95} if k < 2 else {0.8, 0.3, 0.2}
		m.metalness = 1 if k < 2 else 0
		m.roughness = roughness
		append(&s.draws, o3d.Draw{o3d.sphere(&a.geos, 0.45, 48, 24), o3d.compose({-1.1 + 1.1 * f32(k), 0.2, 0}), m})
	}
	return ""
}

// The additive scene over an opaque clear colour (Scene.background_color):
// what adds lands on it, alpha 1 everywhere.
@(private = "file")
build_background_color :: proc(a: ^Parity_Assets) -> string {
	if e := build_additive(a); e != "" {
		return e
	}
	a.scene.background_color = {0.04, 0.05, 0.12, 1}
	return ""
}

// An animated glTF 2.0 model at t = 0.35.
@(private = "file")
build_gltf :: proc(a: ^Parity_Assets) -> string {
	Bin :: struct #packed {
		pos:   [4][3]f32,
		norm:  [4][3]f32,
		uv:    [4][2]f32,
		idx:   [6]u16,
		_pad:  [2]u16,
		times: [2]f32,
		rot:   [2][4]f32,
	}
	s45 := math.sqrt_f32(0.5)
	b := Bin {
		pos   = {{-1, -1, 0}, {1, -1, 0}, {1, 1, 0}, {-1, 1, 0}},
		norm  = {{0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}},
		uv    = {{0, 1}, {1, 1}, {1, 0}, {0, 0}},
		idx   = {0, 1, 2, 0, 2, 3},
		times = {0, 1},
		rot   = {{0, 0, 0, 1}, {0, s45, 0, s45}},
	}
	encoded := base64.encode(([^]u8)(&b)[:size_of(Bin)], allocator = context.temp_allocator)
	src := fmt.tprintf(
		`{{"asset":{{"version":"2.0"}},"buffers":[{{"uri":"data:application/octet-stream;base64,%s","byteLength":%d}}],"bufferViews":[{{"buffer":0,"byteOffset":0,"byteLength":48}},{{"buffer":0,"byteOffset":48,"byteLength":48}},{{"buffer":0,"byteOffset":96,"byteLength":32}},{{"buffer":0,"byteOffset":128,"byteLength":12}},{{"buffer":0,"byteOffset":144,"byteLength":8}},{{"buffer":0,"byteOffset":152,"byteLength":32}}],"accessors":[{{"bufferView":0,"componentType":5126,"count":4,"type":"VEC3"}},{{"bufferView":1,"componentType":5126,"count":4,"type":"VEC3"}},{{"bufferView":2,"componentType":5126,"count":4,"type":"VEC2"}},{{"bufferView":3,"componentType":5123,"count":6,"type":"SCALAR"}},{{"bufferView":4,"componentType":5126,"count":2,"type":"SCALAR"}},{{"bufferView":5,"componentType":5126,"count":2,"type":"VEC4"}}],"materials":[{{"pbrMetallicRoughness":{{"baseColorFactor":[0.3,0.7,0.9,1],"metallicFactor":0.2,"roughnessFactor":0.4}},"doubleSided":true}}],"meshes":[{{"primitives":[{{"attributes":{{"POSITION":0,"NORMAL":1,"TEXCOORD_0":2}},"indices":3,"material":0}}]}}],"nodes":[{{"mesh":0}}],"scenes":[{{"nodes":[0]}}],"scene":0,"animations":[{{"name":"turn","samplers":[{{"input":4,"output":5,"interpolation":"LINEAR"}}],"channels":[{{"sampler":0,"target":{{"node":0,"path":"rotation"}}}}]}}]}}`,
		encoded,
		size_of(Bin),
	)

	err: o3d.Gltf_Error
	a.model, err = o3d.gltf_parse(transmute([]u8)src, &a.geos, &a.textures)
	if err != .None {
		return o3d.GLTF_ERROR_TEXT[err]
	}
	o3d.scene_clear(&a.scene)
	a.scene.camera.position = {0, 0, 3.5}
	append(&a.scene.lights, o3d.Light{kind = .Directional, color = {1, 0.95, 0.85}, intensity = 2.5, position = {2, 3, 4}})
	append(&a.scene.lights, o3d.Light{kind = .Ambient, color = {1, 1, 1}, intensity = 0.4})
	o3d.gltf_draw(&a.scene, &a.model, anim = 0, time = 0.35)
	return ""
}

// A 2-joint skinned glTF bar bent at t = 0.65, casting a shadow onto a floor.
@(private = "file")
build_skinned :: proc(a: ^Parity_Assets) -> string {
	Bin :: struct #packed {
		pos:     [6][3]f32,
		norm:    [6][3]f32,
		joints:  [6][4]u16,
		weights: [6][4]f32,
		idx:     [12]u16,
		ibm:     [2]o3d.Mat4,
		times:   [2]f32,
		rot:     [2][4]f32,
	}
	s45 := math.sqrt_f32(0.5)
	ibm0 := o3d.Mat4(1)
	ibm1 := o3d.Mat4(1)
	ibm1[1, 3] = -1
	b := Bin {
		pos = {
			{-0.3, 0, 0}, {0.3, 0, 0},
			{-0.3, 1, 0}, {0.3, 1, 0},
			{-0.3, 2, 0}, {0.3, 2, 0},
		},
		norm = {
			{0, 0, 1}, {0, 0, 1},
			{0, 0, 1}, {0, 0, 1},
			{0, 0, 1}, {0, 0, 1},
		},
		joints = {
			{0, 0, 0, 0}, {0, 0, 0, 0},
			{0, 1, 0, 0}, {0, 1, 0, 0},
			{1, 0, 0, 0}, {1, 0, 0, 0},
		},
		weights = {
			{1, 0, 0, 0}, {1, 0, 0, 0},
			{0.5, 0.5, 0, 0}, {0.5, 0.5, 0, 0},
			{1, 0, 0, 0}, {1, 0, 0, 0},
		},
		idx   = {0, 1, 3, 0, 3, 2, 2, 3, 5, 2, 5, 4},
		ibm   = {ibm0, ibm1},
		times = {0, 1},
		rot   = {{0, 0, 0, 1}, {0, 0, s45, s45}},
	}
	encoded := base64.encode(([^]u8)(&b)[:size_of(Bin)], allocator = context.temp_allocator)
	src := fmt.tprintf(
		`{{"asset":{{"version":"2.0"}},"buffers":[{{"uri":"data:application/octet-stream;base64,%s","byteLength":%d}}],"bufferViews":[{{"buffer":0,"byteOffset":0,"byteLength":72}},{{"buffer":0,"byteOffset":72,"byteLength":72}},{{"buffer":0,"byteOffset":144,"byteLength":48}},{{"buffer":0,"byteOffset":192,"byteLength":96}},{{"buffer":0,"byteOffset":288,"byteLength":24}},{{"buffer":0,"byteOffset":312,"byteLength":128}},{{"buffer":0,"byteOffset":440,"byteLength":8}},{{"buffer":0,"byteOffset":448,"byteLength":32}}],"accessors":[{{"bufferView":0,"componentType":5126,"count":6,"type":"VEC3"}},{{"bufferView":1,"componentType":5126,"count":6,"type":"VEC3"}},{{"bufferView":2,"componentType":5123,"count":6,"type":"VEC4"}},{{"bufferView":3,"componentType":5126,"count":6,"type":"VEC4"}},{{"bufferView":4,"componentType":5123,"count":12,"type":"SCALAR"}},{{"bufferView":5,"componentType":5126,"count":2,"type":"MAT4"}},{{"bufferView":6,"componentType":5126,"count":2,"type":"SCALAR"}},{{"bufferView":7,"componentType":5126,"count":2,"type":"VEC4"}}],"materials":[{{"pbrMetallicRoughness":{{"baseColorFactor":[0.9,0.4,0.2,1],"metallicFactor":0.1,"roughnessFactor":0.4}},"doubleSided":true}}],"meshes":[{{"primitives":[{{"attributes":{{"POSITION":0,"NORMAL":1,"JOINTS_0":2,"WEIGHTS_0":3}},"indices":4,"material":0}}]}}],"skins":[{{"inverseBindMatrices":5,"joints":[1,2]}}],"nodes":[{{"mesh":0,"skin":0}},{{"children":[2]}},{{"translation":[0,1,0]}}],"scenes":[{{"nodes":[0,1]}}],"scene":0,"animations":[{{"name":"bend","samplers":[{{"input":6,"output":7,"interpolation":"LINEAR"}}],"channels":[{{"sampler":0,"target":{{"node":2,"path":"rotation"}}}}]}}]}}`,
		encoded,
		size_of(Bin),
	)
	err: o3d.Gltf_Error
	a.model, err = o3d.gltf_parse(transmute([]u8)src, &a.geos, &a.textures)
	if err != .None {
		return o3d.GLTF_ERROR_TEXT[err]
	}
	o3d.scene_clear(&a.scene)
	a.scene.camera = {position = {0.4, 1.4, 3.8}, target = {-0.2, 0.8, 0}, fov = 45, near = 0.1, far = 30}
	append(&a.scene.lights, o3d.Light{
		kind        = .Directional,
		color       = {1, 0.95, 0.85},
		intensity   = 2.6,
		position    = {2, 4, 3},
		target      = {0, 0.5, 0},
		cast_shadow = true,
		shadow      = {map_size = 512, height = 6, bias = -0.002, normal_bias = 0.01},
	})
	append(&a.scene.lights, o3d.Light{kind = .Ambient, color = {1, 1, 1}, intensity = 0.35})
	floor := o3d.MATERIAL_DEFAULT
	floor.color, floor.roughness, floor.flags = {0.45, 0.48, 0.52}, 0.9, {.Receive_Shadow}
	append(&a.scene.draws, o3d.Draw{geometry = o3d.plane(&a.geos, 5, 5), model = o3d.compose({0, 0, 0}, {-math.PI / 2, 0, 0}), material = floor})
	o3d.gltf_draw(&a.scene, &a.model, anim = 0, time = 0.65, flags = {.Cast_Shadow, .Receive_Shadow})
	return ""
}

// Morph targets + skinning parity case: a glTF scene with a two-target morphed
// prism on the left and a morphed + skinned bar on the right (morph applied
// before skinning), both casting shadows onto a floor.
@(private = "file")
build_morph :: proc(a: ^Parity_Assets) -> string {
	Bin :: struct #packed {
		pos:      [6][3]f32,
		norm:     [6][3]f32,
		d_pos0:   [6][3]f32,
		d_norm0:  [6][3]f32,
		d_pos1:   [6][3]f32,
		d_norm1:  [6][3]f32,
		joints:   [6][4]u16,
		weights:  [6][4]f32,
		idx:      [12]u16,
		ibm:      [2]o3d.Mat4,
		times:    [2]f32,
		rot:      [2][4]f32,
		mweights: [4]f32,
	}
	s45 := math.sqrt_f32(0.5)
	ibm0 := o3d.Mat4(1)
	ibm1 := o3d.Mat4(1)
	ibm1[1, 3] = -1
	b := Bin {
		pos = {
			{-0.3, 0, 0}, {0.3, 0, 0},
			{-0.3, 1, 0}, {0.3, 1, 0},
			{-0.3, 2, 0}, {0.3, 2, 0},
		},
		norm = {
			{0, 0, 1}, {0, 0, 1},
			{0, 0, 1}, {0, 0, 1},
			{0, 0, 1}, {0, 0, 1},
		},
		// Target 0 ("Flare"): flares the middle and top vertices outward and tilts normals.
		d_pos0 = {
			{0, 0, 0}, {0, 0, 0},
			{-0.25, 0, 0.25}, {0.25, 0, 0.25},
			{-0.45, 0.3, -0.2}, {0.45, 0.3, -0.2},
		},
		d_norm0 = {
			{0, 0, 0}, {0, 0, 0},
			{-0.3, 0.1, 0}, {0.3, 0.1, 0},
			{-0.4, 0.2, 0}, {0.4, 0.2, 0},
		},
		// Target 1 ("Twist"): shifts top vertices in X/Z.
		d_pos1 = {
			{0, 0, 0}, {0, 0, 0},
			{0.15, 0.1, 0.1}, {0.15, 0.1, -0.1},
			{0.35, 0.2, 0.25}, {0.35, 0.2, -0.25},
		},
		d_norm1 = {
			{0, 0, 0}, {0, 0, 0},
			{0.2, 0, 0.2}, {0.2, 0, -0.2},
			{0.3, 0, 0.3}, {0.3, 0, -0.3},
		},
		joints = {
			{0, 0, 0, 0}, {0, 0, 0, 0},
			{0, 1, 0, 0}, {0, 1, 0, 0},
			{1, 0, 0, 0}, {1, 0, 0, 0},
		},
		weights = {
			{1, 0, 0, 0}, {1, 0, 0, 0},
			{0.5, 0.5, 0, 0}, {0.5, 0.5, 0, 0},
			{1, 0, 0, 0}, {1, 0, 0, 0},
		},
		idx      = {0, 1, 3, 0, 3, 2, 2, 3, 5, 2, 5, 4},
		ibm      = {ibm0, ibm1},
		times    = {0, 1},
		rot      = {{0, 0, 0, 1}, {0, 0, s45, s45}},
		mweights = {0, 0, 1, 0.5},
	}
	encoded := base64.encode(([^]u8)(&b)[:size_of(Bin)], allocator = context.temp_allocator)
	src := fmt.tprintf(
		`{{"asset":{{"version":"2.0"}},"buffers":[{{"uri":"data:application/octet-stream;base64,%s","byteLength":%d}}],"bufferViews":[{{"buffer":0,"byteOffset":0,"byteLength":72}},{{"buffer":0,"byteOffset":72,"byteLength":72}},{{"buffer":0,"byteOffset":144,"byteLength":72}},{{"buffer":0,"byteOffset":216,"byteLength":72}},{{"buffer":0,"byteOffset":288,"byteLength":72}},{{"buffer":0,"byteOffset":360,"byteLength":72}},{{"buffer":0,"byteOffset":432,"byteLength":48}},{{"buffer":0,"byteOffset":480,"byteLength":96}},{{"buffer":0,"byteOffset":576,"byteLength":24}},{{"buffer":0,"byteOffset":600,"byteLength":128}},{{"buffer":0,"byteOffset":728,"byteLength":8}},{{"buffer":0,"byteOffset":736,"byteLength":32}},{{"buffer":0,"byteOffset":768,"byteLength":16}}],"accessors":[{{"bufferView":0,"componentType":5126,"count":6,"type":"VEC3"}},{{"bufferView":1,"componentType":5126,"count":6,"type":"VEC3"}},{{"bufferView":2,"componentType":5126,"count":6,"type":"VEC3"}},{{"bufferView":3,"componentType":5126,"count":6,"type":"VEC3"}},{{"bufferView":4,"componentType":5126,"count":6,"type":"VEC3"}},{{"bufferView":5,"componentType":5126,"count":6,"type":"VEC3"}},{{"bufferView":6,"componentType":5123,"count":6,"type":"VEC4"}},{{"bufferView":7,"componentType":5126,"count":6,"type":"VEC4"}},{{"bufferView":8,"componentType":5123,"count":12,"type":"SCALAR"}},{{"bufferView":9,"componentType":5126,"count":2,"type":"MAT4"}},{{"bufferView":10,"componentType":5126,"count":2,"type":"SCALAR"}},{{"bufferView":11,"componentType":5126,"count":2,"type":"VEC4"}},{{"bufferView":12,"componentType":5126,"count":4,"type":"SCALAR"}}],"materials":[{{"pbrMetallicRoughness":{{"baseColorFactor":[0.25,0.75,0.85,1],"metallicFactor":0.15,"roughnessFactor":0.35}},"doubleSided":true}},{{"pbrMetallicRoughness":{{"baseColorFactor":[0.9,0.45,0.2,1],"metallicFactor":0.1,"roughnessFactor":0.4}},"doubleSided":true}}],"meshes":[{{"weights":[0,0],"extras":{{"targetNames":["Flare","Twist"]}},"primitives":[{{"attributes":{{"POSITION":0,"NORMAL":1}},"targets":[{{"POSITION":2,"NORMAL":3}},{{"POSITION":4,"NORMAL":5}}],"indices":8,"material":0}}]}},{{"weights":[0,0],"extras":{{"targetNames":["Flare","Twist"]}},"primitives":[{{"attributes":{{"POSITION":0,"NORMAL":1,"JOINTS_0":6,"WEIGHTS_0":7}},"targets":[{{"POSITION":2,"NORMAL":3}},{{"POSITION":4,"NORMAL":5}}],"indices":8,"material":1}}]}}],"skins":[{{"inverseBindMatrices":9,"joints":[2,3]}}],"nodes":[{{"mesh":0,"translation":[-0.85,0,0]}},{{"mesh":1,"skin":0,"translation":[0.85,0,0]}},{{"children":[3]}},{{"translation":[0,1,0]}}],"scenes":[{{"nodes":[0,1,2]}}],"scene":0,"animations":[{{"name":"morph_and_bend","samplers":[{{"input":10,"output":12,"interpolation":"LINEAR"}},{{"input":10,"output":11,"interpolation":"LINEAR"}}],"channels":[{{"sampler":0,"target":{{"node":0,"path":"weights"}}}},{{"sampler":0,"target":{{"node":1,"path":"weights"}}}},{{"sampler":1,"target":{{"node":3,"path":"rotation"}}}}]}}]}}`,
		encoded,
		size_of(Bin),
	)
	err: o3d.Gltf_Error
	a.model, err = o3d.gltf_parse(transmute([]u8)src, &a.geos, &a.textures)
	if err != .None {
		return o3d.GLTF_ERROR_TEXT[err]
	}
	o3d.scene_clear(&a.scene)
	a.scene.camera = {position = {0.2, 1.5, 4.2}, target = {0, 0.9, 0}, fov = 45, near = 0.1, far = 30}
	append(&a.scene.lights, o3d.Light{
		kind        = .Directional,
		color       = {1, 0.95, 0.85},
		intensity   = 2.6,
		position    = {2, 4, 3},
		target      = {0, 0.5, 0},
		cast_shadow = true,
		shadow      = {map_size = 512, height = 6, bias = -0.002, normal_bias = 0.01},
	})
	append(&a.scene.lights, o3d.Light{kind = .Ambient, color = {1, 1, 1}, intensity = 0.35})
	floor := o3d.MATERIAL_DEFAULT
	floor.color, floor.roughness, floor.flags = {0.45, 0.48, 0.52}, 0.9, {.Receive_Shadow}
	append(&a.scene.draws, o3d.Draw{geometry = o3d.plane(&a.geos, 5, 5), model = o3d.compose({0, 0, 0}, {-math.PI / 2, 0, 0}), material = floor})
	o3d.gltf_draw(
		&a.scene,
		&a.model,
		anim   = 0,
		time   = 0.7,
		flags  = {.Cast_Shadow, .Receive_Shadow},
		morphs = {{name = "Twist", weight = 0.8}},
	)
	return ""
}

// Manim-style 3D scene: InstancedMesh (per-instance transform + color),
// LineSegments / polyline / wireframe, Points (world-space and pixel-space),
// AgX tone mapping, and Linear fog.
@(private = "file")
build_manim :: proc(a: ^Parity_Assets) -> string {
	o3d.scene_clear(&a.scene)
	a.scene.camera = {position = {2.2, 1.8, 3.6}, target = {0, 0.2, 0}, fov = 45, near = 0.1, far = 30}
	a.scene.tone_mapping = .AgX
	a.scene.tone_mapping_exposure = 1.1
	a.scene.fog = {kind = .Linear, color = {0.08, 0.1, 0.14}, near = 3.0, far = 12.0}
	append(&a.scene.lights, o3d.Light{kind = .Directional, color = {1, 0.95, 0.85}, intensity = 2.5, position = {3, 4, 3}})
	append(&a.scene.lights, o3d.Light{kind = .Ambient, color = {1, 1, 1}, intensity = 0.35})

	// 1. Instanced spheres with per-instance transform and color.
	inst_mat := o3d.MATERIAL_DEFAULT
	inst_mat.color, inst_mat.roughness = {1, 1, 1}, 0.35
	append(&a.scene.draws, o3d.Draw{o3d.sphere(&a.geos, 0.22, 24, 16), o3d.compose(), inst_mat})
	insts: [5]o3d.Instance
	for i in 0 ..< 5 {
		t := f32(i) / 4.0
		ang := (t - 0.5) * 2.2
		insts[i] = {
			transform = o3d.compose({math.sin(ang) * 1.1, 0.15 + 0.3 * math.cos(ang * 2), math.cos(ang) * 0.6 - 0.3}),
			color     = {0.2 + 0.7 * t, 0.8 - 0.4 * t, 0.9 - 0.6 * t},
			has_color = true,
		}
	}
	o3d.scene_set_draw_instances(&a.scene, 0, o3d.scene_instances(&a.scene, insts[:]))

	// 2. A 3D parametric helix polyline in world-space thickness.
	helix_pts: [24]o3d.Vec3
	for i in 0 ..< len(helix_pts) {
		t := f32(i) / f32(len(helix_pts) - 1)
		theta := t * math.TAU * 1.5
		helix_pts[i] = {0.85 * math.cos(theta), -0.4 + 1.3 * t, 0.85 * math.sin(theta)}
	}
	line_mat := o3d.MATERIAL_DEFAULT
	line_mat.color = {1.0, 0.75, 0.2}
	line_mat.size = 0.045
	line_mat.flags = {.Basic}
	append(&a.scene.draws, o3d.Draw{o3d.polyline(&a.geos, helix_pts[:]), o3d.compose(), line_mat})

	// 3. A wireframe box in pixel-space thickness (No_Size_Attenuation).
	wire_mat := o3d.MATERIAL_DEFAULT
	wire_mat.color = {0.35, 0.85, 1.0}
	wire_mat.size = 3.0
	wire_mat.flags = {.Basic, .No_Size_Attenuation}
	append(&a.scene.draws, o3d.Draw{o3d.wireframe(&a.geos, o3d.box(&a.geos, 0.9, 0.9, 0.9)), o3d.compose({-0.8, 0.4, 0.3}, {0.3, 0.5, 0}), wire_mat})

	// 4. Points cloud in world-space size.
	pts := [6]o3d.Vec3 {
		{-1.2, -0.2, 0.8}, {-0.6, 0.9, 0.7}, {0.0, 1.1, 0.4},
		{ 0.6,  0.8, 0.6}, { 1.1, 0.2, 0.7}, {0.4, -0.3, 0.9},
	}
	pt_mat := o3d.MATERIAL_DEFAULT
	pt_mat.color = {1.0, 0.4, 0.7}
	pt_mat.size = 0.09
	pt_mat.flags = {.Basic}
	append(&a.scene.draws, o3d.Draw{o3d.points(&a.geos, pts[:]), o3d.compose(), pt_mat})

	return ""
}

// A capsule, a per-frame tube, and a ShaderMaterial's instances styled by
// instanceColor and instanceId.
@(private = "file")
build_shapes :: proc(a: ^Parity_Assets) -> string {
	o3d.scene_clear(&a.scene)
	a.scene.camera = {position = {0, 0.6, 4}, target = {0, 0, 0}, fov = 45, near = 0.1, far = 30}
	append(&a.scene.lights, o3d.Light{kind = .Directional, color = {1, 1, 1}, intensity = 2.5, position = {2, 3, 4}})
	append(&a.scene.lights, o3d.Light{kind = .Ambient, color = {1, 1, 1}, intensity = 0.4})

	cap_mat := o3d.MATERIAL_DEFAULT
	cap_mat.color, cap_mat.roughness = {0.2, 0.8, 0.55}, 0.4
	append(&a.scene.draws, o3d.Draw{o3d.capsule(&a.geos, 0.3, 0.7, 6, 16), o3d.compose({-1.2, 0, 0}, {0, 0, 0.5}), cap_mat})

	pts: [16]o3d.Vec3
	for i in 0 ..< len(pts) {
		t := f32(i) / f32(len(pts) - 1)
		pts[i] = {0.5 * math.cos(t * math.TAU), -0.8 + 1.6 * t, 0.5 * math.sin(t * math.TAU)}
	}
	tube_mat := o3d.MATERIAL_DEFAULT
	tube_mat.color, tube_mat.roughness = {1, 0.75, 0.2}, 0.5
	append(&a.scene.draws, o3d.Draw{o3d.scene_tube(&a.scene, pts[:], 0.08, 10), o3d.compose(), tube_mat})

	id, errors := o3d.shader(&a.shaders,
		"varying vec3 vNormal; void main() { vNormal = normalMatrix * normal; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }",
		"varying vec3 vNormal; void main() { float l = 0.4 + 0.6 * max(dot(normalize(vNormal), vec3(0.0, 0.0, 1.0)), 0.0); gl_FragColor = vec4(instanceColor * l * (0.5 + 0.25 * instanceId), 1.0); }")
	if id == 0 || len(errors) > 0 {
		return o3d.format_shader_errors(errors, context.temp_allocator)
	}
	sh_mat := o3d.MATERIAL_DEFAULT
	sh_mat.shader = id
	di := len(a.scene.draws)
	append(&a.scene.draws, o3d.Draw{o3d.box(&a.geos, 0.3, 0.3, 0.3), o3d.compose({1.2, 0, 0}), sh_mat})
	insts := [3]o3d.Instance {
		{transform = o3d.compose({0, -0.5, 0}, {0.3, 0.4, 0}), color = {1, 0.3, 0.4}, has_color = true},
		{transform = o3d.compose({0, 0, 0}, {0.3, 0.8, 0})},
		{transform = o3d.compose({0, 0.5, 0}, {0.3, 1.2, 0}), color = {0.3, 0.6, 1}, has_color = true},
	}
	o3d.scene_set_draw_instances(&a.scene, di, o3d.scene_instances(&a.scene, insts[:]))
	return ""
}

// ExtrudeGeometry from contours grouped by shapes_from_contours: a bevelled
// "B" with smoothed walls, its caps and sides drawn with two materials, a
// flat-faceted ring through it, and a ShapeGeometry plate behind.
build_extrude :: proc(a: ^Parity_Assets) -> string {
	o3d.scene_clear(&a.scene)
	a.scene.camera = {position = {0.6, 1.2, 5}, target = {0, 0, 0}, fov = 45, near = 0.1, far = 30}
	append(&a.scene.lights, o3d.Light{kind = .Directional, color = {1, 1, 1}, intensity = 2.5, position = {2, 3, 4}})
	append(&a.scene.lights, o3d.Light{kind = .Hemisphere, color = {0.8, 0.9, 1}, ground = {0.3, 0.2, 0.2}, intensity = 0.8})

	body := []o3d.Vec2{{0, 0}, {2.2, 0}, {3, 0.8}, {3, 2.2}, {2.6, 2.5}, {3, 2.8}, {3, 4.2}, {2.2, 5}, {0, 5}}
	upper := []o3d.Vec2{{1, 3}, {1, 4}, {2, 4}, {2, 3}}
	lower := []o3d.Vec2{{1, 1}, {1, 2}, {2, 2}, {2, 1}}
	shapes := o3d.shapes_from_contours({upper, body, lower})
	defer o3d.shapes_destroy(shapes)
	o := o3d.EXTRUDE_DEFAULT
	o.depth, o.bevel_thickness, o.bevel_size, o.bevel_segments, o.crease_angle = 0.8, 0.2, 0.15, 4, math.PI / 4
	b := o3d.extrude_geometry(&a.geos, shapes, o)
	at := o3d.compose({-0.9, -1.2, 0}, {0, -0.5, 0}, {0.5, 0.5, 0.5})
	caps := o3d.MATERIAL_DEFAULT
	caps.color, caps.roughness = {0.95, 0.35, 0.3}, 0.35
	sides := o3d.MATERIAL_DEFAULT
	sides.color, sides.roughness, sides.metalness = {0.95, 0.8, 0.3}, 0.3, 0.6
	append(&a.scene.draws, o3d.Draw{b.caps, at, caps}, o3d.Draw{b.sides, at, sides})

	outer, inner: [24]o3d.Vec2
	for i in 0 ..< len(outer) {
		t := f32(i) / f32(len(outer)) * math.TAU
		outer[i] = {math.cos(t), math.sin(t)}
		inner[i] = 0.6 * outer[i]
	}
	ring := []o3d.Shape{{outer[:], {inner[:]}}}
	flat := o3d.EXTRUDE_DEFAULT
	flat.depth, flat.bevel_thickness, flat.bevel_size, flat.bevel_segments = 0.3, 0.08, 0.05, 2
	ring_mat := o3d.MATERIAL_DEFAULT
	ring_mat.color, ring_mat.roughness = {0.3, 0.6, 0.95}, 0.5
	append(&a.scene.draws, o3d.Draw{o3d.extrude_geometry(&a.geos, ring, flat).geometry, o3d.compose({0.9, 0, 0}, {0.4, 0.9, 0}), ring_mat})

	plate := []o3d.Shape{{[]o3d.Vec2{{-2, -1.5}, {2, -1.5}, {2, 1.5}, {-2, 1.5}}, {inner[:]}}}
	plate_mat := o3d.MATERIAL_DEFAULT
	plate_mat.color, plate_mat.roughness = {0.85, 0.85, 0.8}, 0.8
	append(&a.scene.draws, o3d.Draw{o3d.shape_geometry(&a.geos, plate), o3d.compose({0, 0, -1}), plate_mat})
	return ""
}

// A heart of two cubics and a ring of two circles (cubic arcs), as paths.
@(private = "file")
curve_paths :: proc(heart, ring: ^o3d.Path) {
	o3d.path_move_to(heart, {0, -1})
	o3d.path_bezier_curve_to(heart, {-1.6, -0.1}, {-0.9, 1.2}, {0, 0.5})
	o3d.path_bezier_curve_to(heart, {0.9, 1.2}, {1.6, -0.1}, {0, -1})
	for r in ([2]f32{1, 0.55}) {
		k := r * 0.5522847498
		o3d.path_move_to(ring, {r, 0})
		o3d.path_bezier_curve_to(ring, {r, k}, {k, r}, {0, r})
		o3d.path_bezier_curve_to(ring, {-k, r}, {-r, k}, {-r, 0})
		o3d.path_bezier_curve_to(ring, {-r, -k}, {-k, -r}, {0, -r})
		o3d.path_bezier_curve_to(ring, {k, -r}, {r, -k}, {r, 0})
	}
}

// Exact shapes (curve.odin): an opaque heart (coverage by sample) and a
// slanted even-odd ring (transparent: coverage by area) over a lit plate, a
// small far heart (its pixel footprint near its bands), and a colour-mapped
// ring, both of whose edges cross a mesh's.
build_curves :: proc(a: ^Parity_Assets) -> string {
	o3d.scene_clear(&a.scene)
	a.scene.camera = {position = {0.3, 0.5, 4}, target = {0, 0, 0}, fov = 45, near = 0.1, far = 30}
	append(&a.scene.lights, o3d.Light{kind = .Directional, color = {1, 1, 1}, intensity = 2.5, position = {2, 3, 4}})
	append(&a.scene.lights, o3d.Light{kind = .Hemisphere, color = {0.8, 0.9, 1}, ground = {0.3, 0.2, 0.2}, intensity = 0.8})
	heart, ring: o3d.Path
	defer o3d.path_destroy(&heart)
	defer o3d.path_destroy(&ring)
	curve_paths(&heart, &ring)
	heart_geo := o3d.exact_shape_geometry(&a.geos, &heart)
	ring_geo := o3d.exact_shape_geometry(&a.geos, &ring, .Even_Odd)

	plate := o3d.MATERIAL_DEFAULT
	plate.color, plate.roughness = {0.35, 0.4, 0.5}, 0.8
	append(&a.scene.draws, o3d.Draw{o3d.plane(&a.geos, 5, 3), o3d.compose({0, 0, -0.6}), plate})
	red := o3d.MATERIAL_DEFAULT
	red.color, red.roughness = {0.95, 0.3, 0.3}, 0.4
	append(&a.scene.draws, o3d.Draw{heart_geo, o3d.compose({-1, 0.3, 0}, {0, 0.4, 0.2}, {0.8, 0.8, 0.8}), red})
	glass := o3d.MATERIAL_DEFAULT
	glass.color, glass.roughness, glass.flags, glass.side = {0.4, 0.8, 1}, 0.3, {.Transparent}, .Double
	append(&a.scene.draws, o3d.Draw{ring_geo, o3d.compose({0.9, 0.3, 0.2}, {-0.9, 0.3, 0}, {0.7, 0.7, 0.7}), glass})
	small := o3d.MATERIAL_DEFAULT
	small.color, small.flags = {1, 0.9, 0.4}, {.Basic, .Transparent}
	append(&a.scene.draws, o3d.Draw{heart_geo, o3d.compose({0, -0.7, 0.5}, {}, {0.06, 0.06, 0.06}), small})
	T :: 16
	px: [T * T * 4]u8
	for y in 0 ..< T {
		for x in 0 ..< T {
			i := (y * T + x) * 4
			px[i], px[i + 1], px[i + 2], px[i + 3] = u8(x * 16), 200, u8(y * 16), 255
		}
	}
	mapped := o3d.MATERIAL_DEFAULT
	mapped.color_map = o3d.texture_add(&a.textures, px[:], T, T)
	append(&a.scene.draws, o3d.Draw{ring_geo, o3d.compose({0.2, -0.5, 0.3}, {-0.3, 0, 0}, {0.35, 0.35, 0.35}), mapped})
	return ""
}

// Exact shapes' shadows: an opaque heart and a transparent ring standing over
// a floor, both cast their outlines (not their quads) into the map.
build_curve_shadow :: proc(a: ^Parity_Assets) -> string {
	o3d.scene_clear(&a.scene)
	a.scene.camera = {position = {1.5, 2, 4}, target = {0, 0.4, 0}, fov = 45, near = 0.1, far = 30}
	append(&a.scene.lights, o3d.Light{kind = .Directional, color = {1, 1, 1}, intensity = 2.5, position = {1.5, 4, 2.5}, cast_shadow = true, shadow = {map_size = 1024, height = 6, bias = -0.0005, normal_bias = 0.01}})
	append(&a.scene.lights, o3d.Light{kind = .Ambient, color = {1, 1, 1}, intensity = 0.4})
	heart, ring: o3d.Path
	defer o3d.path_destroy(&heart)
	defer o3d.path_destroy(&ring)
	curve_paths(&heart, &ring)
	floor := o3d.MATERIAL_DEFAULT
	floor.color, floor.roughness, floor.flags = {0.8, 0.8, 0.75}, 0.9, {.Receive_Shadow}
	append(&a.scene.draws, o3d.Draw{o3d.plane(&a.geos, 6, 6), o3d.compose({0, 0, 0}, {-math.PI / 2, 0, 0}), floor})
	red := o3d.MATERIAL_DEFAULT
	red.color, red.flags, red.side = {0.95, 0.3, 0.3}, {.Cast_Shadow, .Receive_Shadow}, .Double
	append(&a.scene.draws, o3d.Draw{o3d.exact_shape_geometry(&a.geos, &heart), o3d.compose({-0.7, 0.8, 0}, {0, 0.5, 0}, {0.6, 0.6, 0.6}), red})
	glass := o3d.MATERIAL_DEFAULT
	glass.color, glass.flags, glass.side = {0.4, 0.8, 1}, {.Transparent, .Cast_Shadow}, .Double
	append(&a.scene.draws, o3d.Draw{o3d.exact_shape_geometry(&a.geos, &ring, .Even_Odd), o3d.compose({0.8, 0.7, 0.2}, {0, -0.4, 0}, {0.55, 0.55, 0.55}), glass})
	return ""
}

// Glyph runs (curve.odin): sets shared across runs and glyphs, a y-down
// glyph from ovg's builder (as a font's), glyphs whose quads overlap and a
// mark over another glyph. An opaque run stands tilted over a shadowed
// floor and casts its glyphs' outlines; a transparent mirrored run (font
// units, y down) lies far along the floor (grazing, small); a
// colour-mapped one faces the eye.
build_text :: proc(a: ^Parity_Assets) -> string {
	o3d.scene_clear(&a.scene)
	a.scene.camera = {position = {0.1, 1.2, 2.5}, target = {-0.1, 0.5, 0.4}, fov = 45, near = 0.1, far = 30}
	append(&a.scene.lights, o3d.Light{kind = .Directional, color = {1, 1, 1}, intensity = 2.5, position = {1, 4, 3}, cast_shadow = true, shadow = {map_size = 1024, height = 6, bias = -0.0005, normal_bias = 0.01}})
	append(&a.scene.lights, o3d.Light{kind = .Hemisphere, color = {0.8, 0.9, 1}, ground = {0.3, 0.2, 0.2}, intensity = 0.8})
	heart, ring: o3d.Path
	defer o3d.path_destroy(&heart)
	defer o3d.path_destroy(&ring)
	curve_paths(&heart, &ring)
	heart_set := o3d.curve_set_add(&a.geos, &heart)
	ring_set := o3d.curve_set_add(&a.geos, &ring, .Even_Odd)
	// A bowl with a stem, y down from the pen as a font's glyph.
	b: curves.Builder
	defer curves.builder_destroy(&b)
	curves.move_to(&b, {0, 0})
	curves.line_to(&b, {0, -1.4})
	curves.line_to(&b, {0.25, -1.4})
	curves.line_to(&b, {0.25, -0.7})
	curves.quad_to(&b, {1.1, -1.1}, {1.1, -0.35})
	curves.quad_to(&b, {1.1, 0.4}, {0.25, 0})
	curves.close(&b)
	words, origin, extent, ok := curves.build(&b)
	if !ok {
		return "text: the glyph's curves did not build"
	}
	defer delete(words)
	bowl_set := o3d.curve_set_add_words(&a.geos, words, origin, extent)
	if heart_set == 0 || ring_set == 0 || bowl_set == 0 {
		return "text: a curve set was not made"
	}
	upright := []o3d.Glyph{{heart_set, {0, 0}}, {ring_set, {0.37, 0}}, {heart_set, {0.68, 0}}, {ring_set, {0.68, 0.05}}, {heart_set, {1.01, 0}}}
	opaque_run := o3d.glyph_run_geometry(&a.geos, upright, 0.22)
	font_run := o3d.glyph_run_geometry(&a.geos, []o3d.Glyph{{bowl_set, {0, 0}}, {bowl_set, {0.27, 0}}, {heart_set, {0.68, 0.12}}, {bowl_set, {0.88, 0}}}, {0.25, -0.25})

	floor := o3d.MATERIAL_DEFAULT
	floor.color, floor.roughness, floor.flags = {0.8, 0.8, 0.75}, 0.9, {.Receive_Shadow}
	append(&a.scene.draws, o3d.Draw{o3d.plane(&a.geos, 8, 8), o3d.compose({0, 0, 0}, {-math.PI / 2, 0, 0}), floor})
	red := o3d.MATERIAL_DEFAULT
	red.color, red.roughness, red.flags, red.side = {0.95, 0.3, 0.3}, 0.4, {.Cast_Shadow, .Receive_Shadow}, .Double
	append(&a.scene.draws, o3d.Draw{opaque_run, o3d.compose({-0.7, 0.55, 0}, {-0.2, 0.3, 0}), red})
	ink := o3d.MATERIAL_DEFAULT
	ink.color, ink.flags = {0.1, 0.15, 0.4}, {.Basic, .Transparent}
	append(&a.scene.draws, o3d.Draw{font_run, o3d.compose({-0.4, 0.002, 0.75}, {-math.PI / 2, 0, 0.1}, {0.8, 0.8, 0.8}), ink})
	T :: 16
	px: [T * T * 4]u8
	for y in 0 ..< T {
		for x in 0 ..< T {
			i := (y * T + x) * 4
			px[i], px[i + 1], px[i + 2], px[i + 3] = u8(x * 16), 200, u8(y * 16), 255
		}
	}
	mapped := o3d.MATERIAL_DEFAULT
	mapped.color_map, mapped.flags = o3d.texture_add(&a.textures, px[:], T, T), {.Transparent}
	append(&a.scene.draws, o3d.Draw{font_run, o3d.compose({-0.3, 0.95, 0.3}, {}, {0.6, 0.6, 0.6}), mapped})
	return ""
}

// The frame's depth as distance (`record_depth_into`, recorded after the
// render in its command buffer) against `o3d.depth_image` of the software
// renderer's, over the depth check's cases and the labels: drawn where the
// software renderer drew and within 1e-3 of its distance (relatively), and
// the given far where nothing was; at most 1 in 2000 texels may differ (as
// in `depth_check`).
depth_image_check :: proc(g: ^GPU) -> (err: string) {
	NAMES :: [?]string{"shadow", "alpha_map", "curves", "labels"}
	FAR :: 1000
	for name in NAMES {
		c: Parity_Case
		for pc in PARITY_CASES {
			if pc.name == name {
				c = pc
			}
		}
		a: Parity_Assets
		defer parity_assets_destroy(&a)
		if e := c.build(&a); e != "" {
			return fmt.tprintf("depth image: %s: %s", name, e)
		}
		a.scene.depth = true
		w, h := u32(c.width), u32(c.height)
		out := target(g, w, h)
		defer sdl.ReleaseGPUTexture(g.device, out)
		img := depth_image_target(g, w, h)
		defer sdl.ReleaseGPUTexture(g.device, img)
		cmd := sdl.AcquireGPUCommandBuffer(g.device)
		if e := render(g, &a.scene, &a.geos, &a.shaders, out, w, h, cmd, &a.textures); e != nil {
			_ = sdl.CancelGPUCommandBuffer(cmd)
			return fmt.tprintf("depth image: %s: render: %v", name, e)
		}
		if e := record_depth_into(g, &g.job, img, FAR, cmd); e != nil {
			_ = sdl.CancelGPUCommandBuffer(cmd)
			return fmt.tprintf("depth image: %s: %v", name, e)
		}
		fence := sdl.SubmitGPUCommandBufferAndAcquireFence(cmd)
		if fence == nil {
			return "depth image: submit failed"
		}
		_ = sdl.WaitForGPUFences(g.device, true, &fence, 1)
		sdl.ReleaseGPUFence(g.device, fence)
		got, rerr := read_depth_image(g, img, w, h)
		defer delete(got)
		if rerr != nil || len(got) != int(w * h * 4) {
			return fmt.tprintf("depth image: %s: no GPU image", name)
		}
		sw: o3d.Software
		defer o3d.software_destroy(&sw)
		o3d.software_render(&sw, &a.scene, &a.geos, &a.shaders, int(w), int(h), nil, &a.textures)
		want := make([]f16, w * h * 4, context.temp_allocator)
		o3d.depth_image(a.scene.camera, o3d.software_depth(&sw), int(w), int(h), FAR, want)
		differ, drawn := 0, 0
		for i in 0 ..< int(w * h) {
			gv, wv := got[i * 4:][:4], want[i * 4:][:4]
			drawn += 1 if wv[3] == 1 else 0
			same := gv[3] == wv[3] && abs(f32(gv[0]) - f32(wv[0])) <= 1e-3 * f32(wv[0]) && gv[0] == gv[1] && gv[0] == gv[2]
			differ += 0 if same else 1
		}
		if drawn == 0 {
			return fmt.tprintf("depth image: %s: nothing drawn", name)
		}
		if f64(differ) / f64(w * h) > 1.0 / 2000 {
			return fmt.tprintf("depth image: %s: %d of %d texels differ", name, differ, w * h)
		}
	}
	a: Parity_Assets
	defer parity_assets_destroy(&a)
	build_standard(&a)
	img := depth_image_target(g, 16, 16)
	defer sdl.ReleaseGPUTexture(g.device, img)
	out := target(g, 16, 16)
	defer sdl.ReleaseGPUTexture(g.device, out)
	if render(g, &a.scene, &a.geos, &a.shaders, out, 16, 16) != nil {
		return "depth image: render without depth"
	}
	if e := record_depth_into(g, &g.job, img, FAR, nil); e != .Not_Prepared {
		return fmt.tprintf("depth image: without Scene.depth (%v)", e)
	}
	return ""
}

// ---- external textures ------------------------------------------------------------------

// External textures (`o3d.texture_add_external`): a frame drawn with the
// host's GPU texture bound to one is the same bytes as with its pixels
// uploaded as an image (magnified, so neither samples past its base level);
// with nothing bound, or bound for another table, the same as with no map.
// Not a software parity case (the software renderer has no such pixels);
// gpu_test.odin and tools/gpu_check run it. `err` says what differed.
external_check :: proc(g: ^GPU) -> (err: string) {
	TW, TH :: 16, 16
	px: [TW * TH * 4]u8
	for y in 0 ..< TH {
		for x in 0 ..< TW {
			i := (y * TW + x) * 4
			px[i], px[i + 1], px[i + 2], px[i + 3] = u8(x * 16), u8(y * 16), u8(255 - (x ~ y) * 16), 255
		}
	}
	a: Parity_Assets
	defer parity_assets_destroy(&a)
	a.scene.camera = {position = {0, 0, 2.2}, target = {0, 0, 0}, fov = 45, near = 0.1, far = 10}
	append(&a.scene.lights, o3d.Light{kind = .Directional, color = {1, 1, 1}, intensity = 2.5, position = {1, 2, 3}})
	append(&a.scene.lights, o3d.Light{kind = .Ambient, color = {1, 1, 1}, intensity = 0.4})
	m := o3d.MATERIAL_DEFAULT
	m.roughness = 0.6
	append(&a.scene.draws, o3d.Draw{o3d.plane(&a.geos, 2, 2), o3d.compose({0, 0, 0}, {0, 0.3, 0}), m})

	images, externals, other: o3d.Textures
	defer o3d.textures_destroy(&images)
	defer o3d.textures_destroy(&externals)
	defer o3d.textures_destroy(&other)
	image_id := o3d.texture_add(&images, px[:], TW, TH)
	external_id := o3d.texture_add_external(&externals, TW, TH)
	o3d.texture_add_external(&other, TW, TH)

	frame :: proc(g: ^GPU, a: ^Parity_Assets, textures: ^o3d.Textures, id: o3d.Texture_Id) -> []u8 {
		W, H :: 160, 120
		a.scene.draws.material[0].color_map = id
		out := target(g, W, H)
		defer sdl.ReleaseGPUTexture(g.device, out)
		if render(g, &a.scene, &a.geos, nil, out, W, H, textures = textures) != nil {
			return nil
		}
		frame, _ := read(g, out, W, H)
		return frame
	}
	plain := frame(g, &a, nil, 0)
	defer delete(plain)
	image := frame(g, &a, &images, image_id)
	defer delete(image)
	unbound := frame(g, &a, &externals, external_id)
	defer delete(unbound)

	tex := sdl.CreateGPUTexture(g.device, {type = .D2, format = COLOR_FORMAT, usage = {.SAMPLER}, width = TW, height = TH, layer_count_or_depth = 1, num_levels = 1, sample_count = ._1})
	if tex == nil {
		return "external: no GPU texture"
	}
	defer sdl.ReleaseGPUTexture(g.device, tex)
	transfer := sdl.CreateGPUTransferBuffer(g.device, {usage = .UPLOAD, size = len(px)})
	mapped := sdl.MapGPUTransferBuffer(g.device, transfer, false)
	copy(([^]u8)(mapped)[:len(px)], px[:])
	sdl.UnmapGPUTransferBuffer(g.device, transfer)
	cmd := sdl.AcquireGPUCommandBuffer(g.device)
	pass := sdl.BeginGPUCopyPass(cmd)
	sdl.UploadToGPUTexture(pass, {transfer_buffer = transfer, pixels_per_row = TW, rows_per_layer = TH}, {texture = tex, w = TW, h = TH, d = 1}, false)
	sdl.EndGPUCopyPass(pass)
	_ = sdl.SubmitGPUCommandBuffer(cmd)
	sdl.ReleaseGPUTransferBuffer(g.device, transfer)

	texture_set_external(g, &other, external_id, tex)
	stale := frame(g, &a, &externals, external_id)
	defer delete(stale)
	texture_set_external(g, &externals, external_id, tex)
	bound := frame(g, &a, &externals, external_id)
	defer delete(bound)
	texture_set_external(g, &externals, external_id, nil)

	switch {
	case bytes_differ(image, plain) == 0:
		return "external: the image draws as no map"
	case bytes_differ(bound, image) != 0:
		return fmt.tprintf("external: %d bytes differ from the uploaded image's frame", bytes_differ(bound, image))
	case bytes_differ(unbound, plain) != 0:
		return fmt.tprintf("external: unbound, %d bytes differ from no map", bytes_differ(unbound, plain))
	case bytes_differ(stale, plain) != 0:
		return fmt.tprintf("external: bound for another table, %d bytes differ from no map", bytes_differ(stale, plain))
	}
	return ""
}

// How many bytes of two frames differ (all, when either is missing).
@(private = "file")
bytes_differ :: proc(a, b: []u8) -> int {
	n := 0 if len(a) == len(b) && len(a) > 0 else max(len(a), len(b), 1)
	for i in 0 ..< min(len(a), len(b)) {
		n += 1 if a[i] != b[i] else 0
	}
	return n
}

// ---- prepared jobs ----------------------------------------------------------------------

// Jobs (`prepare`, then `record`) as a host recording at points of its own
// command stream uses them: several cases' scenes on shared tables, each a
// job at its case's size, all prepared, then recorded in reverse into one
// command buffer. Each frame is the same bytes as `render` draws alone;
// recording makes nothing (the GPU's and the jobs' resources are the same
// before and after); a job records once per prepare, and not after another
// job's prepare started the mirrors over for other tables. `err` says what
// differed.
jobs_check :: proc(g: ^GPU) -> (err: string) {
	NAMES :: [?]string{"shadow", "manim", "custom", "textures"}
	N :: len(NAMES)
	cases: [N]Parity_Case
	scenes: [N]o3d.Scene
	a: Parity_Assets
	defer parity_assets_destroy(&a)
	defer for &scene in scenes {
		o3d.scene_destroy(&scene)
	}
	for name, i in NAMES {
		for c in PARITY_CASES {
			if c.name == name {
				cases[i] = c
			}
		}
		if e := cases[i].build(&a); e != "" {
			return fmt.tprintf("jobs: %s: %s", name, e)
		}
		scenes[i], a.scene = a.scene, {}
	}

	outs: [N]^sdl.GPUTexture
	wants: [N][]u8
	defer for i in 0 ..< N {
		if outs[i] != nil {
			sdl.ReleaseGPUTexture(g.device, outs[i])
		}
		delete(wants[i])
	}
	for c, i in cases {
		w, h := u32(c.width), u32(c.height)
		outs[i] = target(g, w, h)
		if e := render(g, &scenes[i], &a.geos, &a.shaders, outs[i], w, h, textures = &a.textures); e != nil {
			return fmt.tprintf("jobs: %s: render: %v", c.name, e)
		}
		wants[i], _ = read(g, outs[i], w, h)
	}

	jobs: [N]Job
	defer for &job in jobs {
		job_destroy(g, &job)
	}
	for c, i in cases {
		if e := prepare(g, &jobs[i], &scenes[i], &a.geos, &a.shaders, u32(c.width), u32(c.height), &a.textures); e != nil {
			return fmt.tprintf("jobs: %s: prepare: %v", c.name, e)
		}
	}
	Made :: struct {
		color, depth, resolved: ^sdl.GPUTexture,
		buffers:                [4]^sdl.GPUBuffer,
		textures, custom:       int,
		job_buffers:            [N][3]^sdl.GPUBuffer,
		shadows:                [N][o3d.MAX_SHADOWS][2]^sdl.GPUTexture,
	}
	made :: proc(g: ^GPU, jobs: ^[N]Job) -> (m: Made) {
		m.color, m.depth, m.resolved = g.color, g.depth, g.resolved
		m.buffers = {g.vertices.buffer, g.skin_vertices.buffer, g.morph_deltas.buffer, g.indices.buffer}
		m.textures, m.custom = len(g.textures), len(g.custom)
		for &job, i in jobs {
			m.job_buffers[i] = {job.dynamic_vertices.buffer, job.dynamic_indices.buffer, job.custom_skinned.buffer}
			for si in 0 ..< o3d.MAX_SHADOWS {
				m.shadows[i][si] = {job.shadow_maps[si], job.shadow_depths[si]}
			}
		}
		return
	}
	before := made(g, &jobs)
	cmd := sdl.AcquireGPUCommandBuffer(g.device)
	if cmd == nil {
		return "jobs: no command buffer"
	}
	for i := N - 1; i >= 0; i -= 1 {
		if e := record(g, &jobs[i], outs[i], cmd); e != nil {
			_ = sdl.CancelGPUCommandBuffer(cmd)
			return fmt.tprintf("jobs: %s: record: %v", cases[i].name, e)
		}
	}
	fence := sdl.SubmitGPUCommandBufferAndAcquireFence(cmd)
	if fence == nil {
		return "jobs: submit failed"
	}
	_ = sdl.WaitForGPUFences(g.device, true, &fence, 1)
	sdl.ReleaseGPUFence(g.device, fence)
	if made(g, &jobs) != before {
		return "jobs: recording made or replaced a GPU resource"
	}
	for c, i in cases {
		got, _ := read(g, outs[i], u32(c.width), u32(c.height))
		defer delete(got)
		if n := bytes_differ(got, wants[i]); n != 0 {
			return fmt.tprintf("jobs: %s: %d bytes differ from render's frame", c.name, n)
		}
	}

	if e := record(g, &jobs[0], outs[0], nil); e != .Not_Prepared {
		return fmt.tprintf("jobs: recorded twice for one prepare (%v)", e)
	}
	other: Parity_Assets
	defer parity_assets_destroy(&other)
	test_scene(&other.scene, &other.geos)
	if e := prepare(g, &jobs[0], &scenes[0], &a.geos, &a.shaders, u32(cases[0].width), u32(cases[0].height), &a.textures); e != nil {
		return fmt.tprintf("jobs: prepare again: %v", e)
	}
	if e := prepare(g, &jobs[1], &other.scene, &other.geos, nil, 160, 120); e != nil {
		return fmt.tprintf("jobs: prepare on other tables: %v", e)
	}
	if e := record(g, &jobs[0], outs[0], nil); e != .Not_Prepared {
		return fmt.tprintf("jobs: recorded after another prepare started the mirrors over (%v)", e)
	}
	return ""
}

// The frame's depth (Scene.depth) on both renderers, over cases with
// ShaderMaterials, an alpha-masked cutout, skins, morphs, exact shapes and
// alpha maps: the GPU's map against the software renderer's, texel by
// texel. Pixel centres on triangle edges may fall on either side of the
// edge on different rasterizers: at most 1 in 2000 texels may differ (on
// Vulkan 0 to 2 of some 30000 do; elsewhere within 3e-6).
depth_check :: proc(g: ^GPU) -> (err: string) {
	NAMES :: [?]string{"shadow", "skinned", "morph", "curves", "alpha_map"}
	for name in NAMES {
		c: Parity_Case
		for pc in PARITY_CASES {
			if pc.name == name {
				c = pc
			}
		}
		a: Parity_Assets
		defer parity_assets_destroy(&a)
		if e := c.build(&a); e != "" {
			return fmt.tprintf("depth: %s: %s", name, e)
		}
		a.scene.depth = true
		w, h := u32(c.width), u32(c.height)
		out := target(g, w, h)
		defer sdl.ReleaseGPUTexture(g.device, out)
		if e := render(g, &a.scene, &a.geos, &a.shaders, out, w, h, textures = &a.textures); e != nil {
			return fmt.tprintf("depth: %s: render: %v", name, e)
		}
		gpu_depth, rerr := read_depth(g, depth(g), w, h)
		defer delete(gpu_depth)
		if rerr != nil || len(gpu_depth) != int(w * h) {
			return fmt.tprintf("depth: %s: no GPU depth", name)
		}
		sw: o3d.Software
		defer o3d.software_destroy(&sw)
		o3d.software_render(&sw, &a.scene, &a.geos, &a.shaders, int(w), int(h), nil, &a.textures)
		cpu_depth := o3d.software_depth(&sw)
		if len(cpu_depth) != len(gpu_depth) {
			return fmt.tprintf("depth: %s: no software depth", name)
		}
		differ, drawn := 0, 0
		for d, i in cpu_depth {
			drawn += 1 if d > 0 else 0
			differ += 1 if abs(d - gpu_depth[i]) > 1e-4 else 0
		}
		if drawn == 0 {
			return fmt.tprintf("depth: %s: nothing drawn", name)
		}
		share := f64(differ) / f64(len(cpu_depth))
		if share > 1.0 / 2000 {
			return fmt.tprintf("depth: %s: %d of %d texels differ", name, differ, len(cpu_depth))
		}
	}
	return ""
}

// MeshPhysicalMaterial's layers under a sky and a sun: a clear coat over
// rough red, sheen on dark blue cloth, both on rough green, and a
// high-ior white beside the default one.
@(private = "file")
build_physical :: proc(a: ^Parity_Assets) -> string {
	sky := o3d.SKY_DEFAULT
	sky.sun = {0.5, 0.35, 1}
	id := o3d.sky_texture(&a.textures, sky, 256)
	o3d.environment_prefilter(&a.textures, id)
	s := &a.scene
	o3d.scene_clear(s)
	s.camera = {position = {0, 0.6, 4.2}, target = {0, 0.1, 0}, fov = 50, near = 0.1, far = 50}
	s.background = id
	s.environment = id
	s.tone_mapping, s.tone_mapping_exposure = .AgX, 0.6
	append(&s.lights, o3d.Light{kind = .Directional, color = {1, 0.95, 0.9}, intensity = 2, position = {2, 3.5, 3}})
	sphere := o3d.sphere(&a.geos, 0.42, 48, 24)
	for k in 0 ..< 5 {
		m := o3d.MATERIAL_DEFAULT
		switch k {
		case 0:
			m.color, m.roughness = {0.7, 0.05, 0.04}, 0.6
			m.clearcoat, m.clearcoat_roughness = 1, 0.05
		case 1:
			m.color, m.roughness = {0.05, 0.08, 0.25}, 0.9
			m.sheen_color, m.sheen_roughness = {1, 0.6, 0.8}, 0.4
		case 2:
			m.color, m.roughness = {0.1, 0.5, 0.15}, 0.8
			m.clearcoat, m.clearcoat_roughness = 0.6, 0.3
			m.sheen_color, m.sheen_roughness = {0.8, 0.9, 0.5}, 0.15
		case 3:
			m.color, m.roughness = {0.9, 0.9, 0.9}, 0.2
		case 4:
			m.color, m.roughness, m.ior = {0.9, 0.9, 0.9}, 0.2, 2.4
		}
		x := -1.8 + 0.9 * f32(k)
		append(&s.draws, o3d.Draw{sphere, o3d.compose({x, 0.1, 0}), m})
	}
	return ""
}

// Transmission: glass spheres before coloured boxes, under a sun and an
// environment: thin and clear, thick (refracting) and tinted by
// absorption, rough (the frame's blurrier mips), and one half over the
// empty (transparent) background, which shows through as alpha.
@(private = "file")
build_transmission :: proc(a: ^Parity_Assets) -> string {
	sky := o3d.SKY_DEFAULT
	sky.sun = {0.5, 0.35, 1}
	id := o3d.sky_texture(&a.textures, sky, 256)
	o3d.environment_prefilter(&a.textures, id)
	s := &a.scene
	o3d.scene_clear(s)
	s.camera = {position = {0, 0.5, 4.2}, target = {0, 0.1, 0}, fov = 50, near = 0.1, far = 50}
	s.environment = id
	s.tone_mapping = .ACES_Filmic
	append(&s.lights, o3d.Light{kind = .Directional, color = {1, 0.95, 0.9}, intensity = 2, position = {2, 3.5, 3}})
	box := o3d.box(&a.geos, 0.5, 1.6, 0.5)
	colors := [4]o3d.Vec3{{0.9, 0.2, 0.1}, {0.1, 0.7, 0.2}, {0.1, 0.3, 0.9}, {0.9, 0.8, 0.1}}
	for k in 0 ..< 4 {
		m := o3d.MATERIAL_DEFAULT
		m.color, m.roughness = colors[k], 0.6
		append(&s.draws, o3d.Draw{box, o3d.compose({-1.5 + 0.9 * f32(k), 0.1, -1.2}), m})
	}
	sphere := o3d.sphere(&a.geos, 0.42, 48, 24)
	for k in 0 ..< 4 {
		m := o3d.MATERIAL_DEFAULT
		m.color, m.roughness, m.transmission = {1, 1, 1}, 0.05, 1
		switch k {
		case 1:
			m.thickness, m.ior = 0.8, 1.5
			m.attenuation_color, m.attenuation_distance = {0.4, 0.8, 1}, 0.5
		case 2:
			m.roughness, m.thickness, m.ior = 0.5, 0.4, 1.4
		case 3:
			m.color, m.thickness, m.ior = {1, 0.85, 0.7}, 0.3, 1.33
		}
		append(&s.draws, o3d.Draw{sphere, o3d.compose({-1.35 + 0.9 * f32(k), 0.1 + (0.75 if k == 3 else 0), 0}), m})
	}
	return ""
}

// The layers' maps: a clear coat in stripes (red) with its roughness
// (green) in the same texture and the coat's own normal map over a bumpy
// base; sheen colour and roughness from one texture; transmission (red)
// and thickness (green) from one texture over coloured boxes; and a third
// distinct texture on one material, which both renderers drop.
@(private = "file")
build_layer_maps :: proc(a: ^Parity_Assets) -> string {
	S :: 32
	make_map :: proc(a: ^Parity_Assets, texel: proc(x, y: int) -> [4]u8) -> o3d.Texture_Id {
		px: [S * S * 4]u8
		for y in 0 ..< S {
			for x in 0 ..< S {
				c := texel(x, y)
				copy(px[(y * S + x) * 4:][:4], c[:])
			}
		}
		return o3d.texture_add(&a.textures, px[:], S, S)
	}
	stripes := make_map(a, proc(x, y: int) -> [4]u8 {return {255 if (x / 4) % 2 == 0 else 40, u8(y * 8), 0, 255}})
	bumps := make_map(a, proc(x, y: int) -> [4]u8 {
		fx, fy := math.sin(f32(x) * 0.8), math.sin(f32(y) * 0.8)
		return {u8(128 + 90 * fx), u8(128 + 90 * fy), 230, 255}
	})
	ripples := make_map(a, proc(x, y: int) -> [4]u8 {
		f := math.sin(f32(x + y) * 0.5)
		return {u8(128 + 70 * f), 128, 240, 255}
	})
	cloth := make_map(a, proc(x, y: int) -> [4]u8 {return {u8(255 - x * 4), 140, u8(y * 8), u8(60 + (x ~ y) * 6)}})
	glass := make_map(a, proc(x, y: int) -> [4]u8 {return {255 if (y / 8) % 2 == 0 else 90, u8(x * 8), 0, 255}})
	sky := o3d.SKY_DEFAULT
	sky.sun = {0.5, 0.35, 1}
	id := o3d.sky_texture(&a.textures, sky, 256)
	o3d.environment_prefilter(&a.textures, id)
	s := &a.scene
	o3d.scene_clear(s)
	s.camera = {position = {0, 0.5, 4.2}, target = {0, 0.1, 0}, fov = 50, near = 0.1, far = 50}
	s.environment = id
	s.tone_mapping = .AgX
	append(&s.lights, o3d.Light{kind = .Directional, color = {1, 0.95, 0.9}, intensity = 2, position = {2, 3.5, 3}})
	box := o3d.box(&a.geos, 0.5, 1.6, 0.5)
	colors := [3]o3d.Vec3{{0.9, 0.2, 0.1}, {0.1, 0.7, 0.2}, {0.1, 0.3, 0.9}}
	for k in 0 ..< 3 {
		m := o3d.MATERIAL_DEFAULT
		m.color, m.roughness = colors[k], 0.6
		append(&s.draws, o3d.Draw{box, o3d.compose({-0.4 + 0.6 * f32(k), 0.1, -1.4}), m})
	}
	sphere := o3d.sphere(&a.geos, 0.42, 48, 24)
	for k in 0 ..< 4 {
		m := o3d.MATERIAL_DEFAULT
		switch k {
		case 0:
			m.color, m.roughness = {0.6, 0.05, 0.05}, 0.5
			m.normal_map, m.normal_scale = bumps, {1.5, 1.5}
			m.clearcoat, m.clearcoat_roughness = 1, 0.6
			m.clearcoat_map, m.clearcoat_roughness_map = stripes, stripes
			m.clearcoat_normal_map = ripples
		case 1:
			m.color, m.roughness = {0.05, 0.08, 0.25}, 0.9
			m.sheen_color, m.sheen_roughness = {1, 1, 1}, 1
			m.sheen_color_map, m.sheen_roughness_map = cloth, cloth
		case 2:
			m.color, m.roughness, m.transmission = {1, 1, 1}, 0.15, 1
			m.thickness, m.attenuation_color, m.attenuation_distance = 0.8, {0.5, 0.8, 1}, 0.6
			m.transmission_map, m.thickness_map = glass, glass
		case 3:
			// Three distinct textures: the third (sheen's) is dropped.
			m.color, m.roughness = {0.2, 0.5, 0.2}, 0.7
			m.clearcoat, m.clearcoat_map, m.clearcoat_normal_map = 0.8, stripes, ripples
			m.sheen_color, m.sheen_color_map = {1, 0.6, 0.8}, cloth
		}
		append(&s.draws, o3d.Draw{sphere, o3d.compose({-1.35 + 0.9 * f32(k), 0.1, 0}), m})
	}
	return ""
}
