package o3d_sdlgpu

import "core:encoding/base64"
import "core:fmt"
import "core:math"

import sdl "vendor:sdl3"

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
		if errors != "" {
			return errors
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
		{{-0.7, -0.5, 0}, {0, 0, 1}, {0, 0}, {0, 1}},
		{{ 0.7, -0.5, 0}, {0, 0, 1}, {1, 0}, {1, 0}},
		{{ 0.7,  0.5, 0}, {0, 0, 1}, {1, 1}, {0, 0}},
		{{-0.7,  0.5, 0}, {0, 0, 1}, {0, 1}, {1, 1}},
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
	if id == 0 || errors != "" {
		return errors
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
