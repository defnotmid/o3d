package o3d_sdlgpu

import "core:fmt"
import "core:math"
import "core:testing"

import sdl "vendor:sdl3"

import o3d ".."

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

// The GPU and the software renderer draw the same frame: close everywhere
// (anti-aliasing and float order differ at the edges only).
@(test)
test_gpu_matches_software :: proc(t: ^testing.T) {
	g: GPU
	if init(&g) != nil {
		fmt.println("no GPU device: skipped")
		return
	}
	defer destroy(&g)
	geos: o3d.Geometries
	defer o3d.geometries_destroy(&geos)
	scene: o3d.Scene
	defer o3d.scene_destroy(&scene)
	test_scene(&scene, &geos)
	W, H :: 160, 120
	out := target(&g, W, H)
	testing.expect(t, render(&g, &scene, &geos, nil, out, W, H) == nil)
	gpu, _ := read(&g, out, W, H)
	defer delete(gpu)
	sw: o3d.Software
	defer o3d.software_destroy(&sw)
	cpu := o3d.software_render(&sw, &scene, &geos, nil, W, H)
	// PSNR over the colour channels.
	sum: f64
	for i in 0 ..< len(cpu) {
		d := f64(cpu[i]) - f64(gpu[i])
		sum += d * d
	}
	mse := sum / f64(len(cpu))
	psnr := 10 * math.log10(255 * 255 / max(mse, 1e-9))
	testing.expectf(t, psnr > 32, "PSNR %.1f dB", psnr)
	// Something was drawn.
	opaque := 0
	for i in 0 ..< W * H {
		opaque += 1 if gpu[i * 4 + 3] == 255 else 0
	}
	testing.expectf(t, opaque > W * H / 3, "%d opaque pixels", opaque)
}

// ShaderMaterials of each kind: three.js's GLSL 1 style with uniforms; GLSL 3
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

custom_scene :: proc(t: ^testing.T, scene: ^o3d.Scene, geos: ^o3d.Geometries, shaders: ^o3d.Shaders) {
	test_scene(scene, geos)
	make_shader :: proc(t: ^testing.T, shaders: ^o3d.Shaders, vs, fs: string, glsl3 := false) -> o3d.Shader_Id {
		id, errors := o3d.shader(shaders, vs, fs, glsl3)
		testing.expectf(t, errors == "", "%s", errors)
		return id
	}
	m := o3d.MATERIAL_DEFAULT
	m.shader = make_shader(t, shaders, WAVE_VERT, WAVE_FRAG)
	m.uniforms = o3d.scene_uniforms(scene, {"uTime", {1.3}}, {"uColor", {0.3, 0.8, 0.4}})
	append(&scene.draws, o3d.Draw{o3d.sphere(geos, 0.45, 48, 24), o3d.compose({-1.6, 0.5, 1.2}), m})
	m = o3d.MATERIAL_DEFAULT
	m.shader = make_shader(t, shaders, FACETS_VERT, FACETS_FRAG, true)
	m.uniforms = o3d.scene_uniforms(scene, {"uTints", {0.9, 0.3, 0.3, 0.3, 0.9, 0.3, 0.3, 0.3, 0.9}})
	append(&scene.draws, o3d.Draw{o3d.box(geos, 0.6, 0.6, 0.6), o3d.compose({1.5, 1.4, 0.4}, {0.5, 0.7, 0}), m})
	m = o3d.MATERIAL_DEFAULT
	m.shader = make_shader(t, shaders, RIM_VERT, RIM_FRAG)
	append(&scene.draws, o3d.Draw{o3d.torus(geos, 0.35, 0.12, 16, 40), o3d.compose({-0.9, 1.6, -0.6}, {0.9, 0, 0.3}), m})
	m = o3d.MATERIAL_DEFAULT
	m.shader = make_shader(t, shaders, GLASS_VERT, GLASS_FRAG)
	m.flags, m.side = {.Transparent}, .Double
	m.uniforms = o3d.scene_uniforms(scene, {"uAlpha", {0.55}})
	append(&scene.draws, o3d.Draw{o3d.plane(geos, 1.2, 1.0), o3d.compose({-0.3, 0.9, 1.9}, {0, 2.6, 0}), m})
}

// ShaderMaterials draw the same on the GPU (oksl's SPIR-V / MSL) as on the
// software renderer (oksl's programs over its triangles and quads).
@(test)
test_custom_matches_software :: proc(t: ^testing.T) {
	g: GPU
	if init(&g) != nil {
		fmt.println("no GPU device: skipped")
		return
	}
	defer destroy(&g)
	geos: o3d.Geometries
	defer o3d.geometries_destroy(&geos)
	shaders: o3d.Shaders
	defer o3d.shaders_destroy(&shaders)
	scene: o3d.Scene
	defer o3d.scene_destroy(&scene)
	custom_scene(t, &scene, &geos, &shaders)
	W, H :: 240, 180
	out := target(&g, W, H)
	defer sdl.ReleaseGPUTexture(g.device, out)
	testing.expect(t, render(&g, &scene, &geos, &shaders, out, W, H) == nil)
	gpu, _ := read(&g, out, W, H)
	defer delete(gpu)
	sw: o3d.Software
	defer o3d.software_destroy(&sw)
	cpu := o3d.software_render(&sw, &scene, &geos, &shaders, W, H)
	sum: f64
	worst := 0
	for i in 0 ..< len(cpu) {
		d := f64(cpu[i]) - f64(gpu[i])
		sum += d * d
		worst = max(worst, int(abs(d)))
	}
	mse := sum / f64(len(cpu))
	psnr := 10 * math.log10(255 * 255 / max(mse, 1e-9))
	testing.expectf(t, psnr > 32, "PSNR %.1f dB (worst channel difference %d)", psnr, worst)
	when ODIN_DEBUG {
		fmt.printf("custom materials: PSNR %.1f dB, worst %d\n", psnr, worst)
	}
}

// The custom scene with the key light casting a shadow, every draw casting
// and receiving (ShaderMaterials cast with their plain geometry and do not
// receive).
shadow_scene :: proc(t: ^testing.T, scene: ^o3d.Scene, geos: ^o3d.Geometries, shaders: ^o3d.Shaders) {
	custom_scene(t, scene, geos, shaders)
	for &l in scene.lights {
		if l.kind == .Directional {
			l.cast_shadow = true
			l.shadow = {map_size = 1024, height = 9, bias = -0.0005, normal_bias = 0.01}
		}
	}
	for i in 0 ..< len(scene.draws) {
		scene.draws.material[i].flags += {.Cast_Shadow, .Receive_Shadow}
	}
}

// Shadows draw the same on the GPU as on the software renderer: the map,
// the PCF filter and the shadowed light.
@(test)
test_shadow_matches_software :: proc(t: ^testing.T) {
	g: GPU
	if init(&g) != nil {
		fmt.println("no GPU device: skipped")
		return
	}
	defer destroy(&g)
	geos: o3d.Geometries
	defer o3d.geometries_destroy(&geos)
	shaders: o3d.Shaders
	defer o3d.shaders_destroy(&shaders)
	scene: o3d.Scene
	defer o3d.scene_destroy(&scene)
	shadow_scene(t, &scene, &geos, &shaders)
	W, H :: 240, 180
	out := target(&g, W, H)
	defer sdl.ReleaseGPUTexture(g.device, out)
	testing.expect(t, render(&g, &scene, &geos, &shaders, out, W, H) == nil)
	gpu, _ := read(&g, out, W, H)
	defer delete(gpu)
	sw: o3d.Software
	defer o3d.software_destroy(&sw)
	cpu := o3d.software_render(&sw, &scene, &geos, &shaders, W, H)
	sum: f64
	for i in 0 ..< len(cpu) {
		d := f64(cpu[i]) - f64(gpu[i])
		sum += d * d
	}
	psnr := 10 * math.log10(255 * 255 / max(sum / f64(len(cpu)), 1e-9))
	testing.expectf(t, psnr > 32, "PSNR %.1f dB", psnr)
}

// Material texture maps (color_map, normal_map, roughness_map, emissive_map)
// with mipmaps match between GPU and software rasterizer.
@(test)
test_textures_match_software :: proc(t: ^testing.T) {
	g: GPU
	if init(&g) != nil {
		fmt.println("no GPU device: skipped")
		return
	}
	defer destroy(&g)
	geos: o3d.Geometries
	defer o3d.geometries_destroy(&geos)
	textures: o3d.Textures
	defer o3d.textures_destroy(&textures)
	scene: o3d.Scene
	defer o3d.scene_destroy(&scene)

	TW, TH :: 32, 32
	color_px: [TW * TH * 4]u8
	normal_px: [TW * TH * 4]u8
	rough_px: [TW * TH * 4]u8
	emissive_px: [TW * TH * 4]u8
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
		}
	}
	color_id := o3d.texture_add(&textures, color_px[:], TW, TH)
	normal_id := o3d.texture_add(&textures, normal_px[:], TW, TH)
	rough_id := o3d.texture_add(&textures, rough_px[:], TW, TH)
	emissive_id := o3d.texture_add(&textures, emissive_px[:], TW, TH)

	test_scene(&scene, &geos)
	scene.draws.material[0].color_map = color_id
	scene.draws.material[0].roughness_map = rough_id
	scene.draws.material[1].color_map = color_id
	scene.draws.material[1].normal_map = normal_id
	scene.draws.material[1].roughness_map = rough_id
	scene.draws.material[2].normal_map = normal_id
	scene.draws.material[2].emissive = {1, 1, 1}
	scene.draws.material[2].emissive_intensity = 0.6
	scene.draws.material[2].emissive_map = emissive_id

	W, H :: 200, 150
	out := target(&g, W, H)
	defer sdl.ReleaseGPUTexture(g.device, out)
	testing.expect(t, render(&g, &scene, &geos, nil, out, W, H, textures = &textures) == nil)
	gpu, _ := read(&g, out, W, H)
	defer delete(gpu)
	sw: o3d.Software
	defer o3d.software_destroy(&sw)
	cpu := o3d.software_render(&sw, &scene, &geos, nil, W, H, nil, &textures)
	sum: f64
	for i in 0 ..< len(cpu) {
		d := f64(cpu[i]) - f64(gpu[i])
		sum += d * d
	}
	psnr := 10 * math.log10(255 * 255 / max(sum / f64(len(cpu)), 1e-9))
	testing.expectf(t, psnr > 32, "PSNR %.1f dB", psnr)
	when ODIN_DEBUG {
		fmt.printf("textured materials: PSNR %.1f dB\n", psnr)
	}
}

// Equirectangular environment lighting (prefiltered radiance + diffuse
// irradiance) and blurred background match between GPU and software rasterizer.
@(test)
test_environment_matches_software :: proc(t: ^testing.T) {
	g: GPU
	if init(&g) != nil {
		fmt.println("no GPU device: skipped")
		return
	}
	defer destroy(&g)
	geos: o3d.Geometries
	defer o3d.geometries_destroy(&geos)
	textures: o3d.Textures
	defer o3d.textures_destroy(&textures)
	scene: o3d.Scene
	defer o3d.scene_destroy(&scene)

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
	env_id := o3d.texture_add(&textures, env_px[:], EW, EH)

	test_scene(&scene, &geos)
	clear(&scene.lights) // Pure IBL scene so environment radiance & irradiance dominate!
	scene.environment = env_id
	scene.environment_intensity = 1.2
	scene.background = env_id
	scene.background_blurriness = 0.25

	W, H :: 200, 150
	out := target(&g, W, H)
	defer sdl.ReleaseGPUTexture(g.device, out)
	testing.expect(t, render(&g, &scene, &geos, nil, out, W, H, textures = &textures) == nil)
	gpu, _ := read(&g, out, W, H)
	defer delete(gpu)
	sw: o3d.Software
	defer o3d.software_destroy(&sw)
	cpu := o3d.software_render(&sw, &scene, &geos, nil, W, H, nil, &textures)
	sum: f64
	for i in 0 ..< len(cpu) {
		d := f64(cpu[i]) - f64(gpu[i])
		sum += d * d
	}
	psnr := 10 * math.log10(255 * 255 / max(sum / f64(len(cpu)), 1e-9))
	testing.expectf(t, psnr > 32, "PSNR %.1f dB", psnr)
	when ODIN_DEBUG {
		fmt.printf("environment lighting + background: PSNR %.1f dB\n", psnr)
	}
}

