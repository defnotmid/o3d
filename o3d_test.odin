package o3d

import "core:fmt"
import "core:math"
import "core:math/linalg"
import "core:strings"
import "core:testing"
import "core:thread"

// Every primitive's triangles wind counter-clockwise seen from outside: the
// face normal agrees with the vertex normals. They share one vertex and index
// array, each geometry's indices counting from its first vertex.
@(test)
test_primitive_winding :: proc(t: ^testing.T) {
	g: Geometries
	defer geometries_destroy(&g)
	ids := []Geometry_Id{box(&g, 1, 2, 3, 2, 2, 2), sphere(&g, 1, 16, 8), plane(&g, 2, 1, 3, 2), cylinder(&g, 0.5, 1, 2, 12), cylinder(&g, 0, 1, 1, 8), torus(&g, 1, 0.3, 8, 16)}
	names := []string{"box", "sphere", "plane", "cylinder", "cone", "torus"}
	for id, n in ids {
		vertices, indices, ok := geometry_mesh(&g, id)
		testing.expect(t, ok)
		bad := 0
		for i := 0; i + 2 < len(indices); i += 3 {
			a, b, c := vertices[indices[i]], vertices[indices[i + 1]], vertices[indices[i + 2]]
			face := linalg.cross(b.position - a.position, c.position - a.position)
			if linalg.length(face) < 1e-9 {
				continue // a sphere's pole sliver
			}
			if linalg.dot(face, a.normal + b.normal + c.normal) <= 0 {
				bad += 1
			}
		}
		testing.expectf(t, bad == 0, "%s: %d triangles wound inward", names[n], bad)
	}
	// The same parameters are the same geometry, and the key alone makes it.
	testing.expect_value(t, box(&g, 1, 2, 3, 2, 2, 2), ids[0])
	testing.expect_value(t, primitive(&g, {.Box, {1, 2, 3, 2, 2, 2, 0}}), ids[0])
	testing.expect_value(t, primitive(&g, {.None, {}}), Geometry_Id(0))
}

// The shader's constants are the enums': LIGHT_<KIND> is a Light_Kind's
// value and FLAG_<FLAG> a Shading_Flag's bit.
@(test)
test_shader_constants :: proc(t: ^testing.T) {
	SOURCE :: #load("shaders/o3d.slang", string)
	for kind in Light_Kind {
		line := fmt.tprintf("static const uint LIGHT_%s = %d;", strings.to_upper(fmt.tprint(kind), context.temp_allocator), u32(kind))
		testing.expectf(t, strings.contains(SOURCE, line), "o3d.slang lacks %q", line)
	}
	for flag in Shading_Flag {
		line := fmt.tprintf("static const uint FLAG_%s = %d;", strings.to_upper(fmt.tprint(flag), context.temp_allocator), transmute(u32)Shading_Flags{flag})
		testing.expectf(t, strings.contains(SOURCE, line), "o3d.slang lacks %q", line)
	}
}

// compose matches three.js: a quarter turn about Y takes +X to -Z.
@(test)
test_compose :: proc(t: ^testing.T) {
	m := compose({1, 2, 3}, {0, math.PI / 2, 0}, {2, 2, 2})
	p := m * Vec4{1, 0, 0, 1}
	testing.expectf(t, linalg.length(p.xyz - Vec3{1, 2, 1}) < 1e-5, "got %v", p)
	// The camera at +Z looking at the origin puts the origin in the middle
	// of the view, in front (depth between 0 and 1).
	view, proj := camera_matrices(CAMERA_DEFAULT, 100, 100)
	c := proj * view * Vec4{0, 0, 0, 1}
	testing.expect(t, abs(c.x / c.w) < 1e-6 && abs(c.y / c.w) < 1e-6 && c.z / c.w > 0 && c.z / c.w < 1)
}

// A white sphere lit from the upper left: lit on that side, dark on the
// other, transparent around it.
@(test)
test_software_sphere :: proc(t: ^testing.T) {
	g: Geometries
	defer geometries_destroy(&g)
	scene: Scene
	defer scene_destroy(&scene)
	scene_clear(&scene)
	scene.camera = {position = {0, 0, 4}, fov = 40, near = 0.1, far = 100}
	append(&scene.lights, Light{kind = .Directional, color = {1, 1, 1}, intensity = 3, position = {-1, 1, 1}})
	m := MATERIAL_DEFAULT
	m.roughness = 0.6
	append(&scene.draws, Draw{sphere(&g, 1, 48, 24), compose(), m})
	s: Software
	defer software_destroy(&s)
	W :: 96
	px := software_render(&s, &scene, &g, nil, W, W)
	at :: proc(px: []u8, x, y: int) -> [4]u8 {
		i := (y * W + x) * 4
		return {px[i], px[i + 1], px[i + 2], px[i + 3]}
	}
	testing.expect_value(t, at(px, 2, 2), [4]u8{0, 0, 0, 0})
	lit, dark := at(px, W / 2 - 14, W / 2 - 14), at(px, W / 2 + 22, W / 2 + 22)
	testing.expectf(t, lit[3] == 255 && lit[0] > 150, "lit side %v", lit)
	testing.expectf(t, dark[3] == 255 && dark[0] < 40, "dark side %v", dark)
	// The edge is anti-aliased: pixels on it are partly covered.
	partial := 0
	for i in 0 ..< W * W {
		a := px[i * 4 + 3]
		partial += 1 if a > 0 && a < 255 else 0
	}
	testing.expectf(t, partial > 50, "%d partly covered pixels", partial)
}

// A transparent box in front of an opaque one shows both.
@(test)
test_software_transparent :: proc(t: ^testing.T) {
	g: Geometries
	defer geometries_destroy(&g)
	scene: Scene
	defer scene_destroy(&scene)
	scene_clear(&scene)
	red, glass := MATERIAL_DEFAULT, MATERIAL_DEFAULT
	red.flags, red.color = {.Basic}, {1, 0, 0}
	glass.flags, glass.color, glass.opacity = {.Basic, .Transparent}, {0, 0, 1}, 0.5
	// The glass is listed first: it is still drawn after the opaque box.
	append(&scene.draws, Draw{plane(&g, 1, 1), compose({0, 0, 1}), glass})
	append(&scene.draws, Draw{plane(&g, 1, 1), compose(), red})
	s: Software
	defer software_destroy(&s)
	px := software_render(&s, &scene, &g, nil, 64, 64)
	i := (32 * 64 + 32) * 4
	testing.expectf(t, px[i] > 100 && px[i + 2] > 100 && px[i + 3] == 255, "%v", px[i:i + 4])
}

// Drawn on a pool, a frame is the same bytes as drawn on the caller: the
// chunks (a sphere of more than CHUNK_TRIS triangles makes several) and the
// bands keep the drawing order, blending included.
@(test)
test_software_pool :: proc(t: ^testing.T) {
	g: Geometries
	defer geometries_destroy(&g)
	scene: Scene
	defer scene_destroy(&scene)
	scene_clear(&scene)
	scene.camera = {position = {0, 1, 5}, fov = 50, near = 0.1, far = 100}
	append(&scene.lights, Light{kind = .Directional, color = {1, 1, 1}, intensity = 2.5, position = {2, 3, 2}})
	append(&scene.lights, Light{kind = .Hemisphere, color = {0.8, 0.9, 1}, ground = {0.3, 0.2, 0.2}, intensity = 1})
	big := sphere(&g, 1.2, 128, 64)
	vertices, indices, _ := geometry_mesh(&g, big)
	testing.expect(t, len(indices) / 3 > CHUNK_TRIS && len(vertices) > 0)
	solid, glass := MATERIAL_DEFAULT, MATERIAL_DEFAULT
	solid.color, solid.roughness = {0.9, 0.4, 0.3}, 0.4
	glass.color, glass.opacity, glass.flags, glass.side = {0.3, 0.6, 1}, 0.5, {.Transparent}, .Double
	append(&scene.draws, Draw{big, compose(), solid})
	for i in 0 ..< 12 {
		x := f32(i % 4) - 1.5
		y := f32(i / 4) - 0.5
		append(&scene.draws, Draw{box(&g, 0.5, 0.5, 0.5), compose({x, y, 1.5}, {0.3, f32(i), 0}), glass if i % 3 == 0 else solid})
	}
	W, H :: 200, 150
	alone, pooled: Software
	defer software_destroy(&alone)
	defer software_destroy(&pooled)
	want := software_render(&alone, &scene, &g, nil, W, H)
	pool: thread.Pool
	thread.pool_init(&pool, context.allocator, 4)
	thread.pool_start(&pool)
	defer {
		thread.pool_join(&pool)
		thread.pool_destroy(&pool)
	}
	for _ in 0 ..< 2 { // the second frame reuses the chunks' lists
		got := software_render(&pooled, &scene, &g, nil, W, H, &pool)
		differ := 0
		for i in 0 ..< len(want) {
			differ += 1 if got[i] != want[i] else 0
		}
		testing.expectf(t, differ == 0, "%d bytes differ", differ)
	}
}

// draw_order keeps the draws that can be seen: with geometry, their bounding
// sphere (scaled with the draw) reaching into the view.
@(test)
test_draw_order_culls :: proc(t: ^testing.T) {
	g: Geometries
	defer geometries_destroy(&g)
	scene: Scene
	defer scene_destroy(&scene)
	scene_clear(&scene) // the camera at +5 Z looking at the origin
	ball := sphere(&g, 1, 8, 4)
	m, glass := MATERIAL_DEFAULT, MATERIAL_DEFAULT
	glass.opacity, glass.flags = 0.5, {.Transparent}
	append(&scene.draws, Draw{ball, compose(), m}) // in view
	append(&scene.draws, Draw{ball, compose({0, 0, 10}), m}) // behind the camera
	append(&scene.draws, Draw{ball, compose({20, 0, 0}), m}) // far to the side
	append(&scene.draws, Draw{99, compose(), m}) // no such geometry
	append(&scene.draws, Draw{ball, compose({4, 0, 0}, {}, {3, 3, 3}), m}) // off centre, scaled into view
	append(&scene.draws, Draw{ball, compose({0, 0, 1}), glass}) // in view, blended: last
	view, proj := camera_matrices(scene.camera, 100, 100)
	order: [dynamic]int
	defer delete(order)
	draw_order(&scene, &g, nil, view, proj, &order)
	testing.expectf(t, slice_equal(order[:], []int{0, 4, 5}), "order %v", order)
}

@(private = "file")
slice_equal :: proc(a, b: []int) -> bool {
	if len(a) != len(b) {
		return false
	}
	for x, i in a {
		if x != b[i] {
			return false
		}
	}
	return true
}

// three.js-style ShaderMaterial sources (GLSL 1 style, the default).
WAVE_VERT :: `varying vec3 vNormal;
varying vec2 vUv;
uniform float uTime;
void main() {
  vNormal = normalize(normalMatrix * normal);
  vUv = uv;
  vec3 p = position + normal * 0.08 * sin(uTime + position.y * 6.0);
  gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
}`

WAVE_FRAG :: `varying vec3 vNormal;
varying vec2 vUv;
uniform vec3 uColor;
void main() {
  float l = 0.35 + 0.65 * max(dot(normalize(vNormal), normalize(vec3(0.4, 0.8, 1.0))), 0.0);
  gl_FragColor = vec4(uColor * l + vec3(vUv * 0.15, 0.0), 1.0);
}`

// Shaders compile once per source, report errors with their place, and
// reject what o3d does not provide.
@(test)
test_shader_table :: proc(t: ^testing.T) {
	shaders: Shaders
	defer shaders_destroy(&shaders)
	id, errors := shader(&shaders, WAVE_VERT, WAVE_FRAG)
	testing.expectf(t, id != 0 && errors == "", "errors: %s", errors)
	again, _ := shader(&shaders, WAVE_VERT, WAVE_FRAG)
	testing.expect_value(t, again, id)
	sh := shader_get(&shaders, id)
	// modelMatrix is declared by the prelude but unused: absent; the
	// vertex stage reads every attribute.
	testing.expect_value(t, sh.builtins[.Vertex][.Model_Matrix], -1)
	testing.expect(t, sh.builtins[.Vertex][.Normal_Matrix] >= 0 && sh.builtins[.Vertex][.Model_View_Matrix] >= 0)
	testing.expect(t, sh.attributes[.Position] >= 0 && sh.attributes[.Uv] >= 0)
	testing.expect_value(t, sh.varying_floats, 5)

	cases := []struct {
		vertex, fragment, want: string,
	} {
		{WAVE_VERT, "void main() { gl_FragColor = vec4(undefinedThing); }", "Fragment:1:"},
		{"attribute vec3 color; void main() { gl_Position = vec4(color, 1.0); }", "void main() { gl_FragColor = vec4(1.0); }", "attribute color"},
		{WAVE_VERT, "uniform sampler2D map; varying vec2 vUv; void main() { gl_FragColor = texture2D(map, vUv); }", "textures"},
	}
	for c in cases {
		bad, errs := shader(&shaders, c.vertex, c.fragment)
		testing.expectf(t, bad != 0 && shader_get(&shaders, bad) == nil && strings.contains(errs, c.want), "want %q in %q", c.want, errs)
		// A failure is kept: asked again, the same errors without compiling.
		_, errs2 := shader(&shaders, c.vertex, c.fragment)
		testing.expect_value(t, errs2, errs)
	}
}

// A ShaderMaterial on the software renderer: drawn where the geometry is,
// with its uniforms; the same bytes on a pool; a draw whose shader failed
// is left out.
@(test)
test_software_shader_material :: proc(t: ^testing.T) {
	g: Geometries
	defer geometries_destroy(&g)
	shaders: Shaders
	defer shaders_destroy(&shaders)
	scene: Scene
	defer scene_destroy(&scene)
	scene_clear(&scene)
	scene.camera = {position = {0, 0, 4}, fov = 40, near = 0.1, far = 100}
	wave, errors := shader(&shaders, WAVE_VERT, WAVE_FRAG)
	testing.expectf(t, wave != 0, "%s", errors)
	broken, _ := shader(&shaders, WAVE_VERT, "void main() { nope; }")
	m := MATERIAL_DEFAULT
	m.shader = wave
	m.uniforms = scene_uniforms(&scene, {"uTime", {0.5}}, {"uColor", {0.2, 0.9, 0.3}}, {"notInTheShader", {1}})
	append(&scene.draws, Draw{sphere(&g, 1, 96, 48), compose(), m}) // > CHUNK_TRIS triangles
	m.shader = broken
	append(&scene.draws, Draw{box(&g, 4, 4, 0.1), compose({0, 0, -1}), m})
	W :: 96
	s: Software
	defer software_destroy(&s)
	px := software_render(&s, &scene, &g, &shaders, W, W)
	at :: proc(px: []u8, x, y: int) -> [4]u8 {
		i := (y * W + x) * 4
		return {px[i], px[i + 1], px[i + 2], px[i + 3]}
	}
	// Green from uColor in the middle; nothing drawn in the corner (the
	// broken shader's box behind would cover it).
	c := at(px, W / 2, W / 2)
	testing.expectf(t, c[3] == 255 && c[1] > 150 && c[0] < 120, "centre %v", c)
	testing.expect_value(t, at(px, 2, 2), [4]u8{0, 0, 0, 0})
	pool: thread.Pool
	thread.pool_init(&pool, context.allocator, 4)
	thread.pool_start(&pool)
	defer {
		thread.pool_join(&pool)
		thread.pool_destroy(&pool)
	}
	pooled: Software
	defer software_destroy(&pooled)
	got := software_render(&pooled, &scene, &g, &shaders, W, W, &pool)
	differ := 0
	for i in 0 ..< len(px) {
		differ += 1 if got[i] != px[i] else 0
	}
	testing.expectf(t, differ == 0, "%d bytes differ on a pool", differ)
}

// A long box above two floors, seen from straight above and lit at a tilt
// by a shadow-casting light: its shadow falls beside it (up the screen) on
// the floor that receives (left), not on the one that does not (right),
// with soft edges; a pool draws the same bytes.
@(test)
test_software_shadow :: proc(t: ^testing.T) {
	g: Geometries
	defer geometries_destroy(&g)
	scene: Scene
	defer scene_destroy(&scene)
	scene_clear(&scene)
	scene.camera = {position = {0, 8, 0.01}, fov = 60, near = 0.1, far = 100}
	append(&scene.lights, Light{kind = .Ambient, color = {1, 1, 1}, intensity = 0.3})
	append(&scene.lights, Light{kind = .Directional, color = {1, 1, 1}, intensity = 3, position = {0, 5, 5}, cast_shadow = true, shadow = {map_size = 256, height = 10, bias = -0.001}})
	floor, other, block := MATERIAL_DEFAULT, MATERIAL_DEFAULT, MATERIAL_DEFAULT
	floor.flags = {.Receive_Shadow}
	block.flags = {.Cast_Shadow}
	append(&scene.draws, Draw{plane(&g, 3, 6), compose({-1.6, 0, 0}, {-math.PI / 2, 0, 0}), floor})
	append(&scene.draws, Draw{plane(&g, 3, 6), compose({1.6, 0, 0}, {-math.PI / 2, 0, 0}), other})
	append(&scene.draws, Draw{box(&g, 4.8, 0.3, 1.2), compose({0, 1.5, 0}), block})
	W :: 128
	s: Software
	defer software_destroy(&s)
	px := software_render(&s, &scene, &g, nil, W, W)
	luma :: proc(px: []u8, x, y: int) -> int {
		i := (y * W + x) * 4
		return int(px[i]) + int(px[i + 1]) + int(px[i + 2])
	}
	SHADOW_Y, OPEN_Y :: 42, 24 // the shadow's band and open floor, above the box
	LEFT, RIGHT :: 44, 88
	shadowed, open := luma(px, LEFT, SHADOW_Y), luma(px, LEFT, OPEN_Y)
	testing.expectf(t, open > 0 && shadowed * 2 < open, "receiving floor: shadow %d, open %d", shadowed, open)
	testing.expect_value(t, luma(px, RIGHT, SHADOW_Y), luma(px, RIGHT, OPEN_Y))
	// PCF: the band's edge is between the two.
	edge := luma(px, LEFT, 32)
	testing.expectf(t, edge > shadowed && edge < open, "edge %d between %d and %d", edge, shadowed, open)

	pool: thread.Pool
	thread.pool_init(&pool, context.allocator, 4)
	thread.pool_start(&pool)
	defer {
		thread.pool_join(&pool)
		thread.pool_destroy(&pool)
	}
	pooled: Software
	defer software_destroy(&pooled)
	got := software_render(&pooled, &scene, &g, nil, W, W, &pool)
	differ := 0
	for i in 0 ..< len(px) {
		differ += 1 if got[i] != px[i] else 0
	}
	testing.expectf(t, differ == 0, "%d bytes differ on a pool", differ)
}

// Textures build a full mipmap chain in one shared array and `sample` picks
// the level from screen uv derivatives: magnified reads the fine checker,
// minified reads the averaged top level.
@(test)
test_texture_mipmaps :: proc(t: ^testing.T) {
	textures: Textures
	defer textures_destroy(&textures)
	N :: 16
	px: [N * N * 4]u8
	for y in 0 ..< N {
		for x in 0 ..< N {
			i := (y * N + x) * 4
			v := u8(240) if (x + y) % 2 == 0 else u8(20)
			px[i], px[i + 1], px[i + 2], px[i + 3] = v, v, v, 255
		}
	}
	id := texture_add(&textures, px[:], N, N)
	tex, ok := texture_get(&textures, id)
	testing.expect(t, ok && tex.levels == 5 && tex.texel_count == 16 * 16 + 8 * 8 + 4 * 4 + 2 * 2 + 1)
	// Magnified (zero derivatives -> level 0): texel centres return 240 and 20.
	c0 := sample(&textures, id, {0.5 / N, 1 - 0.5 / N}, {}, {})
	c1 := sample(&textures, id, {1.5 / N, 1 - 0.5 / N}, {}, {})
	testing.expectf(t, c0.r > 0.9 && c1.r < 0.1, "level 0 texels: %v, %v", c0, c1)
	// Minified (1 uv unit per pixel -> level 4, 1x1 average ~ 130/255).
	avg := sample(&textures, id, {0.25, 0.75}, {1, 0}, {0, 1})
	testing.expectf(t, abs(avg.r - 130.0 / 255.0) < 0.02, "minified average: %v", avg)
}

// Standard material textures on the software renderer: color_map (with
// v = 1 at the top of the mesh reading row 0 of the image), normal_map
// (screen-derivative tangents tilting normals toward +X on the left and -X
// on the right), roughness_map, and emissive_map; byte-identical on a pool.
@(test)
test_software_textures :: proc(t: ^testing.T) {
	g: Geometries
	defer geometries_destroy(&g)
	textures: Textures
	defer textures_destroy(&textures)
	scene: Scene
	defer scene_destroy(&scene)
	scene_clear(&scene)
	scene.camera = {position = {0, 0, 3.5}, fov = 45, near = 0.1, far = 100}
	// Light from +X (right): surfaces whose normal tilts toward +X are brighter.
	append(&scene.lights, Light{kind = .Directional, color = {1, 1, 1}, intensity = 3.0, position = {4, 0, 2}})

	T :: 32
	col_px, norm_px, emissive_px: [T * T * 4]u8
	for y in 0 ..< T {
		for x in 0 ..< T {
			i := (y * T + x) * 4
			// Top half red, bottom half green.
			if y < T / 2 {
				col_px[i], col_px[i + 1], col_px[i + 2], col_px[i + 3] = 240, 30, 30, 255
			} else {
				col_px[i], col_px[i + 1], col_px[i + 2], col_px[i + 3] = 30, 240, 30, 255
			}
			// Left half tilts normal toward +X (R=220), right half toward -X (R=35).
			nx := u8(220) if x < T / 2 else u8(35)
			norm_px[i], norm_px[i + 1], norm_px[i + 2], norm_px[i + 3] = nx, 128, 200, 255
			// Emissive spot in the bottom-right quadrant.
			if x >= T / 2 && y >= T / 2 {
				emissive_px[i], emissive_px[i + 1], emissive_px[i + 2], emissive_px[i + 3] = 0, 0, 250, 255
			} else {
				emissive_px[i + 3] = 255
			}
		}
	}
	col_id := texture_add(&textures, col_px[:], T, T)
	norm_id := texture_add(&textures, norm_px[:], T, T)
	em_id := texture_add(&textures, emissive_px[:], T, T)

	m := MATERIAL_DEFAULT
	m.color_map = col_id
	m.normal_map = norm_id
	m.emissive = {1, 1, 1}
	m.emissive_map = em_id
	append(&scene.draws, Draw{plane(&g, 2, 2), compose(), m})

	W :: 96
	s: Software
	defer software_destroy(&s)
	px := software_render(&s, &scene, &g, nil, W, W, nil, &textures)
	at :: proc(px: []u8, x, y: int) -> [4]u8 {
		i := (y * W + x) * 4
		return {px[i], px[i + 1], px[i + 2], px[i + 3]}
	}
	top_left := at(px, W / 2 - 16, W / 2 - 16)
	top_right := at(px, W / 2 + 16, W / 2 - 16)
	bot_left := at(px, W / 2 - 16, W / 2 + 16)
	bot_right := at(px, W / 2 + 16, W / 2 + 16)
	// Top is red, bottom is green; left tilts toward the +X light so it is brighter than right;
	// bottom-right has blue emission.
	testing.expectf(t, top_left[0] > top_left[1] * 2, "top_left red: %v", top_left)
	testing.expectf(t, top_left[0] > top_right[0] + 40, "normal map +X vs -X: left %v, right %v", top_left, top_right)
	testing.expectf(t, bot_left[1] > bot_left[0] * 2, "bot_left green: %v", bot_left)
	testing.expectf(t, bot_right[2] > 200, "bot_right blue emissive: %v", bot_right)

	pool: thread.Pool
	thread.pool_init(&pool, context.allocator, 4)
	thread.pool_start(&pool)
	defer {
		thread.pool_join(&pool)
		thread.pool_destroy(&pool)
	}
	pooled: Software
	defer software_destroy(&pooled)
	got := software_render(&pooled, &scene, &g, nil, W, W, &pool, &textures)
	differ := 0
	for i in 0 ..< len(px) {
		differ += 1 if got[i] != px[i] else 0
	}
	testing.expectf(t, differ == 0, "%d bytes differ on a pool", differ)
}

// Equirectangular environment lighting and background: with no lights at all,
// a metal sphere reflects the panorama (sharper at low roughness, blurred at
// high roughness), a dielectric sphere receives diffuse irradiance, the
// background draws behind them, and a pool is byte-identical.
@(test)
test_software_environment :: proc(t: ^testing.T) {
	// Round-trip equirect_dir / equirect_uv.
	for uv in ([3]Vec2{{0.25, 0.75}, {0.5, 0.5}, {0.8, 0.2}}) {
		rt := equirect_uv(equirect_dir(uv))
		testing.expectf(t, linalg.length(rt - uv) < 1e-5, "equirect round-trip %v -> %v", uv, rt)
	}

	g: Geometries
	defer geometries_destroy(&g)
	textures: Textures
	defer textures_destroy(&textures)
	scene: Scene
	defer scene_destroy(&scene)

	// 64x32 equirectangular panorama: blue sky (+Y, top rows), warm ground
	// (-Y, bottom rows), bright sun at +Z (reflected by the center of a sphere
	// seen from +Z).
	EW, EH :: 64, 32
	env_px: [EW * EH * 4]u8
	for y in 0 ..< EH {
		v := 1 - (f32(y) + 0.5) / f32(EH)
		for x in 0 ..< EW {
			u := (f32(x) + 0.5) / f32(EW)
			dir := equirect_dir({u, v})
			sky := clamp(dir.y * 0.5 + 0.5, 0, 1)
			col := linalg.lerp(Vec3{0.55, 0.25, 0.1}, Vec3{0.15, 0.45, 0.95}, sky)
			sun := math.pow(max(linalg.dot(dir, Vec3{0, 0, 1}), 0), 32)
			col = linalg.clamp(col + Vec3{0.85, 0.85, 0.7} * sun, 0, 1)
			i := (y * EW + x) * 4
			env_px[i + 0] = srgb8(col.r)
			env_px[i + 1] = srgb8(col.g)
			env_px[i + 2] = srgb8(col.b)
			env_px[i + 3] = 255
		}
	}
	env_id := texture_add(&textures, env_px[:], EW, EH)

	scene_clear(&scene)
	scene.camera = {position = {0, 0, 4}, fov = 40, near = 0.1, far = 100}
	mirror := MATERIAL_DEFAULT
	mirror.metalness, mirror.roughness = 1, 0.06
	rough_metal := MATERIAL_DEFAULT
	rough_metal.metalness, rough_metal.roughness = 1, 0.85
	matte := MATERIAL_DEFAULT
	matte.metalness, matte.roughness = 0, 1

	W :: 96
	s: Software
	defer software_destroy(&s)
	at :: proc(px: []u8, x, y: int) -> [4]u8 {
		i := (y * W + x) * 4
		return {px[i], px[i + 1], px[i + 2], px[i + 3]}
	}

	// 1. Without environment or lights, the metal sphere is completely black.
	append(&scene.draws, Draw{sphere(&g, 1, 32, 16), compose(), mirror})
	px_dark := software_render(&s, &scene, &g, nil, W, W, nil, &textures)
	testing.expect_value(t, at(px_dark, W / 2, W / 2), [4]u8{0, 0, 0, 255})

	// 2. With scene.environment and scene.background, mirror metal reflects the
	// +Z sun at its center, blue sky above, warm ground below, and the background
	// is opaque.
	scene.environment = env_id
	scene.background = env_id
	px_mirror := software_render(&s, &scene, &g, nil, W, W, nil, &textures)
	center_mirror := at(px_mirror, W / 2, W / 2)
	top_mirror := at(px_mirror, W / 2, W / 2 - 14)
	bot_mirror := at(px_mirror, W / 2, W / 2 + 14)
	bg_corner := at(px_mirror, 4, 4)
	testing.expectf(t, center_mirror[0] > 200 && center_mirror[1] > 200, "mirror sun reflection: %v", center_mirror)
	testing.expectf(t, top_mirror[2] > top_mirror[0], "mirror upper sky blue: %v", top_mirror)
	testing.expectf(t, bot_mirror[0] > bot_mirror[2], "mirror lower ground warm: %v", bot_mirror)
	testing.expectf(t, bg_corner[3] == 255 && bg_corner[2] > 100, "opaque background corner: %v", bg_corner)

	// 3. Rough metal blurs the sun highlight so the peak center brightness is lower.
	scene.draws.material[0] = rough_metal
	px_rough := software_render(&s, &scene, &g, nil, W, W, nil, &textures)
	center_rough := at(px_rough, W / 2, W / 2)
	testing.expectf(t, center_rough[0] > 80 && center_rough[0] < center_mirror[0] - 20, "rough metal softer than mirror: rough=%v mirror=%v", center_rough, center_mirror)

	// 4. Matte dielectric sphere is lit by diffuse irradiance (sky on top, ground on bottom).
	scene.draws.material[0] = matte
	px_matte := software_render(&s, &scene, &g, nil, W, W, nil, &textures)
	top_matte := at(px_matte, W / 2, W / 2 - 16)
	bot_matte := at(px_matte, W / 2, W / 2 + 16)
	testing.expectf(t, top_matte[2] > bot_matte[2] && bot_matte[0] > top_matte[0] - 15, "matte irradiance top=%v bot=%v", top_matte, bot_matte)

	// 5. Pool renders byte-identical pixels.
	pool: thread.Pool
	thread.pool_init(&pool, context.allocator, 4)
	thread.pool_start(&pool)
	defer {
		thread.pool_join(&pool)
		thread.pool_destroy(&pool)
	}
	pooled: Software
	defer software_destroy(&pooled)
	got := software_render(&pooled, &scene, &g, nil, W, W, &pool, &textures)
	differ := 0
	for i in 0 ..< len(px_matte) {
		differ += 1 if got[i] != px_matte[i] else 0
	}
	testing.expectf(t, differ == 0, "%d bytes differ on a pool", differ)
}



// The same parameters every frame make nothing new; a parameter that varies
// makes a mesh per value, which geometries_made shows (scale instead).
@(test)
test_geometries_made :: proc(t: ^testing.T) {
	g: Geometries
	defer geometries_destroy(&g)
	for frame in 0 ..< 3 {
		_ = sphere(&g, 1, 16, 8)
		_ = box(&g, 1, 2, 3)
		_ = frame
	}
	testing.expect_value(t, geometries_made(&g), 2)
	for frame in 0 ..< 3 {
		_ = sphere(&g, 1 + f32(frame) * 0.1, 16, 8) // a radius animated the wrong way
	}
	testing.expect_value(t, geometries_made(&g), 4) // 1.0 was made already; 1.1, 1.2 are new
}

// An environment's derived maps are made by environment_prefilter, once, at
// load: a render after it adds nothing to the textures (no work in frame
// time); one without it derives them on first use.
@(test)
test_environment_prefilter :: proc(t: ^testing.T) {
	textures: Textures
	defer textures_destroy(&textures)
	W, H :: 64, 32
	pixels: [W * H * 4]u8
	for i in 0 ..< W * H {
		pixels[i * 4 + 0], pixels[i * 4 + 1], pixels[i * 4 + 2], pixels[i * 4 + 3] = u8(i % W * 4), 120, u8(i / W * 8), 255
	}
	env := texture_add(&textures, pixels[:], W, H)
	_, _, before := texture_environment(&textures, env)
	testing.expect(t, !before, "not derived until asked")
	rad, irr, ok := environment_prefilter(&textures, env)
	testing.expect(t, ok && rad != 0 && irr != 0)
	r2, i2, found := texture_environment(&textures, env)
	testing.expect(t, found && r2 == rad && i2 == irr, "looked up as derived")
	again, _, _ := environment_prefilter(&textures, env)
	testing.expect_value(t, again, rad) // once

	g: Geometries
	defer geometries_destroy(&g)
	scene: Scene
	defer scene_destroy(&scene)
	scene_clear(&scene)
	scene.environment = env
	m := MATERIAL_DEFAULT
	m.metalness, m.roughness = 1, 0.3
	append(&scene.draws, Draw{sphere(&g, 1, 16, 8), compose(), m})
	texels, count := len(textures.texels), len(textures.list)
	s: Software
	defer software_destroy(&s)
	_ = software_render(&s, &scene, &g, nil, 32, 32, nil, &textures)
	testing.expect(t, len(textures.texels) == texels && len(textures.list) == count, "a render after prefiltering derives nothing")
}

// A table's generation is taken with its first entry, and a table made
// after one was destroyed (same address, same ids) gets a new one.
@(test)
test_generations :: proc(t: ^testing.T) {
	g: Geometries
	testing.expect_value(t, g.generation, Generation(0))
	_ = box(&g)
	first := g.generation
	testing.expect(t, first != 0)
	_ = sphere(&g)
	testing.expect_value(t, g.generation, first)
	geometries_destroy(&g)
	_ = box(&g)
	defer geometries_destroy(&g)
	testing.expect(t, g.generation != 0 && g.generation != first, "a new table, a new generation")
}

// Alpha_Mask discards fragments and shadow casters below alpha_cutoff (cut-out
// shadows); ao_map occludes indirect light only (ambient/hemisphere/environment,
// not direct lights) and reads uv1 when tex_coord == 1; Texture_Transform shifts,
// rotates, and scales UVs.
@(test)
test_alpha_mask_ao_and_uv_transform :: proc(t: ^testing.T) {
	g: Geometries
	defer geometries_destroy(&g)
	textures: Textures
	defer textures_destroy(&textures)
	scene: Scene
	defer scene_destroy(&scene)
	s: Software
	defer software_destroy(&s)

	// 1. Cut-out shadow and main-pass discard: a 32x32 texture whose left half has
	//    alpha = 0 and right half has alpha = 255.
	T :: 32
	mask_px: [T * T * 4]u8
	for y in 0 ..< T {
		for x in 0 ..< T {
			i := (y * T + x) * 4
			if x >= T / 2 {
				mask_px[i], mask_px[i + 1], mask_px[i + 2], mask_px[i + 3] = 220, 60, 60, 255
			}
		}
	}
	mask_id := texture_add(&textures, mask_px[:], T, T)

	scene_clear(&scene)
	scene.camera = {position = {0, 0, 4}, fov = 45, near = 0.1, far = 50}
	append(&scene.lights, Light{kind = .Ambient, color = {1, 1, 1}, intensity = 0.25})
	append(&scene.lights, Light{
		kind        = .Directional,
		color       = {1, 1, 1},
		intensity   = 3,
		position    = {0, 0, 5},
		cast_shadow = true,
		shadow      = {map_size = 256, height = 4, bias = -0.002},
	})
	floor := MATERIAL_DEFAULT
	floor.color, floor.flags = {0.8, 0.8, 0.8}, {.Receive_Shadow}
	cutout := MATERIAL_DEFAULT
	cutout.color_map = mask_id
	cutout.alpha_cutoff = 0.5
	cutout.flags = {.Alpha_Mask, .Cast_Shadow}
	cutout.side = .Double
	append(&scene.draws, Draw{plane(&g, 3, 3), compose({0, 0, -1}), floor})
	append(&scene.draws, Draw{plane(&g, 2, 2), compose({0, 0, 0}), cutout})

	W :: 64
	px := software_render(&s, &scene, &g, nil, W, W, nil, &textures)
	// Left side of cutout (x = 22, y = 32): discarded in both passes -> floor is directly lit by directional light!
	// Right side of cutout (x = 42, y = 32): opaque red cutout.
	left_px := px[(32 * W + 22) * 4:][:4]
	right_px := px[(32 * W + 42) * 4:][:4]
	testing.expectf(t, left_px[1] > 180 && left_px[2] > 180, "left half discarded in main+shadow passes, floor stays bright: %v", left_px)
	testing.expectf(t, right_px[0] > 180 && right_px[1] < 100, "right half opaque red cutout: %v", right_px)

	// Now view the floor directly (move cutout toward light so it only casts shadow on right half of floor):
	scene.draws.model[1] = compose({0, 0, 2.5}) // behind camera's near or between light and floor
	scene.camera.position = {0, 0, 1.5}
	px_sh := software_render(&s, &scene, &g, nil, W, W, nil, &textures)
	floor_left := px_sh[(32 * W + 20) * 4]
	floor_right := px_sh[(32 * W + 44) * 4]
	testing.expectf(t, floor_left > floor_right + 60, "cut-out shadow: left floor (%d) unshadowed, right floor (%d) shadowed", floor_left, floor_right)

	// 2. AO map on UV1 occludes indirect light (Ambient) only, leaving direct light intact,
	//    and Texture_Transform offset shifts UVs.
	ao_px: [T * T * 4]u8
	for y in 0 ..< T {
		for x in 0 ..< T {
			i := (y * T + x) * 4
			// UV1 u < 0.5 has AO = 0 (fully occluded indirect), u >= 0.5 has AO = 255 (unoccluded).
			r := u8(0) if x < T / 2 else u8(255)
			ao_px[i], ao_px[i + 1], ao_px[i + 2], ao_px[i + 3] = r, 255, 255, 255
		}
	}
	ao_id := texture_add(&textures, ao_px[:], T, T)

	// Custom quad where uv is constant {0, 0} (so UV0 would always read AO=0), while uv1 spans [0, 1] horizontally.
	q_verts := [4]Vertex {
		{{-1, -1, 0}, {0, 0, 1}, {0, 0}, {0, 0}},
		{{ 1, -1, 0}, {0, 0, 1}, {0, 0}, {1, 0}},
		{{ 1,  1, 0}, {0, 0, 1}, {0, 0}, {1, 1}},
		{{-1,  1, 0}, {0, 0, 1}, {0, 0}, {0, 1}},
	}
	q_idx := [6]u32{0, 1, 2, 0, 2, 3}
	quad_id := geometry_add(&g, q_verts[:], q_idx[:])

	scene_clear(&scene)
	scene.camera = {position = {0, 0, 3}, fov = 45, near = 0.1, far = 50}
	append(&scene.lights, Light{kind = .Ambient, color = {1, 1, 1}, intensity = 2.0})
	ao_mat := MATERIAL_DEFAULT
	ao_mat.ao_map = ao_id
	ao_mat.ao_map_intensity = 1.0
	ao_mat.ao_transform = {scale = {1, 1}, tex_coord = 1}
	append(&scene.draws, Draw{quad_id, compose(), ao_mat})
	px_ao := software_render(&s, &scene, &g, nil, W, W, nil, &textures)
	testing.expectf(t, px_ao[(32 * W + 20) * 4] < 10 && px_ao[(32 * W + 44) * 4] > 200, "AO on UV1 occludes left ambient (%d) and keeps right (%d)", px_ao[(32 * W + 20) * 4], px_ao[(32 * W + 44) * 4])

	// With direct light only (no ambient), AO map does not darken the left half.
	clear(&scene.lights)
	append(&scene.lights, Light{kind = .Directional, color = {1, 1, 1}, intensity = 2.5, position = {0, 0, 5}})
	px_dir := software_render(&s, &scene, &g, nil, W, W, nil, &textures)
	testing.expectf(t, abs(int(px_dir[(32 * W + 20) * 4]) - int(px_dir[(32 * W + 44) * 4])) <= 2, "AO does not occlude direct light")

	// Shifting UV1 by offset = {0.5, 0} swaps the two halves under ambient light.
	clear(&scene.lights)
	append(&scene.lights, Light{kind = .Ambient, color = {1, 1, 1}, intensity = 2.0})
	scene.draws.material[0].ao_transform.offset = {0.5, 0}
	px_shift := software_render(&s, &scene, &g, nil, W, W, nil, &textures)
	testing.expectf(t, px_shift[(32 * W + 20) * 4] > 200 && px_shift[(32 * W + 44) * 4] < 10, "KHR_texture_transform offset={0.5,0} swaps AO halves")
}

