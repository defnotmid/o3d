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
	ids := []Geometry_Id{box(&g, 1, 2, 3, 2, 2, 2), sphere(&g, 1, 16, 8), plane(&g, 2, 1, 3, 2), cylinder(&g, 0.5, 1, 2, 12), cylinder(&g, 0, 1, 1, 8), torus(&g, 1, 0.3, 8, 16), capsule(&g, 0.5, 1, 4, 8)}
	names := []string{"box", "sphere", "plane", "cylinder", "cone", "torus", "capsule"}
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
		{"uniform sampler2D map; void main() { gl_Position = texture2D(map, vec2(0.5)); }", "void main() { gl_FragColor = vec4(1.0); }", "fragment shaders only"},
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

// An external texture (pixels bound on the GPU) has no texels: the software
// renderer samples it white, draws its material as with no map, and makes no
// environment maps from it; images added after it still sample their own.
@(test)
test_external_texture_software :: proc(t: ^testing.T) {
	textures: Textures
	defer textures_destroy(&textures)
	ext := texture_add_external(&textures, 64, 32)
	px := [4]u8{200, 10, 10, 255}
	img := texture_add(&textures, px[:], 1, 1)
	tex, ok := texture_get(&textures, ext)
	testing.expect(t, ok && tex.external && tex.levels == 0 && tex.width == 64 && tex.height == 32)
	testing.expect(t, sample(&textures, ext, {0.3, 0.6}, {1, 0}, {0, 1}) == {1, 1, 1, 1})
	testing.expect(t, sample_lod(&textures, ext, {0.3, 0.6}, 2) == {})
	testing.expect(t, sample(&textures, img, {0.5, 0.5}, {}, {}).r > 0.7)
	_, _, prefiltered := environment_prefilter(&textures, ext)
	testing.expect(t, !prefiltered)

	g: Geometries
	defer geometries_destroy(&g)
	scene: Scene
	defer scene_destroy(&scene)
	scene_clear(&scene)
	scene.camera = {position = {0, 0, 3}, fov = 45, near = 0.1, far = 100}
	append(&scene.lights, Light{kind = .Directional, color = {1, 1, 1}, intensity = 2, position = {1, 2, 3}})
	m := MATERIAL_DEFAULT
	append(&scene.draws, Draw{plane(&g, 2, 2), compose(), m})
	W :: 48
	alone, mapped: Software
	defer software_destroy(&alone)
	defer software_destroy(&mapped)
	want := software_render(&alone, &scene, &g, nil, W, W, nil, &textures)
	scene.draws.material[0].color_map = ext
	got := software_render(&mapped, &scene, &g, nil, W, W, nil, &textures)
	differ := 0
	for i in 0 ..< len(want) {
		differ += 1 if got[i] != want[i] else 0
	}
	testing.expectf(t, differ == 0, "%d bytes differ from no map", differ)
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

// Tone mapping modes (Linear, Reinhard, Cineon, ACES_Filmic, AgX, Neutral) with
// exposure compress overbright HDR values into [0, 1], and Fog (Linear and Exp2)
// blends shaded draws toward fog.color by camera distance without touching
// background pixels.
@(test)
test_tone_mapping_and_fog :: proc(t: ^testing.T) {
	hdr := Vec3{4.0, 2.0, 0.5}
	testing.expect_value(t, tone_map(hdr, .None, 1.0), hdr)
	testing.expect_value(t, tone_map(Vec3{0.3, 0.2, 0.1}, .Linear, 2.0), Vec3{0.6, 0.4, 0.2})
	for mode in Tone_Mapping {
		if mode == .None || mode == .Linear {
			continue
		}
		mapped := tone_map(hdr, mode, 1.0)
		testing.expectf(t, mapped.r > 0.5 && mapped.r < 1.0 && mapped.g > 0.3 && mapped.g < 1.0, "%v compresses HDR into (0, 1): %v", mode, mapped)
		brighter := tone_map(hdr, mode, 2.0)
		testing.expectf(t, brighter.r > mapped.r && brighter.g > mapped.g, "%v exposure increases brightness: %v vs %v", mode, brighter, mapped)
	}

	g: Geometries
	defer geometries_destroy(&g)
	scene: Scene
	defer scene_destroy(&scene)
	s: Software
	defer software_destroy(&s)

	W :: 64
	scene_clear(&scene)
	scene.camera = {position = {0, 0, 5}, fov = 45, near = 0.1, far = 50}
	m := MATERIAL_DEFAULT
	m.color = {1, 0, 0}
	m.flags = {.Basic}
	box_id := box(&g, 0.8, 0.8, 0.2)

	// Linear fog from dist = 2 to dist = 10 toward pure blue {0, 0, 1}.
	scene.fog = {kind = .Linear, color = {0, 0, 1}, near = 2, far = 10}
	append(&scene.draws, Draw{box_id, compose({0, 0, 3}), m}) // dist ~ 2
	px_near := software_render(&s, &scene, &g, nil, W, W)
	near_r, near_b := px_near[(32 * W + 32) * 4], px_near[(32 * W + 32) * 4 + 2]
	bg_a := px_near[(2 * W + 2) * 4 + 3]

	scene.draws.model[0] = compose({0, 0, -1}) // dist ~ 6
	px_mid := software_render(&s, &scene, &g, nil, W, W)
	mid_r, mid_b := px_mid[(32 * W + 32) * 4], px_mid[(32 * W + 32) * 4 + 2]

	scene.draws.model[0] = compose({0, 0, -6}) // dist ~ 11
	px_far := software_render(&s, &scene, &g, nil, W, W)
	far_r, far_b := px_far[(32 * W + 32) * 4], px_far[(32 * W + 32) * 4 + 2]

	testing.expectf(t, near_r > 230 && near_b < 60, "near box un-fogged red: r=%d b=%d", near_r, near_b)
	testing.expectf(t, mid_r > 120 && mid_b > 120, "mid box blended red+blue: r=%d b=%d", mid_r, mid_b)
	testing.expectf(t, far_b > 240 && far_r < 20, "far box fogged blue: r=%d b=%d", far_r, far_b)
	testing.expect_value(t, bg_a, u8(0))

	// Exp2 fog increases monotonically with distance.
	scene.fog = {kind = .Exp2, color = {0, 1, 0}, density = 0.18}
	scene.draws.model[0] = compose({0, 0, 3})
	g_near := software_render(&s, &scene, &g, nil, W, W)[(32 * W + 32) * 4 + 1]
	scene.draws.model[0] = compose({0, 0, -1})
	g_mid := software_render(&s, &scene, &g, nil, W, W)[(32 * W + 32) * 4 + 1]
	scene.draws.model[0] = compose({0, 0, -6})
	g_far := software_render(&s, &scene, &g, nil, W, W)[(32 * W + 32) * 4 + 1]
	testing.expectf(t, g_near < g_mid && g_mid < g_far, "Exp2 fog green channel increases with distance: %d < %d < %d", g_near, g_mid, g_far)
}

// SpotLight cone falloff (inner/outer penumbra), perspective spot shadow map,
// and multiple simultaneous shadow-casting lights (Directional + Spot).
@(test)
test_spot_light_and_multi_shadows :: proc(t: ^testing.T) {
	g: Geometries
	defer geometries_destroy(&g)
	scene: Scene
	defer scene_destroy(&scene)
	s: Software
	defer software_destroy(&s)

	W :: 64
	scene_clear(&scene)
	scene.camera = {position = {0, 0, 4}, fov = 45, near = 0.1, far = 50}
	// Spot light at (0, 0, 3) aimed at (0, 0, 0) with narrow cone angle = 20 deg, penumbra = 0.5.
	append(&scene.lights, Light{
		kind      = .Spot,
		color     = {1, 1, 1},
		intensity = 15,
		position  = {0, 0, 3},
		target    = {0, 0, 0},
		angle     = 20 * math.RAD_PER_DEG,
		penumbra  = 0.5,
	})
	floor := MATERIAL_DEFAULT
	floor.color, floor.roughness = {0.9, 0.9, 0.9}, 0.8
	floor.flags = {.Receive_Shadow}
	append(&scene.draws, Draw{plane(&g, 6, 6), compose(), floor})

	px := software_render(&s, &scene, &g, nil, W, W)
	center_r := px[(32 * W + 32) * 4]
	edge_r := px[(32 * W + 4) * 4]
	testing.expectf(t, center_r > 160 && edge_r == 0, "spot light illuminates center (%d) and cuts off outside cone (%d)", center_r, edge_r)

	// Two shadow-casting lights at once:
	// Light 0: Red Directional light at (-0.6, 0, 5) -> (-0.6, 0, 0), blocked at (-0.6, 0, 2.5)
	// Light 1: Blue Spot light at (+0.6, 0, 4) -> (+0.6, 0, 0), blocked at (+0.6, 0, 2.5)
	// Camera at (0, 0, 2.0) looking at (0, 0, 0), so blockers at z = 2.5 are behind camera:
	// - Left floor (-0.6, 0, 0): occluded from Red Directional, lit by Blue Spot -> blue > red
	// - Right floor (+0.6, 0, 0): occluded from Blue Spot, lit by Red Directional -> red > blue
	// - Center floor (0, 0, 0): unoccluded from both -> both red and blue bright!
	clear(&scene.lights)
	append(&scene.lights, Light{
		kind        = .Directional,
		color       = {1, 0, 0},
		intensity   = 4,
		position    = {0, 0, 5},
		target      = {0, 0, 0},
		cast_shadow = true,
		shadow      = {map_size = 256, height = 4, bias = -0.002},
	})
	append(&scene.lights, Light{
		kind        = .Spot,
		color       = {0, 0, 1},
		intensity   = 40,
		position    = {0.6, 0, 4},
		target      = {0.6, 0, 0},
		angle       = 55 * math.RAD_PER_DEG,
		penumbra    = 0.2,
		cast_shadow = true,
		shadow      = {map_size = 256, bias = -0.002},
	})
	caster := MATERIAL_DEFAULT
	caster.flags = {.Cast_Shadow}
	scene.camera.position = {0, 0, 2.0}
	append(&scene.draws, Draw{box(&g, 0.35, 0.35, 0.1), compose({-0.5, 0, 2.6}), caster})
	append(&scene.draws, Draw{box(&g, 0.35, 0.35, 0.1), compose({ 0.6, 0, 2.6}), caster})
	px_two := software_render(&s, &scene, &g, nil, W, W)
	left_px := px_two[(32 * W + 16) * 4:][:4]
	center_px := px_two[(32 * W + 32) * 4:][:4]
	right_px := px_two[(32 * W + 48) * 4:][:4]
	testing.expectf(t, left_px[0] < 20 && left_px[2] > 100, "left floor shadowed from red directional, lit by blue spot: %v", left_px)
	testing.expectf(t, right_px[2] < 20 && right_px[0] > 100, "right floor shadowed from blue spot, lit by red directional: %v", right_px)
	testing.expectf(t, center_px[0] > 100 && center_px[2] > 100, "center floor lit by both lights: %v", center_px)
}

// InstancedMesh (per-instance transform + color), LineSegments (line_segments,
// polyline, wireframe), and Points (world-space and pixel-space).
@(test)
test_instancing_lines_and_points :: proc(t: ^testing.T) {
	g: Geometries
	defer geometries_destroy(&g)
	scene: Scene
	defer scene_destroy(&scene)
	s: Software
	defer software_destroy(&s)

	W :: 64
	scene_clear(&scene)
	scene.camera = {position = {0, 0, 4}, fov = 45, near = 0.1, far = 50}
	m := MATERIAL_DEFAULT
	m.color = {1, 1, 1}
	m.flags = {.Basic}

	// 1. InstancedMesh: one draw with 2 instances (left red, right green).
	append(&scene.draws, Draw{box(&g, 0.8, 0.8, 0.2), compose(), m})
	insts := [2]Instance {
		{transform = compose({-0.8, 0, 0}), color = {1, 0, 0}, has_color = true},
		{transform = compose({ 0.8, 0, 0}), color = {0, 1, 0}, has_color = true},
	}
	scene_set_draw_instances(&scene, 0, scene_instances(&scene, insts[:]))
	px := software_render(&s, &scene, &g, nil, W, W)
	left_px := px[(32 * W + 18) * 4:][:4]
	right_px := px[(32 * W + 46) * 4:][:4]
	testing.expectf(t, left_px[0] > 220 && left_px[1] < 20, "left instance is red: %v", left_px)
	testing.expectf(t, right_px[1] > 220 && right_px[0] < 20, "right instance is green: %v", right_px)

	// 2. LineSegments and Points: horizontal line across y = 0.5 and point at (0, -0.5, 0) in pixel space.
	scene_clear(&scene)
	scene.camera = {position = {0, 0, 4}, fov = 45, near = 0.1, far = 50}
	line_mat := MATERIAL_DEFAULT
	line_mat.color = {1, 0.8, 0}
	line_mat.size = 6 // 6 pixels thick
	line_mat.flags = {.Basic, .No_Size_Attenuation}
	seg_id := line_segments(&g, {{-1, 0.5, 0}, {1, 0.5, 0}})
	append(&scene.draws, Draw{seg_id, compose(), line_mat})

	pt_mat := MATERIAL_DEFAULT
	pt_mat.color = {0, 0.8, 1}
	pt_mat.size = 10 // 10 pixels wide
	pt_mat.flags = {.Basic, .No_Size_Attenuation}
	pt_id := points(&g, {{0, -0.5, 0}})
	append(&scene.draws, Draw{pt_id, compose(), pt_mat})

	// Also verify polyline and wireframe geometry creation.
	poly_id := polyline(&g, {{-1, 0, 0}, {0, 1, 0}, {1, 0, 0}}, closed = true)
	wire_id := wireframe(&g, box(&g, 1, 1, 1))
	poly_geo, poly_ok := geometry_get(&g, poly_id)
	wire_geo, wire_ok := geometry_get(&g, wire_id)
	testing.expect(t, poly_ok && poly_geo.mode == .Lines && poly_geo.index_count == 6)
	testing.expect(t, wire_ok && wire_geo.mode == .Lines && wire_geo.index_count == 60)

	px_lp := software_render(&s, &scene, &g, nil, W, W)
	// Line at y = 0.5 projects near row y = 22, x = 32; Point at y = -0.5 projects near row y = 42, x = 32.
	line_hit := px_lp[(22 * W + 32) * 4:][:4]
	pt_hit := px_lp[(42 * W + 32) * 4:][:4]
	testing.expectf(t, line_hit[0] > 200 && line_hit[1] > 150, "pixel-sized line segment rendered at (32, 22): %v", line_hit)
	testing.expectf(t, pt_hit[2] > 200 && pt_hit[1] > 150, "pixel-sized point rendered at (32, 42): %v", pt_hit)
}

// Verifies MeshToonMaterial (`.Toon` 3-band quantized cel-shading) and
// inverted-hull cartoon outlines (`Material.outline_width`, `Material.outline_color`).
@(test)
test_toon_shading_and_outlines :: proc(t: ^testing.T) {
	// 1. Direct `shade()` test: within each of the 3 cel-shading plateaus
	// (shadow: nl in 0..0.10, midtone: nl in 0.22..0.46, lit: nl in 0.62..0.80)
	// the diffuse shading is flat/quantized, with sharp steps between plateaus.
	sm := Shading_Material{
		base      = {0.8, 0.4, 0.2},
		opacity   = 1,
		roughness = 0.9, // high roughness to isolate 3-band diffuse steps
		flags     = {.Toon},
	}
	lights := [1]Shading_Light{
		{
			kind   = .Directional,
			color  = {math.PI, math.PI, math.PI},
			vector = {0, 0, 1},
		},
	}
	c_shadow_a := shade(&sm, {}, {0, 0, -1}, {0, 0, 5}, lights[:]) // nl = 0.0
	c_shadow_b := shade(&sm, {}, linalg.normalize(Vec3{1, 0, 0.08}), {0, 0, 5}, lights[:]) // nl ~ 0.08
	c_mid_a    := shade(&sm, {}, linalg.normalize(Vec3{1, 0, 0.30}), {0, 0, 5}, lights[:]) // nl ~ 0.29
	c_mid_b    := shade(&sm, {}, linalg.normalize(Vec3{1, 0, 0.45}), {0, 0, 5}, lights[:]) // nl ~ 0.41
	c_lit_a    := shade(&sm, {}, linalg.normalize(Vec3{1, 0, 0.90}), {0, 0, 5}, lights[:]) // nl ~ 0.67
	c_lit_b    := shade(&sm, {}, {0, 0, 1}, {0, 0, 5}, lights[:]) // nl = 1.0

	testing.expect(t, linalg.length(c_shadow_a - c_shadow_b) < 1e-5, "shadow band plateau is flat")
	testing.expect(t, linalg.length(c_mid_a - c_mid_b) < 1e-5, "midtone band plateau is flat")
	testing.expect(t, linalg.length(c_lit_a - c_lit_b) < 1e-5, "lit band plateau is flat")
	testing.expect(t, c_mid_a.r > c_shadow_a.r * 2.5 && c_lit_a.r > c_mid_a.r * 1.35, "distinct quantized steps between shadow, midtone, and lit bands")

	// 2. Inverted-hull outline rendering test: render a sphere with and without
	// a bright green outline (`outline_width = 0.12`, `outline_color = {0, 1, 0}`)
	// and verify the silhouette ring is filled with the outline color while the
	// interior remains the sphere's base color.
	g: Geometries
	defer geometries_destroy(&g)
	scene: Scene
	defer scene_destroy(&scene)
	s: Software
	defer software_destroy(&s)

	W :: 64
	sph := sphere(&g, 0.8, 24, 16)
	scene_clear(&scene)
	scene.camera = {position = {0, 0, 3}, fov = 45, near = 0.1, far = 50}
	append(&scene.lights, Light{kind = .Directional, color = {1, 1, 1}, intensity = 3, position = {2, 3, 4}})

	mat := MATERIAL_DEFAULT
	mat.color = {1, 0.1, 0.1}
	mat.flags = {.Toon}
	append(&scene.draws, Draw{sph, compose(), mat})
	px_first := software_render(&s, &scene, &g, nil, W, W)
	px_no_outline := make([]u8, len(px_first), context.temp_allocator)
	copy(px_no_outline, px_first)

	// Enable outline on the draw.
	scene.draws.material[0].outline_width = 0.12
	scene.draws.material[0].outline_color = {0, 1, 0}
	px_outline := software_render(&s, &scene, &g, nil, W, W)

	// Center pixel (32, 32) is red in both; silhouette ring has green outline pixels where px_no_outline was transparent.
	center_px := px_outline[(32 * W + 32) * 4:][:4]
	testing.expectf(t, center_px[0] > 150 && center_px[1] < 80, "sphere interior remains red: %v", center_px)

	outline_ring_pixels := 0
	for i in 0 ..< W * W {
		if px_no_outline[i * 4 + 3] == 0 && px_outline[i * 4 + 3] == 255 {
			if px_outline[i * 4 + 1] > 200 && px_outline[i * 4 + 0] < 20 {
				outline_ring_pixels += 1
			}
		}
	}
	testing.expectf(t, outline_ring_pixels > 40, "inverted-hull outline forms a silhouette ring around the sphere (%d pixels)", outline_ring_pixels)
}




// A tube is per-frame triangles that expansion hands the renderers as
// dynamic geometry, wound outward; an instanced ShaderMaterial draw becomes a
// draw per instance whose instanceColor and instanceId are that instance's.
@(test)
test_tube_and_instance_builtins :: proc(t: ^testing.T) {
	g: Geometries
	defer geometries_destroy(&g)
	shaders: Shaders
	defer shaders_destroy(&shaders)
	scene: Scene
	defer scene_destroy(&scene)
	exp: Expanded_Draws
	defer expanded_draws_destroy(&exp)

	tube := scene_tube(&scene, {{-1, 0, 0}, {0, 1, 0}, {1, 0, 0}}, 0.1, 8)
	testing.expect(t, tube >= SCENE_GEO_BIT && tube < DYNAMIC_GEO_BIT)
	geo := scene.path_geos[tube - SCENE_GEO_BIT]
	verts := scene.path_vertices[geo.first_vertex:][:geo.vertex_count]
	indices := scene.path_indices[geo.first_index:][:geo.index_count]
	bad := 0
	for i := 0; i + 2 < len(indices); i += 3 {
		a, b, c := verts[indices[i]], verts[indices[i + 1]], verts[indices[i + 2]]
		if linalg.dot(linalg.cross(b.position - a.position, c.position - a.position), a.normal + b.normal + c.normal) < 0 {
			bad += 1
		}
	}
	testing.expectf(t, bad == 0, "tube: %d triangles wound inward", bad)
	append(&scene.draws, Draw{tube, compose(), MATERIAL_DEFAULT})

	id, errs := shader(&shaders, "void main() { gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }", "void main() { gl_FragColor = vec4(instanceColor * (instanceId + 1.0), 1.0); }")
	testing.expectf(t, id != 0 && errs == "", "errors: %s", errs)
	m := MATERIAL_DEFAULT
	m.shader = id
	append(&scene.draws, Draw{capsule(&g, 0.5, 1), compose(), m})
	scene_set_draw_instances(&scene, 1, scene_instances(&scene, {{transform = compose()}, {transform = compose({2, 0, 0}), color = {1, 0, 0}, has_color = true}}))

	view, proj := linalg.MATRIX4F32_IDENTITY, linalg.MATRIX4F32_IDENTITY
	out := expand_draws(&exp, &scene, &g, view, proj, 64, 64)
	testing.expect_value(t, len(out.draws), 3)
	testing.expect(t, out.draws[0].geometry >= DYNAMIC_GEO_BIT)

	blocks: Shader_Blocks
	defer shader_blocks_destroy(&blocks)
	shader_blocks(out, &shaders, view, proj, &blocks)
	sh := shader_get(&shaders, id)
	color_at, id_at := sh.builtins[.Fragment][.Instance_Color], sh.builtins[.Fragment][.Instance_Id]
	testing.expect(t, color_at >= 0 && id_at >= 0)
	want := [2]struct {
		color: Vec3,
		id:    f32,
	}{{{1, 1, 1}, 0}, {{1, 0, 0}, 1}}
	for w, k in want {
		block := blocks.bytes[blocks.at[1 + k][.Fragment]:]
		color := (^Vec3)(&block[color_at])^
		index := (^f32)(&block[id_at])^
		testing.expectf(t, linalg.length(color - w.color) < 1e-5 && index == w.id, "instance %d: instanceColor %v, instanceId %v", k, color, index)
	}
}

// A ShaderMaterial's sampler2D: found by name, and read as three.js reads
// it (uv (0.25, 0.75) is the image's top-left quarter; texelFetch's (0, 1)
// the top row's first texel). A vertex shader may not sample.
@(test)
test_shader_maps :: proc(t: ^testing.T) {
	shaders: Shaders
	defer shaders_destroy(&shaders)
	g: Geometries
	defer geometries_destroy(&g)
	textures: Textures
	defer textures_destroy(&textures)
	VERT :: `void main() { gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`
	FRAG :: `uniform sampler2D other;
uniform sampler2D map;
uniform float uFetch;
void main() {
  vec4 a = texture2D(map, vec2(0.25, 0.75));
  vec4 b = texelFetch(map, ivec2(0, 1), 0);
  gl_FragColor = vec4(mix(a.rgb, b.rgb, uFetch) * texture2D(other, vec2(0.5)).rgb, 1.0);
}`
	id, errors := shader(&shaders, VERT, FRAG)
	testing.expect_value(t, errors, "")
	testing.expect_value(t, shader_map(&shaders, id, "map"), 1)
	testing.expect_value(t, shader_map(&shaders, id, "other"), 0)
	testing.expect_value(t, shader_map(&shaders, id, "nope"), -1)
	_, verrors := shader(&shaders, `uniform sampler2D map; void main() { gl_Position = texture2D(map, vec2(0.5)); }`, `void main() { gl_FragColor = vec4(1.0); }`)
	testing.expect(t, verrors != "", "a vertex shader's sampler is refused")

	// 2x2, rows from the top: red, green / blue, white.
	px := [16]u8{255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255}
	tex := texture_add(&textures, px[:], 2, 2)
	scene: Scene
	defer scene_destroy(&scene)
	for fetch in ([]f32{0, 1}) {
		scene_clear(&scene)
		scene.camera = {position = {0, 0, 1}, fov = 90, near = 0.1, far = 10}
		m := MATERIAL_DEFAULT
		m.shader = id
		m.shader_maps[shader_map(&shaders, id, "map")] = tex
		m.uniforms = scene_uniforms(&scene, {"uFetch", {fetch}})
		append(&scene.draws, Draw{plane(&g, 4, 4), compose({0, 0, 0}), m})
		s: Software
		out := software_render(&s, &scene, &g, &shaders, 8, 8, nil, &textures)
		// `other` has no texture: white.
		testing.expectf(t, out[(4 * 8 + 4) * 4] == 255 && out[(4 * 8 + 4) * 4 + 1] < 8, "fetch %v: the top-left texel (red), got %v", fetch, out[(4 * 8 + 4) * 4:][:4])
		software_destroy(&s)
	}
}
