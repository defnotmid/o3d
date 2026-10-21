package o3d

import "core:fmt"
import "core:math"
import "core:math/linalg"
import "core:slice"
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
		{{-1, -1, 0}, {0, 0, 1}, {0, 0}, {0, 0}, {}},
		{{ 1, -1, 0}, {0, 0, 1}, {0, 0}, {1, 0}, {}},
		{{ 1,  1, 0}, {0, 0, 1}, {0, 0}, {1, 1}, {}},
		{{-1,  1, 0}, {0, 0, 1}, {0, 0}, {0, 1}, {}},
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
	shader_blocks(out, &shaders, view, proj, nil, &blocks)
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

// A transparent material whose only transparency is its alpha map blends:
// where the map's green is 0, what is behind shows through.
@(test)
test_alpha_map_blends :: proc(t: ^testing.T) {
	g: Geometries
	defer geometries_destroy(&g)
	textures: Textures
	defer textures_destroy(&textures)
	scene: Scene
	defer scene_destroy(&scene)
	clear_px := [4]u8{0, 0, 0, 255} // green 0: fully transparent
	clear_id := texture_add(&textures, clear_px[:], 1, 1)
	scene_clear(&scene)
	scene.camera = {position = {0, 0, 2}, fov = 60, near = 0.1, far = 10}
	behind, front := MATERIAL_DEFAULT, MATERIAL_DEFAULT
	behind.color, behind.flags = {1, 0, 0}, {.Basic}
	front.color, front.flags, front.alpha_map = {1, 1, 1}, {.Basic, .Transparent}, clear_id
	append(&scene.draws, Draw{plane(&g, 4, 4), compose({0, 0, -1}), behind})
	append(&scene.draws, Draw{plane(&g, 4, 4), compose({0, 0, 0}), front})
	s: Software
	defer software_destroy(&s)
	px := software_render(&s, &scene, &g, nil, 8, 8, nil, &textures)
	c := px[(4 * 8 + 4) * 4:][:4]
	testing.expectf(t, c[0] > 240 && c[1] < 16 && c[2] < 16, "the red plane behind shows through: %v", c)
}

// HDR textures: a Radiance file decodes (by odecs) to linear half floats
// past 1, their mipmaps average in float, an environment's radiance and
// irradiance keep values past 1, and sampling reads them linear.
@(test)
test_hdr_textures :: proc(t: ^testing.T) {
	textures: Textures
	defer textures_destroy(&textures)
	// 2x2 flat RGBE: (1, 0.5, 0.25), then 128, 128, 128 three times.
	hdr := "#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 2 +X 2\n"
	data := make([dynamic]u8, context.temp_allocator)
	append(&data, hdr)
	append(&data, 128, 64, 32, 129)
	for _ in 0 ..< 3 {
		append(&data, 128, 128, 128, 136)
	}
	id, ok := texture_decode(&textures, data[:])
	testing.expect(t, ok, "the .hdr decodes")
	tex, _ := texture_get(&textures, id)
	testing.expect(t, tex.hdr && tex.width == 2 && tex.levels == 2, "an HDR texture with its mip")
	top := textures.hdr_texels[tex.mips[0]]
	testing.expect_value(t, [3]f32{f32(top.r), f32(top.g), f32(top.b)}, [3]f32{1, 0.5, 0.25})
	mip := textures.hdr_texels[tex.mips[1]]
	testing.expect(t, abs(f32(mip.r) - (1 + 3 * 128) / 4.0) < 0.5, "the mip averages in float")
	// Linear on reading: no sRGB decode.
	c := sample_lod_linear(&textures, id, {0.25, 0.75}, 0)
	testing.expect(t, abs(c.r - 1) < 1e-3 && abs(c.b - 0.25) < 1e-3, "the top-left texel, linear")

	// An environment at 100 everywhere: its maps stay at 100.
	EW, EH :: 16, 8
	env: [EW * EH * 4]f32
	for &v in env {
		v = 100
	}
	env_id := texture_add_hdr(&textures, env[:], EW, EH)
	rad, irr, eok := environment_prefilter(&textures, env_id)
	testing.expect(t, eok, "prefiltered")
	rt, _ := texture_get(&textures, rad)
	it, _ := texture_get(&textures, irr)
	testing.expect(t, rt.hdr && it.hdr, "HDR maps")
	testing.expect(t, abs(sample_lod_linear(&textures, irr, {0.5, 0.5}, 0).g - 100) < 0.5, "irradiance 100")
	testing.expect(t, abs(sample_lod_linear(&textures, rad, {0.3, 0.6}, 3).r - 100) < 0.5, "a rough radiance level 100")
}

// A shadow-only material (three.js's ShadowMaterial) draws nothing where
// it is lit and its colour at its opacity where a caster's shadow falls
// (test_software_shadow's box, light and bands).
@(test)
test_shadow_only :: proc(t: ^testing.T) {
	g: Geometries
	defer geometries_destroy(&g)
	scene: Scene
	defer scene_destroy(&scene)
	scene_clear(&scene)
	scene.camera = {position = {0, 8, 0.01}, fov = 60, near = 0.1, far = 100}
	append(&scene.lights, Light{kind = .Directional, color = {1, 1, 1}, intensity = 3, position = {0, 5, 5}, cast_shadow = true, shadow = {map_size = 256, height = 10, bias = -0.001}})
	under, catcher, block := MATERIAL_DEFAULT, MATERIAL_DEFAULT, MATERIAL_DEFAULT
	under.color, under.flags = {1, 0, 0}, {.Basic}
	catcher.color, catcher.opacity, catcher.flags = {0, 0, 1}, 0.5, {.Shadow_Only}
	block.flags = {.Cast_Shadow, .Basic}
	append(&scene.draws, Draw{plane(&g, 6, 6), compose({0, -0.01, 0}, {-math.PI / 2, 0, 0}), under})
	append(&scene.draws, Draw{plane(&g, 6, 6), compose({0, 0, 0}, {-math.PI / 2, 0, 0}), catcher})
	append(&scene.draws, Draw{box(&g, 4.8, 0.3, 1.2), compose({0, 1.5, 0}), block})
	W :: 128
	s: Software
	defer software_destroy(&s)
	px := software_render(&s, &scene, &g, nil, W, W)
	at :: proc(px: []u8, x, y: int) -> [4]u8 {
		i := (y * W + x) * 4
		return {px[i], px[i + 1], px[i + 2], px[i + 3]}
	}
	SHADOW_Y, OPEN_Y, X :: 42, 24, 44
	lit := at(px, X, OPEN_Y)
	testing.expectf(t, lit.r > 250 && lit.b < 4, "lit: nothing drawn over the red, got %v", lit)
	shade := at(px, X, SHADOW_Y)
	testing.expectf(t, shade.b > 120 && shade.r > 100 && shade.r < 220, "shadowed: blue at half over red, got %v", shade)
}

// camera_project puts a point where the renderers draw it: a small sphere's
// centre projects onto its pixels; a point behind the camera is not in
// front; nearer points have smaller depths.
@(test)
test_camera_project :: proc(t: ^testing.T) {
	g: Geometries
	defer geometries_destroy(&g)
	scene: Scene
	defer scene_destroy(&scene)
	W, H :: 96, 64
	cameras := [2]Camera{{position = {3, 2, 5}, target = {0, 0.5, 0}, fov = 45, near = 0.1, far = 50}, {position = {0, 6, 6}, target = {0, 0, 0}, orthographic = true, height = 8, near = 0.1, far = 50}}
	for cam in cameras {
		scene_clear(&scene)
		scene.camera = cam
		dot := MATERIAL_DEFAULT
		dot.color, dot.flags = {1, 0, 0}, {.Basic}
		at := Vec3{1.2, 0.8, -0.7}
		append(&scene.draws, Draw{sphere(&g, 0.08, 12, 8), compose(at), dot})
		s: Software
		px := software_render(&s, &scene, &g, nil, W, H)
		pixel, depth, in_front := camera_project(cam, W, H, at)
		testing.expect(t, in_front && depth > 0 && depth < 1, "in front, depth in 0..1")
		x, y := int(pixel.x), int(pixel.y)
		i := (y * W + x) * 4
		testing.expectf(t, x >= 0 && x < W && y >= 0 && y < H && px[i] > 200 && px[i + 1] < 40, "the sphere is drawn at %v: %v", pixel, px[i:][:4])
		software_destroy(&s)
		_, near_depth, _ := camera_project(cam, W, H, linalg.lerp(cam.position, at, 0.5))
		testing.expect(t, near_depth > depth, "a nearer point has a greater depth (reversed)")
	}
	_, _, in_front := camera_project(cameras[0], W, H, cameras[0].position * 2 - cameras[0].target)
	testing.expect(t, !in_front, "behind the camera")
}

// The frame's depth (Scene.depth): what camera_project says of a point on
// the nearest opaque surface; a point behind it is hidden (its projected
// depth is past the map's); blended draws are not in it; nothing without
// asking.
@(test)
test_frame_depth :: proc(t: ^testing.T) {
	g: Geometries
	defer geometries_destroy(&g)
	scene: Scene
	defer scene_destroy(&scene)
	scene_clear(&scene)
	scene.camera = {position = {0.3, 0.4, 4}, target = {0, 0, 0}, fov = 50, near = 0.1, far = 20}
	scene.depth = true
	wall, glass := MATERIAL_DEFAULT, MATERIAL_DEFAULT
	wall.flags = {.Basic}
	glass.flags, glass.opacity = {.Basic, .Transparent}, 0.5
	append(&scene.draws, Draw{plane(&g, 6, 6), compose({0, 0, 0}), wall})
	append(&scene.draws, Draw{plane(&g, 6, 6), compose({0, 0, 1}), glass})
	W, H :: 80, 60
	s: Software
	defer software_destroy(&s)
	software_render(&s, &scene, &g, nil, W, H)
	depth := software_depth(&s)
	testing.expect_value(t, len(depth), W * H)
	on_wall := Vec3{0.4, -0.3, 0}
	px, d, _ := camera_project(scene.camera, W, H, on_wall)
	at := depth[int(px.y) * W + int(px.x)]
	testing.expectf(t, abs(at - d) < 2e-3, "the wall's depth (not the glass's): map %v, point %v", at, d)
	_, behind, _ := camera_project(scene.camera, W, H, on_wall - {0, 0, 1})
	testing.expect(t, behind < at, "a point behind the wall is hidden (reversed: less)")
	scene.depth = false
	software_render(&s, &scene, &g, nil, W, H)
	testing.expect_value(t, len(software_depth(&s)), 0)
}

// Particles: drawn farthest first whatever their order (the near one's
// colour over the far one's), added with .Additive, faded near what they
// cut with softness.
@(test)
test_particles :: proc(t: ^testing.T) {
	g: Geometries
	defer geometries_destroy(&g)
	scene: Scene
	defer scene_destroy(&scene)
	W :: 32
	s: Software
	defer software_destroy(&s)
	at :: proc(px: []u8, x, y: int) -> [4]u8 {
		i := (y * W + x) * 4
		return {px[i], px[i + 1], px[i + 2], px[i + 3]}
	}
	frame :: proc(scene: ^Scene) {
		scene_clear(scene)
		scene.camera = {position = {0, 0, 5}, fov = 40, near = 0.1, far = 20}
	}
	// Near red listed first, far blue second: red still on top.
	frame(&scene)
	m := MATERIAL_DEFAULT
	pair := [2]Particle{{position = {0, 0, 1}, size = 2, color = {1, 0, 0, 1}}, {position = {0, 0, -1}, size = 2, color = {0, 0, 1, 1}}}
	append(&scene.draws, Draw{scene_particles(&scene, pair[:]), compose({0, 0, 0}), m})
	c := at(software_render(&s, &scene, &g, nil, W, W), W / 2, W / 2)
	testing.expectf(t, c.r > 250 && c.b < 4, "the near particle on top: %v", c)

	// Added: two half reds make a full red.
	frame(&scene)
	add := MATERIAL_DEFAULT
	add.color, add.flags = {1, 0, 0}, {.Additive}
	halves := [2]Particle{{position = {0, 0, 0}, size = 2, color = {1, 1, 1, 0.5}}, {position = {0, 0, 0.5}, size = 2, color = {1, 1, 1, 0.5}}}
	append(&scene.draws, Draw{scene_particles(&scene, halves[:]), compose({0, 0, 0}), add})
	c = at(software_render(&s, &scene, &g, nil, W, W), W / 2, W / 2)
	testing.expectf(t, c.r > 250, "two halves added: %v", c)

	// Soft: a particle half through a wall fades toward where it meets it.
	frame(&scene)
	wall := MATERIAL_DEFAULT
	wall.flags = {.Basic}
	wall.color = {0, 0, 0}
	append(&scene.draws, Draw{plane(&g, 10, 10), compose({0, 0, 0}, {0, 0.8, 0}), wall})
	soft := MATERIAL_DEFAULT
	soft.color, soft.softness = {1, 1, 1}, 1
	one := [1]Particle{{position = {0, 0, 0.3}, size = 3, color = {1, 1, 1, 1}}}
	append(&scene.draws, Draw{scene_particles(&scene, one[:]), compose({0, 0, 0}), soft})
	px := software_render(&s, &scene, &g, nil, W, W)
	// The wall turns about y: on one side it is behind the particle (faded
	// a little), on the other nearly at it (faded most).
	left, right := at(px, W / 2 - 8, W / 2), at(px, W / 2 + 8, W / 2)
	testing.expectf(t, left.r != right.r && min(left.r, right.r) < 200, "faded by how far the wall is behind: %v %v", left, right)
}

// point_hidden: points on a steeply tilted surface are not hidden by it
// (no bias), a point a little behind it is, one in front is not, one
// behind the camera or off the frame is.
@(test)
test_point_hidden :: proc(t: ^testing.T) {
	g: Geometries
	defer geometries_destroy(&g)
	scene: Scene
	defer scene_destroy(&scene)
	scene_clear(&scene)
	scene.camera = {position = {0, 0.5, 4}, target = {0, 0, 0}, fov = 50, near = 0.1, far = 50}
	scene.depth = true
	wall := MATERIAL_DEFAULT
	wall.flags = {.Basic}
	tilt := compose({0, 0, 0}, {0, 1.2, 0})
	append(&scene.draws, Draw{plane(&g, 6, 6), tilt, wall})
	W, H :: 64, 48
	s: Software
	defer software_destroy(&s)
	software_render(&s, &scene, &g, nil, W, H)
	depths := software_depth(&s)
	hidden_on := 0
	for i in 0 ..< 200 {
		u := f32(i % 20) / 19 * 2.4 - 1.2
		v := f32(i / 20) / 9 * 2 - 1
		on := (tilt * Vec4{u, v, 0, 1}).xyz
		hidden_on += 1 if point_hidden(scene.camera, depths, W, H, on) else 0
	}
	testing.expectf(t, hidden_on == 0, "%d of 200 points on the wall hidden by it", hidden_on)
	n := (tilt * Vec4{0, 0, 1, 0}).xyz
	testing.expect(t, point_hidden(scene.camera, depths, W, H, -n * 0.05), "a point just behind the wall is hidden")
	testing.expect(t, !point_hidden(scene.camera, depths, W, H, n * 0.05), "one in front is not")
	testing.expect(t, point_hidden(scene.camera, depths, W, H, {0, 0.5, 6}), "behind the camera")
	testing.expect(t, point_hidden(scene.camera, depths, W, H, {40, 0, 0}), "off the frame")
}

// A particle's opacity outside 0..1 (an emitter's fade overshooting) is
// clamped as the GPU's 8-bit target clamps it: no channel spills into
// another (the software renderer drew green and blue specks).
@(test)
test_particle_opacity_clamped :: proc(t: ^testing.T) {
	g: Geometries
	defer geometries_destroy(&g)
	scene: Scene
	defer scene_destroy(&scene)
	W :: 16
	s: Software
	defer software_destroy(&s)
	for flags in ([]Material_Flags{{.Additive}, {}}) {
		for a in ([]f32{-0.2, 1.7}) {
			scene_clear(&scene)
			scene.camera = {position = {0, 0, 5}, fov = 40, near = 0.1, far = 20}
			m := MATERIAL_DEFAULT
			m.color, m.flags = {1, 0.5, 0.25}, flags
			one := [1]Particle{{position = {0, 0, 0}, size = 3, color = {1, 1, 1, a}}}
			append(&scene.draws, Draw{scene_particles(&scene, one[:]), compose({0, 0, 0}), m})
			px := software_render(&s, &scene, &g, nil, W, W)
			i := (W / 2 * W + W / 2) * 4
			c := [4]u8{px[i], px[i + 1], px[i + 2], px[i + 3]}
			want := [4]u8{} if a < 0 else {255, 128, 64, 255}
			testing.expectf(t, c == want, "%v, opacity %v: %v, want %v", flags, a, c, want)
		}
	}
}

// Additive particles over a background, as the GPU draws them: the
// background under them at full strength (not dimmed by their alpha) and
// the sum clamped at 255 (it wrapped: red to 0, green/blue specks).
@(test)
test_additive_over_background :: proc(t: ^testing.T) {
	g: Geometries
	defer geometries_destroy(&g)
	textures: Textures
	defer textures_destroy(&textures)
	scene: Scene
	defer scene_destroy(&scene)
	sky_px := [4]u8{100, 50, 25, 255}
	sky := texture_add(&textures, sky_px[:], 1, 1)
	W :: 16
	s: Software
	defer software_destroy(&s)
	for n in 1 ..= 2 {
		scene_clear(&scene)
		scene.camera = {position = {0, 0, 5}, fov = 40, near = 0.1, far = 20}
		scene.background = sky
		m := MATERIAL_DEFAULT
		m.color, m.flags = {200.0 / 255, 200.0 / 255, 200.0 / 255}, {.Additive}
		rows := [2]Particle{{position = {0, 0, 0}, size = 3, color = 1}, {position = {0, 0, 0.1}, size = 3, color = 1}}
		append(&scene.draws, Draw{scene_particles(&scene, rows[:n]), compose({0, 0, 0}), m})
		px := software_render(&s, &scene, &g, nil, W, W, nil, &textures)
		i := (W / 2 * W + W / 2) * 4
		c := [4]u8{px[i], px[i + 1], px[i + 2], px[i + 3]}
		want := [4]u8{255, 250, 225, 255} if n == 1 else {255, 255, 255, 255}
		testing.expectf(t, c == want, "%d added over the sky: %v, want %v", n, c, want)
	}
}

// An additive mesh (any material, not only particles): drawn with the
// transparent ones, so an opaque plane behind it listed after it is under
// it, and its colour added to that plane's and to the sky's, clamped.
@(test)
test_additive_mesh :: proc(t: ^testing.T) {
	g: Geometries
	defer geometries_destroy(&g)
	textures: Textures
	defer textures_destroy(&textures)
	scene: Scene
	defer scene_destroy(&scene)
	sky_px := [4]u8{100, 50, 25, 255}
	scene.background = texture_add(&textures, sky_px[:], 1, 1)
	scene.camera = {position = {0, 0, 5}, fov = 40, near = 0.1, far = 20}
	add := MATERIAL_DEFAULT
	add.color, add.flags = {200.0 / 255, 200.0 / 255, 200.0 / 255}, {.Basic, .Additive}
	append(&scene.draws, Draw{plane(&g, 2, 2), compose({0, 0, 0}), add})
	red := MATERIAL_DEFAULT
	red.color, red.flags = {1, 0, 0}, {.Basic}
	append(&scene.draws, Draw{plane(&g, 0.5, 0.5), compose({0, 0, -1}), red})
	W :: 32
	s: Software
	defer software_destroy(&s)
	px := software_render(&s, &scene, &g, nil, W, W, nil, &textures)
	at :: proc(px: []u8, x, y: int) -> [4]u8 {
		i := (y * W + x) * 4
		return {px[i], px[i + 1], px[i + 2], px[i + 3]}
	}
	testing.expectf(t, at(px, W / 2, W / 2) == {255, 200, 200, 255}, "added to the plane behind: %v", at(px, W / 2, W / 2))
	testing.expectf(t, at(px, W / 2, 9) == {255, 250, 225, 255}, "added to the sky: %v", at(px, W / 2, 9))
	testing.expectf(t, at(px, 1, 1) == {100, 50, 25, 255}, "the sky: %v", at(px, 1, 1))
}

// A probed particle (`occlusion_radius`), drawn over everything: whole in
// the open, faded by the share of its disc a wall's edge hides, gone
// behind the wall, and gone with its point off the frame though its quad
// is on it.
@(test)
test_particle_occlusion :: proc(t: ^testing.T) {
	g: Geometries
	defer geometries_destroy(&g)
	scene: Scene
	defer scene_destroy(&scene)
	W :: 32
	s: Software
	defer software_destroy(&s)
	frame :: proc(scene: ^Scene, g: ^Geometries, s: ^Software, wall_x: f32, p: Particle, radius: f32 = 4) -> f32 {
		scene_clear(scene)
		scene.camera = {position = {0, 0, 5}, fov = 40, near = 0.1, far = 20}
		if wall_x < 10 {
			// A wall whose right edge is at x = wall_x.
			black := MATERIAL_DEFAULT
			black.color, black.flags = {0, 0, 0}, {.Basic}
			append(&scene.draws, Draw{plane(g, 4, 4), compose({wall_x - 2, 0, 1}), black})
		}
		m := MATERIAL_DEFAULT
		m.color, m.flags = {200.0 / 255, 200.0 / 255, 200.0 / 255}, {.Basic, .Additive, .No_Size_Attenuation}
		m.depth_offset, m.occlusion_radius = 1e6, radius
		rows := [1]Particle{p}
		append(&scene.draws, Draw{scene_particles(scene, rows[:]), compose({0, 0, 0}), m})
		px := software_render(s, scene, g, nil, W, W)
		return f32(px[(W / 2 * W + W / 2) * 4]) / 200
	}
	open := Particle{position = {0, 0, 0}, size = 6, color = 1}
	testing.expectf(t, abs(frame(&scene, &g, &s, 100, open) - 1) < 0.01, "in the open: whole")
	half := frame(&scene, &g, &s, 0, open)
	testing.expectf(t, half > 0.3 && half < 0.7, "half behind the wall's edge: %v", half)
	testing.expectf(t, frame(&scene, &g, &s, 2, open) == 0, "behind the wall: gone")
	testing.expectf(t, frame(&scene, &g, &s, 2, open, radius = 0) > 0.95, "behind the wall, not probed: drawn over it")
	// Its point past the frame's right edge, its quad moved back onto it.
	off := Particle{position = {2.5, 0, 0}, size = 6, color = 1, offset = {-3.66, 0}}
	testing.expectf(t, frame(&scene, &g, &s, 100, off) == 0, "its point off the frame: gone")
	testing.expectf(t, frame(&scene, &g, &s, 100, off, radius = 0) > 0.95, "its quad on the frame (not probed: drawn)")
	off.position.x, off.offset.x = 0.1, -0.4
	testing.expectf(t, frame(&scene, &g, &s, 100, off) > 0.95, "its point on the frame: whole")
}

// Vertex colours (`.Vertex_Colors`): a triangle's corners' colours
// interpolated in linear light, their alpha fading a blended panel over a
// plane behind, a polyline's gradient along it and points' own colours;
// without the flag the material's colour.
@(test)
test_vertex_colors :: proc(t: ^testing.T) {
	g: Geometries
	defer geometries_destroy(&g)
	scene: Scene
	defer scene_destroy(&scene)
	W :: 64
	s: Software
	defer software_destroy(&s)
	at :: proc(px: []u8, x, y: int) -> [4]u8 {
		i := (y * W + x) * 4
		return {px[i], px[i + 1], px[i + 2], px[i + 3]}
	}
	near :: proc(a, b: [4]u8, tol: int) -> bool {
		for k in 0 ..< 4 {
			if abs(int(a[k]) - int(b[k])) > tol {
				return false
			}
		}
		return true
	}
	ortho :: proc(scene: ^Scene) {
		scene_clear(scene)
		scene.camera = {position = {0, 0, 5}, orthographic = true, height = 2, near = 0.1, far = 20}
	}
	// A quad, its left corners red and opaque, its right blue and clear.
	quad := [4]Vertex{
		{position = {-1, -1, 0}, normal = {0, 0, 1}, color = {255, 0, 0, 255}},
		{position = {1, -1, 0}, normal = {0, 0, 1}, color = {0, 0, 255, 0}},
		{position = {1, 1, 0}, normal = {0, 0, 1}, color = {0, 0, 255, 0}},
		{position = {-1, 1, 0}, normal = {0, 0, 1}, color = {255, 0, 0, 255}},
	}
	idx := [6]u32{0, 1, 2, 0, 2, 3}
	q := geometry_add(&g, quad[:], idx[:])
	ortho(&scene)
	m := MATERIAL_DEFAULT
	m.flags = {.Basic, .Vertex_Colors}
	append(&scene.draws, Draw{q, compose({0, 0, 0}), m})
	px := software_render(&s, &scene, &g, nil, W, W)
	testing.expectf(t, near(at(px, 0, W / 2), {255, 0, 0, 255}, 24), "the left edge red: %v", at(px, 0, W / 2))
	testing.expectf(t, near(at(px, W - 1, W / 2), {0, 0, 255, 255}, 24), "the right edge blue (opaque: alpha not read): %v", at(px, W - 1, W / 2))
	// Halfway, in linear light: each channel half its linear value.
	mid := u8(math.round(linear_to_srgb(0.5) * 255))
	testing.expectf(t, near(at(px, W / 2, W / 2), {mid, 0, mid, 255}, 3), "the middle: %v, want %v", at(px, W / 2, W / 2), mid)
	// Blended over a white plane behind: clear at the right edge.
	ortho(&scene)
	white := MATERIAL_DEFAULT
	white.flags = {.Basic}
	append(&scene.draws, Draw{plane(&g, 4, 4), compose({0, 0, -1}), white})
	m.flags += {.Transparent}
	append(&scene.draws, Draw{q, compose({0, 0, 0}), m})
	px = software_render(&s, &scene, &g, nil, W, W)
	testing.expectf(t, near(at(px, 0, W / 2), {255, 0, 0, 255}, 24), "opaque at the left: %v", at(px, 0, W / 2))
	testing.expectf(t, near(at(px, W - 1, W / 2), {255, 255, 255, 255}, 24), "clear at the right: %v", at(px, W - 1, W / 2))
	// Without the flag: the material's colour.
	ortho(&scene)
	m.flags = {.Basic}
	m.color = {0, 1, 0}
	append(&scene.draws, Draw{q, compose({0, 0, 0}), m})
	px = software_render(&s, &scene, &g, nil, W, W)
	testing.expectf(t, at(px, 0, W / 2) == {0, 255, 0, 255}, "the material's colour: %v", at(px, 0, W / 2))
	// A polyline red to blue, and points of their own colours.
	ortho(&scene)
	line_pts := [2]Vec3{{-1.2, 0.5, 0}, {1.2, 0.5, 0}}
	line_c := [2]Vec4{{1, 0, 0, 1}, {0, 0, 1, 1}}
	pt_pts := [2]Vec3{{-0.5, -0.5, 0}, {0.5, -0.5, 0}}
	pt_c := [2]Vec4{{0, 1, 0, 1}, {1, 1, 0, 1}}
	thick := MATERIAL_DEFAULT
	thick.size, thick.flags = 6, {.Basic, .Vertex_Colors, .No_Size_Attenuation}
	append(&scene.draws, Draw{scene_polyline(&scene, line_pts[:], colors = line_c[:]), compose({0, 0, 0}), thick})
	append(&scene.draws, Draw{scene_points(&scene, pt_pts[:], pt_c[:]), compose({0, 0, 0}), thick})
	px = software_render(&s, &scene, &g, nil, W, W)
	y := W / 4
	testing.expectf(t, at(px, 2, y).r > 200 && at(px, 2, y).b < 120, "the line red at its start: %v", at(px, 2, y))
	testing.expectf(t, at(px, W - 3, y).b > 200 && at(px, W - 3, y).r < 120, "blue at its end: %v", at(px, W - 3, y))
	testing.expectf(t, at(px, W / 4, 3 * W / 4) == {0, 255, 0, 255}, "a green point: %v", at(px, W / 4, 3 * W / 4))
	testing.expectf(t, at(px, 3 * W / 4, 3 * W / 4) == {255, 255, 0, 255}, "a yellow point: %v", at(px, 3 * W / 4, 3 * W / 4))
}

// Depth precision: two layers 0.005 apart 200 units out with near 1 and
// far 40000 (a solar panel's, with an Earth in the frame). With depth 0 at
// near the two depths were a few float steps apart near 1 and the far
// layer showed through; reversed, floats are dense there and the near
// layer wins everywhere, drawn either order.
@(test)
test_depth_precision :: proc(t: ^testing.T) {
	g: Geometries
	defer geometries_destroy(&g)
	scene: Scene
	defer scene_destroy(&scene)
	W :: 64
	s: Software
	defer software_destroy(&s)
	for order in 0 ..< 2 {
		scene_clear(&scene)
		scene.camera = {position = {0, 0, 0}, target = {0.3, 0.2, -1}, fov = 40, near = 1, far = 40000}
		red, blue := MATERIAL_DEFAULT, MATERIAL_DEFAULT
		red.color, red.flags = {1, 0, 0}, {.Basic}
		blue.color, blue.flags = {0, 0, 1}, {.Basic}
		// Tilted so each pixel's depths come from the triangles' slopes.
		front := Draw{plane(&g, 400, 400), compose({60, 40, -200}, {0.4, 0.3, 0}), red}
		back := Draw{plane(&g, 400, 400), compose({60, 40, -200} + linalg.normalize(Vec3{0.3, 0.2, -1}) * 0.005, {0.4, 0.3, 0}), blue}
		if order == 0 {
			append(&scene.draws, front, back)
		} else {
			append(&scene.draws, back, front)
		}
		px := software_render(&s, &scene, &g, nil, W, W)
		through := 0
		for i := 0; i < len(px); i += 4 {
			if px[i + 2] > px[i] {
				through += 1
			}
		}
		testing.expectf(t, through == 0, "order %d: the far layer shows at %d of %d pixels", order, through, W * W)
	}
}

// Triplanar mapping on a mesh without uvs: projected in the object's
// frame the pattern moves with it (the same pixels a quarter of a unit
// along), in the world's it stays (the same pixels where it was); by uv
// the mesh reads one texel, a flat colour.
@(test)
test_triplanar :: proc(t: ^testing.T) {
	g: Geometries
	defer geometries_destroy(&g)
	textures: Textures
	defer textures_destroy(&textures)
	scene: Scene
	defer scene_destroy(&scene)
	// 4 x 4 cells of 4 texels: a quarter of a unit each.
	checker: [16 * 16 * 4]u8
	for i in 0 ..< 256 {
		v := u8(230 if (i % 16 / 4 + i / 64) % 2 == 0 else 30)
		checker[i * 4], checker[i * 4 + 1], checker[i * 4 + 2], checker[i * 4 + 3] = v, v, v, 255
	}
	tex := texture_add(&textures, checker[:], 16, 16)
	sv, si, _ := geometry_mesh(&g, plane(&g, 1.5, 1.5))
	verts := make([]Vertex, len(sv), context.temp_allocator)
	copy(verts, sv)
	for &v in verts {
		v.uv = {}
	}
	quad := geometry_add(&g, verts, si)
	W :: 64 // 32 pixels a unit
	s: Software
	defer software_destroy(&s)
	frame :: proc(scene: ^Scene, g: ^Geometries, s: ^Software, textures: ^Textures, quad: Geometry_Id, tex: Texture_Id, projection: Map_Projection, x: f32) -> []u8 {
		scene_clear(scene)
		scene.camera = {position = {0, 0, 5}, orthographic = true, height = 2, near = 0.1, far = 20}
		m := MATERIAL_DEFAULT
		m.flags, m.color_map, m.projection = {.Basic}, tex, projection
		append(&scene.draws, Draw{quad, compose({x, 0, 0}), m})
		px := software_render(s, scene, g, nil, W, W, nil, textures)
		out := make([]u8, len(px), context.temp_allocator)
		copy(out, px)
		return out
	}
	same :: proc(a, b: []u8, ax, bx, x0, x1: int) -> (worst: int) {
		for y in 16 ..< 48 {
			for x in x0 ..< x1 {
				worst = max(worst, abs(int(a[(y * W + x + ax) * 4]) - int(b[(y * W + x + bx) * 4])))
			}
		}
		return
	}
	a := frame(&scene, &g, &s, &textures, quad, tex, .Object, 0)
	b := frame(&scene, &g, &s, &textures, quad, tex, .Object, 0.25)
	contrast := 0
	for x in 16 ..< 48 {
		contrast = max(contrast, abs(int(a[(36 * W + x) * 4]) - int(a[(36 * W + 16) * 4])))
	}
	testing.expectf(t, contrast > 120, "the checker shows: %d", contrast)
	testing.expectf(t, same(a, b, 0, 8, 16, 48) <= 2, "object: moved with it, %d", same(a, b, 0, 8, 16, 48))
	testing.expectf(t, same(a, b, 0, 0, 16, 48) > 100, "object: not where it was")
	a = frame(&scene, &g, &s, &textures, quad, tex, .World, 0)
	b = frame(&scene, &g, &s, &textures, quad, tex, .World, 0.25)
	testing.expectf(t, same(a, b, 0, 0, 16, 48) <= 2, "world: stays, %d", same(a, b, 0, 0, 16, 48))
	a = frame(&scene, &g, &s, &textures, quad, tex, .Uv, 0)
	testing.expectf(t, same(a, a, 0, 3, 16, 44) == 0, "by uv: one texel")
}

// A ShaderMaterial calling shadowAt: dark under a box the shadow light
// sees, lit beside it, lit everywhere when no light casts; its map takes
// the first channel and the shader's own sampler the next.
@(test)
test_shader_shadow :: proc(t: ^testing.T) {
	g: Geometries
	defer geometries_destroy(&g)
	shaders: Shaders
	defer shaders_destroy(&shaders)
	scene: Scene
	defer scene_destroy(&scene)
	VERT :: `varying vec3 vWorld;
varying vec3 vNormal;
void main() {
  vec4 w = modelMatrix * vec4(position, 1.0);
  vWorld = w.xyz;
  vNormal = mat3(modelMatrix) * normal;
  gl_Position = projectionMatrix * viewMatrix * w;
}`
	FRAG :: `uniform sampler2D tint;
varying vec3 vWorld;
varying vec3 vNormal;
void main() {
  gl_FragColor = vec4(vec3(shadowAt(vWorld, vNormal)) * texture2D(tint, vec2(0.5)).rgb, 1.0);
}`
	id, errors := shader(&shaders, VERT, FRAG)
	testing.expectf(t, errors == "", "compiles: %s", errors)
	sh := shader_get(&shaders, id)
	testing.expect(t, sh != nil && sh.shadow_channel == 0 && shader_map(&shaders, id, "tint") == 1, "the shadow map first, the shader's own next")
	W :: 32
	s: Software
	defer software_destroy(&s)
	for casting in ([]bool{true, false}) {
		scene_clear(&scene)
		scene.camera = {position = {0, 6, 0.01}, target = {0, 0, 0}, orthographic = true, height = 4, near = 0.1, far = 20}
		append(&scene.lights, Light{kind = .Directional, color = 1, intensity = 1, position = {3, 5, 0}, cast_shadow = casting})
		floor := MATERIAL_DEFAULT
		floor.shader, floor.flags = id, {.Receive_Shadow}
		append(&scene.draws, Draw{plane(&g, 4, 4), compose({0, 0, 0}, {-math.PI / 2, 0, 0}), floor})
		caster := MATERIAL_DEFAULT
		caster.flags = {.Cast_Shadow, .Basic}
		append(&scene.draws, Draw{box(&g, 1, 0.2, 1), compose({1, 2, 0}), caster})
		px := software_render(&s, &scene, &g, &shaders, W, W)
		// The sun slants from +x: the box's shadow falls on x -0.7 .. 0.3,
		// beside it (seen from above at x 0.5 .. 1.5).
		under := px[(W / 2 * W + 13) * 4]
		open := px[(W / 2 * W + 4) * 4]
		if casting {
			testing.expectf(t, under < 20 && open > 240, "shadowed under the box (%d), lit beside it (%d)", under, open)
		} else {
			testing.expectf(t, under > 240 && open > 240, "no light casts: lit (%d, %d)", under, open)
		}
	}
}

// Labels as particles: an anchor (`Particle.offset`) puts the quad's
// bottom edge on its point (a wide one too), and `depth_offset` brings one inside a wall in
// front of it, covering the same pixels.
@(test)
test_particle_anchor_depth_offset :: proc(t: ^testing.T) {
	g: Geometries
	defer geometries_destroy(&g)
	scene: Scene
	defer scene_destroy(&scene)
	W :: 32
	s: Software
	defer software_destroy(&s)
	at :: proc(px: []u8, x, y: int) -> [4]u8 {
		i := (y * W + x) * 4
		return {px[i], px[i + 1], px[i + 2], px[i + 3]}
	}
	frame :: proc(scene: ^Scene, offset: Vec2, depth_offset: f32, z: f32, g: ^Geometries, wall: bool, size := Vec2{8, 8}) {
		scene_clear(scene)
		scene.camera = {position = {0, 0, 5}, fov = 40, near = 0.1, far = 20}
		if wall {
			blue := MATERIAL_DEFAULT
			blue.flags, blue.color = {.Basic}, {0, 0, 1}
			append(&scene.draws, Draw{plane(g, 10, 10), compose({0, 0, 0}), blue})
		}
		m := MATERIAL_DEFAULT
		m.color, m.flags, m.depth_offset = {1, 0, 0}, {.No_Size_Attenuation}, depth_offset
		one := [1]Particle{{position = {0, 0, z}, size = size, color = {1, 1, 1, 1}, offset = offset}}
		append(&scene.draws, Draw{scene_particles(scene, one[:]), compose({0, 0, 0}), m})
	}
	red :: proc(px: []u8) -> (n: int) {
		for i := 0; i < len(px); i += 4 {
			n += 1 if px[i] > 250 && px[i + 2] < 4 else 0
		}
		return
	}
	// Centred: rows 12 to 19; anchored at its bottom: rows 8 to 15.
	frame(&scene, {}, 0, 0, &g, false)
	px := software_render(&s, &scene, &g, nil, W, W)
	testing.expect(t, at(px, 16, 13).r > 250 && at(px, 16, 18).r > 250 && at(px, 16, 9).r == 0, "centred on its point")
	frame(&scene, {0, 0.5}, 0, 0, &g, false)
	px = software_render(&s, &scene, &g, nil, W, W)
	testing.expectf(t, at(px, 16, 9).r > 250 && at(px, 16, 14).r > 250 && at(px, 16, 17).r == 0, "its bottom edge on its point: %v %v", at(px, 16, 9), at(px, 16, 17))
	free_px := red(px)
	testing.expect_value(t, free_px, 64)
	// 12 by 4 pixels, anchored: columns 10 to 21, rows 12 to 15.
	frame(&scene, {0, 0.5}, 0, 0, &g, false, {12, 4})
	px = software_render(&s, &scene, &g, nil, W, W)
	testing.expect(t, red(px) == 48 && at(px, 10, 12).r > 250 && at(px, 21, 15).r > 250 && at(px, 9, 13).r == 0 && at(px, 16, 16).r == 0 && at(px, 16, 11).r == 0, "a wide quad, anchored")

	// A little inside the wall: hidden, and in front of it moved.
	frame(&scene, {0, 0.5}, 0, -0.05, &g, true)
	px = software_render(&s, &scene, &g, nil, W, W)
	testing.expect_value(t, red(px), 0)
	frame(&scene, {0, 0.5}, 0.2, -0.05, &g, true)
	px = software_render(&s, &scene, &g, nil, W, W)
	testing.expect_value(t, red(px), free_px)
}

// A premultiplied colour map with .Premultiplied_Color_Map draws as its
// straight one does (one colour, its alpha varying: where colours vary,
// filtering premultiplied texels is what differs).
@(test)
test_premultiplied_color_map :: proc(t: ^testing.T) {
	g: Geometries
	defer geometries_destroy(&g)
	textures: Textures
	defer textures_destroy(&textures)
	straight, premul: [4 * 4 * 4]u8
	for i in 0 ..< 16 {
		a := u8(40 + i * 13)
		c := [3]u8{230, 140, 60}
		for ch in 0 ..< 3 {
			straight[i * 4 + ch] = c[ch]
			premul[i * 4 + ch] = u8((u32(c[ch]) * u32(a) + 127) / 255)
		}
		straight[i * 4 + 3], premul[i * 4 + 3] = a, a
	}
	ids := [2]Texture_Id{texture_add(&textures, straight[:], 4, 4), texture_add(&textures, premul[:], 4, 4)}
	W :: 32
	frames: [2][]u8
	for id, k in ids {
		scene: Scene
		defer scene_destroy(&scene)
		scene_clear(&scene)
		scene.camera = {position = {0, 0, 2}, fov = 40, near = 0.1, far = 20}
		back := MATERIAL_DEFAULT
		back.flags, back.color = {.Basic}, {0.2, 0.6, 0.3}
		append(&scene.draws, Draw{plane(&g, 4, 4), compose({0, 0, -0.5}), back})
		m := MATERIAL_DEFAULT
		m.flags, m.color_map = {.Basic, .Transparent}, id
		if k == 1 {
			m.flags += {.Premultiplied_Color_Map}
		}
		append(&scene.draws, Draw{plane(&g, 1.2, 1.2), compose({0, 0, 0}), m})
		s: Software
		defer software_destroy(&s)
		px := software_render(&s, &scene, &g, nil, W, W, nil, &textures)
		frames[k] = make([]u8, len(px), context.temp_allocator)
		copy(frames[k], px)
	}
	worst := 0
	for b, i in frames[0] {
		worst = max(worst, abs(int(b) - int(frames[1][i])))
	}
	// Rounding: 8-bit premultiplied texels lose up to half a level / alpha.
	testing.expectf(t, worst <= 3, "differs by %d", worst)
	testing.expect(t, frames[0][(16 * W + 16) * 4] > 100, "the map is drawn")
}

// The host's pixels bound to an external texture (software_set_external)
// draw as the same image uploaded; unbound, as no map.
@(test)
test_software_external :: proc(t: ^testing.T) {
	g: Geometries
	defer geometries_destroy(&g)
	TW :: 8
	px: [TW * TW * 4]u8
	for y in 0 ..< TW {
		for x in 0 ..< TW {
			i := (y * TW + x) * 4
			px[i], px[i + 1], px[i + 2], px[i + 3] = u8(x * 32), u8(y * 32), u8(255 - (x ~ y) * 32), u8(255 - x * 20)
		}
	}
	images, externals: Textures
	defer textures_destroy(&images)
	defer textures_destroy(&externals)
	image_id := texture_add(&images, px[:], TW, TW)
	external_id := texture_add_external(&externals, TW, TW)
	W :: 48
	s: Software
	defer software_destroy(&s)
	frame :: proc(s: ^Software, g: ^Geometries, textures: ^Textures, id: Texture_Id) -> []u8 {
		scene: Scene
		defer scene_destroy(&scene)
		scene_clear(&scene)
		scene.camera = {position = {0, 0, 2}, fov = 40, near = 0.1, far = 20}
		m := MATERIAL_DEFAULT
		m.flags, m.color_map = {.Basic, .Transparent}, id
		append(&scene.draws, Draw{plane(g, 1.2, 1.2), compose({0, 0, 0}, {0, 0.3, 0}), m})
		out := software_render(s, &scene, g, nil, W, W, nil, textures)
		frame := make([]u8, len(out), context.temp_allocator)
		copy(frame, out)
		return frame
	}
	plain := frame(&s, &g, nil, 0)
	image := frame(&s, &g, &images, image_id)
	unbound := frame(&s, &g, &externals, external_id)
	software_set_external(&s, external_id, px[:], TW, TW)
	bound := frame(&s, &g, &externals, external_id)
	software_set_external(&s, external_id, nil, 0, 0)
	unbound_again := frame(&s, &g, &externals, external_id)
	testing.expect(t, !slice.equal(image, plain), "the image draws")
	testing.expect(t, slice.equal(bound, image), "bound: as the image")
	testing.expect(t, slice.equal(unbound, plain) && slice.equal(unbound_again, plain), "unbound: as no map")
}

// depth_image: distance along the camera's axis where drawn (a 1), the
// given far where not (a 0).
@(test)
test_depth_image :: proc(t: ^testing.T) {
	g: Geometries
	defer geometries_destroy(&g)
	scene: Scene
	defer scene_destroy(&scene)
	scene_clear(&scene)
	scene.camera = {position = {0, 0, 4}, fov = 40, near = 0.1, far = 50}
	scene.depth = true
	m := MATERIAL_DEFAULT
	m.flags = {.Basic}
	append(&scene.draws, Draw{plane(&g, 1, 1), compose({0, 0, 1.5}), m})
	W :: 32
	s: Software
	defer software_destroy(&s)
	software_render(&s, &scene, &g, nil, W, W)
	out := make([]f16, W * W * 4, context.temp_allocator)
	depth_image(scene.camera, software_depth(&s), W, W, 99, out)
	c := (16 * W + 16) * 4
	testing.expectf(t, abs(f32(out[c]) - 2.5) < 2e-3 && out[c + 3] == 1, "the plane 2.5 away: %v", out[c:][:4])
	testing.expectf(t, out[0] == 99 && out[1] == 99 && out[3] == 0, "nothing: far, a 0: %v", out[:4])
}
