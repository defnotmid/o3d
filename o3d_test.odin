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
