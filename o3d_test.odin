package o3d

import "core:fmt"
import "core:math"
import "core:math/linalg"
import "core:strings"
import "core:testing"

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
	px := software_render(&s, &scene, &g, W, W)
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
	px := software_render(&s, &scene, &g, 64, 64)
	i := (32 * 64 + 32) * 4
	testing.expectf(t, px[i] > 100 && px[i + 2] > 100 && px[i + 3] == 255, "%v", px[i:i + 4])
}
