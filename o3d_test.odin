package o3d

import "core:math"
import "core:math/linalg"
import "core:testing"

// Every primitive's triangles wind counter-clockwise seen from outside: the
// face normal agrees with the vertex normals.
@(test)
test_primitive_winding :: proc(t: ^testing.T) {
	g: Geometries
	defer geometries_destroy(&g)
	ids := []Geometry_Id{box(&g, 1, 2, 3, 2, 2, 2), sphere(&g, 1, 16, 8), plane(&g, 2, 1, 3, 2), cylinder(&g, 0.5, 1, 2, 12), cylinder(&g, 0, 1, 1, 8), torus(&g, 1, 0.3, 8, 16)}
	names := []string{"box", "sphere", "plane", "cylinder", "cone", "torus"}
	for id, n in ids {
		geo, ok := geometry_get(&g, id)
		testing.expect(t, ok)
		bad := 0
		for i := 0; i + 2 < len(geo.indices); i += 3 {
			a, b, c := geo.vertices[geo.indices[i]], geo.vertices[geo.indices[i + 1]], geo.vertices[geo.indices[i + 2]]
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
	// The same parameters are the same geometry.
	testing.expect_value(t, box(&g, 1, 2, 3, 2, 2, 2), ids[0])
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
	red.basic, red.color = true, {1, 0, 0}
	glass.basic, glass.color, glass.opacity, glass.transparent = true, {0, 0, 1}, 0.5, true
	// The glass is listed first: it is still drawn after the opaque box.
	append(&scene.draws, Draw{plane(&g, 1, 1), compose({0, 0, 1}), glass})
	append(&scene.draws, Draw{plane(&g, 1, 1), compose(), red})
	s: Software
	defer software_destroy(&s)
	px := software_render(&s, &scene, &g, 64, 64)
	i := (32 * 64 + 32) * 4
	testing.expectf(t, px[i] > 100 && px[i + 2] > 100 && px[i + 3] == 255, "%v", px[i:i + 4])
}
