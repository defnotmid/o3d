package o3d

import "core:math"
import "core:math/linalg"
import "core:testing"

@(private = "file")
square :: proc(half: f32, ccw: bool) -> [4]Vec2 {
	if ccw {
		return {{-half, -half}, {half, -half}, {half, half}, {-half, half}}
	}
	return {{-half, -half}, {-half, half}, {half, half}, {half, -half}}
}

@(private = "file")
circle :: proc(points: []Vec2, r: f32, centre: Vec2 = {}, ccw := true) {
	for &p, i in points {
		a := f32(i) / f32(len(points)) * math.TAU * (1 if ccw else -1)
		p = centre + r * Vec2{math.cos(a), math.sin(a)}
	}
}

// The triangles' total area (counter-clockwise positive) over `points`.
@(private = "file")
triangles_area :: proc(points: []Vec2, tris: []u32) -> (area: f32, clockwise: int) {
	for t := 0; t + 2 < len(tris); t += 3 {
		a, b, c := points[tris[t]], points[tris[t + 1]], points[tris[t + 2]]
		s := ((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)) / 2
		area += s
		if s < 0 {
			clockwise += 1
		}
	}
	return
}

// ShapeUtils.triangulateShape: a 4x4 square with a 2x2 hole is 8 triangles
// (n + 2h - 2) covering 12, all counter-clockwise, in either winding; a
// closing duplicate point changes nothing; past 80 points (earcut's z-order
// hashing) the same holds.
@(test)
test_triangulate_shape :: proc(t: ^testing.T) {
	for ccw in ([]bool{true, false}) {
		outer, hole := square(2, ccw), square(1, !ccw)
		holes := [][]Vec2{hole[:]}
		tris: [dynamic]u32
		defer delete(tris)
		testing.expect_value(t, triangulate_shape(outer[:], holes, &tris), 8)
		pts := [8]Vec2{outer[0], outer[1], outer[2], outer[3], hole[0], hole[1], hole[2], hole[3]}
		area, cw := triangles_area(pts[:], tris[:])
		testing.expectf(t, abs(area - 12) < 1e-5, "area %v", area)
		testing.expect_value(t, cw, 0)
	}
	{
		closed := [5]Vec2{{0, 0}, {1, 0}, {1, 1}, {0, 1}, {0, 0}}
		tris: [dynamic]u32
		defer delete(tris)
		testing.expect_value(t, triangulate_shape(closed[:], nil, &tris), 2)
		area, _ := triangles_area(closed[:], tris[:])
		testing.expect(t, abs(area - 1) < 1e-6)
	}
	{
		pts: [150]Vec2
		circle(pts[:100], 2)
		circle(pts[100:], 1, {0.2, 0}, ccw = false)
		holes := [][]Vec2{pts[100:]}
		tris: [dynamic]u32
		defer delete(tris)
		testing.expect_value(t, triangulate_shape(pts[:100], holes, &tris), 150)
		area, cw := triangles_area(pts[:], tris[:])
		want := abs(shape_area(pts[:100])) - abs(shape_area(pts[100:]))
		testing.expectf(t, abs(area - want) < 1e-3, "area %v, want %v", area, want)
		testing.expect_value(t, cw, 0)
	}
	{
		// Repeated and collinear points, and a two-point hole, are survived.
		odd := [7]Vec2{{0, 0}, {1, 0}, {1, 0}, {2, 0}, {2, 2}, {0, 2}, {0, 1}}
		stub := [2]Vec2{{0.5, 0.5}, {0.6, 0.6}}
		holes := [][]Vec2{stub[:]}
		tris: [dynamic]u32
		defer delete(tris)
		triangulate_shape(odd[:], holes, &tris)
		all := [9]Vec2{odd[0], odd[1], odd[2], odd[3], odd[4], odd[5], odd[6], stub[0], stub[1]}
		area, _ := triangles_area(all[:], tris[:])
		testing.expectf(t, abs(area - 4) < 1e-5, "area %v", area)
	}
}

// ShapePath.toShapes: a "B" (an outline, two counters wound the other way)
// is one shape with two holes; a dot inside a counter is a shape of its
// own; with every contour wound the same way, nonzero fills the counters
// in and even-odd keeps them as holes.
@(test)
test_shapes_from_contours :: proc(t: ^testing.T) {
	body := [4]Vec2{{0, 0}, {3, 0}, {3, 5}, {0, 5}} // counter-clockwise
	upper := [4]Vec2{{1, 3}, {1, 4}, {2, 4}, {2, 3}} // clockwise
	lower := [4]Vec2{{1, 1}, {1, 2}, {2, 2}, {2, 1}}
	dot := [4]Vec2{{1.4, 1.4}, {1.6, 1.4}, {1.6, 1.6}, {1.4, 1.6}}
	{
		contours := [][]Vec2{upper[:], body[:], lower[:]}
		shapes := shapes_from_contours(contours)
		defer shapes_destroy(shapes)
		testing.expect_value(t, len(shapes), 1)
		if len(shapes) == 1 {
			testing.expect_value(t, raw_data(shapes[0].contour), raw_data(body[:]))
			testing.expect_value(t, len(shapes[0].holes), 2)
		}
	}
	{
		contours := [][]Vec2{body[:], upper[:], lower[:], dot[:]}
		shapes := shapes_from_contours(contours)
		defer shapes_destroy(shapes)
		testing.expect_value(t, len(shapes), 2)
		if len(shapes) == 2 {
			testing.expect_value(t, len(shapes[0].holes), 2)
			testing.expect_value(t, raw_data(shapes[1].contour), raw_data(dot[:]))
			testing.expect_value(t, len(shapes[1].holes), 0)
		}
	}
	{
		same_upper := [4]Vec2{upper[3], upper[2], upper[1], upper[0]} // counter-clockwise
		same_lower := [4]Vec2{lower[3], lower[2], lower[1], lower[0]}
		contours := [][]Vec2{body[:], same_upper[:], same_lower[:]}
		nonzero := shapes_from_contours(contours, .Nonzero)
		defer shapes_destroy(nonzero)
		testing.expect_value(t, len(nonzero), 1)
		if len(nonzero) == 1 {
			testing.expect_value(t, len(nonzero[0].holes), 0)
		}
		even_odd := shapes_from_contours(contours, .Even_Odd)
		defer shapes_destroy(even_odd)
		testing.expect_value(t, len(even_odd), 1)
		if len(even_odd) == 1 {
			testing.expect_value(t, len(even_odd[0].holes), 2)
		}
	}
}

// ShapeGeometry: the square with a hole is 8 triangles at z = 0 facing +Z,
// covering 12, with the points' x and y as UVs.
@(test)
test_shape_geometry :: proc(t: ^testing.T) {
	g: Geometries
	defer geometries_destroy(&g)
	outer, hole := square(2, true), square(1, true) // the hole is rewound
	holes := [][]Vec2{hole[:]}
	shapes := []Shape{{outer[:], holes}}
	vertices, indices, ok := geometry_mesh(&g, shape_geometry(&g, shapes))
	testing.expect(t, ok)
	testing.expect_value(t, len(indices), 24)
	area: f32
	for i := 0; i + 2 < len(indices); i += 3 {
		a, b, c := vertices[indices[i]], vertices[indices[i + 1]], vertices[indices[i + 2]]
		face := linalg.cross(b.position - a.position, c.position - a.position)
		area += face.z / 2
		testing.expect(t, face.z > 0)
	}
	testing.expectf(t, abs(area - 12) < 1e-5, "area %v", area)
	for v in vertices {
		testing.expect_value(t, v.normal, Vec3{0, 0, 1})
		testing.expect_value(t, v.uv, v.position.xy)
		testing.expect_value(t, v.position.z, 0)
	}
}

// The mesh's signed volume (positive when every face winds outward) and
// how many triangles' vertex normals disagree with their winding.
@(private = "file")
mesh_volume :: proc(vertices: []Vertex, indices: []u32) -> (volume: f32, inward: int) {
	for i := 0; i + 2 < len(indices); i += 3 {
		a, b, c := vertices[indices[i]], vertices[indices[i + 1]], vertices[indices[i + 2]]
		volume += linalg.dot(a.position, linalg.cross(b.position, c.position)) / 6
		face := linalg.cross(b.position - a.position, c.position - a.position)
		if linalg.length(face) > 1e-9 && linalg.dot(face, a.normal + b.normal + c.normal) <= 0 {
			inward += 1
		}
	}
	return
}

// ExtrudeGeometry: the square with a hole, without a bevel, is 16 cap and 16
// wall triangles enclosing area * depth with every face outward; with
// three.js's default bevel the walls are 7 bands; the groups are the
// whole's index ranges, caps then sides; caps face +-Z.
@(test)
test_extrude_geometry :: proc(t: ^testing.T) {
	g: Geometries
	defer geometries_destroy(&g)
	outer, hole := square(2, false), square(1, false)
	holes := [][]Vec2{hole[:]}
	shapes := []Shape{{outer[:], holes}}

	flat := EXTRUDE_DEFAULT
	flat.bevel_enabled, flat.depth = false, 0.5
	e := extrude_geometry(&g, shapes, flat)
	vertices, indices, ok := geometry_mesh(&g, e.geometry)
	testing.expect(t, ok)
	testing.expect_value(t, len(indices), (16 + 16) * 3)
	volume, inward := mesh_volume(vertices, indices)
	testing.expectf(t, abs(volume - 6) < 1e-4, "volume %v", volume)
	testing.expect_value(t, inward, 0)
	caps, sides: Geometry
	caps, _ = geometry_get(&g, e.caps)
	sides, _ = geometry_get(&g, e.sides)
	whole, _ := geometry_get(&g, e.geometry)
	testing.expect_value(t, caps.index_count, 16 * 3)
	testing.expect_value(t, sides.first_index, whole.first_index + caps.index_count)
	testing.expect_value(t, caps.index_count + sides.index_count, whole.index_count)
	testing.expect_value(t, caps.first_vertex, whole.first_vertex)
	cv, ci, _ := geometry_mesh(&g, e.caps)
	for i in ci {
		v := cv[i]
		testing.expect_value(t, v.normal, Vec3{0, 0, -1} if v.position.z == 0 else Vec3{0, 0, 1})
		testing.expect_value(t, v.uv, v.position.xy)
	}
	sv, si, _ := geometry_mesh(&g, e.sides)
	for i in si {
		v := sv[i]
		testing.expect(t, abs(v.normal.z) < 1e-6)
		testing.expect(t, v.uv.y == 1 - v.position.z)
	}

	bevelled := extrude_geometry(&g, shapes)
	vertices, indices, _ = geometry_mesh(&g, bevelled.geometry)
	sides, _ = geometry_get(&g, bevelled.sides)
	testing.expect_value(t, sides.index_count, 8 * 7 * 2 * 3)
	volume, inward = mesh_volume(vertices, indices)
	testing.expectf(t, volume > 12, "volume %v", volume)
	testing.expect_value(t, inward, 0)
	lo, hi: f32 = 1e9, -1e9
	for v in vertices {
		lo, hi = min(lo, v.position.z), max(hi, v.position.z)
	}
	testing.expect(t, abs(lo + 0.2) < 1e-6 && abs(hi - 1.2) < 1e-6)
}

// A crease angle smooths a round wall (its normals radial and level) and
// its bevel, while the caps stay flat; without one every face is flat.
@(test)
test_extrude_crease :: proc(t: ^testing.T) {
	g: Geometries
	defer geometries_destroy(&g)
	ring: [48]Vec2
	circle(ring[:], 1)
	shapes := []Shape{{contour = ring[:]}}
	o := EXTRUDE_DEFAULT
	o.bevel_enabled = false
	o.crease_angle = math.PI / 6
	e := extrude_geometry(&g, shapes, o)
	sv, si, _ := geometry_mesh(&g, e.sides)
	for i in si {
		v := sv[i]
		want := linalg.normalize(Vec3{v.position.x, v.position.y, 0})
		testing.expectf(t, linalg.dot(v.normal, want) > 0.9999, "wall normal %v at %v", v.normal, v.position)
	}
	vertices, indices, _ := geometry_mesh(&g, e.geometry)
	_, inward := mesh_volume(vertices, indices)
	testing.expect_value(t, inward, 0)
	// Smooth walls share vertices around the ring (2 rows of 48, plus seams
	// where WorldUVGenerator's u turns from x to y), where flat ones take 4
	// a quad; the caps' 96 come first in the shared range.
	testing.expectf(t, len(sv) - 96 < 48 * 3, "%d wall vertices", len(sv) - 96)

	o.bevel_enabled = true
	b := extrude_geometry(&g, shapes, o)
	vertices, indices, _ = geometry_mesh(&g, b.geometry)
	_, inward = mesh_volume(vertices, indices)
	testing.expect_value(t, inward, 0)
	cv, ci, _ := geometry_mesh(&g, b.caps)
	for i in ci {
		testing.expect(t, abs(cv[i].normal.z) == 1)
	}
}
