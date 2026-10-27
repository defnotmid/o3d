package o3d

import "core:math"
import "core:math/linalg"
import "core:testing"

import curves "deps:ocurves"

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

// The polygons as closed contours of lines.
@(private = "file")
polygons :: proc(c: ^curves.Contours, rings: ..[]Vec2) {
	for r in rings {
		curves.contours_begin(c, r[0])
		for p in r[1:] {
			curves.contours_line(c, p)
		}
		curves.contours_end(c, true, false)
	}
}

// The polygons' shapes by `rule`, over `s`.
@(private = "file")
polygon_shapes :: proc(t: ^testing.T, s: ^Shapes, rule: curves.Fill_Rule, rings: ..[]Vec2, loc := #caller_location) {
	c: curves.Contours
	defer curves.contours_destroy(&c)
	polygons(&c, ..rings)
	testing.expect_value(t, shapes_from_contours(s, &c, rule), curves.Bool_Error.None, loc = loc)
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
// hashing) the same holds; repeated and collinear points and a two-point
// hole are survived.
@(test)
test_triangulate_shape :: proc(t: ^testing.T) {
	s: Shapes
	defer shapes_destroy(&s)
	tris: [dynamic]u32
	defer delete(tris)
	// Shape 0's triangles: how many, their area over its points, and how
	// many wind clockwise.
	cut :: proc(s: ^Shapes, tris: ^[dynamic]u32) -> (n: int, area: f32, cw: int) {
		clear(tris)
		if len(s.shapes) != 1 {
			return -1, 0, 0
		}
		n = triangulate_shape(s, 0, tris)
		area, cw = triangles_area(s.points[s.rings[s.shapes[0][0]][0]:], tris[:])
		return
	}
	for ccw in ([]bool{true, false}) {
		outer, hole := square(2, ccw), square(1, !ccw)
		polygon_shapes(t, &s, .Non_Zero, outer[:], hole[:])
		n, area, cw := cut(&s, &tris)
		testing.expect_value(t, n, 8)
		testing.expectf(t, abs(area - 12) < 1e-5, "area %v", area)
		testing.expect_value(t, cw, 0)
	}
	{
		closed := [5]Vec2{{0, 0}, {1, 0}, {1, 1}, {0, 1}, {0, 0}}
		polygon_shapes(t, &s, .Non_Zero, closed[:])
		n, area, _ := cut(&s, &tris)
		testing.expect_value(t, n, 2)
		testing.expectf(t, abs(area - 1) < 1e-6, "area %v", area)
	}
	{
		pts: [150]Vec2
		circle(pts[:100], 2)
		circle(pts[100:], 1, {0.2, 0}, ccw = false)
		polygon_shapes(t, &s, .Non_Zero, pts[:100], pts[100:])
		n, area, cw := cut(&s, &tris)
		testing.expect_value(t, n, 150)
		want := abs(shape_area(pts[:100])) - abs(shape_area(pts[100:]))
		testing.expectf(t, abs(area - want) < 1e-3, "area %v, want %v", area, want)
		testing.expect_value(t, cw, 0)
	}
	{
		odd := [7]Vec2{{0, 0}, {1, 0}, {1, 0}, {2, 0}, {2, 2}, {0, 2}, {0, 1}}
		stub := [2]Vec2{{0.5, 0.5}, {0.6, 0.6}}
		polygon_shapes(t, &s, .Non_Zero, odd[:], stub[:])
		_, area, _ := cut(&s, &tris)
		testing.expectf(t, abs(area - 4) < 1e-5, "area %v", area)
	}
}

// ShapePath.toShapes, by the filled area: a "B" (an outline, two counters
// wound the other way) is one shape with two holes, the outline clockwise
// and the holes counter-clockwise; a dot inside a counter is a shape of its
// own; with every contour wound the same way, nonzero fills the counters in
// and even-odd keeps them as holes; two overlapping squares are one outline
// (three.js keeps both, overlapping). Flattened, a line keeps its end and a
// curve `curve_segments` points (the quadratic's midpoint where three.js
// puts it), repeats dropped, and contours that fill nothing are gone.
@(test)
test_shapes_from_contours :: proc(t: ^testing.T) {
	body := [4]Vec2{{0, 0}, {3, 0}, {3, 5}, {0, 5}} // counter-clockwise
	upper := [4]Vec2{{1, 3}, {1, 4}, {2, 4}, {2, 3}} // clockwise
	lower := [4]Vec2{{1, 1}, {1, 2}, {2, 2}, {2, 1}}
	dot := [4]Vec2{{1.4, 1.4}, {1.6, 1.4}, {1.6, 1.6}, {1.4, 1.6}}
	s: Shapes
	defer shapes_destroy(&s)
	polygon_shapes(t, &s, .Non_Zero, upper[:], body[:], lower[:])
	testing.expect_value(t, len(s.shapes), 1)
	if len(s.shapes) == 1 {
		testing.expect_value(t, s.shapes[0], [2]u32{0, 3})
		testing.expectf(t, abs(shape_area(shapes_ring(&s, 0)) + 15) < 1e-5, "outline area %v", shape_area(shapes_ring(&s, 0)))
		testing.expect(t, !shape_is_clockwise(shapes_ring(&s, 1)) && !shape_is_clockwise(shapes_ring(&s, 2)))
	}
	polygon_shapes(t, &s, .Non_Zero, body[:], upper[:], lower[:], dot[:])
	testing.expect_value(t, len(s.shapes), 2)
	rings: [2]u32
	for sh in s.shapes {
		if sh[1] == 3 {
			rings[0] += 1
		} else if sh[1] == 1 && abs(shape_area(shapes_ring(&s, int(sh[0]))) + 0.04) < 1e-5 {
			rings[1] += 1
		}
	}
	testing.expect_value(t, rings, [2]u32{1, 1})

	same_upper := [4]Vec2{upper[3], upper[2], upper[1], upper[0]} // counter-clockwise
	same_lower := [4]Vec2{lower[3], lower[2], lower[1], lower[0]}
	polygon_shapes(t, &s, .Non_Zero, body[:], same_upper[:], same_lower[:])
	testing.expect_value(t, len(s.shapes), 1)
	testing.expect_value(t, len(s.rings), 1)
	polygon_shapes(t, &s, .Even_Odd, body[:], same_upper[:], same_lower[:])
	testing.expect_value(t, len(s.shapes), 1)
	testing.expect_value(t, len(s.rings), 3)

	left, right := square(1, true), [4]Vec2{{0, -1}, {2, -1}, {2, 1}, {0, 1}}
	polygon_shapes(t, &s, .Non_Zero, left[:], right[:])
	testing.expect_value(t, len(s.shapes), 1)
	testing.expect_value(t, len(s.rings), 1)
	if len(s.rings) == 1 {
		testing.expectf(t, abs(shape_area(shapes_ring(&s, 0)) + 6) < 1e-5, "union area %v", shape_area(shapes_ring(&s, 0)))
	}

	c: curves.Contours
	defer curves.contours_destroy(&c)
	curves.contours_begin(&c, {0, 0})
	curves.contours_line(&c, {2, 0})
	curves.contours_line(&c, {2, 0})
	curves.contours_quad(&c, {2, 2}, {0, 2})
	curves.contours_end(&c, true, false)
	curves.contours_begin(&c, {5, 5})
	curves.contours_line(&c, {6, 5})
	curves.contours_end(&c, false, false)
	testing.expect_value(t, shapes_from_contours(&s, &c, curve_segments = 4), curves.Bool_Error.None)
	testing.expect_value(t, len(s.rings), 1)
	if len(s.rings) == 1 {
		ring := shapes_ring(&s, 0)
		testing.expect_value(t, len(ring), 1 + 1 + 4)
		mid := false
		for p in ring {
			mid ||= linalg.length(p - Vec2{1.5, 1.5}) < 1e-6 // t = 1/2: (2,0)/4 + (2,2)/2 + (0,2)/4
		}
		testing.expect(t, mid)
	}
}

// ShapeGeometry: the square with a hole (even-odd: either winding) is 8
// triangles at z = 0 facing +Z, covering 12, with the points' x and y as
// UVs.
@(test)
test_shape_geometry :: proc(t: ^testing.T) {
	g: Geometries
	defer geometries_destroy(&g)
	outer, hole := square(2, true), square(1, true)
	s: Shapes
	defer shapes_destroy(&s)
	polygon_shapes(t, &s, .Even_Odd, outer[:], hole[:])
	vertices, indices, ok := geometry_mesh(&g, shape_geometry(&g, &s))
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
	outer, hole := square(2, false), square(1, true)
	s: Shapes
	defer shapes_destroy(&s)
	polygon_shapes(t, &s, .Non_Zero, outer[:], hole[:])
	shapes := &s

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

// Bevel offsets as the straight skeleton makes them: a comb's teeth taken
// in by 0.4 are 0.2 wide (its front cap the exact area), by 0.6 gone (no
// cap, a closed roof); a small hole closes under a bevel wider than it (no
// wall about it); pixels touching at a corner offset either way; a sharp
// corner is cut past the miter limit and mitered under it. Every result
// winds outward and encloses volume.
@(test)
test_extrude_offset :: proc(t: ^testing.T) {
	g: Geometries
	defer geometries_destroy(&g)
	// The front cap's area (its faces facing +Z at the top), the mesh's
	// volume and inward faces.
	solid :: proc(t: ^testing.T, g: ^Geometries, o: Extrude_Options, rings: ..[]Vec2, loc := #caller_location) -> (cap: f32, e: Extruded) {
		s: Shapes
		defer shapes_destroy(&s)
		polygon_shapes(t, &s, .Non_Zero, ..rings, loc = loc)
		e = extrude_geometry(g, &s, o)
		vertices, indices, _ := geometry_mesh(g, e.geometry)
		volume, inward := mesh_volume(vertices, indices)
		testing.expectf(t, volume > 0 && inward == 0, "volume %v, %v inward", volume, inward, loc = loc)
		cv, ci, _ := geometry_mesh(g, e.caps)
		for i := 0; i + 2 < len(ci); i += 3 {
			a, b, c := cv[ci[i]].position, cv[ci[i + 1]].position, cv[ci[i + 2]].position
			if a.z > o.depth {
				cap += ((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)) / 2
			}
		}
		return
	}
	comb := []Vec2{{0, 0}, {5, 0}, {5, 4}, {4, 4}, {4, 1}, {3, 1}, {3, 4}, {2, 4}, {2, 1}, {1, 1}, {1, 4}, {0, 4}}
	o := EXTRUDE_DEFAULT
	o.bevel_size, o.bevel_offset = 0.4, -0.4
	cap, _ := solid(t, &g, o, comb)
	// The base 4.2 x 0.2 and three teeth 0.2 x 3.
	testing.expectf(t, abs(cap - 2.64) < 1e-4, "taken in by 0.4: cap %v", cap)
	o.bevel_size, o.bevel_offset = 0.6, -0.6
	cap, _ = solid(t, &g, o, comb)
	testing.expectf(t, cap == 0, "taken in by 0.6: cap %v", cap)

	outer, hole := square(2, true), square(0.2, false)
	o.bevel_size, o.bevel_offset = 0.25, 0
	_, e := solid(t, &g, o, outer[:], hole[:])
	vertices, _, _ := geometry_mesh(&g, e.geometry)
	for v in vertices {
		if v.position.z >= 0 && v.position.z <= o.depth {
			testing.expectf(t, max(abs(v.position.x), abs(v.position.y)) > 1, "no wall about the closed hole: %v", v.position)
		}
	}

	// Pixels touching at a corner as tools/mascot traces them: a ring of
	// eight around an empty middle, its corner pixel out, so the outline
	// turns around each pixel's corner at (2, -2) and passes it twice.
	pinch := []Vec2{{0, 0}, {3, 0}, {3, -2}, {2, -2}, {2, -1}, {1, -1}, {1, -2}, {2, -2}, {2, -3}, {0, -3}}
	o.bevel_size, o.bevel_offset = 0.1, 0
	solid(t, &g, o, pinch)
	o.bevel_size, o.bevel_offset = 0.1, -0.1
	solid(t, &g, o, pinch)

	// A 10 degree point out at x = 4: cut 0.1 past it, or mitered out to
	// 0.1 / sin(5 degrees).
	spike := []Vec2{{0, 0}, {0, 0.7}, {4, 0.35}}
	o.bevel_size, o.bevel_offset = 0.1, 0
	reach :: proc(g: ^Geometries, e: Extruded) -> (x: f32) {
		vertices, _, _ := geometry_mesh(g, e.geometry)
		for v in vertices {
			x = max(x, v.position.x)
		}
		return
	}
	_, e = solid(t, &g, o, spike)
	testing.expectf(t, abs(reach(&g, e) - 4.1) < 1e-4, "cut past the limit: %v", reach(&g, e))
	o.miter_limit = 20
	_, e = solid(t, &g, o, spike)
	want := 4 + 0.1 / math.sin(f32(math.atan2(0.35, 4)))
	testing.expectf(t, abs(reach(&g, e) - want) < 1e-3, "mitered under it: %v, want %v", reach(&g, e), want)
}

// A crease angle smooths a round wall (its normals radial and level) and
// its bevel, while the caps stay flat; without one every face is flat.
@(test)
test_extrude_crease :: proc(t: ^testing.T) {
	g: Geometries
	defer geometries_destroy(&g)
	ring: [48]Vec2
	circle(ring[:], 1)
	s: Shapes
	defer shapes_destroy(&s)
	polygon_shapes(t, &s, .Non_Zero, ring[:])
	shapes := &s
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
