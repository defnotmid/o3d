package o3d

import "core:math"
import "core:slice"

import curves "deps:ocurves"

// Shapes: three.js's ShapePath.toShapes, ShapeUtils, ShapeGeometry and
// ExtrudeGeometry. A path's contours (ocurves' `Contours`, curve.odin) are
// made shapes once (`shapes_from_contours`): what their fill rule fills, as
// outlines with their holes, flattened to points; the geometries are built
// from those.
//
// A shape's geometry is made once and kept like any other (`Geometries`
// never frees): build it when the outline is first seen, keep the id, and
// move it per frame through the draw's transform. Unlike the primitives,
// shapes are not looked up by their parameters, so calling `extrude_geometry`
// every frame makes a new mesh every frame (`geometries_made` shows it).

// Shapes in one table: every ring's points in `points` (once round, no
// closing point repeating the first), each ring's range of them in `rings`
// (first, count), each shape's range of rings in `shapes` (first, count:
// its outline, then its holes), so a shape's points are one range too.
// Outlines wind clockwise (Y up), holes counter-clockwise, as three.js's
// geometries wind them. `scratch` is the booleans', kept warm across calls.
Shapes :: struct {
	points:  [dynamic]Vec2,
	rings:   [dynamic][2]u32,
	shapes:  [dynamic][2]u32,
	scratch: curves.Bool_Scratch,
}

shapes_destroy :: proc(s: ^Shapes) {
	delete(s.points)
	delete(s.rings)
	delete(s.shapes)
	curves.bool_scratch_destroy(&s.scratch)
	s^ = {}
}

// ShapePath.toShapes with getPoints(curve_segments): the area contours `c`
// fill by `fill_rule`, as shapes written over `s`. Where three.js takes
// each contour's winding and containment as given, this is the filled
// area itself (ocurves' simplify): contours that cross or overlap are
// merged, one that changes nothing is gone, a solid inside a hole is a
// shape of its own. A line keeps its end; a curve takes `curve_segments`
// points at even steps of its parameter, as three.js flattens it, and a
// part of one (cut where contours meet) its share of them; a point equal
// to the one before is dropped. On an error `s` is left empty.
shapes_from_contours :: proc(s: ^Shapes, c: ^curves.Contours, fill_rule := curves.Fill_Rule.Non_Zero, curve_segments := 12) -> curves.Bool_Error {
	clear(&s.points)
	clear(&s.rings)
	clear(&s.shapes)
	input := [1]curves.Bool_Input{{count = u32(len(c.contours)), rule = fill_rule}}
	budget := curves.DEFAULT_BUDGET
	curves.bool_run(&s.scratch, c, input[:], .Union, &budget) or_return
	n := max(curve_segments, 1)
	out := &s.scratch.out
	for f in s.scratch.faces {
		first := u32(len(s.rings))
		for ct, k in out.contours[f[0]:][:f[1]] {
			start := len(s.points)
			pts := out.pts[ct.start:][:3 * ct.segs + 1]
			shapes_push(s, pts[0])
			for kind, i in out.kinds[ct.seg:][:ct.segs] {
				q := pts[3 * i:][:4]
				if kind == .Line {
					shapes_push(s, q[3])
					continue
				}
				t := s.scratch.out_t[int(ct.seg) + i]
				m := max(int(math.ceil(f64(n) * abs(f64(t[1]) - f64(t[0])))), 1)
				p0, c1, c2, p3 := wide(q[0]), wide(q[1]), wide(q[2]), wide(q[3])
				for j in 1 ..= m {
					u := f64(j) / f64(m)
					w := 1 - u
					p := w * w * w * p0 + 3 * w * w * u * c1 + 3 * w * u * u * c2 + u * u * u * p3
					shapes_push(s, {f32(p.x), f32(p.y)})
				}
			}
			if len(s.points) - start > 1 && s.points[len(s.points) - 1] == s.points[start] {
				pop(&s.points)
			}
			if len(s.points) - start < 3 {
				resize(&s.points, start)
				if k == 0 {
					break // no outline: no shape
				}
				continue
			}
			append(&s.rings, [2]u32{u32(start), u32(len(s.points) - start)})
		}
		if u32(len(s.rings)) > first {
			append(&s.shapes, [2]u32{first, u32(len(s.rings)) - first})
		}
	}
	return .None
}

@(private = "file")
shapes_push :: #force_inline proc(s: ^Shapes, p: Vec2) {
	if len(s.points) == 0 || s.points[len(s.points) - 1] != p {
		append(&s.points, p)
	}
}

@(private = "file")
wide :: #force_inline proc "contextless" (p: Vec2) -> [2]f64 {
	return {f64(p.x), f64(p.y)}
}

// Ring `i` of `s`'s points.
shapes_ring :: proc(s: ^Shapes, i: int) -> []Vec2 {
	r := s.rings[i]
	return s.points[r[0]:][:r[1]]
}

// ShapeUtils.area: the signed area, positive when counter-clockwise (Y up).
shape_area :: proc "contextless" (contour: []Vec2) -> f32 {
	return f32(ring_area(contour))
}

// ShapeUtils.isClockWise.
shape_is_clockwise :: proc "contextless" (contour: []Vec2) -> bool {
	return ring_area(contour) < 0
}

// A ring's signed area in f64, positive when counter-clockwise (Y up).
@(private = "file")
ring_area :: proc "contextless" (r: [][2]$T) -> f64 {
	a: f64
	for p, q := len(r) - 1, 0; q < len(r); p, q = q, q + 1 {
		a += f64(r[p].x) * f64(r[q].y) - f64(r[q].x) * f64(r[p].y)
	}
	return a * 0.5
}

// ShapeUtils.triangulateShape: earcut (mapbox/earcut, as three.js carries
// it) of shape `i` of `s`, its outline with its holes. Appends three
// indices per triangle into the shape's points (from its first: the
// outline's, then each hole's), every triangle counter-clockwise (Y up);
// returns how many triangles it appended.
triangulate_shape :: proc(s: ^Shapes, i: int, triangles: ^[dynamic]u32) -> int {
	sp: Shape_Points
	defer shape_points_destroy(&sp)
	shape_points(&sp, s, i)
	before := len(triangles)
	triangulate_rings(&sp, sp.points[:], triangles)
	return (len(triangles) - before) / 3
}

// ---- ShapeGeometry and ExtrudeGeometry ---------------------------------------------------

// ShapeGeometry(shapes): the shapes' flat triangulated faces at z = 0,
// facing +Z, with three.js's UVs (the points' x and y).
shape_geometry :: proc(g: ^Geometries, shapes: ^Shapes) -> Geometry_Id {
	clear(&g.scratch.vertices)
	clear(&g.scratch.indices)
	s: Shape_Points
	defer shape_points_destroy(&s)
	tris: [dynamic]u32
	defer delete(tris)
	for i in 0 ..< len(shapes.shapes) {
		shape_points(&s, shapes, i)
		base := u32(len(g.scratch.vertices))
		clear(&tris)
		triangulate_rings(&s, s.points[:], &tris)
		for p in s.points {
			q := Vec2{f32(p.x), f32(p.y)}
			append(&g.scratch.vertices, Vertex{position = {q.x, q.y, 0}, normal = {0, 0, 1}, uv = q})
		}
		for i in tris {
			append(&g.scratch.indices, base + i)
		}
	}
	return geometry_add(g, g.scratch.vertices[:], g.scratch.indices[:])
}

// ExtrudeGeometry's options, three.js's names and defaults
// (`EXTRUDE_DEFAULT`). Without a bevel the shape spans z in [0, depth]; a
// bevel adds `bevel_thickness` beyond both ends, its caps the shape offset
// by `bevel_offset`, its walls by `bevel_size + bevel_offset` (out of the
// shape: positive), on a quarter ellipse between. An offset is the shape's
// straight skeleton's: each edge moved that far along its normal, corners
// where their lines meet, parts too thin for it gone and parts it brings
// together joined, so any offset makes a closed solid (three.js's bevel
// turns thin parts inside out).
Extrude_Options :: struct {
	depth:           f32, // 1
	steps:           int, // 1: segments along the depth
	bevel_enabled:   bool, // true
	bevel_thickness: f32, // 0.2
	bevel_size:      f32, // 0.1 (three.js: bevelThickness - 0.1)
	bevel_offset:    f32, // 0
	bevel_segments:  int, // 3
	// 4 (SVG's stroke-miterlimit): a corner the offset sharpens to more
	// than this many times its distance out (a miter) is cut square across
	// its bisector instead (1: every corner that juts).
	miter_limit:     f32,
	// o3d's own: side faces meeting at less than this angle (radians) share
	// smoothed normals, so curved walls and round bevels shade smoothly. 0
	// is three.js's: every face flat. Caps always stay flat, with hard
	// edges where they meet the walls.
	crease_angle:    f32,
}

EXTRUDE_DEFAULT :: Extrude_Options {
	depth           = 1,
	steps           = 1,
	bevel_enabled   = true,
	bevel_thickness = 0.2,
	bevel_size      = 0.1,
	bevel_segments  = 3,
	miter_limit     = 4,
}

// An extruded geometry and three.js's two material groups, each a geometry
// of its own sharing the whole's vertices: draw `geometry` with one
// material, or `caps` (group 0: the front and back faces) and `sides`
// (group 1: the walls and bevels) with two.
Extruded :: struct {
	geometry, caps, sides: Geometry_Id,
}

// ExtrudeGeometry(shapes, options): front and back caps, side walls and
// bevel rings, with three.js's WorldUVGenerator UVs, flat normals (or
// smoothed walls by `crease_angle`), and the caps' triangles before the
// sides' (three.js orders its groups per shape; here each group is one
// range). `steps` below 1 is taken as 1, `bevel_segments` below 0 as 0.
extrude_geometry :: proc(g: ^Geometries, shapes: ^Shapes, options := EXTRUDE_DEFAULT) -> Extruded {
	o := options
	o.steps = max(o.steps, 1)
	if !o.bevel_enabled || o.bevel_segments <= 0 {
		o.bevel_segments, o.bevel_thickness, o.bevel_size, o.bevel_offset = 0, 0, 0, 0
	}
	x: Extruder
	defer extruder_destroy(&x)
	for i in 0 ..< len(shapes.shapes) {
		extrude_shape(&x, shapes, i, o)
	}
	// Every triangle's corners, caps then sides, in order.
	side_normals(&x, o.crease_angle)
	cap_indices := len(x.caps) * 3
	resize(&x.corners, cap_indices + len(x.sides) * 3)
	resize(&x.corner_point, len(x.corners))
	for t, ti in x.caps {
		a, b, c := x.points[t[0]], x.points[t[1]], x.points[t[2]]
		n := face_normal(a, b, c)
		for p, k in ([3][3]f64{a, b, c}) {
			x.corners[ti * 3 + k] = {position = vec3(p), normal = n, uv = {f32(p.x), f32(p.y)}}
			x.corner_point[ti * 3 + k] = t[k]
		}
	}
	for t, ti in x.sides {
		for k in 0 ..< 3 {
			p := x.points[t.corners[k]]
			uv := Vec2{f32(p.x if t.u_from_x else p.y), f32(1 - p.z)}
			x.corners[cap_indices + ti * 3 + k] = {position = vec3(p), normal = x.normals[ti * 3 + k], uv = uv}
			x.corner_point[cap_indices + ti * 3 + k] = t.corners[k]
		}
	}
	shared_vertices(&x, g)
	whole := geometry_add(g, g.scratch.vertices[:], g.scratch.indices[:])
	return {
		geometry = whole,
		caps     = geometry_group(g, whole, 0, u32(cap_indices)),
		sides    = geometry_group(g, whole, u32(cap_indices), u32(len(g.scratch.indices) - cap_indices)),
	}
}

// The corners as vertices and indices in `g.scratch`, equal ones (as
// values: -0 is 0) one vertex, numbered in the order they first come. Equal corners
// share a point, so each is looked for among its point's earlier corners
// only: the corners grouped by point (counted, prefix-summed, filled in
// place order).
@(private = "file")
shared_vertices :: proc(x: ^Extruder, g: ^Geometries) {
	n := len(x.corners)
	corners := x.corners[:]
	resize(&x.point_start, len(x.points) + 1)
	slice.zero(x.point_start[:])
	for p in x.corner_point {
		x.point_start[p + 1] += 1
	}
	for i in 1 ..< len(x.point_start) {
		x.point_start[i] += x.point_start[i - 1]
	}
	resize(&x.by_point, n)
	resize(&x.first, n)
	for p, i in x.corner_point {
		x.by_point[x.point_start[p]] = u32(i)
		x.point_start[p] += 1
	}
	// point_start[p] is now where p's span ends; it starts at p - 1's end.
	for p in 0 ..< len(x.points) {
		span := x.by_point[x.point_start[p - 1] if p > 0 else 0:x.point_start[p]]
		for c, k in span {
			x.first[c] = c
			for e in span[:k] {
				if x.first[e] == e && corners[c] == corners[e] {
					x.first[c] = e
					break
				}
			}
		}
	}
	// Numbered in place order: a first corner takes the next number, the
	// rest their first's (x.first then holds the numbers at first places).
	table_grow_non_zero(&g.scratch.indices, n)
	clear(&g.scratch.vertices)
	for f, i in x.first {
		if int(f) == i {
			x.first[i] = u32(len(g.scratch.vertices))
			append(&g.scratch.vertices, corners[i])
			g.scratch.indices[i] = x.first[i]
		} else {
			g.scratch.indices[i] = x.first[f]
		}
	}
}

// three.js's geometry.addGroup(start, count): a geometry drawing `count` of
// `src`'s indices from `start`, sharing its vertices (nothing is copied), to
// draw part of a mesh with a material of its own. 0 when `src` is not a
// geometry or the range is not in it.
geometry_group :: proc(g: ^Geometries, src: Geometry_Id, start, count: u32) -> Geometry_Id {
	geo, ok := geometry_get(g, src)
	if !ok || start + count > geo.index_count || start + count < start {
		return 0
	}
	generation_take(&g.generation)
	geo.first_index += start
	geo.index_count = count
	append(&g.list, geo)
	return Geometry_Id(len(g.list))
}

// ---- building ------------------------------------------------------------------------------
// In f64, as three.js builds in JavaScript's numbers: earcut's and the UV
// generator's exact comparisons then decide as they do there.

@(private = "file")
vec3 :: #force_inline proc "contextless" (p: [3]f64) -> Vec3 {
	return {f32(p.x), f32(p.y), f32(p.z)}
}

// A shape's points in f64: `points` is the outline then each hole, `rings`
// their lengths.
@(private = "file")
Shape_Points :: struct {
	points: [dynamic][2]f64,
	rings:  [dynamic]int,
	// Scratch for `triangulate_rings`, kept across shapes: the holes'
	// ranges and the ear clipper's nodes.
	starts, ends: [dynamic]int,
	nodes:        [dynamic]Ear_Node,
}

@(private = "file")
shape_points_destroy :: proc(s: ^Shape_Points) {
	delete(s.points)
	delete(s.rings)
	delete(s.starts)
	delete(s.ends)
	delete(s.nodes)
}

// Fills `s` from shape `i` of `shapes`.
@(private = "file")
shape_points :: proc(s: ^Shape_Points, shapes: ^Shapes, i: int) {
	clear(&s.points)
	clear(&s.rings)
	for r in shapes.rings[shapes.shapes[i][0]:][:shapes.shapes[i][1]] {
		for p in shapes.points[r[0]:][:r[1]] {
			append(&s.points, wide(p))
		}
		append(&s.rings, int(r[1]))
	}
}

// ShapeUtils.triangulateShape over `s`'s rings, at `points` (its points or
// the same rings moved).
@(private = "file")
triangulate_rings :: proc(s: ^Shape_Points, points: [][2]f64, triangles: ^[dynamic]u32) {
	resize(&s.starts, len(s.rings) - 1)
	resize(&s.ends, len(s.rings) - 1)
	k := s.rings[0]
	for l, h in s.rings[1:] {
		s.starts[h], s.ends[h] = k, k + l
		k += l
	}
	clear(&s.nodes)
	e := Earcut{nodes = s.nodes, triangles = triangles}
	earcut(&e, points, s.rings[0], s.starts[:], s.ends[:])
	s.nodes = e.nodes // as it grew
}

@(private = "file")
Side_Triangle :: struct {
	corners:  [3]u32, // into `Extruder.points`
	u_from_x: bool, // WorldUVGenerator's wall u: the quad's x, else its y
}

// ExtrudeGeometry's state across its shapes: every shape's points (three.js's
// placeholder), the cap and side triangles into them, then the sides'
// corner normals and the emitted vertices.
@(private = "file")
Extruder :: struct {
	points:  [dynamic][3]f64,
	caps:    [dynamic][3]u32,
	sides:   [dynamic]Side_Triangle,
	normals: [dynamic]Vec3, // per side corner
	// The corners and their points, and scratch to share equal ones
	// (`shared_vertices`): the corners by point (a span each) and each
	// corner's first equal one.
	corners:      [dynamic]Vertex,
	corner_point: [dynamic]u32,
	point_start:  [dynamic]u32,
	by_point:     [dynamic]u32,
	first:        [dynamic]u32,
	shape:        Shape_Points,
	// The bevel's rings from the caps to the walls (`path`), those of one
	// direction in the order its wavefront reaches them (`run`), the
	// wavefront (`lines`, `wave`), and the loops at the caps' and the
	// walls' rings.
	path:         [dynamic]Ring_Entry,
	run:          [dynamic]Ring_Entry,
	lines:        [dynamic]Wave_Line,
	wave:         [dynamic]Wave_Vertex,
	samples:      [dynamic]Wave_Sample,
	edge_lines:   [dynamic][3]f64,
	edge_from:    [dynamic]i32,
	chain_a:      [dynamic]i32,
	chain_b:      [dynamic]i32,
	seen:         [dynamic]bool,
	cap_loops:    Loops,
	wall_loops:   Loops,
	// Scratch for the caps: one outline and its holes, their points'
	// indices, the loops' areas, and the triangles.
	cap_shape:    Shape_Points,
	cap_ids:      [dynamic][2]u32,
	areas:        [dynamic]f64,
	outline_of:   [dynamic]int,
	faces:        [dynamic]u32,
}

@(private = "file")
extruder_destroy :: proc(x: ^Extruder) {
	delete(x.points)
	delete(x.caps)
	delete(x.sides)
	delete(x.normals)
	delete(x.corners)
	delete(x.corner_point)
	delete(x.point_start)
	delete(x.by_point)
	delete(x.first)
	shape_points_destroy(&x.shape)
	delete(x.path)
	delete(x.run)
	delete(x.lines)
	delete(x.wave)
	delete(x.samples)
	delete(x.edge_lines)
	delete(x.edge_from)
	delete(x.chain_a)
	delete(x.chain_b)
	delete(x.seen)
	loops_destroy(&x.cap_loops)
	loops_destroy(&x.wall_loops)
	shape_points_destroy(&x.cap_shape)
	delete(x.cap_ids)
	delete(x.areas)
	delete(x.outline_of)
	delete(x.faces)
}

// A ring of the bevel: its offset `d` from the contours (out: positive),
// the cos of its angle on the bevel's quarter (its z from the walls' ends
// is bevel_thickness times it), its place from the caps (0) to the walls
// (bevel_segments), and in a run its wavefront time (the offset's size).
@(private = "file")
Ring_Entry :: struct {
	t, d, c, b: f64,
}

// Loops of the wavefront kept at a ring, in the contours' winding: their
// points (where, and their back and front bevel points), and their
// lengths.
@(private = "file")
Loops :: struct {
	at:          [dynamic][2]f64,
	ids:         [dynamic][2]u32,
	lens:        [dynamic]int,
}

@(private = "file")
loops_destroy :: proc(l: ^Loops) {
	delete(l.at)
	delete(l.ids)
	delete(l.lens)
}

@(private = "file")
loops_clear :: proc(l: ^Loops) {
	clear(&l.at)
	clear(&l.ids)
	clear(&l.lens)
}

// ExtrudeGeometry's addShape, without extrudePath. The bevel's rings are
// the shape offset as its straight skeleton offsets it (`wave_run`): each
// edge moved its distance along its normal, corners mitered where its
// lines meet (cut square to their bisector past `miter_limit`), parts
// that close up gone, parts that meet joined; the caps and walls are the
// loops at its first and last ring. The bevel's faces run between its
// rings and the times its wavefront changes, so they meet the caps and
// walls whatever the offsets do to the outline.
@(private = "file")
extrude_shape :: proc(x: ^Extruder, shapes: ^Shapes, i: int, o: Extrude_Options) {
	s := &x.shape
	shape_points(s, shapes, i)
	loops_clear(&x.cap_loops)
	loops_clear(&x.wall_loops)
	segs := o.bevel_segments
	clear(&x.path)
	for b in 0 ..= segs {
		a := f64(b) / f64(max(segs, 1)) * math.PI / 2
		append(&x.path, Ring_Entry{d = f64(o.bevel_offset) + f64(o.bevel_size) * math.sin(a), c = math.cos(a), b = f64(b)})
	}
	// Where the offsets cross the contours, a ring there: both directions'
	// wavefronts start from it.
	for k := 0; k + 1 < len(x.path); k += 1 {
		d0, d1 := x.path[k].d, x.path[k + 1].d
		if (d0 < 0 && d1 > 0) || (d0 > 0 && d1 < 0) {
			a := math.asin(clamp(-f64(o.bevel_offset) / f64(o.bevel_size), 0, 1))
			inject_at(&x.path, k + 1, Ring_Entry{d = 0, c = math.cos(a), b = a / (math.PI / 2) * f64(segs)})
			break
		}
	}
	// A wavefront out of the shape for the offsets out (or none moving),
	// one into it for those in.
	outward, inward := false, false
	for e in x.path {
		outward ||= e.d > 0
		inward ||= e.d < 0
	}
	for sigma in ([2]f64{1, -1}) {
		if (sigma > 0 && inward && !outward) || (sigma < 0 && !inward) {
			continue
		}
		clear(&x.run)
		for e in x.path {
			if sigma * e.d >= 0 {
				r := e
				r.t = sigma * e.d
				append(&x.run, r)
			}
		}
		forward := x.run[0].t <= x.run[len(x.run) - 1].t
		if !forward {
			slice.reverse(x.run[:])
		}
		wave_run(x, s, o, sigma, forward)
	}
	add_caps(x)
	add_walls(x, o)
}

// A line of the wavefront: a contour's edge (or a corner's chamfer)
// moving along `normal` at unit speed, its points x at time t where
// dot(normal, x) = c + t; `dir` runs along it the loop's way (`normal` is
// to its left).
@(private = "file")
Wave_Line :: struct {
	normal, dir: [2]f64,
	c:           f64,
}

// A vertex of the wavefront: at `p` at time `t0`, moving at `v` (where its
// lines meet); its loop's neighbours, the line from it to `next` (the one
// before it is `prev`'s), its last sample (-1: none yet), when its edge to
// `next` began, and whether its lines turn away from where they move
// (reflex: it can run into an edge ahead).
@(private = "file")
Wave_Vertex :: struct {
	p, v:             [2]f64,
	t0:               f64,
	prev, next, line: i32,
	sample:           i32,
	since:            f64,
	reflex, alive:    bool,
}

// Where a vertex was at time `t` (a ring, or an event that changed its
// edges): its back and front bevel points, and its sample before in the
// band (-1: none).
@(private = "file")
Wave_Sample :: struct {
	back, front: u32,
	t:           f64,
	prev:        i32,
}

// A wavefront event at time `t`: edge `a` -> `a.next` shrinks to nothing,
// or (`split`) reflex vertex `a` runs into edge `b` -> `b.next`.
@(private = "file")
Wave_Event :: struct {
	t:     f64,
	a, b:  i32,
	split: bool,
}

@(private = "file")
dot2 :: #force_inline proc "contextless" (a, b: [2]f64) -> f64 {
	return a.x * b.x + a.y * b.y
}

@(private = "file")
cross2 :: #force_inline proc "contextless" (a, b: [2]f64) -> f64 {
	return a.x * b.y - a.y * b.x
}

@(private = "file")
wave_at :: #force_inline proc "contextless" (w: Wave_Vertex, t: f64) -> [2]f64 {
	return w.p + w.v * (t - w.t0)
}

// Where two lines' meeting moves: v with dot(n1, v) = dot(n2, v) = 1; a
// line's own normal when they run on as one; still when they face each
// other.
@(private = "file")
wave_velocity :: proc "contextless" (a, b: Wave_Line) -> [2]f64 {
	det := cross2(a.normal, b.normal)
	if abs(det) < 1e-12 {
		return a.normal if dot2(a.normal, b.normal) > 0 else {}
	}
	return {(b.normal.y - a.normal.y) / det, (a.normal.x - b.normal.x) / det}
}

// The shape's contours as a wavefront moving out of it (`sigma` 1: each
// contour's left, away from the shape) or into it (-1: the loops run the
// other way), its vertices in the points' order. Moving, a reflex corner
// whose miter passes `miter` times the offset (or that turns right back)
// is cut: a chamfer line through it along the lines' bisector, from
// nothing.
@(private = "file")
wave_init :: proc(x: ^Extruder, s: ^Shape_Points, sigma: f64, moving: bool, miter: f64) {
	clear(&x.lines)
	clear(&x.wave)
	k := 0
	for n in s.rings {
		ring := s.points[k:][:n]
		base := i32(len(x.wave))
		for i in 0 ..< n {
			next := (i + 1) % n if sigma > 0 else (i + n - 1) % n
			prev := (i + n - 1) % n if sigma > 0 else (i + 1) % n
			d := ring[next] - ring[i]
			dir := d / math.sqrt(dot2(d, d))
			normal := [2]f64{-dir.y, dir.x}
			append(&x.lines, Wave_Line{normal, dir, dot2(normal, ring[i])})
			append(&x.wave, Wave_Vertex{p = ring[i], prev = base + i32(prev), next = base + i32(next), line = i32(len(x.lines) - 1), sample = -1, alive = true})
		}
		k += n
	}
	for i in 0 ..< len(x.wave) {
		w := x.wave[i]
		l_in, l_out := x.lines[x.wave[w.prev].line], x.lines[w.line]
		turn := cross2(l_in.dir, l_out.dir)
		back := abs(turn) < 1e-12 && dot2(l_in.dir, l_out.dir) < 0 && dot2(l_in.normal, x.wave[w.next].p - w.p) <= 0
		x.wave[i].reflex = turn < -1e-12 || back
		x.wave[i].v = wave_velocity(l_in, l_out)
		v := x.wave[i].v
		if !moving || !x.wave[i].reflex || (!back && dot2(v, v) <= miter * miter) {
			continue
		}
		nc := l_in.normal + l_out.normal
		nc = nc / math.sqrt(dot2(nc, nc)) if dot2(nc, nc) > 1e-18 else l_in.dir
		chamfer := Wave_Line{nc, {nc.y, -nc.x}, dot2(nc, w.p)}
		append(&x.lines, chamfer)
		append(&x.wave, Wave_Vertex{p = w.p, prev = i32(i), next = w.next, line = w.line, sample = -1, alive = true, v = wave_velocity(chamfer, l_out), reflex = cross2(chamfer.dir, l_out.dir) < -1e-12})
		second := i32(len(x.wave) - 1)
		x.wave[w.next].prev = second
		x.wave[i].next, x.wave[i].line = second, i32(len(x.lines) - 1)
		x.wave[i].v = wave_velocity(l_in, chamfer)
		x.wave[i].reflex = cross2(l_in.dir, chamfer.dir) < -1e-12
	}
}

// The wavefront from the contours through `x.run`'s rings (`forward`: from
// the caps' side), drawing its bevel faces (`wave_face`) between them and
// the events between them, and keeping the loops at the caps' and walls'
// rings. A ring samples every vertex and closes every edge's face; an
// event only the vertices whose edges it changes and those edges' faces.
@(private = "file")
wave_run :: proc(x: ^Extruder, s: ^Shape_Points, o: Extrude_Options, sigma: f64, forward: bool) {
	lo, hi := s.points[0], s.points[0]
	for p in s.points {
		lo, hi = {min(lo.x, p.x), min(lo.y, p.y)}, {max(hi.x, p.x), max(hi.y, p.y)}
	}
	eps := 1e-9 * max(hi.x - lo.x, hi.y - lo.y, 1e-9)
	entries := x.run[:]
	wave_init(x, s, sigma, entries[len(entries) - 1].t > eps, f64(o.miter_limit))
	clear(&x.samples)
	f := Wave_Faces{o = o, forward = forward, reversed = sigma < 0, eps = eps}
	segs := f64(o.bevel_segments)
	t: f64
	// A skeleton has at most a few events a vertex; past that the input is
	// degenerate and the rest of the run takes none.
	budget := 4 * len(x.wave) + 16
	// The next event to the last ring: the wavefront changes only at events,
	// so it stands until one is made.
	last := entries[len(entries) - 1].t
	ev, ok := wave_next_event(x, t, last, eps)
	for e, k in entries {
		for ; budget > 0; budget -= 1 {
			if !ok || ev.t >= e.t {
				break
			}
			// Its place on the bevel's quarter, by its offset (nothing
			// moves with no bevel size).
			f.c = 1
			if o.bevel_size != 0 {
				u := clamp((sigma * ev.t - f64(o.bevel_offset)) / f64(o.bevel_size), 0, 1)
				f.c = math.sqrt(1 - u * u)
			}
			wave_event(x, &f, ev, k > 0)
			t = ev.t
			ev, ok = wave_next_event(x, t, last, eps)
		}
		f.c = e.c
		for i in 0 ..< len(x.wave) {
			if x.wave[i].alive {
				wave_sample(x, &f, i32(i), e.t)
			}
		}
		if k > 0 {
			for i in 0 ..< len(x.wave) {
				if x.wave[i].alive {
					wave_face(x, &f, i32(i), e.t)
				}
			}
		}
		for &w in x.wave {
			if w.alive {
				w.since = e.t
				x.samples[w.sample].prev = -1
			}
		}
		t = e.t
		if e.b == 0 {
			wave_loops(x, &x.cap_loops, f.reversed, eps)
		}
		if e.b == segs {
			wave_loops(x, &x.wall_loops, f.reversed, eps)
		}
	}
}

// The first event from `t` to `until` (none: not ok): an edge shrinking to
// nothing, or a reflex vertex running into an edge ahead of it (within its
// ends then), edges first at a tie.
@(private = "file")
wave_next_event :: proc(x: ^Extruder, t, until, eps: f64) -> (ev: Wave_Event, ok: bool) {
	ev.t = until
	for a, ai in x.wave {
		if !a.alive {
			continue
		}
		b := x.wave[a.next]
		l := x.lines[a.line]
		rate := dot2(b.v - a.v, l.dir)
		if rate >= 0 {
			continue
		}
		te := t + max(dot2(wave_at(b, t) - wave_at(a, t), l.dir), 0) / -rate
		if te <= ev.t {
			ev, ok = {te, i32(ai), a.next, false}, true
		}
	}
	// The live edges' lines laid out for the sweep below: each reflex vertex
	// against all of them, the extent checked only for the few that it
	// would meet in time.
	clear(&x.edge_lines)
	clear(&x.edge_from)
	for p, pi in x.wave {
		if p.alive {
			l := x.lines[p.line]
			append(&x.edge_lines, [3]f64{l.normal.x, l.normal.y, l.c + t})
			append(&x.edge_from, i32(pi))
		}
	}
	for v, vi in x.wave {
		if !v.alive || !v.reflex {
			continue
		}
		pv := wave_at(v, t)
		for e, k in x.edge_lines {
			rate := e.x * v.v.x + e.y * v.v.y - 1
			gap := e.x * pv.x + e.y * pv.y - e.z
			if rate >= 0 || gap < -eps {
				continue
			}
			te := t + max(gap, 0) / -rate
			if te > ev.t || (ok && te == ev.t) {
				continue
			}
			pi := x.edge_from[k]
			p := x.wave[pi]
			if pi == i32(vi) || p.next == i32(vi) {
				continue
			}
			l := x.lines[p.line]
			q := x.wave[p.next]
			hit, from := wave_at(v, te), wave_at(p, te)
			along, length := dot2(hit - from, l.dir), dot2(wave_at(q, te) - from, l.dir)
			// At a neighbour's end of its edge it is that neighbour's
			// edge shrinking (an edge event), not a split.
			lo := eps if pi == v.next else -eps
			hi := length - eps if p.next == v.prev else length + eps
			if along < lo || along > hi {
				continue
			}
			ev, ok = {te, i32(vi), pi, true}, true
		}
	}
	return
}

// A vertex where lines `a` and `b` meet at `p` at time `t`.
@(private = "file")
wave_vertex :: proc(x: ^Extruder, p: [2]f64, t: f64, prev, next, line_in, line: i32) -> i32 {
	a, b := x.lines[line_in], x.lines[line]
	append(&x.wave, Wave_Vertex{p = p, t0 = t, v = wave_velocity(a, b), prev = prev, next = next, line = line, sample = -1, reflex = cross2(a.dir, b.dir) < -1e-12, alive = true})
	i := i32(len(x.wave) - 1)
	x.wave[prev].next, x.wave[next].prev = i, i
	return i
}

// An event at its time on the bevel (`drawing`: within the run's rings):
// the vertices whose edges it changes are sampled there and those edges'
// faces closed, then the event made (`wave_apply`) and its new vertices
// and edges start there.
@(private = "file")
wave_event :: proc(x: ^Extruder, f: ^Wave_Faces, ev: Wave_Event, drawing: bool) {
	if drawing {
		a := x.wave[ev.a]
		// The vertices to sample, and the edges (by their first vertex) that
		// end.
		touched: [5]i32
		ending: [3]i32
		if ev.split {
			p := x.wave[ev.b]
			touched = {a.prev, ev.a, a.next, ev.b, p.next}
			ending = {a.prev, ev.a, ev.b}
		} else {
			b := x.wave[ev.b]
			touched = {a.prev, ev.a, ev.b, b.next, b.next}
			ending = {a.prev, ev.a, ev.b}
		}
		for v in touched {
			wave_sample(x, f, v, ev.t)
		}
		for e, i in ending {
			if i == 0 || (e != ending[0] && (i == 1 || e != ending[1])) {
				wave_face(x, f, e, ev.t)
			}
		}
	}
	first := len(x.wave)
	wave_apply(x, ev)
	if !drawing {
		return
	}
	for i in first ..< len(x.wave) {
		if x.wave[i].alive {
			wave_sample(x, f, i32(i), ev.t)
			x.wave[i].since = ev.t
			x.wave[x.wave[i].prev].since = ev.t
		}
	}
}

// The event: a shrunk edge's ends become one vertex where its neighbours'
// lines meet; a reflex vertex splits the edge it ran into, its loop then
// two (or two loops one). Loops left with under three vertices are gone.
@(private = "file")
wave_apply :: proc(x: ^Extruder, ev: Wave_Event) {
	a := x.wave[ev.a]
	if !ev.split {
		x.wave[ev.a].alive, x.wave[ev.b].alive = false, false
		if a.prev == ev.b {
			return
		}
		b := x.wave[ev.b]
		c := wave_vertex(x, wave_at(a, ev.t), ev.t, a.prev, b.next, x.wave[a.prev].line, b.line)
		wave_prune(x, c)
		return
	}
	p := x.wave[ev.b]
	hit := wave_at(a, ev.t)
	x.wave[ev.a].alive = false
	// Before the hit: the vertex's way in, then the edge on to its end; after
	// it: the edge from its start, then the vertex's way out.
	one := wave_vertex(x, hit, ev.t, a.prev, p.next, x.wave[a.prev].line, p.line)
	two := wave_vertex(x, hit, ev.t, ev.b, a.next, p.line, a.line)
	wave_prune(x, one)
	wave_prune(x, two)
}

// Removes the loop through `i` when it has fewer than three vertices.
@(private = "file")
wave_prune :: proc(x: ^Extruder, i: i32) {
	n, j := 1, x.wave[i].next
	for ; j != i && n < 3; j = x.wave[j].next {
		n += 1
	}
	if j != i {
		return
	}
	for k := 0; k < n; k += 1 {
		x.wave[j].alive = false
		j = x.wave[j].next
	}
}

// How a run draws its faces: the bevel's options, the faces' winding (the
// rings' order from the caps, `forward`; the loops' way, `reversed` into
// the shape), the cos of the bevel's quarter where it samples, and how
// small an area is none.
@(private = "file")
Wave_Faces :: struct {
	o:                 Extrude_Options,
	forward, reversed: bool,
	c, eps:            f64,
}

// Vertex `i` sampled at time `t` (once a time): its back and front bevel
// points there.
@(private = "file")
wave_sample :: proc(x: ^Extruder, f: ^Wave_Faces, i: i32, t: f64) {
	w := &x.wave[i]
	if w.sample >= 0 && x.samples[w.sample].t == t {
		return
	}
	p := wave_at(w^, t)
	th, depth := f64(f.o.bevel_thickness), f64(f.o.depth)
	back := add_point(x, {p.x, p.y, -th * f.c})
	front := add_point(x, {p.x, p.y, depth + th * f.c})
	append(&x.samples, Wave_Sample{back, front, t, w.sample})
	w.sample = i32(len(x.samples) - 1)
}

@(private = "file")
add_point :: #force_inline proc(x: ^Extruder, p: [3]f64) -> u32 {
	append(&x.points, p)
	return u32(len(x.points) - 1)
}

// The face edge `a` -> `a.next` swept from when it began to time `t` (both
// ends sampled there), on the back and front bevels: its ends' samples
// between (each a chain, in time), zipped into triangles from the caps'
// side as three.js winds its layers' quads.
@(private = "file")
wave_face :: proc(x: ^Extruder, f: ^Wave_Faces, a: i32, t: f64) {
	w := x.wave[a]
	if w.since >= t {
		return
	}
	chain :: proc(x: ^Extruder, out: ^[dynamic]i32, from: i32, since: f64) {
		clear(out)
		for j := from; j >= 0 && x.samples[j].t >= since; j = x.samples[j].prev {
			append(out, j)
		}
	}
	chain(x, &x.chain_a, w.sample, w.since)
	chain(x, &x.chain_b, x.wave[w.next].sample, w.since)
	// Chains run back in time: the back bevel's lower rings are the caps'
	// side (early when forward), the front's the walls'.
	wave_zip(x, f, x.chain_a[:], x.chain_b[:], !f.forward, false)
	wave_zip(x, f, x.chain_a[:], x.chain_b[:], f.forward, true)
}

// Triangles between two chains of samples (`a`'s and `b`'s, late to early;
// `late_low`: the late ones are the lower rings), advancing the side whose
// next sample is lower (b at a tie, as three.js's quads split), wound as
// `side_quad`'s.
@(private = "file")
wave_zip :: proc(x: ^Extruder, f: ^Wave_Faces, a, b: []i32, late_low, front: bool) {
	m, n := len(a) - 1, len(b) - 1
	if m < 0 || n < 0 {
		return
	}
	// The k-th sample from the lower rings, and how high it is.
	at :: proc(x: ^Extruder, chain: []i32, k: int, late_low, front: bool) -> (u32, f64) {
		s := x.samples[chain[k if late_low else len(chain) - 1 - k]]
		return s.front if front else s.back, -s.t if late_low else s.t
	}
	a0, _ := at(x, a, 0, late_low, front)
	b0, _ := at(x, b, 0, late_low, front)
	pa, pb := x.points[a0], x.points[b0]
	if pa.xy == pb.xy {
		am, _ := at(x, a, m, late_low, front)
		bn, _ := at(x, b, n, late_low, front)
		pa, pb = x.points[am], x.points[bn]
	}
	from_x := abs(pa.y - pb.y) < abs(pa.x - pb.x)
	for i, j := 0, 0; i < m || j < n; {
		ai, _ := at(x, a, i, late_low, front)
		bj, _ := at(x, b, j, late_low, front)
		advance_a := j >= n
		if !advance_a && i < m {
			_, ha := at(x, a, i + 1, late_low, front)
			_, hb := at(x, b, j + 1, late_low, front)
			advance_a = ha < hb
		}
		if advance_a {
			an, _ := at(x, a, i + 1, late_low, front)
			if f.reversed {
				side_triangle(x, {ai, bj, an}, from_x, f.eps)
			} else {
				side_triangle(x, {ai, an, bj}, from_x, f.eps)
			}
			i += 1
		} else {
			bn, _ := at(x, b, j + 1, late_low, front)
			if f.reversed {
				side_triangle(x, {bj, bn, ai}, from_x, f.eps)
			} else {
				side_triangle(x, {bj, ai, bn}, from_x, f.eps)
			}
			j += 1
		}
	}
}

// A side quad from edge a -> b at its lower ring (`a_lo`, `b_lo`) to its
// upper (three.js's layer order), the loop running the contours' way or
// (`reversed`) against it; its triangles that have an area.
@(private = "file")
side_quad :: proc(x: ^Extruder, a_lo, b_lo, a_hi, b_hi: u32, reversed: bool, eps: f64) {
	pa, pb := x.points[a_lo], x.points[b_lo]
	if pa.xy == pb.xy {
		pa, pb = x.points[a_hi], x.points[b_hi]
	}
	from_x := abs(pa.y - pb.y) < abs(pa.x - pb.x)
	if reversed {
		side_triangle(x, {a_lo, b_lo, a_hi}, from_x, eps)
		side_triangle(x, {b_lo, b_hi, a_hi}, from_x, eps)
	} else {
		side_triangle(x, {b_lo, a_lo, b_hi}, from_x, eps)
		side_triangle(x, {a_lo, a_hi, b_hi}, from_x, eps)
	}
}

@(private = "file")
side_triangle :: proc(x: ^Extruder, corners: [3]u32, from_x: bool, eps: f64) {
	a, b, c := x.points[corners[0]], x.points[corners[1]], x.points[corners[2]]
	u, v := b - a, c - a
	n := [3]f64{u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.z, u.x * v.y - u.y * v.x}
	if n.x * n.x + n.y * n.y + n.z * n.z <= eps * eps * eps * eps {
		return
	}
	append(&x.sides, Side_Triangle{corners, from_x})
}

// The wavefront's live loops into `loops`, each from its first vertex in
// the contours' winding (`reversed`: against the loops' way), points that
// fall together kept once; loops of under three points left out.
@(private = "file")
wave_loops :: proc(x: ^Extruder, loops: ^Loops, reversed: bool, eps: f64) {
	resize(&x.seen, len(x.wave))
	slice.fill(x.seen[:], false)
	for w, i in x.wave {
		if !w.alive || x.seen[i] {
			continue
		}
		start := len(loops.at)
		for j := i32(i); !x.seen[j]; j = x.wave[j].prev if reversed else x.wave[j].next {
			x.seen[j] = true
			q := x.samples[x.wave[j].sample]
			p := x.points[q.back].xy
			if len(loops.at) > start {
				d := p - loops.at[len(loops.at) - 1]
				if dot2(d, d) <= eps * eps {
					continue
				}
			}
			append(&loops.at, p)
			append(&loops.ids, [2]u32{q.back, q.front})
		}
		for len(loops.at) - start > 1 {
			d := loops.at[start] - loops.at[len(loops.at) - 1]
			if dot2(d, d) > eps * eps {
				break
			}
			pop(&loops.at)
			pop(&loops.ids)
		}
		if len(loops.at) - start < 3 {
			resize(&loops.at, start)
			resize(&loops.ids, start)
			continue
		}
		append(&loops.lens, len(loops.at) - start)
	}
}

// The caps from the loops at the caps' ring: each outline (clockwise) with
// the holes inside it (the smallest outline about each), triangulated,
// the back's reversed.
@(private = "file")
add_caps :: proc(x: ^Extruder) {
	l := &x.cap_loops
	resize(&x.areas, len(l.lens))
	resize(&x.outline_of, len(l.lens))
	k := 0
	for n, r in l.lens {
		x.areas[r] = ring_area(l.at[k:][:n])
		k += n
	}
	// Each hole's outline: the smallest one about its first point.
	k = 0
	for n, r in l.lens {
		x.outline_of[r] = -1
		if x.areas[r] > 0 {
			best := max(f64)
			j := 0
			for m, q in l.lens {
				if x.areas[q] < 0 && -x.areas[q] < best && inside_loop(l.at[k], l.at[j:][:m]) {
					best, x.outline_of[r] = -x.areas[q], q
				}
				j += m
			}
		}
		k += n
	}
	s := &x.cap_shape
	k = 0
	for n, r in l.lens {
		if x.areas[r] >= 0 {
			k += n
			continue
		}
		clear(&s.points)
		clear(&s.rings)
		clear(&x.cap_ids)
		append(&s.points, ..l.at[k:][:n])
		append(&x.cap_ids, ..l.ids[k:][:n])
		append(&s.rings, n)
		j := 0
		for m, q in l.lens {
			if x.outline_of[q] == r {
				append(&s.points, ..l.at[j:][:m])
				append(&x.cap_ids, ..l.ids[j:][:m])
				append(&s.rings, m)
			}
			j += m
		}
		clear(&x.faces)
		triangulate_rings(s, s.points[:], &x.faces)
		for t := 0; t + 2 < len(x.faces); t += 3 {
			f := x.faces[t:][:3]
			append(&x.caps, [3]u32{x.cap_ids[f[2]][0], x.cap_ids[f[1]][0], x.cap_ids[f[0]][0]})
		}
		for t := 0; t + 2 < len(x.faces); t += 3 {
			f := x.faces[t:][:3]
			append(&x.caps, [3]u32{x.cap_ids[f[0]][1], x.cap_ids[f[1]][1], x.cap_ids[f[2]][1]})
		}
		k += n
	}
}

// Whether `p` is inside a loop (even-odd).
@(private = "file")
inside_loop :: proc "contextless" (p: [2]f64, loop: [][2]f64) -> bool {
	inside := false
	for i, j := 0, len(loop) - 1; i < len(loop); j, i = i, i + 1 {
		a, b := loop[i], loop[j]
		if (a.y > p.y) != (b.y > p.y) && p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x {
			inside = !inside
		}
	}
	return inside
}

// The walls along the loops at the walls' ring, from the back bevel's end
// (z 0) to the front's (z depth) in `steps`.
@(private = "file")
add_walls :: proc(x: ^Extruder, o: Extrude_Options) {
	l := &x.wall_loops
	k := 0
	for n in l.lens {
		ids := l.ids[k:][:n]
		for i in 0 ..< n {
			a, b := ids[i], ids[(i + 1) % n]
			a_lo, b_lo := a[0], b[0]
			for st in 1 ..= o.steps {
				a_hi, b_hi := a[1], b[1]
				if st < o.steps {
					z := f64(o.depth) / f64(o.steps) * f64(st)
					a_hi = add_point(x, {x.points[a[0]].x, x.points[a[0]].y, z})
					b_hi = add_point(x, {x.points[b[0]].x, x.points[b[0]].y, z})
				}
				side_quad(x, a_lo, b_lo, a_hi, b_hi, false, 0)
				a_lo, b_lo = a_hi, b_hi
			}
		}
		k += n
	}
}

// A triangle's unit normal (counter-clockwise front), +Z when degenerate.
@(private = "file")
face_normal :: proc "contextless" (a, b, c: [3]f64) -> Vec3 {
	u, v := b - a, c - a
	n := [3]f64{u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.z, u.x * v.y - u.y * v.x}
	l := math.sqrt(n.x * n.x + n.y * n.y + n.z * n.z)
	return vec3(n / l) if l > 1e-30 else {0, 0, 1}
}

// The side triangles' corner normals: each face's own (three.js), or with a
// crease angle the sum of the side faces at that point within the angle of
// this one, each weighted by its corner's angle there (so a quad's two
// triangles count as the quad, whichever way its diagonal runs).
@(private = "file")
side_normals :: proc(x: ^Extruder, crease_angle: f32) {
	n := len(x.sides)
	resize(&x.normals, n * 3)
	units := make([]Vec3, n)
	defer delete(units)
	for t, i in x.sides {
		units[i] = face_normal(x.points[t.corners[0]], x.points[t.corners[1]], x.points[t.corners[2]])
	}
	if crease_angle <= 0 {
		for i in 0 ..< n {
			x.normals[i * 3], x.normals[i * 3 + 1], x.normals[i * 3 + 2] = units[i], units[i], units[i]
		}
		return
	}
	// The faces at each point, grouped by point (counting sort).
	start := make([]u32, len(x.points) + 1)
	defer delete(start)
	for t in x.sides {
		for p in t.corners {
			start[p + 1] += 1
		}
	}
	for i in 1 ..< len(start) {
		start[i] += start[i - 1]
	}
	at := make([]u32, n * 3)
	defer delete(at)
	fill := make([]u32, len(x.points))
	defer delete(fill)
	for t, i in x.sides {
		for p in t.corners {
			at[start[p] + fill[p]] = u32(i)
			fill[p] += 1
		}
	}
	cos_crease := math.cos(min(crease_angle, math.PI))
	for t, i in x.sides {
		for p, k in t.corners {
			sum: Vec3
			for f in at[start[p]:start[p + 1]] {
				u := units[f]
				if u.x * units[i].x + u.y * units[i].y + u.z * units[i].z >= cos_crease {
					sum += u * corner_angle(x, x.sides[f].corners, p)
				}
			}
			l := math.sqrt(sum.x * sum.x + sum.y * sum.y + sum.z * sum.z)
			x.normals[i * 3 + k] = sum / l if l > 1e-20 else units[i]
		}
	}
}

// The angle of triangle `corners` at its point `p`.
@(private = "file")
corner_angle :: proc(x: ^Extruder, corners: [3]u32, p: u32) -> f32 {
	k := 0 if corners[0] == p else 1 if corners[1] == p else 2
	o := x.points[p]
	u := x.points[corners[(k + 1) % 3]] - o
	v := x.points[corners[(k + 2) % 3]] - o
	lu := math.sqrt(u.x * u.x + u.y * u.y + u.z * u.z)
	lv := math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
	if lu < 1e-30 || lv < 1e-30 {
		return 0
	}
	return f32(math.acos(clamp((u.x * v.x + u.y * v.y + u.z * v.z) / (lu * lv), -1, 1)))
}

// ---- earcut -------------------------------------------------------------------------------
// A port of mapbox/earcut (ISC, LICENSES/earcut-ISC.txt) as three.js
// carries it, over a node pool: links are indices, -1 for none.

@(private = "file")
Ear_Node :: struct {
	i:              u32, // the point's index
	x, y:           f64,
	prev, next:     i32, // in the polygon ring
	z:              i32, // z-order curve value
	prev_z, next_z: i32, // in z order
	steiner:        bool,
}

@(private = "file")
Earcut :: struct {
	nodes:                 [dynamic]Ear_Node,
	triangles:             ^[dynamic]u32,
	min_x, min_y, inv_size: f64,
}

@(private = "file")
earcut :: proc(e: ^Earcut, data: [][2]f64, outer_len: int, hole_starts, hole_ends: []int) {
	reserve(&e.nodes, len(data) * 3 / 2 + 8)
	outer := linked_list(e, data, 0, outer_len, true)
	if outer < 0 || e.nodes[outer].next == e.nodes[outer].prev {
		return
	}
	if len(hole_starts) > 0 {
		outer = eliminate_holes(e, data, hole_starts, hole_ends, outer)
	}
	if len(data) > 80 {
		lo, hi := data[0], data[0]
		for p in data[1:outer_len] {
			lo, hi = {min(lo.x, p.x), min(lo.y, p.y)}, {max(hi.x, p.x), max(hi.y, p.y)}
		}
		e.min_x, e.min_y = lo.x, lo.y
		size := max(hi.x - lo.x, hi.y - lo.y)
		e.inv_size = 32767 / size if size != 0 else 0
	}
	earcut_linked(e, outer, 0)
}

@(private = "file")
N :: #force_inline proc "contextless" (e: ^Earcut, i: i32) -> ^Ear_Node {
	return &e.nodes[i]
}

@(private = "file")
linked_list :: proc(e: ^Earcut, data: [][2]f64, start, end: int, clockwise: bool) -> i32 {
	last: i32 = -1
	sum: f64
	for i, j := start, end - 1; i < end; j, i = i, i + 1 {
		sum += (data[j].x - data[i].x) * (data[i].y + data[j].y)
	}
	if clockwise == (sum > 0) {
		for i in start ..< end {
			last = insert_node(e, u32(i), data[i], last)
		}
	} else {
		for i := end - 1; i >= start; i -= 1 {
			last = insert_node(e, u32(i), data[i], last)
		}
	}
	if last >= 0 && equals(e, last, N(e, last).next) {
		remove_node(e, last)
		last = N(e, last).next
	}
	return last
}

// Drops repeated and collinear points.
@(private = "file")
filter_points :: proc(e: ^Earcut, start: i32, end: i32 = -1) -> i32 {
	if start < 0 {
		return start
	}
	end := end if end >= 0 else start
	p := start
	for {
		again := false
		if !N(e, p).steiner && (equals(e, p, N(e, p).next) || area(e, N(e, p).prev, p, N(e, p).next) == 0) {
			remove_node(e, p)
			p = N(e, p).prev
			end = p
			if p == N(e, p).next {
				break
			}
			again = true
		} else {
			p = N(e, p).next
		}
		if !again && p == end {
			break
		}
	}
	return end
}

@(private = "file")
earcut_linked :: proc(e: ^Earcut, ear_in: i32, pass: int) {
	ear := ear_in
	if ear < 0 {
		return
	}
	if pass == 0 && e.inv_size != 0 {
		index_curve(e, ear)
	}
	stop := ear
	for N(e, ear).prev != N(e, ear).next {
		prev, next := N(e, ear).prev, N(e, ear).next
		if is_ear_hashed(e, ear) if e.inv_size != 0 else is_ear(e, ear) {
			append(e.triangles, N(e, prev).i, N(e, ear).i, N(e, next).i)
			remove_node(e, ear)
			ear = N(e, next).next
			stop = ear
			continue
		}
		ear = next
		if ear == stop {
			switch pass {
			case 0:
				earcut_linked(e, filter_points(e, ear), 1)
			case 1:
				ear = cure_local_intersections(e, filter_points(e, ear))
				earcut_linked(e, ear, 2)
			case 2:
				split_earcut(e, ear)
			}
			break
		}
	}
}

@(private = "file")
is_ear :: proc(e: ^Earcut, ear: i32) -> bool {
	a, b, c := N(e, ear).prev, ear, N(e, ear).next
	if area(e, a, b, c) >= 0 {
		return false // reflex
	}
	pa, pb, pc := N(e, a)^, N(e, b)^, N(e, c)^
	x0, y0 := min(pa.x, pb.x, pc.x), min(pa.y, pb.y, pc.y)
	x1, y1 := max(pa.x, pb.x, pc.x), max(pa.y, pb.y, pc.y)
	for p := pc.next; p != a; p = N(e, p).next {
		q := N(e, p)
		if q.x >= x0 && q.x <= x1 && q.y >= y0 && q.y <= y1 &&
		   point_in_triangle_except_first(pa.x, pa.y, pb.x, pb.y, pc.x, pc.y, q.x, q.y) &&
		   area(e, q.prev, p, q.next) >= 0 {
			return false
		}
	}
	return true
}

@(private = "file")
is_ear_hashed :: proc(e: ^Earcut, ear: i32) -> bool {
	a, b, c := N(e, ear).prev, ear, N(e, ear).next
	if area(e, a, b, c) >= 0 {
		return false
	}
	pa, pb, pc := N(e, a)^, N(e, b)^, N(e, c)^
	x0, y0 := min(pa.x, pb.x, pc.x), min(pa.y, pb.y, pc.y)
	x1, y1 := max(pa.x, pb.x, pc.x), max(pa.y, pb.y, pc.y)
	min_z, max_z := z_order(e, x0, y0), z_order(e, x1, y1)
	blocks :: proc(e: ^Earcut, p, a, c: i32, pa, pb, pc: Ear_Node, x0, y0, x1, y1: f64) -> bool {
		q := N(e, p)
		return q.x >= x0 && q.x <= x1 && q.y >= y0 && q.y <= y1 && p != a && p != c &&
		       point_in_triangle_except_first(pa.x, pa.y, pb.x, pb.y, pc.x, pc.y, q.x, q.y) && area(e, q.prev, p, q.next) >= 0
	}
	p, n := N(e, ear).prev_z, N(e, ear).next_z
	for p >= 0 && N(e, p).z >= min_z && n >= 0 && N(e, n).z <= max_z {
		if blocks(e, p, a, c, pa, pb, pc, x0, y0, x1, y1) {
			return false
		}
		p = N(e, p).prev_z
		if blocks(e, n, a, c, pa, pb, pc, x0, y0, x1, y1) {
			return false
		}
		n = N(e, n).next_z
	}
	for p >= 0 && N(e, p).z >= min_z {
		if blocks(e, p, a, c, pa, pb, pc, x0, y0, x1, y1) {
			return false
		}
		p = N(e, p).prev_z
	}
	for n >= 0 && N(e, n).z <= max_z {
		if blocks(e, n, a, c, pa, pb, pc, x0, y0, x1, y1) {
			return false
		}
		n = N(e, n).next_z
	}
	return true
}

@(private = "file")
cure_local_intersections :: proc(e: ^Earcut, start_in: i32) -> i32 {
	start := start_in
	p := start
	for {
		a, b := N(e, p).prev, N(e, N(e, p).next).next
		if !equals(e, a, b) && intersects(e, a, p, N(e, p).next, b) && locally_inside(e, a, b) && locally_inside(e, b, a) {
			append(e.triangles, N(e, a).i, N(e, p).i, N(e, b).i)
			remove_node(e, p)
			remove_node(e, N(e, p).next)
			p, start = b, b
		}
		p = N(e, p).next
		if p == start {
			break
		}
	}
	return filter_points(e, p)
}

@(private = "file")
split_earcut :: proc(e: ^Earcut, start: i32) {
	a := start
	for {
		for b := N(e, N(e, a).next).next; b != N(e, a).prev; b = N(e, b).next {
			if N(e, a).i != N(e, b).i && is_valid_diagonal(e, a, b) {
				c := split_polygon(e, a, b)
				a = filter_points(e, a, N(e, a).next)
				c = filter_points(e, c, N(e, c).next)
				earcut_linked(e, a, 0)
				earcut_linked(e, c, 0)
				return
			}
		}
		a = N(e, a).next
		if a == start {
			break
		}
	}
}

@(private = "file")
eliminate_holes :: proc(e: ^Earcut, data: [][2]f64, hole_starts, hole_ends: []int, outer_in: i32) -> i32 {
	outer := outer_in
	queue := make([dynamic]i32, 0, len(hole_starts))
	defer delete(queue)
	for start, h in hole_starts {
		list := linked_list(e, data, start, hole_ends[h], false)
		if list < 0 {
			continue
		}
		if list == N(e, list).next {
			N(e, list).steiner = true
		}
		append(&queue, leftmost(e, list))
	}
	// A stable insertion sort by x, y, then the slope to the next point.
	less :: proc(e: ^Earcut, a, b: i32) -> bool {
		na, nb := N(e, a), N(e, b)
		if na.x != nb.x {
			return na.x < nb.x
		}
		if na.y != nb.y {
			return na.y < nb.y
		}
		sa := (N(e, na.next).y - na.y) / (N(e, na.next).x - na.x)
		sb := (N(e, nb.next).y - nb.y) / (N(e, nb.next).x - nb.x)
		return sa < sb
	}
	for i in 1 ..< len(queue) {
		for j := i; j > 0 && less(e, queue[j], queue[j - 1]); j -= 1 {
			queue[j], queue[j - 1] = queue[j - 1], queue[j]
		}
	}
	for h in queue {
		outer = eliminate_hole(e, h, outer)
	}
	return outer
}

@(private = "file")
eliminate_hole :: proc(e: ^Earcut, hole, outer: i32) -> i32 {
	bridge := find_hole_bridge(e, hole, outer)
	if bridge < 0 {
		return outer
	}
	bridge_reverse := split_polygon(e, bridge, hole)
	filter_points(e, bridge_reverse, N(e, bridge_reverse).next)
	return filter_points(e, bridge, N(e, bridge).next)
}

// David Eberly's algorithm: a point of the outline the hole's leftmost point
// can bridge to.
@(private = "file")
find_hole_bridge :: proc(e: ^Earcut, hole, outer: i32) -> i32 {
	p := outer
	hx, hy := N(e, hole).x, N(e, hole).y
	qx := math.inf_f64(-1)
	m: i32 = -1
	if equals(e, hole, p) {
		return p
	}
	for {
		np, nn := N(e, p), N(e, N(e, p).next)
		if equals(e, hole, np.next) {
			return np.next
		} else if hy <= np.y && hy >= nn.y && nn.y != np.y {
			x := np.x + (hy - np.y) * (nn.x - np.x) / (nn.y - np.y)
			if x <= hx && x > qx {
				qx = x
				m = p if np.x < nn.x else np.next
				if x == hx {
					return m // the hole touches the outline's segment
				}
			}
		}
		p = np.next
		if p == outer {
			break
		}
	}
	if m < 0 {
		return -1
	}
	stop := m
	mx, my := N(e, m).x, N(e, m).y
	tan_min := math.inf_f64(1)
	p = m
	for {
		q := N(e, p)
		if hx >= q.x && q.x >= mx && hx != q.x &&
		   point_in_triangle(hx if hy < my else qx, hy, mx, my, qx if hy < my else hx, hy, q.x, q.y) {
			tan := abs(hy - q.y) / (hx - q.x)
			if locally_inside(e, p, hole) &&
			   (tan < tan_min || (tan == tan_min && (q.x > N(e, m).x || (q.x == N(e, m).x && sector_contains_sector(e, m, p))))) {
				m = p
				tan_min = tan
			}
		}
		p = q.next
		if p == stop {
			break
		}
	}
	return m
}

@(private = "file")
sector_contains_sector :: proc(e: ^Earcut, m, p: i32) -> bool {
	return area(e, N(e, m).prev, m, N(e, p).prev) < 0 && area(e, N(e, p).next, m, N(e, m).next) < 0
}

@(private = "file")
index_curve :: proc(e: ^Earcut, start: i32) {
	p := start
	for {
		q := N(e, p)
		if q.z == 0 {
			q.z = z_order(e, q.x, q.y)
		}
		q.prev_z, q.next_z = q.prev, q.next
		p = q.next
		if p == start {
			break
		}
	}
	N(e, N(e, p).prev_z).next_z = -1
	N(e, p).prev_z = -1
	sort_linked(e, p)
}

// Simon Tatham's linked-list merge sort, by z.
@(private = "file")
sort_linked :: proc(e: ^Earcut, list_in: i32) -> i32 {
	list := list_in
	in_size := 1
	for {
		p := list
		list = -1
		tail: i32 = -1
		merges := 0
		for p >= 0 {
			merges += 1
			q := p
			p_size := 0
			for _ in 0 ..< in_size {
				p_size += 1
				q = N(e, q).next_z
				if q < 0 {
					break
				}
			}
			q_size := in_size
			for p_size > 0 || (q_size > 0 && q >= 0) {
				el: i32
				if p_size != 0 && (q_size == 0 || q < 0 || N(e, p).z <= N(e, q).z) {
					el = p
					p = N(e, p).next_z
					p_size -= 1
				} else {
					el = q
					q = N(e, q).next_z
					q_size -= 1
				}
				if tail >= 0 {
					N(e, tail).next_z = el
				} else {
					list = el
				}
				N(e, el).prev_z = tail
				tail = el
			}
			p = q
		}
		N(e, tail).next_z = -1
		in_size *= 2
		if merges <= 1 {
			break
		}
	}
	return list
}

// A point's z-order (Morton code) in the outline's bounds, 15 bits a side.
@(private = "file")
z_order :: proc "contextless" (e: ^Earcut, px, py: f64) -> i32 {
	x := u32(i32((px - e.min_x) * e.inv_size))
	y := u32(i32((py - e.min_y) * e.inv_size))
	x = (x | (x << 8)) & 0x00FF00FF
	x = (x | (x << 4)) & 0x0F0F0F0F
	x = (x | (x << 2)) & 0x33333333
	x = (x | (x << 1)) & 0x55555555
	y = (y | (y << 8)) & 0x00FF00FF
	y = (y | (y << 4)) & 0x0F0F0F0F
	y = (y | (y << 2)) & 0x33333333
	y = (y | (y << 1)) & 0x55555555
	return i32(x | (y << 1))
}

@(private = "file")
leftmost :: proc(e: ^Earcut, start: i32) -> i32 {
	p, left := start, start
	for {
		q, l := N(e, p), N(e, left)
		if q.x < l.x || (q.x == l.x && q.y < l.y) {
			left = p
		}
		p = q.next
		if p == start {
			break
		}
	}
	return left
}

@(private = "file")
point_in_triangle :: proc "contextless" (ax, ay, bx, by, cx, cy, px, py: f64) -> bool {
	return (cx - px) * (ay - py) >= (ax - px) * (cy - py) &&
	       (ax - px) * (by - py) >= (bx - px) * (ay - py) &&
	       (bx - px) * (cy - py) >= (cx - px) * (by - py)
}

@(private = "file")
point_in_triangle_except_first :: proc "contextless" (ax, ay, bx, by, cx, cy, px, py: f64) -> bool {
	return !(ax == px && ay == py) && point_in_triangle(ax, ay, bx, by, cx, cy, px, py)
}

// Whether a diagonal between a and b is inside the polygon and crosses none
// of its edges.
@(private = "file")
is_valid_diagonal :: proc(e: ^Earcut, a, b: i32) -> bool {
	na, nb := N(e, a), N(e, b)
	return N(e, na.next).i != nb.i && N(e, na.prev).i != nb.i && !intersects_polygon(e, a, b) &&
	       ((locally_inside(e, a, b) && locally_inside(e, b, a) && middle_inside(e, a, b) &&
	            (area(e, na.prev, a, nb.prev) != 0 || area(e, a, nb.prev, b) != 0)) ||
	        (equals(e, a, b) && area(e, na.prev, a, na.next) > 0 && area(e, nb.prev, b, nb.next) > 0))
}

@(private = "file")
area :: #force_inline proc "contextless" (e: ^Earcut, p, q, r: i32) -> f64 {
	np, nq, nr := &e.nodes[p], &e.nodes[q], &e.nodes[r]
	return (nq.y - np.y) * (nr.x - nq.x) - (nq.x - np.x) * (nr.y - nq.y)
}

@(private = "file")
equals :: #force_inline proc "contextless" (e: ^Earcut, a, b: i32) -> bool {
	return e.nodes[a].x == e.nodes[b].x && e.nodes[a].y == e.nodes[b].y
}

@(private = "file")
intersects :: proc "contextless" (e: ^Earcut, p1, q1, p2, q2: i32) -> bool {
	sign :: proc "contextless" (v: f64) -> int {
		return 1 if v > 0 else -1 if v < 0 else 0
	}
	o1 := sign(area(e, p1, q1, p2))
	o2 := sign(area(e, p1, q1, q2))
	o3 := sign(area(e, p2, q2, p1))
	o4 := sign(area(e, p2, q2, q1))
	if o1 != o2 && o3 != o4 {
		return true
	}
	if o1 == 0 && on_segment(e, p1, p2, q1) {return true}
	if o2 == 0 && on_segment(e, p1, q2, q1) {return true}
	if o3 == 0 && on_segment(e, p2, p1, q2) {return true}
	if o4 == 0 && on_segment(e, p2, q1, q2) {return true}
	return false
}

// For collinear p, q, r: whether q lies on the segment pr.
@(private = "file")
on_segment :: proc "contextless" (e: ^Earcut, p, q, r: i32) -> bool {
	np, nq, nr := &e.nodes[p], &e.nodes[q], &e.nodes[r]
	return nq.x <= max(np.x, nr.x) && nq.x >= min(np.x, nr.x) && nq.y <= max(np.y, nr.y) && nq.y >= min(np.y, nr.y)
}

@(private = "file")
intersects_polygon :: proc(e: ^Earcut, a, b: i32) -> bool {
	ia, ib := N(e, a).i, N(e, b).i
	p := a
	for {
		q := N(e, p)
		if q.i != ia && N(e, q.next).i != ia && q.i != ib && N(e, q.next).i != ib && intersects(e, p, q.next, a, b) {
			return true
		}
		p = q.next
		if p == a {
			break
		}
	}
	return false
}

@(private = "file")
locally_inside :: proc "contextless" (e: ^Earcut, a, b: i32) -> bool {
	na := &e.nodes[a]
	if area(e, na.prev, a, na.next) < 0 {
		return area(e, a, b, na.next) >= 0 && area(e, a, na.prev, b) >= 0
	}
	return area(e, a, b, na.prev) < 0 || area(e, a, na.next, b) < 0
}

@(private = "file")
middle_inside :: proc(e: ^Earcut, a, b: i32) -> bool {
	p := a
	inside := false
	px, py := (N(e, a).x + N(e, b).x) / 2, (N(e, a).y + N(e, b).y) / 2
	for {
		q, n := N(e, p), N(e, N(e, p).next)
		if (q.y > py) != (n.y > py) && n.y != q.y && px < (n.x - q.x) * (py - q.y) / (n.y - q.y) + q.x {
			inside = !inside
		}
		p = q.next
		if p == a {
			break
		}
	}
	return inside
}

// Links a to b with a bridge: two rings if they were one, one if two; the
// copies of a and b start the other ring.
@(private = "file")
split_polygon :: proc(e: ^Earcut, a, b: i32) -> i32 {
	a2 := new_node(e, N(e, a).i, {N(e, a).x, N(e, a).y})
	b2 := new_node(e, N(e, b).i, {N(e, b).x, N(e, b).y})
	an, bp := N(e, a).next, N(e, b).prev
	N(e, a).next = b
	N(e, b).prev = a
	N(e, a2).next = an
	N(e, an).prev = a2
	N(e, b2).next = a2
	N(e, a2).prev = b2
	N(e, bp).next = b2
	N(e, b2).prev = bp
	return b2
}

@(private = "file")
new_node :: proc(e: ^Earcut, i: u32, p: [2]f64) -> i32 {
	append(&e.nodes, Ear_Node{i = i, x = p.x, y = p.y, prev = -1, next = -1, prev_z = -1, next_z = -1})
	return i32(len(e.nodes) - 1)
}

@(private = "file")
insert_node :: proc(e: ^Earcut, i: u32, p: [2]f64, last: i32) -> i32 {
	n := new_node(e, i, p)
	if last < 0 {
		N(e, n).prev, N(e, n).next = n, n
	} else {
		N(e, n).next = N(e, last).next
		N(e, n).prev = last
		N(e, N(e, last).next).prev = n
		N(e, last).next = n
	}
	return n
}

@(private = "file")
remove_node :: proc "contextless" (e: ^Earcut, p: i32) {
	n := &e.nodes[p]
	e.nodes[n.next].prev = n.prev
	e.nodes[n.prev].next = n.next
	if n.prev_z >= 0 {
		e.nodes[n.prev_z].next_z = n.next_z
	}
	if n.next_z >= 0 {
		e.nodes[n.next_z].prev_z = n.prev_z
	}
}
