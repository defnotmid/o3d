package o3d

import "core:math"
import "core:slice"

// Shapes: three.js's Shape, ShapeUtils, ShapePath.toShapes, ShapeGeometry
// and ExtrudeGeometry, over outlines already flattened to points (what
// three.js's `shape.extractPoints(curveSegments)` gives: fonts' and SVGs'
// curves are flattened by the caller, so there is no curveSegments here).
//
// A shape's geometry is made once and kept like any other (`Geometries`
// never frees): build it when the outline is first seen, keep the id, and
// move it per frame through the draw's transform. Unlike the primitives,
// shapes are not looked up by their parameters, so calling `extrude_geometry`
// every frame makes a new mesh every frame (`geometries_made` shows it).

// An outline and its holes, as points in the XY plane (three.js's Shape
// after extractPoints: `shape` and `holes`). Either winding is taken: the
// geometries wind the outline clockwise and the holes counter-clockwise, as
// three.js does. A closing point equal to the first is ignored.
Shape :: struct {
	contour: []Vec2,
	holes:   [][]Vec2,
}

// SVG's fill rules, for `shapes_from_contours`.
Fill_Rule :: enum u8 {
	Nonzero,
	Even_Odd,
}

// ShapeUtils.area: the signed area, positive when counter-clockwise (Y up).
shape_area :: proc "contextless" (contour: []Vec2) -> f32 {
	return f32(signed_area_f64(contour))
}

// ShapeUtils.isClockWise.
shape_is_clockwise :: proc "contextless" (contour: []Vec2) -> bool {
	return signed_area_f64(contour) < 0
}

@(private = "file")
signed_area_f64 :: proc "contextless" (contour: []Vec2) -> f64 {
	n := len(contour)
	a: f64
	for p, q := n - 1, 0; q < n; p, q = q, q + 1 {
		a += f64(contour[p].x) * f64(contour[q].y) - f64(contour[q].x) * f64(contour[p].y)
	}
	return a * 0.5
}

// ShapeUtils.triangulateShape: earcut (mapbox/earcut, as three.js carries
// it) of an outline with holes. Appends three indices per triangle into the
// outline's points followed by each hole's, in order, every triangle
// counter-clockwise (Y up) whatever the input's winding; returns how many
// triangles it appended. Holes of fewer than three points are skipped.
triangulate_shape :: proc(contour: []Vec2, holes: [][]Vec2, triangles: ^[dynamic]u32) -> int {
	n := len(contour)
	for h in holes {
		n += len(h)
	}
	data := make([][2]f64, n)
	defer delete(data)
	// Each kept hole's range of points (a skipped hole's points stay in
	// `data`, so the indices still count them).
	hole_starts := make([dynamic]int, 0, len(holes))
	defer delete(hole_starts)
	hole_ends := make([dynamic]int, 0, len(holes))
	defer delete(hole_ends)
	k := 0
	for p in contour {
		data[k] = {f64(p.x), f64(p.y)}
		k += 1
	}
	for h in holes {
		if len(h) >= 3 {
			append(&hole_starts, k)
			append(&hole_ends, k + len(h))
		}
		for p in h {
			data[k] = {f64(p.x), f64(p.y)}
			k += 1
		}
	}
	before := len(triangles)
	e := Earcut{triangles = triangles}
	defer delete(e.nodes)
	earcut(&e, data, len(contour), hole_starts[:], hole_ends[:])
	return (len(triangles) - before) / 3
}

// ShapePath.toShapes: groups closed contours (as a font's glyph or an SVG
// path gives them: any windings, nested, with counters like the holes of
// "B" and "8") into shapes with holes by `fill_rule`'s containment. A
// contour that does not change what is filled (a second copy, a
// same-direction contour inside a nonzero fill) is dropped, and a solid
// inside a hole is a shape of its own. Shapes refer to `contours`' slices
// (not copied: keep them alive); free the result with `shapes_destroy`.
shapes_from_contours :: proc(contours: [][]Vec2, fill_rule := Fill_Rule.Nonzero, allocator := context.allocator) -> []Shape {
	Entry :: struct {
		contour:        int,
		lo, hi:         [2]f64,
		interior:       [2]f64,
		abs_area:       f64,
		winding:        int,
		container:      int, // an entry, -1 for none
		exclude, hole:  bool,
		shape:          int,
	}
	inside :: proc(w: int, rule: Fill_Rule) -> bool {
		return w != 0 if rule == .Nonzero else w & 1 != 0
	}
	entries := make([dynamic]Entry, 0, len(contours))
	defer delete(entries)
	for c, ci in contours {
		if len(c) < 3 {
			continue
		}
		area := signed_area_f64(c)
		if area == 0 {
			continue
		}
		lo, hi := [2]f64{math.F64_MAX, math.F64_MAX}, [2]f64{-math.F64_MAX, -math.F64_MAX}
		for p in c {
			q := [2]f64{f64(p.x), f64(p.y)}
			lo, hi = {min(lo.x, q.x), min(lo.y, q.y)}, {max(hi.x, q.x), max(hi.y, q.y)}
		}
		append(&entries, Entry{contour = ci, lo = lo, hi = hi, interior = interior_point(c, lo, hi), abs_area = abs(area), winding = -1 if area < 0 else 1, container = -1})
	}
	// Largest first: whatever could contain an entry comes before it.
	slice.stable_sort_by(entries[:], proc(a, b: Entry) -> bool {return a.abs_area > b.abs_area})
	for i in 0 ..< len(entries) {
		e := &entries[i]
		container_winding := 0
		for j := i - 1; j >= 0; j -= 1 {
			c := &entries[j]
			if !(c.lo.x <= e.lo.x && c.lo.y <= e.lo.y && e.hi.x <= c.hi.x && e.hi.y <= c.hi.y) {
				continue
			}
			if !point_in_polygon(e.interior, contours[c.contour]) {
				continue
			}
			e.container = c.container if c.exclude else j
			container_winding = c.winding
			e.winding += container_winding
			break
		}
		if inside(e.winding, fill_rule) == inside(container_winding, fill_rule) {
			e.exclude = true
		}
	}
	shape_count := 0
	for &e in entries {
		if e.exclude {
			continue
		}
		e.hole = e.container >= 0 && !entries[e.container].hole
		if !e.hole {
			e.shape = shape_count
			shape_count += 1
		}
	}
	hole_counts := make([]int, shape_count)
	defer delete(hole_counts)
	for e in entries {
		if !e.exclude && e.hole {
			hole_counts[entries[e.container].shape] += 1
		}
	}
	shapes := make([]Shape, shape_count, allocator)
	for e in entries {
		if !e.exclude && !e.hole {
			shapes[e.shape] = {contour = contours[e.contour], holes = make([][]Vec2, hole_counts[e.shape], allocator)}
			hole_counts[e.shape] = 0
		}
	}
	for e in entries {
		if !e.exclude && e.hole {
			s := &shapes[entries[e.container].shape]
			s.holes[hole_counts[entries[e.container].shape]] = contours[e.contour]
			hole_counts[entries[e.container].shape] += 1
		}
	}
	return shapes
}

shapes_destroy :: proc(shapes: []Shape, allocator := context.allocator) {
	for s in shapes {
		delete(s.holes, allocator)
	}
	delete(shapes, allocator)
}

// Even-odd ray casting, for simple polygons.
@(private = "file")
point_in_polygon :: proc "contextless" (p: [2]f64, polygon: []Vec2) -> bool {
	inside := false
	n := len(polygon)
	for i, j := 0, n - 1; i < n; j, i = i, i + 1 {
		a := [2]f64{f64(polygon[i].x), f64(polygon[i].y)}
		b := [2]f64{f64(polygon[j].x), f64(polygon[j].y)}
		if (a.y > p.y) != (b.y > p.y) && p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x {
			inside = !inside
		}
	}
	return inside
}

// A point inside a simple polygon: its bounds' centre, or else the middle of
// the first two crossings of a horizontal line through it (paper.js's
// getInteriorPoint, as ShapePath.toShapes uses it).
@(private = "file")
interior_point :: proc(polygon: []Vec2, lo, hi: [2]f64) -> [2]f64 {
	p := (lo + hi) * 0.5
	if point_in_polygon(p, polygon) {
		return p
	}
	first, second := math.F64_MAX, math.F64_MAX
	count := 0
	n := len(polygon)
	for i in 0 ..< n {
		a := [2]f64{f64(polygon[i].x), f64(polygon[i].y)}
		b := [2]f64{f64(polygon[(i + 1) % n].x), f64(polygon[(i + 1) % n].y)}
		if (a.y > p.y) != (b.y > p.y) {
			x := a.x + (p.y - a.y) * (b.x - a.x) / (b.y - a.y)
			count += 1
			if x < first {
				first, second = x, first
			} else if x < second {
				second = x
			}
		}
	}
	if count > 1 {
		p.x = (first + second) / 2
	}
	return p
}

// ---- ShapeGeometry and ExtrudeGeometry ---------------------------------------------------

// ShapeGeometry(shapes): the shapes' flat triangulated faces at z = 0,
// facing +Z, with three.js's UVs (the points' x and y).
shape_geometry :: proc(g: ^Geometries, shapes: []Shape) -> Geometry_Id {
	clear(&g.scratch.vertices)
	clear(&g.scratch.indices)
	s: Shape_Points
	defer shape_points_destroy(&s)
	tris: [dynamic]u32
	defer delete(tris)
	for shape in shapes {
		if !shape_points(&s, shape, merge = false) {
			continue
		}
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
// bevel adds `bevel_thickness` beyond both ends and grows the outline by
// `bevel_size + bevel_offset`.
Extrude_Options :: struct {
	depth:           f32, // 1
	steps:           int, // 1: segments along the depth
	bevel_enabled:   bool, // true
	bevel_thickness: f32, // 0.2
	bevel_size:      f32, // 0.1 (three.js: bevelThickness - 0.1)
	bevel_offset:    f32, // 0
	bevel_segments:  int, // 3
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
extrude_geometry :: proc(g: ^Geometries, shapes: []Shape, options := EXTRUDE_DEFAULT) -> Extruded {
	o := options
	o.steps = max(o.steps, 1)
	if !o.bevel_enabled || o.bevel_segments <= 0 {
		o.bevel_segments, o.bevel_thickness, o.bevel_size, o.bevel_offset = 0, 0, 0, 0
	}
	x: Extruder
	defer extruder_destroy(&x)
	for shape in shapes {
		extrude_shape(&x, shape, o)
	}
	clear(&g.scratch.vertices)
	clear(&g.scratch.indices)
	emit := proc(x: ^Extruder, g: ^Geometries, corner: Vertex) {
		if i, ok := x.emitted[corner]; ok {
			append(&g.scratch.indices, i)
			return
		}
		i := u32(len(g.scratch.vertices))
		x.emitted[corner] = i
		append(&g.scratch.vertices, corner)
		append(&g.scratch.indices, i)
	}
	for t in x.caps {
		a, b, c := x.points[t[0]], x.points[t[1]], x.points[t[2]]
		n := face_normal(a, b, c)
		for p in ([3][3]f64{a, b, c}) {
			emit(&x, g, {position = vec3(p), normal = n, uv = {f32(p.x), f32(p.y)}})
		}
	}
	cap_indices := len(g.scratch.indices)
	side_normals(&x, o.crease_angle)
	for t, ti in x.sides {
		for k in 0 ..< 3 {
			p := x.points[t.corners[k]]
			uv := Vec2{f32(p.x if t.u_from_x else p.y), f32(1 - p.z)}
			emit(&x, g, {position = vec3(p), normal = x.normals[ti * 3 + k], uv = uv})
		}
	}
	whole := geometry_add(g, g.scratch.vertices[:], g.scratch.indices[:])
	return {
		geometry = whole,
		caps     = geometry_group(g, whole, 0, u32(cap_indices)),
		sides    = geometry_group(g, whole, u32(cap_indices), u32(len(g.scratch.indices) - cap_indices)),
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

// A shape's points, rewound (the outline clockwise, holes counter-clockwise)
// and, for extrusion, with repeated points merged: `points` is the outline
// then each hole, `rings` their lengths (each 3 or more).
@(private = "file")
Shape_Points :: struct {
	points: [dynamic][2]f64,
	rings:  [dynamic]int,
}

@(private = "file")
shape_points_destroy :: proc(s: ^Shape_Points) {
	delete(s.points)
	delete(s.rings)
}

// Fills `s` from `shape`; false when its outline has fewer than 3 points.
@(private = "file")
shape_points :: proc(s: ^Shape_Points, shape: Shape, merge: bool) -> bool {
	clear(&s.points)
	clear(&s.rings)
	add :: proc(s: ^Shape_Points, ring: []Vec2, clockwise, merge: bool) -> bool {
		start := len(s.points)
		for p in ring {
			append(&s.points, [2]f64{f64(p.x), f64(p.y)})
		}
		r := s.points[start:]
		if (ring_area(r) < 0) != clockwise {
			slice.reverse(r)
		}
		if merge {
			resize(&s.points, start + merge_overlapping_points(r))
		} else if len(r) > 2 && r[len(r) - 1] == r[0] {
			pop(&s.points) // ShapeUtils's removeDupEndPts
		}
		if len(s.points) - start < 3 {
			resize(&s.points, start)
			return false
		}
		append(&s.rings, len(s.points) - start)
		return true
	}
	if !add(s, shape.contour, true, merge) {
		return false
	}
	for h in shape.holes {
		add(s, h, false, merge)
	}
	return true
}

@(private = "file")
ring_area :: proc "contextless" (r: [][2]f64) -> f64 {
	a: f64
	for p, q := len(r) - 1, 0; q < len(r); p, q = q, q + 1 {
		a += r[p].x * r[q].y - r[q].x * r[p].y
	}
	return a * 0.5
}

// ShapeUtils.triangulateShape over `s`'s rings, at `points` (its points or
// the same rings moved).
@(private = "file")
triangulate_rings :: proc(s: ^Shape_Points, points: [][2]f64, triangles: ^[dynamic]u32) {
	starts := make([]int, len(s.rings) - 1)
	defer delete(starts)
	ends := make([]int, len(s.rings) - 1)
	defer delete(ends)
	k := s.rings[0]
	for l, h in s.rings[1:] {
		starts[h], ends[h] = k, k + l
		k += l
	}
	e := Earcut{triangles = triangles}
	defer delete(e.nodes)
	earcut(&e, points, s.rings[0], starts, ends)
}

// ExtrudeGeometry's mergeOverlappingPoints: drops points (wrapping around)
// within a scaled 1e-10 of the one before; returns how many are left at the
// front of `points`.
@(private = "file")
merge_overlapping_points :: proc(points: [][2]f64) -> int {
	THRESHOLD_SQ :: 1e-10 * 1e-10
	n := len(points)
	if n == 0 {
		return 0
	}
	prev := points[0]
	for i := 1; i <= n && n > 0; i += 1 {
		cur_i := i % n
		cur := points[cur_i]
		d := cur - prev
		scale := max(abs(cur.x), abs(cur.y), abs(prev.x), abs(prev.y))
		if d.x * d.x + d.y * d.y <= THRESHOLD_SQ * scale * scale {
			copy(points[cur_i:], points[cur_i + 1:n])
			n -= 1
			i -= 1
			continue
		}
		prev = cur
	}
	return n
}

@(private = "file")
Side_Triangle :: struct {
	corners:  [3]u32, // into `Extruder.points`
	u_from_x: bool, // WorldUVGenerator's wall u: the quad's x, else its y
}

// ExtrudeGeometry's state across its shapes: every shape's layered points
// (three.js's placeholder), the cap and side triangles into them, then the
// sides' corner normals and the emitted vertices.
@(private = "file")
Extruder :: struct {
	points:  [dynamic][3]f64,
	caps:    [dynamic][3]u32,
	sides:   [dynamic]Side_Triangle,
	normals: [dynamic]Vec3, // per side corner
	emitted: map[Vertex]u32,
	shape:   Shape_Points,
	moves:   [dynamic][2]f64,
	ring:    [dynamic][2]f64,
	faces:   [dynamic]u32,
}

@(private = "file")
extruder_destroy :: proc(x: ^Extruder) {
	delete(x.points)
	delete(x.caps)
	delete(x.sides)
	delete(x.normals)
	delete(x.emitted)
	shape_points_destroy(&x.shape)
	delete(x.moves)
	delete(x.ring)
	delete(x.faces)
}

// ExtrudeGeometry's addShape, without extrudePath.
@(private = "file")
extrude_shape :: proc(x: ^Extruder, shape: Shape, o: Extrude_Options) {
	s := &x.shape
	if !shape_points(s, shape, merge = true) {
		return
	}
	vertices := s.points[:]
	vlen := len(vertices)
	// Each point's bevel direction, within its own ring.
	resize(&x.moves, vlen)
	{
		k := 0
		for n in s.rings {
			ring := vertices[k:][:n]
			for i in 0 ..< n {
				x.moves[k + i] = bevel_vec(ring[i], ring[(i + n - 1) % n], ring[(i + 1) % n])
			}
			k += n
		}
	}
	moves := x.moves[:]
	segs := o.bevel_segments
	base := u32(len(x.points))
	add_layer :: proc(x: ^Extruder, vertices, moves: [][2]f64, bs, z: f64) {
		for v, i in vertices {
			p := v + moves[i] * bs
			append(&x.points, [3]f64{p.x, p.y, z})
		}
	}
	bevel_at :: proc(o: Extrude_Options, b: int) -> (z, bs: f64) {
		t := f64(b) / f64(o.bevel_segments)
		return f64(o.bevel_thickness) * math.cos(t * math.PI / 2), f64(o.bevel_size) * math.sin(t * math.PI / 2) + f64(o.bevel_offset)
	}
	// The caps' triangles: the outline as the outermost bevel ring has it.
	clear(&x.faces)
	if segs == 0 {
		triangulate_rings(s, vertices, &x.faces)
	} else {
		clear(&x.ring)
		for v, i in vertices {
			append(&x.ring, v + moves[i] * f64(o.bevel_offset))
		}
		triangulate_rings(s, x.ring[:], &x.faces)
	}
	// Layers, back to front: the back bevel's rings, the walls' steps + 1,
	// the front bevel's rings.
	for b in 0 ..< segs {
		z, bs := bevel_at(o, b)
		add_layer(x, vertices, moves, bs, -z)
	}
	wall := f64(o.bevel_size) + f64(o.bevel_offset)
	for st in 0 ..= o.steps {
		add_layer(x, vertices, moves, wall, f64(o.depth) / f64(o.steps) * f64(st))
	}
	for b := segs - 1; b >= 0; b -= 1 {
		z, bs := bevel_at(o, b)
		add_layer(x, vertices, moves, bs, f64(o.depth) + z)
	}
	// Back faces reversed, front faces as triangulated.
	top := base + u32(vlen * (o.steps + 2 * segs))
	for t := 0; t + 2 < len(x.faces); t += 3 {
		f := x.faces[t:][:3]
		append(&x.caps, [3]u32{base + f[2], base + f[1], base + f[0]})
	}
	for t := 0; t + 2 < len(x.faces); t += 3 {
		f := x.faces[t:][:3]
		append(&x.caps, [3]u32{top + f[0], top + f[1], top + f[2]})
	}
	// Side walls, ring by ring.
	layers := o.steps + 2 * segs
	offset := 0
	for n in s.rings {
		for i := n - 1; i >= 0; i -= 1 {
			j, k := i, (i + n - 1) % n
			for st in 0 ..< layers {
				l1, l2 := vlen * st, vlen * (st + 1)
				a := base + u32(offset + j + l1)
				b := base + u32(offset + k + l1)
				c := base + u32(offset + k + l2)
				d := base + u32(offset + j + l2)
				pa, pb := x.points[a], x.points[b]
				from_x := abs(pa.y - pb.y) < abs(pa.x - pb.x)
				append(&x.sides, Side_Triangle{{a, b, d}, from_x}, Side_Triangle{{b, c, d}, from_x})
			}
		}
		offset += n
	}
}

// ExtrudeGeometry's getBevelVec: where `p` moves for a unit bevel, the
// meeting of its two edges' lines shifted one unit to their left (outward
// on a clockwise outline), kept from spiking past sqrt(2).
@(private = "file")
bevel_vec :: proc "contextless" (p, prev, next: [2]f64) -> [2]f64 {
	EPSILON :: 2.220446049250313e-16
	vp := p - prev
	vn := next - p
	vp_lensq := vp.x * vp.x + vp.y * vp.y
	collinear := vp.x * vn.y - vp.y * vn.x
	trans: [2]f64
	shrink: f64
	if abs(collinear) > EPSILON {
		vp_len := math.sqrt(vp_lensq)
		vn_len := math.sqrt(vn.x * vn.x + vn.y * vn.y)
		prev_shift := [2]f64{prev.x - vp.y / vp_len, prev.y + vp.x / vp_len}
		next_shift := [2]f64{next.x - vn.y / vn_len, next.y + vn.x / vn_len}
		sf := ((next_shift.x - prev_shift.x) * vn.y - (next_shift.y - prev_shift.y) * vn.x) / collinear
		trans = prev_shift + vp * sf - p
		lensq := trans.x * trans.x + trans.y * trans.y
		if lensq <= 2 {
			return trans
		}
		shrink = math.sqrt(lensq / 2)
	} else {
		same := false
		if vp.x > EPSILON {
			same = vn.x > EPSILON
		} else if vp.x < -EPSILON {
			same = vn.x < -EPSILON
		} else {
			same = math.sign(vp.y) == math.sign(vn.y)
		}
		if same {
			trans = {-vp.y, vp.x}
			shrink = math.sqrt(vp_lensq)
		} else {
			trans = vp
			shrink = math.sqrt(vp_lensq / 2)
		}
	}
	return trans / shrink
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
