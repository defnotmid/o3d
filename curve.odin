package o3d

import "core:math/linalg"

import "deps:ovg/curves"

// Paths with their curves (three.js's Path and ShapePath: moveTo, lineTo,
// quadraticCurveTo, bezierCurveTo), drawn two ways:
//
// - flattened to points (`path_contours`, three.js's getPoints with
//   `curve_segments` divisions per curve), for `shapes_from_contours`,
//   `shape_geometry` and `extrude_geometry`;
// - exact (`exact_shape_geometry`): one quad over the path's bounds whose
//   pixels take their coverage from the curves themselves (ovg's curves
//   package: the exact area of the pixel inside the shape, from the curve
//   pieces in the grid cells it covers and their backdrop windings). No
//   facets at any distance, and edges anti-aliased by the area each pixel
//   covers rather than by 4x MSAA's samples.
//
// Text is the same with a quad per glyph (`glyph_run_geometry`): each
// glyph a curve set (`curve_set_add`, or `curve_set_add_words` for an ovg
// font's glyph shapes), made once and placed by the caller's layout, the
// run one geometry and one draw.
//
// An exact shape or run is a geometry like any other (`Geometries` keeps
// its curve sets, made once and never freed) with `Draw_Mode.Curves`; any
// standard material draws it (MeshStandard, Basic, Toon, textured: its UVs
// are the points' x and y, as ShapeGeometry's). How the edge is resolved
// follows the material:
//
// - `.Transparent`: each pixel's covered area is its alpha, blended over
//   what is behind (drawn with the other blended draws, farthest first,
//   depth kept): the exact edge;
// - otherwise opaque: the curves decide which of the pixel's four samples
//   are inside, so the edges are 4x MSAA's but the curves exact, written
//   with depth and drawn in any order.
//
// Shadows take the shape's outline (texels outside it are discarded). Each
// frame each quad grows by half a pixel around its edges (in
// `expand_draws`, for the view it is drawn in), so the edge pixels outside
// the bounds are drawn. A ShaderMaterial on an exact shape draws the bare
// quad: `uv1` holds the shape coordinate (`curves`' units) for its own use.

Path_Segment_Kind :: enum u8 {
	Move,
	Line,
	Quadratic,
	Cubic,
}

// A move, line, or curve to `to`, with its control points (`c1` for a
// quadratic; `c1`, `c2` for a cubic).
Path_Segment :: struct {
	kind:       Path_Segment_Kind,
	c1, c2, to: Vec2,
}

// Contours of lines and Bézier curves in the XY plane (Y up), each begun by
// a move and closed when filled (three.js's ShapePath of subpaths); either
// winding fills, holes are contours inside others by the fill rule.
Path :: struct {
	segments: [dynamic]Path_Segment,
}

path_destroy :: proc(p: ^Path) {
	delete(p.segments)
	p^ = {}
}

path_clear :: proc(p: ^Path) {
	clear(&p.segments)
}

// Starts a contour at `to` (moveTo).
path_move_to :: proc(p: ^Path, to: Vec2) {
	append(&p.segments, Path_Segment{kind = .Move, to = to})
}

// lineTo; a path not yet begun begins at `to`.
path_line_to :: proc(p: ^Path, to: Vec2) {
	if len(p.segments) == 0 {
		path_move_to(p, to)
		return
	}
	append(&p.segments, Path_Segment{kind = .Line, to = to})
}

// quadraticCurveTo: control point `c`, end `to`.
path_quadratic_curve_to :: proc(p: ^Path, c, to: Vec2) {
	if len(p.segments) == 0 {
		path_move_to(p, c)
	}
	append(&p.segments, Path_Segment{kind = .Quadratic, c1 = c, to = to})
}

// bezierCurveTo: control points `c1`, `c2`, end `to`.
path_bezier_curve_to :: proc(p: ^Path, c1, c2, to: Vec2) {
	if len(p.segments) == 0 {
		path_move_to(p, c1)
	}
	append(&p.segments, Path_Segment{kind = .Cubic, c1 = c1, c2 = c2, to = to})
}

// closePath: a line back to the contour's start, unless it is there.
path_close :: proc(p: ^Path) {
	#reverse for s in p.segments {
		if s.kind == .Move {
			if last := p.segments[len(p.segments) - 1].to; last != s.to {
				append(&p.segments, Path_Segment{kind = .Line, to = s.to})
			}
			return
		}
	}
}

// The contours flattened to points as three.js's getPoints(divisions) does:
// a line is its end, a curve `curve_segments` points along it (at even
// steps of its parameter, computed in f64 as three.js does), a point equal
// to the one before dropped. Contours with fewer than 3 points are left
// out. Free with `contours_destroy`.
path_contours :: proc(p: ^Path, curve_segments := 12, allocator := context.allocator) -> [][]Vec2 {
	n := max(curve_segments, 1)
	out := make([dynamic][]Vec2, allocator)
	points := make([dynamic]Vec2, context.temp_allocator)
	push :: proc(points: ^[dynamic]Vec2, q: [2]f64) {
		v := Vec2{f32(q.x), f32(q.y)}
		if len(points) == 0 || points[len(points) - 1] != v {
			append(points, v)
		}
	}
	flush :: proc(out: ^[dynamic][]Vec2, points: ^[dynamic]Vec2, allocator := context.allocator) {
		if len(points) >= 3 {
			c := make([]Vec2, len(points), allocator)
			copy(c, points[:])
			append(out, c)
		}
		clear(points)
	}
	at: [2]f64
	for s in p.segments {
		p0 := at
		c1 := [2]f64{f64(s.c1.x), f64(s.c1.y)}
		c2 := [2]f64{f64(s.c2.x), f64(s.c2.y)}
		to := [2]f64{f64(s.to.x), f64(s.to.y)}
		switch s.kind {
		case .Move:
			flush(&out, &points, allocator)
			push(&points, to)
		case .Line:
			push(&points, to)
		case .Quadratic:
			for i in 1 ..= n {
				t := f64(i) / f64(n)
				k := 1 - t
				push(&points, k * k * p0 + 2 * k * t * c1 + t * t * to)
			}
		case .Cubic:
			for i in 1 ..= n {
				t := f64(i) / f64(n)
				k := 1 - t
				push(&points, k * k * k * p0 + 3 * k * k * t * c1 + 3 * k * t * t * c2 + t * t * t * to)
			}
		}
		at = to
	}
	flush(&out, &points, allocator)
	return out[:]
}

contours_destroy :: proc(contours: [][]Vec2, allocator := context.allocator) {
	for c in contours {
		delete(c, allocator)
	}
	delete(contours, allocator)
}

// A shape's curves kept in `Geometries` (made once, never freed) for any
// number of exact geometries to draw: a path's (`curve_set_add`) or a
// block ovg's curves package built (`curve_set_add_words`: an ovg font's
// glyph shapes). 0 is none.
Curve_Set_Id :: distinct u32

// A curve set: its block's first word in `Geometries.curve_words`, how its
// shape units map to the units it was made in (`p = origin + q * extent`),
// its bounds `lo`..`hi` in shape units, and its fill rule.
Curve_Set :: struct {
	first_word: u32,
	origin:     Vec2,
	extent:     f32,
	lo, hi:     Vec2,
	even_odd:   bool,
}

// A `.Curves` quad's entry in its geometry's table: its set's first word,
// and this bit for the even-odd rule.
CURVE_EVEN_ODD :: u32(1) << 31

// A path's curves as a curve set, in the path's units: cubics become
// quadratics within `tolerance` (path units; 0: 1/4096 of each cubic's
// size). 0 when the path fills nothing (no curves, or no extent).
curve_set_add :: proc(g: ^Geometries, p: ^Path, fill_rule := Fill_Rule.Nonzero, tolerance: f32 = 0) -> Curve_Set_Id {
	b := curves.Builder{tol = tolerance}
	defer curves.builder_destroy(&b)
	for s in p.segments {
		switch s.kind {
		case .Move:
			curves.move_to(&b, s.to)
		case .Line:
			if !b.open {
				curves.move_to(&b, b.last)
			}
			curves.line_to(&b, s.to)
		case .Quadratic:
			if !b.open {
				curves.move_to(&b, b.last)
			}
			curves.quad_to(&b, s.c1, s.to)
		case .Cubic:
			if !b.open {
				curves.move_to(&b, b.last)
			}
			curves.cubic_to(&b, s.c1, s.c2, s.to)
		}
	}
	words, origin, extent, ok := curves.build(&b)
	if !ok {
		return 0
	}
	defer delete(words)
	return curve_set_add_words(g, words, origin, extent, fill_rule)
}

// A block of ovg curves' words (`curves.build`'s, or an ovg `Shape`'s
// `data` with its `origin` and `extent`: a font's glyph from
// `font_glyph_shape`, in font units with y down) as a curve set, copied.
// 0 when the block is too short for its header or fills nothing.
curve_set_add_words :: proc(g: ^Geometries, words: []u32, origin: Vec2, extent: f32, fill_rule := Fill_Rule.Nonzero) -> Curve_Set_Id {
	if len(words) < CURVE_HEADER_WORDS || int(words[6]) > len(words) || !(extent > 0) {
		return 0
	}
	x0, y0, x1, y1 := curves.bounds(words)
	if !(x1 > x0) || !(y1 > y0) {
		return 0
	}
	generation_take(&g.generation)
	append(&g.curve_sets, Curve_Set{first_word = u32(len(g.curve_words)), origin = origin, extent = extent, lo = {x0, y0}, hi = {x1, y1}, even_odd = fill_rule == .Even_Odd})
	append(&g.curve_words, ..words)
	return Curve_Set_Id(len(g.curve_sets))
}

curve_set_get :: proc "contextless" (g: ^Geometries, id: Curve_Set_Id) -> (Curve_Set, bool) {
	if id == 0 || int(id) > len(g.curve_sets) {
		return {}, false
	}
	return g.curve_sets[id - 1], true
}

// The words of ovg curves' header (bounds and offsets), before its cells.
@(private)
CURVE_HEADER_WORDS :: 12

// A glyph of a run: curve set `set` with its units' origin at `position`
// (run units).
Glyph :: struct {
	set:      Curve_Set_Id,
	position: Vec2,
}

// Text with exact edges: one quad a glyph over its set's bounds at z = 0,
// facing +Z, each drawn with its set's coverage per pixel
// (`Draw_Mode.Curves`; see the top of this file), all one geometry and one
// draw. Layout is the caller's: each glyph's set is placed at its
// `position`, its units scaled by `scale` (run units per set unit; a
// negative y turns a font's y-down units Y up). Glyphs may share sets and
// overlap. UVs are the run's x and y (as ShapeGeometry's points); `uv1` is
// the glyph's shape coordinate. Glyphs without a set are left out; 0 when
// none is left.
glyph_run_geometry :: proc(g: ^Geometries, glyphs: []Glyph, scale: Vec2 = 1) -> Geometry_Id {
	if scale.x == 0 || scale.y == 0 {
		return 0
	}
	vertices := make([dynamic]Vertex, 0, 4 * len(glyphs), context.temp_allocator)
	indices := make([dynamic]u32, 0, 6 * len(glyphs), context.temp_allocator)
	entries := make([dynamic]u32, 0, len(glyphs), context.temp_allocator)
	// Mirrored (one axis flipped), the corners turn clockwise: wound back
	// to face +Z.
	order := [6]u32{0, 2, 1, 0, 3, 2} if scale.x * scale.y < 0 else [6]u32{0, 1, 2, 0, 2, 3}
	for glyph in glyphs {
		set := curve_set_get(g, glyph.set) or_continue
		qs := [4]Vec2{set.lo, {set.hi.x, set.lo.y}, set.hi, {set.lo.x, set.hi.y}}
		base := u32(len(vertices))
		for q in qs {
			p := glyph.position + (set.origin + q * set.extent) * scale
			append(&vertices, Vertex{position = {p.x, p.y, 0}, normal = {0, 0, 1}, uv = p, uv1 = q})
		}
		for k in order {
			append(&indices, base + k)
		}
		append(&entries, set.first_word | (CURVE_EVEN_ODD if set.even_odd else 0))
	}
	if len(entries) == 0 {
		return 0
	}
	table := u32(len(g.curve_words))
	append(&g.curve_words, ..entries[:])
	id := geometry_add(g, vertices[:], indices[:], mode = .Curves)
	g.list[id - 1].curve_table = table
	return id
}

// ShapeGeometry with exact edges: one quad over the path's bounds at z = 0,
// facing +Z, drawn with its curves' coverage per pixel (`Draw_Mode.Curves`;
// see the top of this file): a run of one glyph, the path's curve set
// (`curve_set_add`) at the origin. UVs are the points' x and y, as
// ShapeGeometry's; `uv1` is the shape coordinate. 0 when the path fills
// nothing.
exact_shape_geometry :: proc(g: ^Geometries, p: ^Path, fill_rule := Fill_Rule.Nonzero, tolerance: f32 = 0) -> Geometry_Id {
	set := curve_set_add(g, p, fill_rule, tolerance)
	if set == 0 {
		return 0
	}
	glyph := [1]Glyph{{set = set}}
	return glyph_run_geometry(g, glyph[:])
}

// The words of a `.Curves` quad's set from its table entry (offsets are
// relative to its first word), and its fill rule.
@(private)
curve_entry_words :: #force_inline proc "contextless" (g: ^Geometries, entry: u32) -> (words: []u32, even_odd: bool) {
	first := int(entry &~ CURVE_EVEN_ODD)
	if first >= len(g.curve_words) {
		return
	}
	return g.curve_words[first:], entry & CURVE_EVEN_ODD != 0
}

// The table entry of a `.Curves` geometry's quad (its vertices
// `4 * quad ..< 4 * quad + 4`); 0 when out of range.
@(private)
curve_quad_entry :: #force_inline proc "contextless" (g: ^Geometries, geo: Geometry, quad: u32) -> u32 {
	i := int(geo.curve_table) + int(quad)
	if geo.mode != .Curves || i >= len(g.curve_words) {
		return 0
	}
	return g.curve_words[i]
}

// Whether the shape covers the point `rc` (shape units) by its fill rule.
@(private)
curves_inside :: #force_inline proc "contextless" (words: []u32, rc: Vec2, even_odd: bool) -> bool {
	w := curves.winding(words, rc)
	return w & 1 != 0 if even_odd else w != 0
}

// The quads of a `.Curves` geometry (each four vertices in its object's XY
// plane, `uv` its x and y) grown for the view so each edge moves out by
// half the footprint of the pixel there, which is as far outside the
// bounds as a pixel's centre can be and still be covered: `x` by half the
// object's x extent of the pixel (the box around its parallelogram, as
// `curves.footprint_of` takes it), `y` likewise; UVs and the shape
// coordinate move with it (each quad's own shape units per object unit,
// signed: a mirrored glyph's run the other way). A corner behind the eye
// stays (the near plane clips it); along a grazing view an edge moves at
// most the quad's size.
@(private)
dilate_curve_quads :: proc "contextless" (src, dst: []Vertex, mvp: Mat4, width, height: int) {
	half := Vec2{f32(width), f32(height)} * 0.5
	col0 := Vec4{mvp[0, 0], mvp[1, 0], mvp[2, 0], mvp[3, 0]}
	col1 := Vec4{mvp[0, 1], mvp[1, 1], mvp[2, 1], mvp[3, 1]}
	for q := 0; q + 4 <= len(src); q += 4 {
		quad := src[q:][:4]
		lo, hi := quad[0].position.xy, quad[0].position.xy
		for v in quad {
			lo = linalg.min(lo, v.position.xy)
			hi = linalg.max(hi, v.position.xy)
		}
		centre := (lo + hi) * 0.5
		size := hi - lo
		dp := quad[2].position.xy - quad[0].position.xy
		du := quad[2].uv1 - quad[0].uv1
		shape_scale := Vec2{du.x / dp.x if dp.x != 0 else 0, du.y / dp.y if dp.y != 0 else 0}
		for v, k in quad {
			out := v
			c := mvp * Vec4{v.position.x, v.position.y, v.position.z, 1}
			if c.w > 1e-6 {
				// Pixels per object unit along x and y at the corner.
				jx := (col0.xy * c.w - c.xy * col0.w) / (c.w * c.w) * half
				jy := (col1.xy * c.w - c.xy * col1.w) / (c.w * c.w) * half
				det := abs(jx.x * jy.y - jx.y * jy.x)
				grow := size
				if det > 1e-20 {
					grow.x = min(0.5 * (abs(jy.x) + abs(jy.y)) / det, size.x)
					grow.y = min(0.5 * (abs(jx.x) + abs(jx.y)) / det, size.y)
				}
				d := Vec2{grow.x if v.position.x > centre.x else -grow.x, grow.y if v.position.y > centre.y else -grow.y}
				out.position.xy += d
				out.uv += d
				out.uv1 += d * shape_scale
			}
			dst[q + k] = out
		}
	}
}
