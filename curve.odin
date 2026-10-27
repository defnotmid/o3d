package o3d

import "core:math/linalg"

import curves "deps:ocurves"

// Paths with their curves are ocurves' `Contours` (lines and cubics in one
// table: three.js's Path and ShapePath, moveTo as `curves.contours_begin`,
// lineTo `curves.contours_line`, bezierCurveTo `curves.contours_cubic`,
// quadraticCurveTo `curves.contours_quad`), drawn two ways:
//
// - flattened to points (`shapes_from_contours`, three.js's toShapes and
//   getPoints with `curve_segments` divisions per curve), for
//   `shape_geometry` and `extrude_geometry`;
// - exact (`exact_shape_geometry`): one quad over the path's bounds whose
//   pixels take their coverage from the curves themselves (ocurves'
//   curves package: the exact area of the pixel inside the shape, from the curve
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

// A shape's curves kept in `Geometries` (made once, never freed) for any
// number of exact geometries to draw: a path's (`curve_set_add`) or a
// block ocurves built (`curve_set_add_words`: an ovg font's
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
	rule:       curves.Fill_Rule,
}

// A `.Curves` quad's entry in its geometry's table: its set's first word,
// and this bit for the even-odd rule (shaders/o3d.slang reads it too).
CURVE_EVEN_ODD :: u32(1) << 31

// Contours' curves as a curve set, in their units: cubics become
// quadratics within `tolerance` (their units; 0: 1/4096 of each cubic's
// size). 0 when they fill nothing (no curves, or no extent).
curve_set_add :: proc(g: ^Geometries, c: ^curves.Contours, fill_rule := curves.Fill_Rule.Non_Zero, tolerance: f32 = 0) -> Curve_Set_Id {
	b := curves.Builder{tol = tolerance}
	defer curves.builder_destroy(&b)
	curves.builder_add_contours(&b, c, 0, len(c.contours))
	words, origin, extent, ok := curves.build(&b)
	if !ok {
		return 0
	}
	defer delete(words)
	return curve_set_add_words(g, words, origin, extent, fill_rule)
}

// A block of curves' words (`curves.build`'s, or an ovg `Shape`'s
// `data` with its `origin` and `extent`: a font's glyph from
// `font_glyph_shape`, in font units with y down) as a curve set, copied.
// 0 when the block is too short for its header or fills nothing.
curve_set_add_words :: proc(g: ^Geometries, words: []u32, origin: Vec2, extent: f32, fill_rule := curves.Fill_Rule.Non_Zero) -> Curve_Set_Id {
	if len(words) < curves.HEADER || int(words[6]) > len(words) || !(extent > 0) {
		return 0
	}
	x0, y0, x1, y1 := curves.bounds(words)
	if !(x1 > x0) || !(y1 > y0) {
		return 0
	}
	generation_take(&g.generation)
	append(&g.curve_sets, Curve_Set{first_word = u32(len(g.curve_words)), origin = origin, extent = extent, lo = {x0, y0}, hi = {x1, y1}, rule = fill_rule})
	append(&g.curve_words, ..words)
	return Curve_Set_Id(len(g.curve_sets))
}

curve_set_get :: proc "contextless" (g: ^Geometries, id: Curve_Set_Id) -> (Curve_Set, bool) {
	if id == 0 || int(id) > len(g.curve_sets) {
		return {}, false
	}
	return g.curve_sets[id - 1], true
}

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
		append(&entries, set.first_word | (CURVE_EVEN_ODD if set.rule == .Even_Odd else 0))
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

// ShapeGeometry with exact edges: one quad over the contours' bounds at
// z = 0, facing +Z, drawn with their curves' coverage per pixel
// (`Draw_Mode.Curves`; see the top of this file): a run of one glyph, their
// curve set (`curve_set_add`) at the origin. UVs are the points' x and y,
// as ShapeGeometry's; `uv1` is the shape coordinate. 0 when they fill
// nothing.
exact_shape_geometry :: proc(g: ^Geometries, c: ^curves.Contours, fill_rule := curves.Fill_Rule.Non_Zero, tolerance: f32 = 0) -> Geometry_Id {
	set := curve_set_add(g, c, fill_rule, tolerance)
	if set == 0 {
		return 0
	}
	glyph := [1]Glyph{{set = set}}
	return glyph_run_geometry(g, glyph[:])
}

// The words of a `.Curves` quad's set from its table entry (offsets are
// relative to its first word), and its fill rule.
@(private)
curve_entry_words :: #force_inline proc "contextless" (g: ^Geometries, entry: u32) -> (words: []u32, rule: curves.Fill_Rule) {
	first := int(entry &~ CURVE_EVEN_ODD)
	if first >= len(g.curve_words) {
		return
	}
	return g.curve_words[first:], .Even_Odd if entry & CURVE_EVEN_ODD != 0 else .Non_Zero
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
curves_inside :: #force_inline proc "contextless" (words: []u32, rc: Vec2, rule: curves.Fill_Rule) -> bool {
	return curves.fill_inside(curves.winding(words, rc), rule)
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
