package o3d

import "core:math"
import "core:testing"

import curves "deps:ocurves"

// A circle of four cubic arcs (the usual 0.5523 handles), counter-clockwise
// unless `cw`.
@(private = "file")
circle_path :: proc(p: ^curves.Contours, r: f32, centre: Vec2 = {}, cw := false) {
	k := r * 0.5522847498
	s: f32 = -1 if cw else 1
	curves.contours_begin(p, centre + {r, 0})
	curves.contours_cubic(p, centre + {r, s * k}, centre + {k, s * r}, centre + {0, s * r})
	curves.contours_cubic(p, centre + {-k, s * r}, centre + {-r, s * k}, centre + {-r, 0})
	curves.contours_cubic(p, centre + {-r, -s * k}, centre + {-k, -s * r}, centre + {0, -s * r})
	curves.contours_cubic(p, centre + {k, -s * r}, centre + {r, -s * k}, centre + {r, 0})
	curves.contours_end(p, true, false)
}

// An exact shape is one quad over its bounds, facing +Z, its UVs the points
// and uv1 the shape coordinate; its curve set is kept and its quad's entry
// names it; a path with no curves makes nothing.
@(test)
test_exact_shape_geometry :: proc(t: ^testing.T) {
	g: Geometries
	defer geometries_destroy(&g)
	p: curves.Contours
	defer curves.contours_destroy(&p)
	circle_path(&p, 2, {1, 1})
	id := exact_shape_geometry(&g, &p)
	geo, ok := geometry_get(&g, id)
	testing.expect(t, ok)
	testing.expect_value(t, geo.mode, Draw_Mode.Curves)
	set, set_ok := curve_set_get(&g, 1)
	testing.expect(t, set_ok)
	testing.expect_value(t, set.extent, f32(4))
	words, rule := curve_entry_words(&g, curve_quad_entry(&g, geo, 0))
	testing.expect(t, len(words) > 0 && rule == .Non_Zero)
	vertices, indices, _ := geometry_mesh(&g, id)
	testing.expect_value(t, len(vertices), 4)
	testing.expect_value(t, len(indices), 6)
	testing.expect_value(t, vertices[0].position, Vec3{-1, -1, 0})
	testing.expect_value(t, vertices[2].position, Vec3{3, 3, 0})
	testing.expect_value(t, vertices[2].uv, Vec2{3, 3})
	testing.expect_value(t, vertices[2].uv1, Vec2{1, 1})
	testing.expect(t, curves_inside(words, {0.5, 0.5}, .Non_Zero))
	testing.expect(t, !curves_inside(words, {0.02, 0.02}, .Non_Zero))

	empty: curves.Contours
	defer curves.contours_destroy(&empty)
	curves.contours_begin(&empty, {1, 0})
	testing.expect_value(t, exact_shape_geometry(&g, &empty), Geometry_Id(0))
}

// The pixels' alpha summed (in pixels), how many are partly covered, and
// how many of those are not a whole number of 4x MSAA's samples.
@(private = "file")
alpha_area :: proc(px: []u8) -> (area: f64, partial, between_samples: int) {
	for i := 3; i < len(px); i += 4 {
		area += f64(px[i]) / 255
		if px[i] != 0 && px[i] != 255 {
			partial += 1
			if px[i] != 64 && px[i] != 128 && px[i] != 191 {
				between_samples += 1
			}
		}
	}
	return
}

// A circle drawn exact: transparent, its pixels' alpha is the area each
// covers (the total within 0.2% of pi r^2, the edge in many levels where
// 4x MSAA has 3); opaque, by sample (within 1%, only 4x MSAA's levels);
// the flattened mesh (12 segments a curve) misses more. Even-odd leaves a ring
// of two circles where nonzero fills both. Seen from a slant, it keeps its
// area too.
@(test)
test_exact_shape_coverage :: proc(t: ^testing.T) {
	g: Geometries
	defer geometries_destroy(&g)
	p: curves.Contours
	defer curves.contours_destroy(&p)
	circle_path(&p, 0.7)
	exact := exact_shape_geometry(&g, &p)
	shapes: Shapes
	defer shapes_destroy(&shapes)
	testing.expect_value(t, shapes_from_contours(&shapes, &p), curves.Bool_Error.None)
	mesh := shape_geometry(&g, &shapes)

	SIZE :: 64
	scene: Scene
	defer scene_destroy(&scene)
	scene.camera = {position = {0, 0, 5}, orthographic = true, height = 2, near = 0.1, far = 100}
	sw: Software
	defer software_destroy(&sw)
	draw :: proc(scene: ^Scene, sw: ^Software, g: ^Geometries, id: Geometry_Id, flags: Material_Flags) -> (area: f64, partial, between_samples: int) {
		clear(&scene.draws)
		m := MATERIAL_DEFAULT
		m.flags = flags
		append(&scene.draws, Draw{geometry = id, model = 1, material = m})
		return alpha_area(software_render(sw, scene, g, nil, SIZE, SIZE))
	}
	r_px := 0.7 * f64(SIZE) / 2
	want := math.PI * r_px * r_px
	area, partial, between := draw(&scene, &sw, &g, exact, {.Basic, .Transparent})
	testing.expectf(t, abs(area - want) / want < 0.002, "area %v, want %v", area, want)
	testing.expectf(t, partial > 100 && between > partial / 2, "partial pixels %v, %v between samples", partial, between)
	area, partial, between = draw(&scene, &sw, &g, exact, {.Basic})
	testing.expectf(t, abs(area - want) / want < 0.01, "sampled area %v, want %v", area, want)
	testing.expectf(t, partial > 50 && between == 0, "sampled: partial pixels %v, %v between samples", partial, between)
	flat_area, _, _ := draw(&scene, &sw, &g, mesh, {.Basic})
	testing.expectf(t, abs(flat_area - want) > abs(area - want), "flattened %v, exact %v, want %v", flat_area, area, want)

	ring: curves.Contours
	defer curves.contours_destroy(&ring)
	circle_path(&ring, 0.7)
	circle_path(&ring, 0.4)
	nonzero := exact_shape_geometry(&g, &ring)
	even_odd := exact_shape_geometry(&g, &ring, .Even_Odd)
	area, _, _ = draw(&scene, &sw, &g, nonzero, {.Basic, .Transparent})
	testing.expectf(t, abs(area - want) / want < 0.002, "nonzero %v, want %v", area, want)
	inner := 0.4 * f64(SIZE) / 2
	area, _, _ = draw(&scene, &sw, &g, even_odd, {.Basic, .Transparent})
	testing.expectf(t, abs(area - (want - math.PI * inner * inner)) / want < 0.003, "even-odd %v", area)

	// Turned 60 degrees about Y: the area halves.
	clear(&scene.draws)
	m := MATERIAL_DEFAULT
	m.flags = {.Basic, .Transparent}
	m.side = .Double
	c, s := math.cos(f32(math.PI / 3)), math.sin(f32(math.PI / 3))
	turn := Mat4{c, 0, s, 0, 0, 1, 0, 0, -s, 0, c, 0, 0, 0, 0, 1}
	append(&scene.draws, Draw{geometry = exact, model = turn, material = m})
	area, _, _ = alpha_area(software_render(&sw, &scene, &g, nil, SIZE, SIZE))
	testing.expectf(t, abs(area - want / 2) / want < 0.004, "slanted %v, want %v", area, want / 2)
}

// A glyph run: one quad a glyph with a set (sets shared, a glyph without
// one left out), each at its position and scaled, its entry naming its set
// and fill rule; a mirrored run is wound back to face +Z; words from
// curves' builder make the same set as the path; bad words or no glyphs make
// nothing.
@(test)
test_glyph_run_geometry :: proc(t: ^testing.T) {
	g: Geometries
	defer geometries_destroy(&g)
	p: curves.Contours
	defer curves.contours_destroy(&p)
	circle_path(&p, 1, {1, 1})
	disc := curve_set_add(&g, &p)
	ring_path: curves.Contours
	defer curves.contours_destroy(&ring_path)
	circle_path(&ring_path, 1, {1, 1})
	circle_path(&ring_path, 0.5, {1, 1})
	ring := curve_set_add(&g, &ring_path, .Even_Odd)
	glyphs := []Glyph{{disc, {0, 0}}, {0, {9, 9}}, {ring, {3, 0}}, {disc, {6, 0}}}
	id := glyph_run_geometry(&g, glyphs, 2)
	geo, _ := geometry_get(&g, id)
	vertices, indices, _ := geometry_mesh(&g, id)
	testing.expect_value(t, len(vertices), 12)
	testing.expect_value(t, len(indices), 18)
	testing.expect_value(t, vertices[4].position, Vec3{3, 0, 0})
	testing.expect_value(t, vertices[6].position, Vec3{7, 4, 0})
	testing.expect_value(t, vertices[10].uv, Vec2{10, 4})
	testing.expect_value(t, vertices[10].uv1, Vec2{1, 1})
	disc_set, _ := curve_set_get(&g, disc)
	ring_set, _ := curve_set_get(&g, ring)
	testing.expect_value(t, curve_quad_entry(&g, geo, 0), disc_set.first_word)
	testing.expect_value(t, curve_quad_entry(&g, geo, 1), ring_set.first_word | CURVE_EVEN_ODD)
	testing.expect_value(t, curve_quad_entry(&g, geo, 2), disc_set.first_word)
	words, rule := curve_entry_words(&g, curve_quad_entry(&g, geo, 1))
	testing.expect(t, rule == .Even_Odd && !curves_inside(words, {0.5, 0.5}, .Even_Odd) && curves_inside(words, {0.5, 0.15}, .Even_Odd))

	// Y flipped (a font's y-down units): counter-clockwise still.
	flipped := glyph_run_geometry(&g, glyphs[:1], {1, -1})
	fv, fi, _ := geometry_mesh(&g, flipped)
	a, b, c := fv[fi[0]].position.xy, fv[fi[1]].position.xy, fv[fi[2]].position.xy
	testing.expect(t, (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y) > 0)
	testing.expect_value(t, fv[2].position, Vec3{2, -2, 0})

	b_: curves.Builder
	defer curves.builder_destroy(&b_)
	curves.move_to(&b_, {2, 1})
	curves.quad_to(&b_, {2, 2}, {1, 2})
	curves.line_to(&b_, {1, 1})
	curves.close(&b_)
	w, origin, extent, ok := curves.build(&b_)
	defer delete(w)
	testing.expect(t, ok)
	from_words := curve_set_add_words(&g, w, origin, extent)
	ws, _ := curve_set_get(&g, from_words)
	testing.expect_value(t, ws.origin, origin)
	testing.expect_value(t, curve_set_add_words(&g, w[:4], origin, extent), Curve_Set_Id(0))
	testing.expect_value(t, glyph_run_geometry(&g, {{0, {}}}), Geometry_Id(0))
	testing.expect_value(t, glyph_run_geometry(&g, glyphs, {1, 0}), Geometry_Id(0))
}

// A run's glyphs each cover their own area (a disc twice, an even-odd ring
// between), drawn as one transparent draw; the same run mirrored (y down)
// covers the same.
@(test)
test_glyph_run_coverage :: proc(t: ^testing.T) {
	g: Geometries
	defer geometries_destroy(&g)
	p: curves.Contours
	defer curves.contours_destroy(&p)
	circle_path(&p, 0.2)
	disc := curve_set_add(&g, &p)
	ring_path: curves.Contours
	defer curves.contours_destroy(&ring_path)
	circle_path(&ring_path, 0.2)
	circle_path(&ring_path, 0.1)
	ring := curve_set_add(&g, &ring_path, .Even_Odd)
	glyphs := []Glyph{{disc, {-0.6, 0}}, {ring, {0, 0}}, {disc, {0.6, 0}}}
	run := glyph_run_geometry(&g, glyphs)
	mirrored := glyph_run_geometry(&g, glyphs, {1, -1})

	SIZE :: 128
	scene: Scene
	defer scene_destroy(&scene)
	scene.camera = {position = {0, 0, 5}, orthographic = true, height = 2, near = 0.1, far = 100}
	sw: Software
	defer software_destroy(&sw)
	px_per := f64(SIZE) / 2
	r, r_in := 0.2 * px_per, 0.1 * px_per
	want := 3 * math.PI * r * r - math.PI * r_in * r_in
	for id in ([]Geometry_Id{run, mirrored}) {
		clear(&scene.draws)
		m := MATERIAL_DEFAULT
		m.flags = {.Basic, .Transparent}
		append(&scene.draws, Draw{geometry = id, model = 1, material = m})
		area, _, _ := alpha_area(software_render(&sw, &scene, &g, nil, SIZE, SIZE))
		testing.expectf(t, abs(area - want) / want < 0.003, "run %v: area %v, want %v", id, area, want)
	}
}
