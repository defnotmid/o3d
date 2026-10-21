package o3d

import "core:math/linalg"
import "core:testing"

// testdata/usd_parts.usdz: tools/usd_fixture/make.py says what it holds.
@(private = "file")
USD_PARTS :: #load("testdata/usd_parts.usdz")

// The layer's tables and values, compressed arrays among them.
@(test)
test_usd_read_layer :: proc(t: ^testing.T) {
	entries, zerr := zip_read_directory(USD_PARTS)
	testing.expect_value(t, zerr, Zip_Error.None)
	bytes, ok := zip_extract(USD_PARTS, entries, "parts.usdc")
	testing.expect(t, ok, "the layer is in the USDZ")
	l, err := usd_read_layer(bytes)
	defer usd_layer_destroy(&l)
	testing.expect_value(t, err, Usd_Error.None)

	token :: proc(l: ^Usd_Layer, s: string) -> i32 {
		for t, i in l.tokens {
			if t == s {
				return i32(i)
			}
		}
		return -1
	}
	path :: proc(l: ^Usd_Layer, names: []string, property := "") -> i32 {
		p := l.root
		for n in names {
			p = usd_child(l, p, token(l, n), false)
		}
		return usd_child(l, p, token(l, property), true) if property != "" else p
	}
	grid := path(&l, {"World", "Grid"})
	testing.expect(t, grid >= 0 && l.specs[l.paths[grid].spec].type == .Prim, "/World/Grid is a prim")

	// 100 face vertex indices (compressed ints) and 32 floats.
	ints: [dynamic]i64
	defer delete(ints)
	rep, has := usd_field(&l, l.paths[path(&l, {"World", "Grid"}, "faceVertexIndices")].spec, .Default)
	testing.expect(t, has && usd_rep_is_array(rep), "an array default")
	testing.expect(t, usd_ints(&l, rep, &ints), "ints read")
	testing.expect_value(t, len(ints), 100)
	sum: i64
	for v in ints {
		sum += v
	}
	testing.expect_value(t, sum, 1750) // each quad's corners sum to 4(6y + x) + 14
	floats: [dynamic]f32
	defer delete(floats)
	rep, _ = usd_field(&l, l.paths[path(&l, {"World", "Grid"}, "weights")].spec, .Default)
	comps, fok := usd_floats(&l, rep, &floats)
	testing.expect(t, fok && comps == 1 && len(floats) == 32 && floats[31] == 15.5, "32 floats, i / 2")

	// A token and the stage's metadata.
	rep, _ = usd_field(&l, l.paths[l.root].spec, .Up_Axis)
	axis, _ := usd_string(&l, rep)
	testing.expect_value(t, axis, "Z")
	rep, _ = usd_field(&l, l.paths[l.root].spec, .Meters_Per_Unit)
	meters, _ := usd_number(&l, rep)
	testing.expect_value(t, meters, 0.01)
}

// Composition, transforms, subsets, bindings, materials and textures.
@(test)
test_usd_parse :: proc(t: ^testing.T) {
	geos: Geometries
	defer geometries_destroy(&geos)
	textures: Textures
	defer textures_destroy(&textures)
	m, err := usd_parse(USD_PARTS, &geos, &textures)
	defer model_destroy(&m)
	testing.expect_value(t, err, Usd_Error.None)
	testing.expect_value(t, len(m.warnings), 0)

	a, b := model_node(&m, "A"), model_node(&m, "B")
	testing.expect(t, a != 0 && b != 0, "instances by name")
	red, net := model_material(&m, "Red"), model_material(&m, "Net")
	testing.expect(t, red != 0 && net != 0, "materials by name")

	scene: Scene
	defer scene_destroy(&scene)
	gltf_draw(&scene, &m)
	// Each instance's Part: the subset's quad bound to Red, the triangle to
	// the instance's own (B's Net, A none). B's Part shares A's geometry.
	part_of :: proc(m: ^Model, parent: Model_Node_Id) -> int {
		for i in 0 ..< len(m.nodes) {
			if m.nodes.parent[i] == i32(parent) - 1 && m.nodes.name[i] == "Part" {
				return i
			}
		}
		return -1
	}
	pa, pb := part_of(&m, a), part_of(&m, b)
	testing.expect(t, pa >= 0 && pb >= 0, "each instance has its prototype's Part")
	ma, mb := m.meshes[m.nodes.mesh[pa]], m.meshes[m.nodes.mesh[pb]]
	testing.expect(t, ma.primitive_count == 2 && mb.primitive_count == 2, "a primitive per part")
	prim_a := m.primitives[ma.first_primitive:][:2]
	prim_b := m.primitives[mb.first_primitive:][:2]
	testing.expect(t, prim_a[0].geometry == prim_b[0].geometry && prim_a[1].geometry == prim_b[1].geometry, "instances share geometry")
	testing.expect(t, prim_a[0].material_id == red && prim_b[0].material_id == red, "the subset's binding")
	testing.expect(t, prim_a[1].material_id == 0 && prim_b[1].material_id == net, "the instance's own binding")
	testing.expect_value(t, prim_a[0].material.roughness, f32(0.25))
	testing.expect(t, linalg.length(prim_a[0].material.color - Vec3{1, 0, 0}) < 1e-6, "Red's colour")

	// A's ops (translate, rotate 90 about z, scale 2) in centimeters, Z up:
	// the prototype's (1, 0, 0) is (10, 2, 0) on the stage, (0.1, 0, -0.02)
	// in o3d's meters, Y up.
	world := model_node_world(&m, Model_Node_Id(pa + 1))
	p := (world * Vec4{1, 0, 0, 1}).xyz
	testing.expect(t, linalg.length(p - Vec3{0.1, 0, -0.02}) < 1e-6, "transform ops, units and axis")
	// B's translate has only time samples: the first, (0, 50, 0).
	p = (model_node_world(&m, Model_Node_Id(pb + 1)) * Vec4{0, 0, 0, 1}).xyz
	testing.expect(t, linalg.length(p - Vec3{0, 0, -0.5}) < 1e-6, "the first time sample")

	// Face-varying indexed st: the quad's corners (0,0) (1,0) (1,1) (0,1).
	g, _ := geometry_get(&geos, prim_a[0].geometry)
	quad := geos.vertices[g.first_vertex:][:g.vertex_count]
	testing.expect(t, len(quad) == 6 && quad[0].uv == {0, 0} && quad[1].uv == {1, 0} && quad[2].uv == {1, 1} && quad[5].uv == {0, 1}, "uvs by their indices")
	testing.expect(t, linalg.length(quad[0].normal - Vec3{0, 0, 1}) < 1e-6, "smooth normals without authored ones")

	// Net: the colour texture, and the mask's red channel as the alpha map's
	// green, through the UsdTransform2d (scale (2, 3), translation (0.5,
	// 0.25): glTF's offset (0.5, 1 - 0.25 - 3)), cut at 0.5.
	nm := prim_b[1].material
	testing.expect(t, nm.color_map != 0 && nm.alpha_map != 0, "colour and alpha maps")
	testing.expect(t, .Alpha_Mask in nm.flags && nm.alpha_cutoff == 0.5, "opacityThreshold cuts")
	testing.expect_value(t, nm.alpha_transform, Texture_Transform{scale = {2, 3}, offset = {0.5, -2.25}})
	mask, _ := texture_get(&textures, nm.alpha_map)
	top := textures.texels[mask.mips[0]]
	bottom := textures.texels[mask.mips[0] + 2]
	testing.expect(t, top.g == 255 && bottom.g == 0, "the red channel, read where o3d reads alpha (green)")

	// The grid: no material, its display colour.
	grid := model_node(&m, "Grid")
	gm := m.primitives[m.meshes[m.nodes.mesh[grid - 1]].first_primitive].material
	testing.expect(t, abs(gm.color.g - linear_to_srgb(0.5)) < 1e-6 && gm.color.r == 0, "displayColor")
}

@(test)
test_usd_errors :: proc(t: ^testing.T) {
	geos: Geometries
	defer geometries_destroy(&geos)
	parse :: proc(bytes: []u8, geos: ^Geometries, limits: ^Limits = nil) -> Usd_Error {
		m, err := usd_parse(bytes, geos, limits = limits)
		model_destroy(&m)
		return err
	}
	testing.expect_value(t, parse(nil, &geos), Usd_Error.Empty)
	testing.expect_value(t, parse(transmute([]u8)string("#usda 1.0\n"), &geos), Usd_Error.Not_Usdc)
	entries, _ := zip_read_directory(USD_PARTS)
	layer, _ := zip_extract(USD_PARTS, entries, "parts.usdc")
	testing.expect_value(t, parse(layer[:len(layer) - 100], &geos), Usd_Error.Invalid)
	old := make([]u8, len(layer), context.temp_allocator)
	copy(old, layer)
	old[9] = 3 // version 0.3.0
	testing.expect_value(t, parse(old, &geos), Usd_Error.Unsupported_Version)
	// A USDZ with only its textures.
	no_layer := make([]u8, len(USD_PARTS), context.temp_allocator)
	copy(no_layer, USD_PARTS)
	for e in entries {
		if e.name == "parts.usdc" {
			// Rename the central directory's entry.
			for i in 0 ..< len(no_layer) - 10 {
				if string(no_layer[i:][:10]) == "parts.usdc" {
					no_layer[i + 6] = 'x'
				}
			}
		}
	}
	testing.expect_value(t, parse(no_layer, &geos), Usd_Error.No_Layer)
	// Past the values limit: the file is not read.
	limits := DEFAULT_LIMITS
	limits[.Gltf_Values] = 64
	testing.expect_value(t, parse(USD_PARTS, &geos, &limits), Usd_Error.Limit)
}

// model_bounds: the fixture's rest-pose box, from its nodes' transforms,
// the stage's centimeters and Z up (stage x 0..10, y 0..51 (B at y 50),
// z 0 -> o3d x 0..0.1, y 0, z -0.51..0).
@(test)
test_usd_model_bounds :: proc(t: ^testing.T) {
	geos: Geometries
	defer geometries_destroy(&geos)
	m, err := usd_parse(USD_PARTS, &geos)
	defer model_destroy(&m)
	testing.expect_value(t, err, Usd_Error.None)
	lo, hi, ok := model_bounds(&m, &geos)
	testing.expect(t, ok, "it has geometry")
	testing.expectf(t, linalg.length(lo - Vec3{0, 0, -0.51}) < 1e-5 && linalg.length(hi - Vec3{0.1, 0, 0}) < 1e-5, "box %v .. %v", lo, hi)
}
