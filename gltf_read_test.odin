package o3d

import "core:encoding/base64"
import "core:fmt"
import "core:math"
import "core:math/linalg"
import "core:strings"
import "core:testing"

// A GLB: its JSON chunk (padded with spaces) and binary chunk.
@(private = "file")
make_glb :: proc(json_text: string, bin: []u8, allocator := context.temp_allocator) -> []u8 {
	json_len := (len(json_text) + 3) &~ 3
	bin_len := (len(bin) + 3) &~ 3
	out := make([dynamic]u8, 0, 28 + json_len + bin_len, allocator)
	put :: proc(out: ^[dynamic]u8, v: int) {
		append(out, u8(v), u8(v >> 8), u8(v >> 16), u8(v >> 24))
	}
	append(&out, 'g', 'l', 'T', 'F')
	put(&out, 2)
	put(&out, 28 + json_len + bin_len)
	put(&out, json_len)
	put(&out, 0x4E4F534A)
	append(&out, json_text)
	for len(out) < 20 + json_len {
		append(&out, ' ')
	}
	put(&out, bin_len)
	put(&out, 0x004E4942)
	append(&out, ..bin)
	for len(out) < cap(out) {
		append(&out, 0)
	}
	return out[:]
}

// Sparse positions and indices, normalized bytes and a GLB binary chunk.
@(test)
test_gltf_read_glb_and_sparse :: proc(t: ^testing.T) {
	Bin :: struct #packed {
		pos:        [3][3]f32,
		pos_at:     [4]u8,
		pos_values: [3]f32,
		idx:        [4]u16,
		idx_at:     [4]u8,
		idx_values: [2]u16,
		uv:         [8]i8,
	}
	b := Bin {
		pos        = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}},
		pos_at     = {1, 0, 0, 0},
		pos_values = {5, 5, 5},
		idx        = {0, 1, 2, 0},
		idx_at     = {0, 0, 0, 0},
		idx_values = {2, 0},
		uv         = {-128, 127, 0, -64, 64, 0, 0, 0},
	}
	src := `{"asset": {"version": "2.0"},
  "buffers": [{"byteLength": 76}],
  "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 36}, {"buffer": 0, "byteOffset": 36, "byteLength": 4},
    {"buffer": 0, "byteOffset": 40, "byteLength": 12}, {"buffer": 0, "byteOffset": 52, "byteLength": 6},
    {"buffer": 0, "byteOffset": 60, "byteLength": 1}, {"buffer": 0, "byteOffset": 64, "byteLength": 2},
    {"buffer": 0, "byteOffset": 68, "byteLength": 6}],
  "accessors": [
    {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
     "sparse": {"count": 1, "indices": {"bufferView": 1, "componentType": 5121}, "values": {"bufferView": 2}}},
    {"bufferView": 3, "componentType": 5123, "count": 3, "type": "SCALAR",
     "sparse": {"count": 1, "indices": {"bufferView": 4, "componentType": 5121}, "values": {"bufferView": 5}}},
    {"bufferView": 6, "componentType": 5120, "normalized": true, "count": 3, "type": "VEC2"}],
  "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "TEXCOORD_0": 2}, "indices": 1}]}],
  "nodes": [{"mesh": 0}]}`
	glb := make_glb(src, ([^]u8)(&b)[:size_of(Bin)])

	doc, err := gltf_read(glb)
	defer gltf_document_destroy(&doc)
	testing.expect_value(t, err, Gltf_Error.None)
	pos: [9]f32
	testing.expect_value(t, gltf_accessor_floats(&doc, 0, pos[:]), 9)
	testing.expect_value(t, pos, [9]f32{0, 0, 0, 5, 5, 5, 0, 1, 0})
	idx: [3]u32
	testing.expect_value(t, gltf_accessor_uints(&doc, 1, idx[:]), 3)
	testing.expect_value(t, idx, [3]u32{2, 1, 2})
	uv: [6]f32
	gltf_accessor_floats(&doc, 2, uv[:])
	testing.expect_value(t, uv, [6]f32{-1, 1, 0, -64.0 / 127, 64.0 / 127, 0})
	// Fewer floats than the accessor has: whole elements only.
	testing.expect_value(t, gltf_accessor_floats(&doc, 0, pos[:5]), 3)

	geos: Geometries
	defer geometries_destroy(&geos)
	m, merr := gltf_parse(glb, &geos)
	defer model_destroy(&m)
	testing.expect_value(t, merr, Gltf_Error.None)
	testing.expect_value(t, len(geos.vertices), 3)
	testing.expect(t, len(geos.indices) == 3 && geos.indices[0] == 2 && geos.indices[1] == 1 && geos.indices[2] == 2, "sparse indices")
	testing.expect_value(t, geos.vertices[1].position, Vec3{5, 5, 5})
	testing.expect_value(t, geos.vertices[0].uv, Vec2{-1, 0})
}

@(private = "file")
read_error :: proc(src: string) -> Gltf_Error {
	doc, err := gltf_read(transmute([]u8)src)
	gltf_document_destroy(&doc)
	return err
}

@(test)
test_gltf_read_errors :: proc(t: ^testing.T) {
	A :: `{"asset": {"version": "2.0"}, `
	testing.expect_value(t, read_error(A + `"nodes": [{"children": [2]}, {"children": [2]}, {}]}`), Gltf_Error.Invalid_Gltf)
	testing.expect_value(t, read_error(A + `"nodes": [{"children": [1]}, {"children": [0]}]}`), Gltf_Error.Invalid)
	testing.expect_value(t, read_error(A + `"nodes": [{"children": [0]}]}`), Gltf_Error.Invalid)
	testing.expect_value(t, read_error(A + `"nodes": [{"mesh": 3}]}`), Gltf_Error.Invalid_Gltf)
	testing.expect_value(t, read_error(A + `"scenes": [{"nodes": [1]}], "nodes": [{"children": [1]}, {}]}`), Gltf_Error.Invalid_Gltf)
	testing.expect_value(t, read_error(A + `"nodes": {"a": {}}}`), Gltf_Error.Legacy_Gltf)
	testing.expect_value(t, read_error(`{"asset": {"version": "1.0"}}`), Gltf_Error.Legacy_Gltf)
	testing.expect_value(t, read_error(A + `"nodes": [}`), Gltf_Error.Invalid_Json)
	testing.expect_value(t, read_error(A + `"buffers": [{"uri": "http://example.com/a.bin", "byteLength": 4}]}`), Gltf_Error.Unknown_Format)
	testing.expect_value(t, read_error(A + `"buffers": [{"uri": "data:application/octet-stream,AAAA", "byteLength": 3}]}`), Gltf_Error.Unknown_Format)
	testing.expect_value(t, read_error(A + `"buffers": [{"uri": "data:application/octet-stream;base64,AAAA", "byteLength": 4}]}`), Gltf_Error.Buffer_Io)
	// An accessor past its view, a view past its buffer.
	B :: A + `"buffers": [{"uri": "data:application/octet-stream;base64,AAAAAAAAAAAAAAAA", "byteLength": 12}], `
	testing.expect_value(t, read_error(B + `"bufferViews": [{"buffer": 0, "byteLength": 12}], "accessors": [{"bufferView": 0, "componentType": 5126, "count": 2, "type": "VEC3"}]}`), Gltf_Error.Invalid)
	testing.expect_value(t, read_error(B + `"bufferViews": [{"buffer": 0, "byteOffset": 4, "byteLength": 12}]}`), Gltf_Error.Invalid)
	testing.expect_value(t, read_error(B + `"bufferViews": [{"buffer": 0, "byteLength": 12}], "accessors": [{"bufferView": 0, "componentType": 5126, "count": 1, "type": "VEC3"}]}`), Gltf_Error.None)

	glb := make_glb(`{"asset": {"version": "2.0"}, "buffers": [{"byteLength": 8}]}`, {1, 2, 3, 4})
	testing.expect_value(t, read_error(string(glb)), Gltf_Error.Data_Too_Short)
	testing.expect_value(t, read_error(string(glb[:30])), Gltf_Error.Data_Too_Short)
}

@(test)
test_gltf_limits :: proc(t: ^testing.T) {
	geos: Geometries
	defer geometries_destroy(&geos)
	A :: `{"asset": {"version": "2.0"}, `

	// Nested JSON past the depth limit fails before it is parsed.
	deep := strings.concatenate({A, `"extras": `, strings.repeat("[", 5000, context.temp_allocator), strings.repeat("]", 5000, context.temp_allocator), "}"}, context.temp_allocator)
	m, err := gltf_parse(transmute([]u8)deep, &geos)
	testing.expect_value(t, err, Gltf_Error.Limit)
	testing.expect(t, len(m.warnings) == 1 && strings.contains(m.warnings[0], "nesting depth"), "the limit is named")
	model_destroy(&m)

	limits := DEFAULT_LIMITS
	limits[.Gltf_Nodes] = 2
	m, err = gltf_parse(transmute([]u8)string(A + `"nodes": [{}, {}, {}]}`), &geos, limits = &limits)
	testing.expect_value(t, err, Gltf_Error.Limit)
	model_destroy(&m)

	// A hierarchy deeper than the limit.
	limits = DEFAULT_LIMITS
	limits[.Depth] = 3
	chain := A + `"nodes": [{"children": [1]}, {"children": [2]}, {"children": [3]}, {"children": [4]}, {}]}`
	m, err = gltf_parse(transmute([]u8)string(chain), &geos, limits = &limits)
	testing.expect_value(t, err, Gltf_Error.Limit)
	model_destroy(&m)

	// A buffer of more bytes than allowed.
	limits = DEFAULT_LIMITS
	limits[.Gltf_Buffer_Bytes] = 8
	m, err = gltf_parse(transmute([]u8)string(A + `"buffers": [{"uri": "data:application/octet-stream;base64,AAAAAAAAAAAAAAAA", "byteLength": 12}]}`), &geos, limits = &limits)
	testing.expect_value(t, err, Gltf_Error.Limit)
	model_destroy(&m)

	// Values past the limit skip the primitive (an accessor without a view
	// is all zeros, so its count is all it costs to ask for).
	limits = DEFAULT_LIMITS
	limits[.Gltf_Values] = 1000
	ask :: proc(count: int) -> string {
		return fmt.tprintf(`{{"asset": {{"version": "2.0"}}, "accessors": [{{"componentType": 5126, "count": %d, "type": "VEC3"}}], "meshes": [{{"primitives": [{{"attributes": {{"POSITION": 0}}}}]}}], "nodes": [{{"mesh": 0}}]}}`, count)
	}
	m, err = gltf_parse(transmute([]u8)ask(100), &geos, limits = &limits)
	testing.expect_value(t, err, Gltf_Error.None)
	testing.expect_value(t, len(m.primitives), 1)
	testing.expect_value(t, len(m.warnings), 0)
	model_destroy(&m)
	m, err = gltf_parse(transmute([]u8)ask(1 << 40), &geos, limits = &limits)
	testing.expect_value(t, err, Gltf_Error.None)
	testing.expect_value(t, len(m.primitives), 0)
	testing.expect(t, len(m.warnings) == 1 && strings.contains(m.warnings[0], "accessor values"), "the skipped primitive is warned of")
	model_destroy(&m)
}

@(test)
test_gltf_resolve :: proc(t: ^testing.T) {
	testing.expect_value(t, gltf_resolve("models", "a%20b%2Fc.bin", context.temp_allocator), "models/a b/c.bin")
	testing.expect_value(t, gltf_resolve("", "x%zz%4", context.temp_allocator), "x%zz%4")
	encoded := base64.encode(transmute([]u8)string("hi"), allocator = context.temp_allocator)
	src := fmt.tprintf(`{{"asset": {{"version": "2.0"}}, "buffers": [{{"uri": "data:application/octet-stream;base64,%s", "byteLength": 2}}]}}`, encoded)
	doc, err := gltf_read(transmute([]u8)src)
	defer gltf_document_destroy(&doc)
	testing.expect_value(t, err, Gltf_Error.None)
	testing.expect_value(t, string(doc.buffers[0]), "hi")
}

// Parts by durable id: names found once, rows by node and material, a
// node's row holding for the nodes under it, transforms on a node (one
// given by a matrix too), world matrices to attach to, and the extensions
// a material has that o3d does not draw named in a warning.
@(test)
test_model_parts :: proc(t: ^testing.T) {
	tri := [9]f32{0, 0, 0, 1, 0, 0, 0, 1, 0}
	data := base64.encode(([^]u8)(&tri)[:size_of(tri)], allocator = context.temp_allocator)
	src := fmt.tprintf(`{{"asset": {{"version": "2.0"}}, "extensionsUsed": ["KHR_materials_transmission"],
  "buffers": [{{"uri": "data:application/octet-stream;base64,%s", "byteLength": 36}}],
  "bufferViews": [{{"buffer": 0, "byteLength": 36}}],
  "accessors": [{{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "min": [0, 0, 0], "max": [1, 1, 0]}}],
  "materials": [{{"name": "Hull"}}, {{"name": "Steel", "extensions": {{"KHR_materials_transmission": {{"transmissionFactor": 1}}}}}}],
  "meshes": [{{"primitives": [{{"attributes": {{"POSITION": 0}}, "material": 0}}, {{"attributes": {{"POSITION": 0}}, "material": 1}}]}}],
  "nodes": [{{"name": "Ship", "mesh": 0, "children": [1, 2]}}, {{"name": "Net", "mesh": 0}},
    {{"name": "Wheel", "translation": [2, 0, 0], "children": [3]}}, {{"name": "Hub", "mesh": 0, "matrix": [2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 3, 0, 1]}}],
  "scenes": [{{"nodes": [0]}}]}}`, data)
	geos: Geometries
	defer geometries_destroy(&geos)
	m, err := gltf_parse(transmute([]u8)src, &geos)
	defer model_destroy(&m)
	testing.expect_value(t, err, Gltf_Error.None)
	testing.expect(t, len(m.warnings) == 1 && m.warnings[0] == `KHR_materials_transmission is not supported: drawn without it (materials "Steel")`, "the material is named")

	ship, net, wheel, hub := model_node(&m, "Ship"), model_node(&m, "Net"), model_node(&m, "Wheel"), model_node(&m, "Hub")
	steel := model_material(&m, "Steel")
	testing.expect(t, ship != 0 && net != 0 && wheel != 0 && hub != 0 && steel != 0 && model_node(&m, "Nope") == 0 && model_node(&m, "") == 0, "ids by name")

	scene: Scene
	defer scene_destroy(&scene)
	parts: Model_Parts
	defer model_parts_destroy(&parts)
	model_parts_init(&parts, &m)
	gltf_draw(&scene, &m, flags = {.Cast_Shadow, .Receive_Shadow}, parts = &parts)
	testing.expect_value(t, len(scene.draws), 6)

	// The ship casts no shadow but its net does; the net is hidden in the
	// second drawing; steel is drawn as another material.
	parts.nodes[ship - 1].remove = {.Cast_Shadow}
	parts.nodes[net - 1].add = {.Cast_Shadow}
	glass := MATERIAL_DEFAULT
	glass.opacity, glass.flags = 0.3, {.Transparent}
	parts.materials[steel - 1].replace = true
	parts.replacements[steel - 1] = glass
	scene_clear(&scene)
	gltf_draw(&scene, &m, flags = {.Cast_Shadow}, parts = &parts)
	testing.expect_value(t, len(scene.draws), 6)
	casts := 0
	for mat in scene.draws.material[:len(scene.draws)] {
		casts += 1 if .Cast_Shadow in mat.flags else 0
	}
	testing.expect_value(t, casts, 2) // Net's two primitives
	testing.expect_value(t, scene.draws.material[1].opacity, f32(0.3))
	testing.expect(t, .Transparent in scene.draws.material[1].flags && .Cast_Shadow not_in scene.draws.material[1].flags, "steel replaced, then the ship's row")

	parts.nodes[net - 1].hidden = true
	parts.materials[model_material(&m, "Hull") - 1].hidden = true
	scene_clear(&scene)
	gltf_draw(&scene, &m, parts = &parts)
	testing.expect_value(t, len(scene.draws), 2) // Ship's and Hub's steel

	// Turning the wheel turns the hub under it; moving the hub keeps its
	// matrix's scale.
	parts.transforms[wheel - 1] = {set = {.Rotation}, rotation = linalg.quaternion_angle_axis_f32(math.PI / 2, {0, 0, 1})}
	parts.transforms[hub - 1] = {set = {.Translation}, translation = {0, 1, 0}}
	scene_clear(&scene)
	gltf_draw(&scene, &m, parts = &parts)
	hub_world := model_node_world(&m, hub)
	testing.expect(t, linalg.length(Vec3{hub_world[0, 3], hub_world[1, 3], hub_world[2, 3]} - Vec3{1, 0, 0}) < 1e-5, "the hub follows the wheel")
	testing.expect(t, linalg.length(Vec3{hub_world[0, 1], hub_world[1, 1], hub_world[2, 1]} - Vec3{-2, 0, 0}) < 1e-5, "and keeps its scale")
	wheel_world := model_node_world(&m, wheel)
	testing.expect(t, abs(wheel_world[0, 3] - 2) < 1e-6, "the wheel keeps its translation")
}
