package o3d

import "core:encoding/base64"
import "core:fmt"
import "core:math"
import "core:math/linalg"
import "core:mem"
import "core:strings"
import "core:testing"

// Builds a self-contained glTF 2.0 JSON (with embedded base64 binary buffer)
// containing:
// - A 2-triangle quad mesh (positions, normals, UVs, indices)
// - A PBR material with baseColorFactor and an embedded 1x1 PNG texture
// - A 2-node parent -> child hierarchy
// - An animation ("slide_and_spin") with LINEAR translation on the parent and
//   LINEAR rotation + CUBICSPLINE scale on the child over t in [0, 1].
@(private = "file")
make_test_gltf :: proc(allocator := context.temp_allocator) -> string {
	// Binary buffer layout (all 4-byte aligned):
	// 0..48:    4 vec3 positions
	// 48..96:   4 vec3 normals
	// 96..128:  4 vec2 uvs
	// 128..144: 6 u16 indices (12 bytes) + 4 bytes padding
	// 144..152: 2 f32 animation times: [0.0, 1.0]
	// 152..176: 2 vec3 translations: [(-1, 0, 0), (1, 0, 0)]
	// 176..208: 2 vec4 rotations: identity -> 90 deg around Z
	// 208..280: 6 vec3 cubic spline scale: (in=0, val=(1,1,1), out=0), (in=0, val=(2,2,1), out=0)
	Bin :: struct #packed {
		pos:   [4][3]f32,
		norm:  [4][3]f32,
		uv:    [4][2]f32,
		idx:   [6]u16,
		_pad:  [2]u16,
		times: [2]f32,
		trans: [2][3]f32,
		rot:   [2][4]f32,
		scale: [6][3]f32,
	}
	#assert(size_of(Bin) == 280)

	s45 := math.sqrt_f32(0.5)
	b := Bin {
		pos = {
			{-0.8, -0.8, 0},
			{ 0.8, -0.8, 0},
			{ 0.8,  0.8, 0},
			{-0.8,  0.8, 0},
		},
		norm = {
			{0, 0, 1},
			{0, 0, 1},
			{0, 0, 1},
			{0, 0, 1},
		},
		uv = {
			{0, 1},
			{1, 1},
			{1, 0},
			{0, 0},
		},
		idx   = {0, 1, 2, 0, 2, 3},
		times = {0, 1},
		trans = {{-1, 0, 0}, {1, 0, 0}},
		rot   = {{0, 0, 0, 1}, {0, 0, s45, s45}},
		scale = {
			{0, 0, 0}, {1, 1, 1}, {0, 0, 0},
			{0, 0, 0}, {2, 2, 1}, {0, 0, 0},
		},
	}
	b64, _ := base64.encode(([^]u8)(&b)[:size_of(Bin)], allocator = allocator)

	// Minimal valid 1x1 white RGBA PNG in base64.
	PNG_1X1 :: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4////fwAJ+wP9KobjigAAAABJRU5ErkJggg=="

	return fmt.aprintf(
		`{{
  "asset": {{"version": "2.0"}},
  "buffers": [{{"uri": "data:application/octet-stream;base64,%s", "byteLength": 280}}],
  "bufferViews": [
    {{"buffer": 0, "byteOffset": 0,   "byteLength": 48}},
    {{"buffer": 0, "byteOffset": 48,  "byteLength": 48}},
    {{"buffer": 0, "byteOffset": 96,  "byteLength": 32}},
    {{"buffer": 0, "byteOffset": 128, "byteLength": 12}},
    {{"buffer": 0, "byteOffset": 144, "byteLength": 8}},
    {{"buffer": 0, "byteOffset": 152, "byteLength": 24}},
    {{"buffer": 0, "byteOffset": 176, "byteLength": 32}},
    {{"buffer": 0, "byteOffset": 208, "byteLength": 72}}
  ],
  "accessors": [
    {{"bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3", "min": [-0.8, -0.8, 0], "max": [0.8, 0.8, 0]}},
    {{"bufferView": 1, "componentType": 5126, "count": 4, "type": "VEC3"}},
    {{"bufferView": 2, "componentType": 5126, "count": 4, "type": "VEC2"}},
    {{"bufferView": 3, "componentType": 5123, "count": 6, "type": "SCALAR"}},
    {{"bufferView": 4, "componentType": 5126, "count": 2, "type": "SCALAR", "min": [0], "max": [1]}},
    {{"bufferView": 5, "componentType": 5126, "count": 2, "type": "VEC3"}},
    {{"bufferView": 6, "componentType": 5126, "count": 2, "type": "VEC4"}},
    {{"bufferView": 7, "componentType": 5126, "count": 6, "type": "VEC3"}}
  ],
  "images": [{{"uri": "data:image/png;base64,%s"}}],
  "textures": [{{"source": 0}}],
  "materials": [{{
    "pbrMetallicRoughness": {{
      "baseColorFactor": [0.8, 0.2, 0.1, 1.0],
      "baseColorTexture": {{"index": 0}},
      "metallicFactor": 0.1,
      "roughnessFactor": 0.6
    }},
    "doubleSided": true
  }}],
  "meshes": [{{
    "primitives": [{{
      "attributes": {{"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2}},
      "indices": 3,
      "material": 0
    }}]
  }}],
  "nodes": [
    {{"name": "root", "children": [1], "translation": [-1, 0, 0]}},
    {{"name": "quad", "mesh": 0}}
  ],
  "scenes": [{{"nodes": [0]}}],
  "scene": 0,
  "animations": [{{
    "name": "slide_and_spin",
    "samplers": [
      {{"input": 4, "output": 5, "interpolation": "LINEAR"}},
      {{"input": 4, "output": 6, "interpolation": "LINEAR"}},
      {{"input": 4, "output": 7, "interpolation": "CUBICSPLINE"}}
    ],
    "channels": [
      {{"sampler": 0, "target": {{"node": 0, "path": "translation"}}}},
      {{"sampler": 1, "target": {{"node": 1, "path": "rotation"}}}},
      {{"sampler": 2, "target": {{"node": 1, "path": "scale"}}}}
    ]
  }}]
}}`,
		b64,
		PNG_1X1,
		allocator = allocator,
	)
}

@(test)
test_gltf_parse_and_animate :: proc(t: ^testing.T) {
	src := make_test_gltf(context.temp_allocator)
	geos: Geometries
	defer geometries_destroy(&geos)
	textures: Textures
	defer textures_destroy(&textures)

	model, err := gltf_parse(transmute([]u8)src, &geos, &textures)
	defer model_destroy(&model)
	testing.expect_value(t, err, Gltf_Error.None)
	testing.expect_value(t, len(model.meshes), 1)
	testing.expect_value(t, len(model.primitives), 1)
	testing.expect_value(t, len(model.nodes), 2)
	testing.expect_value(t, len(model.animations), 1)
	testing.expect_value(t, gltf_animation(&model, "slide_and_spin"), 0)
	testing.expect_value(t, gltf_animation(&model, "missing"), -1)
	testing.expect_value(t, len(textures.list), 1)

	// Topological node invariant: parent < i.
	for i in 0 ..< len(model.nodes) {
		testing.expect(t, model.nodes[i].parent < i32(i), "parent must precede child in topological order")
	}

	scene: Scene
	defer scene_destroy(&scene)

	// At t = 0: parent translation is (-1, 0, 0), child scale is (1, 1, 1).
	scene_clear(&scene)
	gltf_draw(&scene, &model, anim = 0, time = 0)
	testing.expect_value(t, len(scene.draws), 1)
	m0 := scene.draws[0].model
	testing.expect(t, abs(m0[0, 3] - (-1.0)) < 1e-4, "at t=0 x translation should be -1")
	testing.expect(t, abs(linalg.length(Vec3{m0[0, 0], m0[1, 0], m0[2, 0]}) - 1.0) < 1e-4, "at t=0 x scale should be 1")

	// At t = 0.5: parent translation is (0, 0, 0), child scale is (1.5, 1.5, 1).
	scene_clear(&scene)
	gltf_draw(&scene, &model, anim = 0, time = 0.5)
	m_mid := scene.draws[0].model
	testing.expect(t, abs(m_mid[0, 3]) < 1e-4, "at t=0.5 x translation should be 0")
	testing.expect(t, abs(linalg.length(Vec3{m_mid[0, 0], m_mid[1, 0], m_mid[2, 0]}) - 1.5) < 1e-4, "at t=0.5 x scale should be 1.5")

	// At t = 1.0 (non-looping): parent translation is (+1, 0, 0), child scale is (2, 2, 1).
	scene_clear(&scene)
	gltf_draw(&scene, &model, anim = 0, time = 1.0, loop = false)
	m1 := scene.draws[0].model
	testing.expect(t, abs(m1[0, 3] - 1.0) < 1e-4, "at t=1.0 x translation should be +1")
	testing.expect(t, abs(linalg.length(Vec3{m1[0, 0], m1[1, 0], m1[2, 0]}) - 2.0) < 1e-4, "at t=1.0 x scale should be 2")

	// Render at t = 0 (quad on left) vs t = 1 (quad on right) and verify pixels move.
	sw: Software
	defer software_destroy(&sw)

	scene_clear(&scene)
	scene.camera.position = {0, 0, 4}
	append(&scene.lights, Light{kind = .Directional, color = {1, 1, 1}, intensity = 3, position = {0, 0, 5}})
	gltf_draw(&scene, &model, anim = 0, time = 0, loop = false)
	px0 := software_render(&sw, &scene, &geos, nil, 64, 64, nil, &textures)
	left_t0 := px0[(32 * 64 + 18) * 4 + 3]
	right_t0 := px0[(32 * 64 + 46) * 4 + 3]
	testing.expect(t, left_t0 > 200 && right_t0 == 0, "at t=0 quad should cover left side")

	// Verify 0 warm-frame allocations during animated gltf_draw + software_render.
	track: mem.Tracking_Allocator
	mem.tracking_allocator_init(&track, context.allocator)
	defer mem.tracking_allocator_destroy(&track)
	{
		context.allocator = mem.tracking_allocator(&track)
		scene_clear(&scene)
		scene.camera.position = {0, 0, 4}
		append(&scene.lights, Light{kind = .Directional, color = {1, 1, 1}, intensity = 3, position = {0, 0, 5}})
		gltf_draw(&scene, &model, anim = 0, time = 1.0, loop = false)
		px1 := software_render(&sw, &scene, &geos, nil, 64, 64, nil, &textures)
		left_t1 := px1[(32 * 64 + 18) * 4 + 3]
		right_t1 := px1[(32 * 64 + 46) * 4 + 3]
		testing.expect(t, left_t1 == 0 && right_t1 > 200, "at t=1 quad should cover right side")
	}
	testing.expect_value(t, len(track.allocation_map), 0)
	testing.expect_value(t, track.total_memory_allocated, 0)
}

// The loader follows three.js's GLTFLoader and the spec: the default scene
// only; OPAQUE ignores alpha, BLEND blends; the metallicRoughness texture
// is both roughness (green) and metalness (blue) map; a primitive without
// normals is flat shaded; an index past the vertices rejects the file.
@(test)
test_gltf_conventions :: proc(t: ^testing.T) {
	base := make_test_gltf(context.temp_allocator)
	load :: proc(t: ^testing.T, src: string) -> (m: Model, geos: Geometries, textures: Textures, ok: bool) {
		err: Gltf_Error
		m, err = gltf_parse(transmute([]u8)src, &geos, &textures)
		ok = err == .None
		return
	}
	variant :: proc(src: string, pairs: ..string) -> string {
		out := src
		for i := 0; i + 1 < len(pairs); i += 2 {
			out, _ = strings.replace_all(out, pairs[i], pairs[i + 1], context.temp_allocator)
		}
		return out
	}
	{
		// Opaque with alpha 0.5: drawn opaque. A second scene's extra root
		// node is not drawn.
		src := variant(base, `[0.8, 0.2, 0.1, 1.0]`, `[0.8, 0.2, 0.1, 0.5]`, `"roughnessFactor": 0.6`, `"roughnessFactor": 0.6, "metallicRoughnessTexture": {"index": 0}`, `"scenes": [{"nodes": [0]}]`, `"scenes": [{"nodes": [0]}, {"nodes": [2]}]`, `{"name": "quad", "mesh": 0}`, `{"name": "quad", "mesh": 0}, {"name": "other", "mesh": 0}`)
		m, geos, textures, ok := load(t, src)
		defer model_destroy(&m)
		defer geometries_destroy(&geos)
		defer textures_destroy(&textures)
		testing.expect(t, ok)
		mat := m.primitives[0].material
		testing.expect(t, .Transparent not_in mat.flags && mat.opacity == 1, "OPAQUE ignores alpha")
		testing.expect(t, mat.roughness_map != 0 && mat.metalness_map == mat.roughness_map, "metallicRoughness is both maps")
		testing.expect_value(t, len(m.nodes), 2) // root and quad: not the other scene's node
	}
	{
		src := variant(base, `"doubleSided": true`, `"doubleSided": true, "alphaMode": "BLEND"`, `{"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2}`, `{"POSITION": 0, "TEXCOORD_0": 2}`)
		m, geos, textures, ok := load(t, src)
		defer model_destroy(&m)
		defer geometries_destroy(&geos)
		defer textures_destroy(&textures)
		testing.expect(t, ok)
		mat := m.primitives[0].material
		testing.expect(t, .Transparent in mat.flags && blended(mat), "BLEND blends (its colour map's alpha)")
		testing.expect(t, .Flat_Shading in mat.flags, "no normals: flat shaded")
	}
	{
		// The index accessor reads past the four vertices (its view holds
		// u16 indices; read as u32 they are out of range): rejected.
		src := variant(base, `{"bufferView": 3, "componentType": 5123, "count": 6, "type": "SCALAR"}`, `{"bufferView": 3, "componentType": 5125, "count": 3, "type": "SCALAR"}`)
		m, geos, textures, ok := load(t, src)
		defer model_destroy(&m)
		defer geometries_destroy(&geos)
		defer textures_destroy(&textures)
		testing.expect(t, !ok || len(m.primitives) == 0, "out-of-range indices rejected")
	}
}

// A failed load says why: no data, not glTF, a buffer file missing,
// validation; and a model posed with callers' pose buffers is the same as
// posed with its own, the model only read.
@(test)
test_gltf_errors_and_poses :: proc(t: ^testing.T) {
	geos: Geometries
	defer geometries_destroy(&geos)
	textures: Textures
	defer textures_destroy(&textures)
	_, empty := gltf_parse(nil, &geos, &textures)
	testing.expect_value(t, empty, Gltf_Error.Empty)
	junk := "not a model"
	_, unknown := gltf_parse(transmute([]u8)junk, &geos, &textures)
	testing.expect(t, unknown != .None && GLTF_ERROR_TEXT[unknown] != "", "junk is an error with words")
	missing := `{"asset": {"version": "2.0"}, "buffers": [{"uri": "nowhere.bin", "byteLength": 12}]}`
	_, buffer := gltf_parse(transmute([]u8)missing, &geos, &textures, "/nonexistent")
	testing.expect_value(t, buffer, Gltf_Error.Buffer_Not_Found)
	_, no_file := gltf_load("/nonexistent/model.glb", &geos, &textures)
	testing.expect_value(t, no_file, Gltf_Error.File_Not_Found)

	src := make_test_gltf(context.temp_allocator)
	model, err := gltf_parse(transmute([]u8)src, &geos, &textures)
	defer model_destroy(&model)
	testing.expect_value(t, err, Gltf_Error.None)
	own, mine: Scene
	defer scene_destroy(&own)
	defer scene_destroy(&mine)
	pose: Model_Pose
	defer model_pose_destroy(&pose)
	gltf_draw(&own, &model, anim = 0, time = 0.4)
	gltf_draw(&mine, &model, anim = 0, time = 0.4, pose = &pose)
	testing.expect_value(t, len(mine.draws), len(own.draws))
	testing.expect_value(t, mine.draws[0].model, own.draws[0].model)
}
