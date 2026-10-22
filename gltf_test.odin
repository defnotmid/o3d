package o3d

import "core:encoding/base64"
import "core:fmt"
import "core:math"
import "core:math/linalg"
import "core:mem"
import "core:os"
import "core:strings"
import "core:testing"
import image "deps:odecs/image"

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

// Hand-made skinned glTF 2.0: a vertical bar from y = 0 to y = 2 (6 vertices,
// 4 triangles) rigged to 2 joints:
// - joint 0 at (0, 0, 0), inverseBind = identity
// - joint 1 at (0, 1, 0) (child of joint 0), inverseBind = translate(0, -1, 0)
// Bottom vertices (y = 0) are weighted 100% to joint 0; middle vertices (y = 1)
// have unnormalized weights (2, 2, 0, 0) on joints (0, 1); top vertices (y = 2)
// are weighted 100% to joint 1.
// Animation "bend" rotates joint 1 by +90 deg around Z at t = 1 (bending the
// top of the bar from (0, 2, 0) to (-1, 1, 0)).
@(private = "file")
make_skinned_test_gltf :: proc(allocator := context.temp_allocator) -> string {
	Bin :: struct #packed {
		pos:     [6][3]f32,
		norm:    [6][3]f32,
		joints:  [6][4]u16,
		weights: [6][4]f32,
		idx:     [12]u16,
		ibm:     [2]Mat4,
		times:   [2]f32,
		rot:     [2][4]f32,
	}
	s45 := math.sqrt_f32(0.5)
	ibm0 := linalg.MATRIX4F32_IDENTITY
	ibm1 := linalg.MATRIX4F32_IDENTITY
	ibm1[1, 3] = -1 // translate(0, -1, 0)
	b := Bin {
		pos = {
			{-0.25, 0, 0}, {0.25, 0, 0},
			{-0.25, 1, 0}, {0.25, 1, 0},
			{-0.25, 2, 0}, {0.25, 2, 0},
		},
		norm = {
			{0, 0, 1}, {0, 0, 1},
			{0, 0, 1}, {0, 0, 1},
			{0, 0, 1}, {0, 0, 1},
		},
		joints = {
			{0, 0, 0, 0}, {0, 0, 0, 0},
			{0, 1, 0, 0}, {0, 1, 0, 0},
			{1, 0, 0, 0}, {1, 0, 0, 0},
		},
		// Unnormalized weights on the middle row to verify load-time normalization.
		weights = {
			{1, 0, 0, 0}, {1, 0, 0, 0},
			{2, 2, 0, 0}, {2, 2, 0, 0},
			{1, 0, 0, 0}, {1, 0, 0, 0},
		},
		idx   = {0, 1, 3, 0, 3, 2, 2, 3, 5, 2, 5, 4},
		ibm   = {ibm0, ibm1},
		times = {0, 1},
		rot   = {{0, 0, 0, 1}, {0, 0, s45, s45}},
	}
	b64, _ := base64.encode(([^]u8)(&b)[:size_of(Bin)], allocator = allocator)
	return fmt.aprintf(
		`{{
  "asset": {{"version": "2.0"}},
  "buffers": [{{"uri": "data:application/octet-stream;base64,%s", "byteLength": %d}}],
  "bufferViews": [
    {{"buffer": 0, "byteOffset": 0,   "byteLength": 72}},
    {{"buffer": 0, "byteOffset": 72,  "byteLength": 72}},
    {{"buffer": 0, "byteOffset": 144, "byteLength": 48}},
    {{"buffer": 0, "byteOffset": 192, "byteLength": 96}},
    {{"buffer": 0, "byteOffset": 288, "byteLength": 24}},
    {{"buffer": 0, "byteOffset": 312, "byteLength": 128}},
    {{"buffer": 0, "byteOffset": 440, "byteLength": 8}},
    {{"buffer": 0, "byteOffset": 448, "byteLength": 32}}
  ],
  "accessors": [
    {{"bufferView": 0, "componentType": 5126, "count": 6, "type": "VEC3", "min": [-0.25, 0, 0], "max": [0.25, 2, 0]}},
    {{"bufferView": 1, "componentType": 5126, "count": 6, "type": "VEC3"}},
    {{"bufferView": 2, "componentType": 5123, "count": 6, "type": "VEC4"}},
    {{"bufferView": 3, "componentType": 5126, "count": 6, "type": "VEC4"}},
    {{"bufferView": 4, "componentType": 5123, "count": 12, "type": "SCALAR"}},
    {{"bufferView": 5, "componentType": 5126, "count": 2, "type": "MAT4"}},
    {{"bufferView": 6, "componentType": 5126, "count": 2, "type": "SCALAR", "min": [0], "max": [1]}},
    {{"bufferView": 7, "componentType": 5126, "count": 2, "type": "VEC4"}}
  ],
  "materials": [{{
    "pbrMetallicRoughness": {{"baseColorFactor": [0.9, 0.4, 0.2, 1.0], "metallicFactor": 0.0, "roughnessFactor": 0.5}},
    "doubleSided": true
  }}],
  "meshes": [{{
    "primitives": [{{
      "attributes": {{"POSITION": 0, "NORMAL": 1, "JOINTS_0": 2, "WEIGHTS_0": 3}},
      "indices": 4,
      "material": 0
    }}]
  }}],
  "skins": [{{
    "inverseBindMatrices": 5,
    "joints": [1, 2]
  }}],
  "nodes": [
    {{"name": "bar", "mesh": 0, "skin": 0}},
    {{"name": "j0", "children": [2]}},
    {{"name": "j1", "translation": [0, 1, 0]}}
  ],
  "scenes": [{{"nodes": [0, 1]}}],
  "scene": 0,
  "animations": [{{
    "name": "bend",
    "samplers": [{{"input": 6, "output": 7, "interpolation": "LINEAR"}}],
    "channels": [{{"sampler": 0, "target": {{"node": 2, "path": "rotation"}}}}]
  }}]
}}`,
		b64,
		size_of(Bin),
		allocator = allocator,
	)
}

@(test)
test_gltf_skinning :: proc(t: ^testing.T) {
	src := make_skinned_test_gltf(context.temp_allocator)
	geos: Geometries
	defer geometries_destroy(&geos)
	model, err := gltf_parse(transmute([]u8)src, &geos)
	defer model_destroy(&model)
	testing.expect_value(t, err, Gltf_Error.None)
	testing.expect_value(t, len(model.skins), 1)
	testing.expect_value(t, model.skins[0].joint_count, u32(2))

	// Check normalized weights in skin_vertices.
	skin_verts := geometry_skin(&geos, model.primitives[0].geometry)
	testing.expect_value(t, len(skin_verts), 6)
	testing.expect(t, abs(skin_verts[2].weights.x - 0.5) < 1e-5 && abs(skin_verts[2].weights.y - 0.5) < 1e-5, "middle vertex weights normalized to 0.5, 0.5")

	scene: Scene
	defer scene_destroy(&scene)
	sw: Software
	defer software_destroy(&sw)

	// Bind pose (t = 0): top of bar is straight up at (0, 1.75, 0), nothing at (-0.75, 1, 0).
	scene_clear(&scene)
	scene.camera = {position = {0, 1, 3.5}, target = {0, 1, 0}, fov = 50, near = 0.1, far = 20}
	append(&scene.lights, Light{kind = .Directional, color = {1, 1, 1}, intensity = 3, position = {0, 1, 5}})
	gltf_draw(&scene, &model, anim = 0, time = 0, loop = false)
	testing.expect_value(t, len(scene_draw_joints(&scene, 0)), 2)
	px0 := software_render(&sw, &scene, &geos, nil, 64, 64)
	top_t0 := px0[(22 * 64 + 32) * 4 + 3]  // near (0, 1.5)
	left_t0 := px0[(32 * 64 + 22) * 4 + 3] // near (-0.5, 1.0)
	testing.expect(t, top_t0 > 200 && left_t0 == 0, "at t=0 bar is straight up")

	// Bent pose (t = 1) with zero warm-frame heap allocations: top half bends 90 deg left to (-1, 1, 0).
	track: mem.Tracking_Allocator
	mem.tracking_allocator_init(&track, context.allocator)
	defer mem.tracking_allocator_destroy(&track)
	{
		context.allocator = mem.tracking_allocator(&track)
		scene_clear(&scene)
		scene.camera = {position = {0, 1, 3.5}, target = {0, 1, 0}, fov = 50, near = 0.1, far = 20}
		append(&scene.lights, Light{kind = .Directional, color = {1, 1, 1}, intensity = 3, position = {0, 1, 5}})
		gltf_draw(&scene, &model, anim = 0, time = 1, loop = false)
		px1 := software_render(&sw, &scene, &geos, nil, 64, 64)
		top_t1 := px1[(22 * 64 + 32) * 4 + 3]
		left_t1 := px1[(32 * 64 + 22) * 4 + 3]
		testing.expect(t, top_t1 == 0 && left_t1 > 200, "at t=1 top half of bar bends to the left")
	}
	testing.expect_value(t, len(track.allocation_map), 0)
	testing.expect_value(t, track.total_memory_allocated, 0)

	// ShaderMaterial on a skinned draw: receives posed positions.
	shaders: Shaders
	defer shaders_destroy(&shaders)
	sh_id, sh_err := shader(
		&shaders,
		`void main() { gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
		`void main() { gl_FragColor = vec4(0.2, 0.9, 0.3, 1.0); }`,
	)
	testing.expect_value(t, len(sh_err), 0)
	custom_mat := MATERIAL_DEFAULT
	custom_mat.shader = sh_id
	custom_mat.side = .Double
	scene_clear(&scene)
	scene.camera = {position = {0, 1, 3.5}, target = {0, 1, 0}, fov = 50, near = 0.1, far = 20}
	gltf_draw(&scene, &model, anim = 0, time = 1, loop = false, override_material = &custom_mat)
	px_sh := software_render(&sw, &scene, &geos, &shaders, 64, 64)
	testing.expect(t, px_sh[(32 * 64 + 22) * 4 + 1] > 200, "ShaderMaterial on skinned draw renders bent bar on left")
}

// Hand-made two-target cube (8 vertices, 12 triangles):
// - Target 0 ("Tall"): stretches top 4 vertices (y > 0) by +1.2 in Y.
// - Target 1 ("Wide"): stretches right/left vertices by +/-1.2 in X and tilts normals.
// - Animation "morph_anim": LINEAR weights from [0, 0] at t=0 to [1, 0] at t=1.
@(private = "file")
make_morph_cube_gltf :: proc(allocator := context.temp_allocator) -> string {
	Bin :: struct #packed {
		pos:     [8][3]f32,
		norm:    [8][3]f32,
		d_pos0:  [8][3]f32,
		d_norm0: [8][3]f32,
		d_pos1:  [8][3]f32,
		d_norm1: [8][3]f32,
		idx:     [36]u16,
		times:   [2]f32,
		weights: [4]f32,
	}
	b := Bin {
		pos = {
			{-0.4, -0.4,  0.4}, { 0.4, -0.4,  0.4}, { 0.4,  0.4,  0.4}, {-0.4,  0.4,  0.4},
			{-0.4, -0.4, -0.4}, { 0.4, -0.4, -0.4}, { 0.4,  0.4, -0.4}, {-0.4,  0.4, -0.4},
		},
		norm = {
			{0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1},
			{0, 0, -1}, {0, 0, -1}, {0, 0, -1}, {0, 0, -1},
		},
		d_pos0 = {
			{0, 0, 0}, {0, 0, 0}, {0, 1.2, 0}, {0, 1.2, 0},
			{0, 0, 0}, {0, 0, 0}, {0, 1.2, 0}, {0, 1.2, 0},
		},
		d_norm0 = {
			{0, 0, 0}, {0, 0, 0}, {0, 0.2, 0}, {0, 0.2, 0},
			{0, 0, 0}, {0, 0, 0}, {0, 0.2, 0}, {0, 0.2, 0},
		},
		d_pos1 = {
			{-1.0, 0, 0}, {1.0, 0, 0}, {1.0, 0, 0}, {-1.0, 0, 0},
			{-1.0, 0, 0}, {1.0, 0, 0}, {1.0, 0, 0}, {-1.0, 0, 0},
		},
		d_norm1 = {
			{-0.2, 0, 0}, {0.2, 0, 0}, {0.2, 0, 0}, {-0.2, 0, 0},
			{-0.2, 0, 0}, {0.2, 0, 0}, {0.2, 0, 0}, {-0.2, 0, 0},
		},
		idx = {
			0, 1, 2, 0, 2, 3, // front
			5, 4, 7, 5, 7, 6, // back
			4, 0, 3, 4, 3, 7, // left
			1, 5, 6, 1, 6, 2, // right
			3, 2, 6, 3, 6, 7, // top
			4, 5, 1, 4, 1, 0, // bottom
		},
		times   = {0, 1},
		weights = {0, 0, 1, 0},
	}
	b64, _ := base64.encode(([^]u8)(&b)[:size_of(Bin)], allocator = allocator)
	return fmt.aprintf(
		`{{
  "asset": {{"version": "2.0"}},
  "buffers": [{{"uri": "data:application/octet-stream;base64,%s", "byteLength": %d}}],
  "bufferViews": [
    {{"buffer": 0, "byteOffset": 0,   "byteLength": 96}},
    {{"buffer": 0, "byteOffset": 96,  "byteLength": 96}},
    {{"buffer": 0, "byteOffset": 192, "byteLength": 96}},
    {{"buffer": 0, "byteOffset": 288, "byteLength": 96}},
    {{"buffer": 0, "byteOffset": 384, "byteLength": 96}},
    {{"buffer": 0, "byteOffset": 480, "byteLength": 96}},
    {{"buffer": 0, "byteOffset": 576, "byteLength": 72}},
    {{"buffer": 0, "byteOffset": 648, "byteLength": 8}},
    {{"buffer": 0, "byteOffset": 656, "byteLength": 16}}
  ],
  "accessors": [
    {{"bufferView": 0, "componentType": 5126, "count": 8, "type": "VEC3", "min": [-0.4, -0.4, -0.4], "max": [0.4, 0.4, 0.4]}},
    {{"bufferView": 1, "componentType": 5126, "count": 8, "type": "VEC3"}},
    {{"bufferView": 2, "componentType": 5126, "count": 8, "type": "VEC3"}},
    {{"bufferView": 3, "componentType": 5126, "count": 8, "type": "VEC3"}},
    {{"bufferView": 4, "componentType": 5126, "count": 8, "type": "VEC3"}},
    {{"bufferView": 5, "componentType": 5126, "count": 8, "type": "VEC3"}},
    {{"bufferView": 6, "componentType": 5123, "count": 36, "type": "SCALAR"}},
    {{"bufferView": 7, "componentType": 5126, "count": 2, "type": "SCALAR", "min": [0], "max": [1]}},
    {{"bufferView": 8, "componentType": 5126, "count": 4, "type": "SCALAR"}}
  ],
  "materials": [{{
    "pbrMetallicRoughness": {{"baseColorFactor": [0.9, 0.3, 0.2, 1.0], "metallicFactor": 0.0, "roughnessFactor": 0.5}},
    "doubleSided": true
  }}],
  "meshes": [{{
    "weights": [0.0, 0.0],
    "extras": {{"targetNames": ["Tall", "Wide"]}},
    "primitives": [{{
      "attributes": {{"POSITION": 0, "NORMAL": 1}},
      "targets": [
        {{"POSITION": 2, "NORMAL": 3}},
        {{"POSITION": 4, "NORMAL": 5}}
      ],
      "indices": 6,
      "material": 0
    }}]
  }}],
  "nodes": [{{"name": "cube", "mesh": 0}}],
  "scenes": [{{"nodes": [0]}}],
  "scene": 0,
  "animations": [{{
    "name": "morph_anim",
    "samplers": [{{"input": 7, "output": 8, "interpolation": "LINEAR"}}],
    "channels": [{{"sampler": 0, "target": {{"node": 0, "path": "weights"}}}}]
  }}]
}}`,
		b64,
		size_of(Bin),
		allocator = allocator,
	)
}

@(test)
test_gltf_morph_targets :: proc(t: ^testing.T) {
	src := make_morph_cube_gltf(context.temp_allocator)
	geos: Geometries
	defer geometries_destroy(&geos)
	model, err := gltf_parse(transmute([]u8)src, &geos)
	defer model_destroy(&model)
	testing.expect_value(t, err, Gltf_Error.None)
	testing.expect_value(t, gltf_morph_count(&model), 2)
	testing.expect_value(t, gltf_morph_target(&model, "Tall"), 0)
	testing.expect_value(t, gltf_morph_target(&model, "Wide"), 1)
	testing.expect_value(t, gltf_morph_target(&model, "Missing"), -1)

	geo, ok := geometry_get(&geos, model.primitives[0].geometry)
	testing.expect(t, ok && geo.morph_targets == 2 && geo.morph_radius >= 1.19)

	scene: Scene
	defer scene_destroy(&scene)
	sw: Software
	defer software_destroy(&sw)

	setup_cam :: proc(scene: ^Scene) {
		scene_clear(scene)
		scene.camera = {position = {0, 0.4, 3.8}, target = {0, 0.4, 0}, fov = 50, near = 0.1, far = 20}
		append(&scene.lights, Light{kind = .Directional, color = {1, 1, 1}, intensity = 3, position = {0, 1, 5}})
	}

	// 1. Weights 0: cube goes up to y = 0.4, so y ~ 1.2 (row 12) is empty and x ~ -1.1 (col 12) is empty.
	setup_cam(&scene)
	gltf_draw(&scene, &model)
	px0 := software_render(&sw, &scene, &geos, nil, 64, 64)
	top_0 := px0[(12 * 64 + 32) * 4 + 3]
	side_0 := px0[(40 * 64 + 12) * 4 + 3]
	center_0 := px0[(40 * 64 + 32) * 4 + 3]
	testing.expect(t, center_0 > 200 && top_0 == 0 && side_0 == 0, "weights 0: compact cube at center only")

	// 2. Weight 1 on target 0 (by index): top stretches to y = 1.6, covering row 12.
	setup_cam(&scene)
	gltf_draw(&scene, &model, morphs = {{index = 0, weight = 1}})
	px_tall := software_render(&sw, &scene, &geos, nil, 64, 64)
	testing.expect(t, px_tall[(12 * 64 + 32) * 4 + 3] > 200 && px_tall[(40 * 64 + 12) * 4 + 3] == 0, "Tall=1 reaches top")

	// 3. Animated at t = 1.0: morph_anim drives target 0 to 1.0.
	setup_cam(&scene)
	gltf_draw(&scene, &model, anim = 0, time = 1.0, loop = false)
	px_anim := software_render(&sw, &scene, &geos, nil, 64, 64)
	testing.expect(t, px_anim[(12 * 64 + 32) * 4 + 3] > 200, "animated weights at t=1 reach top")

	// 4. Overridden by name after animation, with zero warm-frame allocations:
	//    suppress "Tall" to 0 and set "Wide" to 1.
	pose: Model_Pose
	defer model_pose_destroy(&pose)
	gltf_draw(&scene, &model, anim = 0, time = 1.0, loop = false, pose = &pose, morphs = {{name = "Tall", weight = 0}, {name = "Wide", weight = 1}})
	_ = software_render(&sw, &scene, &geos, nil, 64, 64)

	track: mem.Tracking_Allocator
	mem.tracking_allocator_init(&track, context.allocator)
	defer mem.tracking_allocator_destroy(&track)
	{
		context.allocator = mem.tracking_allocator(&track)
		setup_cam(&scene)
		gltf_draw(&scene, &model, anim = 0, time = 1.0, loop = false, pose = &pose, morphs = {{name = "Tall", weight = 0}, {name = "Wide", weight = 1}})
		px_wide := software_render(&sw, &scene, &geos, nil, 64, 64)
		testing.expect(t, px_wide[(12 * 64 + 32) * 4 + 3] == 0 && px_wide[(40 * 64 + 12) * 4 + 3] > 200, "override Tall=0, Wide=1 stretches horizontally")
	}
	testing.expect_value(t, len(track.allocation_map), 0)
	testing.expect_value(t, track.total_memory_allocated, 0)

	// 5. Bounding sphere covers morphed height: place the cube below the view frustum so
	//    unmorphed (radius ~0.69) is culled, while Tall=1.5 expands radius into the view.
	scene_clear(&scene)
	scene.camera = {position = {0, 0, 4}, target = {0, 0, 0}, fov = 30, near = 0.1, far = 20}
	view, proj := camera_matrices(scene.camera, 64, 64)
	order: [dynamic]int
	defer delete(order)
	gltf_draw(&scene, &model, transform = compose({0, -1.9, 0}))
	draw_order(&scene, &geos, nil, view, proj, &order)
	testing.expect_value(t, len(order), 0)

	scene_clear(&scene)
	gltf_draw(&scene, &model, transform = compose({0, -1.9, 0}), morphs = {{name = "Tall", weight = 1.5}})
	draw_order(&scene, &geos, nil, view, proj, &order)
	testing.expect_value(t, len(order), 1)
}

@(test)
test_gltf_extensions_and_materials :: proc(t: ^testing.T) {
	// Quantized mesh with TEXCOORD_0 (normalized u16), TEXCOORD_1 (float),
	// KHR_materials_emissive_strength, KHR_texture_transform, occlusionTexture on UV1,
	// and alphaMode: MASK with alphaCutoff.
	Bin :: struct #packed {
		pos_i16: [4][4]i16, // 4 vertices, 3D i16 padded to 4
		norm_i8: [4][4]i8,  // 4 vertices, 3D normalized i8 padded to 4
		uv0_u16: [4][2]u16, // normalized u16
		uv1_f32: [4][2]f32,
		idx:     [6]u16,
		_pad:    [1]u16,
	}
	b := Bin {
		pos_i16 = {{-1, -1, 0, 0}, {1, -1, 0, 0}, {1, 1, 0, 0}, {-1, 1, 0, 0}},
		norm_i8 = {{0, 0, 127, 0}, {0, 0, 127, 0}, {0, 0, 127, 0}, {0, 0, 127, 0}},
		uv0_u16 = {{0, 65535}, {65535, 65535}, {65535, 0}, {0, 0}},
		uv1_f32 = {{0, 0}, {1, 0}, {1, 1}, {0, 1}},
		idx     = {0, 1, 2, 0, 2, 3},
	}
	b64, _ := base64.encode(([^]u8)(&b)[:size_of(Bin)], allocator = context.temp_allocator)
	PNG_1X1 :: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4////fwAJ+wP9KobjigAAAABJRU5ErkJggg=="
	src := fmt.tprintf(
		`{{
  "asset": {{"version": "2.0"}},
  "extensionsUsed": ["KHR_mesh_quantization", "KHR_materials_emissive_strength", "KHR_texture_transform"],
  "extensionsRequired": ["KHR_mesh_quantization"],
  "buffers": [{{"uri": "data:application/octet-stream;base64,%s", "byteLength": %d}}],
  "bufferViews": [
    {{"buffer": 0, "byteOffset": 0,  "byteLength": 32, "byteStride": 8}},
    {{"buffer": 0, "byteOffset": 32, "byteLength": 16, "byteStride": 4}},
    {{"buffer": 0, "byteOffset": 48, "byteLength": 16}},
    {{"buffer": 0, "byteOffset": 64, "byteLength": 32}},
    {{"buffer": 0, "byteOffset": 96, "byteLength": 12}}
  ],
  "accessors": [
    {{"bufferView": 0, "componentType": 5122, "count": 4, "type": "VEC3"}},
    {{"bufferView": 1, "componentType": 5120, "count": 4, "type": "VEC3", "normalized": true}},
    {{"bufferView": 2, "componentType": 5123, "count": 4, "type": "VEC2", "normalized": true}},
    {{"bufferView": 3, "componentType": 5126, "count": 4, "type": "VEC2"}},
    {{"bufferView": 4, "componentType": 5123, "count": 6, "type": "SCALAR"}}
  ],
  "images": [{{"uri": "data:image/png;base64,%s"}}],
  "textures": [{{"source": 0}}],
  "materials": [{{
    "alphaMode": "MASK",
    "alphaCutoff": 0.65,
    "emissiveFactor": [0.2, 0.4, 0.8],
    "extensions": {{
      "KHR_materials_emissive_strength": {{"emissiveStrength": 3.5}}
    }},
    "pbrMetallicRoughness": {{
      "baseColorTexture": {{
        "index": 0,
        "extensions": {{
          "KHR_texture_transform": {{"offset": [0.25, 0.5], "rotation": 0.5, "scale": [2.0, 3.0], "texCoord": 1}}
        }}
      }}
    }},
    "occlusionTexture": {{"index": 0, "texCoord": 1, "strength": 0.75}},
    "doubleSided": true
  }}],
  "meshes": [{{
    "primitives": [{{
      "attributes": {{"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2, "TEXCOORD_1": 3}},
      "indices": 4,
      "material": 0
    }}]
  }}],
  "nodes": [{{"mesh": 0}}],
  "scenes": [{{"nodes": [0]}}],
  "scene": 0
}}`,
		b64,
		size_of(Bin),
		PNG_1X1,
	)

	geos: Geometries
	defer geometries_destroy(&geos)
	textures: Textures
	defer textures_destroy(&textures)
	model, err := gltf_parse(transmute([]u8)src, &geos, &textures)
	defer model_destroy(&model)
	testing.expect_value(t, err, Gltf_Error.None)
	verts, _, ok := geometry_mesh(&geos, model.primitives[0].geometry)
	testing.expect(t, ok && len(verts) == 4)
	testing.expect(t, linalg.length(verts[2].position - Vec3{1, 1, 0}) < 1e-4, "quantized i16 position")
	testing.expect(t, linalg.length(verts[2].normal - Vec3{0, 0, 1}) < 1e-2, "quantized normalized i8 normal")
	testing.expect(t, linalg.length(verts[2].uv - Vec2{1, 1}) < 1e-4, "quantized normalized u16 uv0 flipped")
	testing.expect(t, linalg.length(verts[2].uv1 - Vec2{1, 0}) < 1e-4, "uv1 flipped")

	mat := model.primitives[0].material
	testing.expect(t, .Alpha_Mask in mat.flags && abs(mat.alpha_cutoff - 0.65) < 1e-5, "MASK mode and alphaCutoff")
	testing.expect(t, abs(mat.emissive_intensity - 3.5) < 1e-5, "KHR_materials_emissive_strength")
	testing.expect(t, mat.ao_map != 0 && abs(mat.ao_map_intensity - 0.75) < 1e-5 && mat.ao_transform.tex_coord == 1, "occlusionTexture on UV1")
	testing.expect(t, mat.color_transform.offset == {0.25, 0.5} && mat.color_transform.scale == {2, 3} && mat.color_transform.tex_coord == 1, "KHR_texture_transform")
}

// COLOR_0 (normalized u8 with alpha, float without; linear in the file)
// into the vertices' sRGB colours, its material reading them; a primitive
// without it does not.
@(test)
test_gltf_vertex_colors :: proc(t: ^testing.T) {
	Bin :: struct #packed {
		pos:  [3][3]f32,
		col8: [3][4]u8,
		colf: [3][3]f32,
	}
	b := Bin {
		pos  = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}},
		col8 = {{255, 0, 0, 128}, {0, 255, 0, 255}, {0, 0, 255, 0}},
		colf = {{0.2158605, 0.2158605, 0.2158605}, {1, 1, 1}, {0, 0, 0}}, // sRGB 128 / 255
	}
	b64, _ := base64.encode(([^]u8)(&b)[:size_of(Bin)], allocator = context.temp_allocator)
	src := fmt.tprintf(
		`{{
  "asset": {{"version": "2.0"}},
  "buffers": [{{"uri": "data:application/octet-stream;base64,%s", "byteLength": %d}}],
  "bufferViews": [
    {{"buffer": 0, "byteOffset": 0, "byteLength": 36}},
    {{"buffer": 0, "byteOffset": 36, "byteLength": 12}},
    {{"buffer": 0, "byteOffset": 48, "byteLength": 36}}
  ],
  "accessors": [
    {{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "min": [0, 0, 0], "max": [1, 1, 0]}},
    {{"bufferView": 1, "componentType": 5121, "count": 3, "type": "VEC4", "normalized": true}},
    {{"bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC3"}}
  ],
  "meshes": [{{"primitives": [
    {{"attributes": {{"POSITION": 0, "COLOR_0": 1}}}},
    {{"attributes": {{"POSITION": 0, "COLOR_0": 2}}}},
    {{"attributes": {{"POSITION": 0}}}}
  ]}}],
  "nodes": [{{"mesh": 0}}],
  "scenes": [{{"nodes": [0]}}],
  "scene": 0
}}`,
		b64,
		size_of(Bin),
	)
	geos: Geometries
	defer geometries_destroy(&geos)
	model, err := gltf_parse(transmute([]u8)src, &geos)
	defer model_destroy(&model)
	testing.expect_value(t, err, Gltf_Error.None)
	testing.expect_value(t, len(model.primitives), 3)
	v8, _, _ := geometry_mesh(&geos, model.primitives[0].geometry)
	testing.expect(t, v8[0].color == {255, 0, 0, 128} && v8[1].color == {0, 255, 0, 255} && v8[2].color == {0, 0, 255, 0}, "normalized u8 with alpha")
	vf, _, _ := geometry_mesh(&geos, model.primitives[1].geometry)
	testing.expectf(t, vf[0].color == {128, 128, 128, 255} && vf[1].color == {255, 255, 255, 255}, "linear floats as sRGB, alpha 1: %v", vf[0].color)
	testing.expect(t, .Vertex_Colors in model.primitives[0].material.flags && .Vertex_Colors in model.primitives[1].material.flags, "read by their materials")
	testing.expect(t, .Vertex_Colors not_in model.primitives[2].material.flags, "not without COLOR_0")
}

@(test)
test_gltf_unsupported_extensions :: proc(t: ^testing.T) {
	geos: Geometries
	defer geometries_destroy(&geos)

	for ext in ([]string{"KHR_draco_mesh_compression", "EXT_meshopt_compression", "KHR_texture_basisu"}) {
		src := fmt.tprintf(`{{"asset":{{"version":"2.0"}},"extensionsUsed":[%q],"extensionsRequired":[%q]}}`, ext, ext)
		m, err := gltf_parse(transmute([]u8)src, &geos)
		defer model_destroy(&m)
		testing.expect_value(t, err, Gltf_Error.Unsupported_Extension)
		testing.expect_value(t, m.unsupported_extension, ext)
		testing.expect(t, GLTF_ERROR_TEXT[err] != "")
	}

	// Optional extensions in extensionsUsed (not in extensionsRequired) are ignored.
	opt := `{"asset":{"version":"2.0"},"extensionsUsed":["KHR_materials_ior","KHR_lights_punctual","KHR_materials_transmission"],"scenes":[{"nodes":[]}],"scene":0}`
	m_opt, err_opt := gltf_parse(transmute([]u8)opt, &geos)
	defer model_destroy(&m_opt)
	testing.expect_value(t, err_opt, Gltf_Error.None)
}

// KHR_materials_unlit sets Material_Flag.Basic, and KHR_lights_punctual imports
// directional, point, and spot lights on nodes and appends them to scene.lights
// transformed by the node's world matrix (aiming along local -Z).
@(test)
test_gltf_unlit_and_lights_punctual :: proc(t: ^testing.T) {
	geos: Geometries
	defer geometries_destroy(&geos)
	scene: Scene
	defer scene_destroy(&scene)

	Bin :: struct #packed {
		pos: [3][3]f32,
	}
	b := Bin{pos = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}}
	encoded := base64.encode(([^]u8)(&b)[:size_of(Bin)], allocator = context.temp_allocator)
	src := fmt.tprintf(
		`{{"asset":{{"version":"2.0"}},"extensionsUsed":["KHR_materials_unlit","KHR_lights_punctual"],"extensionsRequired":["KHR_materials_unlit","KHR_lights_punctual"],"extensions":{{"KHR_lights_punctual":{{"lights":[{{"type":"directional","color":[1,0.5,0.25],"intensity":2.5}},{{"type":"point","color":[0.2,0.8,1],"intensity":10,"range":6}},{{"type":"spot","color":[1,1,1],"intensity":20,"range":12,"spot":{{"innerConeAngle":0.2,"outerConeAngle":0.5}}}}]}}}},"buffers":[{{"uri":"data:application/octet-stream;base64,%s","byteLength":%d}}],"bufferViews":[{{"buffer":0,"byteOffset":0,"byteLength":36}}],"accessors":[{{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3"}}],"materials":[{{"pbrMetallicRoughness":{{"baseColorFactor":[0.8,0.2,0.4,1]}},"extensions":{{"KHR_materials_unlit":{{}}}}}}],"meshes":[{{"primitives":[{{"attributes":{{"POSITION":0}},"material":0}}]}}],"nodes":[{{"mesh":0}},{{"translation":[1,2,3],"extensions":{{"KHR_lights_punctual":{{"light":0}}}}}},{{"translation":[-2,1,0],"extensions":{{"KHR_lights_punctual":{{"light":1}}}}}},{{"translation":[0,4,2],"extensions":{{"KHR_lights_punctual":{{"light":2}}}}}}],"scenes":[{{"nodes":[0,1,2,3]}}],"scene":0}}`,
		encoded,
		size_of(Bin),
	)
	m, err := gltf_parse(transmute([]u8)src, &geos)
	defer model_destroy(&m)
	testing.expect_value(t, err, Gltf_Error.None)
	testing.expect_value(t, len(m.lights), 3)

	scene_clear(&scene)
	gltf_draw(&scene, &m, transform = compose({10, 0, 0}))
	testing.expect_value(t, len(scene.draws), 1)
	testing.expect(t, .Basic in scene.draws.material[0].flags, "KHR_materials_unlit sets .Basic")
	testing.expect_value(t, len(scene.lights), 3)

	// Directional light at (11, 2, 3) aiming along -Z -> target (11, 2, 2).
	l0 := scene.lights[0]
	testing.expect_value(t, l0.kind, Light_Kind.Directional)
	testing.expect(t, linalg.length(l0.position - Vec3{11, 2, 3}) < 1e-4)
	testing.expect(t, linalg.length(l0.target - Vec3{11, 2, 2}) < 1e-4)
	testing.expect_value(t, l0.intensity, f32(2.5))

	// Point light at (8, 1, 0) with distance = 6.
	l1 := scene.lights[1]
	testing.expect_value(t, l1.kind, Light_Kind.Point)
	testing.expect(t, linalg.length(l1.position - Vec3{8, 1, 0}) < 1e-4)
	testing.expect_value(t, l1.distance, f32(6))

	// Spot light at (10, 4, 2) aiming at (10, 4, 1) with outer angle = 0.5, inner = 0.2 -> penumbra = (0.5 - 0.2) / 0.5 = 0.6.
	l2 := scene.lights[2]
	testing.expect_value(t, l2.kind, Light_Kind.Spot)
	testing.expect(t, linalg.length(l2.position - Vec3{10, 4, 2}) < 1e-4)
	testing.expect(t, linalg.length(l2.target - Vec3{10, 4, 1}) < 1e-4)
	testing.expect(t, abs(l2.angle - 0.5) < 1e-5)
	testing.expect(t, abs(l2.penumbra - 0.6) < 1e-5)
}

@(private = "file")
Test_Zip_Entry :: struct {
	name: string,
	data: string,
}

@(private = "file")
put_u16_le :: proc(buf: ^[dynamic]u8, v: u16) {
	append(buf, u8(v), u8(v >> 8))
}

@(private = "file")
put_u32_le :: proc(buf: ^[dynamic]u8, v: u32) {
	append(buf, u8(v), u8(v >> 8), u8(v >> 16), u8(v >> 24))
}

// Builds a minimal Stored (method 0) ZIP archive in memory for testing `.3mf`.
@(private = "file")
make_test_zip :: proc(entries: []Test_Zip_Entry, allocator := context.temp_allocator) -> []u8 {
	buf := make([dynamic]u8, 0, 1024, allocator)
	offsets := make([]u32, len(entries), context.temp_allocator)
	for e, i in entries {
		offsets[i] = u32(len(buf))
		put_u32_le(&buf, 0x04034b50) // LFH signature
		put_u16_le(&buf, 20)         // version needed
		put_u16_le(&buf, 0)          // flags
		put_u16_le(&buf, 0)          // method = 0 (stored)
		put_u16_le(&buf, 0)          // mod time
		put_u16_le(&buf, 0)          // mod date
		put_u32_le(&buf, 0)          // crc32
		put_u32_le(&buf, u32(len(e.data)))
		put_u32_le(&buf, u32(len(e.data)))
		put_u16_le(&buf, u16(len(e.name)))
		put_u16_le(&buf, 0)          // extra len
		append(&buf, ..transmute([]u8)e.name)
		append(&buf, ..transmute([]u8)e.data)
	}
	cd_offset := u32(len(buf))
	for e, i in entries {
		put_u32_le(&buf, 0x02014b50) // CDH signature
		put_u16_le(&buf, 20)         // version made by
		put_u16_le(&buf, 20)         // version needed
		put_u16_le(&buf, 0)          // flags
		put_u16_le(&buf, 0)          // method = 0 (stored)
		put_u16_le(&buf, 0)          // mod time
		put_u16_le(&buf, 0)          // mod date
		put_u32_le(&buf, 0)          // crc32
		put_u32_le(&buf, u32(len(e.data)))
		put_u32_le(&buf, u32(len(e.data)))
		put_u16_le(&buf, u16(len(e.name)))
		put_u16_le(&buf, 0)          // extra len
		put_u16_le(&buf, 0)          // comment len
		put_u16_le(&buf, 0)          // disk start
		put_u16_le(&buf, 0)          // internal attr
		put_u32_le(&buf, 0)          // external attr
		put_u32_le(&buf, offsets[i])
		append(&buf, ..transmute([]u8)e.name)
	}
	cd_size := u32(len(buf)) - cd_offset
	put_u32_le(&buf, 0x06054b50) // EOCD signature
	put_u16_le(&buf, 0)
	put_u16_le(&buf, 0)
	put_u16_le(&buf, u16(len(entries)))
	put_u16_le(&buf, u16(len(entries)))
	put_u32_le(&buf, cd_size)
	put_u32_le(&buf, cd_offset)
	put_u16_le(&buf, 0)
	return buf[:]
}

@(test)
test_threemf_parse_and_draw :: proc(t: ^testing.T) {
	geos: Geometries
	defer geometries_destroy(&geos)
	scene: Scene
	defer scene_destroy(&scene)

	rels_xml := `<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
	<Relationship Target="/3D/root.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>`

	root_xml := `<?xml version="1.0" encoding="UTF-8"?>
<model unit="meter" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
       xmlns:m="http://schemas.microsoft.com/3dmanufacturing/material/2015/02"
       xmlns:p="http://schemas.microsoft.com/3dmanufacturing/production/2015/06"
       requiredextensions="m p">
	<resources>
		<m:pbmetallicdisplayproperties id="10">
			<m:pbmetallic name="Gold" metallicness="0.85" roughness="0.15"/>
		</m:pbmetallicdisplayproperties>
		<basematerials id="1" displaypropertiesid="10">
			<base name="Gold" displaycolor="#FFCC00"/>
		</basematerials>
		<m:colorgroup id="2">
			<m:color color="#FF0000FF"/>
			<m:color color="#00FF0080"/>
		</m:colorgroup>
		<object id="3" name="DualTri" pid="1" pindex="0" type="model">
			<mesh>
				<vertices>
					<vertex x="0" y="0" z="0"/>
					<vertex x="1" y="0" z="0"/>
					<vertex x="0" y="0" z="2"/>
					<vertex x="0" y="2" z="0"/>
				</vertices>
				<triangles>
					<triangle v1="0" v2="1" v3="2"/>
					<triangle v1="0" v2="3" v3="1" pid="2" p1="1"/>
				</triangles>
			</mesh>
		</object>
		<object id="4" name="Assembly" type="model">
			<components>
				<component objectid="3" transform="1 0 0 0 1 0 0 0 1 2 4 6"/>
				<component p:path="/3D/Parts/sub.model" objectid="1"/>
			</components>
		</object>
	</resources>
	<build>
		<item objectid="4"/>
	</build>
</model>`

	// Sub-part in centimeters (0.01 scale relative to root's meter unit).
	sub_xml := `<?xml version="1.0" encoding="UTF-8"?>
<model unit="centimeter" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">
	<resources>
		<object id="1" name="SubTri" type="model">
			<mesh>
				<vertices>
					<vertex x="0" y="0" z="0"/>
					<vertex x="100" y="0" z="0"/>
					<vertex x="0" y="0" z="100"/>
				</vertices>
				<triangles>
					<triangle v1="0" v2="1" v3="2"/>
				</triangles>
			</mesh>
		</object>
	</resources>
	<build/>
</model>`

	zip_bytes := make_test_zip({
		{name = "_rels/.rels", data = rels_xml},
		{name = "3D/root.model", data = root_xml},
		{name = "3D/Parts/sub.model", data = sub_xml},
	})

	m, err := threemf_parse(zip_bytes, &geos)
	defer model_destroy(&m)
	testing.expect_value(t, err, Threemf_Error.None)
	testing.expect_value(t, len(m.meshes), 2)
	testing.expect(t, model_node(&m, "DualTri") != 0 && model_node(&m, "SubTri") != 0, "3MF object names")
	// DualTri has 2 triangles with different materials -> split into 2 primitives.
	testing.expect_value(t, m.meshes[0].primitive_count, u32(2))
	testing.expect_value(t, m.meshes[1].primitive_count, u32(1))

	// First primitive uses basematerials id=1 + pbmetallicdisplayproperties id=10.
	p0 := m.primitives[0]
	testing.expect(t, abs(p0.material.metalness - 0.85) < 1e-5)
	testing.expect(t, abs(p0.material.roughness - 0.15) < 1e-5)
	testing.expect(t, abs(p0.material.color.r - 1.0) < 1e-4)
	testing.expect(t, abs(p0.material.color.g - 0xcc / 255.0) < 1e-4)

	// The <base> is a file material by its name; the colour group's are not.
	gold := model_material(&m, "Gold")
	testing.expect(t, gold != 0 && len(m.material_names) == 1, "the base material by name")
	testing.expect_value(t, p0.material_id, gold)
	testing.expect_value(t, m.primitives[1].material_id, Model_Material_Id(0))

	// Second primitive uses colorgroup id=2 index=1 (#00FF0080 -> transparent).
	p1 := m.primitives[1]
	testing.expect(t, .Transparent in p1.material.flags)
	testing.expect(t, abs(p1.material.opacity - 128.0 / 255.0) < 1e-4)

	// Z-up (0, 0, 2) in 3MF becomes Y-up (0, 2, 0) in o3d.
	g0, ok0 := geometry_get(&geos, p0.geometry)
	testing.expect(t, ok0)
	v2 := geos.vertices[g0.first_vertex + 2]
	testing.expect(t, linalg.length(v2.position - Vec3{0, 2, 0}) < 1e-5)

	// Sub-part vertex (100, 0, 0) cm is scaled to (1, 0, 0) m.
	p_sub := m.primitives[2]
	g_sub, ok_sub := geometry_get(&geos, p_sub.geometry)
	testing.expect(t, ok_sub)
	v_sub1 := geos.vertices[g_sub.first_vertex + 1]
	testing.expect(t, linalg.length(v_sub1.position - Vec3{1, 0, 0}) < 1e-5)

	// Component translation (2, 4, 6) in 3MF Z-up becomes (2, 6, -4) in o3d Y-up.
	scene_clear(&scene)
	gltf_draw(&scene, &m)
	testing.expect_value(t, len(scene.draws), 3)
	d0_xf := scene.draws.model[0]
	testing.expect(t, abs(d0_xf[0, 3] - 2) < 1e-5)
	testing.expect(t, abs(d0_xf[1, 3] - 6) < 1e-5)
	testing.expect(t, abs(d0_xf[2, 3] - (-4)) < 1e-5)
}

@(test)
test_threemf_errors :: proc(t: ^testing.T) {
	geos: Geometries
	defer geometries_destroy(&geos)

	// Broken ZIP bytes.
	m_bad, err_bad := threemf_parse(transmute([]u8)string("not a zip encoded archive"), &geos)
	defer model_destroy(&m_bad)
	testing.expect_value(t, err_bad, Threemf_Error.Not_Zip)

	// Unsupported required extension (e.g. Slice extension).
	slice_xml := `<?xml version="1.0" encoding="UTF-8"?>
<model unit="millimeter" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
       xmlns:s="http://schemas.microsoft.com/3dmanufacturing/slice/2015/07"
       requiredextensions="s">
	<resources/>
	<build/>
</model>`
	zip_slice := make_test_zip({{name = "3D/3dmodel.model", data = slice_xml}})
	m_ext, err_ext := threemf_parse(zip_slice, &geos)
	defer model_destroy(&m_ext)
	testing.expect_value(t, err_ext, Threemf_Error.Unsupported_Extension)
	testing.expect_value(t, m_ext.unsupported_extension, "http://schemas.microsoft.com/3dmanufacturing/slice/2015/07")
	testing.expect(t, THREEMF_ERROR_TEXT[err_ext] != "")
}

// Verifies 3MF crease-angle normal smoothing: faces meeting at < 60° share
// smooth averaged normals along their shared edge, while faces meeting at 90°
// (>= 60°) keep crisp hard-edge normals.
@(test)
test_threemf_crease_normals :: proc(t: ^testing.T) {
	geos: Geometries
	defer geometries_destroy(&geos)

	model_xml := `<?xml version="1.0" encoding="UTF-8"?>
<model unit="meter" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">
	<resources>
		<object id="1" type="model">
			<mesh>
				<vertices>
					<vertex x="0" y="0" z="0"/>
					<vertex x="0" y="1" z="0"/>
					<vertex x="-1" y="0" z="0"/>
					<vertex x="1" y="0" z="-0.3"/>
					<vertex x="0" y="0" z="-1"/>
				</vertices>
				<triangles>
					<triangle v1="0" v2="1" v3="2"/>
					<triangle v1="0" v2="3" v3="1"/>
					<triangle v1="0" v2="4" v3="1"/>
				</triangles>
			</mesh>
		</object>
	</resources>
	<build>
		<item objectid="1"/>
	</build>
</model>`
	zip_bytes := make_test_zip({{name = "3D/3dmodel.model", data = model_xml}})
	m, err := threemf_parse(zip_bytes, &geos)
	defer model_destroy(&m)
	testing.expect_value(t, err, Threemf_Error.None)
	testing.expect_value(t, len(m.primitives), 1)

	g, ok := geometry_get(&geos, m.primitives[0].geometry)
	testing.expect(t, ok && g.vertex_count == 9)
	verts := geos.vertices[g.first_vertex:][:g.vertex_count]

	// Tri 0 (v0, v1, v2) has normal (0, 1, 0) in o3d; Tri 1 (v0, v3, v1) has normal ~ normalize(0.3, 1, 0)
	// (~16.7° crease < 60°), so vertex 0 in Tri 0 (verts[0]) and Tri 1 (verts[3]) share an averaged normal.
	testing.expect(t, verts[0].normal.y > 0.9 && verts[0].normal.x > 0.05, "gentle crease is smoothed across shared edge")
	testing.expect(t, linalg.length(verts[0].normal - verts[3].normal) < 1e-4, "shared gentle-crease vertices get identical averaged normal")

	// Tri 2 (v0, v4, v1) has normal (1, 0, 0) in o3d (90° from Tri 0, >= 60° crease),
	// so Tri 2's normal is NOT blended into Tri 0's vertex 0 normal, and Tri 2 keeps its sharp (1, 0, 0) normal.
	testing.expect(t, linalg.length(verts[6].normal - Vec3{1, 0, 0}) < 1e-4, "sharp 90-deg crease keeps crisp hard-edge normal")
}

// Verifies Wavefront .obj + .mtl parsing, N-gon fan triangulation, negative
// indices, per-vertex colors, crease-angle smooth normals, and `gltf_draw`
// with `.Toon` and `outline_width`.
@(test)
test_obj_parse_and_mtl :: proc(t: ^testing.T) {
	geos: Geometries
	defer geometries_destroy(&geos)
	textures: Textures
	defer textures_destroy(&textures)
	scene: Scene
	defer scene_destroy(&scene)

	obj_src := `# Test Wavefront OBJ with quad face, negative indices, and vertex colors
o MascotBody
v -1.0  0.0  0.0 1.0 0.2 0.2
v  1.0  0.0  0.0 1.0 0.2 0.2
v  1.0  1.0  0.0 0.2 1.0 0.2
v -1.0  1.0  0.0 0.2 1.0 0.2
vt 0.0 0.0
vt 1.0 0.0
vt 1.0 1.0
vt 0.0 1.0
f 1/1 2/2 3/3 4/4

o MascotCap
v  0.0  1.0  0.0
v  1.0  1.0  0.0
v  0.5  1.8  0.0
vn 0.0  0.0  1.0
f -3//1 -2//1 -1//1
`
	m, err := obj_parse(transmute([]u8)obj_src, &geos, &textures)
	defer model_destroy(&m)
	testing.expect_value(t, err, Obj_Error.None)
	testing.expect_value(t, len(m.nodes), 2)
	testing.expect_value(t, len(m.primitives), 2)
	testing.expect_value(t, model_node(&m, "MascotCap"), Model_Node_Id(2))

	// First primitive: quad fan-triangulated into 2 triangles (6 vertices), with 2-color palette texture.
	g0, ok0 := geometry_get(&geos, m.primitives[0].geometry)
	testing.expect(t, ok0 && g0.vertex_count == 6 && g0.index_count == 6)
	testing.expect(t, m.primitives[0].material.color_map != 0, "multi-color vertex colors baked into palette texture")
	v0 := geos.vertices[g0.first_vertex]
	testing.expect(t, linalg.length(v0.normal - Vec3{0, 0, 1}) < 1e-4, "auto-computed face/smooth normal points along +Z")

	// Second primitive: triangle using negative indices and explicit vn.
	g1, ok1 := geometry_get(&geos, m.primitives[1].geometry)
	testing.expect(t, ok1 && g1.vertex_count == 3)

	// Draw with .Toon and outline_width: expand_draws emits 2 front draws + 2 back-face outline draws = 4 draws.
	scene_clear(&scene)
	gltf_draw(&scene, &m, flags = {.Toon}, outline_width = 0.03, outline_color = {0.05, 0.03, 0.02})
	testing.expect_value(t, len(scene.draws), 2)
	testing.expect(t, .Toon in scene.draws.material[0].flags)
	testing.expect(t, abs(scene.draws.material[0].outline_width - 0.03) < 1e-6)
}


// A 2x2 RGBA PNG (red, green / blue, half-transparent white), and whether
// texture `id`'s top level holds exactly its pixels.
@(private = "file")
TEST_PIXELS :: [16]u8{255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 128}

@(private = "file")
test_png :: proc() -> []u8 {
	pixels := TEST_PIXELS
	png, _ := image.encode_png(pixels[:], 2, 2, context.temp_allocator)
	return png
}

@(private = "file")
holds_test_pixels :: proc(textures: ^Textures, id: Texture_Id) -> bool {
	tex := texture_get(textures, id) or_return
	pixels := TEST_PIXELS
	return tex.width == 2 && tex.height == 2 && mem.compare(mem.slice_to_bytes(textures.texels[tex.mips[0]:][:4]), pixels[:]) == 0
}

// A 3MF's `<m:texture2d>` is decoded from its archive entry into a material's
// colour map.
@(test)
test_threemf_texture2d :: proc(t: ^testing.T) {
	geos: Geometries
	defer geometries_destroy(&geos)
	textures: Textures
	defer textures_destroy(&textures)

	rels_xml := `<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
	<Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>`
	model_xml := `<?xml version="1.0" encoding="UTF-8"?>
<model unit="millimeter" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
       xmlns:m="http://schemas.microsoft.com/3dmanufacturing/material/2015/02" requiredextensions="m">
	<resources>
		<m:texture2d id="1" path="/3D/Textures/a.png" contenttype="image/png"/>
		<m:texture2d id="2" path="/3D/Textures/bad.png" contenttype="image/png"/>
		<m:texture2dgroup id="3" texid="1">
			<m:tex2coord u="0" v="0"/>
			<m:tex2coord u="1" v="0"/>
			<m:tex2coord u="0" v="1"/>
		</m:texture2dgroup>
		<object id="4" type="model" pid="3" pindex="0">
			<mesh>
				<vertices>
					<vertex x="0" y="0" z="0"/>
					<vertex x="1" y="0" z="0"/>
					<vertex x="0" y="1" z="0"/>
				</vertices>
				<triangles>
					<triangle v1="0" v2="1" v3="2" p1="0" p2="1" p3="2"/>
				</triangles>
			</mesh>
		</object>
	</resources>
	<build>
		<item objectid="4"/>
	</build>
</model>`
	zip_bytes := make_test_zip({
		{name = "_rels/.rels", data = rels_xml},
		{name = "3D/3dmodel.model", data = model_xml},
		{name = "3D/Textures/a.png", data = string(test_png())},
		{name = "3D/Textures/bad.png", data = "\x89PNG\r\n\x1a\nnot a png"},
	})
	m, err := threemf_parse(zip_bytes, &geos, &textures)
	defer model_destroy(&m)
	testing.expect_value(t, err, Threemf_Error.None)
	testing.expect_value(t, len(textures.list), 1)
	testing.expect_value(t, len(m.primitives), 1)
	testing.expect(t, holds_test_pixels(&textures, m.primitives[0].material.color_map), "the texture2d's pixels")
}

// An MTL's `map_Kd` is loaded from a file next to the OBJ.
@(test)
test_obj_map_kd :: proc(t: ^testing.T) {
	geos: Geometries
	defer geometries_destroy(&geos)
	textures: Textures
	defer textures_destroy(&textures)

	dir, derr := os.make_directory_temp("", "o3d_obj_*", context.temp_allocator)
	testing.expect_value(t, derr, nil)
	defer os.remove_all(dir)
	write :: proc(dir, name: string, data: []u8) -> bool {
		return os.write_entire_file(fmt.tprintf("%s/%s", dir, name), data) == nil
	}
	testing.expect(t, write(dir, "a.png", test_png()))
	testing.expect(t, write(dir, "a.mtl", transmute([]u8)string("newmtl lit\nKd 1 1 1\nmap_Kd -bm 1.0 a.png\nnewmtl missing\nmap_Kd none.png\n")))
	obj_src := `mtllib a.mtl
v 0 0 0
v 1 0 0
v 0 1 0
vt 0 0
vt 1 0
vt 0 1
usemtl lit
f 1/1 2/2 3/3
usemtl missing
f 3/3 2/2 1/1
`
	m, err := obj_parse(transmute([]u8)obj_src, &geos, &textures, dir)
	defer model_destroy(&m)
	testing.expect_value(t, err, Obj_Error.None)
	testing.expect_value(t, len(m.primitives), 2)
	testing.expect_value(t, len(textures.list), 1)
	testing.expect(t, holds_test_pixels(&textures, m.primitives[0].material.color_map), "map_Kd's pixels")
	testing.expect_value(t, m.primitives[1].material.color_map, Texture_Id(0))
	testing.expect_value(t, m.primitives[1].material_id, model_material(&m, "missing"))
	testing.expect_value(t, model_material(&m, "lit"), Model_Material_Id(1))
}

// model_node_bounds: a node's box holds its meshes' and its children's in
// the model's space; a node with nothing under it is empty; model_bounds
// is the roots' together.
@(test)
test_model_node_bounds :: proc(t: ^testing.T) {
	geos: Geometries
	defer geometries_destroy(&geos)
	m: Model
	defer model_destroy(&m)
	unit := box(&geos) // -0.5 .. 0.5
	append(&m.primitives, Model_Primitive{geometry = unit})
	append(&m.meshes, Model_Mesh{first_primitive = 0, primitive_count = 1})
	node :: proc(m: ^Model, parent, mesh: i32, local: Mat4) {
		append(&m.nodes, Model_Node{parent = parent, mesh = mesh, skin = -1, light = -1, local = local})
	}
	// 0: a root moved 10 along x holding 1 (its box, up 2, twice the size)
	// and 2 (nothing); 3: another root, a box at the origin.
	node(&m, -1, -1, linalg.matrix4_translate_f32({10, 0, 0}))
	node(&m, 0, 0, linalg.matrix4_translate_f32({0, 2, 0}) * linalg.matrix4_scale_f32({2, 2, 2}))
	node(&m, 0, -1, linalg.MATRIX4F32_IDENTITY)
	node(&m, -1, 0, linalg.MATRIX4F32_IDENTITY)
	boxes: [4][2]Vec3
	model_node_bounds(&m, &geos, boxes[:])
	near :: proc(a, b: Vec3) -> bool {
		return linalg.length(a - b) < 1e-5
	}
	testing.expectf(t, near(boxes[1][0], {9, 1, -1}) && near(boxes[1][1], {11, 3, 1}), "the child's own box: %v", boxes[1])
	testing.expectf(t, boxes[0] == boxes[1], "the root holds its child's: %v", boxes[0])
	testing.expectf(t, boxes[2][0].x > boxes[2][1].x, "nothing under it: empty, %v", boxes[2])
	testing.expectf(t, near(boxes[3][0], {-0.5, -0.5, -0.5}) && near(boxes[3][1], {0.5, 0.5, 0.5}), "the other root: %v", boxes[3])
	lo, hi, ok := model_bounds(&m, &geos)
	testing.expectf(t, ok && near(lo, {-0.5, -0.5, -1}) && near(hi, {11, 3, 1}), "the roots together: %v .. %v", lo, hi)
}
