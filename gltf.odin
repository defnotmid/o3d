package o3d

import "core:encoding/base64"
import "core:fmt"
import "core:math"
import "core:math/linalg"
import "core:os"
import "core:slice"
import "core:strings"

// glTF 2.0 / GLB import: `gltf_read` (gltf_read.odin) reads the file, and
// this builds meshes, metallic-roughness PBR materials and textures, a
// topologically ordered node hierarchy, and keyframe animations (`STEP`,
// `LINEAR`, `CUBICSPLINE`) from it, stored in flat arrays.
// `gltf_draw` evaluates the hierarchy (and optional animation at `time`) in a
// single forward pass and appends draws to `Scene` with zero heap allocations.
//
// As three.js's GLTFLoader: the default scene's nodes (every root when the
// file names none); baseColor and emissive factors converted to o3d's sRGB;
// the metallicRoughness texture as both roughness (green) and metalness
// (blue) map; occlusionTexture as ao_map (red channel and strength, applied
// to indirect light); TEXCOORD_0 and TEXCOORD_1 mapped per texture slot with
// KHR_texture_transform; KHR_materials_emissive_strength and
// KHR_materials_unlit; KHR_mesh_quantization integer vertex attributes;
// alphaMode OPAQUE ignores alpha, MASK discards below alphaCutoff (including
// in the shadow pass), BLEND blends; a primitive without normals is flat
// shaded, as the spec asks; morph targets (`POSITION` and `NORMAL` deltas,
// mesh/node default `weights`, `extras.targetNames`, and `"weights"`
// animation channels; `TANGENT` deltas are ignored) and skins (`JOINTS_0`,
// `WEIGHTS_0`, `inverseBindMatrices`) deform meshes in `gltf_draw` (morph
// before skin). Unimplemented extensions in `extensionsRequired` fail with
// `.Unsupported_Extension` (`Model.unsupported_extension`); optional ones in
// `extensionsUsed` are ignored.

// Why a glTF did not load; GLTF_ERROR_TEXT says it in words.
Gltf_Error :: enum u8 {
	None,
	Empty, // no bytes
	File_Not_Found,
	Io,
	Data_Too_Short,
	Unknown_Format,
	Invalid_Json,
	Invalid_Gltf,
	Legacy_Gltf,
	Out_Of_Memory,
	// An external buffer (.bin) the file names.
	Buffer_Not_Found,
	Buffer_Io,
	// Failed validation: an accessor, view or index past its data.
	Invalid,
	// extensionsRequired names an extension o3d does not implement;
	// `Model.unsupported_extension` has its name.
	Unsupported_Extension,
	// A `Max` limit (limits.odin) was passed; `Model.warnings` says which.
	Limit,
}

@(rodata)
GLTF_ERROR_TEXT := [Gltf_Error]string {
	.None                  = "",
	.Empty                 = "no data",
	.File_Not_Found        = "file not found",
	.Io                    = "could not read the file",
	.Data_Too_Short        = "the data ends early",
	.Unknown_Format        = "not glTF or GLB",
	.Invalid_Json          = "invalid JSON",
	.Invalid_Gltf          = "not valid glTF",
	.Legacy_Gltf           = "glTF 1.0 (2.0 is supported)",
	.Out_Of_Memory         = "out of memory",
	.Buffer_Not_Found      = "a buffer file it names was not found",
	.Buffer_Io             = "a buffer file it names could not be read",
	.Invalid               = "an accessor, view or index points past its data",
	.Unsupported_Extension = "requires an unsupported glTF extension",
	.Limit                 = "too large to read (a limit was reached)",
}

Animation_Path :: enum u8 {
	Translation,
	Rotation,
	Scale,
	Weights,
}

Interpolation :: enum u8 {
	Linear,
	Step,
	Cubic_Spline,
}

// Overrides one morph target's weight in `gltf_draw`: by `name` (when
// non-empty, matching `extras.targetNames`) or 0-based target `index`.
Morph_Weight :: struct {
	name:   string,
	index:  int,
	weight: f32,
}

Model_Mesh :: struct {
	first_primitive, primitive_count: u32,
	first_weight, weight_count:       u32,
	first_name, name_count:           u32,
}

Model_Primitive :: struct {
	geometry:    Geometry_Id,
	material:    Material,
	// The file's material it came from (0: none).
	material_id: Model_Material_Id,
	skinned:     bool,
}

// A model's node, by its place in `Model.nodes` plus one (0: none). Ids
// last as long as the model: find them once by name (`model_node`).
Model_Node_Id :: distinct u32

// One of a model's file materials (a glTF material, an MTL `newmtl`), by
// its place in `Model.material_names` plus one (0: none).
Model_Material_Id :: distinct u32

Model_Skin :: struct {
	first_joint, joint_count: u32,
}

Model_Node :: struct {
	name:                       string, // "" if none
	parent:                     i32, // -1 for root nodes; always < current node index
	mesh:                       i32, // -1 if none, else index in `Model.meshes`
	skin:                       i32, // -1 if none, else index in `Model.skins`
	light:                      i32, // -1 if none, else index in `Model.lights`
	first_weight, weight_count: u32, // rest morph weights in `Model.node_weights`
	has_matrix:                 bool,
	translation:                Vec3,
	rotation:                   linalg.Quaternionf32,
	scale:                      Vec3,
	local:                      Mat4,
}

Model_Animation :: struct {
	name:                         string,
	duration:                     f32,
	first_channel, channel_count: u32,
}

Model_Channel :: struct {
	node:                  u32,
	path:                  Animation_Path,
	interpolation:         Interpolation,
	first_time, key_count: u32,
	first_value:           u32,
	weight_count:          u32, // for `.Weights`: morph targets per keyframe
}

Model :: struct {
	meshes:                [dynamic]Model_Mesh,
	primitives:            [dynamic]Model_Primitive,
	lights:                [dynamic]Light,
	nodes:                 #soa[dynamic]Model_Node,
	skins:                 [dynamic]Model_Skin,
	skin_joints:           [dynamic]u32,
	inverse_bind:          [dynamic]Mat4,
	mesh_weights:          [dynamic]f32,
	mesh_target_names:     [dynamic]string,
	target_names:          [dynamic]string, // unique non-empty morph target names in the model (borrowed from `mesh_target_names`)
	node_weights:          [dynamic]f32,
	animations:            [dynamic]Model_Animation,
	channels:              [dynamic]Model_Channel,
	times:                 [dynamic]f32,
	values:                [dynamic]Vec4,
	morph_values:          [dynamic]f32,
	// By Model_Material_Id - 1: each file material's name ("" if none).
	material_names:        [dynamic]string,
	// Set when `gltf_load` or `gltf_parse` returns `.Unsupported_Extension`.
	unsupported_extension: string,
	// What was skipped (limits reached, images that did not decode); with
	// `.Limit`, the limit.
	warnings:              [dynamic]string,
	// The pose `gltf_draw` uses when the caller gives none.
	scratch:               Model_Pose,
}

// Where `gltf_draw` poses a model: per node, the animated transform and the
// world matrix, plus the posed joint matrices for skins and morph target
// weights. A model is only read while it is drawn with a caller's pose, so
// threads may pose one model at once, each with its own (sized on first use,
// reused: no allocation after).
Model_Pose :: struct {
	translation:   [dynamic]Vec3,
	rotation:      [dynamic]linalg.Quaternionf32,
	scale:         [dynamic]Vec3,
	dirty:         [dynamic]bool,
	world:         [dynamic]Mat4,
	joints:        [dynamic]Mat4,
	morph_weights: [dynamic]f32,
	// Per node: its `Model_Parts` row with its ancestors' folded in.
	parts:         [dynamic]Part_State,
}

// What one drawing of a model changes, part by part: tables by durable id
// (`model_parts_init` sizes them to the model; a zero row changes nothing).
// The caller keeps them, like a `Model_Pose`, and sets rows by id.
Model_Parts :: struct {
	// By Model_Node_Id - 1: a node's row holds for the nodes under it too
	// (hidden stays hidden; a nearer row's flags win).
	nodes:        [dynamic]Part_State,
	// By Model_Node_Id - 1: replaces parts of the node's own (animated)
	// transform; the nodes under it follow.
	transforms:   [dynamic]Part_Transform,
	// By Model_Material_Id - 1: for every primitive of that material, after
	// its nodes' rows.
	materials:    [dynamic]Part_State,
	// By Model_Material_Id - 1: the material drawn instead where its
	// state's `replace` is set.
	replacements: [dynamic]Material,
}

Part_State :: struct {
	hidden:      bool,
	// Drawn with (`replacements`; material rows only).
	replace:     bool,
	add, remove: Material_Flags,
}

Transform_Part :: enum u8 {
	Translation,
	Rotation,
	Scale,
}

Part_Transform :: struct {
	set:         bit_set[Transform_Part; u8],
	translation: Vec3,
	rotation:    linalg.Quaternionf32,
	scale:       Vec3,
}

// Sizes `parts` to `m`'s nodes and materials, every row zero.
model_parts_init :: proc(parts: ^Model_Parts, m: ^Model) {
	resize(&parts.nodes, len(m.nodes))
	resize(&parts.transforms, len(m.nodes))
	resize(&parts.materials, len(m.material_names))
	resize(&parts.replacements, len(m.material_names))
	slice.zero(parts.nodes[:])
	slice.zero(parts.transforms[:])
	slice.zero(parts.materials[:])
	slice.zero(parts.replacements[:])
}

model_parts_destroy :: proc(parts: ^Model_Parts) {
	delete(parts.nodes)
	delete(parts.transforms)
	delete(parts.materials)
	delete(parts.replacements)
	parts^ = {}
}

// The first node named `name` (0: none).
model_node :: proc "contextless" (m: ^Model, name: string) -> Model_Node_Id {
	if len(name) == 0 {
		return 0
	}
	for n, i in m.nodes.name[:len(m.nodes)] {
		if n == name {
			return Model_Node_Id(i + 1)
		}
	}
	return 0
}

// The first file material named `name` (0: none).
model_material :: proc "contextless" (m: ^Model, name: string) -> Model_Material_Id {
	if len(name) == 0 {
		return 0
	}
	for n, i in m.material_names {
		if n == name {
			return Model_Material_Id(i + 1)
		}
	}
	return 0
}

// The model's box in its rest pose and its own space (its nodes'
// transforms, no animation, skins and morphs at rest): its size for
// placing and framing. `ok` false: it has no geometry.
model_bounds :: proc(m: ^Model, geos: ^Geometries) -> (lo, hi: Vec3, ok: bool) {
	n := len(m.nodes)
	world := make([]Mat4, n, context.temp_allocator)
	lo, hi = {max(f32), max(f32), max(f32)}, {min(f32), min(f32), min(f32)}
	for i in 0 ..< n {
		parent := m.nodes.parent[i]
		world[i] = world[parent] * m.nodes.local[i] if parent >= 0 else m.nodes.local[i]
		mi := m.nodes.mesh[i]
		if mi < 0 || int(mi) >= len(m.meshes) {
			continue
		}
		mesh := m.meshes[mi]
		for prim in m.primitives[mesh.first_primitive:][:mesh.primitive_count] {
			g := geometry_get(geos, prim.geometry) or_continue
			if g.vertex_count == 0 {
				continue
			}
			for k in 0 ..< 8 {
				corner := Vec3{g.box_max.x if k & 1 != 0 else g.box_min.x, g.box_max.y if k & 2 != 0 else g.box_min.y, g.box_max.z if k & 4 != 0 else g.box_min.z}
				q := (world[i] * Vec4{corner.x, corner.y, corner.z, 1}).xyz
				lo, hi = linalg.min(lo, q), linalg.max(hi, q)
			}
			ok = true
		}
	}
	if !ok {
		return {}, {}, false
	}
	return
}

// Where `node` was in the world at the last `gltf_draw` with `pose` (nil:
// the model's own), to attach other draws to it.
model_node_world :: proc "contextless" (m: ^Model, node: Model_Node_Id, pose: ^Model_Pose = nil) -> Mat4 {
	p := pose if pose != nil else &m.scratch
	if node == 0 || int(node) > len(p.world) {
		return linalg.MATRIX4F32_IDENTITY
	}
	return p.world[node - 1]
}

model_pose_destroy :: proc(p: ^Model_Pose) {
	delete(p.translation)
	delete(p.rotation)
	delete(p.scale)
	delete(p.dirty)
	delete(p.world)
	delete(p.joints)
	delete(p.morph_weights)
	delete(p.parts)
	p^ = {}
}

model_destroy :: proc(m: ^Model) {
	for a in m.animations {
		delete(a.name)
	}
	for s in m.mesh_target_names {
		delete(s)
	}
	for s in m.nodes.name[:len(m.nodes)] {
		delete(s)
	}
	for s in m.material_names {
		delete(s)
	}
	delete(m.material_names)
	delete(m.meshes)
	delete(m.primitives)
	delete(m.lights)
	delete(m.nodes)
	delete(m.skins)
	delete(m.skin_joints)
	delete(m.inverse_bind)
	delete(m.mesh_weights)
	delete(m.mesh_target_names)
	delete(m.target_names)
	delete(m.node_weights)
	delete(m.animations)
	delete(m.channels)
	delete(m.times)
	delete(m.values)
	delete(m.morph_values)
	delete(m.unsupported_extension)
	for w in m.warnings {
		delete(w)
	}
	delete(m.warnings)
	model_pose_destroy(&m.scratch)
	m^ = {}
}


// Loads a `.gltf` or `.glb` file from disk into flat `Model`, `Geometries`,
// and `Textures` tables; `err` says why not (GLTF_ERROR_TEXT). `limits`
// bounds the work (nil: DEFAULT_LIMITS).
gltf_load :: proc(path: string, geos: ^Geometries, textures: ^Textures = nil, limits: ^Limits = nil) -> (m: Model, err: Gltf_Error) {
	bytes, read_err := os.read_entire_file(path, context.allocator)
	if read_err != nil {
		return {}, .Io if os.exists(path) else .File_Not_Found
	}
	defer delete(bytes)
	slash := max(strings.last_index_byte(path, '/'), strings.last_index_byte(path, '\\'))
	return gltf_parse(bytes, geos, textures, path[:slash] if slash >= 0 else "", limits)
}

// Parses a `.gltf` or `.glb` byte slice in memory into flat `Model`,
// `Geometries`, and `Textures` tables; `err` says why not (GLTF_ERROR_TEXT).
// Files it names resolve against `base_dir` (the working directory when
// empty).
gltf_parse :: proc(bytes: []u8, geos: ^Geometries, textures: ^Textures = nil, base_dir := "", limits: ^Limits = nil) -> (m: Model, err: Gltf_Error) {
	doc: Gltf_Document
	doc, err = gltf_read(bytes, base_dir, limits)
	defer gltf_document_destroy(&doc)
	#partial switch err {
	case .None:
	case .Unsupported_Extension:
		return {unsupported_extension = strings.clone(doc.unsupported_extension)}, err
	case .Limit:
		budget_warn(&doc.budget, &m.warnings)
		return {warnings = m.warnings}, err
	case:
		return {}, err
	}
	m = build_model(&doc, geos, textures, base_dir)
	budget_warn(&doc.budget, &m.warnings)
	return m, .None
}

// Finds an animation's index by name (-1 if not found).
gltf_animation :: proc "contextless" (m: ^Model, name: string) -> int {
	for a, i in m.animations {
		if a.name == name {
			return i
		}
	}
	return -1
}

// Finds a morph target's 0-based index by name (`extras.targetNames`, -1 if not found).
gltf_morph_target :: proc "contextless" (m: ^Model, name: string) -> int {
	if len(name) == 0 {
		return -1
	}
	for mesh in m.meshes {
		for tname, ti in m.mesh_target_names[mesh.first_name:][:mesh.name_count] {
			if tname == name {
				return ti
			}
		}
	}
	return -1
}

// Maximum number of morph targets on any mesh in `m` (0 when none).
gltf_morph_count :: proc "contextless" (m: ^Model) -> int {
	n := 0
	for mesh in m.meshes {
		n = max(n, int(mesh.weight_count))
	}
	return n
}

// Evaluates `m`'s node hierarchy (and optional keyframe animation `anim` at
// `time` seconds) under `transform` and appends its mesh primitives to
// `scene.draws`. `flags` (e.g. `{.Cast_Shadow, .Receive_Shadow, .Toon}`) are
// merged into each primitive's material flags; `override_material` replaces
// every primitive's material when non-nil; `morphs` overrides morph target
// weights (by target name or 0-based index) after any animated `"weights"`
// channels; `outline_width` (> 0) and `outline_color` enable inverted-hull
// cartoon outlines on all primitives. `pose` is where it is posed (nil: the
// model's own, so one thread at a time); with the caller's, the model is only
// read. `parts` (`model_parts_init`) changes nodes and materials by id:
// hidden, flags, transforms, replaced materials (a replacement wins over
// `override_material`). Zero allocations on warm frames.
gltf_draw :: proc(scene: ^Scene, m: ^Model, transform: Mat4 = linalg.MATRIX4F32_IDENTITY, anim: int = -1, time: f32 = 0, loop := true, flags: Material_Flags = {}, pose: ^Model_Pose = nil, override_material: ^Material = nil, morphs: []Morph_Weight = nil, outline_width: f32 = 0, outline_color: Vec3 = {}, parts: ^Model_Parts = nil) {
	n := len(m.nodes)
	if n == 0 {
		return
	}
	p := pose if pose != nil else &m.scratch
	if len(p.world) < n {
		resize(&p.translation, n)
		resize(&p.rotation, n)
		resize(&p.scale, n)
		resize(&p.dirty, n)
		resize(&p.world, n)
	}
	if len(p.joints) < len(m.skin_joints) {
		resize(&p.joints, len(m.skin_joints))
	}
	if len(p.morph_weights) < len(m.node_weights) {
		resize(&p.morph_weights, len(m.node_weights))
	}
	node_parts := parts != nil && len(parts.nodes) >= n
	material_parts := parts != nil && len(parts.materials) >= len(m.material_names) && len(parts.replacements) >= len(m.material_names)
	moved := parts != nil && len(parts.transforms) >= n
	if node_parts && len(p.parts) < n {
		resize(&p.parts, n)
	}
	if len(m.node_weights) > 0 {
		copy(p.morph_weights[:len(m.node_weights)], m.node_weights[:])
	}
	has_anim := anim >= 0 && anim < len(m.animations)
	if has_anim || moved {
		copy(p.translation[:n], m.nodes.translation[:n])
		copy(p.rotation[:n], m.nodes.rotation[:n])
		copy(p.scale[:n], m.nodes.scale[:n])
		slice.zero(p.dirty[:n])
	}
	if has_anim {
		a := m.animations[anim]
		t := time
		if a.duration > 0 {
			if loop {
				t = math.mod(t, a.duration)
				if t < 0 {
					t += a.duration
				}
			} else {
				t = clamp(t, 0, a.duration)
			}
		} else {
			t = 0
		}
		for ch in m.channels[a.first_channel:][:a.channel_count] {
			if ch.key_count == 0 || int(ch.node) >= n {
				continue
			}
			if ch.path == .Weights {
				nw := m.nodes.weight_count[ch.node]
				if nw > 0 {
					sample_weights(m, ch, t, p.morph_weights[m.nodes.first_weight[ch.node]:][:nw])
				}
				continue
			}
			v := sample_channel(m, ch, t)
			switch ch.path {
			case .Translation:
				p.translation[ch.node] = v.xyz
			case .Rotation:
				p.rotation[ch.node] = quaternion(x = v.x, y = v.y, z = v.z, w = v.w)
			case .Scale:
				p.scale[ch.node] = v.xyz
			case .Weights:
			}
			p.dirty[ch.node] = true
		}
	}
	if moved {
		for xf, i in parts.transforms[:n] {
			if xf.set == {} {
				continue
			}
			if m.nodes.has_matrix[i] && !p.dirty[i] {
				p.translation[i], p.rotation[i], p.scale[i] = trs_of(m.nodes.local[i])
			}
			if .Translation in xf.set {
				p.translation[i] = xf.translation
			}
			if .Rotation in xf.set {
				p.rotation[i] = xf.rotation
			}
			if .Scale in xf.set {
				p.scale[i] = xf.scale
			}
			p.dirty[i] = true
		}
	}

	parents := m.nodes.parent[:n]
	meshes := m.nodes.mesh[:n]
	skins := m.nodes.skin[:n]
	lights := m.nodes.light[:n]
	locals := m.nodes.local[:n]
	world := p.world[:n]
	for i in 0 ..< n {
		local := locals[i]
		if (has_anim || moved) && p.dirty[i] {
			local = linalg.matrix4_from_trs_f32(p.translation[i], p.rotation[i], p.scale[i])
		}
		parent := parents[i]
		world[i] = (world[parent] * local) if parent >= 0 else (transform * local)
		if node_parts {
			row := parts.nodes[i]
			if parent >= 0 {
				up := p.parts[parent]
				row.hidden ||= up.hidden
				row.add, row.remove = (up.add - row.remove) + row.add, (up.remove - row.add) + row.remove
			}
			p.parts[i] = row
		}
	}
	for i in 0 ..< n {
		if node_parts && p.parts[i].hidden {
			continue
		}
		if li := lights[i]; li >= 0 && int(li) < len(m.lights) {
			l := m.lights[li]
			w := world[i]
			pos := Vec3{w[0, 3], w[1, 3], w[2, 3]}
			// KHR_lights_punctual directional and spot lights shine along local -Z.
			fwd := Vec3{-w[0, 2], -w[1, 2], -w[2, 2]}
			dir := linalg.normalize(fwd) if linalg.length(fwd) > 1e-9 else Vec3{0, 0, -1}
			l.position = pos
			l.target = pos + dir
			append(&scene.lights, l)
		}
		mi := meshes[i]
		if mi < 0 || int(mi) >= len(m.meshes) {
			continue
		}
		var_joints: Joint_Range
		if si := skins[i]; si >= 0 && int(si) < len(m.skins) {
			skin := m.skins[si]
			jcount := int(skin.joint_count)
			if jcount > 0 {
				inv_mesh := linalg.inverse(world[i])
				out_joints := p.joints[skin.first_joint:][:jcount]
				skin_nodes := m.skin_joints[skin.first_joint:][:jcount]
				ibms := m.inverse_bind[skin.first_joint:][:jcount]
				for j in 0 ..< jcount {
					jn := skin_nodes[j]
					jw := world[jn] if int(jn) < n else linalg.MATRIX4F32_IDENTITY
					out_joints[j] = inv_mesh * jw * ibms[j]
				}
				var_joints = scene_joints(scene, out_joints)
			}
		}
		mesh := m.meshes[mi]
		var_morphs: Morph_Range
		if wcount := int(m.nodes.weight_count[i]); wcount > 0 {
			node_w := p.morph_weights[m.nodes.first_weight[i]:][:wcount]
			if len(morphs) > 0 {
				names := m.mesh_target_names[mesh.first_name:][:mesh.name_count]
				for ov in morphs {
					if len(ov.name) > 0 {
						for tname, ti in names {
							if tname == ov.name && ti < wcount {
								node_w[ti] = ov.weight
							}
						}
					} else if ov.index >= 0 && ov.index < wcount {
						node_w[ov.index] = ov.weight
					}
				}
			}
			var_morphs = scene_morphs(scene, node_w)
		}
		for prim in m.primitives[mesh.first_primitive:][:mesh.primitive_count] {
			mat := override_material^ if override_material != nil else prim.material
			row: Part_State
			if material_parts && prim.material_id != 0 {
				row = parts.materials[prim.material_id - 1]
				if row.hidden {
					continue
				}
				if row.replace {
					mat = parts.replacements[prim.material_id - 1]
				}
			}
			mat.flags += flags
			if node_parts {
				mat.flags = (mat.flags - p.parts[i].remove) + p.parts[i].add
			}
			mat.flags = (mat.flags - row.remove) + row.add
			if outline_width > 0 {
				mat.outline_width = outline_width
				mat.outline_color = outline_color
			}
			di := len(scene.draws)
			append(&scene.draws, Draw{prim.geometry, world[i], mat})
			if prim.skinned && var_joints.count > 0 {
				scene_set_draw_joints(scene, di, var_joints)
			}
			if var_morphs.count > 0 {
				scene_set_draw_morphs(scene, di, var_morphs)
			}
		}
	}
}


// A node matrix's translation, rotation and scale (a mirroring matrix
// takes a negative x scale).
@(private = "file")
trs_of :: proc "contextless" (mt: Mat4) -> (t: Vec3, r: linalg.Quaternionf32, s: Vec3) {
	t = {mt[0, 3], mt[1, 3], mt[2, 3]}
	cols := [3]Vec3{{mt[0, 0], mt[1, 0], mt[2, 0]}, {mt[0, 1], mt[1, 1], mt[2, 1]}, {mt[0, 2], mt[1, 2], mt[2, 2]}}
	s = {linalg.length(cols[0]), linalg.length(cols[1]), linalg.length(cols[2])}
	if linalg.dot(linalg.cross(cols[0], cols[1]), cols[2]) < 0 {
		s.x = -s.x
	}
	rot: linalg.Matrix3f32
	for c in 0 ..< 3 {
		axis := cols[c] / s[c] if s[c] != 0 else Vec3{}
		rot[0, c], rot[1, c], rot[2, c] = axis.x, axis.y, axis.z
	}
	r = linalg.quaternion_from_matrix3_f32(rot)
	return
}

// ---- model builder ----------------------------------------------------------------------

@(private = "file")
build_model :: proc(doc: ^Gltf_Document, geos: ^Geometries, textures: ^Textures, base_dir: string) -> (m: Model) {
	// 1. Decode images into `textures` once per glTF image.
	img_ids := make([]Texture_Id, len(doc.images), context.temp_allocator)
	if textures != nil {
		for i in 0 ..< len(doc.images) {
			img_ids[i] = load_gltf_image(doc, textures, i, base_dir, &m.warnings)
		}
	}

	// 2. Import KHR_lights_punctual lights.
	reserve(&m.lights, len(doc.lights))
	for gl in doc.lights {
		kind: Light_Kind = .Point
		switch gl.kind {
		case .Directional:
			kind = .Directional
		case .Point:
			kind = .Point
		case .Spot:
			kind = .Spot
		case .Invalid:
		}
		outer := gl.outer_cone if gl.outer_cone > 0 else math.PI / 4
		inner := clamp(gl.inner_cone, 0, outer)
		penumbra := clamp(1 - inner / outer, 0, 1) if outer > 0 else 0
		append(&m.lights, Light{
			kind      = kind,
			color     = linear_to_srgb_vec3(Vec3(gl.color)),
			intensity = gl.intensity,
			distance  = max(gl.range, 0),
			decay     = 2,
			angle     = outer,
			penumbra  = penumbra,
		})
	}

	// 3. Build meshes and primitives into `geos` and `m.primitives`, and
	// say which extensions the file uses that o3d does not draw.
	reserve(&m.material_names, len(doc.materials))
	for gm in doc.materials {
		append(&m.material_names, strings.clone(gm.name))
	}
	warn_unread_extensions(doc, &m.warnings)
	reserve(&m.meshes, len(doc.meshes))
	scratch := Build_Scratch{
		verts   = make([dynamic]Vertex, context.temp_allocator),
		skin    = make([dynamic]Skin_Vertex, context.temp_allocator),
		morphs  = make([dynamic]Morph_Delta, context.temp_allocator),
		indices = make([dynamic]u32, context.temp_allocator),
		floats  = make([dynamic]f32, context.temp_allocator),
		uints   = make([dynamic]u32, context.temp_allocator),
	}

	for gm in doc.meshes {
		first_prim := u32(len(m.primitives))
		weights := doc.floats[gm.weights.first:][:gm.weights.count]
		names := doc.strings[gm.target_names.first:][:gm.target_names.count]
		max_targets := max(len(weights), len(names))
		for &gp in doc.primitives[gm.primitives.first:][:gm.primitives.count] {
			if gp.mode != 4 {
				continue
			}
			max_targets = max(max_targets, int(gp.targets.count))
			geo_id, has_normals, has_skin, has_color := build_primitive(doc, geos, &gp, &scratch)
			if geo_id == 0 {
				continue
			}
			material := convert_material(doc, gp.material, img_ids)
			if !has_normals {
				material.flags += {.Flat_Shading}
			}
			if has_color {
				material.flags += {.Vertex_Colors}
			}
			append(&m.primitives, Model_Primitive{geometry = geo_id, material = material, material_id = Model_Material_Id(gp.material + 1), skinned = has_skin})
		}
		first_weight := u32(len(m.mesh_weights))
		for ti in 0 ..< max_targets {
			append(&m.mesh_weights, weights[ti] if ti < len(weights) else 0)
		}
		first_name := u32(len(m.mesh_target_names))
		name_count := min(len(names), max_targets)
		for ti in 0 ..< name_count {
			cloned := strings.clone(names[ti]) if len(names[ti]) > 0 else ""
			append(&m.mesh_target_names, cloned)
			if len(cloned) > 0 && !slice.contains(m.target_names[:], cloned) {
				append(&m.target_names, cloned)
			}
		}
		append(&m.meshes, Model_Mesh{
			first_primitive = first_prim,
			primitive_count = u32(len(m.primitives)) - first_prim,
			first_weight    = first_weight,
			weight_count    = u32(max_targets),
			first_name      = first_name,
			name_count      = u32(name_count),
		})
	}

	// 4. Flatten the default scene's node hierarchy (every root when the file
	// names no scene) in topological order (parents always before children).
	node_map := make([]i32, len(doc.nodes), context.temp_allocator)
	slice.fill(node_map, -1)
	scene := doc.scene
	if scene < 0 && len(doc.scenes) > 0 {
		scene = 0
	}
	if scene >= 0 {
		roots := doc.scenes[scene]
		for root in doc.node_lists[roots.first:][:roots.count] {
			emit_node(doc, root, -1, &m, node_map)
		}
	} else {
		for n, i in doc.nodes {
			if n.parent < 0 {
				emit_node(doc, i32(i), -1, &m, node_map)
			}
		}
	}
	for gs in doc.skins {
		for jn in doc.node_lists[gs.joints.first:][:gs.joints.count] {
			emit_node(doc, jn, -1, &m, node_map)
		}
	}

	// 5. Flatten skins (joint node indices and inverse bind matrices).
	build_skins(doc, &m, node_map, &scratch.floats)

	num_nodes := len(m.nodes)
	resize(&m.scratch.translation, num_nodes)
	resize(&m.scratch.rotation, num_nodes)
	resize(&m.scratch.scale, num_nodes)
	resize(&m.scratch.dirty, num_nodes)
	resize(&m.scratch.world, num_nodes)
	resize(&m.scratch.joints, len(m.skin_joints))
	resize(&m.scratch.morph_weights, len(m.node_weights))

	// 6. Flatten animations and keyframe samplers.
	build_animations(doc, &m, node_map, &scratch.floats)
	return m
}

// One warning per extension the file uses that o3d does not implement,
// naming the materials that have it: they are drawn without it.
@(private = "file")
warn_unread_extensions :: proc(doc: ^Gltf_Document, warnings: ^[dynamic]string) {
	unread := make([dynamic]string, context.temp_allocator)
	for ext in doc.extensions_used {
		if !slice.contains(GLTF_EXTENSIONS[:], ext) && !slice.contains(unread[:], ext) {
			append(&unread, ext)
		}
	}
	for gm in doc.materials {
		for ext in doc.strings[gm.unread_extensions.first:][:gm.unread_extensions.count] {
			if !slice.contains(unread[:], ext) {
				append(&unread, ext)
			}
		}
	}
	slice.sort(unread[:])
	for ext in unread {
		b := strings.builder_make(context.temp_allocator)
		fmt.sbprintf(&b, "%s is not supported: drawn without it", ext)
		count := 0
		for gm, mi in doc.materials {
			if !slice.contains(doc.strings[gm.unread_extensions.first:][:gm.unread_extensions.count], ext) {
				continue
			}
			fmt.sbprint(&b, " (materials " if count == 0 else ", ")
			if len(gm.name) > 0 {
				fmt.sbprintf(&b, "%q", gm.name)
			} else {
				fmt.sbprintf(&b, "#%d", mi)
			}
			count += 1
		}
		if count > 0 {
			strings.write_byte(&b, ')')
		}
		append(warnings, strings.clone(strings.to_string(b)))
	}
}

// An image's texture (0: none): from its buffer view, a data: URI, or a
// file beside the glTF.
@(private = "file")
load_gltf_image :: proc(doc: ^Gltf_Document, t: ^Textures, i: int, base_dir: string, warnings: ^[dynamic]string) -> Texture_Id {
	img := doc.images[i]
	data, owned: []u8
	defer delete(owned)
	if img.view >= 0 {
		v := doc.views[img.view]
		if buf := doc.buffers[v.buffer]; buf != nil && v.length > 0 {
			data = buf[v.offset:][:v.length]
		}
	}
	if data == nil && len(img.uri) > 0 {
		if strings.has_prefix(img.uri, "data:") {
			comma := strings.index_byte(img.uri, ',')
			if comma < 0 {
				return 0
			}
			decoded, err := base64.decode(img.uri[comma + 1:], allocator = context.temp_allocator)
			if err != nil {
				return 0
			}
			data = decoded
		} else {
			read_err: os.Error
			data, read_err = os.read_entire_file(gltf_resolve(base_dir, img.uri, context.temp_allocator), context.allocator)
			if read_err != nil {
				return 0
			}
			owned = data
		}
	}
	if data == nil {
		return 0
	}
	if !budget_take(&doc.budget, .Image_Pixels, texture_pixels(data)) {
		return 0
	}
	id, ok := texture_decode(t, data)
	if !ok {
		append(warnings, strings.clone(fmt.tprintf("glTF image %d could not be decoded", i)))
	}
	return id
}

@(private = "file")
Build_Scratch :: struct {
	verts:   [dynamic]Vertex,
	skin:    [dynamic]Skin_Vertex,
	morphs:  [dynamic]Morph_Delta,
	indices: [dynamic]u32,
	floats:  [dynamic]f32,
	uints:   [dynamic]u32,
}

// Accessor `acc` into `out` (resized to `n` and zero past what it has).
@(private = "file")
unpack_floats_to :: proc(doc: ^Gltf_Document, acc: i32, out: ^[dynamic]f32, n: int) {
	table_grow_non_zero(out, n)
	wrote := gltf_accessor_floats(doc, acc, out[:n])
	slice.zero(out[wrote:n])
}

// A triangle primitive into `geos` (0: none), and whether it had normals,
// skinning and vertex colours (COLOR_0, which glTF multiplies the base
// colour by).
@(private = "file")
build_primitive :: proc(doc: ^Gltf_Document, geos: ^Geometries, prim: ^Gltf_Primitive, s: ^Build_Scratch) -> (id: Geometry_Id, has_normals, has_skin, has_color: bool) {
	if prim.position < 0 || doc.accessors[prim.position].count == 0 {
		return 0, false, false, false
	}
	vcount := doc.accessors[prim.position].count
	count_of :: proc(doc: ^Gltf_Document, acc: i32, vcount: int) -> bool {
		return acc >= 0 && doc.accessors[acc].count == vcount
	}
	has_normals = count_of(doc, prim.normal, vcount)
	has_uv := count_of(doc, prim.texcoord[0], vcount)
	has_uv1 := count_of(doc, prim.texcoord[1], vcount)
	has_skin = count_of(doc, prim.joints, vcount) && count_of(doc, prim.weights, vcount)
	color_comps := type_components(doc.accessors[prim.color].type) if prim.color >= 0 else 0
	has_color = count_of(doc, prim.color, vcount) && (color_comps == 3 || color_comps == 4)
	targets := doc.targets[prim.targets.first:][:prim.targets.count]
	icount := doc.accessors[prim.indices].count if prim.indices >= 0 else 0

	// What this primitive unpacks, counted before any of it is (counts come
	// from the file, so in i128).
	per_vertex := 3 + (3 if has_normals else 0) + (2 if has_uv else 0) + (2 if has_uv1 else 0) + (color_comps if has_color else 0) + (8 if has_skin else 0) + 14 * len(targets)
	if !budget_take_wide(&doc.budget, .Gltf_Values, i128(vcount) * i128(per_vertex) + i128(icount)) {
		return 0, false, false, false
	}

	verts, floats := &s.verts, &s.floats
	table_grow(verts, vcount)
	unpack_floats_to(doc, prim.position, floats, vcount * 3)
	for i in 0 ..< vcount {
		verts[i].position = {floats[i * 3], floats[i * 3 + 1], floats[i * 3 + 2]}
		verts[i].normal = {0, 1, 0}
		verts[i].uv = {}
		verts[i].uv1 = {}
		verts[i].color = {}
	}

	if has_normals {
		unpack_floats_to(doc, prim.normal, floats, vcount * 3)
		for i in 0 ..< vcount {
			n := Vec3{floats[i * 3], floats[i * 3 + 1], floats[i * 3 + 2]}
			verts[i].normal = linalg.normalize(n) if linalg.length2(n) > 1e-12 else Vec3{0, 1, 0}
		}
	}

	if has_uv {
		unpack_floats_to(doc, prim.texcoord[0], floats, vcount * 2)
		for i in 0 ..< vcount {
			// glTF TEXCOORD has v = 0 at top; o3d's Vertex.uv has v = 1 at top.
			verts[i].uv = {floats[i * 2], 1 - floats[i * 2 + 1]}
		}
	}

	if has_uv1 {
		unpack_floats_to(doc, prim.texcoord[1], floats, vcount * 2)
		for i in 0 ..< vcount {
			verts[i].uv1 = {floats[i * 2], 1 - floats[i * 2 + 1]}
		}
	}

	if has_color {
		// Linear (glTF's), kept in sRGB as o3d's colours are.
		unpack_floats_to(doc, prim.color, floats, vcount * color_comps)
		for i in 0 ..< vcount {
			c := floats[i * color_comps:][:color_comps]
			verts[i].color = vertex_color({linear_to_srgb(c[0]), linear_to_srgb(c[1]), linear_to_srgb(c[2]), c[3] if color_comps == 4 else 1})
		}
	}

	if has_skin {
		table_grow_non_zero(&s.skin, vcount)
		unpack_floats_to(doc, prim.weights, floats, vcount * 4)
		// Up to 4 joint components per vertex (fewer leave the rest 0).
		jcomps := type_components(doc.accessors[prim.joints].type)
		table_grow_non_zero(&s.uints, vcount * jcomps)
		gltf_accessor_uints(doc, prim.joints, s.uints[:vcount * jcomps])
		for i in 0 ..< vcount {
			j: [4]u32
			if jcomps <= 4 {
				copy(j[:jcomps], s.uints[i * jcomps:][:jcomps])
			}
			w := Vec4{
				max(floats[i * 4 + 0], 0),
				max(floats[i * 4 + 1], 0),
				max(floats[i * 4 + 2], 0),
				max(floats[i * 4 + 3], 0),
			}
			sum := w.x + w.y + w.z + w.w
			w = w / sum if sum > 1e-8 else Vec4{1, 0, 0, 0}
			joints: [4]u16
			for k in 0 ..< 4 {
				joints[k] = u16(min(j[k], u32(MAX_JOINTS - 1))) if w[k] > 0 else 0
			}
			s.skin[i] = {joints, w}
		}
	}

	has_morphs := len(targets) > 0
	if has_morphs {
		morphs := &s.morphs
		clear(morphs)
		table_grow(morphs, len(targets) * vcount)
		for t, ti in targets {
			base := ti * vcount
			if count_of(doc, t.position, vcount) {
				unpack_floats_to(doc, t.position, floats, vcount * 3)
				for i in 0 ..< vcount {
					morphs[base + i].position = {floats[i * 3], floats[i * 3 + 1], floats[i * 3 + 2], 0}
				}
			}
			if count_of(doc, t.normal, vcount) {
				unpack_floats_to(doc, t.normal, floats, vcount * 3)
				for i in 0 ..< vcount {
					morphs[base + i].normal = {floats[i * 3], floats[i * 3 + 1], floats[i * 3 + 2], 0}
				}
			}
		}
	}

	indices := &s.indices
	if prim.indices >= 0 {
		table_grow_non_zero(indices, icount)
		gltf_accessor_uints(doc, prim.indices, indices[:icount])
		// Sparse indices are not checked by gltf_read; a renderer indexes
		// vertices by them unchecked.
		for i in indices^ {
			if int(i) >= vcount {
				return 0, false, false, false
			}
		}
	} else {
		table_grow_non_zero(indices, vcount)
		for i in 0 ..< vcount {
			indices[i] = u32(i)
		}
	}

	// Without NORMAL the primitive is flat shaded (its material says so);
	// smooth normals from the faces still give shadows' normal bias one.
	if !has_normals {
		for &v in verts^ {
			v.normal = {}
		}
		for t := 0; t + 2 < len(indices); t += 3 {
			i0, i1, i2 := indices[t], indices[t + 1], indices[t + 2]
			fn := linalg.cross(verts[i1].position - verts[i0].position, verts[i2].position - verts[i0].position)
			verts[i0].normal += fn
			verts[i1].normal += fn
			verts[i2].normal += fn
		}
		for &v in verts^ {
			v.normal = linalg.normalize(v.normal) if linalg.length2(v.normal) > 1e-12 else Vec3{0, 1, 0}
		}
	}

	return geometry_add(geos, verts[:], indices[:], s.skin[:vcount] if has_skin else nil, s.morphs[:] if has_morphs else nil), has_normals, has_skin, has_color
}

@(private = "file")
tex_from_ref :: proc(doc: ^Gltf_Document, ref: Gltf_Texture_Ref, img_ids: []Texture_Id) -> Texture_Id {
	if ref.texture < 0 {
		return 0
	}
	img := doc.textures[ref.texture]
	return img_ids[img] if img >= 0 else 0
}

@(private = "file")
ref_transform :: proc "contextless" (ref: Gltf_Texture_Ref) -> (t: Texture_Transform) {
	t.scale = {1, 1}
	t.tex_coord = u8(clamp(ref.tex_coord, 0, 1))
	if ref.has_transform {
		t.offset = ref.offset
		t.rotation = ref.rotation
		t.scale = ref.transform_scale
		if ref.has_tex_coord {
			t.tex_coord = u8(clamp(ref.transform_coord, 0, 1))
		}
	}
	return
}

@(private = "file")
convert_material :: proc(doc: ^Gltf_Document, mi: i32, img_ids: []Texture_Id) -> Material {
	if mi < 0 {
		return MATERIAL_DEFAULT
	}
	gm := &doc.materials[mi]
	m := MATERIAL_DEFAULT
	if gm.has_pbr {
		m.color = linear_to_srgb_vec3(gm.base_color.rgb)
		m.opacity = gm.base_color[3]
		m.metalness = gm.metallic
		m.roughness = gm.roughness
		m.color_map = tex_from_ref(doc, gm.base_color_texture, img_ids)
		m.color_transform = ref_transform(gm.base_color_texture)
		// Roughness in green, metalness in blue, one texture.
		m.roughness_map = tex_from_ref(doc, gm.metal_rough_texture, img_ids)
		m.roughness_transform = ref_transform(gm.metal_rough_texture)
		m.metalness_map = m.roughness_map
		m.metalness_transform = m.roughness_transform
	}
	m.normal_map = tex_from_ref(doc, gm.normal_texture, img_ids)
	m.normal_transform = ref_transform(gm.normal_texture)
	m.emissive_map = tex_from_ref(doc, gm.emissive_texture, img_ids)
	m.emissive_transform = ref_transform(gm.emissive_texture)
	m.ao_map = tex_from_ref(doc, gm.occlusion_texture, img_ids)
	m.ao_transform = ref_transform(gm.occlusion_texture)
	m.ao_map_intensity = gm.occlusion_texture.scale if gm.occlusion_texture.texture >= 0 else 1
	m.emissive = linear_to_srgb_vec3(Vec3(gm.emissive))
	m.emissive_intensity = gm.emissive_strength
	if gm.double_sided {
		m.side = .Double
	}
	if gm.unlit {
		m.flags += {.Basic}
	}
	switch gm.alpha_mode {
	case .Opaque:
		m.opacity = 1
	case .Mask:
		m.flags += {.Alpha_Mask}
		m.alpha_cutoff = gm.alpha_cutoff
	case .Blend:
		m.flags += {.Transparent}
	}
	return m
}

// Emits node `ni` then its children; a node whose parent is not emitted
// emits that parent first (skin joints outside the scene).
@(private = "file")
emit_node :: proc(doc: ^Gltf_Document, ni: i32, parent_idx: i32, m: ^Model, node_map: []i32) {
	if ni < 0 || node_map[ni] >= 0 {
		return
	}
	gn := &doc.nodes[ni]
	parent := parent_idx
	if parent < 0 && gn.parent >= 0 {
		if node_map[gn.parent] < 0 {
			emit_node(doc, gn.parent, -1, m, node_map)
		}
		parent = node_map[gn.parent]
	}

	mesh_idx: i32 = -1
	first_weight, weight_count: u32
	if gn.mesh >= 0 {
		mesh_idx = gn.mesh
		mesh := m.meshes[gn.mesh]
		if mesh.weight_count > 0 && budget_take(&doc.budget, .Gltf_Values, int(mesh.weight_count)) {
			first_weight = u32(len(m.node_weights))
			weight_count = mesh.weight_count
			defaults := m.mesh_weights[mesh.first_weight:][:mesh.weight_count]
			node_w := doc.floats[gn.weights.first:][:gn.weights.count]
			for ti in 0 ..< int(weight_count) {
				append(&m.node_weights, node_w[ti] if ti < len(node_w) else defaults[ti])
			}
		}
	}

	t := Vec3(gn.translation) if gn.has_translation else Vec3{0, 0, 0}
	r := quaternion(x = gn.rotation[0], y = gn.rotation[1], z = gn.rotation[2], w = gn.rotation[3]) if gn.has_rotation else linalg.QUATERNIONF32_IDENTITY
	s := Vec3(gn.scale) if gn.has_scale else Vec3{1, 1, 1}
	local: Mat4
	if gn.has_matrix {
		cm := gn.matrix_
		local = Mat4{
			cm[0], cm[4], cm[8],  cm[12],
			cm[1], cm[5], cm[9],  cm[13],
			cm[2], cm[6], cm[10], cm[14],
			cm[3], cm[7], cm[11], cm[15],
		}
	} else {
		local = linalg.matrix4_from_trs_f32(t, r, s)
	}

	my_idx := i32(len(m.nodes))
	node_map[ni] = my_idx
	append(&m.nodes, Model_Node{
		name         = strings.clone(gn.name),
		parent       = parent,
		mesh         = mesh_idx,
		skin         = gn.skin,
		light        = gn.light,
		first_weight = first_weight,
		weight_count = weight_count,
		has_matrix   = gn.has_matrix,
		translation  = t,
		rotation     = r,
		scale        = s,
		local        = local,
	})
	for child := gn.first_child; child >= 0; child = doc.nodes[child].next {
		emit_node(doc, child, my_idx, m, node_map)
	}
}

@(private = "file")
build_skins :: proc(doc: ^Gltf_Document, m: ^Model, node_map: []i32, floats: ^[dynamic]f32) {
	if len(doc.skins) == 0 || len(doc.nodes) == 0 {
		return
	}
	reserve(&m.skins, len(doc.skins))
	for gs in doc.skins {
		first_joint := u32(len(m.skin_joints))
		joints := doc.node_lists[gs.joints.first:][:gs.joints.count]
		jcount := min(len(joints), MAX_JOINTS)
		ibm_acc := gs.inverse_bind_matrices
		has_ibm := ibm_acc >= 0 && doc.accessors[ibm_acc].count >= jcount && budget_take(&doc.budget, .Gltf_Values, jcount * 16)
		if has_ibm {
			unpack_floats_to(doc, ibm_acc, floats, jcount * 16)
		}
		for j in 0 ..< jcount {
			append(&m.skin_joints, u32(max(node_map[joints[j]], 0)))
			ibm := linalg.MATRIX4F32_IDENTITY
			if has_ibm {
				cm := floats[j * 16:]
				ibm = Mat4{
					cm[0], cm[4], cm[8],  cm[12],
					cm[1], cm[5], cm[9],  cm[13],
					cm[2], cm[6], cm[10], cm[14],
					cm[3], cm[7], cm[11], cm[15],
				}
			}
			append(&m.inverse_bind, ibm)
		}
		append(&m.skins, Model_Skin{
			first_joint = first_joint,
			joint_count = u32(jcount),
		})
	}
}

// ---- animations -------------------------------------------------------------------------

@(private = "file")
build_animations :: proc(doc: ^Gltf_Document, m: ^Model, node_map: []i32, floats: ^[dynamic]f32) {
	if len(doc.animations) == 0 || len(doc.nodes) == 0 {
		return
	}
	reserve(&m.animations, len(doc.animations))
	for ga in doc.animations {
		first_ch := u32(len(m.channels))
		duration: f32
		for gc in doc.channels[ga.channels.first:][:ga.channels.count] {
			if gc.node < 0 || node_map[gc.node] < 0 {
				continue
			}
			node_idx := u32(node_map[gc.node])

			path: Animation_Path
			comps: int
			switch gc.path {
			case .Translation:
				path, comps = .Translation, 3
			case .Rotation:
				path, comps = .Rotation, 4
			case .Scale:
				path, comps = .Scale, 3
			case .Weights:
				path = .Weights
				comps = int(m.nodes.weight_count[node_idx])
				if comps <= 0 {
					continue
				}
			case .Invalid:
				continue
			}

			gs := doc.samplers[gc.sampler]
			kcount := doc.accessors[gs.input].count
			if kcount == 0 {
				continue
			}
			interp: Interpolation
			switch gs.interpolation {
			case .Linear:
				interp = .Linear
			case .Step:
				interp = .Step
			case .Cubic_Spline:
				interp = .Cubic_Spline
			}
			val_mult := 3 if interp == .Cubic_Spline else 1
			total_vecs := kcount * val_mult
			if i128(doc.accessors[gs.output].count) < i128(total_vecs) * i128(comps if path == .Weights else 1) || !budget_take_wide(&doc.budget, .Gltf_Values, i128(kcount) + i128(total_vecs) * i128(comps)) {
				continue
			}

			first_time := u32(len(m.times))
			table_grow_non_zero(&m.times, int(first_time) + kcount)
			wrote := gltf_accessor_floats(doc, gs.input, m.times[first_time:])
			slice.zero(m.times[int(first_time) + wrote:])
			if m.times[int(first_time) + kcount - 1] > duration {
				duration = m.times[int(first_time) + kcount - 1]
			}

			if path == .Weights {
				first_val := u32(len(m.morph_values))
				total_floats := total_vecs * comps
				table_grow_non_zero(&m.morph_values, int(first_val) + total_floats)
				wrote = gltf_accessor_floats(doc, gs.output, m.morph_values[first_val:])
				slice.zero(m.morph_values[int(first_val) + wrote:])
				append(&m.channels, Model_Channel{
					node          = node_idx,
					path          = .Weights,
					interpolation = interp,
					first_time    = first_time,
					key_count     = u32(kcount),
					first_value   = first_val,
					weight_count  = u32(comps),
				})
				continue
			}

			unpack_floats_to(doc, gs.output, floats, total_vecs * comps)

			first_val := u32(len(m.values))
			table_grow_non_zero(&m.values, int(first_val) + total_vecs)
			for k in 0 ..< total_vecs {
				base := k * comps
				if comps == 4 {
					m.values[int(first_val) + k] = {floats[base], floats[base + 1], floats[base + 2], floats[base + 3]}
				} else {
					m.values[int(first_val) + k] = {floats[base], floats[base + 1], floats[base + 2], 0}
				}
			}

			append(&m.channels, Model_Channel{
				node          = node_idx,
				path          = path,
				interpolation = interp,
				first_time    = first_time,
				key_count     = u32(kcount),
				first_value   = first_val,
			})
		}
		append(&m.animations, Model_Animation{
			name          = strings.clone(ga.name),
			duration      = duration,
			first_channel = first_ch,
			channel_count = u32(len(m.channels)) - first_ch,
		})
	}
}

@(private = "file")
sample_weights :: proc "contextless" (m: ^Model, ch: Model_Channel, t: f32, dst: []f32) {
	wcount := min(int(ch.weight_count), len(dst))
	if wcount <= 0 {
		return
	}
	times := m.times[ch.first_time:][:ch.key_count]
	vals := m.morph_values[ch.first_value:]
	if ch.key_count == 1 || t <= times[0] {
		base := wcount if ch.interpolation == .Cubic_Spline else 0
		copy(dst[:wcount], vals[base:][:wcount])
		return
	}
	last := int(ch.key_count) - 1
	if t >= times[last] {
		base := (last * 3 + 1) * wcount if ch.interpolation == .Cubic_Spline else last * wcount
		copy(dst[:wcount], vals[base:][:wcount])
		return
	}
	lo, hi := 0, last
	for lo + 1 < hi {
		mid := int(uint(lo + hi) >> 1)
		if times[mid] <= t {
			lo = mid
		} else {
			hi = mid
		}
	}
	if ch.interpolation == .Step {
		copy(dst[:wcount], vals[lo * wcount:][:wcount])
		return
	}
	dt := max(times[lo + 1] - times[lo], 1e-6)
	u := clamp((t - times[lo]) / dt, 0, 1)
	if ch.interpolation == .Cubic_Spline {
		u2 := u * u
		u3 := u2 * u
		h00 := 2 * u3 - 3 * u2 + 1
		h10 := u3 - 2 * u2 + u
		h01 := -2 * u3 + 3 * u2
		h11 := u3 - u2
		b0 := lo * 3 * wcount
		b1 := (lo + 1) * 3 * wcount
		for i in 0 ..< wcount {
			p0 := vals[b0 + wcount + i]
			m0 := vals[b0 + 2 * wcount + i] * dt
			p1 := vals[b1 + wcount + i]
			m1 := vals[b1 + i] * dt
			dst[i] = h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1
		}
		return
	}
	b0 := lo * wcount
	b1 := (lo + 1) * wcount
	for i in 0 ..< wcount {
		dst[i] = linalg.lerp(vals[b0 + i], vals[b1 + i], u)
	}
}

@(private = "file")
sample_channel :: proc "contextless" (m: ^Model, ch: Model_Channel, t: f32) -> Vec4 {
	times := m.times[ch.first_time:][:ch.key_count]
	vals := m.values[ch.first_value:]
	if ch.key_count == 1 || t <= times[0] {
		return vals[1] if ch.interpolation == .Cubic_Spline else vals[0]
	}
	last := int(ch.key_count) - 1
	if t >= times[last] {
		return vals[last * 3 + 1] if ch.interpolation == .Cubic_Spline else vals[last]
	}

	// Binary search for interval times[lo] <= t <= times[lo + 1].
	lo, hi := 0, last
	for lo + 1 < hi {
		mid := int(uint(lo + hi) >> 1)
		if times[mid] <= t {
			lo = mid
		} else {
			hi = mid
		}
	}

	if ch.interpolation == .Step {
		return vals[lo]
	}

	dt := max(times[lo + 1] - times[lo], 1e-6)
	u := clamp((t - times[lo]) / dt, 0, 1)

	if ch.interpolation == .Cubic_Spline {
		u2 := u * u
		u3 := u2 * u
		p0 := vals[lo * 3 + 1]
		m0 := vals[lo * 3 + 2] * dt
		p1 := vals[(lo + 1) * 3 + 1]
		m1 := vals[(lo + 1) * 3 + 0] * dt
		v := (2 * u3 - 3 * u2 + 1) * p0 + (u3 - 2 * u2 + u) * m0 + (-2 * u3 + 3 * u2) * p1 + (u3 - u2) * m1
		return linalg.normalize(v) if ch.path == .Rotation else v
	}

	v0, v1 := vals[lo], vals[lo + 1]
	if ch.path == .Rotation {
		q0 := quaternion(x = v0.x, y = v0.y, z = v0.z, w = v0.w)
		q1 := quaternion(x = v1.x, y = v1.y, z = v1.z, w = v1.w)
		q := linalg.quaternion_slerp_f32(q0, q1, u)
		return {q.x, q.y, q.z, q.w}
	}
	return linalg.lerp(v0, v1, u)
}
