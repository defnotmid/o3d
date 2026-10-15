package o3d

import "core:encoding/base64"
import "core:math"
import "core:math/linalg"
import "core:slice"
import "core:strings"
import "vendor:cgltf"

// glTF 2.0 / GLB import (`vendor:cgltf`): meshes, metallic-roughness PBR
// materials and textures, topologically ordered node hierarchy, and keyframe
// animations (`STEP`, `LINEAR`, `CUBICSPLINE`) stored in flat arrays.
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
}

@(private = "file", rodata)
SUPPORTED_EXTENSIONS := [?]string {
	"KHR_materials_unlit",
	"KHR_materials_emissive_strength",
	"KHR_texture_transform",
	"KHR_mesh_quantization",
	"KHR_lights_punctual",
}

@(private = "file")
unsupported_required_extension :: proc(data: ^cgltf.data) -> (string, bool) {
	outer: for raw in data.extensions_required {
		if raw == nil {
			continue
		}
		ext := string(raw)
		for sup in SUPPORTED_EXTENSIONS {
			if ext == sup {
				continue outer
			}
		}
		return strings.clone(ext), true
	}
	return "", false
}

// A cgltf result as a Gltf_Error; `buffers`: from loading external buffers.
@(private = "file")
gltf_error :: proc "contextless" (r: cgltf.result, buffers := false) -> Gltf_Error {
	switch r {
	case .success:
		return .None
	case .data_too_short:
		return .Data_Too_Short
	case .unknown_format:
		return .Unknown_Format
	case .invalid_json:
		return .Invalid_Json
	case .invalid_gltf, .invalid_options:
		return .Invalid_Gltf
	case .file_not_found:
		return .Buffer_Not_Found if buffers else .File_Not_Found
	case .io_error:
		return .Buffer_Io if buffers else .Io
	case .out_of_memory:
		return .Out_Of_Memory
	case .legacy_gltf:
		return .Legacy_Gltf
	}
	return .Invalid_Gltf
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
	geometry: Geometry_Id,
	material: Material,
	skinned:  bool,
}

Model_Skin :: struct {
	first_joint, joint_count: u32,
}

Model_Node :: struct {
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
	// Set when `gltf_load` or `gltf_parse` returns `.Unsupported_Extension`.
	unsupported_extension: string,
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
}

model_pose_destroy :: proc(p: ^Model_Pose) {
	delete(p.translation)
	delete(p.rotation)
	delete(p.scale)
	delete(p.dirty)
	delete(p.world)
	delete(p.joints)
	delete(p.morph_weights)
	p^ = {}
}

model_destroy :: proc(m: ^Model) {
	for a in m.animations {
		delete(a.name)
	}
	for s in m.mesh_target_names {
		delete(s)
	}
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
	model_pose_destroy(&m.scratch)
	m^ = {}
}

// Loads a `.gltf` or `.glb` file from disk into flat `Model`, `Geometries`,
// and `Textures` tables; `err` says why not (GLTF_ERROR_TEXT).
gltf_load :: proc(path: string, geos: ^Geometries, textures: ^Textures = nil) -> (m: Model, err: Gltf_Error) {
	cpath := strings.clone_to_cstring(path, context.temp_allocator)
	opts: cgltf.options
	data, res := cgltf.parse_file(opts, cpath)
	if res != .success || data == nil {
		return {}, gltf_error(res) if res != .success else .Invalid_Gltf
	}
	defer cgltf.free(data)
	if ext, bad := unsupported_required_extension(data); bad {
		return {unsupported_extension = ext}, .Unsupported_Extension
	}
	if r := cgltf.load_buffers(opts, data, cpath); r != .success {
		return {}, gltf_error(r, buffers = true)
	}
	if cgltf.validate(data) != .success {
		return {}, .Invalid
	}
	slash := strings.last_index_byte(path, '/')
	base_dir := path[:slash] if slash >= 0 else ""
	return build_model(data, geos, textures, base_dir), .None
}

// Parses a `.gltf` or `.glb` byte slice in memory into flat `Model`,
// `Geometries`, and `Textures` tables; `err` says why not (GLTF_ERROR_TEXT).
gltf_parse :: proc(bytes: []u8, geos: ^Geometries, textures: ^Textures = nil, base_dir := "") -> (m: Model, err: Gltf_Error) {
	if len(bytes) == 0 {
		return {}, .Empty
	}
	opts: cgltf.options
	data, res := cgltf.parse(opts, raw_data(bytes), uint(len(bytes)))
	if res != .success || data == nil {
		return {}, gltf_error(res) if res != .success else .Invalid_Gltf
	}
	defer cgltf.free(data)
	if ext, bad := unsupported_required_extension(data); bad {
		return {unsupported_extension = ext}, .Unsupported_Extension
	}
	cpath: cstring = ""
	if len(base_dir) > 0 {
		dummy := strings.concatenate({base_dir, "/scene.gltf"}, context.temp_allocator)
		cpath = strings.clone_to_cstring(dummy, context.temp_allocator)
	}
	if r := cgltf.load_buffers(opts, data, cpath); r != .success {
		return {}, gltf_error(r, buffers = true)
	}
	if cgltf.validate(data) != .success {
		return {}, .Invalid
	}
	return build_model(data, geos, textures, base_dir), .None
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
// read. Zero allocations on warm frames.
gltf_draw :: proc(scene: ^Scene, m: ^Model, transform: Mat4 = linalg.MATRIX4F32_IDENTITY, anim: int = -1, time: f32 = 0, loop := true, flags: Material_Flags = {}, pose: ^Model_Pose = nil, override_material: ^Material = nil, morphs: []Morph_Weight = nil, outline_width: f32 = 0, outline_color: Vec3 = {}) {
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
	if len(m.node_weights) > 0 {
		copy(p.morph_weights[:len(m.node_weights)], m.node_weights[:])
	}
	has_anim := anim >= 0 && anim < len(m.animations)
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
		copy(p.translation[:n], m.nodes.translation[:n])
		copy(p.rotation[:n], m.nodes.rotation[:n])
		copy(p.scale[:n], m.nodes.scale[:n])
		slice.zero(p.dirty[:n])

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

	parents := m.nodes.parent[:n]
	meshes := m.nodes.mesh[:n]
	skins := m.nodes.skin[:n]
	lights := m.nodes.light[:n]
	locals := m.nodes.local[:n]
	world := p.world[:n]
	for i in 0 ..< n {
		local := locals[i]
		if has_anim && p.dirty[i] {
			local = linalg.matrix4_from_trs_f32(p.translation[i], p.rotation[i], p.scale[i])
		}
		parent := parents[i]
		world[i] = (world[parent] * local) if parent >= 0 else (transform * local)
	}
	for i in 0 ..< n {
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
			mat.flags += flags
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

// ---- model builder ----------------------------------------------------------------------

@(private = "file")
build_model :: proc(data: ^cgltf.data, geos: ^Geometries, textures: ^Textures, base_dir: string) -> (m: Model) {
	// 1. Decode images into `textures` once per glTF image.
	img_ids := make([]Texture_Id, len(data.images), context.temp_allocator)
	if textures != nil {
		for &img, i in data.images {
			img_ids[i] = load_gltf_image(textures, &img, base_dir)
		}
	}

	// 2. Import KHR_lights_punctual lights.
	reserve(&m.lights, len(data.lights))
	for &gl in data.lights {
		kind: Light_Kind = .Point
		switch gl.type {
		case .directional:
			kind = .Directional
		case .point:
			kind = .Point
		case .spot:
			kind = .Spot
		case .invalid:
		}
		outer := gl.spot_outer_cone_angle if gl.spot_outer_cone_angle > 0 else math.PI / 4
		inner := clamp(gl.spot_inner_cone_angle, 0, outer)
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

	// 3. Build meshes and primitives into `geos` and `m.primitives`.
	reserve(&m.meshes, len(data.meshes))
	v_scratch := make([dynamic]Vertex, context.temp_allocator)
	s_scratch := make([dynamic]Skin_Vertex, context.temp_allocator)
	d_scratch := make([dynamic]Morph_Delta, context.temp_allocator)
	i_scratch := make([dynamic]u32, context.temp_allocator)
	f_scratch := make([dynamic]f32, context.temp_allocator)

	for &gm in data.meshes {
		first_prim := u32(len(m.primitives))
		max_targets := max(len(gm.weights), len(gm.target_names))
		for &gp in gm.primitives {
			if gp.type != .triangles {
				continue
			}
			max_targets = max(max_targets, len(gp.targets))
			geo_id, has_normals, has_skin := build_primitive(geos, &gp, &v_scratch, &s_scratch, &d_scratch, &i_scratch, &f_scratch)
			if geo_id == 0 {
				continue
			}
			material := convert_material(data, gp.material, img_ids)
			if !has_normals {
				material.flags += {.Flat_Shading}
			}
			append(&m.primitives, Model_Primitive{geometry = geo_id, material = material, skinned = has_skin})
		}
		first_weight := u32(len(m.mesh_weights))
		for ti in 0 ..< max_targets {
			w := gm.weights[ti] if ti < len(gm.weights) else 0
			append(&m.mesh_weights, w)
		}
		first_name := u32(len(m.mesh_target_names))
		name_count := min(len(gm.target_names), max_targets)
		for ti in 0 ..< name_count {
			raw_name := gm.target_names[ti]
			cloned := strings.clone(string(raw_name)) if raw_name != nil && len(raw_name) > 0 else ""
			append(&m.mesh_target_names, cloned)
			if len(cloned) > 0 {
				seen := false
				for existing in m.target_names {
					if existing == cloned {
						seen = true
						break
					}
				}
				if !seen {
					append(&m.target_names, cloned)
				}
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
	node_map := make([]i32, len(data.nodes), context.temp_allocator)
	slice.fill(node_map, -1)
	scene := data.scene
	if scene == nil && len(data.scenes) > 0 {
		scene = &data.scenes[0]
	}
	if scene != nil {
		for root in scene.nodes {
			emit_node(data, root, -1, &m, node_map)
		}
	} else {
		for &gn in data.nodes {
			if gn.parent == nil {
				emit_node(data, &gn, -1, &m, node_map)
			}
		}
	}
	for &gs in data.skins {
		for jn in gs.joints {
			emit_node(data, jn, -1, &m, node_map)
		}
	}

	// 5. Flatten skins (joint node indices and inverse bind matrices).
	build_skins(data, &m, node_map, &f_scratch)

	num_nodes := len(m.nodes)
	resize(&m.scratch.translation, num_nodes)
	resize(&m.scratch.rotation, num_nodes)
	resize(&m.scratch.scale, num_nodes)
	resize(&m.scratch.dirty, num_nodes)
	resize(&m.scratch.world, num_nodes)
	resize(&m.scratch.joints, len(m.skin_joints))
	resize(&m.scratch.morph_weights, len(m.node_weights))

	// 6. Flatten animations and keyframe samplers.
	build_animations(data, &m, node_map, &f_scratch)
	return m
}

@(private = "file")
load_gltf_image :: proc(t: ^Textures, img: ^cgltf.image, base_dir: string) -> Texture_Id {
	if img.buffer_view != nil {
		ptr := cgltf.buffer_view_data(img.buffer_view)
		if ptr != nil && img.buffer_view.size > 0 {
			id, _ := texture_decode(t, ptr[:img.buffer_view.size])
			return id
		}
	}
	if img.uri != nil {
		uri := string(img.uri)
		if strings.has_prefix(uri, "data:") {
			if comma := strings.index_byte(uri, ','); comma >= 0 {
				decoded, err := base64.decode(uri[comma + 1:], allocator = context.temp_allocator)
				if err == nil {
					id, _ := texture_decode(t, decoded)
					return id
				}
			}
			return 0
		}
		path := strings.concatenate({base_dir, "/", uri}, context.temp_allocator) if len(base_dir) > 0 else uri
		id, _ := texture_load(t, path)
		return id
	}
	return 0
}

// A triangle primitive into `geos` (0: none), and whether it had normals and skinning.
@(private = "file")
build_primitive :: proc(geos: ^Geometries, prim: ^cgltf.primitive, verts: ^[dynamic]Vertex, skin: ^[dynamic]Skin_Vertex, morphs: ^[dynamic]Morph_Delta, indices: ^[dynamic]u32, floats: ^[dynamic]f32) -> (id: Geometry_Id, has_normals, has_skin: bool) {
	pos_acc, norm_acc, uv_acc, uv1_acc, joints_acc, weights_acc: ^cgltf.accessor
	for &attr in prim.attributes {
		switch attr.type {
		case .position:
			if attr.index == 0 {
				pos_acc = attr.data
			}
		case .normal:
			if attr.index == 0 {
				norm_acc = attr.data
			}
		case .texcoord:
			if attr.index == 0 {
				uv_acc = attr.data
			} else if attr.index == 1 {
				uv1_acc = attr.data
			}
		case .joints:
			if attr.index == 0 {
				joints_acc = attr.data
			}
		case .weights:
			if attr.index == 0 {
				weights_acc = attr.data
			}
		case .invalid, .tangent, .color, .custom:
		}
	}
	if pos_acc == nil || pos_acc.count == 0 {
		return 0, false, false
	}
	vcount := int(pos_acc.count)
	table_grow(verts, vcount)

	table_grow_non_zero(floats, vcount * 3)
	_ = cgltf.accessor_unpack_floats(pos_acc, raw_data(floats^), uint(vcount * 3))
	for i in 0 ..< vcount {
		verts[i].position = {floats[i * 3], floats[i * 3 + 1], floats[i * 3 + 2]}
		verts[i].normal = {0, 1, 0}
		verts[i].uv = {}
		verts[i].uv1 = {}
	}

	if norm_acc != nil && int(norm_acc.count) == vcount {
		_ = cgltf.accessor_unpack_floats(norm_acc, raw_data(floats^), uint(vcount * 3))
		for i in 0 ..< vcount {
			n := Vec3{floats[i * 3], floats[i * 3 + 1], floats[i * 3 + 2]}
			verts[i].normal = linalg.normalize(n) if linalg.length2(n) > 1e-12 else Vec3{0, 1, 0}
		}
	}

	if uv_acc != nil && int(uv_acc.count) == vcount {
		table_grow_non_zero(floats, vcount * 2)
		_ = cgltf.accessor_unpack_floats(uv_acc, raw_data(floats^), uint(vcount * 2))
		for i in 0 ..< vcount {
			// glTF TEXCOORD has v = 0 at top; o3d's Vertex.uv has v = 1 at top.
			verts[i].uv = {floats[i * 2], 1 - floats[i * 2 + 1]}
		}
	}

	if uv1_acc != nil && int(uv1_acc.count) == vcount {
		table_grow_non_zero(floats, vcount * 2)
		_ = cgltf.accessor_unpack_floats(uv1_acc, raw_data(floats^), uint(vcount * 2))
		for i in 0 ..< vcount {
			verts[i].uv1 = {floats[i * 2], 1 - floats[i * 2 + 1]}
		}
	}

	has_skin = joints_acc != nil && weights_acc != nil && int(joints_acc.count) == vcount && int(weights_acc.count) == vcount
	if has_skin {
		table_grow_non_zero(skin, vcount)
		table_grow_non_zero(floats, vcount * 4)
		_ = cgltf.accessor_unpack_floats(weights_acc, raw_data(floats^), uint(vcount * 4))
		for i in 0 ..< vcount {
			j: [4]u32
			_ = cgltf.accessor_read_uint(joints_acc, uint(i), &j[0], 4)
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
			skin[i] = {joints, w}
		}
	}

	has_morphs := len(prim.targets) > 0
	if has_morphs {
		tcount := len(prim.targets)
		clear(morphs)
		table_grow(morphs, tcount * vcount)
		table_grow_non_zero(floats, vcount * 3)
		for &target, ti in prim.targets {
			t_pos, t_norm: ^cgltf.accessor
			for &attr in target.attributes {
				switch attr.type {
				case .position:
					t_pos = attr.data
				case .normal:
					t_norm = attr.data
				case .invalid, .tangent, .texcoord, .color, .joints, .weights, .custom:
				}
			}
			base := ti * vcount
			if t_pos != nil && int(t_pos.count) == vcount {
				_ = cgltf.accessor_unpack_floats(t_pos, raw_data(floats^), uint(vcount * 3))
				for i in 0 ..< vcount {
					morphs[base + i].position = {floats[i * 3], floats[i * 3 + 1], floats[i * 3 + 2], 0}
				}
			}
			if t_norm != nil && int(t_norm.count) == vcount {
				_ = cgltf.accessor_unpack_floats(t_norm, raw_data(floats^), uint(vcount * 3))
				for i in 0 ..< vcount {
					morphs[base + i].normal = {floats[i * 3], floats[i * 3 + 1], floats[i * 3 + 2], 0}
				}
			}
		}
	}

	if prim.indices != nil {
		icount := int(prim.indices.count)
		table_grow_non_zero(indices, icount)
		_ = cgltf.accessor_unpack_indices(prim.indices, raw_data(indices^), size_of(u32), uint(icount))
		// validate checked them; a renderer indexes vertices by them unchecked.
		for i in indices^ {
			if int(i) >= vcount {
				return 0, false, false
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
	has_normals = norm_acc != nil && int(norm_acc.count) == vcount
	if !has_normals {
		for &v in verts^ {
			v.normal = {}
		}
		for t := 0; t + 2 < len(indices); t += 3 {
			i0, i1, i2 := indices[t], indices[t + 1], indices[t + 2]
			if int(i0) >= vcount || int(i1) >= vcount || int(i2) >= vcount {
				continue
			}
			fn := linalg.cross(verts[i1].position - verts[i0].position, verts[i2].position - verts[i0].position)
			verts[i0].normal += fn
			verts[i1].normal += fn
			verts[i2].normal += fn
		}
		for &v in verts^ {
			v.normal = linalg.normalize(v.normal) if linalg.length2(v.normal) > 1e-12 else Vec3{0, 1, 0}
		}
	}

	return geometry_add(geos, verts[:], indices[:], skin[:] if has_skin else nil, morphs[:] if has_morphs else nil), has_normals, has_skin
}

@(private = "file")
tex_from_view :: proc(data: ^cgltf.data, view: cgltf.texture_view, img_ids: []Texture_Id) -> Texture_Id {
	if view.texture == nil || view.texture.image_ == nil || len(data.images) == 0 {
		return 0
	}
	idx := int(uintptr(view.texture.image_) - uintptr(raw_data(data.images))) / size_of(cgltf.image)
	if idx < 0 || idx >= len(img_ids) {
		return 0
	}
	return img_ids[idx]
}

@(private = "file")
view_transform :: proc "contextless" (view: cgltf.texture_view) -> (t: Texture_Transform) {
	t.scale = {1, 1}
	t.tex_coord = u8(clamp(view.texcoord, 0, 1))
	if view.has_transform {
		xf := view.transform
		t.offset = xf.offset
		t.rotation = xf.rotation
		t.scale = xf.scale
		if xf.has_texcoord {
			t.tex_coord = u8(clamp(xf.texcoord, 0, 1))
		}
	}
	return
}

@(private = "file")
convert_material :: proc(data: ^cgltf.data, gm: ^cgltf.material, img_ids: []Texture_Id) -> Material {
	if gm == nil {
		return MATERIAL_DEFAULT
	}
	m := MATERIAL_DEFAULT
	if gm.has_pbr_metallic_roughness {
		pbr := gm.pbr_metallic_roughness
		m.color = linear_to_srgb_vec3(pbr.base_color_factor.rgb)
		m.opacity = pbr.base_color_factor[3]
		m.metalness = pbr.metallic_factor
		m.roughness = pbr.roughness_factor
		m.color_map = tex_from_view(data, pbr.base_color_texture, img_ids)
		m.color_transform = view_transform(pbr.base_color_texture)
		// Roughness in green, metalness in blue, one texture.
		m.roughness_map = tex_from_view(data, pbr.metallic_roughness_texture, img_ids)
		m.roughness_transform = view_transform(pbr.metallic_roughness_texture)
		m.metalness_map = m.roughness_map
		m.metalness_transform = m.roughness_transform
	}
	m.normal_map = tex_from_view(data, gm.normal_texture, img_ids)
	m.normal_transform = view_transform(gm.normal_texture)
	m.emissive_map = tex_from_view(data, gm.emissive_texture, img_ids)
	m.emissive_transform = view_transform(gm.emissive_texture)
	m.ao_map = tex_from_view(data, gm.occlusion_texture, img_ids)
	m.ao_transform = view_transform(gm.occlusion_texture)
	m.ao_map_intensity = gm.occlusion_texture.scale if gm.occlusion_texture.texture != nil else 1
	m.emissive = linear_to_srgb_vec3(Vec3(gm.emissive_factor))
	m.emissive_intensity = gm.emissive_strength.emissive_strength if gm.has_emissive_strength else 1
	if gm.double_sided {
		m.side = .Double
	}
	if gm.unlit {
		m.flags += {.Basic}
	}
	switch gm.alpha_mode {
	case .opaque:
		m.opacity = 1
	case .mask:
		m.flags += {.Alpha_Mask}
		m.alpha_cutoff = gm.alpha_cutoff
	case .blend:
		m.flags += {.Transparent}
	}
	return m
}

@(private = "file")
emit_node :: proc(data: ^cgltf.data, gn: ^cgltf.node, parent_idx: i32, m: ^Model, node_map: []i32) {
	if gn == nil || len(data.nodes) == 0 {
		return
	}
	raw_idx := int(uintptr(gn) - uintptr(raw_data(data.nodes))) / size_of(cgltf.node)
	if raw_idx < 0 || raw_idx >= len(node_map) || node_map[raw_idx] >= 0 {
		return
	}
	parent := parent_idx
	if parent < 0 && gn.parent != nil {
		p_raw := int(uintptr(gn.parent) - uintptr(raw_data(data.nodes))) / size_of(cgltf.node)
		if p_raw >= 0 && p_raw < len(node_map) {
			if node_map[p_raw] < 0 {
				emit_node(data, gn.parent, -1, m, node_map)
			}
			parent = node_map[p_raw]
		}
	}

	mesh_idx: i32 = -1
	first_weight, weight_count: u32
	if gn.mesh != nil && len(data.meshes) > 0 {
		mi := int(uintptr(gn.mesh) - uintptr(raw_data(data.meshes))) / size_of(cgltf.mesh)
		if mi >= 0 && mi < len(m.meshes) {
			mesh_idx = i32(mi)
			mesh := m.meshes[mi]
			if mesh.weight_count > 0 {
				first_weight = u32(len(m.node_weights))
				weight_count = mesh.weight_count
				defaults := m.mesh_weights[mesh.first_weight:][:mesh.weight_count]
				for ti in 0 ..< int(weight_count) {
					w := gn.weights[ti] if ti < len(gn.weights) else defaults[ti]
					append(&m.node_weights, w)
				}
			}
		}
	}
	skin_idx: i32 = -1
	if gn.skin != nil && len(data.skins) > 0 {
		si := int(uintptr(gn.skin) - uintptr(raw_data(data.skins))) / size_of(cgltf.skin)
		if si >= 0 && si < len(data.skins) {
			skin_idx = i32(si)
		}
	}
	light_idx: i32 = -1
	if gn.light != nil && len(data.lights) > 0 {
		li := int(uintptr(gn.light) - uintptr(raw_data(data.lights))) / size_of(cgltf.light)
		if li >= 0 && li < len(m.lights) {
			light_idx = i32(li)
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
	node_map[raw_idx] = my_idx
	append(&m.nodes, Model_Node{
		parent       = parent,
		mesh         = mesh_idx,
		skin         = skin_idx,
		light        = light_idx,
		first_weight = first_weight,
		weight_count = weight_count,
		has_matrix   = bool(gn.has_matrix),
		translation  = t,
		rotation     = r,
		scale        = s,
		local        = local,
	})
	for child in gn.children {
		emit_node(data, child, my_idx, m, node_map)
	}
}

@(private = "file")
build_skins :: proc(data: ^cgltf.data, m: ^Model, node_map: []i32, floats: ^[dynamic]f32) {
	if len(data.skins) == 0 || len(data.nodes) == 0 {
		return
	}
	reserve(&m.skins, len(data.skins))
	for &gs in data.skins {
		first_joint := u32(len(m.skin_joints))
		jcount := min(len(gs.joints), MAX_JOINTS)
		has_ibm := gs.inverse_bind_matrices != nil && int(gs.inverse_bind_matrices.count) >= jcount
		if has_ibm {
			table_grow_non_zero(floats, jcount * 16)
			_ = cgltf.accessor_unpack_floats(gs.inverse_bind_matrices, raw_data(floats^), uint(jcount * 16))
		}
		for j in 0 ..< jcount {
			jn := gs.joints[j]
			node_idx: u32
			if jn != nil {
				raw_node := int(uintptr(jn) - uintptr(raw_data(data.nodes))) / size_of(cgltf.node)
				if raw_node >= 0 && raw_node < len(node_map) && node_map[raw_node] >= 0 {
					node_idx = u32(node_map[raw_node])
				}
			}
			append(&m.skin_joints, node_idx)
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
build_animations :: proc(data: ^cgltf.data, m: ^Model, node_map: []i32, floats: ^[dynamic]f32) {
	if len(data.animations) == 0 || len(data.nodes) == 0 {
		return
	}
	reserve(&m.animations, len(data.animations))
	for &ga in data.animations {
		first_ch := u32(len(m.channels))
		duration: f32
		for &gc in ga.channels {
			if gc.sampler == nil || gc.target_node == nil || gc.sampler.input == nil || gc.sampler.output == nil {
				continue
			}
			raw_node := int(uintptr(gc.target_node) - uintptr(raw_data(data.nodes))) / size_of(cgltf.node)
			if raw_node < 0 || raw_node >= len(node_map) || node_map[raw_node] < 0 {
				continue
			}
			node_idx := u32(node_map[raw_node])

			path: Animation_Path
			comps: int
			switch gc.target_path {
			case .translation:
				path, comps = .Translation, 3
			case .rotation:
				path, comps = .Rotation, 4
			case .scale:
				path, comps = .Scale, 3
			case .weights:
				path = .Weights
				comps = int(m.nodes.weight_count[node_idx])
				if comps <= 0 {
					continue
				}
			case .invalid:
				continue
			}

			gs := gc.sampler
			kcount := int(gs.input.count)
			if kcount == 0 {
				continue
			}
			interp: Interpolation
			switch gs.interpolation {
			case .linear:
				interp = .Linear
			case .step:
				interp = .Step
			case .cubic_spline:
				interp = .Cubic_Spline
			}
			val_mult := 3 if interp == .Cubic_Spline else 1
			total_vecs := kcount * val_mult
			expected_count := total_vecs * comps if path == .Weights else total_vecs
			if int(gs.output.count) < expected_count {
				continue
			}

			first_time := u32(len(m.times))
			table_grow_non_zero(&m.times, int(first_time) + kcount)
			_ = cgltf.accessor_unpack_floats(gs.input, raw_data(m.times[first_time:]), uint(kcount))
			if m.times[int(first_time) + kcount - 1] > duration {
				duration = m.times[int(first_time) + kcount - 1]
			}

			if path == .Weights {
				first_val := u32(len(m.morph_values))
				total_floats := total_vecs * comps
				table_grow_non_zero(&m.morph_values, int(first_val) + total_floats)
				_ = cgltf.accessor_unpack_floats(gs.output, raw_data(m.morph_values[first_val:]), uint(total_floats))
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

			table_grow_non_zero(floats, total_vecs * comps)
			_ = cgltf.accessor_unpack_floats(gs.output, raw_data(floats^), uint(total_vecs * comps))

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
			name          = strings.clone(string(ga.name)) if ga.name != nil else "",
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
