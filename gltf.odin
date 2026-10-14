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
// (blue) map; alphaMode OPAQUE ignores alpha, BLEND blends, MASK draws
// opaque (no alpha cutoff yet); a primitive without normals is flat shaded,
// as the spec asks. Not imported: skins, morph targets, cameras, lights,
// extensions (KHR_*), texture transforms, a second uv set.

Animation_Path :: enum u8 {
	Translation,
	Rotation,
	Scale,
}

Interpolation :: enum u8 {
	Linear,
	Step,
	Cubic_Spline,
}

Model_Mesh :: struct {
	first_primitive, primitive_count: u32,
}

Model_Primitive :: struct {
	geometry: Geometry_Id,
	material: Material,
}

Model_Node :: struct {
	parent:      i32, // -1 for root nodes; always < current node index
	mesh:        i32, // -1 if none, else index in `Model.meshes`
	has_matrix:  bool,
	translation: Vec3,
	rotation:    linalg.Quaternionf32,
	scale:       Vec3,
	local:       Mat4,
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
}

Model :: struct {
	meshes:     [dynamic]Model_Mesh,
	primitives: [dynamic]Model_Primitive,
	nodes:      #soa[dynamic]Model_Node,
	animations: [dynamic]Model_Animation,
	channels:   [dynamic]Model_Channel,
	times:      [dynamic]f32,
	values:     [dynamic]Vec4,
	// Preallocated per-node evaluation scratch so `gltf_draw` never allocates.
	scratch:    struct {
		translation: [dynamic]Vec3,
		rotation:    [dynamic]linalg.Quaternionf32,
		scale:       [dynamic]Vec3,
		dirty:       [dynamic]bool,
		world:       [dynamic]Mat4,
	},
}

model_destroy :: proc(m: ^Model) {
	for a in m.animations {
		delete(a.name)
	}
	delete(m.meshes)
	delete(m.primitives)
	delete(m.nodes)
	delete(m.animations)
	delete(m.channels)
	delete(m.times)
	delete(m.values)
	delete(m.scratch.translation)
	delete(m.scratch.rotation)
	delete(m.scratch.scale)
	delete(m.scratch.dirty)
	delete(m.scratch.world)
	m^ = {}
}

// Loads a `.gltf` or `.glb` file from disk into flat `Model`, `Geometries`,
// and `Textures` tables.
gltf_load :: proc(path: string, geos: ^Geometries, textures: ^Textures = nil) -> (m: Model, ok: bool) {
	cpath := strings.clone_to_cstring(path, context.temp_allocator)
	opts: cgltf.options
	data, res := cgltf.parse_file(opts, cpath)
	if res != .success || data == nil {
		return {}, false
	}
	defer cgltf.free(data)
	if cgltf.load_buffers(opts, data, cpath) != .success || cgltf.validate(data) != .success {
		return {}, false
	}
	slash := strings.last_index_byte(path, '/')
	base_dir := path[:slash] if slash >= 0 else ""
	return build_model(data, geos, textures, base_dir)
}

// Parses a `.gltf` or `.glb` byte slice in memory into flat `Model`,
// `Geometries`, and `Textures` tables.
gltf_parse :: proc(bytes: []u8, geos: ^Geometries, textures: ^Textures = nil, base_dir := "") -> (m: Model, ok: bool) {
	if len(bytes) == 0 {
		return {}, false
	}
	opts: cgltf.options
	data, res := cgltf.parse(opts, raw_data(bytes), uint(len(bytes)))
	if res != .success || data == nil {
		return {}, false
	}
	defer cgltf.free(data)
	cpath: cstring = ""
	if len(base_dir) > 0 {
		dummy := strings.concatenate({base_dir, "/scene.gltf"}, context.temp_allocator)
		cpath = strings.clone_to_cstring(dummy, context.temp_allocator)
	}
	if cgltf.load_buffers(opts, data, cpath) != .success || cgltf.validate(data) != .success {
		return {}, false
	}
	return build_model(data, geos, textures, base_dir)
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

// Evaluates `m`'s node hierarchy (and optional keyframe animation `anim` at
// `time` seconds) under `transform` and appends its mesh primitives to
// `scene.draws`. `flags` (e.g. `{.Cast_Shadow, .Receive_Shadow}`) are merged
// into each primitive's material flags. Zero allocations on warm frames.
gltf_draw :: proc(scene: ^Scene, m: ^Model, transform: Mat4 = linalg.MATRIX4F32_IDENTITY, anim: int = -1, time: f32 = 0, loop := true, flags: Material_Flags = {}) {
	n := len(m.nodes)
	if n == 0 {
		return
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
		copy(m.scratch.translation[:n], m.nodes.translation[:n])
		copy(m.scratch.rotation[:n], m.nodes.rotation[:n])
		copy(m.scratch.scale[:n], m.nodes.scale[:n])
		slice.zero(m.scratch.dirty[:n])

		for ch in m.channels[a.first_channel:][:a.channel_count] {
			if ch.key_count == 0 || int(ch.node) >= n {
				continue
			}
			v := sample_channel(m, ch, t)
			switch ch.path {
			case .Translation:
				m.scratch.translation[ch.node] = v.xyz
			case .Rotation:
				m.scratch.rotation[ch.node] = quaternion(x = v.x, y = v.y, z = v.z, w = v.w)
			case .Scale:
				m.scratch.scale[ch.node] = v.xyz
			}
			m.scratch.dirty[ch.node] = true
		}
	}

	parents := m.nodes.parent[:n]
	meshes := m.nodes.mesh[:n]
	locals := m.nodes.local[:n]
	world := m.scratch.world[:n]
	for i in 0 ..< n {
		local := locals[i]
		if has_anim && m.scratch.dirty[i] {
			local = linalg.matrix4_from_trs_f32(m.scratch.translation[i], m.scratch.rotation[i], m.scratch.scale[i])
		}
		p := parents[i]
		world[i] = (world[p] * local) if p >= 0 else (transform * local)
		mi := meshes[i]
		if mi < 0 || int(mi) >= len(m.meshes) {
			continue
		}
		mesh := m.meshes[mi]
		for prim in m.primitives[mesh.first_primitive:][:mesh.primitive_count] {
			mat := prim.material
			mat.flags += flags
			append(&scene.draws, Draw{
				geometry = prim.geometry,
				model    = world[i],
				material = mat,
			})
		}
	}
}

// ---- model builder ----------------------------------------------------------------------

@(private = "file")
build_model :: proc(data: ^cgltf.data, geos: ^Geometries, textures: ^Textures, base_dir: string) -> (m: Model, ok: bool) {
	// 1. Decode images into `textures` once per glTF image.
	img_ids := make([]Texture_Id, len(data.images), context.temp_allocator)
	if textures != nil {
		for &img, i in data.images {
			img_ids[i] = load_gltf_image(textures, &img, base_dir)
		}
	}

	// 2. Build meshes and primitives into `geos` and `m.primitives`.
	reserve(&m.meshes, len(data.meshes))
	v_scratch := make([dynamic]Vertex, context.temp_allocator)
	i_scratch := make([dynamic]u32, context.temp_allocator)
	f_scratch := make([dynamic]f32, context.temp_allocator)

	for &gm in data.meshes {
		first_prim := u32(len(m.primitives))
		for &gp in gm.primitives {
			if gp.type != .triangles {
				continue
			}
			geo_id, has_normals := build_primitive(geos, &gp, &v_scratch, &i_scratch, &f_scratch)
			if geo_id == 0 {
				continue
			}
			material := convert_material(data, gp.material, img_ids)
			if !has_normals {
				material.flags += {.Flat_Shading}
			}
			append(&m.primitives, Model_Primitive{geometry = geo_id, material = material})
		}
		append(&m.meshes, Model_Mesh{
			first_primitive = first_prim,
			primitive_count = u32(len(m.primitives)) - first_prim,
		})
	}

	// 3. Flatten the default scene's node hierarchy (every root when the file
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

	num_nodes := len(m.nodes)
	resize(&m.scratch.translation, num_nodes)
	resize(&m.scratch.rotation, num_nodes)
	resize(&m.scratch.scale, num_nodes)
	resize(&m.scratch.dirty, num_nodes)
	resize(&m.scratch.world, num_nodes)

	// 4. Flatten animations and keyframe samplers.
	build_animations(data, &m, node_map, &f_scratch)
	return m, true
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

// A triangle primitive into `geos` (0: none), and whether it had normals.
@(private = "file")
build_primitive :: proc(geos: ^Geometries, prim: ^cgltf.primitive, verts: ^[dynamic]Vertex, indices: ^[dynamic]u32, floats: ^[dynamic]f32) -> (id: Geometry_Id, has_normals: bool) {
	pos_acc, norm_acc, uv_acc: ^cgltf.accessor
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
			}
		case .invalid, .tangent, .color, .joints, .weights, .custom:
		}
	}
	if pos_acc == nil || pos_acc.count == 0 {
		return 0, false
	}
	vcount := int(pos_acc.count)
	table_grow(verts, vcount)

	table_grow_non_zero(floats, vcount * 3)
	_ = cgltf.accessor_unpack_floats(pos_acc, raw_data(floats^), uint(vcount * 3))
	for i in 0 ..< vcount {
		verts[i].position = {floats[i * 3], floats[i * 3 + 1], floats[i * 3 + 2]}
		verts[i].normal = {0, 1, 0}
		verts[i].uv = {}
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
			// glTF TEXCOORD_0 has v = 0 at top; o3d's Vertex.uv has v = 1 at top.
			verts[i].uv = {floats[i * 2], 1 - floats[i * 2 + 1]}
		}
	}

	if prim.indices != nil {
		icount := int(prim.indices.count)
		table_grow_non_zero(indices, icount)
		_ = cgltf.accessor_unpack_indices(prim.indices, raw_data(indices^), size_of(u32), uint(icount))
		// validate checked them; a renderer indexes vertices by them unchecked.
		for i in indices^ {
			if int(i) >= vcount {
				return 0, false
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

	return geometry_add(geos, verts[:], indices[:]), has_normals
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
linear_to_srgb_vec3 :: #force_inline proc "contextless" (c: [3]f32) -> Vec3 {
	return {linear_to_srgb(c[0]), linear_to_srgb(c[1]), linear_to_srgb(c[2])}
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
		// Roughness in green, metalness in blue, one texture.
		m.roughness_map = tex_from_view(data, pbr.metallic_roughness_texture, img_ids)
		m.metalness_map = m.roughness_map
	}
	m.normal_map = tex_from_view(data, gm.normal_texture, img_ids)
	m.emissive_map = tex_from_view(data, gm.emissive_texture, img_ids)
	m.emissive = linear_to_srgb_vec3(gm.emissive_factor)
	m.emissive_intensity = 1
	if gm.double_sided {
		m.side = .Double
	}
	if gm.unlit {
		m.flags += {.Basic}
	}
	switch gm.alpha_mode {
	case .opaque, .mask:
		m.opacity = 1 // alpha ignored (MASK's cutoff is not supported yet)
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
	if gn.mesh != nil && len(data.meshes) > 0 {
		mi := int(uintptr(gn.mesh) - uintptr(raw_data(data.meshes))) / size_of(cgltf.mesh)
		if mi >= 0 && mi < len(m.meshes) {
			mesh_idx = i32(mi)
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
		parent      = parent,
		mesh        = mesh_idx,
		has_matrix  = bool(gn.has_matrix),
		translation = t,
		rotation    = r,
		scale       = s,
		local       = local,
	})
	for child in gn.children {
		emit_node(data, child, my_idx, m, node_map)
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
			path: Animation_Path
			comps: int
			switch gc.target_path {
			case .translation:
				path, comps = .Translation, 3
			case .rotation:
				path, comps = .Rotation, 4
			case .scale:
				path, comps = .Scale, 3
			case .invalid, .weights:
				continue
			}
			raw_node := int(uintptr(gc.target_node) - uintptr(raw_data(data.nodes))) / size_of(cgltf.node)
			if raw_node < 0 || raw_node >= len(node_map) || node_map[raw_node] < 0 {
				continue
			}
			node_idx := u32(node_map[raw_node])

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
			if int(gs.output.count) < kcount * val_mult {
				continue
			}

			first_time := u32(len(m.times))
			table_grow_non_zero(&m.times, int(first_time) + kcount)
			_ = cgltf.accessor_unpack_floats(gs.input, raw_data(m.times[first_time:]), uint(kcount))
			if m.times[int(first_time) + kcount - 1] > duration {
				duration = m.times[int(first_time) + kcount - 1]
			}

			total_vecs := kcount * val_mult
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
