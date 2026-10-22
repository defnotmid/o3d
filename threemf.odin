package o3d

import "base:runtime"
import "core:encoding/xml"
import "core:math/linalg"
import "core:os"
import "core:slice"
import "core:strconv"
import "core:strings"

// 3MF (3D Manufacturing Format) import: reads `.3mf` ZIP archives (`3D/3dmodel.model`
// and any production-extension sub-model `.model` parts) into the same flat
// `Model`, `Geometries` and `Textures` tables as `gltf_load`, so `gltf_draw`
// draws 3MF models through the same path with zero warm-frame allocations.
//
// Conventions:
// - 3MF is right-handed with +Z up (+X right, +Y back); o3d is right-handed
//   with +Y up (+X right, -Z forward). Vertex positions and 3x4 affine
//   component/build transforms are converted via the +1-determinant basis
//   change `(x, y, z)_3mf -> (x, z, -y)_o3d`, preserving counter-clockwise
//   triangle winding.
// - Root `<model unit="...">` coordinates stay in the root model's unit (as
//   three.js's ThreeMFLoader does); sub-models with a different `unit` scale
//   into the root model's unit.
// - `<basematerials>` (`displaycolor`), `<m:colorgroup>` (`color`),
//   `<m:pbmetallicdisplayproperties>` (`metallicness`, `roughness`), and
//   `<m:texture2d>` + `<m:texture2dgroup>` (`u`, `v`) map to `o3d.Material`
//   and `Textures`. Multi-material meshes split into one `Model_Primitive`
//   per distinct material; flat face normals are computed per triangle.
//   Each `<base>` is a file material (`Model.material_names`: its `name`),
//   its primitives' `material_id`; colour and texture groups' are 0.

// Why a `.3mf` archive did not load; THREEMF_ERROR_TEXT says it in words.
Threemf_Error :: enum u8 {
	None,
	Empty,
	File_Not_Found,
	Io,
	Data_Too_Short,
	Not_Zip,
	Corrupt_Zip,
	Missing_Model,
	Invalid_Xml,
	Invalid_Model,
	Out_Of_Memory,
	// `requiredextensions` names a 3MF extension o3d does not implement;
	// `Model.unsupported_extension` has its namespace or prefix.
	Unsupported_Extension,
}

@(rodata)
THREEMF_ERROR_TEXT := [Threemf_Error]string {
	.None                  = "",
	.Empty                 = "no data",
	.File_Not_Found        = "file not found",
	.Io                    = "could not read the file",
	.Data_Too_Short        = "the data ends early",
	.Not_Zip               = "not a 3MF ZIP archive",
	.Corrupt_Zip           = "corrupt ZIP archive",
	.Missing_Model         = "no 3D model part in the archive",
	.Invalid_Xml           = "invalid XML in model part",
	.Invalid_Model         = "not a valid 3MF model",
	.Out_Of_Memory         = "out of memory",
	.Unsupported_Extension = "requires an unsupported 3MF extension",
}

@(private = "file", rodata)
SUPPORTED_3MF_EXTENSIONS := [?]string {
	"http://schemas.microsoft.com/3dmanufacturing/core/2015/02",
	"http://schemas.microsoft.com/3dmanufacturing/material/2015/02",
	"http://schemas.microsoft.com/3dmanufacturing/production/2015/06",
}

@(private = "file", rodata)
THREEMF_UNITS := [?]struct {
	name:   string,
	meters: f32,
} {
	{"micron", 1e-6},
	{"millimeter", 1e-3},
	{"centimeter", 1e-2},
	{"inch", 0.0254},
	{"foot", 0.3048},
	{"meter", 1.0},
}

// Loads a `.3mf` file from disk into flat `Model`, `Geometries`, and
// `Textures` tables; `err` says why not (`THREEMF_ERROR_TEXT`).
threemf_load :: proc(path: string, geos: ^Geometries, textures: ^Textures = nil) -> (m: Model, err: Threemf_Error) {
	if !os.exists(path) {
		return {}, .File_Not_Found
	}
	data, read_err := os.read_entire_file(path, context.allocator)
	if read_err != nil {
		return {}, .Io
	}
	defer delete(data)
	return threemf_parse(data, geos, textures)
}

// Parses a `.3mf` archive in memory into flat `Model`, `Geometries`, and
// `Textures` tables; `err` says why not (`THREEMF_ERROR_TEXT`).
threemf_parse :: proc(archive: []u8, geos: ^Geometries, textures: ^Textures = nil) -> (m: Model, err: Threemf_Error) {
	if len(archive) == 0 {
		return {}, .Empty
	}
	// The build's scratch is gone on return: what the model keeps is copied.
	runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()
	if len(archive) < 4 {
		return {}, .Data_Too_Short
	}
	entries, zip_err := zip_read_directory(archive)
	switch zip_err {
	case .None:
	case .Data_Too_Short:
		return {}, .Data_Too_Short
	case .Not_Zip:
		return {}, .Not_Zip
	case .Corrupt_Zip:
		return {}, .Corrupt_Zip
	}
	root_path := threemf_root_model_path(archive, entries)
	if root_path == "" {
		return {}, .Missing_Model
	}

	b: Threemf_Builder
	b.archive = archive
	b.entries = entries
	b.geos = geos
	b.textures = textures
	b.model = &m

	root_part, part_err := threemf_load_part(&b, root_path, is_root = true)
	if part_err != .None {
		if part_err == .Unsupported_Extension {
			return m, .Unsupported_Extension
		}
		model_destroy(&m)
		return {}, part_err
	}

	part := &b.parts[root_part]
	for item in b.items[part.first_item:][:part.item_count] {
		threemf_emit_node(&b, item.part, item.object_id, -1, item.transform, 0)
	}
	num_nodes := len(m.nodes)
	resize(&m.scratch.translation, num_nodes)
	resize(&m.scratch.rotation, num_nodes)
	resize(&m.scratch.scale, num_nodes)
	resize(&m.scratch.dirty, num_nodes)
	resize(&m.scratch.world, num_nodes)
	return m, .None
}

// Discovers the primary 3D model part via `_rels/.rels` (relationship type
// `.../3dmodel`), falling back to `3D/3dmodel.model` or the first `.model` entry.
@(private = "file")
threemf_root_model_path :: proc(archive: []u8, entries: []Zip_Entry) -> string {
	if rels, ok := zip_extract(archive, entries, "_rels/.rels"); ok && len(rels) > 0 {
		if doc, err := xml.parse_bytes(rels, allocator = context.temp_allocator); err == .None && doc.element_count > 0 {
			for el in doc.elements[:doc.element_count] {
				if el.kind != .Element || !strings.equal_fold(xml_local(el.ident), "Relationship") {
					continue
				}
				target, t_ok := xml_attr(el, "Target")
				rel_type, r_ok := xml_attr(el, "Type")
				if t_ok && r_ok && strings.has_suffix(strings.to_lower(rel_type, context.temp_allocator), "/3dmodel") {
					clean := trim_archive_path(target)
					if _, exists := zip_extract(archive, entries, clean); exists {
						return clean
					}
				}
			}
		}
	}
	for e in entries {
		if strings.equal_fold(e.name, "3D/3dmodel.model") {
			return e.name
		}
	}
	for e in entries {
		if strings.has_suffix(strings.to_lower(e.name, context.temp_allocator), ".model") {
			return e.name
		}
	}
	return ""
}

// ---- flat 3MF builder tables ------------------------------------------------------

@(private = "file")
Threemf_PBR_Prop :: struct {
	metalness, roughness: f32,
}

@(private = "file")
Threemf_PBR_Group :: struct {
	id:                u32,
	first_prop, count: u32,
}

@(private = "file")
Threemf_Tex2D :: struct {
	id:  u32,
	tex: Texture_Id,
}

@(private = "file")
Threemf_Group_Kind :: enum u8 {
	Base_Materials,
	Color_Group,
	Texture_Group,
}

@(private = "file")
Threemf_Group :: struct {
	id:                u32,
	kind:              Threemf_Group_Kind,
	tex:               Texture_Id, // for `.Texture_Group`
	first_item, count: u32,        // slice in `materials` or `uvs`
}

@(private = "file")
Threemf_Component :: struct {
	part:      i32,
	object_id: u32,
	transform: Mat4,
}

@(private = "file")
Threemf_Object :: struct {
	id:                     u32,
	name:                   string,
	mesh:                   i32, // index in `Model.meshes`, or -1
	first_comp, comp_count: u32,
}

@(private = "file")
Threemf_Part :: struct {
	path:                         string,
	unit_meters:                  f32,
	unit_scale:                   f32,
	first_pbr, pbr_count:         u32,
	first_tex, tex_count:         u32,
	first_group, group_count:     u32,
	first_object, object_count:   u32,
	first_item, item_count:       u32,
}

@(private = "file")
Threemf_Builder :: struct {
	archive:     []u8,
	entries:     []Zip_Entry,
	geos:        ^Geometries,
	textures:    ^Textures,
	model:       ^Model,
	root_meters: f32,
	parts:       [dynamic]Threemf_Part,
	pbr_props:   [dynamic]Threemf_PBR_Prop,
	pbr_groups:  [dynamic]Threemf_PBR_Group,
	tex2ds:      [dynamic]Threemf_Tex2D,
	materials:   [dynamic]Material,
	// By `materials` row: its file material (a `<base>`'s; 0: none).
	material_ids: [dynamic]Model_Material_Id,
	uvs:         [dynamic]Vec2,
	groups:      [dynamic]Threemf_Group,
	components:  [dynamic]Threemf_Component,
	objects:     [dynamic]Threemf_Object,
	items:       [dynamic]Threemf_Component,
}

@(private = "file")
xml_local :: #force_inline proc "contextless" (s: string) -> string {
	if i := strings.index_byte(s, ':'); i >= 0 {
		return s[i + 1:]
	}
	return s
}

@(private = "file")
xml_attr :: proc "contextless" (el: xml.Element, name: string) -> (string, bool) {
	for a in el.attribs {
		if a.key == name || xml_local(a.key) == name {
			return a.val, true
		}
	}
	return "", false
}

@(private = "file")
parse_u32_attr :: proc(el: xml.Element, name: string) -> (u32, bool) {
	s, ok := xml_attr(el, name)
	if !ok {
		return 0, false
	}
	v, p_ok := strconv.parse_uint(strings.trim_space(s), 10)
	return u32(v), p_ok
}

@(private = "file")
parse_f32_attr :: proc(el: xml.Element, name: string, default: f32 = 0) -> (f32, bool) {
	s, ok := xml_attr(el, name)
	if !ok {
		return default, false
	}
	v, p_ok := strconv.parse_f32(strings.trim_space(s))
	return v if p_ok else default, p_ok
}

@(private = "file")
unit_in_meters :: proc "contextless" (name: string) -> f32 {
	for u in THREEMF_UNITS {
		if u.name == name {
			return u.meters
		}
	}
	return 1e-3 // 3MF default is millimeter
}

// Parses `#RRGGBB` or `#RRGGBBAA` sRGB hex color.
@(private = "file")
parse_hex_color :: proc(s: string) -> (color: Vec3, opacity: f32, ok: bool) {
	str := strings.trim_space(s)
	if len(str) == 0 || str[0] != '#' {
		return {0.8, 0.8, 0.8}, 1, false
	}
	hex := str[1:]
	if len(hex) != 6 && len(hex) != 8 {
		return {0.8, 0.8, 0.8}, 1, false
	}
	r, r_ok := strconv.parse_uint(hex[0:2], 16)
	g, g_ok := strconv.parse_uint(hex[2:4], 16)
	b, b_ok := strconv.parse_uint(hex[4:6], 16)
	if !r_ok || !g_ok || !b_ok {
		return {0.8, 0.8, 0.8}, 1, false
	}
	a := uint(255)
	if len(hex) == 8 {
		a_val, a_ok := strconv.parse_uint(hex[6:8], 16)
		if !a_ok {
			return {0.8, 0.8, 0.8}, 1, false
		}
		a = a_val
	}
	return {f32(r) / 255.0, f32(g) / 255.0, f32(b) / 255.0}, f32(a) / 255.0, true
}

// Converts a 3MF 3x4 row-vector affine matrix ("m00 m01 m02 m10 ... m32" in
// Z-up space, with translation scaled by `unit_scale`) into o3d's column-vector
// Y-up `Mat4` via `B * M * B^-1`, where `B(x, y, z) = (x, z, -y)`.
@(private = "file")
parse_threemf_transform :: proc(s: string, unit_scale: f32) -> Mat4 {
	m: [12]f32 = {
		1, 0, 0,
		0, 1, 0,
		0, 0, 1,
		0, 0, 0,
	}
	rest := s
	count := 0
	for count < 12 {
		rest = strings.trim_left_space(rest)
		if len(rest) == 0 {
			break
		}
		end := 0
		for end < len(rest) && rest[end] != ' ' && rest[end] != '\t' && rest[end] != '\r' && rest[end] != '\n' {
			end += 1
		}
		v, ok := strconv.parse_f32(rest[:end])
		if !ok {
			return linalg.MATRIX4F32_IDENTITY
		}
		m[count] = v
		count += 1
		rest = rest[end:]
	}
	if count < 12 {
		return linalg.MATRIX4F32_IDENTITY
	}
	m00, m01, m02 := m[0], m[1], m[2]
	m10, m11, m12 := m[3], m[4], m[5]
	m20, m21, m22 := m[6], m[7], m[8]
	tx := m[9] * unit_scale
	ty := m[10] * unit_scale
	tz := m[11] * unit_scale
	return Mat4 {
		m00,  m20, -m10,  tx,
		m02,  m22, -m12,  tz,
		-m01, -m21,  m11, -ty,
		0,    0,    0,    1,
	}
}

@(private = "file")
check_required_3mf_extensions :: proc(el: xml.Element) -> (unsupported: string, bad: bool) {
	req_str: string
	has_req := false
	for a in el.attribs {
		if a.key == "requiredextensions" {
			req_str = a.val
			has_req = true
			break
		}
	}
	if !has_req {
		return "", false
	}
	rest := req_str
	outer: for {
		rest = strings.trim_left_space(rest)
		if len(rest) == 0 {
			break
		}
		end := 0
		for end < len(rest) && rest[end] != ' ' && rest[end] != '\t' && rest[end] != '\r' && rest[end] != '\n' {
			end += 1
		}
		prefix := rest[:end]
		rest = rest[end:]

		uri := prefix
		xmlns_key := strings.concatenate({"xmlns:", prefix}, context.temp_allocator)
		for a in el.attribs {
			if a.key == xmlns_key {
				uri = a.val
				break
			}
		}
		for sup in SUPPORTED_3MF_EXTENSIONS {
			if uri == sup {
				continue outer
			}
		}
		return strings.clone(uri), true
	}
	return "", false
}

@(private = "file")
find_pbr_prop :: proc "contextless" (b: ^Threemf_Builder, part: ^Threemf_Part, dp_id: u32, index: u32) -> (Threemf_PBR_Prop, bool) {
	if dp_id == 0 {
		return {}, false
	}
	for g in b.pbr_groups[part.first_pbr:][:part.pbr_count] {
		if g.id == dp_id && index < g.count {
			return b.pbr_props[g.first_prop + index], true
		}
	}
	return {}, false
}

@(private = "file")
find_tex2d :: proc "contextless" (b: ^Threemf_Builder, part: ^Threemf_Part, id: u32) -> Texture_Id {
	for t in b.tex2ds[part.first_tex:][:part.tex_count] {
		if t.id == id {
			return t.tex
		}
	}
	return 0
}

@(private = "file")
find_group :: proc "contextless" (b: ^Threemf_Builder, part: ^Threemf_Part, id: u32) -> (int, bool) {
	if id == 0 {
		return -1, false
	}
	for i in 0 ..< int(part.group_count) {
		gi := int(part.first_group) + i
		if b.groups[gi].id == id {
			return gi, true
		}
	}
	return -1, false
}

// Loads a `.model` part from the archive (returning its index in `b.parts` if
// already loaded).
@(private = "file")
threemf_load_part :: proc(b: ^Threemf_Builder, raw_path: string, is_root := false) -> (part_idx: int, err: Threemf_Error) {
	clean := trim_archive_path(raw_path)
	for p, i in b.parts {
		if strings.equal_fold(p.path, clean) {
			return i, .None
		}
	}
	xml_bytes, extracted := zip_extract(b.archive, b.entries, clean)
	if !extracted || len(xml_bytes) == 0 {
		return -1, .Missing_Model
	}
	doc, xml_err := xml.parse_bytes(xml_bytes, allocator = context.temp_allocator)
	if xml_err != .None || doc == nil || doc.element_count == 0 {
		return -1, .Invalid_Xml
	}

	model_id := xml.Element_ID(0)
	found_model := false
	for el, idx in doc.elements[:doc.element_count] {
		if el.kind == .Element && xml_local(el.ident) == "model" {
			model_id = xml.Element_ID(idx)
			found_model = true
			break
		}
	}
	if !found_model {
		return -1, .Invalid_Model
	}
	model_el := doc.elements[model_id]
	if ext, bad := check_required_3mf_extensions(model_el); bad {
		b.model.unsupported_extension = ext
		return -1, .Unsupported_Extension
	}

	unit_name, has_unit := xml_attr(model_el, "unit")
	meters := unit_in_meters(unit_name) if has_unit else 1e-3
	if is_root {
		b.root_meters = meters
	}
	unit_scale := meters / b.root_meters if b.root_meters > 0 else 1

	if b.parts.allocator.procedure == nil {
		b.parts.allocator = context.temp_allocator
		b.pbr_props.allocator = context.temp_allocator
		b.pbr_groups.allocator = context.temp_allocator
		b.tex2ds.allocator = context.temp_allocator
		b.materials.allocator = context.temp_allocator
		b.material_ids.allocator = context.temp_allocator
		b.uvs.allocator = context.temp_allocator
		b.groups.allocator = context.temp_allocator
		b.components.allocator = context.temp_allocator
		b.objects.allocator = context.temp_allocator
		b.items.allocator = context.temp_allocator
	}

	part_idx = len(b.parts)
	append(&b.parts, Threemf_Part {
		path        = clean,
		unit_meters = meters,
		unit_scale  = unit_scale,
	})

	// Pass 1: `<resources>` (display properties, 2D textures, property groups, objects).
	for v in model_el.value {
		child_id := v.(xml.Element_ID) or_continue
		child := doc.elements[child_id]
		if child.kind != .Element || xml_local(child.ident) != "resources" {
			continue
		}
		threemf_parse_resources(b, part_idx, doc, child) or_return
	}

	// Pass 2: `<build>` items.
	first_item := u32(len(b.items))
	item_count := u32(0)
	for v in model_el.value {
		child_id := v.(xml.Element_ID) or_continue
		child := doc.elements[child_id]
		if child.kind != .Element || xml_local(child.ident) != "build" {
			continue
		}
		for iv in child.value {
			item_id := iv.(xml.Element_ID) or_continue
			item_el := doc.elements[item_id]
			if item_el.kind != .Element || xml_local(item_el.ident) != "item" {
				continue
			}
			obj_id := parse_u32_attr(item_el, "objectid") or_continue
			target_part := part_idx
			if p_path, has_p := xml_attr(item_el, "path"); has_p && p_path != "" {
				target_part = threemf_load_part(b, p_path) or_return
			}
			xform := linalg.MATRIX4F32_IDENTITY
			if tf_str, has_tf := xml_attr(item_el, "transform"); has_tf {
				xform = parse_threemf_transform(tf_str, unit_scale)
			}
			append(&b.items, Threemf_Component {
				part      = i32(target_part),
				object_id = obj_id,
				transform = xform,
			})
			item_count += 1
		}
	}
	b.parts[part_idx].first_item = first_item
	b.parts[part_idx].item_count = item_count
	return part_idx, .None
}

@(private = "file")
threemf_parse_resources :: proc(b: ^Threemf_Builder, part_idx: int, doc: ^xml.Document, res_el: xml.Element) -> Threemf_Error {
	// Sub-pass A: `<pbmetallicdisplayproperties>` and `<texture2d>` first so
	// `<basematerials>`, `<colorgroup>` and `<texture2dgroup>` can resolve them
	// regardless of declaration order.
	first_pbr := u32(len(b.pbr_groups))
	pbr_count := u32(0)
	first_tex := u32(len(b.tex2ds))
	tex_count := u32(0)

	for v in res_el.value {
		el_id := v.(xml.Element_ID) or_continue
		el := doc.elements[el_id]
		if el.kind != .Element {
			continue
		}
		switch xml_local(el.ident) {
		case "pbmetallicdisplayproperties":
			gid := parse_u32_attr(el, "id") or_continue
			first_prop := u32(len(b.pbr_props))
			prop_count := u32(0)
			for pv in el.value {
				pid := pv.(xml.Element_ID) or_continue
				pel := doc.elements[pid]
				if pel.kind != .Element || xml_local(pel.ident) != "pbmetallic" {
					continue
				}
				metal, _ := parse_f32_attr(pel, "metallicness", 0)
				rough, _ := parse_f32_attr(pel, "roughness", 1)
				append(&b.pbr_props, Threemf_PBR_Prop{clamp(metal, 0, 1), clamp(rough, 0, 1)})
				prop_count += 1
			}
			append(&b.pbr_groups, Threemf_PBR_Group{id = gid, first_prop = first_prop, count = prop_count})
			pbr_count += 1

		case "texture2d":
			tid := parse_u32_attr(el, "id") or_continue
			path, has_path := xml_attr(el, "path")
			tex_id: Texture_Id
			if has_path && b.textures != nil {
				if img_bytes, ok := zip_extract(b.archive, b.entries, path); ok && len(img_bytes) > 0 {
					tex_id, _ = texture_decode(b.textures, img_bytes)
				}
			}
			append(&b.tex2ds, Threemf_Tex2D{id = tid, tex = tex_id})
			tex_count += 1
		}
	}
	b.parts[part_idx].first_pbr = first_pbr
	b.parts[part_idx].pbr_count = pbr_count
	b.parts[part_idx].first_tex = first_tex
	b.parts[part_idx].tex_count = tex_count

	// Sub-pass B: `<basematerials>`, `<colorgroup>`, and `<texture2dgroup>`.
	first_group := u32(len(b.groups))
	group_count := u32(0)
	for v in res_el.value {
		el_id := v.(xml.Element_ID) or_continue
		el := doc.elements[el_id]
		if el.kind != .Element {
			continue
		}
		switch xml_local(el.ident) {
		case "basematerials":
			gid := parse_u32_attr(el, "id") or_continue
			dp_id, _ := parse_u32_attr(el, "displaypropertiesid")
			first_mat := u32(len(b.materials))
			mat_count := u32(0)
			for mv in el.value {
				mid := mv.(xml.Element_ID) or_continue
				mel := doc.elements[mid]
				if mel.kind != .Element || xml_local(mel.ident) != "base" {
					continue
				}
				mat := MATERIAL_DEFAULT
				mat.roughness = 0.5
				if c_str, has_c := xml_attr(mel, "displaycolor"); has_c {
					if col, op, c_ok := parse_hex_color(c_str); c_ok {
						mat.color = col
						mat.opacity = op
						if op < 1 {
							mat.flags += {.Transparent}
						}
					}
				}
				if pbr, has_pbr := find_pbr_prop(b, &b.parts[part_idx], dp_id, mat_count); has_pbr {
					mat.metalness = pbr.metalness
					mat.roughness = pbr.roughness
				}
				append(&b.materials, mat)
				name, _ := xml_attr(mel, "name")
				append(&b.model.material_names, strings.clone(name))
				append(&b.material_ids, Model_Material_Id(len(b.model.material_names)))
				mat_count += 1
			}
			append(&b.groups, Threemf_Group{id = gid, kind = .Base_Materials, first_item = first_mat, count = mat_count})
			group_count += 1

		case "colorgroup":
			gid := parse_u32_attr(el, "id") or_continue
			dp_id, _ := parse_u32_attr(el, "displaypropertiesid")
			first_mat := u32(len(b.materials))
			mat_count := u32(0)
			for cv in el.value {
				cid := cv.(xml.Element_ID) or_continue
				cel := doc.elements[cid]
				if cel.kind != .Element || xml_local(cel.ident) != "color" {
					continue
				}
				mat := MATERIAL_DEFAULT
				mat.roughness = 0.5
				if c_str, has_c := xml_attr(cel, "color"); has_c {
					if col, op, c_ok := parse_hex_color(c_str); c_ok {
						mat.color = col
						mat.opacity = op
						if op < 1 {
							mat.flags += {.Transparent}
						}
					}
				}
				if pbr, has_pbr := find_pbr_prop(b, &b.parts[part_idx], dp_id, mat_count); has_pbr {
					mat.metalness = pbr.metalness
					mat.roughness = pbr.roughness
				}
				append(&b.materials, mat)
				append(&b.material_ids, 0)
				mat_count += 1
			}
			append(&b.groups, Threemf_Group{id = gid, kind = .Color_Group, first_item = first_mat, count = mat_count})
			group_count += 1

		case "texture2dgroup":
			gid := parse_u32_attr(el, "id") or_continue
			tid, _ := parse_u32_attr(el, "texid")
			tex := find_tex2d(b, &b.parts[part_idx], tid)
			first_uv := u32(len(b.uvs))
			uv_count := u32(0)
			for uv_v in el.value {
				uid := uv_v.(xml.Element_ID) or_continue
				uel := doc.elements[uid]
				if uel.kind != .Element || xml_local(uel.ident) != "tex2coord" {
					continue
				}
				u, _ := parse_f32_attr(uel, "u", 0)
				v_coord, _ := parse_f32_attr(uel, "v", 0)
				append(&b.uvs, Vec2{u, v_coord})
				uv_count += 1
			}
			append(&b.groups, Threemf_Group{id = gid, kind = .Texture_Group, tex = tex, first_item = first_uv, count = uv_count})
			group_count += 1
		}
	}
	b.parts[part_idx].first_group = first_group
	b.parts[part_idx].group_count = group_count

	// Sub-pass C: `<object>` elements (meshes and component assemblies).
	// Collect object elements first because a `<component p:path="...">` may
	// append new sub-parts to `b.parts` and `b.objects`.
	obj_els := make([dynamic]xml.Element_ID, 0, 16, context.temp_allocator)
	for v in res_el.value {
		el_id := v.(xml.Element_ID) or_continue
		el := doc.elements[el_id]
		if el.kind == .Element && xml_local(el.ident) == "object" {
			append(&obj_els, el_id)
		}
	}

	built_objs := make([dynamic]Threemf_Object, 0, len(obj_els), context.temp_allocator)
	unit_scale := b.parts[part_idx].unit_scale
	for obj_el_id in obj_els {
		el := doc.elements[obj_el_id]
		oid := parse_u32_attr(el, "id") or_continue
		if obj_type, has_type := xml_attr(el, "type"); has_type {
			if obj_type != "model" && obj_type != "other" {
				continue
			}
		}
		default_pid, _ := parse_u32_attr(el, "pid")
		default_pindex, has_default_pindex := parse_u32_attr(el, "pindex")

		mesh_idx := i32(-1)
		first_comp := u32(len(b.components))
		comp_count := u32(0)

		for ov in el.value {
			child_id := ov.(xml.Element_ID) or_continue
			child := doc.elements[child_id]
			if child.kind != .Element {
				continue
			}
			switch xml_local(child.ident) {
			case "mesh":
				mesh_idx = threemf_build_mesh(b, part_idx, doc, child, default_pid, default_pindex, has_default_pindex) or_return

			case "components":
				for cv in child.value {
					comp_id := cv.(xml.Element_ID) or_continue
					comp_el := doc.elements[comp_id]
					if comp_el.kind != .Element || xml_local(comp_el.ident) != "component" {
						continue
					}
					target_oid := parse_u32_attr(comp_el, "objectid") or_continue
					target_part := part_idx
					if p_path, has_p := xml_attr(comp_el, "path"); has_p && p_path != "" {
						target_part = threemf_load_part(b, p_path) or_return
					}
					xform := linalg.MATRIX4F32_IDENTITY
					if tf_str, has_tf := xml_attr(comp_el, "transform"); has_tf {
						xform = parse_threemf_transform(tf_str, unit_scale)
					}
					append(&b.components, Threemf_Component {
						part      = i32(target_part),
						object_id = target_oid,
						transform = xform,
					})
					comp_count += 1
				}
			}
		}
		name, _ := xml_attr(el, "name")
		append(&built_objs, Threemf_Object {
			id         = oid,
			name       = strings.clone(name, context.temp_allocator),
			mesh       = mesh_idx,
			first_comp = first_comp,
			comp_count = comp_count,
		})
	}

	first_obj := u32(len(b.objects))
	append(&b.objects, ..built_objs[:])
	b.parts[part_idx].first_object = first_obj
	b.parts[part_idx].object_count = u32(len(built_objs))
	return .None
}

@(private = "file")
Threemf_Tri :: struct {
	v:          [3]u32,
	uv:         [3]Vec2,
	material:   Material,
	material_id: Model_Material_Id,
	sort_key:   u64,
}

@(private = "file")
threemf_resolve_triangle :: proc(
	b: ^Threemf_Builder,
	part_idx: int,
	el: xml.Element,
	default_pid: u32,
	default_pindex: u32,
	has_default_pindex: bool,
) -> (mat: Material, uvs: [3]Vec2, key: u64, id: Model_Material_Id) {
	part := &b.parts[part_idx]
	pid, has_pid := parse_u32_attr(el, "pid")
	if !has_pid {
		pid = default_pid
	}
	p1, has_p1 := parse_u32_attr(el, "p1")
	if !has_p1 && has_default_pindex {
		p1 = default_pindex
		has_p1 = true
	}
	p2, has_p2 := parse_u32_attr(el, "p2")
	if !has_p2 {
		p2 = p1
	}
	p3, has_p3 := parse_u32_attr(el, "p3")
	if !has_p3 {
		p3 = p1
	}

	gi, found := find_group(b, part, pid)
	if !found || !has_p1 {
		mat = MATERIAL_DEFAULT
		mat.roughness = 0.5
		return mat, {}, 0, 0
	}
	g := b.groups[gi]
	switch g.kind {
	case .Base_Materials, .Color_Group:
		if p1 >= g.count {
			mat = MATERIAL_DEFAULT
			mat.roughness = 0.5
			return mat, {}, 0, 0
		}
		if g.kind == .Color_Group && (p2 != p1 || p3 != p1) && p2 < g.count && p3 < g.count {
			m1 := b.materials[g.first_item + p1]
			m2 := b.materials[g.first_item + p2]
			m3 := b.materials[g.first_item + p3]
			lin := (linear(m1.color) + linear(m2.color) + linear(m3.color)) * (1.0 / 3.0)
			mat = m1
			mat.color = linear_to_srgb_vec3(lin)
			mat.opacity = (m1.opacity + m2.opacity + m3.opacity) * (1.0 / 3.0)
			if mat.opacity < 1 {
				mat.flags += {.Transparent}
			}
			key = (u64(u32(gi) + 1) << 32) | u64(0x80000000 | (p1 ~ (p2 << 10) ~ (p3 << 20)))
			return mat, {}, key, 0
		}
		mat = b.materials[g.first_item + p1]
		key = (u64(u32(gi) + 1) << 32) | u64(p1)
		return mat, {}, key, b.material_ids[g.first_item + p1]

	case .Texture_Group:
		mat = MATERIAL_DEFAULT
		mat.roughness = 0.5
		mat.color_map = g.tex
		if p1 < g.count {
			uvs[0] = b.uvs[g.first_item + p1]
		}
		if p2 < g.count {
			uvs[1] = b.uvs[g.first_item + p2]
		}
		if p3 < g.count {
			uvs[2] = b.uvs[g.first_item + p3]
		}
		key = (u64(u32(gi) + 1) << 32)
		return mat, uvs, key, 0
	}
	mat = MATERIAL_DEFAULT
	mat.roughness = 0.5
	return mat, {}, 0, 0
}

@(private = "file")
threemf_build_mesh :: proc(
	b: ^Threemf_Builder,
	part_idx: int,
	doc: ^xml.Document,
	mesh_el: xml.Element,
	default_pid: u32,
	default_pindex: u32,
	has_default_pindex: bool,
) -> (mesh_idx: i32, err: Threemf_Error) {
	unit_scale := b.parts[part_idx].unit_scale
	positions := make([dynamic]Vec3, 0, 64, context.temp_allocator)
	tris := make([dynamic]Threemf_Tri, 0, 64, context.temp_allocator)

	for v in mesh_el.value {
		child_id := v.(xml.Element_ID) or_continue
		child := doc.elements[child_id]
		if child.kind != .Element {
			continue
		}
		switch xml_local(child.ident) {
		case "vertices":
			for vv in child.value {
				vid := vv.(xml.Element_ID) or_continue
				vel := doc.elements[vid]
				if vel.kind != .Element || xml_local(vel.ident) != "vertex" {
					continue
				}
				x, x_ok := parse_f32_attr(vel, "x")
				y, y_ok := parse_f32_attr(vel, "y")
				z, z_ok := parse_f32_attr(vel, "z")
				if !x_ok || !y_ok || !z_ok {
					return -1, .Invalid_Model
				}
				// 3MF Z-up (x, y, z) -> o3d Y-up (x, z, -y).
				append(&positions, Vec3{x * unit_scale, z * unit_scale, -y * unit_scale})
			}

		case "triangles":
			for tv in child.value {
				tid := tv.(xml.Element_ID) or_continue
				tel := doc.elements[tid]
				if tel.kind != .Element || xml_local(tel.ident) != "triangle" {
					continue
				}
				v1, ok1 := parse_u32_attr(tel, "v1")
				v2, ok2 := parse_u32_attr(tel, "v2")
				v3, ok3 := parse_u32_attr(tel, "v3")
				if !ok1 || !ok2 || !ok3 {
					return -1, .Invalid_Model
				}
				mat, uvs, key, id := threemf_resolve_triangle(b, part_idx, tel, default_pid, default_pindex, has_default_pindex)
				append(&tris, Threemf_Tri {
					v           = {v1, v2, v3},
					uv          = uvs,
					material    = mat,
					material_id = id,
					sort_key    = key,
				})
			}
		}
	}

	v_count := u32(len(positions))
	if v_count == 0 || len(tris) == 0 {
		return -1, .Invalid_Model
	}
	for t in tris {
		if t.v[0] >= v_count || t.v[1] >= v_count || t.v[2] >= v_count {
			return -1, .Invalid_Model
		}
	}

	// Group triangles by material key only when a mesh uses multiple materials.
	multi_mat := false
	first_key := tris[0].sort_key
	for t in tris[1:] {
		if t.sort_key != first_key {
			multi_mat = true
			break
		}
	}
	if multi_mat {
		slice.sort_by(tris[:], proc(a, b: Threemf_Tri) -> bool {
			return a.sort_key < b.sort_key
		})
	}

	// Precompute area-weighted and unit face normals + a flat per-vertex
	// triangle incidence list so shared vertices within a 60° crease angle
	// (`cos(60°) = 0.5`) receive smooth normals while hard edges stay sharp.
	COS_CREASE :: f32(0.5)
	total_tris := len(tris)
	tri_area_n := make([]Vec3, total_tris, context.temp_allocator)
	tri_unit_n := make([]Vec3, total_tris, context.temp_allocator)
	head := make([]i32, v_count, context.temp_allocator)
	slice.fill(head, -1)
	next_corner := make([]i32, total_tris * 3, context.temp_allocator)

	for t, ti in tris {
		p0 := positions[t.v[0]]
		p1 := positions[t.v[1]]
		p2 := positions[t.v[2]]
		c := linalg.cross(p1 - p0, p2 - p0)
		clen := linalg.length(c)
		tri_area_n[ti] = c
		tri_unit_n[ti] = c / clen if clen > 1e-12 else Vec3{0, 1, 0}
		for k in 0 ..< 3 {
			vi := t.v[k]
			ci := i32(ti * 3 + k)
			next_corner[ci] = head[vi]
			head[vi] = ci
		}
	}

	first_prim := u32(len(b.model.primitives))
	prim_count := u32(0)
	run_start := 0
	for run_start < total_tris {
		key := tris[run_start].sort_key
		run_end := run_start + 1
		for run_end < total_tris && tris[run_end].sort_key == key {
			run_end += 1
		}
		run := tris[run_start:run_end]
		nt := len(run)
		verts := make([]Vertex, nt * 3, context.temp_allocator)
		indices := make([]u32, nt * 3, context.temp_allocator)
		for t, local_ti in run {
			ti := run_start + local_ti
			fn := tri_unit_n[ti]
			base := local_ti * 3
			for k in 0 ..< 3 {
				vi := t.v[k]
				acc := Vec3{}
				for ci := head[vi]; ci >= 0; ci = next_corner[ci] {
					other_ti := int(ci) / 3
					if linalg.dot(fn, tri_unit_n[other_ti]) >= COS_CREASE {
						acc += tri_area_n[other_ti]
					}
				}
				alen := linalg.length(acc)
				vn := acc / alen if alen > 1e-12 else fn
				verts[base + k] = {position = positions[vi], normal = vn, uv = t.uv[k]}
				indices[base + k] = u32(base + k)
			}
		}
		geo_id := geometry_add(b.geos, verts, indices)
		append(&b.model.primitives, Model_Primitive {
			geometry = geo_id,
			material    = run[0].material,
			material_id = run[0].material_id,
		})
		prim_count += 1
		run_start = run_end
	}

	mesh_idx = i32(len(b.model.meshes))
	append(&b.model.meshes, Model_Mesh {
		first_primitive = first_prim,
		primitive_count = prim_count,
	})
	return mesh_idx, .None
}

// Emits topologically ordered `Model_Node`s (`parent < i`) for `(part_idx, object_id)`.
@(private = "file")
threemf_emit_node :: proc(b: ^Threemf_Builder, part_idx: i32, object_id: u32, parent: i32, transform: Mat4, depth: int) {
	if depth > 64 || part_idx < 0 || int(part_idx) >= len(b.parts) {
		return
	}
	part := b.parts[part_idx]
	obj_idx := -1
	for i in 0 ..< int(part.object_count) {
		oi := int(part.first_object) + i
		if b.objects[oi].id == object_id {
			obj_idx = oi
			break
		}
	}
	if obj_idx < 0 {
		return
	}
	obj := b.objects[obj_idx]
	node_idx := i32(len(b.model.nodes))
	append(&b.model.nodes, Model_Node {
		name       = strings.clone(obj.name),
		parent     = parent,
		mesh       = obj.mesh,
		skin       = -1,
		light      = -1,
		has_matrix = true,
		scale      = {1, 1, 1},
		rotation   = linalg.QUATERNIONF32_IDENTITY,
		local      = transform,
	})
	for comp in b.components[obj.first_comp:][:obj.comp_count] {
		threemf_emit_node(b, comp.part, comp.object_id, node_idx, comp.transform, depth + 1)
	}
}
