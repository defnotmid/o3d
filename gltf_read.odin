package o3d

import "core:encoding/base64"
import "core:encoding/json"
import "core:math"
import "core:mem"
import "core:mem/virtual"
import "core:os"
import "core:slice"
import "core:strings"

// glTF 2.0 reading: `gltf_read` turns a `.gltf` (JSON with external,
// data-URI or GLB buffers) or `.glb` into a `Gltf_Document`, the file's own
// tables as flat arrays that refer to each other by index (-1: none), its
// buffers loaded and the whole checked (every index in range, every accessor
// and view inside its data, counts that agree, no node with two parents or
// in a cycle) so that reading it later needs no checks. gltf.odin builds a
// `Model` from it. The JSON is parsed (core:encoding/json) into the
// document's arena with the tables and the buffers read from files and data
// URIs; a GLB's binary chunk is borrowed from the bytes read, so they must
// outlive the document. What o3d does not use (cameras, samplers, names
// other than animations', extras other than `targetNames`, extensions other
// than those o3d implements) is skipped. Bounded by `Limits` (limits.odin).

Gltf_Range :: struct {
	first, count: u32,
}

Gltf_Component :: enum u8 {
	Invalid,
	I8,
	U8,
	I16,
	U16,
	U32,
	F32,
}

Gltf_Type :: enum u8 {
	Invalid,
	Scalar,
	Vec2,
	Vec3,
	Vec4,
	Mat2,
	Mat3,
	Mat4,
}

Gltf_Buffer_View :: struct {
	buffer:         i32,
	offset, length: int,
	// 0: elements tightly packed.
	stride:         int,
}

// A view stored compressed (EXT_meshopt_compression): its bytes decode
// from `length` bytes at `offset` in `buffer` into the view, which is its
// own span of the decoded buffer (the last in `buffers`).
Gltf_Meshopt :: struct {
	view, buffer:   i32,
	offset, length: int,
	count, stride:  int,
	mode:           Meshopt_Mode,
	filter:         Meshopt_Filter,
}

Gltf_Accessor :: struct {
	view:       i32, // -1: all zeros (or only the sparse values)
	sparse:     i32, // -1, or into `Gltf_Document.sparse`
	offset:     int,
	count:      int,
	// The view's byteStride, or the element size.
	stride:     int,
	component:  Gltf_Component,
	type:       Gltf_Type,
	normalized: bool,
}

Gltf_Sparse :: struct {
	count:                          int,
	indices_view, values_view:      i32,
	indices_offset, values_offset:  int,
	indices_component:              Gltf_Component,
}

Gltf_Semantic :: enum u8 {
	Other,
	Position,
	Normal,
	Tangent,
	Texcoord,
	Color,
	Joints,
	Weights,
}

Gltf_Attribute :: struct {
	semantic: Gltf_Semantic,
	set:      u8, // TEXCOORD_<set>
	accessor: i32,
}

Gltf_Primitive :: struct {
	// 0 points, 1 lines, 2 line loop, 3 line strip, 4 triangles, 5 strip,
	// 6 fan; anything else is invalid.
	mode:                        int,
	indices, material:           i32,
	// Into `attributes` (all of them) and `targets`.
	attributes, targets:         Gltf_Range,
	// The attributes o3d reads.
	position, normal, joints, weights: i32,
	texcoord:                    [2]i32,
	color:                       i32, // COLOR_0
}

// A morph target: its attributes, and the POSITION and NORMAL deltas.
Gltf_Target :: struct {
	attributes:       Gltf_Range,
	position, normal: i32,
}

Gltf_Mesh :: struct {
	primitives:   Gltf_Range,
	// Into `floats` and `strings` (extras.targetNames).
	weights:      Gltf_Range,
	target_names: Gltf_Range,
}

Gltf_Texture_Ref :: struct {
	texture:           i32,
	tex_coord:         i32,
	// normalTexture.scale or occlusionTexture.strength.
	scale:             f32,
	// KHR_texture_transform.
	has_transform:     bool,
	has_tex_coord:     bool,
	transform_coord:   i32,
	offset:            [2]f32,
	rotation:          f32,
	transform_scale:   [2]f32,
}

Gltf_Alpha_Mode :: enum u8 {
	Opaque,
	Mask,
	Blend,
}

Gltf_Material :: struct {
	name:                 string,
	has_pbr:              bool,
	base_color:           [4]f32,
	metallic, roughness:  f32,
	base_color_texture:   Gltf_Texture_Ref,
	metal_rough_texture:  Gltf_Texture_Ref,
	normal_texture:       Gltf_Texture_Ref,
	occlusion_texture:    Gltf_Texture_Ref,
	emissive_texture:     Gltf_Texture_Ref,
	emissive:             [3]f32,
	// KHR_materials_emissive_strength (1 without).
	emissive_strength:    f32,
	alpha_mode:           Gltf_Alpha_Mode,
	alpha_cutoff:         f32,
	double_sided:         bool,
	// KHR_materials_unlit.
	unlit:                bool,
	// KHR_materials_ior (1.5 without), _clearcoat, _sheen, _transmission
	// and _volume factors (their textures are not read); an infinite
	// attenuation distance is 0.
	ior:                  f32,
	clearcoat:            f32,
	clearcoat_roughness:  f32,
	sheen_color:          [3]f32,
	sheen_roughness:      f32,
	transmission:         f32,
	thickness:            f32,
	attenuation_color:    [3]f32,
	attenuation_distance: f32,
	// Into `strings`: the material's extensions o3d does not read.
	unread_extensions:    Gltf_Range,
}

Gltf_Image :: struct {
	view: i32,
	uri:  string,
}

Gltf_Node :: struct {
	name:                          string,
	parent, first_child, next:     i32,
	mesh, skin, light:             i32,
	has_matrix:                    bool,
	has_translation, has_rotation: bool,
	has_scale:                     bool,
	// Column-major, as in the file.
	matrix_:                       [16]f32,
	translation:                   [3]f32,
	rotation:                      [4]f32,
	scale:                         [3]f32,
	// Into `floats`.
	weights:                       Gltf_Range,
}

Gltf_Skin :: struct {
	// Into `node_lists`.
	joints:                Gltf_Range,
	inverse_bind_matrices: i32,
}

Gltf_Interpolation :: enum u8 {
	Linear,
	Step,
	Cubic_Spline,
}

Gltf_Path :: enum u8 {
	Invalid,
	Translation,
	Rotation,
	Scale,
	Weights,
}

Gltf_Sampler :: struct {
	input, output: i32,
	interpolation: Gltf_Interpolation,
}

Gltf_Channel :: struct {
	// Into `samplers` (the document's, not the animation's).
	sampler: i32,
	node:    i32,
	path:    Gltf_Path,
}

Gltf_Animation :: struct {
	name:               string,
	channels, samplers: Gltf_Range,
}

Gltf_Light_Kind :: enum u8 {
	Invalid,
	Directional,
	Point,
	Spot,
}

// KHR_lights_punctual.
Gltf_Light :: struct {
	kind:              Gltf_Light_Kind,
	color:             [3]f32,
	intensity, range:  f32,
	inner_cone, outer_cone: f32,
}

Gltf_Document :: struct {
	// Each `byteLength` bytes (nil: a buffer without data).
	buffers:             [][]u8,
	buffer_lengths:      []int,
	views:               []Gltf_Buffer_View,
	meshopt:             [dynamic]Gltf_Meshopt,
	accessors:           []Gltf_Accessor,
	sparse:              [dynamic]Gltf_Sparse,
	images:              []Gltf_Image,
	// Each texture's image (`source`), or -1.
	textures:            []i32,
	materials:           []Gltf_Material,
	meshes:              []Gltf_Mesh,
	primitives:          [dynamic]Gltf_Primitive,
	attributes:          [dynamic]Gltf_Attribute,
	targets:             [dynamic]Gltf_Target,
	nodes:               []Gltf_Node,
	skins:               []Gltf_Skin,
	animations:          []Gltf_Animation,
	samplers:            [dynamic]Gltf_Sampler,
	channels:            [dynamic]Gltf_Channel,
	lights:              []Gltf_Light,
	// Into `node_lists`: each scene's root nodes.
	scenes:              []Gltf_Range,
	// The default scene, or -1.
	scene:               i32,
	node_lists:          [dynamic]i32,
	floats:              [dynamic]f32,
	strings:             [dynamic]string,
	extensions_required: []string,
	extensions_used:     []string,
	// Set with `.Unsupported_Extension`: the first required extension o3d
	// does not implement.
	unsupported_extension: string,
	// What the read used of `limits` and which it reached; building a
	// model goes on counting in it.
	budget:              Budget,
	arena:               ^virtual.Arena,
}

gltf_document_destroy :: proc(doc: ^Gltf_Document) {
	if doc.arena != nil {
		virtual.arena_destroy(doc.arena)
		free(doc.arena)
	}
	doc^ = {}
}

// The extensions o3d implements; any other in extensionsRequired fails the
// file (`.Unsupported_Extension`).
@(rodata)
GLTF_EXTENSIONS := [?]string {
	"KHR_materials_unlit",
	"KHR_materials_emissive_strength",
	"KHR_texture_transform",
	"KHR_mesh_quantization",
	"KHR_lights_punctual",
	"EXT_meshopt_compression",
	"EXT_texture_webp",
	"KHR_materials_ior",
	"KHR_materials_clearcoat",
	"KHR_materials_sheen",
	"KHR_materials_transmission",
	"KHR_materials_volume",
}

// The material extensions o3d reads (others are named in `Model.warnings`).
@(rodata)
GLTF_MATERIAL_EXTENSIONS := [?]string {
	"KHR_materials_unlit",
	"KHR_materials_emissive_strength",
	"KHR_materials_ior",
	"KHR_materials_clearcoat",
	"KHR_materials_sheen",
	"KHR_materials_transmission",
	"KHR_materials_volume",
}

// Reads a `.gltf` or `.glb` in memory into `doc` (destroy it, error or not):
// external buffers resolve against `base_dir` (the working directory when
// empty). A GLB's binary chunk stays in `bytes`.
gltf_read :: proc(bytes: []u8, base_dir := "", limits: ^Limits = nil) -> (doc: Gltf_Document, err: Gltf_Error) {
	budget_init(&doc.budget, limits)
	doc.scene = -1
	if len(bytes) == 0 {
		return doc, .Empty
	}
	doc.arena = new(virtual.Arena)
	if virtual.arena_init_growing(doc.arena) != nil {
		free(doc.arena)
		doc.arena = nil
		return doc, .Out_Of_Memory
	}
	r := Gltf_Reader{doc = &doc, a = virtual.arena_allocator(doc.arena)}
	json_bytes, bin, has_bin := gltf_split_glb(bytes) or_return
	if json_depth(json_bytes) > doc.budget.limits[.Depth] {
		doc.budget.hit += {.Depth}
		return doc, .Limit
	}
	// The JSON tree lives in a scratch arena of its own, gone on return;
	// what the document keeps of it is copied.
	scratch: virtual.Arena
	if virtual.arena_init_growing(&scratch, uint(clamp(len(json_bytes) * 16, 1 << 20, 1 << 30))) != nil {
		return doc, .Out_Of_Memory
	}
	defer virtual.arena_destroy(&scratch)
	root_value, jerr := json.parse(json_bytes, .JSON, false, virtual.arena_allocator(&scratch))
	if jerr == .Out_Of_Memory {
		return doc, .Out_Of_Memory
	} else if jerr != nil {
		return doc, .Invalid_Json
	}
	root, is_object := root_value.(json.Object)
	if !is_object {
		return doc, .Invalid_Gltf
	}
	gltf_parse_root(&r, root)
	if r.err != nil {
		return doc, r.err
	}
	for ext in doc.extensions_required {
		supported := false
		for s in GLTF_EXTENSIONS {
			supported ||= ext == s
		}
		if !supported {
			doc.unsupported_extension = ext
			return doc, .Unsupported_Extension
		}
	}
	gltf_load_buffers(&r, root, bin, has_bin, base_dir)
	if r.err != nil {
		return doc, r.err
	}
	return doc, gltf_validate(&doc)
}

// ---- GLB and JSON -----------------------------------------------------------------------

// The JSON and the binary chunk of a GLB, or all of `bytes` for a `.gltf`.
@(private = "file")
gltf_split_glb :: proc(bytes: []u8) -> (json_bytes, bin: []u8, has_bin: bool, err: Gltf_Error) {
	GLB_MAGIC :: 0x46546C67 // "glTF"
	JSON_CHUNK :: 0x4E4F534A
	BIN_CHUNK :: 0x004E4942
	if len(bytes) < 12 {
		return nil, nil, false, .Data_Too_Short
	}
	u32_at :: proc "contextless" (b: []u8, at: int) -> int {
		return int(u32(b[at]) | u32(b[at + 1]) << 8 | u32(b[at + 2]) << 16 | u32(b[at + 3]) << 24)
	}
	if u32_at(bytes, 0) != GLB_MAGIC {
		return bytes, nil, false, nil
	}
	if version := u32_at(bytes, 4); version != 2 {
		return nil, nil, false, .Legacy_Gltf if version < 2 else .Unknown_Format
	}
	size := len(bytes)
	if u32_at(bytes, 8) > size || size < 20 {
		return nil, nil, false, .Data_Too_Short
	}
	json_len := u32_at(bytes, 12)
	if json_len > size - 20 {
		return nil, nil, false, .Data_Too_Short
	}
	if u32_at(bytes, 16) != JSON_CHUNK {
		return nil, nil, false, .Unknown_Format
	}
	json_bytes = bytes[20:][:json_len]
	rest := size - 20 - json_len
	if rest >= 8 {
		bin_at := 20 + json_len
		bin_len := u32_at(bytes, bin_at)
		if bin_len > rest - 8 {
			return nil, nil, false, .Data_Too_Short
		}
		if u32_at(bytes, bin_at + 4) != BIN_CHUNK {
			return nil, nil, false, .Unknown_Format
		}
		bin = bytes[bin_at + 8:][:bin_len]
		has_bin = true
	}
	return
}

// The deepest nesting of arrays and objects (json.parse recurses into them).
@(private = "file")
json_depth :: proc "contextless" (s: []u8) -> int {
	depth, deepest := 0, 0
	in_string, escaped := false, false
	for c in s {
		if in_string {
			if escaped {
				escaped = false
			} else if c == '\\' {
				escaped = true
			} else if c == '"' {
				in_string = false
			}
			continue
		}
		switch c {
		case '"':
			in_string = true
		case '[', '{':
			depth += 1
			deepest = max(deepest, depth)
		case ']', '}':
			depth -= 1
		}
	}
	return deepest
}

@(private = "file")
Gltf_Reader :: struct {
	doc: ^Gltf_Document,
	a:   mem.Allocator,
	// The first error met (later ones are not looked for).
	err: Gltf_Error,
}

@(private = "file")
fail :: proc "contextless" (r: ^Gltf_Reader, err: Gltf_Error = .Invalid_Gltf) {
	if r.err == nil {
		r.err = err
	}
}

// `o[key]` as an array; a glTF 1.0 file has objects where 2.0 has arrays.
@(private = "file")
get_array :: proc(r: ^Gltf_Reader, o: json.Object, key: string) -> json.Array {
	v, ok := o[key]
	if !ok {
		return nil
	}
	#partial switch a in v {
	case json.Array:
		return a
	case json.Object:
		fail(r, .Legacy_Gltf)
	case:
		fail(r)
	}
	return nil
}

@(private = "file")
get_object :: proc(r: ^Gltf_Reader, o: json.Object, key: string) -> (json.Object, bool) {
	v, ok := o[key]
	if !ok {
		return nil, false
	}
	obj, is_object := v.(json.Object)
	if !is_object {
		fail(r)
	}
	return obj, is_object
}

@(private = "file")
as_object :: proc(r: ^Gltf_Reader, v: json.Value) -> json.Object {
	obj, is_object := v.(json.Object)
	if !is_object {
		fail(r)
	}
	return obj
}

@(private = "file")
get_number :: proc(r: ^Gltf_Reader, o: json.Object, key: string, default: f64) -> f64 {
	v, ok := o[key]
	if !ok {
		return default
	}
	n, is_number := v.(json.Float)
	if !is_number {
		fail(r)
		return default
	}
	return n
}

@(private = "file")
get_f32 :: proc(r: ^Gltf_Reader, o: json.Object, key: string, default: f32) -> f32 {
	return f32(get_number(r, o, key, f64(default)))
}

// A size or offset; negative is 0, as cgltf reads them.
@(private = "file")
get_size :: proc(r: ^Gltf_Reader, o: json.Object, key: string) -> int {
	n := get_number(r, o, key, 0)
	if !(n < 1 << 62) {
		fail(r)
		return 0
	}
	return max(int(n), 0)
}

// An index into a table of `count`, or -1 when absent.
@(private = "file")
get_index :: proc(r: ^Gltf_Reader, o: json.Object, key: string, count: int) -> i32 {
	v, ok := o[key]
	if !ok {
		return -1
	}
	return as_index(r, v, count)
}

@(private = "file")
as_index :: proc(r: ^Gltf_Reader, v: json.Value, count: int) -> i32 {
	n, is_number := v.(json.Float)
	if !is_number || !(n >= 0 && n < f64(count)) || n != math.trunc(n) {
		fail(r)
		return -1
	}
	return i32(n)
}

@(private = "file")
get_int :: proc(r: ^Gltf_Reader, o: json.Object, key: string, default: int) -> int {
	n := get_number(r, o, key, f64(default))
	if !(abs(n) < 1 << 31) {
		fail(r)
		return default
	}
	return int(n)
}

@(private = "file")
get_bool :: proc(r: ^Gltf_Reader, o: json.Object, key: string) -> bool {
	v, ok := o[key]
	if !ok {
		return false
	}
	b, is_bool := v.(json.Boolean)
	if !is_bool {
		fail(r)
	}
	return b
}

@(private = "file")
get_string :: proc(r: ^Gltf_Reader, o: json.Object, key: string) -> (string, bool) {
	v, ok := o[key]
	if !ok {
		return "", false
	}
	s, is_string := v.(json.String)
	if !is_string {
		fail(r)
	}
	return s, is_string
}

// `o[key]` as an array of strings, cloned.
@(private = "file")
get_strings :: proc(r: ^Gltf_Reader, o: json.Object, key: string) -> []string {
	a := get_array(r, o, key)
	out := make([]string, len(a), r.a)
	for e, i in a {
		s, is_string := e.(json.String)
		if !is_string {
			fail(r)
		}
		out[i] = strings.clone(s, r.a)
	}
	return out
}

// `o.name`, cloned ("" without one).
@(private = "file")
get_name :: proc(r: ^Gltf_Reader, o: json.Object) -> string {
	name, _ := get_string(r, o, "name")
	return strings.clone(name, r.a)
}

// `o[key]` as exactly `len(out)` numbers; whether it is there.
@(private = "file")
get_floats :: proc(r: ^Gltf_Reader, o: json.Object, key: string, out: []f32) -> bool {
	v, ok := o[key]
	if !ok {
		return false
	}
	a, is_array := v.(json.Array)
	if !is_array || len(a) != len(out) {
		fail(r)
		return false
	}
	for e, i in a {
		n, is_number := e.(json.Float)
		if !is_number {
			fail(r)
			return false
		}
		out[i] = f32(n)
	}
	return true
}

// Appends `o[key]`'s numbers to `floats`.
@(private = "file")
get_float_list :: proc(r: ^Gltf_Reader, o: json.Object, key: string) -> (rng: Gltf_Range) {
	a := get_array(r, o, key)
	rng.first = u32(len(r.doc.floats))
	for e in a {
		n, is_number := e.(json.Float)
		if !is_number {
			fail(r)
			return
		}
		append(&r.doc.floats, f32(n))
	}
	rng.count = u32(len(a))
	return
}

// Appends `o[key]`'s node indices to `node_lists`.
@(private = "file")
get_node_list :: proc(r: ^Gltf_Reader, o: json.Object, key: string) -> (rng: Gltf_Range) {
	a := get_array(r, o, key)
	rng.first = u32(len(r.doc.node_lists))
	for e in a {
		append(&r.doc.node_lists, as_index(r, e, len(r.doc.nodes)))
	}
	rng.count = u32(len(a))
	return
}

@(private = "file")
component_of :: proc(r: ^Gltf_Reader, o: json.Object, key: string) -> Gltf_Component {
	switch get_int(r, o, key, 0) {
	case 5120:
		return .I8
	case 5121:
		return .U8
	case 5122:
		return .I16
	case 5123:
		return .U16
	case 5125:
		return .U32
	case 5126:
		return .F32
	}
	return .Invalid
}

// ---- the root ---------------------------------------------------------------------------

@(private = "file")
gltf_parse_root :: proc(r: ^Gltf_Reader, root: json.Object) {
	doc := r.doc
	context.allocator = r.a
	if asset, ok := get_object(r, root, "asset"); ok {
		if version, has := get_string(r, asset, "version"); has {
			if v, _ := strconv_prefix_f64(version); v < 2 {
				fail(r, .Legacy_Gltf)
			}
		}
	}
	doc.extensions_required = get_strings(r, root, "extensionsRequired")
	doc.extensions_used = get_strings(r, root, "extensionsUsed")

	// Table lengths first: entries refer to each other in any order.
	buffers := get_array(r, root, "buffers")
	views := get_array(r, root, "bufferViews")
	accessors := get_array(r, root, "accessors")
	images := get_array(r, root, "images")
	textures := get_array(r, root, "textures")
	materials := get_array(r, root, "materials")
	meshes := get_array(r, root, "meshes")
	nodes := get_array(r, root, "nodes")
	skins := get_array(r, root, "skins")
	animations := get_array(r, root, "animations")
	scenes := get_array(r, root, "scenes")
	lights: json.Array
	if exts, ok := get_object(r, root, "extensions"); ok {
		if punctual, has := get_object(r, exts, "KHR_lights_punctual"); has {
			lights = get_array(r, punctual, "lights")
		}
	}
	if r.err != nil {
		return
	}
	if budget_over(&doc.budget, .Gltf_Nodes, len(nodes)) || budget_over(&doc.budget, .Gltf_Accessors, len(accessors)) {
		fail(r, .Limit)
		return
	}
	doc.views = make([]Gltf_Buffer_View, len(views))
	doc.accessors = make([]Gltf_Accessor, len(accessors))
	doc.images = make([]Gltf_Image, len(images))
	doc.textures = make([]i32, len(textures))
	doc.materials = make([]Gltf_Material, len(materials))
	doc.meshes = make([]Gltf_Mesh, len(meshes))
	doc.nodes = make([]Gltf_Node, len(nodes))
	doc.skins = make([]Gltf_Skin, len(skins))
	doc.animations = make([]Gltf_Animation, len(animations))
	doc.scenes = make([]Gltf_Range, len(scenes))
	doc.lights = make([]Gltf_Light, len(lights))
	doc.meshopt = make([dynamic]Gltf_Meshopt)
	doc.sparse = make([dynamic]Gltf_Sparse)
	doc.primitives = make([dynamic]Gltf_Primitive)
	doc.attributes = make([dynamic]Gltf_Attribute)
	doc.targets = make([dynamic]Gltf_Target)
	doc.samplers = make([dynamic]Gltf_Sampler)
	doc.channels = make([dynamic]Gltf_Channel)
	doc.node_lists = make([dynamic]i32)
	doc.floats = make([dynamic]f32)
	doc.strings = make([dynamic]string)

	decoded := 0
	for v, i in views {
		o := as_object(r, v)
		doc.views[i] = {
			buffer = get_index(r, o, "buffer", len(buffers)),
			offset = get_size(r, o, "byteOffset"),
			length = get_size(r, o, "byteLength"),
			stride = get_size(r, o, "byteStride"),
		}
		if doc.views[i].buffer < 0 {
			fail(r)
		}
		if exts, has_exts := get_object(r, o, "extensions"); has_exts {
			if mo, has := get_object(r, exts, "EXT_meshopt_compression"); has {
				parse_meshopt(r, mo, i32(i), len(buffers), &decoded)
			}
		}
	}
	// The decoded views' buffer after the file's, when there are any.
	doc.buffers = make([][]u8, len(buffers) + (1 if len(doc.meshopt) > 0 else 0))
	doc.buffer_lengths = make([]int, len(doc.buffers))
	for v, i in buffers {
		doc.buffer_lengths[i] = get_size(r, as_object(r, v), "byteLength")
	}
	if len(doc.meshopt) > 0 {
		doc.buffer_lengths[len(buffers)] = decoded
	}
	for v, i in accessors {
		parse_accessor(r, as_object(r, v), &doc.accessors[i])
	}
	for v, i in images {
		o := as_object(r, v)
		doc.images[i].view = get_index(r, o, "bufferView", len(views))
		uri, _ := get_string(r, o, "uri")
		doc.images[i].uri = strings.clone(uri, r.a)
	}
	for v, i in textures {
		o := as_object(r, v)
		doc.textures[i] = get_index(r, o, "source", len(images))
		// WebP decodes like any image: its source over the fallback.
		if exts, has_exts := get_object(r, o, "extensions"); has_exts {
			if webp, has := get_object(r, exts, "EXT_texture_webp"); has {
				doc.textures[i] = get_index(r, webp, "source", len(images))
			}
		}
	}
	for v, i in materials {
		parse_material(r, as_object(r, v), &doc.materials[i])
	}
	for v, i in meshes {
		parse_mesh(r, as_object(r, v), &doc.meshes[i])
	}
	for &n in doc.nodes {
		n.parent, n.first_child, n.next = -1, -1, -1
	}
	for v, i in nodes {
		parse_node(r, as_object(r, v), i32(i))
	}
	for v, i in skins {
		o := as_object(r, v)
		doc.skins[i] = {
			joints                = get_node_list(r, o, "joints"),
			inverse_bind_matrices = get_index(r, o, "inverseBindMatrices", len(accessors)),
		}
	}
	for v, i in animations {
		parse_animation(r, as_object(r, v), &doc.animations[i])
	}
	for v, i in lights {
		parse_light(r, as_object(r, v), &doc.lights[i])
	}
	for v, i in scenes {
		doc.scenes[i] = get_node_list(r, as_object(r, v), "nodes")
		for root_node in doc.node_lists[doc.scenes[i].first:][:doc.scenes[i].count] {
			if root_node >= 0 && doc.nodes[root_node].parent >= 0 {
				fail(r)
			}
		}
	}
	doc.scene = get_index(r, root, "scene", len(scenes))
}

// The number a string starts with (atof's reading), and whether it has one.
@(private = "file")
strconv_prefix_f64 :: proc(s: string) -> (f64, bool) {
	n, sign, frac, seen := 0.0, 1.0, 0.0, false
	i := 0
	if i < len(s) && (s[i] == '-' || s[i] == '+') {
		sign = -1 if s[i] == '-' else 1
		i += 1
	}
	for ; i < len(s) && s[i] >= '0' && s[i] <= '9'; i += 1 {
		n = n * 10 + f64(s[i] - '0')
		seen = true
	}
	if i < len(s) && s[i] == '.' {
		scale := 0.1
		for i += 1; i < len(s) && s[i] >= '0' && s[i] <= '9'; i += 1 {
			frac += f64(s[i] - '0') * scale
			scale *= 0.1
			seen = true
		}
	}
	return sign * (n + frac), seen
}

@(private = "file")
parse_accessor :: proc(r: ^Gltf_Reader, o: json.Object, acc: ^Gltf_Accessor) {
	views := len(r.doc.views)
	acc^ = {
		view       = get_index(r, o, "bufferView", views),
		sparse     = -1,
		offset     = get_size(r, o, "byteOffset"),
		count      = get_size(r, o, "count"),
		component  = component_of(r, o, "componentType"),
		normalized = get_bool(r, o, "normalized"),
	}
	if t, ok := get_string(r, o, "type"); ok {
		switch t {
		case "SCALAR":
			acc.type = .Scalar
		case "VEC2":
			acc.type = .Vec2
		case "VEC3":
			acc.type = .Vec3
		case "VEC4":
			acc.type = .Vec4
		case "MAT2":
			acc.type = .Mat2
		case "MAT3":
			acc.type = .Mat3
		case "MAT4":
			acc.type = .Mat4
		}
	}
	sp, has_sparse := get_object(r, o, "sparse")
	if !has_sparse {
		return
	}
	s := Gltf_Sparse{count = get_size(r, sp, "count"), indices_view = -1, values_view = -1}
	if ind, ok := get_object(r, sp, "indices"); ok {
		s.indices_view = get_index(r, ind, "bufferView", views)
		s.indices_offset = get_size(r, ind, "byteOffset")
		s.indices_component = component_of(r, ind, "componentType")
	}
	if val, ok := get_object(r, sp, "values"); ok {
		s.values_view = get_index(r, val, "bufferView", views)
		s.values_offset = get_size(r, val, "byteOffset")
	}
	if s.indices_view < 0 || s.values_view < 0 {
		fail(r)
	}
	acc.sparse = i32(len(r.doc.sparse))
	append(&r.doc.sparse, s)
}

// A compressed view's source and how it decodes; the view becomes the next
// span of the decoded buffer (index `buffers`, `decoded` bytes so far,
// spans 4-byte aligned).
@(private = "file")
parse_meshopt :: proc(r: ^Gltf_Reader, o: json.Object, view: i32, buffers: int, decoded: ^int) {
	m := Gltf_Meshopt {
		view   = view,
		buffer = get_index(r, o, "buffer", buffers),
		offset = get_size(r, o, "byteOffset"),
		length = get_size(r, o, "byteLength"),
		count  = get_size(r, o, "count"),
		stride = get_size(r, o, "byteStride"),
	}
	mode, _ := get_string(r, o, "mode")
	switch mode {
	case "ATTRIBUTES":
		m.mode = .Attributes
	case "TRIANGLES":
		m.mode = .Triangles
	case "INDICES":
		m.mode = .Indices
	case:
		fail(r)
	}
	filter, has_filter := get_string(r, o, "filter")
	switch filter {
	case "NONE":
	case "OCTAHEDRAL":
		m.filter = .Octahedral
	case "QUATERNION":
		m.filter = .Quaternion
	case "EXPONENTIAL":
		m.filter = .Exponential
	case:
		if has_filter {
			fail(r)
		}
	}
	if m.buffer < 0 || m.stride <= 0 || m.stride > 256 || m.filter != .None && m.mode != .Attributes {
		fail(r)
	}
	size := m.count * m.stride
	if r.err != nil || !budget_take(&r.doc.budget, .Gltf_Buffer_Bytes, size) {
		fail(r, .Limit if r.err == nil else r.err)
		return
	}
	v := &r.doc.views[view]
	v.buffer, v.offset, v.length = i32(buffers), decoded^, size
	decoded^ = (decoded^ + size + 3) & ~int(3)
	append(&r.doc.meshopt, m)
}

@(private = "file")
parse_texture_ref :: proc(r: ^Gltf_Reader, o: json.Object, key: string, ref: ^Gltf_Texture_Ref) {
	ref^ = {texture = -1, scale = 1, transform_scale = 1}
	t, ok := get_object(r, o, key)
	if !ok {
		return
	}
	ref.texture = get_index(r, t, "index", len(r.doc.textures))
	ref.tex_coord = i32(get_int(r, t, "texCoord", 0))
	if _, has_scale := t["scale"]; has_scale {
		ref.scale = get_f32(r, t, "scale", 1)
	}
	if _, has_strength := t["strength"]; has_strength {
		ref.scale = get_f32(r, t, "strength", 1)
	}
	exts, has_exts := get_object(r, t, "extensions")
	xf, has_xf := get_object(r, exts, "KHR_texture_transform")
	if !has_exts || !has_xf {
		return
	}
	ref.has_transform = true
	get_floats(r, xf, "offset", ref.offset[:])
	ref.rotation = get_f32(r, xf, "rotation", 0)
	get_floats(r, xf, "scale", ref.transform_scale[:])
	if _, has := xf["texCoord"]; has {
		ref.has_tex_coord = true
		ref.transform_coord = i32(get_int(r, xf, "texCoord", 0))
	}
}

@(private = "file")
parse_material :: proc(r: ^Gltf_Reader, o: json.Object, m: ^Gltf_Material) {
	m^ = {name = get_name(r, o), base_color = 1, metallic = 1, roughness = 1, emissive_strength = 1, alpha_cutoff = 0.5, ior = 1.5, attenuation_color = 1}
	if pbr, ok := get_object(r, o, "pbrMetallicRoughness"); ok {
		m.has_pbr = true
		get_floats(r, pbr, "baseColorFactor", m.base_color[:])
		m.metallic = get_f32(r, pbr, "metallicFactor", 1)
		m.roughness = get_f32(r, pbr, "roughnessFactor", 1)
		parse_texture_ref(r, pbr, "baseColorTexture", &m.base_color_texture)
		parse_texture_ref(r, pbr, "metallicRoughnessTexture", &m.metal_rough_texture)
	} else {
		parse_texture_ref(r, nil, "", &m.base_color_texture)
		parse_texture_ref(r, nil, "", &m.metal_rough_texture)
	}
	parse_texture_ref(r, o, "normalTexture", &m.normal_texture)
	parse_texture_ref(r, o, "occlusionTexture", &m.occlusion_texture)
	parse_texture_ref(r, o, "emissiveTexture", &m.emissive_texture)
	get_floats(r, o, "emissiveFactor", m.emissive[:])
	if mode, ok := get_string(r, o, "alphaMode"); ok {
		switch mode {
		case "MASK":
			m.alpha_mode = .Mask
		case "BLEND":
			m.alpha_mode = .Blend
		}
	}
	m.alpha_cutoff = get_f32(r, o, "alphaCutoff", 0.5)
	m.double_sided = get_bool(r, o, "doubleSided")
	exts, has_exts := get_object(r, o, "extensions")
	if !has_exts {
		return
	}
	m.unread_extensions.first = u32(len(r.doc.strings))
	for name in exts {
		if !slice.contains(GLTF_MATERIAL_EXTENSIONS[:], name) {
			append(&r.doc.strings, strings.clone(name, r.a))
			m.unread_extensions.count += 1
		}
	}
	if ior, ok := get_object(r, exts, "KHR_materials_ior"); ok {
		m.ior = get_f32(r, ior, "ior", 1.5)
	}
	if cc, ok := get_object(r, exts, "KHR_materials_clearcoat"); ok {
		m.clearcoat = get_f32(r, cc, "clearcoatFactor", 0)
		m.clearcoat_roughness = get_f32(r, cc, "clearcoatRoughnessFactor", 0)
	}
	if sh, ok := get_object(r, exts, "KHR_materials_sheen"); ok {
		get_floats(r, sh, "sheenColorFactor", m.sheen_color[:])
		m.sheen_roughness = get_f32(r, sh, "sheenRoughnessFactor", 0)
	}
	if tr, ok := get_object(r, exts, "KHR_materials_transmission"); ok {
		m.transmission = get_f32(r, tr, "transmissionFactor", 0)
	}
	if vol, ok := get_object(r, exts, "KHR_materials_volume"); ok {
		m.thickness = get_f32(r, vol, "thicknessFactor", 0)
		get_floats(r, vol, "attenuationColor", m.attenuation_color[:])
		if d := get_f32(r, vol, "attenuationDistance", 0); d > 0 && d < math.F32_MAX {
			m.attenuation_distance = d
		}
	}
	if _, ok := exts["KHR_materials_unlit"]; ok {
		m.unlit = true
	}
	if es, ok := get_object(r, exts, "KHR_materials_emissive_strength"); ok {
		m.emissive_strength = get_f32(r, es, "emissiveStrength", 1)
	}
}

// One attribute name's semantic and set (TEXCOORD_1: Texcoord, 1).
@(private = "file")
semantic_of :: proc(name: string) -> (Gltf_Semantic, u8) {
	base, set := name, 0
	if us := strings.index_byte(name, '_'); us > 0 {
		base = name[:us]
		for c in name[us + 1:] {
			if c < '0' || c > '9' || set > 255 {
				return .Other, 0
			}
			set = set * 10 + int(c - '0')
		}
	}
	if set > 255 {
		return .Other, 0
	}
	switch base {
	case "POSITION":
		return .Position, u8(set)
	case "NORMAL":
		return .Normal, u8(set)
	case "TANGENT":
		return .Tangent, u8(set)
	case "TEXCOORD":
		return .Texcoord, u8(set)
	case "COLOR":
		return .Color, u8(set)
	case "JOINTS":
		return .Joints, u8(set)
	case "WEIGHTS":
		return .Weights, u8(set)
	}
	return .Other, 0
}

// Appends an attributes object; the accessor for each name o3d reads.
@(private = "file")
parse_attributes :: proc(r: ^Gltf_Reader, o: json.Object) -> (rng: Gltf_Range) {
	doc := r.doc
	rng.first = u32(len(doc.attributes))
	for name, v in o {
		sem, set := semantic_of(name)
		append(&doc.attributes, Gltf_Attribute{semantic = sem, set = set, accessor = as_index(r, v, len(doc.accessors))})
	}
	rng.count = u32(len(doc.attributes)) - rng.first
	return
}

@(private = "file")
attribute :: proc(r: ^Gltf_Reader, o: json.Object, name: string) -> i32 {
	v, ok := o[name]
	return as_index(r, v, len(r.doc.accessors)) if ok else -1
}

@(private = "file")
parse_mesh :: proc(r: ^Gltf_Reader, o: json.Object, mesh: ^Gltf_Mesh) {
	doc := r.doc
	prims := get_array(r, o, "primitives")
	mesh.primitives = {u32(len(doc.primitives)), u32(len(prims))}
	for v in prims {
		po := as_object(r, v)
		p := Gltf_Primitive{
			mode     = get_int(r, po, "mode", 4),
			indices  = get_index(r, po, "indices", len(doc.accessors)),
			material = get_index(r, po, "material", len(doc.materials)),
		}
		attrs, _ := get_object(r, po, "attributes")
		p.attributes = parse_attributes(r, attrs)
		p.position = attribute(r, attrs, "POSITION")
		p.normal = attribute(r, attrs, "NORMAL")
		p.texcoord = {attribute(r, attrs, "TEXCOORD_0"), attribute(r, attrs, "TEXCOORD_1")}
		p.color = attribute(r, attrs, "COLOR_0")
		p.joints = attribute(r, attrs, "JOINTS_0")
		p.weights = attribute(r, attrs, "WEIGHTS_0")
		targets := get_array(r, po, "targets")
		p.targets = {u32(len(doc.targets)), u32(len(targets))}
		for tv in targets {
			to := as_object(r, tv)
			append(&doc.targets, Gltf_Target{
				attributes = parse_attributes(r, to),
				position   = attribute(r, to, "POSITION"),
				normal     = attribute(r, to, "NORMAL"),
			})
		}
		append(&doc.primitives, p)
	}
	mesh.weights = get_float_list(r, o, "weights")
	mesh.target_names.first = u32(len(doc.strings))
	if extras, ok := o["extras"].(json.Object); ok {
		if names, has := extras["targetNames"].(json.Array); has {
			for n in names {
				s, is_string := n.(json.String)
				if !is_string {
					fail(r)
				}
				append(&doc.strings, strings.clone(s, r.a))
			}
			mesh.target_names.count = u32(len(names))
		}
	}
}

@(private = "file")
parse_node :: proc(r: ^Gltf_Reader, o: json.Object, i: i32) {
	doc := r.doc
	n := &doc.nodes[i]
	n.name = get_name(r, o)
	n.mesh = get_index(r, o, "mesh", len(doc.meshes))
	n.skin = get_index(r, o, "skin", len(doc.skins))
	n.light = -1
	n.rotation = {0, 0, 0, 1}
	n.scale = 1
	n.matrix_ = {0 = 1, 5 = 1, 10 = 1, 15 = 1}
	n.has_translation = get_floats(r, o, "translation", n.translation[:])
	n.has_rotation = get_floats(r, o, "rotation", n.rotation[:])
	n.has_scale = get_floats(r, o, "scale", n.scale[:])
	n.has_matrix = get_floats(r, o, "matrix", n.matrix_[:])
	n.weights = get_float_list(r, o, "weights")
	if exts, ok := get_object(r, o, "extensions"); ok {
		if lp, has := get_object(r, exts, "KHR_lights_punctual"); has {
			n.light = get_index(r, lp, "light", len(doc.lights))
		}
	}
	// Children in their order: `first_child`, then each one's `next`.
	last: i32 = -1
	for c in get_array(r, o, "children") {
		ci := as_index(r, c, len(doc.nodes))
		if ci < 0 {
			return
		}
		child := &doc.nodes[ci]
		if child.parent >= 0 {
			fail(r) // a node with two parents
			return
		}
		child.parent = i
		if last < 0 {
			n.first_child = ci
		} else {
			doc.nodes[last].next = ci
		}
		last = ci
	}
}

@(private = "file")
parse_animation :: proc(r: ^Gltf_Reader, o: json.Object, anim: ^Gltf_Animation) {
	doc := r.doc
	if name, ok := get_string(r, o, "name"); ok {
		anim.name = strings.clone(name, r.a)
	}
	samplers := get_array(r, o, "samplers")
	anim.samplers = {u32(len(doc.samplers)), u32(len(samplers))}
	for v in samplers {
		so := as_object(r, v)
		s := Gltf_Sampler{
			input  = get_index(r, so, "input", len(doc.accessors)),
			output = get_index(r, so, "output", len(doc.accessors)),
		}
		if s.input < 0 || s.output < 0 {
			fail(r)
		}
		if interp, has := get_string(r, so, "interpolation"); has {
			switch interp {
			case "STEP":
				s.interpolation = .Step
			case "CUBICSPLINE":
				s.interpolation = .Cubic_Spline
			}
		}
		append(&doc.samplers, s)
	}
	channels := get_array(r, o, "channels")
	anim.channels = {u32(len(doc.channels)), u32(len(channels))}
	for v in channels {
		co := as_object(r, v)
		ch := Gltf_Channel{node = -1}
		local := get_index(r, co, "sampler", len(samplers))
		if local < 0 {
			fail(r)
		}
		ch.sampler = i32(anim.samplers.first) + local
		if target, ok := get_object(r, co, "target"); ok {
			ch.node = get_index(r, target, "node", len(doc.nodes))
			if path, has := get_string(r, target, "path"); has {
				switch path {
				case "translation":
					ch.path = .Translation
				case "rotation":
					ch.path = .Rotation
				case "scale":
					ch.path = .Scale
				case "weights":
					ch.path = .Weights
				}
			}
		}
		append(&doc.channels, ch)
	}
}

@(private = "file")
parse_light :: proc(r: ^Gltf_Reader, o: json.Object, l: ^Gltf_Light) {
	l^ = {color = 1, intensity = 1, outer_cone = math.PI / 4}
	if kind, ok := get_string(r, o, "type"); ok {
		switch kind {
		case "directional":
			l.kind = .Directional
		case "point":
			l.kind = .Point
		case "spot":
			l.kind = .Spot
		}
	}
	get_floats(r, o, "color", l.color[:])
	l.intensity = get_f32(r, o, "intensity", 1)
	l.range = get_f32(r, o, "range", 0)
	if spot, ok := get_object(r, o, "spot"); ok {
		l.inner_cone = get_f32(r, spot, "innerConeAngle", 0)
		l.outer_cone = get_f32(r, spot, "outerConeAngle", math.PI / 4)
	}
}

// ---- buffers ----------------------------------------------------------------------------

@(private = "file")
gltf_load_buffers :: proc(r: ^Gltf_Reader, root: json.Object, bin: []u8, has_bin: bool, base_dir: string) {
	doc := r.doc
	buffers, _ := root["buffers"].(json.Array)
	for v, i in buffers {
		size := doc.buffer_lengths[i]
		uri, has_uri := get_string(r, v.(json.Object) or_else nil, "uri")
		if r.err != nil {
			return
		}
		if !has_uri {
			// The GLB's binary chunk; any other buffer without a uri has no data.
			if i == 0 && has_bin {
				if len(bin) < size {
					fail(r, .Data_Too_Short)
					return
				}
				doc.buffers[0] = bin[:size]
			}
			continue
		}
		if !budget_take(&doc.budget, .Gltf_Buffer_Bytes, size) {
			fail(r, .Limit)
			return
		}
		if strings.has_prefix(uri, "data:") {
			comma := strings.index_byte(uri, ',')
			if comma < 0 || !strings.has_suffix(uri[:comma], ";base64") {
				fail(r, .Unknown_Format)
				return
			}
			data, err := base64.decode(uri[comma + 1:], allocator = r.a)
			if err != nil || len(data) < size {
				fail(r, .Buffer_Io)
				return
			}
			doc.buffers[i] = data[:size]
		} else if strings.contains(uri, "://") {
			fail(r, .Unknown_Format)
			return
		} else {
			doc.buffers[i] = read_buffer_file(r, gltf_resolve(base_dir, uri, r.a), size)
			if r.err != nil {
				return
			}
		}
	}
	if len(doc.meshopt) == 0 {
		return
	}
	decoded, alloc_err := make([]u8, doc.buffer_lengths[len(buffers)], r.a)
	if alloc_err != nil {
		fail(r, .Out_Of_Memory)
		return
	}
	doc.buffers[len(buffers)] = decoded
	for m in doc.meshopt {
		data := doc.buffers[m.buffer]
		if data == nil || !fits(len(data), m.offset, 1, 0, m.length) {
			fail(r, .Invalid)
			return
		}
		src := data[m.offset:][:m.length]
		v := doc.views[m.view]
		dst := decoded[v.offset:][:v.length]
		ok: bool
		switch m.mode {
		case .Attributes:
			ok = meshopt_decode_vertices(dst, src, m.count, m.stride) && meshopt_unfilter(dst, m.count, m.stride, m.filter)
		case .Triangles:
			ok = meshopt_decode_triangles(dst, src, m.count, m.stride)
		case .Indices:
			ok = meshopt_decode_indices(dst, src, m.count, m.stride)
		}
		if !ok {
			fail(r, .Invalid_Compression)
			return
		}
	}
}

// Exactly `size` bytes of the file at `path`.
@(private = "file")
read_buffer_file :: proc(r: ^Gltf_Reader, path: string, size: int) -> []u8 {
	f, err := os.open(path)
	if err != nil {
		fail(r, .Buffer_Io if os.exists(path) else .Buffer_Not_Found)
		return nil
	}
	defer os.close(f)
	data, alloc_err := make([]u8, size, r.a)
	if alloc_err != nil {
		fail(r, .Out_Of_Memory)
		return nil
	}
	if n, read_err := os.read_full(f, data); read_err != nil || n < size {
		fail(r, .Buffer_Io)
		return nil
	}
	return data
}

// A URI from the file relative to `base_dir` (none: as it is), its %XX
// escapes decoded.
@(private)
gltf_resolve :: proc(base_dir, uri: string, allocator: mem.Allocator) -> string {
	unhex :: proc "contextless" (c: u8) -> int {
		switch c {
		case '0' ..= '9':
			return int(c - '0')
		case 'a' ..= 'f':
			return int(c - 'a' + 10)
		case 'A' ..= 'F':
			return int(c - 'A' + 10)
		}
		return -1
	}
	b := strings.builder_make(0, len(base_dir) + 1 + len(uri), allocator)
	if len(base_dir) > 0 {
		strings.write_string(&b, base_dir)
		strings.write_byte(&b, '/')
	}
	for i := 0; i < len(uri); i += 1 {
		if uri[i] == '%' && i + 2 < len(uri) {
			hi, lo := unhex(uri[i + 1]), unhex(uri[i + 2])
			if hi >= 0 && lo >= 0 {
				strings.write_byte(&b, u8(hi * 16 + lo))
				i += 2
				continue
			}
		}
		strings.write_byte(&b, uri[i])
	}
	return strings.to_string(b)
}

// ---- validation -------------------------------------------------------------------------

// Bytes of one element (matrix columns of 1- and 2-byte components are
// padded to 4 bytes).
@(private)
gltf_element_size :: proc "contextless" (t: Gltf_Type, c: Gltf_Component) -> int {
	size := component_size(c)
	#partial switch t {
	case .Mat2:
		if size == 1 {
			return 8
		}
	case .Mat3:
		if size == 1 || size == 2 {
			return 12 * size
		}
	}
	return size * type_components(t)
}

@(private = "file")
component_size :: proc "contextless" (c: Gltf_Component) -> int {
	switch c {
	case .I8, .U8:
		return 1
	case .I16, .U16:
		return 2
	case .U32, .F32:
		return 4
	case .Invalid:
	}
	return 0
}

@(private)
type_components :: proc "contextless" (t: Gltf_Type) -> int {
	switch t {
	case .Scalar:
		return 1
	case .Vec2:
		return 2
	case .Vec3:
		return 3
	case .Vec4, .Mat2:
		return 4
	case .Mat3:
		return 9
	case .Mat4:
		return 16
	case .Invalid:
	}
	return 0
}

// The bytes of `view`, nil when its buffer has none.
@(private = "file")
view_bytes :: proc "contextless" (doc: ^Gltf_Document, view: i32) -> []u8 {
	v := doc.views[view]
	data := doc.buffers[v.buffer]
	if data == nil {
		return nil
	}
	return data[v.offset:][:v.length]
}

// Whether `count` items `stride` apart, each `size` bytes, from `offset`
// fit in `length` bytes.
@(private = "file")
fits :: proc "contextless" (length, offset, count, stride, size: int) -> bool {
	if count == 0 {
		return offset <= length
	}
	if offset > length || size > length - offset {
		return false
	}
	return stride == 0 || count - 1 <= (length - offset - size) / stride
}

@(private = "file")
max_index :: proc "contextless" (data: []u8, c: Gltf_Component, count: int) -> int {
	bound := 0
	switch c {
	case .U8:
		for i in 0 ..< count {
			bound = max(bound, int(data[i]))
		}
	case .U16:
		for i in 0 ..< count {
			bound = max(bound, int((^u16le)(&data[i * 2])^))
		}
	case .U32:
		for i in 0 ..< count {
			bound = max(bound, int((^u32le)(&data[i * 4])^))
		}
	case .I8, .I16, .F32, .Invalid:
	}
	return bound
}

// What cgltf's validation checks (the same rules, so the same files load):
// accessors and views inside their data, sparse indices in range, a
// primitive's attributes and targets of one count with its indices below
// it, morph weights matching the targets, no node in a cycle and no deeper
// than the depth limit, and animation outputs matching their inputs.
@(private = "file")
gltf_validate :: proc(doc: ^Gltf_Document) -> Gltf_Error {
	for v in doc.views {
		if !fits(doc.buffer_lengths[v.buffer], v.offset, 1, 0, v.length) {
			return .Invalid
		}
	}
	for &a in doc.accessors {
		if a.component == .Invalid || a.type == .Invalid {
			return .Invalid
		}
		size := gltf_element_size(a.type, a.component)
		a.stride = size
		if a.view >= 0 {
			v := doc.views[a.view]
			if v.stride != 0 {
				a.stride = v.stride
			}
			if !fits(v.length, a.offset, a.count, a.stride, size) {
				return .Invalid
			}
		}
		if a.sparse < 0 {
			continue
		}
		s := doc.sparse[a.sparse]
		if s.indices_component != .U8 && s.indices_component != .U16 && s.indices_component != .U32 {
			return .Invalid
		}
		iv, vv := doc.views[s.indices_view], doc.views[s.values_view]
		isize := component_size(s.indices_component)
		if s.count > a.count || !fits(iv.length, s.indices_offset, s.count, isize, isize) || !fits(vv.length, s.values_offset, s.count, size, size) {
			return .Invalid
		}
		if data := view_bytes(doc, s.indices_view); data != nil && s.count > 0 {
			if max_index(data[s.indices_offset:], s.indices_component, s.count) >= a.count {
				return .Invalid
			}
		}
	}
	for mesh in doc.meshes {
		prims := doc.primitives[mesh.primitives.first:][:mesh.primitives.count]
		if len(prims) == 0 {
			continue
		}
		targets := prims[0].targets.count
		if mesh.weights.count > 0 && mesh.weights.count != targets || mesh.target_names.count > 0 && mesh.target_names.count != targets {
			return .Invalid
		}
		for p in prims {
			if p.mode < 0 || p.mode > 6 || p.targets.count != targets || p.attributes.count == 0 {
				return .Invalid
			}
			attrs := doc.attributes[p.attributes.first:][:p.attributes.count]
			count := doc.accessors[attrs[0].accessor].count
			if count == 0 {
				return .Invalid
			}
			for attr in attrs {
				if doc.accessors[attr.accessor].count != count {
					return .Invalid
				}
			}
			for t in doc.targets[p.targets.first:][:p.targets.count] {
				for attr in doc.attributes[t.attributes.first:][:t.attributes.count] {
					if doc.accessors[attr.accessor].count != count {
						return .Invalid
					}
				}
			}
			if p.indices < 0 {
				continue
			}
			ia := doc.accessors[p.indices]
			if ia.component != .U8 && ia.component != .U16 && ia.component != .U32 || ia.type != .Scalar || ia.stride != component_size(ia.component) {
				return .Invalid
			}
			if ia.view >= 0 && ia.count > 0 {
				if data := view_bytes(doc, ia.view); data != nil && max_index(data[ia.offset:], ia.component, ia.count) >= count {
					return .Invalid
				}
			}
		}
	}
	for n in doc.nodes {
		if n.weights.count > 0 && n.mesh >= 0 {
			mesh := doc.meshes[n.mesh]
			if mesh.primitives.count > 0 && doc.primitives[mesh.primitives.first].targets.count != n.weights.count {
				return .Invalid
			}
		}
	}
	// Each node's depth up its parent chain, once (-2: on the chain being
	// walked, so meeting it again is a cycle).
	depth := make([]i32, len(doc.nodes), virtual.arena_allocator(doc.arena))
	chain := make([dynamic]i32, virtual.arena_allocator(doc.arena))
	for &d in depth {
		d = -1
	}
	for i in 0 ..< len(doc.nodes) {
		clear(&chain)
		p := i32(i)
		for p >= 0 && depth[p] == -1 {
			depth[p] = -2
			append(&chain, p)
			p = doc.nodes[p].parent
		}
		if p >= 0 && depth[p] == -2 {
			return .Invalid
		}
		d := depth[p] if p >= 0 else -1
		#reverse for c in chain {
			d += 1
			depth[c] = d
			if budget_over(&doc.budget, .Depth, int(d)) {
				return .Limit
			}
		}
	}
	for ch in doc.channels {
		if ch.node < 0 {
			continue
		}
		components := 1
		if ch.path == .Weights {
			mesh := doc.nodes[ch.node].mesh
			if mesh < 0 || doc.meshes[mesh].primitives.count == 0 {
				return .Invalid
			}
			components = int(doc.primitives[doc.meshes[mesh].primitives.first].targets.count)
		}
		s := doc.samplers[ch.sampler]
		values := 3 if s.interpolation == .Cubic_Spline else 1
		if i128(doc.accessors[s.input].count) * i128(components) * i128(values) != i128(doc.accessors[s.output].count) {
			return .Invalid
		}
	}
	return .None
}

// ---- accessors --------------------------------------------------------------------------

// Unpacks accessor `acc` as floats into `out` (as many whole elements as
// fit, at most its count; matrices column by column): normalized integers
// to [0, 1] or [-1, 1], others converted, sparse values applied, zeros
// without a view. How many floats it wrote.
gltf_accessor_floats :: proc(doc: ^Gltf_Document, acc: i32, out: []f32) -> int {
	a := doc.accessors[acc]
	comps := type_components(a.type)
	n := min(len(out) / comps, a.count)
	dst := out[:n * comps]
	src := view_bytes(doc, a.view) if a.view >= 0 else nil
	if src == nil {
		for &f in dst {
			f = 0
		}
	} else {
		unpack_floats(src[a.offset:], a.stride, n, a, dst)
	}
	if a.sparse >= 0 {
		s := doc.sparse[a.sparse]
		indices := view_bytes(doc, s.indices_view)
		values := view_bytes(doc, s.values_view)
		if indices == nil || values == nil {
			return len(dst)
		}
		size := gltf_element_size(a.type, a.component)
		one: [16]f32
		for k in 0 ..< s.count {
			at := sparse_index(indices[s.indices_offset:], s.indices_component, k)
			if at < n {
				unpack_floats(values[s.values_offset + k * size:], size, 1, a, one[:comps])
				copy(dst[at * comps:][:comps], one[:comps])
			}
		}
	}
	return len(dst)
}

// Unpacks accessor `acc`'s integer components into `out` (as many whole
// elements as fit): indices, joints. Signed components wrap as cgltf reads
// them; float components and matrices give 0. How many it wrote.
gltf_accessor_uints :: proc(doc: ^Gltf_Document, acc: i32, out: []u32) -> int {
	a := doc.accessors[acc]
	comps := type_components(a.type)
	n := min(len(out) / comps, a.count)
	dst := out[:n * comps]
	src := view_bytes(doc, a.view) if a.view >= 0 else nil
	matrix_type := a.type == .Mat2 || a.type == .Mat3 || a.type == .Mat4
	if src == nil || matrix_type {
		for &u in dst {
			u = 0
		}
	} else {
		unpack_uints(src[a.offset:], a.stride, n, comps, a.component, dst)
	}
	if a.sparse >= 0 && !matrix_type {
		s := doc.sparse[a.sparse]
		indices := view_bytes(doc, s.indices_view)
		values := view_bytes(doc, s.values_view)
		if indices == nil || values == nil {
			return len(dst)
		}
		size := gltf_element_size(a.type, a.component)
		for k in 0 ..< s.count {
			at := sparse_index(indices[s.indices_offset:], s.indices_component, k)
			if at < n {
				unpack_uints(values[s.values_offset + k * size:], size, 1, comps, a.component, dst[at * comps:][:comps])
			}
		}
	}
	return len(dst)
}

@(private = "file")
sparse_index :: #force_inline proc "contextless" (data: []u8, c: Gltf_Component, k: int) -> int {
	#partial switch c {
	case .U8:
		return int(data[k])
	case .U16:
		return int((^u16le)(&data[k * 2])^)
	case .U32:
		return int((^u32le)(&data[k * 4])^)
	}
	return 0
}

// One sweep per component type over `n` elements `stride` apart.
@(private = "file")
unpack_floats :: proc(src: []u8, stride, n: int, a: Gltf_Accessor, dst: []f32) {
	comps := type_components(a.type)
	if a.component == .F32 && stride == comps * 4 {
		mem.copy(raw_data(dst), raw_data(src), n * comps * 4)
		return
	}
	// Matrix columns of 1- and 2-byte components start 4-byte aligned.
	rows, column := comps, 0
	#partial switch a.type {
	case .Mat2:
		rows = 2
	case .Mat3:
		rows = 3
	case .Mat4:
		rows = 4
	}
	column = (rows * component_size(a.component) + 3) &~ 3 if rows != comps else 0
	// Normalized: divided by the largest value (as cgltf divides), signed
	// ones no lower than -1 (as the spec says).
	sweep :: proc(src: []u8, stride, n, comps, rows, column: int, dst: []f32, $T: typeid, divisor: f32) {
		size :: size_of(T)
		for e in 0 ..< n {
			base := e * stride
			for c in 0 ..< comps {
				at := base + (c / rows) * column + (c % rows) * size if column > 0 else base + c * size
				v := f32((^T)(&src[at])^)
				if divisor != 1 {
					v = max(v / divisor, -1)
				}
				dst[e * comps + c] = v
			}
		}
	}
	norm := a.normalized
	switch a.component {
	case .I8:
		sweep(src, stride, n, comps, rows, column, dst, i8, 127 if norm else 1)
	case .U8:
		sweep(src, stride, n, comps, rows, column, dst, u8, 255 if norm else 1)
	case .I16:
		sweep(src, stride, n, comps, rows, column, dst, i16le, 32767 if norm else 1)
	case .U16:
		sweep(src, stride, n, comps, rows, column, dst, u16le, 65535 if norm else 1)
	case .U32:
		sweep(src, stride, n, comps, rows, column, dst, u32le, 1)
	case .F32:
		sweep(src, stride, n, comps, rows, column, dst, f32le, 1)
	case .Invalid:
	}
}

@(private = "file")
unpack_uints :: proc(src: []u8, stride, n, comps: int, c: Gltf_Component, dst: []u32) {
	sweep :: proc(src: []u8, stride, n, comps: int, dst: []u32, $T: typeid) {
		for e in 0 ..< n {
			for k in 0 ..< comps {
				dst[e * comps + k] = u32((^T)(&src[e * stride + k * size_of(T)])^)
			}
		}
	}
	switch c {
	case .I8:
		sweep(src, stride, n, comps, dst, i8)
	case .U8:
		sweep(src, stride, n, comps, dst, u8)
	case .I16:
		sweep(src, stride, n, comps, dst, i16le)
	case .U16:
		sweep(src, stride, n, comps, dst, u16le)
	case .U32:
		sweep(src, stride, n, comps, dst, u32le)
	case .F32, .Invalid:
		for &u in dst[:n * comps] {
			u = 0
		}
	}
}
