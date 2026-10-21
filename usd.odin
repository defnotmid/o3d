package o3d

import "base:runtime"
import "core:fmt"
import "core:math"
import "core:math/linalg"
import "core:os"
import "core:slice"
import "core:strings"

// USD (`.usdz`, `.usdc`) into the same flat `Model`, `Geometries` and
// `Textures` tables as `gltf_load`, so `gltf_draw` draws it. A USDZ's
// first `.usdc` is its layer and its other entries the textures. The layer
// (usd_read.odin) is composed as USD composes one layer: a prim is the
// specs at its path, then (weaker) those of the prims it references inside
// the layer, each reference expanded in turn; its children are all those
// sources' children by name, and each attribute the strongest opinion.
// What o3d builds from the composed prims:
//   - Xform, Scope and untyped prims: nodes, their `xformOpOrder` ops
//     (transform, translate, scale, rotateX/Y/Z and the rotateXYZ orders,
//     orient; `!invert!`, `!resetXformStack!`) as the node's local matrix;
//   - Mesh: polygons fanned into triangles, `normals` / `primvars:normals`
//     and `primvars:st` by their interpolation (and `:indices`), smooth
//     normals where it has none, `orientation`, `doubleSided`; its
//     `materialBind` GeomSubsets split it into a primitive per subset;
//   - `material:binding` (inherited down the hierarchy) to a Material whose
//     `outputs:surface` is a UsdPreviewSurface: diffuseColor, metallic,
//     roughness, opacity, opacityThreshold, emissiveColor, occlusion and
//     normal, each a value or a UsdUVTexture (its file from the USDZ or
//     beside the layer); `primvars:displayColor` without one;
//   - the stage's `metersPerUnit` and `upAxis`: a root node to o3d's meters
//     and Y-up.
// Meshes that instances share are built once. What is not drawn (other
// prim types, references to other files, material inputs o3d has no slot
// for) is named in `Model.warnings`.

usd_load :: proc(path: string, geos: ^Geometries, textures: ^Textures = nil, limits: ^Limits = nil) -> (m: Model, err: Usd_Error) {
	bytes, read_err := os.read_entire_file(path, context.allocator)
	if read_err != nil {
		return {}, .Io if os.exists(path) else .File_Not_Found
	}
	defer delete(bytes)
	slash := max(strings.last_index_byte(path, '/'), strings.last_index_byte(path, '\\'))
	return usd_parse(bytes, geos, textures, path[:slash] if slash >= 0 else "", limits)
}

// A `.usdz` or `.usdc` in memory; a `.usdc`'s textures resolve against
// `base_dir`.
usd_parse :: proc(bytes: []u8, geos: ^Geometries, textures: ^Textures = nil, base_dir := "", limits: ^Limits = nil) -> (m: Model, err: Usd_Error) {
	if len(bytes) == 0 {
		return {}, .Empty
	}
	// The build's scratch is gone on return: what the model keeps is copied.
	runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()
	b := Usd_Builder{geos = geos, textures = textures, base_dir = base_dir, m = &m}
	layer_bytes := bytes
	if len(bytes) >= 4 && string(bytes[:4]) == "PK\x03\x04" {
		entries, zerr := zip_read_directory(bytes)
		if zerr != .None {
			return {}, .Invalid
		}
		b.archive, b.entries = bytes, entries
		layer_bytes = nil
		for e in entries {
			if !strings.has_suffix(strings.to_lower(e.name, context.temp_allocator), ".usdc") {
				continue
			}
			l: Budget
			budget_init(&l, limits)
			if !budget_take(&l, .Gltf_Buffer_Bytes, int(e.uncomp_size)) {
				budget_warn(&l, &m.warnings)
				return {warnings = m.warnings}, .Limit
			}
			layer_bytes, _ = zip_extract(bytes, entries, e.name)
			break
		}
		if layer_bytes == nil {
			return {}, .No_Layer
		}
	}
	l, lerr := usd_read_layer(layer_bytes, limits)
	defer usd_layer_destroy(&l)
	if lerr != nil {
		if lerr == .Limit {
			budget_warn(&l.budget, &m.warnings)
			return {warnings = m.warnings}, .Limit
		}
		return {}, lerr
	}
	b.l = &l
	usd_build(&b)
	budget_warn(&l.budget, &m.warnings)
	for w in b.warned {
		append(&m.warnings, strings.clone(w))
	}
	num_nodes := len(m.nodes)
	resize(&m.scratch.translation, num_nodes)
	resize(&m.scratch.rotation, num_nodes)
	resize(&m.scratch.scale, num_nodes)
	resize(&m.scratch.dirty, num_nodes)
	resize(&m.scratch.world, num_nodes)
	return m, .None
}

// ---- composition ------------------------------------------------------------------------

@(private = "file")
Usd_Builder :: struct {
	l:          ^Usd_Layer,
	m:          ^Model,
	geos:       ^Geometries,
	textures:   ^Textures,
	base_dir:   string,
	archive:    []u8,
	entries:    []Zip_Entry,
	// Every composed prim's sources (path indices, strongest first): a
	// prim's are a range of it while it and its subtree are built.
	sources:    [dynamic]i32,
	// Meshes built once, by the specs their geometry comes from.
	meshes:     map[u64]i32,
	// Materials by their Material prim's path: an index in
	// `m.material_names`, and the o3d material.
	materials:  map[i32]i32,
	built:      [dynamic]Material,
	// Textures by asset path, and the grey copies of one's channel.
	images:     map[string]Texture_Id,
	channels:   map[u64]Texture_Id,
	// The primvar the material being built reads its texture coordinates
	// from ("": st), and each built material's.
	varname:    string,
	varnames:   [dynamic]string,
	// The model's warnings, each once.
	warned:     [dynamic]string,
	// Warm buffers.
	ints:       [dynamic]i64,
	floats:     [dynamic]f32,
	tokens:     [dynamic]u32,
	paths:      [dynamic]u32,
	names:      [Usd_Prop]i32,
	token_of:   map[string]i32,
	// By primitive: the mesh part it is (a subset's index, or the subset
	// count for the faces in none), for instances to bind their own.
	prim_part:  [dynamic]i32,
}

// The properties o3d reads, by name.
@(private = "file")
Usd_Prop :: enum u8 {
	Xform_Op_Order,
	Points,
	Face_Vertex_Counts,
	Face_Vertex_Indices,
	Normals,
	Primvar_Normals,
	Primvar_Normals_Indices,
	Orientation,
	Double_Sided,
	Display_Color,
	Display_Opacity,
	Element_Type,
	Family_Name,
	Indices,
	Material_Binding,
	Outputs_Surface,
	Info_Id,
	Diffuse_Color,
	Metallic,
	Roughness,
	Opacity,
	Opacity_Threshold,
	Emissive_Color,
	Occlusion,
	Normal,
	Clearcoat,
	Use_Specular_Workflow,
	File,
	Varname,
	Visibility,
	St,
	In,
	Scale,
	Rotation,
	Translation,
}

@(private = "file", rodata)
USD_PROPS := [Usd_Prop]string {
	.Xform_Op_Order          = "xformOpOrder",
	.Points                  = "points",
	.Face_Vertex_Counts      = "faceVertexCounts",
	.Face_Vertex_Indices     = "faceVertexIndices",
	.Normals                 = "normals",
	.Primvar_Normals         = "primvars:normals",
	.Primvar_Normals_Indices = "primvars:normals:indices",
	.Orientation             = "orientation",
	.Double_Sided            = "doubleSided",
	.Display_Color           = "primvars:displayColor",
	.Display_Opacity         = "primvars:displayOpacity",
	.Element_Type            = "elementType",
	.Family_Name             = "familyName",
	.Indices                 = "indices",
	.Material_Binding        = "material:binding",
	.Outputs_Surface         = "outputs:surface",
	.Info_Id                 = "info:id",
	.Diffuse_Color           = "inputs:diffuseColor",
	.Metallic                = "inputs:metallic",
	.Roughness               = "inputs:roughness",
	.Opacity                 = "inputs:opacity",
	.Opacity_Threshold       = "inputs:opacityThreshold",
	.Emissive_Color          = "inputs:emissiveColor",
	.Occlusion               = "inputs:occlusion",
	.Normal                  = "inputs:normal",
	.Clearcoat               = "inputs:clearcoat",
	.Use_Specular_Workflow   = "inputs:useSpecularWorkflow",
	.File                    = "inputs:file",
	.Varname                 = "inputs:varname",
	.Visibility              = "visibility",
	.St                      = "inputs:st",
	.In                      = "inputs:in",
	.Scale                   = "inputs:scale",
	.Rotation                = "inputs:rotation",
	.Translation             = "inputs:translation",
}

// A composed prim: its sources in `Usd_Builder.sources`.
@(private = "file")
Usd_Prim :: struct {
	first, count: int,
}

@(private = "file")
warn :: proc(b: ^Usd_Builder, text: string) {
	if !slice.contains(b.warned[:], text) {
		append(&b.warned, strings.clone(text, context.temp_allocator))
	}
}

@(private = "file")
usd_build :: proc(b: ^Usd_Builder) {
	l := b.l
	b.token_of = make(map[string]i32, len(l.tokens), context.temp_allocator)
	for t, i in l.tokens {
		b.token_of[t] = i32(i)
	}
	for p in Usd_Prop {
		b.names[p] = b.token_of[USD_PROPS[p]] or_else -1
	}
	b.prim_part = make([dynamic]i32, context.temp_allocator)
	b.ints = make([dynamic]i64, context.temp_allocator)
	b.floats = make([dynamic]f32, context.temp_allocator)
	b.tokens = make([dynamic]u32, context.temp_allocator)
	b.paths = make([dynamic]u32, context.temp_allocator)
	b.meshes = make(map[u64]i32, 64, context.temp_allocator)
	b.materials = make(map[i32]i32, 64, context.temp_allocator)
	b.images = make(map[string]Texture_Id, 16, context.temp_allocator)
	b.channels = make(map[u64]Texture_Id, 16, context.temp_allocator)
	b.varnames = make([dynamic]string, context.temp_allocator)
	b.sources = make([dynamic]i32, context.temp_allocator)
	b.built = make([dynamic]Material, context.temp_allocator)
	b.warned = make([dynamic]string, context.temp_allocator)

	// The stage: meters per unit (USD's default is centimeters) and up axis.
	root_spec := l.paths[l.root].spec if l.root >= 0 else -1
	meters := 0.01
	if rep, ok := usd_field(l, root_spec, .Meters_Per_Unit); ok {
		meters = usd_number(l, rep) or_else 0.01
	}
	z_up := false
	if rep, ok := usd_field(l, root_spec, .Up_Axis); ok {
		axis, _ := usd_string(l, rep)
		z_up = axis == "Z"
	}
	stage := linalg.matrix4_scale_f32(f32(meters))
	if z_up {
		stage = linalg.matrix4_rotate_f32(-math.PI / 2, {1, 0, 0}) * stage
	}
	append(&b.m.nodes, Model_Node{name = strings.clone("/"), parent = -1, mesh = -1, skin = -1, light = -1, has_matrix = true, rotation = 1, scale = 1, local = stage})

	// Every root prim that is defined (not an over or a class).
	if root_spec < 0 {
		return
	}
	if rep, ok := usd_field(l, root_spec, .Prim_Children); ok && usd_tokens(l, rep, &b.tokens) {
		roots := slice.clone(b.tokens[:], context.temp_allocator)
		for t in roots {
			first := len(b.sources)
			expand(b, usd_child(l, l.root, i32(t), false), first, 0)
			build_prim(b, {first, len(b.sources) - first}, i32(t), 0, 0, 0)
			resize(&b.sources, first)
		}
	}
}

// Appends `path` and, weaker, what it references in this layer (each in
// turn expanded) to `b.sources`, the prim's from `first`.
@(private = "file")
expand :: proc(b: ^Usd_Builder, path: i32, first: int, depth: int) {
	l := b.l
	if path < 0 || l.paths[path].spec < 0 {
		return
	}
	if depth > b.l.budget.limits[.Depth] {
		b.l.budget.hit += {.Depth}
		return
	}
	append(&b.sources, path)
	rep, ok := usd_field(l, l.paths[path].spec, .References)
	if !ok {
		return
	}
	refs := make([dynamic]Usd_Reference, context.temp_allocator)
	if !usd_references(l, rep, &refs) {
		return
	}
	for r in refs {
		if r.asset != "" {
			warn(b, fmt.tprintf("a reference to %q (another file) is not composed", r.asset))
			continue
		}
		target := r.prim
		if target < 0 {
			target = default_prim(b)
		}
		if target >= 0 && !slice.contains(b.sources[first:], target) {
			expand(b, target, first, depth + 1)
		}
	}
}

@(private = "file")
default_prim :: proc(b: ^Usd_Builder) -> i32 {
	l := b.l
	rep, ok := usd_field(l, l.paths[l.root].spec, .Default_Prim)
	if !ok {
		return -1
	}
	name, _ := usd_string(l, rep)
	return usd_child(l, l.root, find_token(b, name), false)
}

// The strongest spec among a prim's sources with property `name`, and the
// property's path (-1: none).
@(private = "file")
property :: proc(b: ^Usd_Builder, p: Usd_Prim, name: i32) -> i32 {
	for s in b.sources[p.first:][:p.count] {
		if c := usd_child(b.l, s, name, true); c >= 0 && b.l.paths[c].spec >= 0 {
			return c
		}
	}
	return -1
}

// The strongest value of an attribute: its default or first time sample.
@(private = "file")
value :: proc(b: ^Usd_Builder, p: Usd_Prim, prop: Usd_Prop) -> (rep: u64, ok: bool) {
	name := b.names[prop]
	if name < 0 {
		return
	}
	for s in b.sources[p.first:][:p.count] {
		c := usd_child(b.l, s, name, true)
		if c < 0 || b.l.paths[c].spec < 0 {
			continue
		}
		spec := b.l.paths[c].spec
		if rep, ok = usd_field(b.l, spec, .Default); ok {
			return
		}
		if rep, ok = usd_field(b.l, spec, .Time_Samples); ok {
			return
		}
	}
	return
}

// A prim field's strongest opinion.
@(private = "file")
prim_field :: proc(b: ^Usd_Builder, p: Usd_Prim, name: Usd_Name) -> (rep: u64, ok: bool) {
	for s in b.sources[p.first:][:p.count] {
		if rep, ok = usd_field(b.l, b.l.paths[s].spec, name); ok {
			return
		}
	}
	return
}

// A relationship's or connection's first target path (-1: none).
@(private = "file")
target :: proc(b: ^Usd_Builder, p: Usd_Prim, prop: Usd_Prop, field: Usd_Name) -> i32 {
	name := b.names[prop]
	if name < 0 {
		return -1
	}
	for s in b.sources[p.first:][:p.count] {
		c := usd_child(b.l, s, name, true)
		if c < 0 || b.l.paths[c].spec < 0 {
			continue
		}
		if rep, ok := usd_field(b.l, b.l.paths[c].spec, field); ok && usd_paths(b.l, rep, &b.paths) && len(b.paths) > 0 {
			return i32(b.paths[0])
		}
	}
	return -1
}

// Builds a composed prim and its subtree under node `parent`; `binding`
// is the material bound above it (0: none).
@(private = "file")
build_prim :: proc(b: ^Usd_Builder, p: Usd_Prim, name: i32, parent: i32, binding: i32, depth: int) {
	l := b.l
	binding := binding
	if p.count == 0 {
		return
	}
	if depth > l.budget.limits[.Depth] {
		l.budget.hit += {.Depth}
		return
	}
	// Defined: some source is a `def`; a strongest `class` is not drawn.
	defined := false
	for s, i in b.sources[p.first:][:p.count] {
		rep, ok := usd_field(l, l.paths[s].spec, .Specifier)
		if !ok {
			continue
		}
		spec := Usd_Specifier(u32(rep))
		if i == 0 && spec == .Class {
			return
		}
		defined ||= spec == .Def
	}
	if !defined {
		return
	}
	if rep, ok := prim_field(b, p, .Active); ok {
		if active, is := usd_bool(l, rep); is && !active {
			return
		}
	}
	if rep, ok := value(b, p, .Visibility); ok {
		if vis, _ := usd_string(l, rep); vis == "invisible" {
			return
		}
	}
	type_name := ""
	if rep, ok := prim_field(b, p, .Type_Name); ok {
		type_name, _ = usd_string(l, rep)
	}
	switch type_name {
	case "", "Xform", "Scope", "Mesh", "SkelRoot":
	case "Material", "Shader", "NodeGraph", "GeomSubset":
		return
	case "Skeleton", "SkelAnimation", "BlendShape":
		warn(b, "USD skeletons and blend shapes (UsdSkel) are not read: skinned meshes are drawn in their rest pose")
		return
	case "Camera", "DistantLight", "SphereLight", "RectLight", "DiskLight", "DomeLight", "CylinderLight":
		warn(b, fmt.tprintf("USD %s prims are not read", type_name))
		return
	case:
		warn(b, fmt.tprintf("USD %s prims are not drawn", type_name))
		return
	}
	if rep, ok := target_or_none(b, p); ok {
		binding = rep
	}

	local, reset := xform(b, p)
	node := i32(len(b.m.nodes))
	append(&b.m.nodes, Model_Node{
		name       = strings.clone(l.tokens[name] if name >= 0 else ""),
		parent     = -1 if reset else parent,
		mesh       = -1,
		skin       = -1,
		light      = -1,
		has_matrix = true,
		rotation   = 1,
		scale      = 1,
		local      = local if !reset else b.m.nodes.local[0] * local,
	})
	if type_name == "Mesh" {
		b.m.nodes.mesh[node] = build_mesh(b, p, binding)
	}

	// Children: every source's, by name, in the order first met.
	names := make([dynamic]u32, context.temp_allocator)
	for s in b.sources[p.first:][:p.count] {
		rep, ok := usd_field(l, l.paths[s].spec, .Prim_Children)
		if !ok || !usd_tokens(l, rep, &b.tokens) {
			continue
		}
		for t in b.tokens {
			if !slice.contains(names[:], t) {
				append(&names, t)
			}
		}
	}
	for t in names {
		first := len(b.sources)
		for i in 0 ..< p.count {
			expand(b, usd_child(l, b.sources[p.first + i], i32(t), false), first, 0)
		}
		build_prim(b, {first, len(b.sources) - first}, i32(t), node, binding, depth + 1)
		resize(&b.sources, first)
	}
}

// The material a prim binds (false: none of its own).
@(private = "file")
target_or_none :: proc(b: ^Usd_Builder, p: Usd_Prim) -> (material: i32, ok: bool) {
	t := target(b, p, .Material_Binding, .Target_Paths)
	if t < 0 {
		return 0, false
	}
	return build_material(b, t), true
}

// A prim's local matrix from its xform ops (o3d's column vectors), and
// whether it resets the stack (ignores its parents).
@(private = "file")
xform :: proc(b: ^Usd_Builder, p: Usd_Prim) -> (local: Mat4, reset: bool) {
	l := b.l
	local = linalg.MATRIX4F32_IDENTITY
	rep, ok := value(b, p, .Xform_Op_Order)
	if !ok || !usd_tokens(l, rep, &b.tokens) {
		return
	}
	order := slice.clone(b.tokens[:], context.temp_allocator)
	for t in order {
		op := l.tokens[t] if int(t) < len(l.tokens) else ""
		if op == "!resetXformStack!" {
			reset = true
			local = linalg.MATRIX4F32_IDENTITY
			continue
		}
		invert := strings.has_prefix(op, "!invert!")
		name := op[len("!invert!"):] if invert else op
		prop := property(b, p, find_token(b, name))
		if prop < 0 {
			continue
		}
		spec := l.paths[prop].spec
		vrep, has := usd_field(l, spec, .Default)
		if !has {
			vrep, has = usd_field(l, spec, .Time_Samples)
		}
		if !has {
			continue
		}
		m := op_matrix(b, name, vrep)
		local = local * (linalg.inverse(m) if invert else m)
	}
	return
}

// One xform op's matrix (column vectors).
@(private = "file")
op_matrix :: proc(b: ^Usd_Builder, name: string, rep: u64) -> Mat4 {
	l := b.l
	kind := name[len("xformOp:"):] if strings.has_prefix(name, "xformOp:") else name
	if colon := strings.index_byte(kind, ':'); colon >= 0 {
		kind = kind[:colon] // a suffix names the op (xformOp:translate:pivot)
	}
	vec :: proc(l: ^Usd_Layer, rep: u64) -> Vec3 {
		v, n, ok := usd_vector(l, rep)
		if ok && n >= 3 {
			return {f32(v[0]), f32(v[1]), f32(v[2])}
		}
		x, _ := usd_number(l, rep)
		return {f32(x), f32(x), f32(x)}
	}
	deg :: proc(x: f32) -> f32 {
		return x * math.PI / 180
	}
	switch kind {
	case "transform":
		mt, n, ok := usd_matrix(l, rep)
		if !ok || n != 4 {
			return linalg.MATRIX4F32_IDENTITY
		}
		// Row vectors: transposed into o3d's columns.
		out: Mat4
		for i in 0 ..< 4 {
			for j in 0 ..< 4 {
				out[j, i] = f32(mt[i][j])
			}
		}
		return out
	case "translate":
		return linalg.matrix4_translate_f32(vec(l, rep))
	case "scale":
		return linalg.matrix4_scale_f32(vec(l, rep))
	case "orient":
		v, n, ok := usd_vector(l, rep)
		if !ok || n != 4 {
			return linalg.MATRIX4F32_IDENTITY
		}
		q := quaternion(x = f32(v[0]), y = f32(v[1]), z = f32(v[2]), w = f32(v[3]))
		return linalg.matrix4_from_quaternion_f32(linalg.normalize(q))
	case "rotateX", "rotateY", "rotateZ":
		x, _ := usd_number(l, rep)
		axis := Vec3{1, 0, 0} if kind == "rotateX" else (Vec3{0, 1, 0} if kind == "rotateY" else Vec3{0, 0, 1})
		return linalg.matrix4_rotate_f32(deg(f32(x)), axis)
	case "rotateXYZ", "rotateXZY", "rotateYXZ", "rotateYZX", "rotateZXY", "rotateZYX":
		a := vec(l, rep)
		// The first axis named turns first: the last is outermost.
		out := linalg.MATRIX4F32_IDENTITY
		for c in kind[len("rotate"):] {
			i := int(c - 'X')
			axis: Vec3
			axis[i] = 1
			out = linalg.matrix4_rotate_f32(deg(a[i]), axis) * out
		}
		return out
	}
	warn(b, fmt.tprintf("USD xform op %s is not read", name))
	return linalg.MATRIX4F32_IDENTITY
}

// ---- meshes -----------------------------------------------------------------------------

@(private = "file")
Usd_Interpolation :: enum u8 {
	Constant,
	Uniform,
	Vertex,
	Face_Varying,
}

// A primvar: its values (`components` each), indices, and interpolation.
@(private = "file")
Usd_Primvar :: struct {
	values:        []f32,
	components:    int,
	indices:       []i64,
	interpolation: Usd_Interpolation,
	ok:            bool,
}

@(private = "file")
primvar :: proc(b: ^Usd_Builder, p: Usd_Prim, name: string, default_interpolation: Usd_Interpolation) -> (pv: Usd_Primvar) {
	l := b.l
	token := find_token(b, name)
	prop := property(b, p, token)
	if prop < 0 {
		return
	}
	rep, ok := usd_field(l, l.paths[prop].spec, .Default)
	if !ok {
		rep, ok = usd_field(l, l.paths[prop].spec, .Time_Samples)
	}
	if !ok {
		return
	}
	comps, fok := usd_floats(l, rep, &b.floats)
	if !fok {
		return
	}
	pv.values = slice.clone(b.floats[:], context.temp_allocator)
	pv.components = comps
	pv.interpolation = default_interpolation
	// The interpolation is the strongest opinion among the property's specs.
	for s in b.sources[p.first:][:p.count] {
		c := usd_child(l, s, token, true)
		if c < 0 || l.paths[c].spec < 0 {
			continue
		}
		if irep, has := usd_field(l, l.paths[c].spec, .Interpolation); has {
			text, _ := usd_string(l, irep)
			switch text {
			case "constant":
				pv.interpolation = .Constant
			case "uniform":
				pv.interpolation = .Uniform
			case "vertex", "varying":
				pv.interpolation = .Vertex
			case "faceVarying":
				pv.interpolation = .Face_Varying
			}
			break
		}
	}
	if iprop := property(b, p, find_token(b, strings.concatenate({name, ":indices"}, context.temp_allocator))); iprop >= 0 {
		if irep, has := usd_field(l, l.paths[iprop].spec, .Default); has && usd_ints(l, irep, &b.ints) {
			pv.indices = slice.clone(b.ints[:], context.temp_allocator)
		}
	}
	pv.ok = true
	return
}

@(private = "file")
find_token :: proc(b: ^Usd_Builder, s: string) -> i32 {
	return b.token_of[s] or_else -1
}

// A primvar's value for face `face`, corner `corner` (index `corner` of
// all corners), point `point`.
@(private = "file")
primvar_at :: #force_inline proc "contextless" (pv: ^Usd_Primvar, face, corner, point: int) -> (v: [4]f32, ok: bool) {
	i: int
	switch pv.interpolation {
	case .Constant:
		i = 0
	case .Uniform:
		i = face
	case .Vertex:
		i = point
	case .Face_Varying:
		i = corner
	}
	if len(pv.indices) > 0 {
		if i >= len(pv.indices) {
			return
		}
		i = int(pv.indices[i])
	}
	if i < 0 || (i + 1) * pv.components > len(pv.values) {
		return
	}
	for c in 0 ..< min(pv.components, 4) {
		v[c] = pv.values[i * pv.components + c]
	}
	return v, true
}

// A Mesh prim's `Model_Mesh` (a primitive per bound subset): its geometry
// built once for every instance with the same geometry specs.
@(private = "file")
build_mesh :: proc(b: ^Usd_Builder, p: Usd_Prim, binding: i32) -> i32 {
	l := b.l
	// The subsets: composed children typed GeomSubset of family materialBind.
	Subset :: struct {
		faces:    []i64,
		material: i32,
		key:      i32,
	}
	subsets := make([dynamic]Subset, context.temp_allocator)
	names := make([dynamic]u32, context.temp_allocator)
	for s in b.sources[p.first:][:p.count] {
		rep, ok := usd_field(l, l.paths[s].spec, .Prim_Children)
		if !ok || !usd_tokens(l, rep, &b.tokens) {
			continue
		}
		for t in b.tokens {
			if !slice.contains(names[:], t) {
				append(&names, t)
			}
		}
	}
	for t in names {
		first := len(b.sources)
		for i in 0 ..< p.count {
			expand(b, usd_child(l, b.sources[p.first + i], i32(t), false), first, 0)
		}
		sp := Usd_Prim{first, len(b.sources) - first}
		defer resize(&b.sources, first)
		if rep, ok := prim_field(b, sp, .Type_Name); !ok || (usd_string(l, rep) or_else "") != "GeomSubset" {
			continue
		}
		if rep, ok := value(b, sp, .Family_Name); ok && (usd_string(l, rep) or_else "") != "materialBind" {
			continue
		}
		rep, ok := value(b, sp, .Indices)
		if !ok || !usd_ints(l, rep, &b.ints) {
			continue
		}
		material := binding
		if mat, has := target_or_none(b, sp); has {
			material = mat
		}
		append(&subsets, Subset{faces = slice.clone(b.ints[:], context.temp_allocator), material = material, key = property(b, sp, b.names[.Indices])})
	}

	// The geometry's identity: the specs of everything it is built from.
	props := [?]Usd_Prop{.Points, .Face_Vertex_Counts, .Face_Vertex_Indices, .Normals, .Primvar_Normals, .Primvar_Normals_Indices, .Orientation}
	key: u64 = 14695981039346656037
	mix :: #force_inline proc "contextless" (h: ^u64, v: i32) {
		h^ = (h^ ~ u64(u32(v))) * 1099511628211
	}
	for prop in props {
		mix(&key, property(b, p, b.names[prop]))
	}
	// Texture coordinates from the primvar the materials read (st).
	st_name := "primvars:st"
	for s in subsets {
		if s.material > 0 && b.varnames[s.material - 1] != "" {
			st_name = strings.concatenate({"primvars:", b.varnames[s.material - 1]}, context.temp_allocator)
			break
		}
	}
	if binding > 0 && b.varnames[binding - 1] != "" && st_name == "primvars:st" {
		st_name = strings.concatenate({"primvars:", b.varnames[binding - 1]}, context.temp_allocator)
	}
	mix(&key, property(b, p, find_token(b, st_name)))
	for s in subsets {
		mix(&key, s.key)
	}

	double_sided := false
	if rep, ok := value(b, p, .Double_Sided); ok {
		double_sided = usd_bool(l, rep) or_else false
	}
	first_prim := u32(len(b.m.primitives))
	if cached, ok := b.meshes[key]; ok {
		// The same geometry; this instance's own materials.
		src := b.m.meshes[cached]
		for i in 0 ..< src.primitive_count {
			prim := b.m.primitives[src.first_primitive + i]
			part := b.prim_part[src.first_primitive + i]
			prim.material, prim.material_id = material_of(b, p, subsets[part].material if int(part) < len(subsets) else binding)
			if double_sided {
				prim.material.side = .Double
			}
			append(&b.m.primitives, prim)
			append(&b.prim_part, part)
		}
		append(&b.m.meshes, Model_Mesh{first_primitive = first_prim, primitive_count = src.primitive_count})
		return i32(len(b.m.meshes) - 1)
	}

	rep, ok := value(b, p, .Points)
	if !ok {
		return -1
	}
	if _, pok := usd_floats(l, rep, &b.floats); !pok || len(b.floats) % 3 != 0 {
		return -1
	}
	points := slice.reinterpret([]Vec3, slice.clone(b.floats[:], context.temp_allocator))
	counts, corners: []i64
	if rep, ok = value(b, p, .Face_Vertex_Counts); ok && usd_ints(l, rep, &b.ints) {
		counts = slice.clone(b.ints[:], context.temp_allocator)
	}
	if rep, ok = value(b, p, .Face_Vertex_Indices); ok && usd_ints(l, rep, &b.ints) {
		corners = slice.clone(b.ints[:], context.temp_allocator)
	}
	left_handed := false
	if rep, ok = value(b, p, .Orientation); ok {
		left_handed = (usd_string(l, rep) or_else "") == "leftHanded"
	}
	normals := primvar(b, p, "primvars:normals", .Vertex)
	if !normals.ok {
		normals = primvar(b, p, "normals", .Vertex)
	}
	st := primvar(b, p, st_name, .Face_Varying)

	// Faces: each its first corner; a face whose corners run past the
	// indices, or name a point that is not there, is left out.
	face_first := make([]int, len(counts) + 1, context.temp_allocator)
	tris := 0
	for c, f in counts {
		face_first[f + 1] = face_first[f] + int(max(c, 0))
		if c >= 3 {
			tris += int(c) - 2
		}
	}
	if face_first[len(counts)] > len(corners) {
		warn(b, "a USD mesh has fewer face vertex indices than its counts ask for; the rest is left out")
	}
	if !budget_take_wide(&l.budget, .Gltf_Values, i128(tris) * 3 * 10) {
		return -1
	}

	// Smooth normals where the mesh has none: area-weighted by point.
	smooth: []Vec3
	if !normals.ok {
		smooth = make([]Vec3, len(points), context.temp_allocator)
		for f in 0 ..< len(counts) {
			c0 := face_first[f]
			n := face_first[f + 1] - c0
			if n < 3 || face_first[f + 1] > len(corners) {
				continue
			}
			for k in 1 ..< n - 1 {
				i0, i1, i2 := corners[c0], corners[c0 + k], corners[c0 + k + 1]
				if !in_range(i0, len(points)) || !in_range(i1, len(points)) || !in_range(i2, len(points)) {
					continue
				}
				fn := linalg.cross(points[i1] - points[i0], points[i2] - points[i0])
				if left_handed {
					fn = -fn
				}
				smooth[i0] += fn
				smooth[i1] += fn
				smooth[i2] += fn
			}
		}
		for &n in smooth {
			n = linalg.normalize(n) if linalg.length2(n) > 1e-30 else Vec3{0, 1, 0}
		}
	}

	// Which part each face goes to: its subset's, or the last part (the
	// faces in no subset).
	parts := len(subsets) + 1
	face_part := make([]i32, len(counts), context.temp_allocator)
	slice.fill(face_part, i32(len(subsets)))
	for s, si in subsets {
		for f in s.faces {
			if in_range(f, len(counts)) {
				face_part[f] = i32(si)
			}
		}
	}
	// Count, then fill each part's triangles: one vertex per corner.
	part_tris := make([]int, parts, context.temp_allocator)
	for f in 0 ..< len(counts) {
		if n := face_first[f + 1] - face_first[f]; n >= 3 && face_first[f + 1] <= len(corners) {
			part_tris[face_part[f]] += n - 2
		}
	}
	for pi in 0 ..< parts {
		if part_tris[pi] == 0 {
			continue
		}
		verts := make([dynamic]Vertex, 0, part_tris[pi] * 3, context.temp_allocator)
		for f in 0 ..< len(counts) {
			if face_part[f] != i32(pi) {
				continue
			}
			c0 := face_first[f]
			n := face_first[f + 1] - c0
			if n < 3 || face_first[f + 1] > len(corners) {
				continue
			}
			for k in 1 ..< n - 1 {
				tri := [3]int{0, k, k + 1}
				if left_handed {
					tri = {0, k + 1, k}
				}
				ok_tri := true
				vs: [3]Vertex
				for cj, j in tri {
					corner := c0 + cj
					point := corners[corner]
					if !in_range(point, len(points)) {
						ok_tri = false
						break
					}
					v := Vertex{position = points[point]}
					if normals.ok {
						nv, _ := primvar_at(&normals, f, corner, int(point))
						v.normal = linalg.normalize0(nv.xyz)
					} else {
						v.normal = smooth[point]
					}
					if st.ok {
						uv, _ := primvar_at(&st, f, corner, int(point))
						v.uv = uv.xy
					}
					vs[j] = v
				}
				if ok_tri {
					append(&verts, ..vs[:])
				}
			}
		}
		indices := make([]u32, len(verts), context.temp_allocator)
		for &x, i in indices {
			x = u32(i)
		}
		mat := subsets[pi].material if pi < len(subsets) else binding
		material, material_id := material_of(b, p, mat)
		if double_sided {
			material.side = .Double
		}
		append(&b.m.primitives, Model_Primitive{geometry = geometry_add(b.geos, verts[:], indices), material = material, material_id = material_id})
		append(&b.prim_part, i32(pi))
	}
	append(&b.m.meshes, Model_Mesh{first_primitive = first_prim, primitive_count = u32(len(b.m.primitives)) - first_prim})
	b.meshes[key] = i32(len(b.m.meshes) - 1)
	return i32(len(b.m.meshes) - 1)
}

@(private = "file")
in_range :: #force_inline proc "contextless" (i: i64, n: int) -> bool {
	return i >= 0 && i < i64(n)
}

// The material for a primitive: its bound material, else the mesh's
// display colour, else the default.
@(private = "file")
material_of :: proc(b: ^Usd_Builder, p: Usd_Prim, bound: i32) -> (Material, Model_Material_Id) {
	if bound > 0 {
		return b.built[bound - 1], Model_Material_Id(bound)
	}
	mat := MATERIAL_DEFAULT
	mat.color = linear_to_srgb_vec3({0.18, 0.18, 0.18})
	mat.roughness = 0.5
	mat.metalness = 0
	if rep, ok := value(b, p, .Display_Color); ok {
		if comps, fok := usd_floats(b.l, rep, &b.floats); fok && comps == 3 && len(b.floats) >= 3 {
			mat.color = linear_to_srgb_vec3({b.floats[0], b.floats[1], b.floats[2]})
		} else if v, n, vok := usd_vector(b.l, rep); vok && n >= 3 {
			mat.color = linear_to_srgb_vec3({f32(v[0]), f32(v[1]), f32(v[2])})
		}
	}
	return mat, 0
}

// ---- materials --------------------------------------------------------------------------

// A Material prim's index in `m.material_names` plus one (built once).
@(private = "file")
build_material :: proc(b: ^Usd_Builder, path: i32) -> i32 {
	if id, ok := b.materials[path]; ok {
		return id
	}
	l := b.l
	mat := MATERIAL_DEFAULT
	mat.color = linear_to_srgb_vec3({0.18, 0.18, 0.18})
	mat.roughness = 0.5
	mat.metalness = 0
	name := l.tokens[l.paths[path].token] if path >= 0 && l.paths[path].token >= 0 else ""
	b.varname = ""
	first := len(b.sources)
	expand(b, path, first, 0)
	mp := Usd_Prim{first, len(b.sources) - first}
	surface := target(b, mp, .Outputs_Surface, .Connection_Paths)
	resize(&b.sources, first)
	if surface >= 0 {
		shader := l.paths[surface].parent if l.paths[surface].property else surface
		first = len(b.sources)
		expand(b, shader, first, 0)
		sp := Usd_Prim{first, len(b.sources) - first}
		id := ""
		if rep, ok := value(b, sp, .Info_Id); ok {
			id, _ = usd_string(l, rep)
		}
		if id == "UsdPreviewSurface" {
			preview_surface(b, sp, &mat, name)
		} else {
			warn(b, fmt.tprintf("USD surface shader %q is not read (material %q)", id, name))
		}
		resize(&b.sources, first)
	}
	append(&b.built, mat)
	append(&b.varnames, b.varname)
	append(&b.m.material_names, strings.clone(name))
	b.materials[path] = i32(len(b.built))
	return i32(len(b.built))
}

@(private = "file")
preview_surface :: proc(b: ^Usd_Builder, sp: Usd_Prim, mat: ^Material, name: string) {
	// An input's value, or the texture (and its output channel) connected.
	Input :: struct {
		v:       [4]f32,
		n:       int,
		using tex: Usd_Texture,
	}
	input :: proc(b: ^Usd_Builder, sp: Usd_Prim, prop: Usd_Prop) -> (in_: Input) {
		l := b.l
		if conn := target(b, sp, prop, .Connection_Paths); conn >= 0 {
			in_.tex = connected_texture(b, conn)
			if in_.texture != 0 {
				return
			}
		}
		rep, ok := value(b, sp, prop)
		if !ok {
			return
		}
		if v, n, vok := usd_vector(l, rep); vok {
			for i in 0 ..< n {
				in_.v[i] = f32(v[i])
			}
			in_.n = n
		} else if x, xok := usd_number(l, rep); xok {
			in_.v[0], in_.n = f32(x), 1
		}
		return
	}
	// A texture for a slot that reads `reads` ("g", say): one with the
	// connected channel there, if it is another.
	slot :: proc(b: ^Usd_Builder, in_: Input, reads: u8, xf: ^Texture_Transform) -> Texture_Id {
		xf^ = in_.transform
		if b.varname == "" {
			b.varname = in_.varname
		} else if in_.varname != "" && in_.varname != b.varname {
			warn(b, fmt.tprintf("textures reading primvar %q and %q in one material: o3d reads %q", b.varname, in_.varname, b.varname))
		}
		return channel_texture(b, in_.texture, in_.channel, reads)
	}
	diffuse := input(b, sp, .Diffuse_Color)
	if diffuse.texture != 0 {
		mat.color = {1, 1, 1}
		mat.color_map = slot(b, diffuse, 0, &mat.color_transform)
	} else if diffuse.n >= 3 {
		mat.color = linear_to_srgb_vec3(diffuse.v.xyz)
	}
	if metal := input(b, sp, .Metallic); metal.texture != 0 {
		mat.metalness, mat.metalness_map = 1, slot(b, metal, 'b', &mat.metalness_transform)
	} else if metal.n > 0 {
		mat.metalness = clamp(metal.v[0], 0, 1)
	}
	if rough := input(b, sp, .Roughness); rough.texture != 0 {
		mat.roughness, mat.roughness_map = 1, slot(b, rough, 'g', &mat.roughness_transform)
	} else if rough.n > 0 {
		mat.roughness = clamp(rough.v[0], 0, 1)
	}
	if occ := input(b, sp, .Occlusion); occ.texture != 0 {
		mat.ao_map = slot(b, occ, 'r', &mat.ao_transform)
		mat.ao_map_intensity = 1
	}
	if nrm := input(b, sp, .Normal); nrm.texture != 0 {
		mat.normal_map = slot(b, nrm, 0, &mat.normal_transform)
	}
	if em := input(b, sp, .Emissive_Color); em.texture != 0 {
		mat.emissive, mat.emissive_map, mat.emissive_intensity = {1, 1, 1}, slot(b, em, 0, &mat.emissive_transform), 1
	} else if em.n >= 3 && em.v.xyz != {} {
		mat.emissive, mat.emissive_intensity = linear_to_srgb_vec3(em.v.xyz), 1
	}
	opacity := input(b, sp, .Opacity)
	if opacity.texture != 0 {
		mat.alpha_map = slot(b, opacity, 'g', &mat.alpha_transform)
	} else if opacity.n > 0 {
		mat.opacity = clamp(opacity.v[0], 0, 1)
	}
	threshold := input(b, sp, .Opacity_Threshold)
	if threshold.texture != 0 {
		// A threshold per texel: a cut at one half (the same cut for the
		// all-or-nothing opacity textures that come with it).
		mat.flags += {.Alpha_Mask}
		mat.alpha_cutoff = 0.5
	} else if threshold.n > 0 && threshold.v[0] > 0 {
		mat.flags += {.Alpha_Mask}
		mat.alpha_cutoff = threshold.v[0]
	} else if mat.opacity < 1 || opacity.texture != 0 {
		mat.flags += {.Transparent}
	}
	if cc := input(b, sp, .Clearcoat); cc.n > 0 && cc.v[0] > 0 {
		warn(b, fmt.tprintf("USD clearcoat is not drawn (material %q)", name))
	}
	if spec := input(b, sp, .Use_Specular_Workflow); spec.n > 0 && spec.v[0] != 0 {
		warn(b, fmt.tprintf("USD's specular workflow is drawn as metallic-roughness (material %q)", name))
	}
}

// A UsdUVTexture a material input connects: the texture, the channel the
// connection reads ("rgb", "r", ...), its UsdTransform2d, and the primvar
// its coordinates come from.
@(private = "file")
Usd_Texture :: struct {
	texture:   Texture_Id,
	channel:   string,
	transform: Texture_Transform,
	varname:   string,
}

@(private = "file")
connected_texture :: proc(b: ^Usd_Builder, conn: i32) -> (t: Usd_Texture) {
	l := b.l
	if conn < 0 || !l.paths[conn].property {
		return
	}
	channel := l.tokens[l.paths[conn].token]
	t.channel = channel[len("outputs:"):] if strings.has_prefix(channel, "outputs:") else channel
	first := len(b.sources)
	defer resize(&b.sources, first)
	expand(b, l.paths[conn].parent, first, 0)
	tp := Usd_Prim{first, len(b.sources) - first}
	rep, ok := value(b, tp, .Info_Id)
	if !ok || (usd_string(l, rep) or_else "") != "UsdUVTexture" {
		return
	}
	rep, ok = value(b, tp, .File)
	if !ok {
		return
	}
	asset, _ := usd_string(l, rep)
	t.texture = image(b, asset)
	// Its coordinates: a primvar reader, perhaps through a UsdTransform2d.
	st := target(b, tp, .St, .Connection_Paths)
	for _ in 0 ..< 2 {
		if st < 0 || !l.paths[st].property {
			break
		}
		node := l.paths[st].parent
		nf := len(b.sources)
		expand(b, node, nf, 0)
		np := Usd_Prim{nf, len(b.sources) - nf}
		id := ""
		if r, has := value(b, np, .Info_Id); has {
			id, _ = usd_string(l, r)
		}
		next := i32(-1)
		switch id {
		case "UsdTransform2d":
			// result = in * scale, rotated (degrees, counterclockwise), +
			// translation, in USD's st (origin bottom left, as o3d's
			// uvs); in glTF's terms (origin top left) for Texture_Transform.
			scale := [2]f32{1, 1}
			rotation: f32
			tr: [2]f32
			if r, has := value(b, np, .Scale); has {
				if v, n, vok := usd_vector(l, r); vok && n >= 2 {
					scale = {f32(v[0]), f32(v[1])}
				}
			}
			if r, has := value(b, np, .Rotation); has {
				rotation = f32(usd_number(l, r) or_else 0) * math.PI / 180
			}
			if r, has := value(b, np, .Translation); has {
				if v, n, vok := usd_vector(l, r); vok && n >= 2 {
					tr = {f32(v[0]), f32(v[1])}
				}
			}
			c, s := math.cos(rotation), math.sin(rotation)
			t.transform = {rotation = rotation, scale = scale, offset = {tr.x - s * scale.y, 1 - tr.y - c * scale.y}}
			next = target(b, np, .In, .Connection_Paths)
		case "UsdPrimvarReader_float2":
			if r, has := value(b, np, .Varname); has {
				t.varname, _ = usd_string(l, r)
			}
		}
		resize(&b.sources, nf)
		st = next
	}
	return
}

// `id` with channel `channel` of it where a slot reads `reads` ('r', 'g',
// 'b'; 0: the slot reads rgb): itself when it is there already, else a
// grey copy of that channel, made once.
@(private = "file")
channel_texture :: proc(b: ^Usd_Builder, id: Texture_Id, channel: string, reads: u8) -> Texture_Id {
	if id == 0 || reads == 0 || channel == "rgb" || len(channel) != 1 || channel[0] == reads {
		return id
	}
	c := strings.index_byte("rgba", channel[0])
	if c < 0 {
		return id
	}
	key := u64(id) << 8 | u64(c)
	if derived, ok := b.channels[key]; ok {
		return derived
	}
	tex, _ := texture_get(b.textures, id)
	n := int(tex.width * tex.height)
	grey := make([]u8, n * 4, context.temp_allocator)
	for texel, i in b.textures.texels[tex.mips[0]:][:n] {
		v := texel[c]
		grey[i * 4], grey[i * 4 + 1], grey[i * 4 + 2], grey[i * 4 + 3] = v, v, v, 255
	}
	derived := texture_add(b.textures, grey, int(tex.width), int(tex.height))
	b.channels[key] = derived
	return derived
}

// An asset path's texture (decoded once): from the USDZ, or beside the
// layer.
@(private = "file")
image :: proc(b: ^Usd_Builder, asset: string) -> Texture_Id {
	if b.textures == nil || asset == "" {
		return 0
	}
	if id, ok := b.images[asset]; ok {
		return id
	}
	path := asset
	for strings.has_prefix(path, "./") {
		path = path[2:]
	}
	data: []u8
	owned := false
	if b.archive != nil {
		if e, ok := zip_find(b.entries, path); ok && budget_take(&b.l.budget, .Gltf_Buffer_Bytes, int(e.uncomp_size)) {
			data, _ = zip_extract(b.archive, b.entries, path)
		}
	} else {
		full := strings.concatenate({b.base_dir, "/", path}, context.temp_allocator) if b.base_dir != "" else path
		if bytes, err := os.read_entire_file(full, context.allocator); err == nil {
			data, owned = bytes, true
		}
	}
	defer if owned {
		delete(data)
	}
	id: Texture_Id
	if data != nil && budget_take(&b.l.budget, .Image_Pixels, texture_pixels(data)) {
		id, _ = texture_decode(b.textures, data)
	}
	if id == 0 {
		warn(b, fmt.tprintf("texture %q could not be read", asset))
	}
	b.images[strings.clone(asset, context.temp_allocator)] = id
	return id
}
