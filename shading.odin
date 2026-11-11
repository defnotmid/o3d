package o3d

// o3d's own shading: shaders/o3d.vert and shaders/o3d.frag, oksl GLSL
// compiled at load, a program per variant, for both renderers (SPIR-V and
// MSL for the GPU, oksl's programs for the software renderer).
//
// A variant's shape is fixed when it compiles: its entry (what the draw is
// and which pass draws it), its geometry (a mesh, skinned, or quads), the
// material's flags, each map's uv set and where each layer map is, and the
// frame's lights (each one's kind and shadow), shadow maps, environment and
// fog. Its values are uniforms.
//
// The uniforms have one shape: the shaders' declarations are written from
// the blocks' fields (Draw_Block, Material_Block, Frame_Block) as
// u_<field>. A variant's block holds the fields it reads where oksl lays
// them out (std140, in order), and both renderers fill it from the structs
// by the variant's field table (`block_fill`), each field's size checked
// when it compiles.

import "base:runtime"
import "core:fmt"
import "core:reflect"
import "core:strings"

import "deps:oksl"
import curves "deps:ocurves"

// shaders/o3d.frag reads an exact shape's words (its entry's fill rule,
// curve.odin's CURVE_EVEN_ODD) in this layout of ocurves'.
#assert(curves.FORMAT == 1, "ocurves' words have another layout than shaders/o3d.frag reads: use the ocurves this o3d's tag names")

@(private = "file")
VERTEX_GLSL :: #load("shaders/o3d.vert", string)
@(private = "file")
FRAGMENT_GLSL :: #load("shaders/o3d.frag", string)
@(private = "file")
DEPTH_GLSL :: #load("shaders/o3d_depth.frag", string)

// Both stages' helpers, ahead of their declarations.
@(private = "file")
O3D_COMMON :: `precision highp float;
precision highp int;
const float PI = 3.14159265358979323846;
float srgb_to_linear(float c) {
	return c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4);
}
vec3 linear_rgb(vec3 c) {
	return vec3(srgb_to_linear(c.r), srgb_to_linear(c.g), srgb_to_linear(c.b));
}
// A direction's equirectangular coordinates, top-down (sample_mip's).
vec2 equirect_uv(vec3 d) {
	float u = atan(d.z, d.x) * (0.5 / PI) + 0.5;
	float v = 0.5 - asin(clamp(d.y, -1.0, 1.0)) * (1.0 / PI);
	return vec2(u, v);
}
`

// The vertex stage's block: the draw's matrices, its active morph targets
// (bases into the deltas and weights, 4 a row; their count in
// morph_info.x) and where a skinned draw's joints are in the frame's
// joint table (`joint_table`: x now, y at the shutter's opening); the
// frame's depth pass with motion blur: the same at the shutter's opening.
Draw_Block :: struct {
	mvp:                  Mat4,
	model:                Mat4,
	normal:               Mat4,
	morph_bases:          [2][4]u32,
	morph_weights:        [2][4]f32,
	morph_info:           [4]u32,
	skin:                 [4]u32,
	opened_mvp:           Mat4,
	opened_morph_bases:   [2][4]u32,
	opened_morph_weights: [2][4]f32,
	opened_morph_info:    [4]u32,
}

// The fragment stage's block: the material's part, from its
// Shading_Material and its geometry.
Material_Block :: struct {
	color:           [4]f32, // linear base, opacity
	emissive:        [4]f32, // linear, times intensity
	surface:         [4]f32, // metalness, roughness, ao intensity, alpha cutoff
	uv_transforms:   [2 * len(Texture_Slot)][4]f32,
	triplanar:       [4]f32, // repeats per unit, the weights' power
	to_local:        [3][4]f32,
	normal_to_local: [3][4]f32,
	layers:          [4]f32, // f0, clear coat, its roughness, sheen roughness
	sheen:           [4]f32, // sheen colour (linear); w the material's environment intensity
	normal_scales:   [4]f32, // the normal map's x and y scale, the coat normal map's
	transmission:    [4]f32, // transmission, thickness (the model's units), ior
	scale:           [4]f32, // the model's axis scales
	attenuation:     [4]f32, // absorption per world unit
	particle:        [4]f32, // particles: fade distance before the frame's depth
	table:           [4]u32, // the geometry's table: an exact shape's curve sets, particles' rows
}

// The fragment stage's block: the pass's part.
Frame_Block :: struct {
	eye:           [4]f32,
	view_proj:     Mat4, // glass's refraction
	frame:         [4]f32, // the pass's width and height
	camera:        [4]f32, // near, far, 1 when orthographic
	reflect:       [4]f32, // the reflection's last level, its focal (Mirror_View.focal)
	shadow_matrix: [MAX_SHADOWS]Mat4,
	shadow_params: [MAX_SHADOWS][4]f32, // map size, bias, normal bias
	shadow_from:   [MAX_SHADOWS][4]f32,
	env:           [4]f32, // intensity, the radiance's last level
	fog:           [4]f32, // near, far, density
	fog_color:     [4]f32, // linear
	clip_plane:    [4]f32, // the mirror's pass: what is behind it is left out
	// Per light: colour times intensity (w: a spot's cos outer), ground
	// (w: cos inner; a spot's axis), its direction or position, and
	// distance and decay.
	light_color:   [MAX_LIGHTS][4]f32,
	light_ground:  [MAX_LIGHTS][4]f32,
	light_vector:  [MAX_LIGHTS][4]f32,
	light_falloff: [MAX_LIGHTS][4]f32,
}

// What a draw is and which pass draws it (o3d.frag's ENTRY).
Program_Entry :: enum u8 {
	Main, // replaces what is behind it
	Cut, // masked, blended or added
	Mirror, // in the mirror's pass
	Sprite, // a particle quad
	Curves, // an exact shape
	Curves_Mirror, // an exact shape in the mirror's pass
	Shadow, // a shadow map's caster (o3d_depth.frag)
	Velocity, // in the frame's depth pass with motion blur (o3d_depth.frag)
}

// o3d.vert's GEOMETRY.
Program_Geometry :: enum u8 {
	Mesh,
	Skinned,
	Quads, // an exact shape's or particles' quads
}

// The frame's part of a variant's shape: its lights (kind + 1, 0 none;
// shadow slot + 1, 0 none), shadow maps, environment and fog.
Frame_Shape :: struct {
	lights:  [MAX_LIGHTS][2]u8,
	shadows: u8,
	env:     bool,
	env_hdr: bool,
	fog:     Fog_Kind,
}

Program_Key :: struct {
	flags:       Shading_Flags,
	uv_sets:     u32,
	layer_slots: u32,
	entry:       Program_Entry,
	geometry:    Program_Geometry,
	frame:       Frame_Shape,
}

// What a fragment channel binds, by its sampler's name.
Program_Map :: enum u8 {
	None,
	Color,
	Normal,
	Roughness,
	Emissive,
	Metalness,
	Ao,
	Alpha,
	Env_Radiance,
	Env_Irradiance,
	Shadow0,
	Shadow1,
	Shadow2,
	Shadow3,
	Layer0,
	Layer1,
	// sampler2DArray: glass's frame behind (layer 0), the reflection (its
	// light, layer 0; its runs, layer 1).
	Frame,
	// Particles: the frame's depth.
	Scene_Depth,
}

@(rodata)
PROGRAM_MAP_NAMES := [Program_Map]string {
	.None           = "",
	.Color          = "colorMap",
	.Normal         = "normalMap",
	.Roughness      = "roughnessMap",
	.Emissive       = "emissiveMap",
	.Metalness      = "metalnessMap",
	.Ao             = "aoMap",
	.Alpha          = "alphaMap",
	.Env_Radiance   = "envRadianceMap",
	.Env_Irradiance = "envIrradianceMap",
	.Shadow0        = "shadowMap0",
	.Shadow1        = "shadowMap1",
	.Shadow2        = "shadowMap2",
	.Shadow3        = "shadowMap3",
	.Layer0         = "layerMap0",
	.Layer1         = "layerMap1",
	.Frame          = "frameMap",
	.Scene_Depth    = "sceneDepth",
}

// o3d.vert's attributes: their names, where they are in a Vertex or a
// Skin_Vertex (`skin`), and their formats.
Program_Attribute :: enum u8 {
	Position,
	Normal,
	Uv,
	Uv1,
	Color,
	Joints,
	Weights,
}

@(rodata)
PROGRAM_ATTRIBUTES := [Program_Attribute]struct {
	name:   string,
	skin:   bool,
	offset: uintptr,
	format: oksl.Vertex_Format,
} {
	.Position = {"position", false, offset_of(Vertex, position), .Float3},
	.Normal   = {"normal", false, offset_of(Vertex, normal), .Float3},
	.Uv       = {"uv", false, offset_of(Vertex, uv), .Float2},
	.Uv1      = {"uv1", false, offset_of(Vertex, uv1), .Float2},
	.Color    = {"color", false, offset_of(Vertex, color), .Ubyte4_Norm},
	.Joints   = {"joints", true, offset_of(Skin_Vertex, joints), .Ushort4},
	.Weights  = {"weights", true, offset_of(Skin_Vertex, weights), .Float4},
}

// A storage buffer a stage reads, by its block's name.
Program_Buffer :: enum u8 {
	None,
	Morph_Deltas, // the vertex stage's: Geometries.morph_deltas
	Skin_Joints, // the vertex stage's: the frame's joint table (`joint_table`)
	Sprite_Rows, // particles' rows (Expanded_Draws.sprite_rows)
	Curve_Words, // exact shapes' curves (Geometries.curve_words)
}

@(rodata)
PROGRAM_BUFFER_NAMES := [Program_Buffer]string {
	.None         = "",
	.Morph_Deltas = "Morph_Deltas",
	.Skin_Joints  = "Skin_Joints",
	.Sprite_Rows  = "Sprite_Rows",
	.Curve_Words  = "Curve_Words",
}

// A field a stage's block holds: from the struct (part 0 or 1 of the
// stage's parts) at `src` to the block at `dst`.
Block_Field :: struct {
	part, src, dst, size: int,
}

BLOCK_FIELDS :: 32

Block_Fields :: struct {
	fields: [BLOCK_FIELDS]Block_Field,
	count:  int,
	size:   int, // the block's bytes (oksl's block_size)
}

// A compiled variant and what its stages read, found once.
Program :: struct {
	key:        Program_Key,
	// nil: it failed (`errors`).
	pipeline:   ^oksl.Pipeline,
	errors:     string,
	vertex:     Block_Fields, // from Draw_Block
	fragment:   Block_Fields, // from Material_Block and Frame_Block
	// Each attribute's location (-1: not read).
	attributes: [Program_Attribute]i32,
	// Each stage's storage buffers in binding order.
	vertex_buffers:   [2]Program_Buffer,
	fragment_buffers: [2]Program_Buffer,
	maps:       [oksl.SAMPLER_COUNT]Program_Map,
	// What the software renderer's triangle corners keep of the vertex
	// stage's record.
	corners:    Corner_Floats,
}

// A renderer's variants, compiled the first time a draw needs one, and
// its fullscreen passes' programs (`pass_get`).
Programs :: struct {
	list:   [dynamic]Program,
	keys:   map[Program_Key]i32,
	passes: [Pass]Pass_Program,
}

programs_destroy :: proc(p: ^Programs) {
	for &x in p.list {
		if x.pipeline != nil {
			oksl.destroy_pipeline(x.pipeline)
		}
		delete(x.errors)
	}
	for &x in p.passes {
		if x.pipeline != nil {
			oksl.destroy_pipeline(x.pipeline)
		}
		delete(x.errors)
	}
	delete(p.list)
	delete(p.keys)
	p^ = {}
}

frame_shape :: proc "contextless" (lights: []Shading_Light, shadows: int, env: Shading_Environment, env_hdr: bool, fog: Fog_Kind) -> (f: Frame_Shape) {
	for l, i in lights {
		f.lights[i] = {u8(l.kind) + 1, u8(l.shadowed)}
	}
	f.shadows = u8(shadows)
	f.env = env.radiance != 0 && env.intensity > 0
	f.env_hdr = f.env && env_hdr
	f.fog = fog
	return
}

// A draw's key: its material's shape, how it is drawn (`mirrored`: in the
// mirror's pass) and the frame's.
program_key :: proc "contextless" (m: ^Shading_Material, skinned, mirrored: bool, frame: Frame_Shape) -> Program_Key {
	entry: Program_Entry
	geometry := Program_Geometry.Skinned if skinned else .Mesh
	switch {
	case .Sprite in m.flags:
		entry, geometry = .Sprite, .Quads
	case .Curves in m.flags:
		entry, geometry = .Curves_Mirror if mirrored else .Curves, .Quads
	case mirrored:
		entry = .Mirror
	case blend_of(m) != .Opaque:
		entry = .Cut
	}
	layer_slots := m.layer_slots if m.flags & {.Layer_Maps, .Clearcoat_Normal_Map} != {} else 0
	return {flags = m.flags, uv_sets = m.uv_sets, layer_slots = layer_slots, entry = entry, geometry = geometry, frame = frame}
}

// A draw's key in a depth pass (`velocity`: the frame's, with motion
// blur; else a shadow map's): only what those read of its material.
depth_key :: proc "contextless" (m: ^Shading_Material, skinned, velocity: bool) -> Program_Key {
	DEPTH_FLAGS :: Shading_Flags{.Alpha_Mask, .Color_Map, .Alpha_Map, .Triplanar, .Curves}
	geometry := Program_Geometry.Skinned if skinned else .Mesh
	if .Curves in m.flags || .Sprite in m.flags {
		geometry = .Quads
	}
	sets := m.uv_sets & (1 << u32(Texture_Slot.Color) | 1 << u32(Texture_Slot.Alpha))
	return {flags = m.flags & DEPTH_FLAGS, uv_sets = sets, entry = .Velocity if velocity else .Shadow, geometry = geometry}
}

// The variant for `key` (-1: it does not compile; `errors` says why, once
// on stderr), compiled the first time it is asked for.
program_get :: proc(p: ^Programs, key: Program_Key) -> i32 {
	if found, ok := p.keys[key]; ok {
		return found if p.list[found].pipeline != nil else -1
	}
	x := program_compile(key)
	if x.pipeline == nil {
		fmt.eprintfln("o3d: shader variant %v failed:\n%s", key, x.errors)
	}
	append(&p.list, x)
	id := i32(len(p.list) - 1)
	p.keys[key] = id
	return id if x.pipeline != nil else -1
}

@(private = "file")
program_compile :: proc(key: Program_Key) -> (x: Program) {
	runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()
	x.key = key
	vp, fp := vertex_prelude(key), fragment_prelude(key)
	fragment := DEPTH_GLSL if key.entry == .Shadow || key.entry == .Velocity else FRAGMENT_GLSL
	p, errors, ok := oksl.compile_pipeline({vertex = VERTEX_GLSL, fragment = fragment, vertex_prelude = vp, fragment_prelude = fp})
	if !ok {
		x.errors = strings.clone(oksl.format_errors(errors, context.temp_allocator))
		oksl.destroy_pipeline(p)
		return
	}
	bad := strings.builder_make(context.temp_allocator)
	x.vertex = block_fields(p.vertex, {Draw_Block, nil}, &bad)
	x.fragment = block_fields(p.fragment, {Material_Block, Frame_Block}, &bad)
	if strings.builder_len(bad) > 0 {
		x.errors = strings.clone(strings.to_string(bad))
		oksl.destroy_pipeline(p)
		return
	}
	x.attributes = {.Position = -1, .Normal = -1, .Uv = -1, .Uv1 = -1, .Color = -1, .Joints = -1, .Weights = -1}
	for in_ in p.vertex.stage_inputs {
		for a in Program_Attribute {
			if in_.name == PROGRAM_ATTRIBUTES[a].name {
				x.attributes[a] = i32(in_.location)
			}
		}
	}
	x.corners = corner_floats(p)
	for b, i in p.vertex.storage_buffers {
		x.vertex_buffers[i] = program_buffer(b.name)
	}
	for b, i in p.fragment.storage_buffers {
		x.fragment_buffers[i] = program_buffer(b.name)
	}
	for name, ch in p.fragment.channel_names {
		for k in Program_Map {
			if k != .None && name == PROGRAM_MAP_NAMES[k] {
				x.maps[ch] = k
			}
		}
	}
	x.pipeline = p
	return
}

@(private = "file")
program_buffer :: proc(name: string) -> Program_Buffer {
	for b in Program_Buffer {
		if b != .None && name == PROGRAM_BUFFER_NAMES[b] {
			return b
		}
	}
	return .None
}

// A stage's field table: each uniform it reads, from the part whose field
// it is (u_<field>), its size checked against oksl's layout.
@(private = "file")
block_fields :: proc(m: ^oksl.Module, parts: [2]typeid, bad: ^strings.Builder) -> (out: Block_Fields) {
	out.size = m.block_size
	limit := 0
	for t in parts {
		if t != nil {
			limit += type_info_of(t).size
		}
	}
	if m.block_size > limit {
		fmt.sbprintf(bad, "a block of %d bytes, more than its parts' %d\n", m.block_size, limit)
	}
	for t, part in parts {
		if t == nil {
			continue
		}
		for f in reflect.struct_fields_zipped(t) {
			name := strings.concatenate({"u_", f.name}, context.temp_allocator)
			for u in m.uniforms {
				if u.name != name {
					continue
				}
				size := u.count * u.stride if u.count > 0 else u.size
				if size != f.type.size {
					fmt.sbprintf(bad, "%s: %d bytes in the shader, %d in %v\n", name, size, f.type.size, t)
				}
				out.fields[out.count] = {part, int(f.offset), u.offset, f.type.size}
				out.count += 1
			}
		}
	}
	return
}

// A stage's block for a draw: the fields it reads, from its parts, into
// `out` (Block_Fields.size bytes; std140's gaps are left as they are).
block_fill :: proc "contextless" (fields: ^Block_Fields, parts: [2]rawptr, out: []u8) {
	for f in fields.fields[:fields.count] {
		copy(out[f.dst:][:f.size], ([^]u8)(parts[f.part])[f.src:][:f.size])
	}
}

// The parts' declarations, from their fields: u_<field>.
@(private = "file")
write_uniforms :: proc(b: ^strings.Builder, parts: []typeid) {
	for t in parts {
		for f in reflect.struct_fields_zipped(t) {
			name, count := glsl_type(f.type)
			if count > 0 {
				fmt.sbprintf(b, "uniform %s u_%s[%d];\n", name, f.name, count)
			} else {
				fmt.sbprintf(b, "uniform %s u_%s;\n", name, f.name)
			}
		}
	}
}

// A field's GLSL type (and array length), for the types whose std140
// layout is Odin's: 4-byte scalars, [4] vectors of them, Mat4, and arrays
// of those 16- and 64-byte elements.
@(private = "file")
glsl_type :: proc(ti: ^runtime.Type_Info) -> (name: string, count: int) {
	ti := runtime.type_info_base(ti)
	#partial switch v in ti.variant {
	case runtime.Type_Info_Float:
		return "float", 0
	case runtime.Type_Info_Integer:
		return "int" if v.signed else "uint", 0
	case runtime.Type_Info_Matrix:
		assert(v.row_count == 4 && v.column_count == 4)
		return "mat4", 0
	case runtime.Type_Info_Array:
		elem := runtime.type_info_base(v.elem)
		#partial switch e in elem.variant {
		case runtime.Type_Info_Float:
			assert(v.count == 4, "only [4] vectors share std140's layout")
			return "vec4", 0
		case runtime.Type_Info_Integer:
			assert(v.count == 4, "only [4] vectors share std140's layout")
			return "ivec4" if e.signed else "uvec4", 0
		}
		inner, n := glsl_type(v.elem)
		assert(n == 0 && (elem.size == 16 || elem.size == 64), "arrays of 16- or 64-byte elements only")
		return inner, v.count
	}
	panic("a type std140 lays out differently")
}

@(private = "file")
vertex_prelude :: proc(key: Program_Key) -> string {
	b := strings.builder_make(context.temp_allocator)
	strings.write_string(&b, O3D_COMMON)
	fmt.sbprintf(&b, "#define GEOMETRY %d\n", int(key.geometry))
	pass := 1 if key.entry == .Shadow else (2 if key.entry == .Velocity else 0)
	fmt.sbprintf(&b, "#define PASS %d\n", pass)
	write_uniforms(&b, {Draw_Block})
	return strings.to_string(b)
}

// The uv sets a variant's fragment reads (o3d.frag and o3d_depth.frag
// declare a set's varying only then, so a map that reads another fails to
// compile): its maps' sets (a slot is read when its map's flag is set, a
// layer's when the material has that layer's texture; triplanar maps read
// none), a particle's sprite set 0, an exact shape's coverage set 1. The
// depth passes read one set, for an alpha mask: the colour map's, else
// the alpha map's.
#assert(int(Texture_Slot.Clearcoat_Normal) - int(Texture_Slot.Clearcoat) == int(Layer_Map.Clearcoat_Normal) && int(Texture_Slot.Thickness) - int(Texture_Slot.Clearcoat) == int(Layer_Map.Thickness))

@(private = "file")
uv_sets_read :: proc "contextless" (key: Program_Key) -> (sets: bit_set[0 ..< 2]) {
	if .Curves in key.flags {
		sets += {1}
	}
	if key.entry == .Sprite {
		sets += {0}
	}
	if .Triplanar in key.flags {
		return
	}
	slots: bit_set[Texture_Slot]
	if key.entry == .Shadow || key.entry == .Velocity {
		if .Alpha_Mask in key.flags && .Color_Map in key.flags {
			slots += {.Color}
		} else if .Alpha_Mask in key.flags && .Alpha_Map in key.flags {
			slots += {.Alpha}
		}
	} else {
		MAPS :: [?]struct {
			flag: Shading_Flag,
			slot: Texture_Slot,
		}{{.Color_Map, .Color}, {.Normal_Map, .Normal}, {.Roughness_Map, .Roughness}, {.Emissive_Map, .Emissive}, {.Metalness_Map, .Metalness}, {.Ao_Map, .Ao}, {.Alpha_Map, .Alpha}}
		for m in MAPS {
			if m.flag in key.flags {
				slots += {m.slot}
			}
		}
		// A layer's slot follows the clear coat's in Texture_Slot as Layer_Map
		// lists them.
		for lm in Layer_Map {
			t := (key.layer_slots >> (4 * u32(lm))) & 15
			flag := Shading_Flag.Clearcoat_Normal_Map if lm == .Clearcoat_Normal else .Layer_Maps
			if (t == 1 || t == 2) && flag in key.flags {
				slots += {Texture_Slot(u8(Texture_Slot.Clearcoat) + u8(lm))}
			}
		}
	}
	for s in slots {
		sets += {1} if key.uv_sets & (1 << u32(s)) != 0 else {0}
	}
	return
}

@(private = "file")
fragment_prelude :: proc(key: Program_Key) -> string {
	b := strings.builder_make(context.temp_allocator)
	strings.write_string(&b, O3D_COMMON)
	fmt.sbprintf(&b, "#define ENTRY %d\n", int(key.entry))
	for f in key.flags {
		fmt.sbprintf(&b, "#define FLAG_%s 1\n", strings.to_upper(reflect.enum_string(f), context.temp_allocator))
	}
	// The depth passes' alpha mask (shadow_mask's).
	if .Alpha_Mask in key.flags && key.flags & {.Color_Map, .Alpha_Map} != {} && .Triplanar not_in key.flags {
		strings.write_string(&b, "#define MASKED 1\n")
	}
	lit := key.flags & {.Basic, .Shadow_Only} == {}
	if lit {
		strings.write_string(&b, "#define LIT 1\n")
	}
	if key.frame.env {
		strings.write_string(&b, "#define ENV 1\n")
	}
	if key.frame.env_hdr {
		strings.write_string(&b, "#define ENV_HDR 1\n")
	}
	fmt.sbprintf(&b, "#define FOG %d\n", int(key.frame.fog))
	shadow_maps := 0
	if .Shadow_Only in key.flags || lit && .Receive_Shadow in key.flags {
		shadow_maps = int(key.frame.shadows)
	}
	fmt.sbprintf(&b, "#define SHADOW_MAPS %d\n", shadow_maps)
	// Each slot's transform rows and uv set (its varying and derivatives),
	// and the sets the variant reads.
	for s in Texture_Slot {
		name := strings.to_upper(reflect.enum_string(s), context.temp_allocator)
		fmt.sbprintf(&b, "#define ROW0_%s %d\n#define ROW1_%s %d\n", name, 2 * int(s), name, 2 * int(s) + 1)
		if key.uv_sets & (1 << u32(s)) != 0 {
			fmt.sbprintf(&b, "#define UV_%s v_uv1\n#define DUV_%s g_d1x, g_d1y\n", name, name)
		} else {
			fmt.sbprintf(&b, "#define UV_%s v_uv\n#define DUV_%s g_d0x, g_d0y\n", name, name)
		}
	}
	for set in uv_sets_read(key) {
		fmt.sbprintf(&b, "#define UV%d 1\n", set)
	}
	// The layers' maps: in layerMap0, layerMap1, or none (white).
	textures := 0
	for lm in Layer_Map {
		t := (key.layer_slots >> (4 * u32(lm))) & 15
		name := strings.to_upper(reflect.enum_string(lm), context.temp_allocator)
		if t == 1 || t == 2 {
			fmt.sbprintf(&b, "#define LAYER_%s MAP(layerMap%d, %s)\n#define LAYER_TEX_%s %d\n", name, t - 1, name, name, t)
			textures = max(textures, int(t))
		} else {
			fmt.sbprintf(&b, "#define LAYER_%s vec4(1.0)\n#define LAYER_TEX_%s 0\n", name, name)
		}
	}
	fmt.sbprintf(&b, "#define LAYER_TEXTURES %d\n", textures)
	// The shadow maps' factors at the point.
	strings.write_string(&b, "#define RECEIVE_SHADOWS")
	for i in 0 ..< shadow_maps {
		fmt.sbprintf(&b, " g_shadows[%d] = shadow_factor(shadowMap%d, u_shadow_matrix[%d], u_shadow_params[%d], u_shadow_from[%d], v_world, n);", i, i, i, i, i)
	}
	strings.write_string(&b, "\n#define SHADOW_ONLY_MASK")
	for i in 0 ..< shadow_maps {
		fmt.sbprintf(&b, " mask *= shadow_factor(shadowMap%d, u_shadow_matrix[%d], u_shadow_params[%d], u_shadow_from[%d], v_world, sn);", i, i, i, i)
	}
	// The lights, each by its kind (no braces: fmt reads them).
	strings.write_string(&b, "\n#define O3D_LIGHTS")
	for l, k in key.frame.lights {
		if l[0] == 0 {
			continue
		}
		shadow := fmt.tprintf("g_shadows[%d]", l[1] - 1) if l[1] > 0 && int(l[1]) <= MAX_SHADOWS else "1.0"
		switch Light_Kind(l[0] - 1) {
		case .Ambient:
			fmt.sbprintf(&b, " s_out_d += u_light_color[%d].rgb * g_diffuse * (g_ao / PI);", k)
		case .Hemisphere:
			fmt.sbprintf(&b, " s_out_d += mix(u_light_ground[%d].rgb, u_light_color[%d].rgb, 0.5 * g_n.y + 0.5) * g_diffuse * (g_ao / PI);", k, k)
		case .Directional:
			fmt.sbprintf(&b, " direct(u_light_color[%d].rgb * %s, u_light_vector[%d].xyz);", k, shadow, k)
		case .Point:
			fmt.sbprintf(&b, " point_direct(u_light_color[%d].rgb, u_light_vector[%d].xyz, u_light_falloff[%d], %s);", k, k, k, shadow)
		case .Spot:
			fmt.sbprintf(&b, " spot_direct(u_light_color[%d], u_light_ground[%d], u_light_vector[%d].xyz, u_light_falloff[%d], %s);", k, k, k, k, shadow)
		}
	}
	strings.write_string(&b, "\n")
	write_uniforms(&b, {Material_Block, Frame_Block})
	if .Curves in key.flags {
		strings.write_string(&b, curves.GLSL)
	}
	return strings.to_string(b)
}

// ---- the blocks -------------------------------------------------------------------------

// The vertex stage's block for a draw: `mvp` and `model`, its active morph
// targets (`weights`), and where a skinned draw's joints are in the frame's
// joint table (`skin`: now and at the shutter's opening).
draw_block :: proc "contextless" (out: ^Draw_Block, mvp, model: Mat4, geo: Geometry, weights: []f32, skin: [2]u32) {
	out.mvp, out.model, out.normal = mvp, model, normal_matrix(model)
	morphs: [MAX_MORPH_TARGETS]Active_Morph
	n := active_morphs(geo, weights, &morphs)
	out.morph_bases, out.morph_weights = {}, {}
	for i in 0 ..< n {
		out.morph_bases[i >> 2][i & 3] = morphs[i].base
		out.morph_weights[i >> 2][i & 3] = morphs[i].weight
	}
	out.morph_info = {u32(n), 0, 0, 0}
	out.skin = {skin[0], skin[1], 0, 0}
}

// The vertex stage's block's part at the shutter's opening (the frame's
// depth pass with motion blur): the draw's model-view-projection then and
// its morph weights then (its joints then: Draw_Block.skin.y).
draw_block_opened :: proc "contextless" (out: ^Draw_Block, mvp: Mat4, geo: Geometry, weights: []f32) {
	out.opened_mvp = mvp
	morphs: [MAX_MORPH_TARGETS]Active_Morph
	n := active_morphs(geo, weights, &morphs)
	out.opened_morph_bases, out.opened_morph_weights = {}, {}
	for i in 0 ..< n {
		out.opened_morph_bases[i >> 2][i & 3] = morphs[i].base
		out.opened_morph_weights[i >> 2][i & 3] = morphs[i].weight
	}
	out.opened_morph_info = {u32(n), 0, 0, 0}
}

// The frame's joint table: every skinned draw's joints in one array (the
// vertex stage's Skin_Joints), now and, with `opened`, at the shutter's
// opening (as now when it has none); each draw's first index of each in
// `bases` (0 for a draw without joints).
joint_table :: proc(scene: ^Scene, opened: bool, table: ^[dynamic]Mat4, bases: ^[dynamic][2]u32) {
	n := len(scene.draws)
	clear(table)
	table_grow_non_zero(bases, n)
	for i in 0 ..< n {
		joints := scene_draw_joints(scene, i)
		if len(joints) == 0 {
			bases[i] = {}
			continue
		}
		bases[i] = u32(len(table))
		append(table, ..joints)
		if opened {
			if then := scene_draw_opened_joints(scene, i); then != nil {
				bases[i][1] = u32(len(table))
				append(table, ..then)
			}
		}
	}
}

// The material's part (`frame`: a .Triplanar material's projection;
// `table`: an exact shape's or particles' geometry's table).
material_block :: proc "contextless" (m: ^Shading_Material, frame: Projection_Frame = {}, table: u32 = 0) -> (out: Material_Block) {
	out = {
		color         = {m.base.r, m.base.g, m.base.b, m.opacity},
		emissive      = {m.emissive.r, m.emissive.g, m.emissive.b, 0},
		surface       = {m.metalness, m.roughness, m.ao_intensity, m.alpha_cutoff},
		layers        = {m.layers.f0, m.layers.clearcoat, m.layers.clearcoat_roughness, m.layers.sheen_roughness},
		sheen         = {m.layers.sheen.r, m.layers.sheen.g, m.layers.sheen.b, m.env_intensity},
		normal_scales = {m.normal_scale.x, m.normal_scale.y, m.layers.coat_normal_scale.x, m.layers.coat_normal_scale.y},
		transmission  = {m.layers.transmission, m.layers.thickness, m.layers.ior, 0},
		scale         = {m.layers.scale.x, m.layers.scale.y, m.layers.scale.z, 0},
		attenuation   = {m.layers.attenuation.r, m.layers.attenuation.g, m.layers.attenuation.b, 0},
		particle      = {m.softness, 0, 0, 0},
		table         = {table, 0, 0, 0},
	}
	for s, i in Texture_Slot {
		t := m.uv_transforms[s]
		out.uv_transforms[i * 2] = {t.linear[0, 0], t.linear[0, 1], t.offset.x, 0}
		out.uv_transforms[i * 2 + 1] = {t.linear[1, 0], t.linear[1, 1], t.offset.y, 0}
	}
	if .Triplanar in m.flags {
		out.triplanar = {m.projection_scale, m.projection_sharpness, 0, 0}
		to_local, normal_to_local := Row_Mat4(frame.to_local), Row_Mat4(Mat4(frame.normal_to_local))
		out.to_local = {to_local[0], to_local[1], to_local[2]}
		out.normal_to_local = {normal_to_local[0], normal_to_local[1], normal_to_local[2]}
	}
	return
}

// The pass's part, as both renderers make it: through `camera` at `eye`
// (the mirror's pass: its image) and `view_proj`, `width x height`.
frame_block :: proc "contextless" (out: ^Frame_Block, camera: Camera, eye: Vec3, view_proj: Mat4, width, height: int, lights: []Shading_Light, shadows: []Shadow, env: Shading_Environment, fog: Fog) {
	out^ = {}
	near := camera.near if camera.near > 0 else 0.1
	far := camera.far if camera.far > near else near * 1e4
	out.eye = {eye.x, eye.y, eye.z, 0}
	out.view_proj = view_proj
	out.frame = {f32(width), f32(height), 0, 0}
	out.camera = {near, far, 1 if camera.orthographic else 0, 0}
	for sh, i in shadows {
		out.shadow_matrix[i] = sh.view_proj
		out.shadow_params[i] = {f32(sh.size), sh.bias, sh.normal_bias, 0}
		out.shadow_from[i] = sh.from
	}
	if env.radiance != 0 {
		out.env = {env.intensity, env.max_lod, 0, 0}
	}
	if fog.kind != .None {
		fog_near, fog_far, density := fog_terms(fog)
		fc := linear(fog.color)
		out.fog = {fog_near, fog_far, density, 0}
		out.fog_color = {fc.r, fc.g, fc.b, 0}
	}
	for l, i in lights {
		out.light_color[i] = {l.color.r, l.color.g, l.color.b, l.cos_outer}
		out.light_ground[i] = {l.ground.r, l.ground.g, l.ground.b, l.cos_inner}
		out.light_vector[i] = {l.vector.x, l.vector.y, l.vector.z, 0}
		out.light_falloff[i] = {l.distance, l.decay, 0, 0}
	}
}

// ---- fullscreen passes ----------------------------------------------------------------

// o3d's fullscreen passes: shaders/fullscreen.vert and a fragment stage
// each, its uniforms from its block's fields (u_<field>).
Pass :: enum u8 {
	Background, // the frame's background (background.frag)
	Velocity_Far, // the frame's depth pass where nothing is drawn (velocity_far.frag)
	Depth_Image, // the frame's depth as distance (depth_image.frag)
	// post.frag's: the output, bloom's chain, the depth of field's phases
	// and the motion blur's.
	Output,
	Bloom_Bright,
	Bloom_Down,
	Bloom_Blur,
	Focus_Rows,
	Focus_Tiles,
	Focus_Near,
	Focus_Half,
	Focus_Gather,
	Motion_Rows,
	Motion_Tiles,
	Motion_Around,
	Motion_Gather,
}

POST_PASSES :: bit_set[Pass]{.Output, .Bloom_Bright, .Bloom_Down, .Bloom_Blur, .Focus_Rows, .Focus_Tiles, .Focus_Near, .Focus_Half, .Focus_Gather, .Motion_Rows, .Motion_Tiles, .Motion_Around, .Motion_Gather}

// What a pass's sampler reads, by its name.
Pass_Map :: enum u8 {
	None,
	Light,
	Bloom,
	Depth,
	Tiles,
	Half,
	Half_Keys,
	Velocity,
	Nearness,
	Motion,
	Background,
}

@(rodata)
PASS_MAP_NAMES := [Pass_Map]string {
	.None       = "",
	.Light      = "lightMap",
	.Bloom      = "bloomMap",
	.Depth      = "depthMap",
	.Tiles      = "tileMap",
	.Half       = "halfMap",
	.Half_Keys  = "halfKeyMap",
	.Velocity   = "velocityMap",
	.Nearness   = "nearnessMap",
	.Motion     = "motionMap",
	.Background = "bgMap",
}

@(private = "file")
POST_GLSL :: #load("shaders/post.frag", string)

// The post passes' uniforms: the tone mapping, exposure, the light
// texture's size; the focus, blur and its most (pixels), the tiles a blur
// reaches; the camera's near, far, orthographic; the frame's size, its
// tiles across and down; the motion blur's longest half streak, nearness's
// scale and tiles' size, the velocity maps' size and motion tiles across and
// down; bloom's threshold, sigma and gain, its Gaussian's axis and the size
// of the level drawn (post.frag says more).
Post_Block :: struct {
	tone:        [4]f32,
	lens:        [4]f32,
	camera:      [4]f32,
	frame:       [4]f32,
	motion:      [4]f32,
	motion_size: [4]f32,
	bloom:       [4]f32,
	bloom_size:  [4]f32,
}

@(private = "file", rodata)
PASS_SOURCES := #partial [Pass]string {
	.Background   = #load("shaders/background.frag", string),
	.Velocity_Far = #load("shaders/velocity_far.frag", string),
	.Depth_Image  = #load("shaders/depth_image.frag", string),
}

@(private = "file")
FULLSCREEN_GLSL :: #load("shaders/fullscreen.vert", string)

@(private = "file", rodata)
PASS_BLOCKS := #partial [Pass]typeid {
	.Background   = Background_Block,
	.Velocity_Far = Far_Block,
	.Depth_Image  = Depth_Image_Block,
}

Background_Block :: struct {
	inv_view_proj: Mat4,
	frame:         [4]f32, // the target's width and height
	background:    [4]f32, // intensity, level, 1 when the map is linear (HDR)
}

Far_Block :: struct {
	// From a pixel's clip position on the far plane now to where it was at
	// the shutter's opening.
	opened_far: Mat4,
	frame:      [4]f32, // the target's width and height
}

Depth_Image_Block :: struct {
	params: [4]f32, // the camera's near and far, the far for empty pixels, 1 when orthographic
	frame:  [4]f32, // the target's width and height
}

// A fullscreen pass's program and its block's field table (nil pipeline:
// it failed, `errors`).
Pass_Program :: struct {
	pipeline: ^oksl.Pipeline,
	errors:   string,
	block:    Block_Fields,
	// What each fragment channel reads.
	maps:     [oksl.SAMPLER_COUNT]Pass_Map,
}

// What the passes share with Odin, defined in their prelude: the depth of
// field's tile size, the far plane's nearness, Tone_Mapping's values.
@(private = "file")
write_pass_constants :: proc(b: ^strings.Builder) {
	fmt.sbprintf(b, "#define FOCUS_TILE %d\n#define MOTION_FAR %f\n", FOCUS_TILE, f32(MOTION_FAR))
	for m in Tone_Mapping {
		fmt.sbprintf(b, "#define TONE_%s %d\n", strings.to_upper(reflect.enum_string(m), context.temp_allocator), int(m))
	}
}

// The pass's program, compiled the first time it is asked for (nil: it
// does not compile; stderr says why, once).
pass_get :: proc(p: ^Programs, pass: Pass) -> ^Pass_Program {
	x := &p.passes[pass]
	if x.pipeline == nil && x.errors == "" {
		runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()
		b := strings.builder_make(context.temp_allocator)
		strings.write_string(&b, O3D_COMMON)
		write_pass_constants(&b)
		source, block := PASS_SOURCES[pass], PASS_BLOCKS[pass]
		if pass in POST_PASSES {
			source, block = POST_GLSL, Post_Block
			fmt.sbprintf(&b, "#define PASS_%s 1\n", strings.to_upper(reflect.enum_string(pass), context.temp_allocator))
		}
		write_uniforms(&b, {block})
		pl, errors, ok := oksl.compile_pipeline({vertex = FULLSCREEN_GLSL, fragment = source, vertex_prelude = O3D_COMMON, fragment_prelude = strings.to_string(b)})
		bad := strings.builder_make(context.temp_allocator)
		if ok {
			x.block = block_fields(pl.fragment, {block, nil}, &bad)
			for name, ch in pl.fragment.channel_names {
				for k in Pass_Map {
					if k != .None && name == PASS_MAP_NAMES[k] {
						x.maps[ch] = k
					}
				}
			}
		} else {
			strings.write_string(&bad, oksl.format_errors(errors, context.temp_allocator))
		}
		if strings.builder_len(bad) > 0 {
			x.errors = strings.clone(strings.to_string(bad))
			fmt.eprintfln("o3d: the %v pass failed:\n%s", pass, x.errors)
			oksl.destroy_pipeline(pl)
			return nil
		}
		x.pipeline = pl
	}
	return x if x.pipeline != nil else nil
}
