package o3d

import "core:encoding/endian"
import "core:mem"
import "core:mem/virtual"
import "core:slice"

// USD's binary layers (`.usdc`, "Crate"), read in Odin: the file's tables
// as flat arrays by index, as OpenUSD's crateFile.cpp lays them out
// (version 0.4.0 and later: tokens, strings, fields, field sets, paths and
// specs, each compressed with LZ4 and USD's integer coding). A spec's
// fields are a range of `fields`; a field's value is an 8-byte ValueRep,
// read on demand by the typed getters below into the caller's buffers
// (a value inlined in the rep, or at the file offset it holds). Paths are a
// tree by parent index; `usd_child` finds a prim's child or property by
// element token. Composition (references) and the model are usd.odin's.

Usd_Spec_Type :: enum u8 {
	Unknown,
	Attribute,
	Connection,
	Expression,
	Mapper,
	Mapper_Arg,
	Prim,
	Pseudo_Root,
	Relationship,
	Relationship_Target,
	Variant,
	Variant_Set,
}

// Crate's value types (crateDataTypes.h), the ones o3d reads.
Usd_Type :: enum u8 {
	Invalid      = 0,
	Bool         = 1,
	UChar        = 2,
	Int          = 3,
	UInt         = 4,
	Int64        = 5,
	UInt64       = 6,
	Half         = 7,
	Float        = 8,
	Double       = 9,
	String       = 10,
	Token        = 11,
	Asset_Path   = 12,
	Matrix2d     = 13,
	Matrix3d     = 14,
	Matrix4d     = 15,
	Quatd        = 16,
	Quatf        = 17,
	Quath        = 18,
	Vec2d        = 19,
	Vec2f        = 20,
	Vec2h        = 21,
	Vec2i        = 22,
	Vec3d        = 23,
	Vec3f        = 24,
	Vec3h        = 25,
	Vec3i        = 26,
	Vec4d        = 27,
	Vec4f        = 28,
	Vec4h        = 29,
	Vec4i        = 30,
	Dictionary   = 31,
	Token_List_Op = 32,
	String_List_Op = 33,
	Path_List_Op = 34,
	Reference_List_Op = 35,
	Path_Vector  = 40,
	Token_Vector = 41,
	Specifier    = 42,
	Permission   = 43,
	Variability  = 44,
	Time_Samples = 46,
	Payload      = 47,
	Double_Vector = 48,
	Value_Block  = 51,
	Value        = 52,
	Payload_List_Op = 55,
}

Usd_Specifier :: enum u8 {
	Def,
	Over,
	Class,
}

Usd_Path :: struct {
	parent:   i32, // -1: the root
	token:    i32, // the element's token (-1: the root)
	property: bool,
	spec:     i32, // the spec at this path, -1 if none
}

Usd_Field :: struct {
	token: u32,
	rep:   u64,
}

Usd_Spec :: struct {
	path:   u32,
	// Into `field_sets`: the spec's fields.
	fields: Gltf_Range,
	type:   Usd_Spec_Type,
}

// A reference's target: an asset ("" for this layer) and a prim path in it
// (-1: the asset's default prim).
Usd_Reference :: struct {
	asset: string,
	prim:  i32,
}

Usd_Layer :: struct {
	// The file (borrowed).
	bytes:      []u8,
	version:    [3]u8,
	tokens:     []string,
	// Each string's token.
	strings:    []u32,
	fields:     []Usd_Field,
	// Field indices, by spec ranges.
	field_sets: []u32,
	paths:      []Usd_Path,
	// The pseudo-root's path (the stage's metadata are its fields).
	root:       i32,
	specs:      []Usd_Spec,
	// A path's child by (parent, token, property): `usd_child`.
	children:   map[u64]i32,
	// The token of each name o3d asks for (-1: the file has none).
	names:      [Usd_Name]i32,
	budget:     Budget,
	arena:      ^virtual.Arena,
}

// The field and property names o3d reads, found once as tokens.
Usd_Name :: enum u8 {
	Specifier,
	Type_Name,
	Prim_Children,
	Default,
	Time_Samples,
	Target_Paths,
	Connection_Paths,
	References,
	Interpolation,
	Up_Axis,
	Meters_Per_Unit,
	Default_Prim,
	Active,
}

@(rodata)
USD_NAMES := [Usd_Name]string {
	.Specifier        = "specifier",
	.Type_Name        = "typeName",
	.Prim_Children    = "primChildren",
	.Default          = "default",
	.Time_Samples     = "timeSamples",
	.Target_Paths     = "targetPaths",
	.Connection_Paths = "connectionPaths",
	.References       = "references",
	.Interpolation    = "interpolation",
	.Up_Axis          = "upAxis",
	.Meters_Per_Unit  = "metersPerUnit",
	.Default_Prim     = "defaultPrim",
	.Active           = "active",
}

usd_layer_destroy :: proc(l: ^Usd_Layer) {
	if l.arena != nil {
		virtual.arena_destroy(l.arena)
		free(l.arena)
	}
	l^ = {}
}

// Reads a `.usdc` layer's tables (destroy it, error or not). Its values stay
// in `bytes`, which must outlive the layer.
usd_read_layer :: proc(bytes: []u8, limits: ^Limits = nil) -> (l: Usd_Layer, err: Usd_Error) {
	budget_init(&l.budget, limits)
	l.bytes = bytes
	l.root = -1
	if len(bytes) == 0 {
		return l, .Empty
	}
	if len(bytes) < 88 || string(bytes[:8]) != "PXR-USDC" {
		return l, .Not_Usdc if len(bytes) < 8 || string(bytes[:8]) != "PXR-USDC" else .Invalid
	}
	l.version = {bytes[8], bytes[9], bytes[10]}
	if l.version[0] != 0 || l.version[1] < 4 {
		return l, .Unsupported_Version
	}
	l.arena = new(virtual.Arena)
	if virtual.arena_init_growing(l.arena) != nil {
		free(l.arena)
		l.arena = nil
		return l, .Out_Of_Memory
	}
	context.allocator = virtual.arena_allocator(l.arena)
	toc := Usd_Cursor{l = &l, pos = 16}
	toc.pos = int(min(take_u64(&toc), u64(len(bytes))))
	sections: [Usd_Section]Usd_Cursor
	count := take_u64(&toc)
	if count > u64(len(bytes) / 32) {
		return l, .Invalid
	}
	for _ in 0 ..< count {
		name := take_bytes(&toc, 16)
		start, size := take_u64(&toc), take_u64(&toc)
		if toc.failed {
			return l, .Invalid
		}
		name_len := slice.linear_search(name, 0) or_else 16
		for sec in Usd_Section {
			if USD_SECTION_NAMES[sec] == string(name[:name_len]) {
				if start == 0 || start > u64(len(bytes)) || size > u64(len(bytes)) - start {
					return l, .Invalid
				}
				sections[sec] = {l = &l, pos = int(start), end = int(start + size)}
			}
		}
	}
	readers := [Usd_Section]proc(c: ^Usd_Cursor) {
		.Tokens     = read_tokens,
		.Strings    = read_strings,
		.Fields     = read_fields,
		.Field_Sets = read_field_sets,
		.Paths      = read_paths,
		.Specs      = read_specs,
	}
	for sec in Usd_Section {
		if sections[sec].end > 0 {
			readers[sec](&sections[sec])
			if sections[sec].failed {
				return l, .Limit if l.budget.hit != {} else .Invalid
			}
		}
	}
	token_of := make(map[string]i32, len(l.tokens), context.temp_allocator)
	for t, i in l.tokens {
		token_of[t] = i32(i)
	}
	for name in Usd_Name {
		l.names[name] = token_of[USD_NAMES[name]] or_else -1
	}
	return l, nil
}

Usd_Error :: enum u8 {
	None,
	Empty,
	File_Not_Found,
	Io,
	// Not a binary USD layer (a `.usda` text layer is one of these).
	Not_Usdc,
	// A Crate version before 0.4.0 or after 0.x.
	Unsupported_Version,
	// Corrupt or truncated.
	Invalid,
	// A USDZ without a `.usdc` layer at its root.
	No_Layer,
	Out_Of_Memory,
	// Past one of the `Limits`; `Model.warnings` says which.
	Limit,
}

@(rodata)
USD_ERROR_TEXT := [Usd_Error]string {
	.None                = "",
	.Empty               = "the file is empty",
	.File_Not_Found      = "the file was not found",
	.Io                  = "the file could not be read",
	.Not_Usdc            = "not a binary USD file (.usdc; text .usda is not read)",
	.Unsupported_Version = "a USD crate version o3d does not read (before 0.4.0)",
	.Invalid             = "a corrupt or truncated USD file",
	.No_Layer            = "the USDZ has no .usdc layer",
	.Out_Of_Memory       = "out of memory",
	.Limit               = "too large to read (a limit was reached)",
}

// The path's child (or property) with element token `token` (-1: none).
usd_child :: proc(l: ^Usd_Layer, parent: i32, token: i32, property: bool) -> i32 {
	if parent < 0 || token < 0 {
		return -1
	}
	return l.children[child_key(parent, token, property)] or_else -1
}

// The spec's field named `name` (false: none).
usd_field :: proc "contextless" (l: ^Usd_Layer, spec: i32, name: Usd_Name) -> (rep: u64, ok: bool) {
	t := l.names[name]
	if spec < 0 || t < 0 {
		return 0, false
	}
	s := l.specs[spec]
	for fi in l.field_sets[s.fields.first:][:s.fields.count] {
		if l.fields[fi].token == u32(t) {
			return l.fields[fi].rep, true
		}
	}
	return 0, false
}

// A field by its name as a token (attributes' own metadata, say).
usd_field_token :: proc "contextless" (l: ^Usd_Layer, spec: i32, token: i32) -> (rep: u64, ok: bool) {
	if spec < 0 || token < 0 {
		return 0, false
	}
	s := l.specs[spec]
	for fi in l.field_sets[s.fields.first:][:s.fields.count] {
		if l.fields[fi].token == u32(token) {
			return l.fields[fi].rep, true
		}
	}
	return 0, false
}

usd_rep_type :: #force_inline proc "contextless" (rep: u64) -> Usd_Type {
	return Usd_Type((rep >> 48) & 0xff)
}

usd_rep_is_array :: #force_inline proc "contextless" (rep: u64) -> bool {
	return rep & (1 << 63) != 0
}

// ---- values -----------------------------------------------------------------------------

// A token, asset path or string value ("", false: another type).
usd_string :: proc(l: ^Usd_Layer, value: u64) -> (s: string, ok: bool) {
	rep := resolve_rep(l, value) or_return
	if usd_rep_is_array(rep) {
		return "", false
	}
	x := u32(rep)
	#partial switch usd_rep_type(rep) {
	case .Token, .Asset_Path:
		return token_at(l, x)
	case .String:
		if int(x) >= len(l.strings) {
			return "", false
		}
		return token_at(l, l.strings[x])
	}
	return "", false
}

usd_bool :: proc(l: ^Usd_Layer, value: u64) -> (b: bool, ok: bool) {
	rep := resolve_rep(l, value) or_return
	if usd_rep_type(rep) != .Bool || usd_rep_is_array(rep) {
		return false, false
	}
	return u8(rep) != 0, true
}

// A number of any scalar type, as f64.
usd_number :: proc(l: ^Usd_Layer, value: u64) -> (number: f64, ok: bool) {
	rep := resolve_rep(l, value) or_return
	if usd_rep_is_array(rep) {
		return 0, false
	}
	inlined := rep & (1 << 62) != 0
	x := u32(rep)
	#partial switch usd_rep_type(rep) {
	case .Bool, .UChar:
		return f64(u8(x)), true
	case .Int:
		return f64(i32(x)), true
	case .UInt:
		return f64(x), true
	case .Half:
		return f64(f16(transmute(f16)u16(x))), true
	case .Float:
		return f64(transmute(f32)x), true
	case .Double:
		if inlined {
			return f64(transmute(f32)x), true
		}
		c := at(l, rep)
		v := take_f64(&c)
		return v, !c.failed
	case .Int64, .UInt64:
		if inlined {
			return f64(i32(x)), true
		}
		c := at(l, rep)
		v := take_u64(&c)
		return f64(i64(v)) if usd_rep_type(rep) == .Int64 else f64(v), !c.failed
	}
	return 0, false
}

// A vector or quaternion of up to 4 components (a quaternion's imaginary
// part first, then its real part, as GfQuat stores them).
usd_vector :: proc(l: ^Usd_Layer, value: u64) -> (v: [4]f64, n: int, ok: bool) {
	rep := resolve_rep(l, value) or_return
	if usd_rep_is_array(rep) {
		return
	}
	t := usd_rep_type(rep)
	comp, size := vector_shape(t)
	if comp == 0 {
		return
	}
	if rep & (1 << 62) != 0 {
		// Inlined: each component an int8.
		for i in 0 ..< comp {
			v[i] = f64(i8(u8(rep >> (8 * uint(i)))))
		}
		return v, comp, true
	}
	c := at(l, rep)
	for i in 0 ..< comp {
		v[i] = take_scalar(&c, size)
	}
	return v, comp, !c.failed
}

// A 2x2, 3x3 or 4x4 matrix, row-major as GfMatrix stores it (row vectors:
// the translation is the last row).
usd_matrix :: proc(l: ^Usd_Layer, value: u64) -> (m: [4][4]f64, n: int, ok: bool) {
	rep := resolve_rep(l, value) or_return
	if usd_rep_is_array(rep) {
		return
	}
	#partial switch usd_rep_type(rep) {
	case .Matrix2d:
		n = 2
	case .Matrix3d:
		n = 3
	case .Matrix4d:
		n = 4
	case:
		return
	}
	if rep & (1 << 62) != 0 {
		// Inlined: the diagonal as int8s.
		for i in 0 ..< n {
			m[i][i] = f64(i8(u8(rep >> (8 * uint(i)))))
		}
		return m, n, true
	}
	c := at(l, rep)
	for i in 0 ..< n {
		for j in 0 ..< n {
			m[i][j] = take_f64(&c)
		}
	}
	return m, n, !c.failed
}

// An int, uint or int64 array (or one int) into `out`, as i64.
usd_ints :: proc(l: ^Usd_Layer, value: u64, out: ^[dynamic]i64) -> (ok: bool) {
	clear(out)
	rep := resolve_rep(l, value) or_return
	t := usd_rep_type(rep)
	size: int
	#partial switch t {
	case .Int, .UInt:
		size = 4
	case .Int64, .UInt64:
		size = 8
	case:
		return false
	}
	if !usd_rep_is_array(rep) {
		v, one := usd_number(l, rep)
		append(out, i64(v))
		return one
	}
	if rep & 0xffff_ffff_ffff == 0 {
		return true
	}
	c := at(l, rep)
	n := take_count(&c)
	if c.failed || !grow(&c, out, n) {
		return false
	}
	if rep & (1 << 61) != 0 && n >= 16 {
		take_compressed_ints(&c, out[:], size)
	} else {
		for i in 0 ..< n {
			out[i] = i64(i32(take_u32(&c))) if size == 4 else i64(take_u64(&c))
		}
		if t == .UInt {
			for &v in out {
				v = i64(u32(v))
			}
		}
	}
	return !c.failed
}

// A float array of any precision, or an array of vectors (`components`
// each), into `out` as f32: `components` is what the value has.
usd_floats :: proc(l: ^Usd_Layer, value: u64, out: ^[dynamic]f32) -> (components: int, ok: bool) {
	clear(out)
	rep := resolve_rep(l, value) or_return
	if !usd_rep_is_array(rep) {
		return 0, false
	}
	t := usd_rep_type(rep)
	size: int
	#partial switch t {
	case .Half:
		components, size = 1, 2
	case .Float:
		components, size = 1, 4
	case .Double:
		components, size = 1, 8
	case:
		components, size = vector_shape(t)
	}
	if components == 0 {
		return 0, false
	}
	if rep & 0xffff_ffff_ffff == 0 {
		return components, true
	}
	c := at(l, rep)
	n := take_count(&c)
	if c.failed || n > max(int) / components || !grow(&c, out, n * components) {
		return components, false
	}
	if components == 1 && rep & (1 << 61) != 0 && n >= 16 {
		// Compressed scalars: integers ('i') or a table and indices ('t').
		code := take_bytes(&c, 1)
		switch code[0] if len(code) == 1 else 0 {
		case 'i':
			ints := make([dynamic]i64, n, context.temp_allocator)
			take_compressed_ints(&c, ints[:], 4)
			for v, i in ints {
				out[i] = f32(v)
			}
		case 't':
			table_len := int(take_u32(&c))
			if table_len * size > cursor_end(&c) - c.pos {
				fail(&c)
				return components, false
			}
			table := make([]f32, table_len, context.temp_allocator)
			for &v in table {
				v = f32(take_scalar(&c, size))
			}
			ints := make([dynamic]i64, n, context.temp_allocator)
			take_compressed_ints(&c, ints[:], 4)
			for v, i in ints {
				if v < 0 || int(v) >= table_len {
					fail(&c)
					return components, false
				}
				out[i] = table[v]
			}
		case:
			fail(&c)
		}
		return components, !c.failed
	}
	if n * components * size > cursor_end(&c) - c.pos {
		fail(&c)
		return components, false
	}
	if size == 4 {
		mem.copy(raw_data(out[:]), raw_data(c.l.bytes[c.pos:]), n * components * 4)
	} else {
		for i in 0 ..< n * components {
			out[i] = f32(take_scalar(&c, size))
		}
	}
	return components, !c.failed
}

// A token array, token vector or token list op's items, as token indices.
usd_tokens :: proc(l: ^Usd_Layer, value: u64, out: ^[dynamic]u32) -> (ok: bool) {
	clear(out)
	rep := resolve_rep(l, value) or_return
	#partial switch usd_rep_type(rep) {
	case .Token:
		if !usd_rep_is_array(rep) {
			append(out, u32(rep))
			return true
		}
		if rep & 0xffff_ffff_ffff == 0 {
			return true
		}
		c := at(l, rep)
		n := take_count(&c)
		take_u32s(&c, out, n)
		return !c.failed
	case .Token_Vector:
		c := at(l, rep)
		take_u32s(&c, out, take_count(&c))
		return !c.failed
	case .Token_List_Op:
		c := at(l, rep)
		take_list_op(&c, out)
		return !c.failed
	}
	return false
}

// A path list op's or path vector's paths (a list op's explicit, added,
// prepended and appended items, in that order).
usd_paths :: proc(l: ^Usd_Layer, value: u64, out: ^[dynamic]u32) -> (ok: bool) {
	clear(out)
	rep := resolve_rep(l, value) or_return
	#partial switch usd_rep_type(rep) {
	case .Path_List_Op:
		c := at(l, rep)
		take_list_op(&c, out)
	case .Path_Vector:
		c := at(l, rep)
		take_u32s(&c, out, take_count(&c))
	case:
		return false
	}
	for p in out {
		if int(p) >= len(l.paths) {
			return false
		}
	}
	return true
}

// A reference list op's references (explicit, added, prepended, appended).
usd_references :: proc(l: ^Usd_Layer, value: u64, out: ^[dynamic]Usd_Reference) -> (ok: bool) {
	clear(out)
	rep := resolve_rep(l, value) or_return
	if usd_rep_type(rep) != .Reference_List_Op {
		return false
	}
	c := at(l, rep)
	header := take_bytes(&c, 1)
	if c.failed {
		return false
	}
	// Explicit, added, deleted (skipped), ordered (skipped), prepended, appended.
	for bit in ([]u8{1 << 1, 1 << 2, 1 << 3, 1 << 4, 1 << 5, 1 << 6}) {
		if header[0] & bit == 0 {
			continue
		}
		n := take_count(&c)
		for _ in 0 ..< n {
			if c.failed {
				return false
			}
			s := take_u32(&c)
			p := take_u32(&c)
			take_bytes(&c, 16) // layer offset and scale
			skip_dictionary(&c)
			if bit == 1 << 3 || bit == 1 << 4 {
				continue
			}
			asset := ""
			if int(s) < len(l.strings) {
				asset, _ = token_at(l, l.strings[s])
			}
			append(out, Usd_Reference{asset = asset, prim = i32(p) if int(p) < len(l.paths) else -1})
		}
	}
	return !c.failed
}

// ---- reading ----------------------------------------------------------------------------

// A read position in the file: past `end` (0: the file's end) or on bad
// data it fails, and reads after give zeros.
@(private = "file")
Usd_Cursor :: struct {
	l:        ^Usd_Layer,
	pos, end: int,
	failed:   bool,
}

@(private = "file")
Usd_Section :: enum u8 {
	Tokens,
	Strings,
	Fields,
	Field_Sets,
	Paths,
	Specs,
}

@(private = "file", rodata)
USD_SECTION_NAMES := [Usd_Section]string {
	.Tokens     = "TOKENS",
	.Strings    = "STRINGS",
	.Fields     = "FIELDS",
	.Field_Sets = "FIELDSETS",
	.Paths      = "PATHS",
	.Specs      = "SPECS",
}

@(private = "file")
fail :: proc "contextless" (c: ^Usd_Cursor) {
	c.failed = true
}

@(private = "file")
cursor_end :: #force_inline proc "contextless" (c: ^Usd_Cursor) -> int {
	return c.end if c.end > 0 else len(c.l.bytes)
}

@(private = "file")
child_key :: #force_inline proc "contextless" (parent, token: i32, property: bool) -> u64 {
	return u64(u32(parent)) << 33 | u64(u32(token)) << 1 | u64(property)
}

@(private = "file")
token_at :: #force_inline proc "contextless" (l: ^Usd_Layer, t: u32) -> (string, bool) {
	if int(t) >= len(l.tokens) {
		return "", false
	}
	return l.tokens[t], true
}

@(private = "file")
take_bytes :: proc "contextless" (c: ^Usd_Cursor, n: int) -> []u8 {
	end := cursor_end(c)
	if c.failed || n < 0 || c.pos < 0 || n > end - c.pos {
		c.failed = true
		c.pos = end
		return nil
	}
	b := c.l.bytes[c.pos:][:n]
	c.pos += n
	return b
}

@(private = "file")
take_u32 :: proc "contextless" (c: ^Usd_Cursor) -> u32 {
	b := take_bytes(c, 4)
	return endian.unchecked_get_u32le(b) if len(b) == 4 else 0
}

@(private = "file")
take_u64 :: proc "contextless" (c: ^Usd_Cursor) -> u64 {
	b := take_bytes(c, 8)
	return endian.unchecked_get_u64le(b) if len(b) == 8 else 0
}

@(private = "file")
take_f64 :: proc "contextless" (c: ^Usd_Cursor) -> f64 {
	return transmute(f64)take_u64(c)
}

// One float of `size` bytes (2: half, 4: float, 8: double).
@(private = "file")
take_scalar :: proc "contextless" (c: ^Usd_Cursor, size: int) -> f64 {
	switch size {
	case 2:
		b := take_bytes(c, 2)
		return f64(transmute(f16)endian.unchecked_get_u16le(b)) if len(b) == 2 else 0
	case 4:
		return f64(transmute(f32)take_u32(c))
	}
	return take_f64(c)
}

// An array's element count: a u64 from version 0.7.0, a u32 before.
@(private = "file")
take_count :: proc "contextless" (c: ^Usd_Cursor) -> int {
	n := take_u64(c) if c.l.version[1] >= 7 else u64(take_u32(c))
	return fits(c, n)
}

@(private = "file")
take_u32s :: proc(c: ^Usd_Cursor, out: ^[dynamic]u32, n: int) {
	if c.failed || n < 0 || n > (cursor_end(c) - c.pos) / 4 {
		c.failed = true
		return
	}
	resize(out, n)
	for i in 0 ..< n {
		out[i] = take_u32(c)
	}
}

// A list op's items: explicit, added, prepended, appended (deleted and
// ordered items read past).
@(private = "file")
take_list_op :: proc(c: ^Usd_Cursor, out: ^[dynamic]u32) {
	header := take_bytes(c, 1)
	if c.failed {
		return
	}
	items := make([dynamic]u32, context.temp_allocator)
	for bit in ([]u8{1 << 1, 1 << 2, 1 << 3, 1 << 4, 1 << 5, 1 << 6}) {
		if header[0] & bit == 0 {
			continue
		}
		take_u32s(c, &items, int(take_u64(c)))
		if bit != 1 << 3 && bit != 1 << 4 {
			append(out, ..items[:])
		}
	}
}

// Reads past a dictionary (string keys, VtValue values).
@(private = "file")
skip_dictionary :: proc(c: ^Usd_Cursor) {
	n := take_u64(c)
	for _ in 0 ..< n {
		if c.failed {
			return
		}
		take_u32(c) // key
		// A value: a forward offset to its rep, then the rep.
		start := c.pos
		offset := i64(take_u64(c))
		if offset < 8 || offset > i64(len(c.l.bytes) - start) {
			fail(c)
			return
		}
		c.pos = start + int(offset)
		take_u64(c)
	}
}

// A file offset's cursor for a non-inlined rep.
@(private = "file")
at :: #force_inline proc "contextless" (l: ^Usd_Layer, rep: u64) -> Usd_Cursor {
	return {l = l, pos = int(rep & 0xffff_ffff_ffff)}
}

// A value rep with time samples or a VtValue wrapper resolved to the rep it
// holds (time samples: the first sample's); false for a value block.
@(private = "file")
resolve_rep :: proc(l: ^Usd_Layer, rep: u64) -> (u64, bool) {
	rep := rep
	for _ in 0 ..< 8 {
		#partial switch usd_rep_type(rep) {
		case .Value_Block, .Invalid:
			return 0, false
		case .Value:
			// A forward offset, then the rep.
			c := at(l, rep)
			start := c.pos
			offset := i64(take_u64(&c))
			if c.failed || offset < 8 || offset > i64(len(l.bytes) - start) {
				return 0, false
			}
			c.pos = start + int(offset)
			rep = take_u64(&c)
			if c.failed {
				return 0, false
			}
		case .Time_Samples:
			// Forward offset, the times' rep, forward offset, the count, the
			// values' reps.
			c := at(l, rep)
			start := c.pos
			offset := i64(take_u64(&c))
			if c.failed || offset < 8 || offset > i64(len(l.bytes) - start) {
				return 0, false
			}
			c.pos = start + int(offset)
			take_u64(&c)
			start = c.pos
			offset = i64(take_u64(&c))
			if c.failed || offset < 8 || offset > i64(len(l.bytes) - start) {
				return 0, false
			}
			c.pos = start + int(offset)
			if take_u64(&c) == 0 {
				return 0, false
			}
			rep = take_u64(&c)
			if c.failed {
				return 0, false
			}
		case:
			return rep, true
		}
	}
	return 0, false
}

// Components and bytes per component of a vector or quaternion type.
@(private = "file")
vector_shape :: proc "contextless" (t: Usd_Type) -> (components, size: int) {
	#partial switch t {
	case .Vec2h:
		return 2, 2
	case .Vec2f:
		return 2, 4
	case .Vec2d:
		return 2, 8
	case .Vec3h:
		return 3, 2
	case .Vec3f:
		return 3, 4
	case .Vec3d:
		return 3, 8
	case .Vec4h, .Quath:
		return 4, 2
	case .Vec4f, .Quatf:
		return 4, 4
	case .Vec4d, .Quatd:
		return 4, 8
	}
	return 0, 0
}

// Resizes `out` to `n` if the budget allows.
@(private = "file")
grow :: proc(c: ^Usd_Cursor, out: ^[dynamic]$T, n: int) -> bool {
	if !budget_take(&c.l.budget, .Gltf_Values, n) {
		return false
	}
	return resize(out, n) == nil
}

// ---- structural sections ----------------------------------------------------------------

@(private = "file")
read_tokens :: proc(c: ^Usd_Cursor) {
	if c.end == 0 {
		return
	}
	n := take_u64(c)
	raw_size, packed_size := take_u64(c), take_u64(c)
	packed := take_bytes(c, int(min(packed_size, u64(max(int)))))
	if c.failed || raw_size > u64(len(c.l.bytes)) * 256 || n > raw_size {
		fail(c)
		return
	}
	chars := make([]u8, int(raw_size))
	if lz4_frames(packed, chars) != len(chars) {
		fail(c)
		return
	}
	c.l.tokens = make([]string, int(n))
	start, count := 0, 0
	for b, i in chars {
		if b == 0 && count < len(c.l.tokens) {
			c.l.tokens[count] = string(chars[start:i])
			count += 1
			start = i + 1
		}
	}
	if count != len(c.l.tokens) {
		fail(c)
	}
}

@(private = "file")
read_strings :: proc(c: ^Usd_Cursor) {
	if c.end == 0 {
		return
	}
	out: [dynamic]u32
	take_u32s(c, &out, int(take_u64(c)))
	for s in out {
		if int(s) >= len(c.l.tokens) {
			fail(c)
		}
	}
	c.l.strings = out[:]
}

@(private = "file")
read_fields :: proc(c: ^Usd_Cursor) {
	if c.end == 0 {
		return
	}
	n := take_count_raw(c)
	tokens := make([]i64, n, context.temp_allocator)
	take_compressed_ints(c, tokens, 4)
	reps_size := int(take_u64(c))
	packed := take_bytes(c, reps_size)
	if c.failed {
		return
	}
	reps := make([]u64, n, context.temp_allocator)
	if lz4_frames(packed, slice.to_bytes(reps)) != n * 8 {
		fail(c)
		return
	}
	c.l.fields = make([]Usd_Field, n)
	for i in 0 ..< n {
		if tokens[i] < 0 || int(tokens[i]) >= len(c.l.tokens) {
			fail(c)
			return
		}
		c.l.fields[i] = {u32(tokens[i]), reps[i]}
	}
}

@(private = "file")
read_field_sets :: proc(c: ^Usd_Cursor) {
	if c.end == 0 {
		return
	}
	n := take_count_raw(c)
	sets := make([]i64, n, context.temp_allocator)
	take_compressed_ints(c, sets, 4)
	c.l.field_sets = make([]u32, n)
	for v, i in sets {
		// ~0 terminates each spec's run.
		if u32(v) != max(u32) && (v < 0 || int(v) >= len(c.l.fields)) {
			fail(c)
			return
		}
		c.l.field_sets[i] = u32(v)
	}
}

@(private = "file")
read_paths :: proc(c: ^Usd_Cursor) {
	if c.end == 0 {
		return
	}
	l := c.l
	num_paths := take_count_raw(c)
	n := take_count_raw(c)
	if c.failed || n > num_paths {
		fail(c)
		return
	}
	if budget_over(&l.budget, .Gltf_Nodes, num_paths) {
		fail(c)
		return
	}
	path_index := make([]i64, n, context.temp_allocator)
	element := make([]i64, n, context.temp_allocator)
	jumps := make([]i64, n, context.temp_allocator)
	take_compressed_ints(c, path_index, 4)
	take_compressed_ints(c, element, 4)
	take_compressed_ints(c, jumps, 4)
	if c.failed {
		return
	}
	for &v in element {
		v = i64(i32(v))
	}
	for &v in jumps {
		v = i64(i32(v))
	}
	l.paths = make([]Usd_Path, num_paths)
	for &p in l.paths {
		p = {parent = -1, token = -1, spec = -1}
	}
	seen := make([]bool, num_paths, context.temp_allocator)
	l.children = make(map[u64]i32, num_paths)
	// Depth first: a child follows its parent; a sibling is a jump ahead,
	// stacked while the child's subtree is read.
	Pending :: struct {
		index:  int,
		parent: i32,
	}
	stack := make([dynamic]Pending, context.temp_allocator)
	visited := make([]bool, n, context.temp_allocator)
	cur, parent := 0, i32(-1)
	l.root = i32(path_index[0]) if n > 0 && path_index[0] >= 0 && int(path_index[0]) < num_paths else -1
	for n > 0 {
		this := cur
		if this < 0 || this >= n || visited[this] {
			fail(c)
			return
		}
		visited[this] = true
		cur += 1
		pi := path_index[this]
		if pi < 0 || int(pi) >= num_paths || seen[pi] {
			fail(c)
			return
		}
		seen[pi] = true
		if parent >= 0 {
			tok := element[this]
			prop := tok < 0
			tok = abs(tok)
			if int(tok) >= len(l.tokens) {
				fail(c)
				return
			}
			l.paths[pi] = {parent = parent, token = i32(tok), property = prop, spec = -1}
			l.children[child_key(parent, i32(tok), prop)] = i32(pi)
		}
		jump := jumps[this]
		has_child := jump > 0 || jump == -1
		has_sibling := jump >= 0
		if has_child {
			if has_sibling {
				append(&stack, Pending{this + int(jump), parent})
			}
			parent = i32(pi)
		} else if !has_sibling {
			if len(stack) == 0 {
				break
			}
			next := pop(&stack)
			cur, parent = next.index, next.parent
		}
	}
}

@(private = "file")
read_specs :: proc(c: ^Usd_Cursor) {
	if c.end == 0 {
		return
	}
	l := c.l
	n := take_count_raw(c)
	paths := make([]i64, n, context.temp_allocator)
	sets := make([]i64, n, context.temp_allocator)
	types := make([]i64, n, context.temp_allocator)
	take_compressed_ints(c, paths, 4)
	take_compressed_ints(c, sets, 4)
	take_compressed_ints(c, types, 4)
	if c.failed {
		return
	}
	l.specs = make([]Usd_Spec, n)
	for i in 0 ..< n {
		p, fs, t := paths[i], sets[i], types[i]
		if p < 0 || int(p) >= len(l.paths) || fs < 0 || int(fs) >= len(l.field_sets) || t <= 0 || t > i64(max(Usd_Spec_Type)) || fs > 0 && l.field_sets[fs - 1] != max(u32) || l.paths[p].spec >= 0 {
			fail(c)
			return
		}
		end := int(fs)
		for end < len(l.field_sets) && l.field_sets[end] != max(u32) {
			end += 1
		}
		l.specs[i] = {path = u32(p), fields = {u32(fs), u32(end - int(fs))}, type = Usd_Spec_Type(t)}
		l.paths[p].spec = i32(i)
	}
}

// Whether `n` elements can come from the rest of the cursor: each takes at
// least two bits of USD's integer coding, which LZ4 packs at most 255 to 1.
@(private = "file")
fits :: proc "contextless" (c: ^Usd_Cursor, n: u64) -> int {
	if c.failed || n > u64(cursor_end(c) - c.pos) * 1024 + 64 {
		c.failed = true
		return 0
	}
	return int(n)
}

// A section's leading count, charged to the values budget.
@(private = "file")
take_count_raw :: proc "contextless" (c: ^Usd_Cursor) -> int {
	n := fits(c, take_u64(c))
	if !budget_take(&c.l.budget, .Gltf_Values, n) {
		fail(c)
		return 0
	}
	return n
}

// `len(out)` integers of `size` bytes (4 or 8): LZ4-framed, then each the
// difference to the one before, as the most common difference (a 2-bit
// code 0) or an 8/16/32-bit (16/32/64-bit for 8-byte ints) value.
@(private = "file")
take_compressed_ints :: proc(c: ^Usd_Cursor, out: []i64, size: int) {
	n := len(out)
	packed := take_bytes(c, int(min(take_u64(c), u64(max(int)))))
	if c.failed || n == 0 {
		return
	}
	codes := (n * 2 + 7) / 8
	raw := make([]u8, size + codes + n * size, context.temp_allocator)
	got := lz4_frames(packed, raw)
	if got < size + codes {
		fail(c)
		return
	}
	raw = raw[:got]
	common := i64(i32(endian.unchecked_get_u32le(raw))) if size == 4 else i64(endian.unchecked_get_u64le(raw))
	vals := size + codes
	prev: i64
	for i in 0 ..< n {
		code := (raw[size + i / 4] >> (2 * uint(i % 4))) & 3
		d: i64
		switch code {
		case 0:
			d = common
		case 1:
			if vals + size / 4 > len(raw) {
				fail(c)
				return
			}
			d = i64(i8(raw[vals])) if size == 4 else i64(i16(endian.unchecked_get_u16le(raw[vals:])))
			vals += size / 4
		case 2:
			if vals + size / 2 > len(raw) {
				fail(c)
				return
			}
			d = i64(i16(endian.unchecked_get_u16le(raw[vals:]))) if size == 4 else i64(i32(endian.unchecked_get_u32le(raw[vals:])))
			vals += size / 2
		case 3:
			if vals + size > len(raw) {
				fail(c)
				return
			}
			d = i64(i32(endian.unchecked_get_u32le(raw[vals:]))) if size == 4 else i64(endian.unchecked_get_u64le(raw[vals:]))
			vals += size
		}
		prev = i64(i32(prev + d)) if size == 4 else prev + d
		out[i] = prev
	}
}

// ---- LZ4 --------------------------------------------------------------------------------

// TfFastCompression's framing: a chunk count byte (0: one block of the
// rest), else that many blocks each after an i32 size; LZ4 blocks
// decoded into `dst`. The bytes written, or -1.
@(private = "file")
lz4_frames :: proc "contextless" (src: []u8, dst: []u8) -> int {
	if len(src) == 0 {
		return -1
	}
	chunks := int(src[0])
	if chunks == 0 {
		return lz4_block(src[1:], dst)
	}
	pos, total := 1, 0
	for _ in 0 ..< chunks {
		if pos + 4 > len(src) {
			return -1
		}
		size := int(i32(endian.unchecked_get_u32le(src[pos:])))
		pos += 4
		if size < 0 || size > len(src) - pos {
			return -1
		}
		n := lz4_block(src[pos:][:size], dst[total:])
		if n < 0 {
			return -1
		}
		pos += size
		total += n
	}
	return total
}

// One LZ4 block (literals, then a match of an offset back and a length)
// into `dst`: the bytes written, or -1 if it is malformed or does not fit.
@(private = "file")
lz4_block :: proc "contextless" (src: []u8, dst: []u8) -> int {
	s, d := 0, 0
	for s < len(src) {
		token := src[s]
		s += 1
		lit := int(token >> 4)
		if lit == 15 {
			for {
				if s >= len(src) {
					return -1
				}
				b := src[s]
				s += 1
				lit += int(b)
				if b != 255 {
					break
				}
			}
		}
		if lit > len(src) - s || lit > len(dst) - d {
			return -1
		}
		copy(dst[d:][:lit], src[s:][:lit])
		s += lit
		d += lit
		if s == len(src) {
			break // the last sequence has literals only
		}
		if s + 2 > len(src) {
			return -1
		}
		offset := int(src[s]) | int(src[s + 1]) << 8
		s += 2
		if offset == 0 || offset > d {
			return -1
		}
		length := int(token & 15)
		if length == 15 {
			for {
				if s >= len(src) {
					return -1
				}
				b := src[s]
				s += 1
				length += int(b)
				if b != 255 {
					break
				}
			}
		}
		length += 4
		if length > len(dst) - d {
			return -1
		}
		// Byte by byte: a match may overlap what it copies.
		for i in 0 ..< length {
			dst[d + i] = dst[d - offset + i]
		}
		d += length
	}
	return d
}
