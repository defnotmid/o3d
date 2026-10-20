package o3d

import "core:fmt"
import "core:strings"

// Limits on the work one file can cause: a glTF may come from anyone, and a
// few hundred bytes of JSON can ask for billions of vertices, keyframes or
// pixels. Each limit is a row of one table (as ovg's limits.odin): what it
// counts, whether it caps one value (`Max`) or a running total (`Total`),
// and its default. A reader keeps a `Budget`; a `Max` limit passed fails the
// file (`Gltf_Error.Limit`), a `Total` one cuts short the work it guards
// (a primitive, channel or image is skipped), and either is recorded
// (`Budget.hit`, one warning per limit in `Model.warnings`).
Limit :: enum u8 {
	// Nesting of glTF JSON and of its node hierarchy.
	Depth,
	// Nodes of one glTF.
	Gltf_Nodes,
	// Accessors of one glTF.
	Gltf_Accessors,
	// Bytes of one glTF's buffers (the GLB binary chunk, data URIs, files).
	Gltf_Buffer_Bytes,
	// Values (elements times components) unpacked from one glTF's
	// accessors, morph target deltas and node weights included.
	Gltf_Values,
	// Pixels of one glTF's decoded images.
	Image_Pixels,
}

Limits :: [Limit]int

Limit_Kind :: enum u8 {
	// The limit caps one value (a depth, a table's length).
	Max,
	// The limit caps a running total over a read.
	Total,
}

Limit_Info :: struct {
	name: string,
	kind: Limit_Kind,
}

LIMIT_INFO :: [Limit]Limit_Info {
	.Depth             = {"nesting depth", .Max},
	.Gltf_Nodes        = {"glTF nodes", .Max},
	.Gltf_Accessors    = {"glTF accessors", .Max},
	.Gltf_Buffer_Bytes = {"glTF buffer bytes", .Total},
	.Gltf_Values       = {"glTF accessor values", .Total},
	.Image_Pixels      = {"image pixels", .Total},
}

// Far above what real files use; each bounds memory and time to what a
// desktop handles.
DEFAULT_LIMITS :: Limits {
	.Depth             = 1024,
	.Gltf_Nodes        = 1 << 20,
	.Gltf_Accessors    = 1 << 20,
	.Gltf_Buffer_Bytes = 1 << 31,
	.Gltf_Values       = 1 << 28,
	.Image_Pixels      = 1 << 28,
}

Budget :: struct {
	limits: Limits,
	used:   Limits,
	// The limits reached.
	hit:    bit_set[Limit],
}

// Starts a budget: `limits`, or DEFAULT_LIMITS when nil.
budget_init :: proc "contextless" (b: ^Budget, limits: ^Limits = nil) {
	b^ = {limits = limits^ if limits != nil else DEFAULT_LIMITS}
}

// Counts `n` more of a `Total` limit; false (and recorded) when that would
// pass the limit, and nothing is counted.
budget_take :: #force_inline proc "contextless" (b: ^Budget, l: Limit, n := 1) -> bool {
	if n < 0 || n > b.limits[l] - b.used[l] {
		b.hit += {l}
		return false
	}
	b.used[l] += n
	return true
}

// `budget_take` for a count that may not fit in an int.
budget_take_wide :: proc "contextless" (b: ^Budget, l: Limit, n: i128) -> bool {
	if n < 0 || n > i128(b.limits[l] - b.used[l]) {
		b.hit += {l}
		return false
	}
	b.used[l] += int(n)
	return true
}

// Whether `v` passes a `Max` limit (recorded).
budget_over :: #force_inline proc "contextless" (b: ^Budget, l: Limit, v: int) -> bool {
	if v > b.limits[l] {
		b.hit += {l}
		return true
	}
	return false
}

// Appends one warning per limit hit (cloned strings).
@(private)
budget_warn :: proc(b: ^Budget, warnings: ^[dynamic]string) {
	info := LIMIT_INFO
	for l in b.hit {
		cut := "the file is not read" if info[l].kind == .Max else "the rest is skipped"
		append(warnings, strings.clone(fmt.tprintf("limit reached: %s (%d); %s", info[l].name, b.limits[l], cut)))
	}
}
