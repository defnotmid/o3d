package o3d

import "core:encoding/base64"
import "core:encoding/hex"
import "core:fmt"
import "core:slice"
import "core:testing"

// Streams from meshoptimizer's encoder (encodeGltfBuffer) and what they
// encode: an index sequence (u16, with a jump to 65535), a 5x5 grid of
// triangles (u32) ending in one of free indices, and 20 vertices of 4
// bytes.
@(private = "file")
MESHOPT_INDICES :: "d1000404000208c10c05f0ff0fa70c011bcaf60f00000000"
@(private = "file")
MESHOPT_TRIANGLES :: "e1fe1e100e100e100e100efe1e190e180e170e160eff1e160e160e160e160eff1e160e160e160e160eff1e160e160e160e160eff0f0c9f025f090c5f090c5f090cfffa990cb59a0cb49a0c007687566778a9866589689801690000"
@(private = "file")
MESHOPT_VERTICES :: "a0052aaaaaaaaa0000000007004a4a4a4a4a4a4a4a4a4a4a4a4a4a4aff0000004a4a4a4a0a065656565656565656560000000000000000000000000000000000000000000000000000000000000000000000070000"

@(private = "file")
meshopt_fixture :: proc(s: string) -> []u8 {
	b, _ := hex.decode(transmute([]u8)s, context.temp_allocator)
	return b
}

@(private = "file")
meshopt_grid :: proc() -> []u32 {
	tri := make([dynamic]u32, context.temp_allocator)
	for y in u32(0) ..< 5 {
		for x in u32(0) ..< 5 {
			i := y * 6 + x
			append(&tri, i, i + 1, i + 6, i + 1, i + 7, i + 6)
		}
	}
	append(&tri, 100000, 5, 99999)
	return tri[:]
}

@(private = "file")
meshopt_vertex_bytes :: proc() -> []u8 {
	v := make([]u8, 20 * 4, context.temp_allocator)
	for i in 0 ..< 20 {
		v[i * 4 + 0] = u8(i)
		v[i * 4 + 1] = 7
		v[i * 4 + 2] = u8(i * 37)
		v[i * 4 + 3] = u8(i & 1 * 3)
	}
	return v
}

// The three decoders give back what was encoded, and every truncation and
// every single corrupted byte decodes or fails without reading past the
// data (a truncated stream can still be well formed).
@(test)
test_meshopt_decode :: proc(t: ^testing.T) {
	indices := meshopt_fixture(MESHOPT_INDICES)
	want_indices := []u16le{0, 1, 2, 2, 1, 3, 400, 401, 65535, 7, 7, 0, 300}
	got_indices := make([]u16le, len(want_indices), context.temp_allocator)
	testing.expect(t, meshopt_decode_indices(slice.to_bytes(got_indices), indices, len(want_indices), 2))
	testing.expect(t, slice.equal(got_indices, want_indices), "index sequence")

	triangles := meshopt_fixture(MESHOPT_TRIANGLES)
	grid := meshopt_grid()
	got_tri := make([]u32le, len(grid), context.temp_allocator)
	testing.expect(t, meshopt_decode_triangles(slice.to_bytes(got_tri), triangles, len(grid), 4))
	// The encoder may rotate a triangle (keeping its winding).
	for i := 0; i < len(grid); i += 3 {
		a, b, c := u32(got_tri[i]), u32(got_tri[i + 1]), u32(got_tri[i + 2])
		w := grid[i:][:3]
		rotated := a == w[0] && b == w[1] && c == w[2] || a == w[1] && b == w[2] && c == w[0] || a == w[2] && b == w[0] && c == w[1]
		testing.expectf(t, rotated, "triangle %d: %d %d %d, want %v", i / 3, a, b, c, w)
	}

	vertices := meshopt_fixture(MESHOPT_VERTICES)
	want_vertices := meshopt_vertex_bytes()
	got_vertices := make([]u8, len(want_vertices), context.temp_allocator)
	testing.expect(t, meshopt_decode_vertices(got_vertices, vertices, 20, 4))
	testing.expect(t, slice.equal(got_vertices, want_vertices), "vertices")

	scratch := make([]u8, 1024, context.temp_allocator)
	for src, mode in ([3][]u8{indices, triangles, vertices}) {
		decode :: proc(mode: int, dst, src: []u8) -> bool {
			switch mode {
			case 0:
				return meshopt_decode_indices(dst, src, 13, 2)
			case 1:
				return meshopt_decode_triangles(dst, src, 153, 4)
			}
			return meshopt_decode_vertices(dst, src, 20, 4)
		}
		for n in 0 ..< len(src) {
			_ = decode(mode, scratch, src[:n])
		}
		bad := slice.clone(src, context.temp_allocator)
		for i in 0 ..< len(bad) {
			for flip in ([]u8{0x01, 0x80, 0xff}) {
				bad[i] ~= flip
				_ = decode(mode, scratch, bad)
				bad[i] ~= flip
			}
		}
	}
	testing.expect(t, !meshopt_decode_vertices(scratch, vertices, 20, 6), "a stride not a multiple of 4")
	testing.expect(t, !meshopt_decode_triangles(scratch, triangles, 77, 4), "a count not a multiple of 3")
}

// Filters: an octahedral normal back to unit length, a quaternion with its
// largest component restored, an exponential float.
@(test)
test_meshopt_unfilter :: proc(t: ^testing.T) {
	// +z: x = y = 0, z = 127.
	oct := []i8{0, 0, 127, 0}
	testing.expect(t, meshopt_unfilter(slice.to_bytes(oct), 1, 4, .Octahedral))
	testing.expect_value(t, oct[2], 127)
	// Identity with w (index 3) dropped: x, y, z = 0, the 4th the scale
	// with the dropped index in its low bits.
	quat := []i16le{0, 0, 0, i16le(32767 &~ 3 | 3)}
	testing.expect(t, meshopt_unfilter(slice.to_bytes(quat), 1, 8, .Quaternion))
	testing.expect(t, quat[3] == 32767 && quat[0] == 0 && quat[1] == 0 && quat[2] == 0, fmt.tprint(quat))
	// 3 * 2^-1.
	e := []u32le{0xff000003}
	testing.expect(t, meshopt_unfilter(slice.to_bytes(e), 1, 4, .Exponential))
	testing.expect_value(t, transmute(f32)u32(e[0]), 1.5)
	testing.expect(t, !meshopt_unfilter(slice.to_bytes(quat), 1, 4, .Quaternion), "a quaternion is 8 bytes")
}

// A file with compressed views: each view reads as its decoded bytes from
// the decoded buffer, the fallback buffer (no data) left alone; a view that
// does not decode fails the file. A texture's EXT_texture_webp source is
// taken over its `source`.
@(test)
test_gltf_meshopt :: proc(t: ^testing.T) {
	vertices := meshopt_fixture(MESHOPT_VERTICES)
	triangles := meshopt_fixture(MESHOPT_TRIANGLES)
	data := make([dynamic]u8, context.temp_allocator)
	append(&data, ..vertices)
	for len(data) % 4 != 0 {
		append(&data, 0)
	}
	tri_at := len(data)
	append(&data, ..triangles)
	encoded := base64.encode(data[:], allocator = context.temp_allocator)
	make_src :: proc(encoded: string, length, vert_len, tri_at, tri_len: int) -> []u8 {
		return transmute([]u8)fmt.tprintf(
			`{{"asset": {{"version": "2.0"}}, "extensionsRequired": ["EXT_meshopt_compression"],
			"buffers": [{{"uri": "data:application/octet-stream;base64,%s", "byteLength": %d}}, {{"byteLength": 692, "extensions": {{"EXT_meshopt_compression": {{"fallback": true}}}}}}],
			"bufferViews": [
				{{"buffer": 1, "byteLength": 80, "byteStride": 4, "extensions": {{"EXT_meshopt_compression": {{"buffer": 0, "byteLength": %d, "byteStride": 4, "count": 20, "mode": "ATTRIBUTES"}}}}}},
				{{"buffer": 1, "byteOffset": 80, "byteLength": 612, "extensions": {{"EXT_meshopt_compression": {{"buffer": 0, "byteOffset": %d, "byteLength": %d, "byteStride": 4, "count": 153, "mode": "TRIANGLES"}}}}}}],
			"images": [{{"uri": "a.png"}}, {{"uri": "a.webp"}}],
			"textures": [{{"source": 0, "extensions": {{"EXT_texture_webp": {{"source": 1}}}}}}]}}`,
			encoded, length, vert_len, tri_at, tri_len,
		)
	}
	src := make_src(string(encoded), len(data), len(vertices), tri_at, len(triangles))
	doc, err := gltf_read(src)
	testing.expect_value(t, err, Gltf_Error.None)
	if err == nil {
		bytes :: proc(doc: ^Gltf_Document, i: int) -> []u8 {
			v := doc.views[i]
			return doc.buffers[v.buffer][v.offset:][:v.length]
		}
		testing.expect(t, slice.equal(bytes(&doc, 0), meshopt_vertex_bytes()), "decoded vertices")
		want_tri := make([]u8, 153 * 4, context.temp_allocator)
		_ = meshopt_decode_triangles(want_tri, triangles, 153, 4)
		testing.expect(t, slice.equal(bytes(&doc, 1), want_tri), "decoded triangles")
		testing.expect(t, doc.buffers[1] == nil, "the fallback buffer has no data")
		testing.expect_value(t, doc.textures[0], 1)
	}
	gltf_document_destroy(&doc)

	// The triangles' view pointed at the vertices' stream.
	doc, err = gltf_read(make_src(string(encoded), len(data), len(vertices), 0, len(triangles)))
	testing.expect_value(t, err, Gltf_Error.Invalid_Compression)
	gltf_document_destroy(&doc)
}
