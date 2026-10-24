package o3d

import "core:math"
import "core:mem"

// EXT_meshopt_compression's codecs (meshoptimizer's vertex, index and index
// sequence encodings and its filters): each decodes `count` elements of
// `stride` bytes from `src` into `dst` (count * stride bytes) and says
// whether `src` was well formed. Malformed data fails; no read goes past
// `src`.

Meshopt_Mode :: enum u8 {
	Attributes,
	Triangles,
	Indices,
}

Meshopt_Filter :: enum u8 {
	None,
	Octahedral,
	Quaternion,
	Exponential,
}

// Vertex data: per block of up to 256 elements, each byte of the element
// as a column of zigzag deltas from the element before (the first block's
// from the baseline element at the end of `src`), packed in groups of 16
// at 0, 2, 4 or 8 bits.
@(private)
meshopt_decode_vertices :: proc(dst, src: []u8, count, stride: int) -> bool {
	if stride <= 0 || stride > 256 || stride % 4 != 0 || len(dst) < count * stride {
		return false
	}
	tail := max(stride, 32)
	if len(src) < 1 + tail || src[0] != 0xa0 {
		return false
	}
	baseline := src[len(src) - stride:]
	end := len(src) - tail
	at := 1
	block := min((8192 / stride) & ~int(15), 256)
	deltas: [256]u8
	for first := 0; first < count; first += block {
		n := min(block, count - first)
		groups := (n + 15) / 16
		header := (groups + 3) / 4
		for k in 0 ..< stride {
			if end - at < header {
				return false
			}
			head := src[at:][:header]
			at += header
			for g in 0 ..< groups {
				out := deltas[g * 16:][:16]
				ok := true
				switch head[g / 4] >> uint(g % 4 * 2) & 3 {
				case 0:
					for &d in out {
						d = 0
					}
				case 1:
					at, ok = meshopt_unpack(src[:end], at, out, 2)
				case 2:
					at, ok = meshopt_unpack(src[:end], at, out, 4)
				case 3:
					if end - at < 16 {
						return false
					}
					copy(out, src[at:][:16])
					at += 16
				}
				if !ok {
					return false
				}
			}
			p := baseline[k] if first == 0 else dst[(first - 1) * stride + k]
			for i in 0 ..< n {
				d := deltas[i]
				p += (d >> 1) ~ -(d & 1)
				dst[(first + i) * stride + k] = p
			}
		}
	}
	return at == end
}

// 16 values of BITS bits, high bits first; a value of all ones is an
// escape for the next byte after the packed ones.
@(private = "file")
meshopt_unpack :: proc "contextless" (src: []u8, at: int, out: []u8, $BITS: uint) -> (int, bool) {
	SENTINEL :: u8(1 << BITS - 1)
	extra := at + 16 * int(BITS) / 8
	if extra > len(src) {
		return at, false
	}
	for i in 0 ..< 16 {
		v := src[at + i * int(BITS) / 8] >> (8 - BITS - uint(i) * BITS % 8) & SENTINEL
		if v == SENTINEL {
			if extra >= len(src) {
				return at, false
			}
			v = src[extra]
			extra += 1
		}
		out[i] = v
	}
	return extra, true
}

// Triangle indices: per triangle a code byte that names its vertices by an
// edge and a vertex FIFO of recent ones, the next new index, or a free
// index stored as a varint delta; a 16-byte table of common codes ends
// `src`.
@(private)
meshopt_decode_triangles :: proc(dst, src: []u8, count, stride: int) -> bool {
	if count % 3 != 0 || stride != 2 && stride != 4 || len(dst) < count * stride {
		return false
	}
	if len(src) < 1 + count / 3 + 16 || src[0] != 0xe0 && src[0] != 0xe1 {
		return false
	}
	edges: [16][2]u32
	verts: [16]u32
	for &e in edges {
		e = max(u32)
	}
	for &v in verts {
		v = max(u32)
	}
	edge_at, vert_at: u32
	next, last: u32
	fecmax := u8(13 if src[0] == 0xe1 else 15)
	code := 1
	data := 1 + count / 3
	// Each triangle reads at most 16 bytes past `data`: up to `safe_end`
	// it cannot run off `src`.
	safe_end := len(src) - 16
	table := src[safe_end:]
	push_vertex :: #force_inline proc "contextless" (verts: ^[16]u32, at: ^u32, v: u32) {
		verts[at^] = v
		at^ = (at^ + 1) & 15
	}
	push_edge :: #force_inline proc "contextless" (edges: ^[16][2]u32, at: ^u32, a, b: u32) {
		edges[at^] = {a, b}
		at^ = (at^ + 1) & 15
	}
	for i := 0; i < count; i += 3 {
		if data > safe_end {
			return false
		}
		codetri := src[code]
		code += 1
		a, b, c: u32
		if codetri < 0xf0 {
			e := edges[(edge_at - 1 - u32(codetri >> 4)) & 15]
			a, b = e[0], e[1]
			fec := codetri & 15
			if fec == 0 {
				c = next
				next += 1
				push_vertex(&verts, &vert_at, c)
			} else if fec < fecmax {
				c = verts[(vert_at - 1 - u32(fec)) & 15]
			} else {
				// 13 and 14: the last free index -1 and +1.
				c = last + u32(int(fec) * 2 - 27) if fec != 15 else meshopt_index(src, &data, last)
				last = c
				push_vertex(&verts, &vert_at, c)
			}
			push_edge(&edges, &edge_at, c, b)
			push_edge(&edges, &edge_at, a, c)
		} else {
			// A triangle with no cached edge: its first vertex is new (or a
			// free index), the others new, from the FIFO or free.
			// 0xfe and 0xff: the code in the next byte, not the table, and
			// 15 a free index.
			slow := codetri >= 0xfe
			aux := src[data] if slow else table[codetri & 15]
			fea: u8 = 15 if codetri == 0xff else 0
			if slow {
				data += 1
				if aux == 0 {
					next = 0
				}
			}
			feb, fec := aux >> 4, aux & 15
			a = 0
			if fea == 0 {
				a = next
				next += 1
			}
			vertex :: #force_inline proc "contextless" (verts: ^[16]u32, at: u32, fe: u8, next: ^u32) -> u32 {
				if fe == 0 {
					next^ += 1
					return next^ - 1
				}
				return verts[(at - u32(fe)) & 15]
			}
			b = vertex(&verts, vert_at, feb, &next)
			c = vertex(&verts, vert_at, fec, &next)
			if fea == 15 {
				a = meshopt_index(src, &data, last)
				last = a
			}
			if slow && feb == 15 {
				b = meshopt_index(src, &data, last)
				last = b
			}
			if slow && fec == 15 {
				c = meshopt_index(src, &data, last)
				last = c
			}
			push_vertex(&verts, &vert_at, a)
			if feb == 0 || slow && feb == 15 {
				push_vertex(&verts, &vert_at, b)
			}
			if fec == 0 || slow && fec == 15 {
				push_vertex(&verts, &vert_at, c)
			}
			push_edge(&edges, &edge_at, b, a)
			push_edge(&edges, &edge_at, c, b)
			push_edge(&edges, &edge_at, a, c)
		}
		meshopt_write(dst, i, stride, a)
		meshopt_write(dst, i + 1, stride, b)
		meshopt_write(dst, i + 2, stride, c)
	}
	return data == safe_end
}

// Index sequences (non-triangle indices): per index a varint zigzag delta
// from one of two baselines, its low bit choosing which; a 4-byte tail.
@(private)
meshopt_decode_indices :: proc(dst, src: []u8, count, stride: int) -> bool {
	if stride != 2 && stride != 4 || len(dst) < count * stride {
		return false
	}
	if len(src) < 1 + count + 4 || src[0] != 0xd0 && src[0] != 0xd1 {
		return false
	}
	// Each index reads at most 5 bytes: from before `safe_end` it cannot
	// run off `src`.
	safe_end := len(src) - 4
	data := 1
	last: [2]u32
	for i in 0 ..< count {
		if data >= safe_end {
			return false
		}
		v := meshopt_varint(src, &data)
		which := v & 1
		v >>= 1
		last[which] += (v >> 1) ~ -(v & 1)
		meshopt_write(dst, i, stride, last[which])
	}
	return data == safe_end
}

@(private = "file")
meshopt_varint :: #force_inline proc "contextless" (src: []u8, at: ^int) -> u32 {
	lead := src[at^]
	at^ += 1
	if lead < 128 {
		return u32(lead)
	}
	v := u32(lead & 127)
	for shift: u32 = 7; shift <= 28; shift += 7 {
		group := src[at^]
		at^ += 1
		v |= u32(group & 127) << shift
		if group < 128 {
			break
		}
	}
	return v
}

@(private = "file")
meshopt_index :: #force_inline proc "contextless" (src: []u8, at: ^int, last: u32) -> u32 {
	v := meshopt_varint(src, at)
	return last + ((v >> 1) ~ -(v & 1))
}

@(private = "file")
meshopt_write :: #force_inline proc "contextless" (dst: []u8, i, stride: int, v: u32) {
	if stride == 2 {
		(^u16le)(&dst[i * 2])^ = u16le(v)
	} else {
		(^u32le)(&dst[i * 4])^ = u32le(v)
	}
}

// Undoes a filter in place on `count` decoded elements of `stride` bytes:
// octahedral normals (4 or 8 bytes) and quaternions (8) back to unit
// vectors, exponential floats (any multiple of 4) back to f32. Says whether
// the stride suits the filter.
@(private)
meshopt_unfilter :: proc(data: []u8, count, stride: int, filter: Meshopt_Filter) -> bool {
	switch filter {
	case .None:
	case .Octahedral:
		if stride == 4 {
			meshopt_octahedral(mem.slice_data_cast([]i8, data[:count * 4]), count)
		} else if stride == 8 {
			meshopt_octahedral(mem.slice_data_cast([]i16le, data[:count * 8]), count)
		} else {
			return false
		}
	case .Quaternion:
		if stride != 8 {
			return false
		}
		q := mem.slice_data_cast([]i16le, data[:count * 8])
		for i in 0 ..< count {
			e := q[i * 4:][:4]
			// The largest component was dropped: its index in the low 2
			// bits of the fourth, the scale in the rest.
			ss := f32(1 / math.SQRT_TWO) / f32(i32(e[3]) | 3)
			x, y, z := f32(e[0]) * ss, f32(e[1]) * ss, f32(e[2]) * ss
			w := math.sqrt(max(1 - x * x - y * y - z * z, 0))
			qc := int(e[3]) & 3
			e[(qc + 1) & 3] = i16le(meshopt_round(x * 32767))
			e[(qc + 2) & 3] = i16le(meshopt_round(y * 32767))
			e[(qc + 3) & 3] = i16le(meshopt_round(z * 32767))
			e[qc] = i16le(meshopt_round(w * 32767))
		}
	case .Exponential:
		for &v in mem.slice_data_cast([]u32le, data[:count * stride]) {
			// A 24-bit mantissa and an 8-bit exponent, both signed.
			m := i32(u32(v) << 8) >> 8
			e := i32(u32(v)) >> 24
			v = u32le(transmute(u32)(transmute(f32)(u32(e + 127) << 23) * f32(m)))
		}
	}
	return true
}

@(private = "file")
meshopt_octahedral :: proc(n: $T/[]$E, count: int) {
	one := f32(1 << (8 * size_of(E) - 1) - 1)
	for i in 0 ..< count {
		e := n[i * 4:][:3]
		x, y := f32(e[0]), f32(e[1])
		z := f32(e[2]) - abs(x) - abs(y)
		t := min(z, 0)
		x += t if x >= 0 else -t
		y += t if y >= 0 else -t
		s := one / math.sqrt(x * x + y * y + z * z)
		e[0] = E(meshopt_round(x * s))
		e[1] = E(meshopt_round(y * s))
		e[2] = E(meshopt_round(z * s))
	}
}

// Rounded half away from zero.
@(private = "file")
meshopt_round :: #force_inline proc "contextless" (v: f32) -> int {
	return int(v + (0.5 if v >= 0 else -0.5))
}
