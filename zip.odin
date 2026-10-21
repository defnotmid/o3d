package o3d

import "core:bytes"
import "core:compress/zlib"
import "core:strings"

// A flat ZIP reader for the formats that are archives (3MF, USDZ): the
// central directory as a table of entries, and an entry's bytes (stored
// ones borrowed from the archive, deflated ones inflated into the temp
// allocator).

@(private)
Zip_Error :: enum u8 {
	None,
	Data_Too_Short,
	Not_Zip,
	Corrupt_Zip,
}


@(private)
Zip_Entry :: struct {
	name:        string,
	method:      u16,
	comp_size:   u32,
	uncomp_size: u32,
	data_offset: u32,
}

@(private)
u16_le :: #force_inline proc "contextless" (b: []u8, off: int) -> u16 {
	return u16(b[off]) | (u16(b[off + 1]) << 8)
}

@(private)
u32_le :: #force_inline proc "contextless" (b: []u8, off: int) -> u32 {
	return u32(b[off]) | (u32(b[off + 1]) << 8) | (u32(b[off + 2]) << 16) | (u32(b[off + 3]) << 24)
}

@(private)
trim_archive_path :: #force_inline proc "contextless" (p: string) -> string {
	s := p
	for len(s) > 0 && (s[0] == '/' || s[0] == '\\') {
		s = s[1:]
	}
	return s
}

@(private)
zip_read_directory :: proc(archive: []u8) -> (entries: []Zip_Entry, err: Zip_Error) {
	if len(archive) < 4 {
		return nil, .Data_Too_Short
	}
	if u32_le(archive, 0) != 0x04034b50 && u32_le(archive, 0) != 0x06054b50 {
		return nil, .Not_Zip
	}
	if len(archive) < 22 {
		return nil, .Data_Too_Short
	}
	// Scan backwards for the End of Central Directory record (0x06054b50).
	eocd := -1
	min_off := max(0, len(archive) - 22 - 65535)
	for i := len(archive) - 22; i >= min_off; i -= 1 {
		if u32_le(archive, i) == 0x06054b50 {
			eocd = i
			break
		}
	}
	if eocd < 0 {
		return nil, .Not_Zip
	}
	count := int(u16_le(archive, eocd + 10))
	cd_offset := int(u32_le(archive, eocd + 16))
	if cd_offset < 0 || cd_offset > len(archive) {
		return nil, .Corrupt_Zip
	}
	list := make([dynamic]Zip_Entry, 0, count, context.temp_allocator)
	pos := cd_offset
	for _ in 0 ..< count {
		if pos + 46 > len(archive) || u32_le(archive, pos) != 0x02014b50 {
			return nil, .Corrupt_Zip
		}
		method := u16_le(archive, pos + 10)
		comp_size := u32_le(archive, pos + 20)
		uncomp_size := u32_le(archive, pos + 24)
		name_len := int(u16_le(archive, pos + 28))
		extra_len := int(u16_le(archive, pos + 30))
		comment_len := int(u16_le(archive, pos + 32))
		lh_off := int(u32_le(archive, pos + 42))
		if pos + 46 + name_len + extra_len + comment_len > len(archive) {
			return nil, .Corrupt_Zip
		}
		name := trim_archive_path(string(archive[pos + 46 : pos + 46 + name_len]))
		pos += 46 + name_len + extra_len + comment_len

		if lh_off < 0 || lh_off + 30 > len(archive) || u32_le(archive, lh_off) != 0x04034b50 {
			return nil, .Corrupt_Zip
		}
		lh_name_len := int(u16_le(archive, lh_off + 26))
		lh_extra_len := int(u16_le(archive, lh_off + 28))
		data_off := lh_off + 30 + lh_name_len + lh_extra_len
		if data_off < 0 || data_off + int(comp_size) > len(archive) {
			return nil, .Corrupt_Zip
		}
		append(&list, Zip_Entry {
			name        = name,
			method      = method,
			comp_size   = comp_size,
			uncomp_size = uncomp_size,
			data_offset = u32(data_off),
		})
	}
	return list[:], .None
}

// The entry at `path` (false: none).
@(private)
zip_find :: proc(entries: []Zip_Entry, path: string) -> (Zip_Entry, bool) {
	clean := trim_archive_path(path)
	for e in entries {
		if clean != "" && strings.equal_fold(e.name, clean) {
			return e, true
		}
	}
	return {}, false
}

@(private)
zip_extract :: proc(archive: []u8, entries: []Zip_Entry, path: string) -> (data: []u8, ok: bool) {
	clean := trim_archive_path(path)
	if clean == "" {
		return nil, false
	}
	for e in entries {
		if !strings.equal_fold(e.name, clean) {
			continue
		}
		comp := archive[e.data_offset : e.data_offset + e.comp_size]
		switch e.method {
		case 0: // Stored
			if e.comp_size != e.uncomp_size {
				return nil, false
			}
			return comp, true
		case 8: // Deflated
			if e.uncomp_size == 0 {
				return nil, true
			}
			buf: bytes.Buffer
			buf.buf = make([dynamic]u8, 0, int(e.uncomp_size), context.temp_allocator)
			if zlib.inflate_from_byte_array_raw(comp, &buf, expected_output_size = int(e.uncomp_size)) != nil {
				return nil, false
			}
			return buf.buf[:], true
		case:
			return nil, false
		}
	}
	return nil, false
}
