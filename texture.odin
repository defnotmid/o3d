package o3d

import "core:math"
import "core:math/linalg"
import "core:mem"
import "core:os"
import stbi "vendor:stb/image"

// Textures: RGBA8 images with their mipmap chains, made once and referred to
// by id. Every texture's texels across all mip levels live in one shared flat
// array (`Textures.texels`); a texture is its dimensions, level count and
// per-level offsets in `texels`. The GPU mirrors the list, uploading new
// textures with their mip levels at the start of a frame.

// 0 is no texture.
Texture_Id :: distinct u32

MAX_MIPS :: 16

Texture :: struct {
	width, height:            u32,
	levels:                   u32,
	first_texel, texel_count: u32,
	mips:                     [MAX_MIPS]u32,
	env_radiance:             Texture_Id,
	env_irradiance:           Texture_Id,
}

Textures :: struct {
	generation: Generation, // of these contents, for the GPU's mirror
	texels: [dynamic][4]u8,
	list:   [dynamic]Texture,
}

textures_destroy :: proc(t: ^Textures) {
	delete(t.texels)
	delete(t.list)
	t^ = {}
}

texture_get :: proc "contextless" (t: ^Textures, id: Texture_Id) -> (Texture, bool) {
	if t == nil || id == 0 || int(id) > len(t.list) {
		return {}, false
	}
	return t.list[id - 1], true
}

// Adds an RGBA8 image (`width x height`, rows from the top) and builds its
// mipmap chain down to 1x1 in `t.texels`.
texture_add :: proc(t: ^Textures, pixels: []u8, width, height: int) -> Texture_Id {
	if width <= 0 || height <= 0 || len(pixels) < width * height * 4 {
		return 0
	}
	tex := Texture {
		width       = u32(width),
		height      = u32(height),
		first_texel = u32(len(t.texels)),
	}
	// Pass 1: count total texels across all mip levels.
	mw, mh := u32(width), u32(height)
	total: u32
	for l in 0 ..< u32(MAX_MIPS) {
		tex.mips[l] = tex.first_texel + total
		total += mw * mh
		tex.levels = l + 1
		if mw == 1 && mh == 1 {
			break
		}
		mw, mh = max(mw >> 1, 1), max(mh >> 1, 1)
	}
	tex.texel_count = total

	// Pass 2: single contiguous growth, copy level 0 and downsample mip chain.
	table_grow_non_zero(&t.texels, int(tex.first_texel + total))
	mem.copy(raw_data(t.texels[tex.first_texel:]), raw_data(pixels), width * height * 4)

	sw, sh := int(tex.width), int(tex.height)
	for l in 1 ..< int(tex.levels) {
		dw, dh := max(sw >> 1, 1), max(sh >> 1, 1)
		src := t.texels[tex.mips[l - 1]:][:sw * sh]
		dst := t.texels[tex.mips[l]:][:dw * dh]
		downsample_mip(src, sw, sh, dst, dw, dh)
		sw, sh = dw, dh
	}
	generation_take(&t.generation)
	append(&t.list, tex)
	return Texture_Id(len(t.list))
}

// Decodes an image in memory (PNG, JPEG, BMP, TGA, GIF via stb_image) into `t`.
texture_decode :: proc(t: ^Textures, data: []u8) -> (Texture_Id, bool) {
	if len(data) == 0 {
		return 0, false
	}
	w, h, ch: i32
	px := stbi.load_from_memory(raw_data(data), i32(len(data)), &w, &h, &ch, 4)
	if px == nil {
		return 0, false
	}
	defer stbi.image_free(px)
	id := texture_add(t, px[:w * h * 4], int(w), int(h))
	return id, id != 0
}

// Loads an image file from disk into `t`.
texture_load :: proc(t: ^Textures, path: string) -> (Texture_Id, bool) {
	data, err := os.read_entire_file(path, context.allocator)
	if err != nil {
		return 0, false
	}
	defer delete(data)
	return texture_decode(t, data)
}

// 2x2 box downsample for mipmap generation, clamping at odd edges.
@(private = "file")
downsample_mip :: proc "contextless" (src: [][4]u8, sw, sh: int, dst: [][4]u8, dw, dh: int) #no_bounds_check {
	for y in 0 ..< dh {
		row0 := min(y * 2, sh - 1) * sw
		row1 := min(y * 2 + 1, sh - 1) * sw
		dst_row := y * dw
		for x in 0 ..< dw {
			x0, x1 := min(x * 2, sw - 1), min(x * 2 + 1, sw - 1)
			a := linalg.array_cast(src[row0 + x0], u32)
			b := linalg.array_cast(src[row0 + x1], u32)
			c := linalg.array_cast(src[row1 + x0], u32)
			d := linalg.array_cast(src[row1 + x1], u32)
			dst[dst_row + x] = linalg.array_cast((a + b + c + d + 2) / 4, u8)
		}
	}
}

// Trilinearly samples `id` at `uv` (repeating; v = 1 is row 0, the top of the
// image, matching three.js's flipY and the primitive uvs) with mip level
// chosen from screen derivatives `duv_dx` and `duv_dy` as the GPU does.
sample :: proc "contextless" (t: ^Textures, id: Texture_Id, uv, duv_dx, duv_dy: Vec2) -> Vec4 {
	tex, ok := texture_get(t, id)
	if !ok {
		return {1, 1, 1, 1}
	}
	if tex.levels <= 1 {
		return sample_mip(t.texels[tex.mips[0]:], int(tex.width), int(tex.height), uv)
	}
	size := Vec2{f32(tex.width), f32(tex.height)}
	rho2 := max(linalg.length2(duv_dx * size), linalg.length2(duv_dy * size))
	if rho2 <= 1 {
		return sample_mip(t.texels[tex.mips[0]:], int(tex.width), int(tex.height), uv)
	}
	lod := clamp(0.5 * math.log2(rho2), 0, f32(tex.levels - 1))
	l0 := int(lod)
	l1 := min(l0 + 1, int(tex.levels) - 1)
	frac := lod - f32(l0)
	c0 := sample_mip(t.texels[tex.mips[l0]:], int(max(tex.width >> u32(l0), 1)), int(max(tex.height >> u32(l0), 1)), uv)
	if l0 == l1 || frac <= 0 {
		return c0
	}
	c1 := sample_mip(t.texels[tex.mips[l1]:], int(max(tex.width >> u32(l1), 1)), int(max(tex.height >> u32(l1), 1)), uv)
	return linalg.lerp(c0, c1, frac)
}

// Trilinearly samples `id` at explicit `lod` with horizontal repeat and
// vertical pole clamp (for equirectangular environment and background maps).
sample_lod :: proc "contextless" (t: ^Textures, id: Texture_Id, uv: Vec2, lod: f32) -> Vec4 {
	tex, ok := texture_get(t, id)
	if !ok {
		return {0, 0, 0, 0}
	}
	if tex.levels <= 1 || lod <= 0 {
		return sample_mip(t.texels[tex.mips[0]:], int(tex.width), int(tex.height), uv, true)
	}
	lod_clamped := clamp(lod, 0, f32(tex.levels - 1))
	l0 := int(lod_clamped)
	l1 := min(l0 + 1, int(tex.levels) - 1)
	frac := lod_clamped - f32(l0)
	c0 := sample_mip(t.texels[tex.mips[l0]:], int(max(tex.width >> u32(l0), 1)), int(max(tex.height >> u32(l0), 1)), uv, true)
	if l0 == l1 || frac <= 0 {
		return c0
	}
	c1 := sample_mip(t.texels[tex.mips[l1]:], int(max(tex.width >> u32(l1), 1)), int(max(tex.height >> u32(l1), 1)), uv, true)
	return linalg.lerp(c0, c1, frac)
}

BILINEAR_SCALE :: 256
TEXEL_NORM :: 1.0 / (f32(BILINEAR_SCALE * BILINEAR_SCALE) * 255.0)

// Bilinearly samples one mip level with 8-bit fixed-point subtexel weights
// (matching GPU hardware filtering). When `clamp_v` is true, u wraps and v
// clamps to edge (equirectangular poles).
@(private)
sample_mip :: proc "contextless" (mip: [][4]u8, w, h: int, uv: Vec2, clamp_v := false) -> Vec4 #no_bounds_check {
	tc := Vec2{linalg.fract(uv.x), clamp(1 - uv.y, 0, 1) if clamp_v else linalg.fract(1 - uv.y)}
	st := tc * Vec2{f32(w), f32(h)} - 0.5
	f0 := linalg.floor(st)
	p0 := linalg.array_cast(f0, int)
	wt := linalg.clamp(linalg.array_cast(linalg.round((st - f0) * f32(BILINEAR_SCALE)), i32), 0, BILINEAR_SCALE)
	inv_wt := BILINEAR_SCALE - wt
	w00 := inv_wt.x * inv_wt.y
	w10 := wt.x * inv_wt.y
	w01 := inv_wt.x * wt.y
	w11 := wt.x * wt.y

	ix0 := p0.x + w if p0.x < 0 else (p0.x - w if p0.x >= w else p0.x)
	ix1 := 0 if ix0 + 1 >= w else ix0 + 1
	iy0, iy1: int
	if clamp_v {
		iy0 = clamp(p0.y, 0, h - 1)
		iy1 = clamp(p0.y + 1, 0, h - 1)
	} else {
		iy0 = p0.y + h if p0.y < 0 else (p0.y - h if p0.y >= h else p0.y)
		iy1 = 0 if iy0 + 1 >= h else iy0 + 1
	}

	row0, row1 := iy0 * w, iy1 * w
	c00 := linalg.array_cast(mip[row0 + ix0], i32)
	c10 := linalg.array_cast(mip[row0 + ix1], i32)
	c01 := linalg.array_cast(mip[row1 + ix0], i32)
	c11 := linalg.array_cast(mip[row1 + ix1], i32)
	return linalg.array_cast(w00 * c00 + w10 * c10 + w01 * c01 + w11 * c11, f32) * TEXEL_NORM
}

// Perturbs unit surface normal `n` by tangent-space `map_n` (-1..1) from screen
// derivatives of world position (`dp_dx`, `dp_dy`) and uv (`duv_dx`, `duv_dy`),
// as three.js's perturbNormal2Arb does (with screen y running down).
perturb_normal :: proc "contextless" (n, dp_dx, dp_dy: Vec3, duv_dx, duv_dy: Vec2, map_n: Vec3) -> Vec3 {
	q1perp := linalg.cross(n, dp_dy)
	q0perp := linalg.cross(dp_dx, n)
	t := q1perp * duv_dx.x + q0perp * duv_dy.x
	b := q1perp * duv_dx.y + q0perp * duv_dy.y
	det := max(linalg.length2(t), linalg.length2(b))
	scale := linalg.inverse_sqrt(det) if det > 0 else 0
	return linalg.normalize(t * (map_n.x * scale) + b * (map_n.y * scale) + n * map_n.z)
}

