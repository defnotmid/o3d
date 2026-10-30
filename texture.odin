package o3d

import "core:math"
import "core:math/linalg"
import "core:mem"
import "core:os"
import image "deps:oicut/image"

// Textures: RGBA8 images (sRGB-encoded colour) or HDR ones (linear,
// half floats) with their mipmap chains, made once and referred to by id.
// Every texture's texels across all mip levels live in one shared flat
// array (`Textures.texels`, or `hdr_texels`); a texture is its dimensions,
// level count and per-level offsets in it. The GPU mirrors the list, uploading new
// textures with their mip levels at the start of a frame. An external
// texture (`texture_add_external`) has no texels: its pixels are a GPU
// texture the host binds per frame (a video's frames).

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
	external:                 bool, // levels 0: bound per frame on the GPU
	// Texels in `Textures.hdr_texels`: linear, not sRGB-encoded (an
	// environment or background past 1).
	hdr:                      bool,
}

Textures :: struct {
	generation: Generation, // of these contents, for the GPU's mirror
	texels:     [dynamic][4]u8,
	hdr_texels: [dynamic][4]f16,
	list:       [dynamic]Texture,
}

textures_destroy :: proc(t: ^Textures) {
	delete(t.texels)
	delete(t.hdr_texels)
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
	tex := texture_grow(&t.texels, width, height)
	mem.copy(raw_data(t.texels[tex.first_texel:]), raw_data(pixels), width * height * 4)
	return texture_finish(t, &tex)
}

// Adds an HDR image: linear RGBA floats (`width x height`, rows from the
// top), kept as half floats (past 65504 kept at 65504) with their mipmaps.
texture_add_hdr :: proc(t: ^Textures, pixels: []f32, width, height: int) -> Texture_Id {
	if width <= 0 || height <= 0 || len(pixels) < width * height * 4 {
		return 0
	}
	tex := texture_grow(&t.hdr_texels, width, height)
	tex.hdr = true
	halves(t.hdr_texels[tex.first_texel:][:width * height], pixels)
	return texture_finish(t, &tex)
}

@(private = "file")
halves :: proc "contextless" (dst: [][4]f16, src: []f32) #no_bounds_check {
	for &px, i in dst {
		for c in 0 ..< 4 {
			px[c] = to_half(clamp(src[i * 4 + c], -65504, 65504))
		}
	}
}

// Half floats to floats and back by their bits, exactly (to the nearest,
// ties to even; F. Giesen's): Odin's casts call a library routine on CPUs
// without F16C, which made reading half floats the software renderer's
// largest cost.
@(private)
from_half :: #force_inline proc "contextless" (h: f16) -> f32 {
	MAGIC :: f32(1.0 / 16384) // 2^-14: a subnormal's scale, renormalized
	SHIFTED_EXP :: u32(0x7c00) << 13
	u := u32(transmute(u16)h)
	o := (u & 0x7fff) << 13
	exp := o & SHIFTED_EXP
	o += (127 - 15) << 23
	if exp == SHIFTED_EXP {
		o += (128 - 16) << 23 // inf, NaN
	} else if exp == 0 {
		o += 1 << 23 // zero, subnormal
		o = transmute(u32)(transmute(f32)o - MAGIC)
	}
	return transmute(f32)(o | (u & 0x8000) << 16)
}

@(private)
to_half :: #force_inline proc "contextless" (f: f32) -> f16 {
	DENORM_MAGIC :: f32(0.5) // ((127 - 15) + (23 - 10) + 1) << 23
	u := transmute(u32)f
	sign := u & 0x80000000
	u ~= sign
	o: u32
	if u >= (127 + 16) << 23 {
		o = 0x7e00 if u > 255 << 23 else 0x7c00 // NaN, inf
	} else if u < 113 << 23 {
		o = transmute(u32)(transmute(f32)u + DENORM_MAGIC) - transmute(u32)DENORM_MAGIC // subnormal, zero
	} else {
		mant_odd := (u >> 13) & 1
		u = u - ((127 - 15) << 23) + 0xfff
		o = (u + mant_odd) >> 13
	}
	return transmute(f16)u16(o | sign >> 16)
}

@(private)
from_halves :: #force_inline proc "contextless" (h: [4]f16) -> [4]f32 {
	return {from_half(h[0]), from_half(h[1]), from_half(h[2]), from_half(h[3])}
}

@(private)
to_halves :: #force_inline proc "contextless" (c: [4]f32) -> [4]f16 {
	return {to_half(c[0]), to_half(c[1]), to_half(c[2]), to_half(c[3])}
}

// A texture's levels laid out at the end of `table` (texels or
// hdr_texels), grown once to hold them all; level 0 is for the caller.
@(private)
texture_grow :: proc(table: ^[dynamic]$T, width, height: int) -> Texture {
	tex := Texture {
		width       = u32(width),
		height      = u32(height),
		first_texel = u32(len(table)),
	}
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
	table_grow_non_zero(table, int(tex.first_texel + total))
	return tex
}

// Downsamples level 0 of a grown texture into its mip chain and lists it.
@(private)
texture_finish :: proc(t: ^Textures, tex: ^Texture, blit := false) -> Texture_Id {
	sw, sh := int(tex.width), int(tex.height)
	for l in 1 ..< int(tex.levels) {
		dw, dh := max(sw >> 1, 1), max(sh >> 1, 1)
		if tex.hdr && blit {
			downsample_mip_hdr_blit(t.hdr_texels[tex.mips[l - 1]:][:sw * sh], sw, sh, t.hdr_texels[tex.mips[l]:][:dw * dh], dw, dh)
		} else if tex.hdr {
			downsample_mip_hdr(t.hdr_texels[tex.mips[l - 1]:][:sw * sh], sw, sh, t.hdr_texels[tex.mips[l]:][:dw * dh], dw, dh)
		} else {
			downsample_mip(t.texels[tex.mips[l - 1]:][:sw * sh], sw, sh, t.texels[tex.mips[l]:][:dw * dh], dw, dh)
		}
		sw, sh = dw, dh
	}
	generation_take(&t.generation)
	append(&t.list, tex^)
	return Texture_Id(len(t.list))
}

// Adds a texture whose pixels live on the GPU, outside `t`, changing every
// frame (a video): the GPU renderer samples what the host binds to the id
// (`o3d_sdlgpu.texture_set_external`) without mipmaps unless that texture
// has them; the software renderer, with no texels, samples white (the
// material's own colour). `width x height` is what it expects, for the host.
texture_add_external :: proc(t: ^Textures, width, height: int) -> Texture_Id {
	generation_take(&t.generation)
	append(&t.list, Texture{width = u32(max(width, 0)), height = u32(max(height, 0)), first_texel = u32(len(t.texels)), external = true})
	return Texture_Id(len(t.list))
}

// Decodes an image in memory (PNG, JPEG, GIF, BMP, TGA, QOI or WebP, by
// oicut's image package; an animation's first frame; Radiance HDR as an HDR
// texture) into `t`.
texture_decode :: proc(t: ^Textures, data: []u8) -> (Texture_Id, bool) {
	info, err := image.info(data)
	if err != .None || info.width <= 0 || info.height <= 0 {
		return 0, false
	}
	if info.format == .HDR {
		n := info.width * info.height
		floats, aerr := make([]f32, n * 4)
		if aerr != nil {
			return 0, false
		}
		defer delete(floats)
		tex := texture_grow(&t.hdr_texels, info.width, info.height)
		tex.hdr = true
		if image.decode_float_into(data, floats) != .None {
			resize(&t.hdr_texels, int(tex.first_texel))
			return 0, false
		}
		halves(t.hdr_texels[tex.first_texel:][:n], floats)
		return texture_finish(t, &tex), true
	}
	// Straight into level 0 of the texel table.
	tex := texture_grow(&t.texels, info.width, info.height)
	if image.decode_into(data, mem.slice_to_bytes(t.texels[tex.first_texel:][:info.width * info.height])) != .None {
		resize(&t.texels, int(tex.first_texel))
		return 0, false
	}
	return texture_finish(t, &tex), true
}

// The pixels an image in memory would decode to, without decoding it (0:
// not an image texture_decode reads).
@(private)
texture_pixels :: proc(data: []u8) -> int {
	info, err := image.info(data)
	if err != .None {
		return 0
	}
	return info.width * info.height
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

// 2x2 box downsample of an HDR level, in f32.
@(private = "file")
downsample_mip_hdr :: proc "contextless" (src: [][4]f16, sw, sh: int, dst: [][4]f16, dw, dh: int) #no_bounds_check {
	for y in 0 ..< dh {
		row0 := min(y * 2, sh - 1) * sw
		row1 := min(y * 2 + 1, sh - 1) * sw
		for x in 0 ..< dw {
			x0, x1 := min(x * 2, sw - 1), min(x * 2 + 1, sw - 1)
			sum := from_halves(src[row0 + x0]) + from_halves(src[row0 + x1]) + from_halves(src[row1 + x0]) + from_halves(src[row1 + x1])
			dst[y * dw + x] = to_halves(sum * 0.25)
		}
	}
}

// An HDR level downsampled as a GPU's linear blit does (SDL's
// GenerateMipmaps): each texel bilinear at its centre scaled to the level
// above, 8-bit subtexel weights; the 2x2 box for even sizes, odd ones
// spread over the whole level rather than dropping its last row.
@(private = "file")
downsample_mip_hdr_blit :: proc "contextless" (src: [][4]f16, sw, sh: int, dst: [][4]f16, dw, dh: int) #no_bounds_check {
	tap :: #force_inline proc "contextless" (d, sn, dn: int) -> (i0, i1: int, w: f32) {
		p := (f32(d) + 0.5) * f32(sn) / f32(dn) - 0.5
		f := math.floor(p)
		w = math.round((p - f) * BILINEAR_SCALE) / BILINEAR_SCALE
		i := int(f)
		return clamp(i, 0, sn - 1), clamp(i + 1, 0, sn - 1), w
	}
	for y in 0 ..< dh {
		y0, y1, wy := tap(y, sh, dh)
		for x in 0 ..< dw {
			x0, x1, wx := tap(x, sw, dw)
			top := linalg.lerp(from_halves(src[y0 * sw + x0]), from_halves(src[y0 * sw + x1]), wx)
			bottom := linalg.lerp(from_halves(src[y1 * sw + x0]), from_halves(src[y1 * sw + x1]), wx)
			dst[y * dw + x] = to_halves(linalg.lerp(top, bottom, wy))
		}
	}
}

// One level of a texture, bilinearly: an 8-bit texture's raw (sRGB-encoded)
// values, an HDR one's linear ones.
@(private)
sample_level :: #force_inline proc "contextless" (t: ^Textures, tex: ^Texture, l: int, uv: Vec2, clamp_v := false) -> Vec4 {
	w, h := int(max(tex.width >> u32(l), 1)), int(max(tex.height >> u32(l), 1))
	if tex.hdr {
		return sample_mip_hdr(t.hdr_texels[tex.mips[l]:], w, h, uv, clamp_v)
	}
	return sample_mip(t.texels[tex.mips[l]:], w, h, uv, clamp_v)
}

// Trilinearly samples `id` at `uv` (repeating; v = 1 is row 0, the top of the
// image, matching three.js's flipY and the primitive uvs) with mip level
// chosen from screen derivatives `duv_dx` and `duv_dy` as the GPU does.
sample :: proc "contextless" (t: ^Textures, id: Texture_Id, uv, duv_dx, duv_dy: Vec2) -> Vec4 {
	tex, ok := texture_get(t, id)
	if !ok || tex.external {
		return {1, 1, 1, 1}
	}
	if tex.levels <= 1 {
		return sample_level(t, &tex, 0, uv)
	}
	size := Vec2{f32(tex.width), f32(tex.height)}
	rho2 := max(linalg.length2(duv_dx * size), linalg.length2(duv_dy * size))
	if rho2 <= 1 {
		return sample_level(t, &tex, 0, uv)
	}
	lod := clamp(0.5 * math.log2(rho2), 0, f32(tex.levels - 1))
	l0 := int(lod)
	l1 := min(l0 + 1, int(tex.levels) - 1)
	frac := lod - f32(l0)
	c0 := sample_level(t, &tex, l0, uv)
	if l0 == l1 || frac <= 0 {
		return c0
	}
	return linalg.lerp(c0, sample_level(t, &tex, l1, uv), frac)
}

// Bilinearly samples `id`'s top level at `uv`, repeating (as the GPU's
// shadow pass reads alpha masks).
sample_top :: proc "contextless" (t: ^Textures, id: Texture_Id, uv: Vec2) -> Vec4 {
	tex, ok := texture_get(t, id)
	if !ok || tex.external {
		return {1, 1, 1, 1}
	}
	return sample_level(t, &tex, 0, uv)
}

// Trilinearly samples `id` at explicit `lod`, repeating horizontally and,
// with `clamp_v`, clamping vertically at the poles (equirectangular
// environment and background maps); else repeating both ways.
sample_lod :: proc "contextless" (t: ^Textures, id: Texture_Id, uv: Vec2, lod: f32, clamp_v := true) -> Vec4 {
	tex, ok := texture_get(t, id)
	if !ok || tex.external {
		return {0, 0, 0, 0}
	}
	// Not above 0 (magnified, or -inf or NaN where uv does not change).
	if tex.levels <= 1 || !(lod > 0) {
		return sample_level(t, &tex, 0, uv, clamp_v)
	}
	lod_clamped := clamp(lod, 0, f32(tex.levels - 1))
	l0 := int(lod_clamped)
	l1 := min(l0 + 1, int(tex.levels) - 1)
	frac := lod_clamped - f32(l0)
	c0 := sample_level(t, &tex, l0, uv, clamp_v)
	if l0 == l1 || frac <= 0 {
		return c0
	}
	return linalg.lerp(c0, sample_level(t, &tex, l1, uv, clamp_v), frac)
}

// `sample_lod`'s colour, linear: an HDR texture's as it is, an 8-bit one's
// decoded from sRGB (an environment's or background's either way).
sample_lod_linear :: proc "contextless" (t: ^Textures, id: Texture_Id, uv: Vec2, lod: f32) -> Vec3 {
	tex, ok := texture_get(t, id)
	c := sample_lod(t, id, uv, lod)
	return c.rgb if ok && tex.hdr else linear(c.rgb)
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

// `sample_mip` for an HDR level: the same 8-bit subtexel weights (as GPU
// filtering), linear values.
@(private)
sample_mip_hdr :: proc "contextless" (mip: [][4]f16, w, h: int, uv: Vec2, clamp_v := false) -> Vec4 #no_bounds_check {
	tc := Vec2{linalg.fract(uv.x), clamp(1 - uv.y, 0, 1) if clamp_v else linalg.fract(1 - uv.y)}
	st := tc * Vec2{f32(w), f32(h)} - 0.5
	f0 := linalg.floor(st)
	p0 := linalg.array_cast(f0, int)
	wt := linalg.array_cast(linalg.clamp(linalg.array_cast(linalg.round((st - f0) * f32(BILINEAR_SCALE)), i32), 0, BILINEAR_SCALE), f32) * (1.0 / BILINEAR_SCALE)
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
	top := linalg.lerp(from_halves(mip[row0 + ix0]), from_halves(mip[row0 + ix1]), wt.x)
	bottom := linalg.lerp(from_halves(mip[row1 + ix0]), from_halves(mip[row1 + ix1]), wt.x)
	return linalg.lerp(top, bottom, wt.y)
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

