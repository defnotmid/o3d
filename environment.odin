package o3d

import "core:math"
import "core:math/linalg"
import "core:mem"

// Equirectangular environment lighting and background: `scene.environment`'s
// texture has derived maps, as a texture has mipmaps: a GGX radiance mip
// pyramid and a cosine-convolved diffuse irradiance texture in `Textures`,
// made by `environment_prefilter` (heavy: a host calls it at load, with the
// texture), sampled identically by the software renderer (`shade`) and the
// SDL_GPU renderer (`shaders/o3d.slang`). A render that meets a texture
// without them derives them then, in frame time.
// `scene.background` is sampled along each pixel's unprojected camera ray,
// from its sharp mip (blurriness = 0) or the prefiltered radiance pyramid.

ENV_MAX_WIDTH :: 128
ENV_MIPS :: 6
ENV_RAD_SAMPLES :: 32
ENV_IRR_SAMPLES :: 64

// The scene's equirectangular environment in shading form: prefiltered
// radiance mip chain and diffuse irradiance texture in `textures`.
Shading_Environment :: struct {
	textures:   ^Textures,
	radiance:   Texture_Id,
	irradiance: Texture_Id,
	intensity:  f32,
	max_lod:    f32,
}

shading_environment :: proc(scene: ^Scene, textures: ^Textures) -> Shading_Environment {
	rad, irr, ok := texture_environment(textures, scene.environment)
	if !ok {
		rad, irr, ok = environment_prefilter(textures, scene.environment)
	}
	if !ok {
		return {}
	}
	rad_tex, _ := texture_get(textures, rad)
	return {
		textures   = textures,
		radiance   = rad,
		irradiance = irr,
		intensity  = scene.environment_intensity if scene.environment_intensity > 0 else 1,
		max_lod    = f32(max(int(rad_tex.levels) - 1, 0)),
	}
}

shading_background :: proc(scene: ^Scene, textures: ^Textures) -> (tex: Texture_Id, intensity, lod: f32) {
	if _, ok := texture_get(textures, scene.background); !ok {
		return 0, 0, 0
	}
	intensity = scene.background_intensity if scene.background_intensity > 0 else 1
	if scene.background_blurriness > 0 {
		rad, _, ok := texture_environment(textures, scene.background)
		if !ok {
			rad, _, ok = environment_prefilter(textures, scene.background)
		}
		if ok {
			rad_tex, _ := texture_get(textures, rad)
			return rad, intensity, clamp(scene.background_blurriness, 0, 1) * f32(max(int(rad_tex.levels) - 1, 0))
		}
	}
	return scene.background, intensity, 0
}

// Maps a unit direction `d` to equirectangular uvs (three.js's equirectUv:
// v = 1 at +Y zenith, v = 0 at -Y nadir).
equirect_uv :: #force_inline proc "contextless" (d: Vec3) -> Vec2 {
	u := math.atan2(d.z, d.x) * (0.5 / math.PI) + 0.5
	v := math.asin(clamp(d.y, -1, 1)) * (1.0 / math.PI) + 0.5
	return {u, v}
}

// Maps equirectangular uvs back to a unit world direction.
equirect_dir :: #force_inline proc "contextless" (uv: Vec2) -> Vec3 {
	phi := (uv.x - 0.5) * (2 * math.PI)
	theta := (uv.y - 0.5) * math.PI
	ct := math.cos(theta)
	return {ct * math.cos(phi), math.sin(theta), ct * math.sin(phi)}
}

@(private = "file")
orthonormal_basis :: #force_inline proc "contextless" (n: Vec3) -> (t, b: Vec3) {
	up := Vec3{0, 1, 0} if abs(n.y) < 0.999 else Vec3{1, 0, 0}
	t = linalg.normalize(linalg.cross(up, n))
	b = linalg.cross(n, t)
	return
}

// An equirectangular texture's derived environment maps, if
// `environment_prefilter` has made them.
texture_environment :: proc "contextless" (t: ^Textures, id: Texture_Id) -> (radiance, irradiance: Texture_Id, ok: bool) {
	tex := texture_get(t, id) or_return
	if tex.env_radiance == 0 || tex.env_irradiance == 0 {
		return 0, 0, false
	}
	return tex.env_radiance, tex.env_irradiance, true
}

// Derives an equirectangular texture's environment maps, once (as loading
// a texture derives its mipmaps): a GGX-prefiltered radiance mip pyramid
// (`radiance`, mip 0..levels-1 for roughness 0..1) and a cosine-convolved
// diffuse irradiance texture (`irradiance`), both stored in `t` so the CPU
// and GPU sample them identically. Heavy: call it at load for every texture
// used as `scene.environment` or a blurred `scene.background`.
environment_prefilter :: proc(t: ^Textures, id: Texture_Id) -> (radiance, irradiance: Texture_Id, ok: bool) {
	src_tex, found := texture_get(t, id)
	if !found {
		return 0, 0, false
	}
	if src_tex.env_radiance != 0 && src_tex.env_irradiance != 0 {
		return src_tex.env_radiance, src_tex.env_irradiance, true
	}

	// Start from the mip of `id` whose width is at most ENV_MAX_WIDTH.
	base_mip: u32
	rad_w, rad_h := src_tex.width, src_tex.height
	for base_mip + 1 < src_tex.levels && rad_w > ENV_MAX_WIDTH {
		base_mip += 1
		rad_w, rad_h = max(rad_w >> 1, 1), max(rad_h >> 1, 1)
	}

	rad_tex := Texture {
		width       = rad_w,
		height      = rad_h,
		first_texel = u32(len(t.texels)),
	}
	mw, mh := rad_w, rad_h
	rad_total: u32
	for l in 0 ..< u32(ENV_MIPS) {
		rad_tex.mips[l] = rad_tex.first_texel + rad_total
		rad_total += mw * mh
		rad_tex.levels = l + 1
		if mw == 1 && mh == 1 {
			break
		}
		mw, mh = max(mw >> 1, 1), max(mh >> 1, 1)
	}
	rad_tex.texel_count = rad_total

	irr_w, irr_h := min(rad_w, 32), min(rad_h, 16)
	irr_total := irr_w * irr_h
	irr_tex := Texture {
		width       = irr_w,
		height      = irr_h,
		levels      = 1,
		first_texel = rad_tex.first_texel + rad_total,
		texel_count = irr_total,
	}
	irr_tex.mips[0] = irr_tex.first_texel

	table_grow_non_zero(&t.texels, int(irr_tex.first_texel + irr_total))

	// Level 0 of radiance (roughness = 0) is the sharp base equirectangular mip.
	mem.copy(
		raw_data(t.texels[rad_tex.mips[0]:]),
		raw_data(t.texels[src_tex.mips[base_mip]:]),
		int(rad_w * rad_h) * size_of([4]u8),
	)

	// Levels 1..levels-1: GGX importance-sampled prefiltering by roughness.
	GOLDEN_RATIO :: 0.61803398875
	pw, ph := int(rad_w), int(rad_h)
	for l in 1 ..< int(rad_tex.levels) {
		dw, dh := max(pw >> 1, 1), max(ph >> 1, 1)
		roughness := f32(l) / f32(max(int(rad_tex.levels) - 1, 1))
		alpha := roughness * roughness
		a2 := alpha * alpha

		local_dirs: [ENV_RAD_SAMPLES]Vec4
		valid_samples := 0
		for k in 0 ..< ENV_RAD_SAMPLES {
			u1 := (f32(k) + 0.5) / f32(ENV_RAD_SAMPLES)
			phi := (2 * math.PI) * linalg.fract(f32(k) * GOLDEN_RATIO)
			cos_theta := math.sqrt((1 - u1) / (1 + (a2 - 1) * u1))
			sin_theta := math.sqrt(max(1 - cos_theta * cos_theta, 0))
			hx := sin_theta * math.cos(phi)
			hy := sin_theta * math.sin(phi)
			hz := cos_theta
			ld := Vec3{2 * hz * hx, 2 * hz * hy, 2 * hz * hz - 1}
			if ld.z > 0 {
				local_dirs[valid_samples] = {ld.x, ld.y, ld.z, ld.z}
				valid_samples += 1
			}
		}

		prev_mip := t.texels[rad_tex.mips[l - 1]:][:pw * ph]
		dst_mip := t.texels[rad_tex.mips[l]:][:dw * dh]
		for y in 0 ..< dh {
			v := 1 - (f32(y) + 0.5) / f32(dh)
			for x in 0 ..< dw {
				u := (f32(x) + 0.5) / f32(dw)
				r := equirect_dir({u, v})
				tang, bitang := orthonormal_basis(r)
				acc: Vec3
				w_sum: f32
				for s_idx in 0 ..< valid_samples {
					ld := local_dirs[s_idx]
					dir := tang * ld.x + bitang * ld.y + r * ld.z
					s := sample_mip(prev_mip, pw, ph, equirect_uv(dir), true)
					acc += linear(s.rgb) * ld.w
					w_sum += ld.w
				}
				c := acc / max(w_sum, 1e-6)
				dst_mip[y * dw + x] = {srgb8(c.r), srgb8(c.g), srgb8(c.b), 255}
			}
		}
		pw, ph = dw, dh
	}

	// Diffuse irradiance: cosine-weighted hemisphere convolution.
	irr_dirs: [ENV_IRR_SAMPLES]Vec3
	for k in 0 ..< ENV_IRR_SAMPLES {
		u1 := (f32(k) + 0.5) / f32(ENV_IRR_SAMPLES)
		phi := (2 * math.PI) * linalg.fract(f32(k) * GOLDEN_RATIO)
		r := math.sqrt(u1)
		irr_dirs[k] = {r * math.cos(phi), r * math.sin(phi), math.sqrt(1 - u1)}
	}
	irr_src_level := min(2, int(rad_tex.levels) - 1)
	irr_sw := int(max(rad_tex.width >> u32(irr_src_level), 1))
	irr_sh := int(max(rad_tex.height >> u32(irr_src_level), 1))
	irr_src := t.texels[rad_tex.mips[irr_src_level]:][:irr_sw * irr_sh]
	irr_dst := t.texels[irr_tex.mips[0]:][:irr_total]
	for y in 0 ..< int(irr_h) {
		v := 1 - (f32(y) + 0.5) / f32(irr_h)
		for x in 0 ..< int(irr_w) {
			u := (f32(x) + 0.5) / f32(irr_w)
			n := equirect_dir({u, v})
			tang, bitang := orthonormal_basis(n)
			acc: Vec3
			for ld in irr_dirs {
				dir := tang * ld.x + bitang * ld.y + n * ld.z
				s := sample_mip(irr_src, irr_sw, irr_sh, equirect_uv(dir), true)
				acc += linear(s.rgb)
			}
			c := acc * (1.0 / f32(ENV_IRR_SAMPLES))
			irr_dst[y * int(irr_w) + x] = {srgb8(c.r), srgb8(c.g), srgb8(c.b), 255}
		}
	}

	append(&t.list, rad_tex)
	radiance = Texture_Id(len(t.list))
	append(&t.list, irr_tex)
	irradiance = Texture_Id(len(t.list))
	t.list[id - 1].env_radiance = radiance
	t.list[id - 1].env_irradiance = irradiance
	return radiance, irradiance, true
}
