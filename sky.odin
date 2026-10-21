package o3d

import "core:math"
import "core:math/linalg"

// A physical sky, three.js's Sky (examples/jsm/objects/Sky.js: Preetham,
// Shirley and Smits's daylight model with its sun disc): the linear
// radiance along a direction as that shader writes it before tone mapping.
// `sky_texture` bakes it into an HDR equirectangular texture for
// `scene.background` and `scene.environment` alike (three.js draws it as a
// mesh and bakes it with PMREMGenerator for reflections); the scene's tone
// mapping and exposure (`tone_mapping_exposure`) then apply as to any.
Sky :: struct {
	sun:               Vec3, // towards the sun (any length)
	up:                Vec3, // the zenith
	turbidity:         f32, // haze
	rayleigh:          f32, // molecules' scattering, blue
	mie_coefficient:   f32, // aerosols' scattering
	mie_directional_g: f32, // how forward the aerosols scatter, 0..1
}

// three.js's defaults, the sun at the horizon.
SKY_DEFAULT :: Sky {
	sun               = {0, 0, -1},
	up                = {0, 1, 0},
	turbidity         = 2,
	rayleigh          = 1,
	mie_coefficient   = 0.005,
	mie_directional_g = 0.8,
}

// What the shader's vertex stage works out once: the same for every
// direction.
@(private = "file")
Sky_Terms :: struct {
	sun, up:        Vec3,
	sun_e:          f32,
	beta_r, beta_m: Vec3,
	g:              f32,
	horizon:        f32, // how far Lin moves to its horizon's form
	exponent:       f32,
}

@(private = "file")
sky_terms :: proc "contextless" (sky: Sky) -> (k: Sky_Terms) {
	// Preetham's primaries (680, 550, 450 nm): Rayleigh's total and Mie's
	// constant, pre-computed as three.js's.
	TOTAL_RAYLEIGH :: Vec3{5.804542996261093e-6, 1.3562911419845635e-5, 3.0265902468824876e-5}
	MIE_CONST :: Vec3{1.8399918514433978e14, 2.7798023919660528e14, 4.0790479543861094e14}
	CUTOFF_ANGLE :: 1.6110731556870734 // pi / 1.95: the earth's shadow
	STEEPNESS :: 1.5
	EE :: 1000.0
	k.sun = linalg.normalize0(sky.sun)
	k.up = linalg.normalize0(sky.up)
	zenith_cos := clamp(linalg.dot(k.sun, k.up), -1, 1)
	k.sun_e = EE * max(0, 1 - math.exp(-((CUTOFF_ANGLE - math.acos(zenith_cos)) / STEEPNESS)))
	// three.js's fade by the sun's height over 450000 (its position's y; a
	// direction here).
	sun_fade := 1 - clamp(1 - math.exp(k.sun.y / 450000), 0, 1)
	k.beta_r = TOTAL_RAYLEIGH * (sky.rayleigh - (1 - sun_fade))
	k.beta_m = 0.434 * (0.2 * sky.turbidity * 10e-18) * MIE_CONST * sky.mie_coefficient
	k.g = sky.mie_directional_g
	k.horizon = clamp(math.pow(1 - linalg.dot(k.up, k.sun), 5), 0, 1)
	k.exponent = 1 / (1.2 + 1.2 * sun_fade)
	return
}

@(private = "file")
sky_shade :: #force_inline proc "contextless" (k: ^Sky_Terms, dir: Vec3) -> Vec3 {
	RAYLEIGH_ZENITH_LENGTH :: 8.4e3
	MIE_ZENITH_LENGTH :: 1.25e3
	SUN_ANGULAR_DIAMETER_COS :: 0.999956676946448443553574619906976478926848692873900859324
	THREE_OVER_SIXTEEN_PI :: 0.05968310365946075
	ONE_OVER_FOUR_PI :: 0.07957747154594767
	// The optical length, cut off at the horizon (the formula's singularity).
	up_cos := max(0, linalg.dot(k.up, dir))
	zenith := math.acos(up_cos)
	inv := 1 / (up_cos + 0.15 * math.pow(93.885 - zenith * (180 / math.PI), -1.253))
	fex := linalg.exp(-(k.beta_r * (RAYLEIGH_ZENITH_LENGTH * inv) + k.beta_m * (MIE_ZENITH_LENGTH * inv)))
	// In-scattering.
	cos_theta := linalg.dot(dir, k.sun)
	r := cos_theta * 0.5 + 0.5
	r_phase := THREE_OVER_SIXTEEN_PI * (1 + r * r)
	g2 := k.g * k.g
	m_phase := ONE_OVER_FOUR_PI * (1 - g2) / math.pow(1 - 2 * k.g * cos_theta + g2, 1.5)
	scatter := k.sun_e * (k.beta_r * r_phase + k.beta_m * m_phase) / (k.beta_r + k.beta_m)
	lin := pow3(scatter * (1 - fex), 1.5)
	lin *= linalg.lerp(Vec3{1, 1, 1}, pow3(scatter * fex, 0.5), k.horizon)
	// The night sky's floor and the sun's disc.
	disc := smoothstep(SUN_ANGULAR_DIAMETER_COS, SUN_ANGULAR_DIAMETER_COS + 0.00002, cos_theta)
	l0 := fex * (0.1 + k.sun_e * 19000 * disc)
	return pow3((lin + l0) * 0.04 + Vec3{0, 0.0003, 0.00075}, k.exponent)
}

@(private = "file")
pow3 :: #force_inline proc "contextless" (v: Vec3, e: f32) -> Vec3 {
	return {math.pow(v.x, e), math.pow(v.y, e), math.pow(v.z, e)}
}

@(private = "file")
smoothstep :: #force_inline proc "contextless" (lo, hi, x: f32) -> f32 {
	t := clamp((x - lo) / (hi - lo), 0, 1)
	return t * t * (3 - 2 * t)
}

// The sky's linear radiance towards `dir` (a unit vector).
sky_radiance :: proc "contextless" (sky: Sky, dir: Vec3) -> Vec3 {
	k := sky_terms(sky)
	return sky_shade(&k, dir)
}

// Bakes the sky into a new HDR equirectangular texture, `width` by half
// that, with its mipmaps: the background (sharp at about a texel a screen
// pixel or two: 2048 for a 1080p frame at 50 degrees) and, with
// `environment_prefilter`, the environment (any width; it is prefiltered
// from 128 across).
sky_texture :: proc(t: ^Textures, sky: Sky, width: int) -> Texture_Id {
	if width < 2 {
		return 0
	}
	w, h := width, width / 2
	tex := texture_grow(&t.hdr_texels, w, h)
	tex.hdr = true
	k := sky_terms(sky)
	texels := t.hdr_texels[tex.first_texel:][:w * h]
	for y in 0 ..< h {
		v := 1 - (f32(y) + 0.5) / f32(h)
		for x in 0 ..< w {
			c := sky_shade(&k, equirect_dir({(f32(x) + 0.5) / f32(w), v}))
			texels[y * w + x] = {f16(min(c.r, 65504)), f16(min(c.g, 65504)), f16(min(c.b, 65504)), 1}
		}
	}
	return texture_finish(t, &tex)
}
