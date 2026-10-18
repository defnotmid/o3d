// o3d: 3D for Odin, in three.js's vocabulary. A frame is described from
// scratch every time (`Scene`: a camera, lights and draws of geometry with a
// material), so it is a pure function of whatever drives it; only the
// geometry is kept (`Geometries`, made once and reused by id). The
// software renderer (software.odin) is the pixel reference and the CPU
// path; `o3d/sdlgpu` draws the same scene on the GPU through SDL_GPU.
//
// Conventions are three.js's, so what is known about it carries over:
// right-handed and Y up, the camera looking down -Z, rotations as Euler
// angles in radians applied in X, Y, Z order, the field of view in degrees,
// colours in sRGB, MeshStandardMaterial's metalness / roughness model with
// its light units (a white directional light of intensity 1 gives a white
// surface facing it 1/pi of white: scenes use intensities around 2 to 3),
// no tone mapping, output in sRGB. The output is premultiplied RGBA,
// transparent where nothing was drawn, to composite over anything.
package o3d

import "core:math"
import "core:math/linalg"
import "core:sync"

Vec2 :: [2]f32
Vec3 :: [3]f32
Vec4 :: [4]f32
Mat4 :: matrix[4, 4]f32

Error :: enum u8 {
	None,
	Out_Of_Memory,
	// No GPU device, or it can't do what is asked.
	Device,
	Shader,
	// A GPU render recorded without its prepare (o3d_sdlgpu.record), or
	// after another prepare started the tables' mirrors over.
	Not_Prepared,
}

// ---- transforms -------------------------------------------------------------------------

// Translation, Euler XYZ rotation (radians) and scale, composed as three.js's
// Object3D does: T * R * S, with R = Rx * Ry * Rz.
compose :: proc "contextless" (position: Vec3 = {}, rotation: Vec3 = {}, scale: Vec3 = {1, 1, 1}) -> Mat4 {
	cx, sx := math.cos(rotation.x), math.sin(rotation.x)
	cy, sy := math.cos(rotation.y), math.sin(rotation.y)
	cz, sz := math.cos(rotation.z), math.sin(rotation.z)
	// three.js Matrix4.makeRotationFromEuler, order XYZ.
	r := Mat4 {
		cy * cz, -cy * sz, sy, 0,
		cx * sz + sx * sy * cz, cx * cz - sx * sy * sz, -sx * cy, 0,
		sx * sz - cx * sy * cz, sx * cz + cx * sy * sz, cx * cy, 0,
		0, 0, 0, 1,
	}
	m := r * Mat4{scale.x, 0, 0, 0, 0, scale.y, 0, 0, 0, 0, scale.z, 0, 0, 0, 0, 1}
	m[0, 3], m[1, 3], m[2, 3] = position.x, position.y, position.z
	return m
}

// A view matrix: the camera at `eye` looking at `target` (right-handed).
look_at :: proc "contextless" (eye, target: Vec3, up: Vec3 = {0, 1, 0}) -> Mat4 {
	z := eye - target
	if linalg.length(z) < 1e-9 {
		z = {0, 0, 1}
	}
	z = linalg.normalize(z)
	x := linalg.cross(up, z)
	if linalg.length(x) < 1e-9 {
		// Looking straight along `up`: any side will do.
		x = linalg.cross(Vec3{0, 0, 1} if abs(z.y) > 0.9 else Vec3{0, 1, 0}, z)
	}
	x = linalg.normalize(x)
	y := linalg.cross(z, x)
	return Mat4 {
		x.x, x.y, x.z, -linalg.dot(x, eye),
		y.x, y.y, y.z, -linalg.dot(y, eye),
		z.x, z.y, z.z, -linalg.dot(z, eye),
		0, 0, 0, 1,
	}
}

// Perspective projection to clip space with depth 0 (near) to 1 (far) and Y
// up (SDL_GPU's, D3D's and Metal's convention). `fov` is vertical, degrees.
perspective :: proc "contextless" (fov, aspect, near, far: f32) -> Mat4 {
	f := 1 / math.tan(fov * math.RAD_PER_DEG * 0.5)
	return Mat4 {
		f / aspect, 0, 0, 0,
		0, f, 0, 0,
		0, 0, far / (near - far), near * far / (near - far),
		0, 0, -1, 0,
	}
}

// Orthographic projection of a view `height` units tall.
orthographic :: proc "contextless" (height, aspect, near, far: f32) -> Mat4 {
	h := height * 0.5
	w := h * aspect
	return Mat4 {
		1 / w, 0, 0, 0,
		0, 1 / h, 0, 0,
		0, 0, 1 / (near - far), near / (near - far),
		0, 0, 0, 1,
	}
}

// The matrix for normals: the inverse transpose of the model's upper 3x3.
normal_matrix :: proc "contextless" (model: Mat4) -> Mat4 {
	m3 := matrix[3, 3]f32{
		model[0, 0], model[0, 1], model[0, 2],
		model[1, 0], model[1, 1], model[1, 2],
		model[2, 0], model[2, 1], model[2, 2],
	}
	n := linalg.transpose(linalg.inverse(m3))
	return Mat4 {
		n[0, 0], n[0, 1], n[0, 2], 0,
		n[1, 0], n[1, 1], n[1, 2], 0,
		n[2, 0], n[2, 1], n[2, 2], 0,
		0, 0, 0, 1,
	}
}

// ---- colour -----------------------------------------------------------------------------

srgb_to_linear :: proc "contextless" (c: f32) -> f32 {
	return c / 12.92 if c <= 0.04045 else math.pow((c + 0.055) / 1.055, 2.4)
}

linear_to_srgb :: proc "contextless" (value: f32) -> f32 {
	c := clamp(value, 0, 1)
	return c * 12.92 if c <= 0.0031308 else 1.055 * math.pow(c, 1 / 2.4) - 0.055
}

linear :: proc "contextless" (c: Vec3) -> Vec3 {
	return {srgb_to_linear(c.r), srgb_to_linear(c.g), srgb_to_linear(c.b)}
}

linear_to_srgb_vec3 :: #force_inline proc "contextless" (c: Vec3) -> Vec3 {
	return {linear_to_srgb(c.r), linear_to_srgb(c.g), linear_to_srgb(c.b)}
}

srgb8 :: #force_inline proc "contextless" (linear_val: f32) -> u8 {
	return u8(linear_to_srgb(linear_val) * 255 + 0.5)
}

// three.js's tone mapping modes (`TONE_*` in shaders/common.slang).
Tone_Mapping :: enum u8 {
	None        = 0, // NoToneMapping
	Linear      = 1, // LinearToneMapping
	Reinhard    = 2, // ReinhardToneMapping
	Cineon      = 3, // CineonToneMapping
	ACES_Filmic = 4, // ACESFilmicToneMapping (Stephen Hill fit)
	AgX         = 5, // AgXToneMapping
	Neutral     = 6, // NeutralToneMapping (Khronos PBR Neutral)
}

// Maps linear HDR colour `c` to linear [0, 1] with `mode` and `exposure`
// (0 means 1), matching `tone_map` in shaders/common.slang and three.js's
// tonemapping_pars_fragment.glsl.js.
tone_map :: proc "contextless" (c: Vec3, mode: Tone_Mapping, exposure: f32 = 1) -> Vec3 {
	if mode == .None {
		return c
	}
	exp := exposure if exposure > 0 else 1
	v := c * exp
	switch mode {
	case .None:
		return c
	case .Linear:
		return linalg.clamp(v, 0, 1)
	case .Reinhard:
		return linalg.clamp(v / (Vec3{1, 1, 1} + v), 0, 1)
	case .Cineon:
		x := linalg.max(v - Vec3{0.004, 0.004, 0.004}, Vec3{0, 0, 0})
		u := (x * (6.2 * x + Vec3{0.5, 0.5, 0.5})) / (x * (6.2 * x + Vec3{1.7, 1.7, 1.7}) + Vec3{0.06, 0.06, 0.06})
		return {math.pow(u.r, 2.2), math.pow(u.g, 2.2), math.pow(u.b, 2.2)}
	case .ACES_Filmic:
		v *= 1.0 / 0.6
		v = {
			0.59719 * v.r + 0.35458 * v.g + 0.04823 * v.b,
			0.07600 * v.r + 0.90834 * v.g + 0.01566 * v.b,
			0.02840 * v.r + 0.13383 * v.g + 0.83777 * v.b,
		}
		a := v * (v + Vec3{0.0245786, 0.0245786, 0.0245786}) - Vec3{0.000090537, 0.000090537, 0.000090537}
		b := v * (0.983729 * v + Vec3{0.4329510, 0.4329510, 0.4329510}) + Vec3{0.238081, 0.238081, 0.238081}
		v = a / b
		v = {
			 1.60475 * v.r - 0.53108 * v.g - 0.07367 * v.b,
			-0.10208 * v.r + 1.10813 * v.g - 0.00605 * v.b,
			-0.00327 * v.r - 0.07276 * v.g + 1.07602 * v.b,
		}
		return linalg.clamp(v, 0, 1)
	case .AgX:
		// Linear Rec.709 -> Linear Rec.2020 -> AgXInsetMatrix
		r2020 := Vec3{
			0.6274039 * v.r + 0.3292830 * v.g + 0.0433131 * v.b,
			0.0690973 * v.r + 0.9195404 * v.g + 0.0113623 * v.b,
			0.0163914 * v.r + 0.0880133 * v.g + 0.8955953 * v.b,
		}
		r2020 = linalg.max(r2020, Vec3{1e-10, 1e-10, 1e-10})
		inset := Vec3{
			0.85662715 * r2020.r + 0.09512124 * r2020.g + 0.04825161 * r2020.b,
			0.13731897 * r2020.r + 0.76124199 * r2020.g + 0.10143904 * r2020.b,
			0.11189821 * r2020.r + 0.07679942 * r2020.g + 0.81130237 * r2020.b,
		}
		inset = linalg.max(inset, Vec3{1e-10, 1e-10, 1e-10})
		ev := Vec3{math.log2(inset.r), math.log2(inset.g), math.log2(inset.b)}
		x := linalg.clamp((ev - Vec3{-12.47393, -12.47393, -12.47393}) * (1.0 / 16.5), 0, 1)
		x2 := x * x
		x4 := x2 * x2
		x = 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - Vec3{0.00232, 0.00232, 0.00232}
		outset := Vec3{
			 1.12710058 * x.r - 0.11060664 * x.g - 0.01649394 * x.b,
			-0.14132976 * x.r + 1.15782370 * x.g - 0.01649394 * x.b,
			-0.14132976 * x.r - 0.11060664 * x.g + 1.25193641 * x.b,
		}
		outset = linalg.max(outset, Vec3{0, 0, 0})
		lin2020 := Vec3{math.pow(outset.r, 2.2), math.pow(outset.g, 2.2), math.pow(outset.b, 2.2)}
		out709 := Vec3{
			 1.6604910 * lin2020.r - 0.5876411 * lin2020.g - 0.0728499 * lin2020.b,
			-0.1245505 * lin2020.r + 1.1328999 * lin2020.g - 0.0083494 * lin2020.b,
			-0.0181508 * lin2020.r - 0.1005789 * lin2020.g + 1.1187297 * lin2020.b,
		}
		return linalg.clamp(out709, 0, 1)
	case .Neutral:
		START_COMPRESSION :: f32(0.8 - 0.04)
		DESATURATION :: f32(0.15)
		x := min(v.r, v.g, v.b)
		offset: f32 = x - 6.25 * x * x if x < 0.08 else 0.04
		p := v - Vec3{offset, offset, offset}
		peak := max(p.r, p.g, p.b)
		if peak < START_COMPRESSION {
			return linalg.clamp(p, 0, 1)
		}
		d :: f32(1.0 - START_COMPRESSION)
		new_peak := 1.0 - d * d / (peak + d - START_COMPRESSION)
		p *= new_peak / peak
		g := 1.0 - 1.0 / (DESATURATION * (peak - new_peak) + 1.0)
		return linalg.clamp(linalg.lerp(p, Vec3{new_peak, new_peak, new_peak}, g), 0, 1)
	}
	return c
}

// ---- tables -----------------------------------------------------------------------------

// Grows a dynamic array with geometric reserve so arena allocators do not
// strand O(N^2) backing memory across incremental resizes.
table_grow :: #force_inline proc(table: ^[dynamic]$T, length: int) {
	if length > cap(table^) {
		reserve(table, max(length, 2 * cap(table^), 64))
	}
	resize(table, length)
}

table_grow_non_zero :: #force_inline proc(table: ^[dynamic]$T, length: int) {
	if length > cap(table^) {
		reserve(table, max(length, 2 * cap(table^), 64))
	}
	non_zero_resize(table, length)
}


// ---- generations ------------------------------------------------------------------------

// Which contents a retained table (Geometries, Textures, Shaders) holds, for
// what mirrors it (the GPU's buffers, textures and pipelines): taken when
// the table gets its first entry, and new for a table made after one was
// destroyed, though it reuses ids and even its address. 0: empty.
Generation :: distinct u64

@(private = "file")
generations: u64 // atomic: the last one handed out

// Gives a table a generation if it has none.
@(private)
generation_take :: proc "contextless" (g: ^Generation) {
	if g^ == 0 {
		g^ = Generation(sync.atomic_add(&generations, 1) + 1)
	}
}
