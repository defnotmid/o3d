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
Mat3 :: matrix[3, 3]f32
// A Mat4 laid out by rows: `Row_Mat4(m)[i]` is m's row i (`m[i]`, its column).
Row_Mat4 :: #row_major matrix[4, 4]f32

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
	m[3] = {position.x, position.y, position.z, 1}
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

// Perspective projection to clip space with reversed depth, 1 at near to 0
// at far (three.js's reversedDepthBuffer: floats are densest near 0, so
// far surfaces keep their precision), and Y up (SDL_GPU's, D3D's and
// Metal's convention). `fov` is vertical, degrees. Both renderers, the
// camera's and the lights' shadows alike: nearer is greater.
perspective :: proc "contextless" (fov, aspect, near, far: f32) -> Mat4 {
	f := 1 / math.tan(fov * math.RAD_PER_DEG * 0.5)
	return Mat4 {
		f / aspect, 0, 0, 0,
		0, f, 0, 0,
		0, 0, near / (far - near), near * far / (far - near),
		0, 0, -1, 0,
	}
}

// Orthographic projection of a view `height` units tall, depth reversed as
// `perspective`'s.
orthographic :: proc "contextless" (height, aspect, near, far: f32) -> Mat4 {
	h := height * 0.5
	w := h * aspect
	return Mat4 {
		1 / w, 0, 0, 0,
		0, 1 / h, 0, 0,
		0, 0, 1 / (far - near), far / (far - near),
		0, 0, 0, 1,
	}
}

// The matrix for normals: the inverse transpose of the model's upper 3x3.
normal_matrix :: proc "contextless" (model: Mat4) -> Mat4 {
	return Mat4(linalg.transpose(linalg.inverse(Mat3(model))))
}

// ---- colour -----------------------------------------------------------------------------

// The sRGB transfer both ways, per pixel and texel: tables at 1/4096 steps
// over 0..1 (nearest; within half an 8-bit step), the formula past them.
SRGB_STEPS :: 4096

@(private = "file")
srgb_tables: struct {
	to_linear, to_srgb: [SRGB_STEPS + 1]f32,
}

@(init, private = "file")
make_srgb_tables :: proc "contextless" () {
	for i in 0 ..= SRGB_STEPS {
		c := f32(i) / SRGB_STEPS
		srgb_tables.to_linear[i] = srgb_to_linear_exact(c)
		srgb_tables.to_srgb[i] = linear_to_srgb_exact(c)
	}
}

// log2 of a mantissa in [1, 2) by table (linearly interpolated, within
// about 1e-8), for `log2_fast`.
@(private = "file")
log2_mantissa: [SRGB_STEPS + 1]f32

@(init, private = "file")
make_log2_table :: proc "contextless" () {
	for i in 0 ..= SRGB_STEPS {
		log2_mantissa[i] = math.log2(1 + f32(i) / SRGB_STEPS)
	}
}

@(private = "file")
table_lerp :: #force_inline proc "contextless" (table: ^[SRGB_STEPS + 1]f32, t: f32) -> f32 {
	i := min(int(t), SRGB_STEPS - 1)
	return table[i] + (table[i + 1] - table[i]) * (t - f32(i))
}

// log2 of a normal positive float: its exponent and its mantissa's (a
// texture's level; math.log2 goes through f64's ln).
@(private)
log2_fast :: #force_inline proc "contextless" (x: f32) -> f32 {
	bits := transmute(u32)x
	mantissa := f32(bits & 0x7fffff) * (SRGB_STEPS / f32(1 << 23))
	return f32(i32(bits >> 23) - 127) + table_lerp(&log2_mantissa, mantissa)
}

srgb_to_linear :: #force_inline proc "contextless" (c: f32) -> f32 {
	if c >= 0 && c <= 1 {
		return srgb_tables.to_linear[int(c * SRGB_STEPS + 0.5)]
	}
	return srgb_to_linear_exact(c)
}

linear_to_srgb :: #force_inline proc "contextless" (value: f32) -> f32 {
	c := min(value, 1) if value > 0 else 0 // NaN: 0
	return srgb_tables.to_srgb[int(c * SRGB_STEPS + 0.5)]
}

srgb_to_linear_exact :: proc "contextless" (c: f32) -> f32 {
	return c / 12.92 if c <= 0.04045 else math.pow((c + 0.055) / 1.055, 2.4)
}

linear_to_srgb_exact :: proc "contextless" (value: f32) -> f32 {
	c := clamp(value, 0, 1)
	return c * 12.92 if c <= 0.0031308 else 1.055 * math.pow(c, 1 / 2.4) - 0.055
}

linear :: #force_inline proc "contextless" (c: Vec3) -> Vec3 {
	return {srgb_to_linear(c.r), srgb_to_linear(c.g), srgb_to_linear(c.b)}
}

linear_to_srgb_vec3 :: #force_inline proc "contextless" (c: Vec3) -> Vec3 {
	return {linear_to_srgb(c.r), linear_to_srgb(c.g), linear_to_srgb(c.b)}
}

srgb8 :: #force_inline proc "contextless" (linear_val: f32) -> u8 {
	return u8(linear_to_srgb(linear_val) * 255 + 0.5)
}

// three.js's tone mapping modes (`tone_map` in shaders/post.frag).
Tone_Mapping :: enum u8 {
	None        = 0, // NoToneMapping
	Linear      = 1, // LinearToneMapping
	Reinhard    = 2, // ReinhardToneMapping
	Cineon      = 3, // CineonToneMapping
	ACES_Filmic = 4, // ACESFilmicToneMapping (Stephen Hill fit)
	AgX         = 5, // AgXToneMapping
	Neutral     = 6, // NeutralToneMapping (Khronos PBR Neutral)
}

// ---- tables -----------------------------------------------------------------------------

// Grows a dynamic array with geometric reserve so arena allocators do not
// strand O(N^2) backing memory across incremental resizes. The new capacity
// is never zeroed: table_grow zeroes the new elements, table_grow_non_zero
// leaves them for the caller to write.
table_grow :: #force_inline proc(table: ^[dynamic]$T, length: int) {
	if length > cap(table^) {
		non_zero_reserve(table, max(length, 2 * cap(table^), 64))
	}
	resize(table, length)
}

table_grow_non_zero :: #force_inline proc(table: ^[dynamic]$T, length: int) {
	if length > cap(table^) {
		non_zero_reserve(table, max(length, 2 * cap(table^), 64))
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
