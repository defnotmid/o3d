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

srgb8 :: #force_inline proc "contextless" (linear_val: f32) -> u8 {
	return u8(linear_to_srgb(linear_val) * 255 + 0.5)
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
