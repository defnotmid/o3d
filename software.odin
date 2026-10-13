package o3d

import "base:runtime"
import "core:math"
import "core:math/linalg"
import "core:slice"
import "core:sync"
import "core:thread"

// The CPU renderer: the pixel reference for the GPU's and the path without
// one. Four samples per pixel (the GPU path's 4x MSAA), a depth per sample,
// shading once per pixel per triangle at its centre with perspective-correct
// position and normal, near-plane clipping. Rows are split into bands, drawn
// as tasks on the caller's pool (nil: on the caller). Samples hold sRGB
// premultiplied RGBA8, so the resolve averages what the GPU's does.

SAMPLES :: 4
// Sample positions in a pixel (the standard 4x MSAA pattern).
SAMPLE_OFFSETS := [SAMPLES]Vec2{{0.375, 0.125}, {0.875, 0.375}, {0.125, 0.625}, {0.625, 0.875}}
BAND_ROWS :: 32

Software :: struct {
	width, height: int,
	color:         []u32, // per sample, RGBA8 sRGB premultiplied
	depth:         []f32, // per sample, 0 near .. 1 far
	pixels:        []u8, // the resolved frame, RGBA8 premultiplied
	// Scratch per frame.
	tris:          [dynamic]Tri,
	order:         [dynamic]int,
}

software_destroy :: proc(s: ^Software) {
	delete(s.color)
	delete(s.depth)
	delete(s.pixels)
	delete(s.tris)
	delete(s.order)
	s^ = {}
}

// A triangle ready to draw: screen positions (x, y, depth) with 1/w, and
// per vertex world position and normal divided by w.
@(private)
Tri :: struct {
	p:        [3]Vec3,
	inv_w:    [3]f32,
	world:    [3]Vec3, // over w
	normal:   [3]Vec3, // over w
	draw:     int,
	min_y, max_y: int,
}

// Draws `scene` into `width x height` premultiplied RGBA8 (returned,
// borrowed until the next render).
software_render :: proc(s: ^Software, scene: ^Scene, geos: ^Geometries, width, height: int, pool: ^thread.Pool = nil) -> []u8 {
	if s.width != width || s.height != height {
		delete(s.color)
		delete(s.depth)
		delete(s.pixels)
		s.width, s.height = width, height
		s.color = make([]u32, width * height * SAMPLES)
		s.depth = make([]f32, width * height * SAMPLES)
		s.pixels = make([]u8, width * height * 4)
	}
	slice.zero(s.color)
	slice.fill(s.depth, 1)
	clear(&s.tris)

	view, proj := camera_matrices(scene.camera, width, height)
	view_proj := proj * view
	lights: [MAX_LIGHTS]Shading_Light
	nl := shading_lights(scene.lights[:], &lights)

	draw_order(scene, view, &s.order)
	for di in s.order {
		d := &scene.draws[di]
		geo, ok := geometry_get(geos, d.geometry)
		if !ok {
			continue
		}
		mvp := view_proj * d.model
		nm := normal_matrix(d.model)
		for t := 0; t + 2 < len(geo.indices); t += 3 {
			clip, world, normal: [3]Vec4
			for k in 0 ..< 3 {
				vtx := geo.vertices[geo.indices[t + k]]
				p := Vec4{vtx.position.x, vtx.position.y, vtx.position.z, 1}
				clip[k] = mvp * p
				world[k] = d.model * p
				normal[k] = nm * Vec4{vtx.normal.x, vtx.normal.y, vtx.normal.z, 0}
			}
			clip_near(s, clip, world, normal, di, width, height, d.material.side)
		}
	}

	job := Band_Job {
		s       = s,
		scene   = scene,
		lights  = lights[:nl],
		eye     = scene.camera.position,
		bands   = (height + BAND_ROWS - 1) / BAND_ROWS,
		ctx     = context,
	}
	run_bands(&job, pool)

	// Resolve: the average of a pixel's samples.
	for i in 0 ..< width * height {
		acc: [4]u32
		for k in 0 ..< SAMPLES {
			c := s.color[i * SAMPLES + k]
			acc += {c & 0xff, (c >> 8) & 0xff, (c >> 16) & 0xff, c >> 24}
		}
		for ch in 0 ..< 4 {
			s.pixels[i * 4 + ch] = u8((acc[ch] + SAMPLES / 2) / SAMPLES)
		}
	}
	return s.pixels
}

// Clips a triangle against the near plane (z >= 0 in clip space) and keeps
// what is left as one or two screen triangles.
@(private = "file")
clip_near :: proc(s: ^Software, clip, world, normal: [3]Vec4, draw, width, height: int, side: Side) {
	Corner :: struct {
		c, w, n: Vec4,
	}
	in_: [4]Corner
	count := 0
	for k in 0 ..< 3 {
		a := Corner{clip[k], world[k], normal[k]}
		b := Corner{clip[(k + 1) % 3], world[(k + 1) % 3], normal[(k + 1) % 3]}
		if a.c.z >= 0 {
			in_[count] = a
			count += 1
		}
		if (a.c.z >= 0) != (b.c.z >= 0) {
			t := a.c.z / (a.c.z - b.c.z)
			in_[count] = Corner{linalg.lerp(a.c, b.c, t), linalg.lerp(a.w, b.w, t), linalg.lerp(a.n, b.n, t)}
			count += 1
		}
	}
	for k in 1 ..< count - 1 {
		emit(s, {in_[0].c, in_[k].c, in_[k + 1].c}, {in_[0].w, in_[k].w, in_[k + 1].w}, {in_[0].n, in_[k].n, in_[k + 1].n}, draw, width, height, side)
	}
}

@(private = "file")
emit :: proc(s: ^Software, clip, world, normal: [3]Vec4, draw, width, height: int, side: Side) {
	t: Tri
	t.draw = draw
	for k in 0 ..< 3 {
		w := clip[k].w if abs(clip[k].w) > 1e-9 else 1e-9
		iw := 1 / w
		t.inv_w[k] = iw
		t.p[k] = {(clip[k].x * iw * 0.5 + 0.5) * f32(width), (0.5 - clip[k].y * iw * 0.5) * f32(height), clip[k].z * iw}
		t.world[k] = world[k].xyz * iw
		t.normal[k] = normal[k].xyz * iw
	}
	// Screen area: negative is counter-clockwise on screen (y down), the
	// front face.
	area := (t.p[1].x - t.p[0].x) * (t.p[2].y - t.p[0].y) - (t.p[2].x - t.p[0].x) * (t.p[1].y - t.p[0].y)
	front := area < 0
	if area == 0 || (side == .Front && !front) || (side == .Back && front) {
		return
	}
	if !front {
		t.p[1], t.p[2] = t.p[2], t.p[1]
		t.inv_w[1], t.inv_w[2] = t.inv_w[2], t.inv_w[1]
		t.world[1], t.world[2] = t.world[2], t.world[1]
		t.normal[1], t.normal[2] = t.normal[2], t.normal[1]
	}
	lo := min(t.p[0].y, t.p[1].y, t.p[2].y)
	hi := max(t.p[0].y, t.p[1].y, t.p[2].y)
	t.min_y = clamp(int(math.floor(lo)), 0, height)
	t.max_y = clamp(int(math.ceil(hi)), 0, height)
	if t.min_y >= t.max_y {
		return
	}
	append(&s.tris, t)
}

@(private = "file")
Band_Job :: struct {
	s:       ^Software,
	scene:   ^Scene,
	lights:  []Shading_Light,
	eye:     Vec3,
	bands:   int,
	next:    int, // atomic: the next band to draw
	running: int, // atomic: tasks not finished
	ctx:     runtime.Context,
}

// Every band drawn, by the caller and up to a pool thread per band; each
// claims the next band until none is left, and the caller waits for the
// tasks to finish (they may start after the bands ran out).
@(private = "file")
run_bands :: proc(job: ^Band_Job, pool: ^thread.Pool) {
	helpers := 0
	if pool != nil {
		sync.guard(&pool.mutex)
		helpers = min(len(pool.threads), job.bands - 1)
	}
	job.running = helpers
	for i in 0 ..< helpers {
		thread.pool_add_task(pool, context.allocator, band_task, job, i)
	}
	bands_claim(job)
	for sync.atomic_load(&job.running) > 0 {
		thread.yield()
	}
}

@(private = "file")
band_task :: proc(t: thread.Task) {
	job := (^Band_Job)(t.data)
	bands_claim(job)
	sync.atomic_sub(&job.running, 1)
}

@(private = "file")
bands_claim :: proc(job: ^Band_Job) {
	context = job.ctx
	for {
		b := sync.atomic_add(&job.next, 1)
		if b >= job.bands {
			return
		}
		y0 := b * BAND_ROWS
		draw_band(job, y0, min(y0 + BAND_ROWS, job.s.height))
	}
}

@(private = "file")
draw_band :: proc(job: ^Band_Job, y0, y1: int) {
	s := job.s
	for &t in s.tris {
		if t.max_y <= y0 || t.min_y >= y1 {
			continue
		}
		draw := &job.scene.draws[t.draw]
		m := &draw.material
		base := linear(m.color)
		emissive := linear(m.emissive) * m.emissive_intensity
		opacity := clamp(m.opacity, 0, 1)
		blend := blended(m^)
		x_lo := clamp(int(math.floor(min(t.p[0].x, t.p[1].x, t.p[2].x))), 0, s.width)
		x_hi := clamp(int(math.ceil(max(t.p[0].x, t.p[1].x, t.p[2].x))), 0, s.width)
		area := edge(t.p[0].xy, t.p[1].xy, t.p[2].xy)
		if area == 0 {
			continue
		}
		flat_n: Vec3
		if m.flat_shading {
			w0 := t.world[0] / t.inv_w[0]
			flat_n = linalg.normalize(linalg.cross(t.world[1] / t.inv_w[1] - w0, t.world[2] / t.inv_w[2] - w0))
		}
		for y in max(t.min_y, y0) ..< min(t.max_y, y1) {
			for x in x_lo ..< x_hi {
				covered: [SAMPLES]bool
				any := false
				for k in 0 ..< SAMPLES {
					p := Vec2{f32(x) + SAMPLE_OFFSETS[k].x, f32(y) + SAMPLE_OFFSETS[k].y}
					if inside(t, p, area) {
						covered[k], any = true, true
					}
				}
				if !any {
					continue
				}
				// Shade once, at the pixel centre (clamped into the triangle by
				// its barycentrics), perspective-correct.
				c := Vec2{f32(x) + 0.5, f32(y) + 0.5}
				b0 := edge(t.p[1].xy, t.p[2].xy, c) / area
				b1 := edge(t.p[2].xy, t.p[0].xy, c) / area
				b2 := 1 - b0 - b1
				iw := b0 * t.inv_w[0] + b1 * t.inv_w[1] + b2 * t.inv_w[2]
				world := (t.world[0] * b0 + t.world[1] * b1 + t.world[2] * b2) / iw
				n := flat_n
				if !m.flat_shading {
					n = linalg.normalize((t.normal[0] * b0 + t.normal[1] * b1 + t.normal[2] * b2) / iw)
				}
				lit := shade(m, base, world, n, job.eye, job.lights) + emissive
				col := pack(lit, opacity if blend else 1)
				for k in 0 ..< SAMPLES {
					if !covered[k] {
						continue
					}
					p := Vec2{f32(x) + SAMPLE_OFFSETS[k].x, f32(y) + SAMPLE_OFFSETS[k].y}
					s0 := edge(t.p[1].xy, t.p[2].xy, p) / area
					s1 := edge(t.p[2].xy, t.p[0].xy, p) / area
					z := s0 * t.p[0].z + s1 * t.p[1].z + (1 - s0 - s1) * t.p[2].z
					i := (y * s.width + x) * SAMPLES + k
					if z < 0 || z > 1 || z >= s.depth[i] {
						continue
					}
					if blend {
						s.color[i] = over(col, s.color[i])
					} else {
						s.color[i] = col
						s.depth[i] = z
					}
				}
			}
		}
	}
}

@(private = "file")
edge :: #force_inline proc "contextless" (a, b, p: Vec2) -> f32 {
	return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x)
}

// Inside the (front-facing, area < 0) triangle, with a top-left rule so
// shared edges are drawn once.
@(private = "file")
inside :: #force_inline proc "contextless" (t: Tri, p: Vec2, area: f32) -> bool {
	for k in 0 ..< 3 {
		a, b := t.p[k].xy, t.p[(k + 1) % 3].xy
		e := edge(a, b, p)
		if e > 0 {
			return false
		}
		if e == 0 {
			// Top edge (horizontal, going right in screen) or left edge (going down).
			top_left := (a.y == b.y && b.x < a.x) || b.y > a.y
			if !top_left {
				return false
			}
		}
	}
	return true
}

// Linear colour to sRGB premultiplied RGBA8.
@(private = "file")
pack :: proc "contextless" (c: Vec3, alpha: f32) -> u32 {
	r := u32(math.round(linear_to_srgb(c.r) * alpha * 255))
	g := u32(math.round(linear_to_srgb(c.g) * alpha * 255))
	b := u32(math.round(linear_to_srgb(c.b) * alpha * 255))
	a := u32(math.round(alpha * 255))
	return r | g << 8 | b << 16 | a << 24
}

// Premultiplied `src` over `dst` (in the stored sRGB values, as the GPU's
// fixed-function blend on an UNORM target does).
@(private = "file")
over :: proc "contextless" (src, dst: u32) -> u32 {
	inv := 255 - (src >> 24)
	out: u32
	for sh in ([4]u32{0, 8, 16, 24}) {
		s := (src >> sh) & 0xff
		d := (dst >> sh) & 0xff
		out |= min(s + (d * inv + 127) / 255, 255) << sh
	}
	return out
}
