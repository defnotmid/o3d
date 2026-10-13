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
// position and normal, near-plane clipping. Samples hold sRGB premultiplied
// RGBA8, so the resolve averages what the GPU's does.
//
// A frame is three phases, each spread over the caller's pool (nil: run on
// the caller): chunks of the drawing order (up to CHUNK_TRIS triangles) are
// transformed and clipped into their own triangle lists and counted per
// band of rows; once the counts are summed, each chunk lists its triangles
// in the bands, so a band's list keeps the drawing order; then each band
// clears its rows, draws its list and resolves its pixels.

SAMPLES :: 4
// Sample positions in a pixel (the standard 4x MSAA pattern).
SAMPLE_OFFSETS := [SAMPLES]Vec2{{0.375, 0.125}, {0.875, 0.375}, {0.125, 0.625}, {0.625, 0.875}}
// The samples of a pixel a triangle covers.
Sample_Mask :: bit_set[0 ..< SAMPLES; u8]
BAND_ROWS :: 32
CHUNK_TRIS :: 8192

Software :: struct {
	width, height: int,
	color:         []u32, // per sample, RGBA8 sRGB premultiplied
	depth:         []f32, // per sample, 0 near .. 1 far
	pixels:        []u8, // the resolved frame, RGBA8 premultiplied
	// Scratch per frame (kept, with the chunks' triangle lists, to reuse).
	order:         [dynamic]int,
	materials:     [dynamic]Shading_Material, // per draw
	// The frame's chunks are chunks[:chunk_count]; the ones past it keep
	// their lists' memory for later frames.
	chunks:        [dynamic]Chunk,
	chunk_count:   int,
	// Per chunk and band (chunk * bands + band): the chunk's triangles in
	// the band, then where it lists them in band_tris.
	chunk_bands:   [dynamic]int,
	// Band b's triangles are band_tris[band_start[b]:band_start[b + 1]].
	band_start:    [dynamic]int,
	band_tris:     [dynamic]Tri_Ref,
}

software_destroy :: proc(s: ^Software) {
	delete(s.color)
	delete(s.depth)
	delete(s.pixels)
	delete(s.order)
	delete(s.materials)
	for &c in s.chunks {
		delete(c.tris)
	}
	delete(s.chunks)
	delete(s.chunk_bands)
	delete(s.band_start)
	delete(s.band_tris)
	s^ = {}
}

// A triangle ready to draw: screen positions (x, y, depth) with 1/w, and
// per vertex world position and normal divided by w.
@(private)
Tri :: struct {
	p:            [3]Vec3,
	inv_w:        [3]f32,
	world:        [3]Vec3, // over w
	normal:       [3]Vec3, // over w
	draw:         int,
	min_y, max_y: int,
}

// A run of one draw's triangles (indices[first:][:count * 3]) and the
// screen triangles made from them; a structure of arrays, since binning
// reads only the row spans.
@(private)
Chunk :: struct {
	order:        int, // the draw's place in `order`
	first, count: int,
	tris:         #soa[dynamic]Tri,
}

@(private)
Tri_Ref :: struct {
	chunk, tri: u32,
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
	view, proj := camera_matrices(scene.camera, width, height)
	lights: [MAX_LIGHTS]Shading_Light
	nl := shading_lights(scene.lights[:], &lights)
	shading_materials(scene_materials(scene), &s.materials)
	draw_order(scene, geos, view, proj, &s.order)
	make_chunks(s, scene, geos)
	bands := (height + BAND_ROWS - 1) / BAND_ROWS

	job := Frame_Job {
		s         = s,
		scene     = scene,
		geos      = geos,
		view_proj = proj * view,
		lights    = lights[:nl],
		eye       = scene.camera.position,
		bands     = bands,
	}
	resize(&s.chunk_bands, s.chunk_count * bands)
	parallel(pool, s.chunk_count, &job, transform_chunk)
	sum_bands(s, bands)
	parallel(pool, s.chunk_count, &job, list_chunk)
	parallel(pool, bands, &job, draw_band)
	return s.pixels
}

// Splits the drawing order into chunks of at most CHUNK_TRIS triangles.
@(private = "file")
make_chunks :: proc(s: ^Software, scene: ^Scene, geos: ^Geometries) {
	n := 0
	for di, oi in s.order {
		geo, _ := geometry_get(geos, scene.draws[di].geometry)
		tris := int(geo.index_count / 3)
		for first := 0; first < tris; first += CHUNK_TRIS {
			if n == len(s.chunks) {
				append(&s.chunks, Chunk{})
			}
			c := &s.chunks[n]
			c.order, c.first, c.count = oi, first, min(CHUNK_TRIS, tris - first)
			n += 1
		}
	}
	s.chunk_count = n
}

@(private = "file")
Frame_Job :: struct {
	s:         ^Software,
	scene:     ^Scene,
	geos:      ^Geometries,
	view_proj: Mat4,
	lights:    []Shading_Light,
	eye:       Vec3,
	bands:     int,
}

@(private = "file")
band_range :: #force_inline proc "contextless" (min_y, max_y: int) -> (first, last: int) {
	return min_y / BAND_ROWS, (max_y - 1) / BAND_ROWS
}

// Phase 1: a chunk's triangles transformed, clipped and kept as screen
// triangles, then counted per band.
@(private = "file")
transform_chunk :: proc(data: rawptr, ci: int) {
	job := (^Frame_Job)(data)
	s := job.s
	c := &s.chunks[ci]
	clear(&c.tris)
	di := s.order[c.order]
	d := &job.scene.draws[di]
	vertices, all, _ := geometry_mesh(job.geos, d.geometry)
	indices := all[c.first * 3:][:c.count * 3]
	mvp := job.view_proj * d.model
	nm := normal_matrix(d.model)
	for t := 0; t + 2 < len(indices); t += 3 {
		clip, world, normal: [3]Vec4
		for k in 0 ..< 3 {
			vtx := vertices[indices[t + k]]
			p := Vec4{vtx.position.x, vtx.position.y, vtx.position.z, 1}
			clip[k] = mvp * p
			world[k] = d.model * p
			normal[k] = nm * Vec4{vtx.normal.x, vtx.normal.y, vtx.normal.z, 0}
		}
		clip_near(&c.tris, clip, world, normal, di, s.width, s.height, d.material.side)
	}
	counts := s.chunk_bands[ci * job.bands:][:job.bands]
	slice.zero(counts)
	max_ys := c.tris.max_y[:len(c.tris)]
	for min_y, i in c.tris.min_y[:len(c.tris)] {
		first, last := band_range(min_y, max_ys[i])
		for b in first ..= last {
			counts[b] += 1
		}
	}
}

// Between the phases: each band's start, and where in it each chunk lists
// its triangles (chunks in drawing order).
@(private = "file")
sum_bands :: proc(s: ^Software, bands: int) {
	resize(&s.band_start, bands + 1)
	at := 0
	for b in 0 ..< bands {
		s.band_start[b] = at
		for ci in 0 ..< s.chunk_count {
			n := s.chunk_bands[ci * bands + b]
			s.chunk_bands[ci * bands + b] = at
			at += n
		}
	}
	s.band_start[bands] = at
	resize(&s.band_tris, at)
}

// Phase 2: a chunk's triangles listed in the bands they touch.
@(private = "file")
list_chunk :: proc(data: rawptr, ci: int) {
	job := (^Frame_Job)(data)
	s := job.s
	c := &s.chunks[ci]
	at := s.chunk_bands[ci * job.bands:][:job.bands]
	max_ys := c.tris.max_y[:len(c.tris)]
	for min_y, i in c.tris.min_y[:len(c.tris)] {
		first, last := band_range(min_y, max_ys[i])
		for b in first ..= last {
			s.band_tris[at[b]] = {u32(ci), u32(i)}
			at[b] += 1
		}
	}
}

// Clips a triangle against the near plane (z >= 0 in clip space) and keeps
// what is left as one or two screen triangles.
@(private = "file")
clip_near :: proc(tris: ^#soa[dynamic]Tri, clip, world, normal: [3]Vec4, draw, width, height: int, side: Side) {
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
		emit(tris, {in_[0].c, in_[k].c, in_[k + 1].c}, {in_[0].w, in_[k].w, in_[k + 1].w}, {in_[0].n, in_[k].n, in_[k + 1].n}, draw, width, height, side)
	}
}

@(private = "file")
emit :: proc(tris: ^#soa[dynamic]Tri, clip, world, normal: [3]Vec4, draw, width, height: int, side: Side) {
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
	append(tris, t)
}

// Runs `work(data, i)` for every i in 0 ..< count: on the caller and up to a
// pool thread per item, each claiming the next item until none is left; the
// caller waits for the tasks to finish (they may start after the items ran
// out).
@(private = "file")
parallel :: proc(pool: ^thread.Pool, count: int, data: rawptr, work: proc(data: rawptr, i: int)) {
	Parallel :: struct {
		count:   int,
		data:    rawptr,
		work:    proc(data: rawptr, i: int),
		next:    int, // atomic: the next item
		running: int, // atomic: tasks not finished
		ctx:     runtime.Context,
	}
	claim :: proc(p: ^Parallel) {
		context = p.ctx
		for {
			i := sync.atomic_add(&p.next, 1)
			if i >= p.count {
				return
			}
			p.work(p.data, i)
		}
	}
	task :: proc(t: thread.Task) {
		p := (^Parallel)(t.data)
		claim(p)
		sync.atomic_sub(&p.running, 1)
	}
	p := Parallel{count = count, data = data, work = work, ctx = context}
	helpers := 0
	if pool != nil {
		sync.guard(&pool.mutex)
		helpers = min(len(pool.threads), count - 1)
	}
	p.running = max(helpers, 0)
	for i in 0 ..< helpers {
		thread.pool_add_task(pool, context.allocator, task, &p, i)
	}
	claim(&p)
	for sync.atomic_load(&p.running) > 0 {
		thread.yield()
	}
}

// Phase 3: a band's rows cleared, its triangles drawn in order, its pixels
// resolved (the average of a pixel's samples).
@(private = "file")
draw_band :: proc(data: rawptr, band: int) {
	job := (^Frame_Job)(data)
	s := job.s
	y0 := band * BAND_ROWS
	y1 := min(y0 + BAND_ROWS, s.height)
	slice.zero(s.color[y0 * s.width * SAMPLES:y1 * s.width * SAMPLES])
	slice.fill(s.depth[y0 * s.width * SAMPLES:y1 * s.width * SAMPLES], 1)
	for ref in s.band_tris[s.band_start[band]:s.band_start[band + 1]] {
		t := s.chunks[ref.chunk].tris[ref.tri] // gathered from the columns once
		m := &s.materials[t.draw]
		blend := .Blended in m.flags
		x_lo := clamp(int(math.floor(min(t.p[0].x, t.p[1].x, t.p[2].x))), 0, s.width)
		x_hi := clamp(int(math.ceil(max(t.p[0].x, t.p[1].x, t.p[2].x))), 0, s.width)
		area := edge(t.p[0].xy, t.p[1].xy, t.p[2].xy)
		if area == 0 {
			continue
		}
		flat_n: Vec3
		if .Flat_Shading in m.flags {
			w0 := t.world[0] / t.inv_w[0]
			flat_n = linalg.normalize(linalg.cross(t.world[1] / t.inv_w[1] - w0, t.world[2] / t.inv_w[2] - w0))
		}
		for y in max(t.min_y, y0) ..< min(t.max_y, y1) {
			for x in x_lo ..< x_hi {
				covered: Sample_Mask
				for k in 0 ..< SAMPLES {
					p := Vec2{f32(x) + SAMPLE_OFFSETS[k].x, f32(y) + SAMPLE_OFFSETS[k].y}
					if inside(t, p, area) {
						covered += {k}
					}
				}
				if covered == {} {
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
				if .Flat_Shading not_in m.flags {
					n = linalg.normalize((t.normal[0] * b0 + t.normal[1] * b1 + t.normal[2] * b2) / iw)
				}
				lit := shade(m, world, n, job.eye, job.lights) + m.emissive
				col := pack(lit, m.opacity)
				for k in covered {
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
	for i in y0 * s.width ..< y1 * s.width {
		acc: [4]u32
		for k in 0 ..< SAMPLES {
			c := s.color[i * SAMPLES + k]
			acc += {c & 0xff, (c >> 8) & 0xff, (c >> 16) & 0xff, c >> 24}
		}
		for ch in 0 ..< 4 {
			s.pixels[i * 4 + ch] = u8((acc[ch] + SAMPLES / 2) / SAMPLES)
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
