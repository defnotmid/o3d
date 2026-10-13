package o3d

import "base:runtime"
import "core:math"
import "core:math/linalg"
import "core:slice"
import "core:sync"
import "core:thread"

import "deps:oksl"

// The CPU renderer: the pixel reference for the GPU's and the path without
// one. Four samples per pixel (the GPU path's 4x MSAA), a depth per sample,
// shading once per pixel per triangle at its centre with perspective-correct
// values, near-plane clipping. Samples hold sRGB premultiplied RGBA8, so the
// resolve averages what the GPU's does.
//
// A frame (and before it, the shadow map, shadow.odin: depth only, one
// sample per texel) is three phases, each spread over the caller's pool (nil: run on
// the caller): chunks of the drawing order (up to CHUNK_TRIS triangles)
// run the vertex stage over the vertices they use, clip and keep screen
// triangles with their corners' values, and count them per band of rows;
// once the counts are summed, each chunk lists its triangles in the bands,
// so a band's list keeps the drawing order; then each band clears its rows,
// draws its list and resolves its pixels.
//
// A triangle's corners carry floats interpolated across it: for the
// standard material its world position and normal, for a ShaderMaterial
// its varyings (run by oksl: the vertex program over the chunk's vertices,
// the fragment program over 2x2 quads of a run of the draw's triangles,
// then the same coverage, depth and blending as the standard material).

SAMPLES :: 4
// Sample positions in a pixel (the standard 4x MSAA pattern).
SAMPLE_OFFSETS := [SAMPLES]Vec2{{0.375, 0.125}, {0.875, 0.375}, {0.125, 0.625}, {0.625, 0.875}}
// The samples of a pixel a triangle covers.
Sample_Mask :: bit_set[0 ..< SAMPLES; u8]
BAND_ROWS :: 32
CHUNK_TRIS :: 8192
// Quads shaded by one oksl call (a run of one draw's triangles).
QUAD_BATCH :: 256
// The standard material's corner floats: world position, normal.
STANDARD_FLOATS :: 6

Software :: struct {
	width, height: int,
	color:         []u32, // per sample, RGBA8 sRGB premultiplied
	depth:         []f32, // per sample, 0 near .. 1 far
	pixels:        []u8, // the resolved frame, RGBA8 premultiplied
	// Scratch per frame (kept, with the chunks' and workers' arrays, to
	// reuse).
	order:         [dynamic]int,
	materials:     [dynamic]Shading_Material, // per draw
	blocks:        Shader_Blocks,
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
	// One per thread that runs a phase (the caller is 0).
	workers:       [dynamic]Worker,
	// The shadow light's depth per texel (shadow.size squared), drawn first.
	shadow_map:    []f32,
}

software_destroy :: proc(s: ^Software) {
	delete(s.color)
	delete(s.depth)
	delete(s.pixels)
	delete(s.order)
	delete(s.materials)
	shader_blocks_destroy(&s.blocks)
	for &c in s.chunks {
		delete(c.tris)
		delete(c.values)
	}
	delete(s.chunks)
	delete(s.chunk_bands)
	delete(s.band_start)
	delete(s.band_tris)
	for &w in s.workers {
		worker_destroy(&w)
	}
	delete(s.workers)
	delete(s.shadow_map)
	s^ = {}
}

// A triangle ready to draw: screen positions (x, y, depth) with 1/w, and
// where its corners' floats are in its chunk's `values` (per corner, the
// draw's floats, divided by w unless flat).
@(private)
Tri :: struct {
	p:            [3]Vec3,
	inv_w:        [3]f32,
	values:       u32,
	draw:         int,
	min_y, max_y: int,
	front:        bool, // facing the camera (before back faces were turned)
}

// A run of one draw's triangles (indices[first:][:count * 3]) and the
// screen triangles made from them; a structure of arrays, since binning
// reads only the row spans.
@(private)
Chunk :: struct {
	order:        int, // the draw's place in `order`
	first, count: int,
	tris:         #soa[dynamic]Tri,
	values:       [dynamic]f32,
}

@(private)
Tri_Ref :: struct {
	chunk, tri: u32,
}

// A thread's working memory: the vertex stage's outputs over a chunk's
// vertices, and a batch of quads for a ShaderMaterial.
@(private)
Worker :: struct {
	vm:        oksl.Scratch,
	positions: [dynamic][4]f32, // clip space
	values:    [dynamic]f32,
	quads:     struct {
		origin:  [dynamic][2]i32,
		helpers: [dynamic]oksl.Quad_Lanes,
		facing:  [dynamic]bool,
		covered: [dynamic][4]Sample_Mask, // per lane
		tri:     [dynamic]Tri_Ref,
		values:  [dynamic]f32, // structure of arrays, as run_fragment reads them
		colours: [dynamic][4]f32,
		shaded:  [dynamic]oksl.Quad_Lanes,
	},
}

@(private = "file")
worker_destroy :: proc(w: ^Worker) {
	oksl.scratch_destroy(&w.vm)
	delete(w.positions)
	delete(w.values)
	delete(w.quads.origin)
	delete(w.quads.helpers)
	delete(w.quads.facing)
	delete(w.quads.covered)
	delete(w.quads.tri)
	delete(w.quads.values)
	delete(w.quads.colours)
	delete(w.quads.shaded)
}

// Draws `scene` into `width x height` premultiplied RGBA8 (returned,
// borrowed until the next render). `shaders` holds the ShaderMaterials'
// shaders (nil: none).
software_render :: proc(s: ^Software, scene: ^Scene, geos: ^Geometries, shaders: ^Shaders, width, height: int, pool: ^thread.Pool = nil) -> []u8 {
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
	shader_blocks(scene, shaders, view, proj, &s.blocks)
	threads := 0
	if pool != nil {
		sync.guard(&pool.mutex)
		threads = len(pool.threads)
	}
	if len(s.workers) < threads + 1 {
		resize(&s.workers, threads + 1)
	}
	job := Frame_Job {
		s       = s,
		scene   = scene,
		geos    = geos,
		shaders = shaders,
		lights  = lights[:nl],
		eye     = scene.camera.position,
		shadow  = frame_shadow(scene),
	}
	if job.shadow.light >= 0 {
		size := job.shadow.size
		if len(s.shadow_map) != size * size {
			delete(s.shadow_map)
			s.shadow_map = make([]f32, size * size)
		}
		shadow_casters(scene, geos, shaders, &job.shadow, &s.order)
		job.view_proj, job.width, job.height, job.depth_only = job.shadow.view_proj, size, size, true
		run_pass(s, scene, geos, &job, pool, draw_depth_band)
	}
	draw_order(scene, geos, shaders, view, proj, &s.order)
	job.view_proj, job.width, job.height, job.depth_only = proj * view, width, height, false
	run_pass(s, scene, geos, &job, pool, draw_band)
	return s.pixels
}

// The phases over the draws in `s.order` into a `job.width x job.height`
// target, its bands drawn by `draw`.
@(private = "file")
run_pass :: proc(s: ^Software, scene: ^Scene, geos: ^Geometries, job: ^Frame_Job, pool: ^thread.Pool, draw: proc(data: rawptr, band, worker: int)) {
	make_chunks(s, scene, geos)
	job.bands = (job.height + BAND_ROWS - 1) / BAND_ROWS
	resize(&s.chunk_bands, s.chunk_count * job.bands)
	parallel(pool, s.chunk_count, job, transform_chunk)
	sum_bands(s, job.bands)
	parallel(pool, s.chunk_count, job, list_chunk)
	parallel(pool, job.bands, job, draw)
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
	s:             ^Software,
	scene:         ^Scene,
	geos:          ^Geometries,
	shaders:       ^Shaders,
	lights:        []Shading_Light,
	eye:           Vec3,
	shadow:        Shadow,
	// The pass: its camera and target, and whether it draws only depth
	// (the shadow map: positions only, the casters' other side).
	view_proj:     Mat4,
	width, height: int,
	depth_only:    bool,
	bands:         int,
}

@(private = "file")
band_range :: #force_inline proc "contextless" (min_y, max_y: int) -> (first, last: int) {
	return min_y / BAND_ROWS, (max_y - 1) / BAND_ROWS
}

// A triangle corner on its way to the screen: clip position and floats.
@(private = "file")
Corner :: struct {
	c: Vec4,
	v: [MAX_VARYING_FLOATS]f32,
}

// Phase 1: a chunk's vertices through the vertex stage, its triangles
// clipped and kept as screen triangles, then counted per band.
@(private = "file")
transform_chunk :: proc(data: rawptr, ci, wi: int) {
	job := (^Frame_Job)(data)
	s := job.s
	w := &s.workers[wi]
	c := &s.chunks[ci]
	clear(&c.tris)
	clear(&c.values)
	di := s.order[c.order]
	d := &job.scene.draws[di]
	vertices, all, _ := geometry_mesh(job.geos, d.geometry)
	indices := all[c.first * 3:][:c.count * 3]
	lo, hi := max(u32), u32(0)
	for i in indices {
		lo, hi = min(lo, i), max(hi, i)
	}
	count := int(hi - lo) + 1
	sh := shader_get(job.shaders, s.materials[di].shader) if !job.depth_only else nil
	nf := sh.varying_floats if sh != nil else STANDARD_FLOATS
	if job.depth_only {
		nf = 0
	}
	flat := shader_flat(sh) if sh != nil else nil
	resize(&w.positions, count)
	resize(&w.values, count * nf)
	if job.depth_only {
		mvp := job.view_proj * d.model
		for vtx, i in vertices[lo:][:count] {
			w.positions[i] = mvp * Vec4{vtx.position.x, vtx.position.y, vtx.position.z, 1}
		}
	} else if sh == nil {
		mvp := job.view_proj * d.model
		nm := normal_matrix(d.model)
		for vtx, i in vertices[lo:][:count] {
			p := Vec4{vtx.position.x, vtx.position.y, vtx.position.z, 1}
			w.positions[i] = mvp * p
			world := d.model * p
			normal := nm * Vec4{vtx.normal.x, vtx.normal.y, vtx.normal.z, 0}
			w.values[i * 6 + 0], w.values[i * 6 + 1], w.values[i * 6 + 2] = world.x, world.y, world.z
			w.values[i * 6 + 3], w.values[i * 6 + 4], w.values[i * 6 + 5] = normal.x, normal.y, normal.z
		}
	} else {
		attrs: [16]oksl.Attribute
		for a in Vertex_Attribute {
			loc := sh.attributes[a]
			if loc < 0 {
				continue
			}
			row := VERTEX_ATTRIBUTES[a]
			attrs[loc] = {rawptr(uintptr(&vertices[lo]) + row.offset), size_of(Vertex), .Float2 if row.components == 2 else .Float3}
		}
		oksl.run_vertex(&sh.pipeline.vertex.program, block(&s.blocks, di, .Vertex), attrs[:], count, w.positions[:], w.values[:], &w.vm)
	}
	side := shadow_side(d.material.side) if job.depth_only else d.material.side
	values := raw_data(w.values) // nil with no floats (the shadow map): never read
	for t := 0; t + 2 < len(indices); t += 3 {
		v := [3]int{int(indices[t] - lo), int(indices[t + 1] - lo), int(indices[t + 2] - lo)}
		clip := [3]Vec4{w.positions[v[0]], w.positions[v[1]], w.positions[v[2]]}
		if clip[0].z >= 0 && clip[1].z >= 0 && clip[2].z >= 0 {
			emit(c, clip, {values[v[0] * nf:], values[v[1] * nf:], values[v[2] * nf:]}, nf, flat, di, job.width, job.height, side)
			continue
		}
		corners: [3]Corner = ---
		for k in 0 ..< 3 {
			corners[k].c = clip[k]
			copy(corners[k].v[:nf], w.values[v[k] * nf:][:nf])
		}
		clip_near(c, &corners, nf, flat, di, job.width, job.height, side)
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

// A draw's uniform block for a stage (nil: none).
@(private = "file")
block :: proc(b: ^Shader_Blocks, draw: int, stage: Shader_Stage) -> []u8 {
	at := b.at[draw][stage]
	return b.bytes[at:] if at >= 0 else nil
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
list_chunk :: proc(data: rawptr, ci, _: int) {
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

// Clips a triangle crossing the near plane (z >= 0 in clip space) and keeps
// what is left as one or two screen triangles; `nf` floats per corner,
// the `flat` ones the first vertex's, carried, not interpolated.
@(private = "file")
clip_near :: proc(c: ^Chunk, tri: ^[3]Corner, nf: int, flat: []u8, draw, width, height: int, side: Side) {
	for f in flat {
		tri[1].v[f], tri[2].v[f] = tri[0].v[f], tri[0].v[f]
	}
	in_: [4]Corner = ---
	count := 0
	for k in 0 ..< 3 {
		a, b := &tri[k], &tri[(k + 1) % 3]
		if a.c.z >= 0 {
			in_[count] = a^
			count += 1
		}
		if (a.c.z >= 0) != (b.c.z >= 0) {
			t := a.c.z / (a.c.z - b.c.z)
			o := &in_[count]
			o.c = linalg.lerp(a.c, b.c, t)
			for f in 0 ..< nf {
				o.v[f] = a.v[f] * (1 - t) + b.v[f] * t
			}
			for f in flat {
				o.v[f] = a.v[f]
			}
			count += 1
		}
	}
	for k in 1 ..< count - 1 {
		emit(c, {in_[0].c, in_[k].c, in_[k + 1].c}, {&in_[0].v[0], &in_[k].v[0], &in_[k + 1].v[0]}, nf, flat, draw, width, height, side)
	}
}

// A clipped triangle to the screen, kept if it faces a drawn side and
// reaches a row: its corners' `nf` floats each (`values`), divided by w
// except the flat ones, the first corner's.
@(private = "file")
emit :: proc(c: ^Chunk, clip: [3]Vec4, values: [3][^]f32, nf: int, flat: []u8, draw, width, height: int, side: Side) {
	t: Tri
	t.draw = draw
	for k in 0 ..< 3 {
		clip := clip[k]
		w := clip.w if abs(clip.w) > 1e-9 else 1e-9
		iw := 1 / w
		t.inv_w[k] = iw
		t.p[k] = {(clip.x * iw * 0.5 + 0.5) * f32(width), (0.5 - clip.y * iw * 0.5) * f32(height), clip.z * iw}
	}
	// Screen area: negative is counter-clockwise on screen (y down), the
	// front face.
	area := (t.p[1].x - t.p[0].x) * (t.p[2].y - t.p[0].y) - (t.p[2].x - t.p[0].x) * (t.p[1].y - t.p[0].y)
	t.front = area < 0
	if area == 0 || (side == .Front && !t.front) || (side == .Back && t.front) {
		return
	}
	order := [3]int{0, 1, 2}
	if !t.front {
		order = {0, 2, 1}
		t.p[1], t.p[2] = t.p[2], t.p[1]
		t.inv_w[1], t.inv_w[2] = t.inv_w[2], t.inv_w[1]
	}
	lo := min(t.p[0].y, t.p[1].y, t.p[2].y)
	hi := max(t.p[0].y, t.p[1].y, t.p[2].y)
	t.min_y = clamp(int(math.floor(lo)), 0, height)
	t.max_y = clamp(int(math.ceil(hi)), 0, height)
	if t.min_y >= t.max_y {
		return
	}
	at := len(c.values)
	t.values = u32(at)
	non_zero_resize(&c.values, at + 3 * nf)
	out := c.values[at:]
	for k, i in order {
		iw := t.inv_w[i]
		for f in 0 ..< nf {
			out[i * nf + f] = values[k][f] * iw
		}
		for f in flat {
			out[i * nf + int(f)] = values[0][f]
		}
	}
	append(&c.tris, t)
}

// Runs `work(data, i, worker)` for every i in 0 ..< count: on the caller
// (worker 0) and up to a pool thread per item (worker 1 + its task), each
// claiming the next item until none is left; the caller waits for the
// tasks to finish (they may start after the items ran out).
@(private = "file")
parallel :: proc(pool: ^thread.Pool, count: int, data: rawptr, work: proc(data: rawptr, i, worker: int)) {
	Parallel :: struct {
		count:   int,
		data:    rawptr,
		work:    proc(data: rawptr, i, worker: int),
		next:    int, // atomic: the next item
		running: int, // atomic: tasks not finished
		ctx:     runtime.Context,
	}
	claim :: proc(p: ^Parallel, worker: int) {
		context = p.ctx
		for {
			i := sync.atomic_add(&p.next, 1)
			if i >= p.count {
				return
			}
			p.work(p.data, i, worker)
		}
	}
	task :: proc(t: thread.Task) {
		p := (^Parallel)(t.data)
		claim(p, 1 + t.user_index)
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
	claim(&p, 0)
	for sync.atomic_load(&p.running) > 0 {
		thread.yield()
	}
}

// Phase 3: a band's rows cleared, its triangles drawn in order, its pixels
// resolved (the average of a pixel's samples).
@(private = "file")
draw_band :: proc(data: rawptr, band, wi: int) {
	job := (^Frame_Job)(data)
	s := job.s
	y0 := band * BAND_ROWS
	y1 := min(y0 + BAND_ROWS, s.height)
	slice.zero(s.color[y0 * s.width * SAMPLES:y1 * s.width * SAMPLES])
	slice.fill(s.depth[y0 * s.width * SAMPLES:y1 * s.width * SAMPLES], 1)
	refs := s.band_tris[s.band_start[band]:s.band_start[band + 1]]
	for r := 0; r < len(refs); {
		t := s.chunks[refs[r].chunk].tris[refs[r].tri] // gathered from the columns once
		if s.materials[t.draw].shader != 0 {
			r += draw_custom(job, &s.workers[wi], refs[r:], y0, y1)
			continue
		}
		draw_standard(job, refs[r], t, y0, y1)
		r += 1
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

// A band of the shadow map: its rows cleared to the far plane, then each
// triangle's depth at the texel centres it covers, nearest kept.
@(private = "file")
draw_depth_band :: proc(data: rawptr, band, _: int) {
	job := (^Frame_Job)(data)
	s := job.s
	size := job.width
	y0 := band * BAND_ROWS
	y1 := min(y0 + BAND_ROWS, job.height)
	slice.fill(s.shadow_map[y0 * size:y1 * size], 1)
	for ref in s.band_tris[s.band_start[band]:s.band_start[band + 1]] {
		t := s.chunks[ref.chunk].tris[ref.tri]
		area := edge(t.p[0].xy, t.p[1].xy, t.p[2].xy)
		if area == 0 {
			continue
		}
		x_lo := clamp(int(math.floor(min(t.p[0].x, t.p[1].x, t.p[2].x))), 0, size)
		x_hi := clamp(int(math.ceil(max(t.p[0].x, t.p[1].x, t.p[2].x))), 0, size)
		for y in max(t.min_y, y0) ..< min(t.max_y, y1) {
			for x in x_lo ..< x_hi {
				c := Vec2{f32(x) + 0.5, f32(y) + 0.5}
				if !inside(t, c, area) {
					continue
				}
				b0 := edge(t.p[1].xy, t.p[2].xy, c) / area
				b1 := edge(t.p[2].xy, t.p[0].xy, c) / area
				z := b0 * t.p[0].z + b1 * t.p[1].z + (1 - b0 - b1) * t.p[2].z
				i := y * size + x
				if z >= 0 && z <= 1 && z < s.shadow_map[i] {
					s.shadow_map[i] = z
				}
			}
		}
	}
}

// A standard-material triangle's rows in the band, shaded per pixel.
@(private = "file")
draw_standard :: #force_no_inline proc(job: ^Frame_Job, ref: Tri_Ref, t: Tri, y0, y1: int) {
	t := t // a local the pixel loop keeps in registers (a parameter is a reference)
	s := job.s
	m := &s.materials[t.draw]
	blend := .Blended in m.flags
	receive := .Receive_Shadow in m.flags && job.shadow.light >= 0
	x_lo := clamp(int(math.floor(min(t.p[0].x, t.p[1].x, t.p[2].x))), 0, s.width)
	x_hi := clamp(int(math.ceil(max(t.p[0].x, t.p[1].x, t.p[2].x))), 0, s.width)
	area := edge(t.p[0].xy, t.p[1].xy, t.p[2].xy)
	if area == 0 {
		return
	}
	// World position and normal over w per corner.
	vs := s.chunks[ref.chunk].values[t.values:][:3 * STANDARD_FLOATS]
	world, normal: [3]Vec3
	for k in 0 ..< 3 {
		world[k] = {vs[k * 6 + 0], vs[k * 6 + 1], vs[k * 6 + 2]}
		normal[k] = {vs[k * 6 + 3], vs[k * 6 + 4], vs[k * 6 + 5]}
	}
	flat_n: Vec3
	if .Flat_Shading in m.flags {
		w0 := world[0] / t.inv_w[0]
		flat_n = linalg.normalize(linalg.cross(world[1] / t.inv_w[1] - w0, world[2] / t.inv_w[2] - w0))
	}
	for y in max(t.min_y, y0) ..< min(t.max_y, y1) {
		for x in x_lo ..< x_hi {
			covered := coverage(t, x, y, area)
			if covered == {} {
				continue
			}
			// Shade once, at the pixel centre, perspective-correct.
			c := Vec2{f32(x) + 0.5, f32(y) + 0.5}
			b0 := edge(t.p[1].xy, t.p[2].xy, c) / area
			b1 := edge(t.p[2].xy, t.p[0].xy, c) / area
			b2 := 1 - b0 - b1
			iw := b0 * t.inv_w[0] + b1 * t.inv_w[1] + b2 * t.inv_w[2]
			p := (world[0] * b0 + world[1] * b1 + world[2] * b2) / iw
			n := flat_n
			if .Flat_Shading not_in m.flags {
				n = linalg.normalize((normal[0] * b0 + normal[1] * b1 + normal[2] * b2) / iw)
			}
			shadow: f32 = 1
			if receive {
				shadow = shadow_factor(s.shadow_map, &job.shadow, p, n)
			}
			lit := shade(m, p, n, job.eye, job.lights, shadow) + m.emissive
			write_samples(s, t, x, y, area, covered, pack(lit, m.opacity), blend)
		}
	}
}

// A run of one ShaderMaterial draw's triangles from `refs` (at least the
// first): their quads in the band shaded by the fragment program in one
// batch, then written in triangle order. Returns how many it took.
@(private = "file")
draw_custom :: #force_no_inline proc(job: ^Frame_Job, w: ^Worker, refs: []Tri_Ref, y0, y1: int) -> int {
	s := job.s
	q := &w.quads
	clear(&q.origin)
	clear(&q.helpers)
	clear(&q.facing)
	clear(&q.covered)
	clear(&q.tri)
	draw := -1
	taken := 0
	for ref in refs {
		t := s.chunks[ref.chunk].tris[ref.tri]
		if draw >= 0 && (t.draw != draw || len(q.origin) >= QUAD_BATCH) {
			break
		}
		draw = t.draw
		taken += 1
		area := edge(t.p[0].xy, t.p[1].xy, t.p[2].xy)
		if area == 0 {
			continue
		}
		// Quads at even pixels: a band (BAND_ROWS even) holds whole quads.
		x_lo := clamp(int(math.floor(min(t.p[0].x, t.p[1].x, t.p[2].x))), 0, s.width) &~ 1
		x_hi := clamp(int(math.ceil(max(t.p[0].x, t.p[1].x, t.p[2].x))), 0, s.width)
		for qy := max(t.min_y, y0) &~ 1; qy < min(t.max_y, y1); qy += 2 {
			for qx := x_lo; qx < x_hi; qx += 2 {
				lanes: [4]Sample_Mask
				helpers: oksl.Quad_Lanes
				for l in 0 ..< 4 {
					x, y := qx + l & 1, qy + l >> 1
					if x < s.width && y < y1 {
						lanes[l] = coverage(t, x, y, area)
					}
					if lanes[l] == {} {
						helpers += {l}
					}
				}
				if card(helpers) == 4 {
					continue
				}
				append(&q.origin, [2]i32{i32(qx), i32(qy)})
				append(&q.helpers, helpers)
				append(&q.facing, t.front)
				append(&q.covered, lanes)
				append(&q.tri, ref)
			}
		}
	}
	nq := len(q.origin)
	if nq == 0 {
		return taken
	}
	sh := shader_get(job.shaders, s.materials[draw].shader)
	nf := sh.varying_floats
	// Each lane's varyings, perspective-correct (helpers' extrapolated).
	resize(&q.values, nf * nq * 4)
	for qi in 0 ..< nq {
		ref := q.tri[qi]
		t := s.chunks[ref.chunk].tris[ref.tri]
		area := edge(t.p[0].xy, t.p[1].xy, t.p[2].xy)
		vs := s.chunks[ref.chunk].values[t.values:][:3 * nf]
		for l in 0 ..< 4 {
			c := Vec2{f32(int(q.origin[qi].x) + l & 1) + 0.5, f32(int(q.origin[qi].y) + l >> 1) + 0.5}
			b0 := edge(t.p[1].xy, t.p[2].xy, c) / area
			b1 := edge(t.p[2].xy, t.p[0].xy, c) / area
			b2 := 1 - b0 - b1
			iw := b0 * t.inv_w[0] + b1 * t.inv_w[1] + b2 * t.inv_w[2]
			j := qi * 4 + l
			for f in 0 ..< nf {
				q.values[f * nq * 4 + j] = (vs[f] * b0 + vs[nf + f] * b1 + vs[2 * nf + f] * b2) / iw
			}
			for f in shader_flat(sh) {
				q.values[int(f) * nq * 4 + j] = vs[f] // kept as is: the first corner's
			}
		}
	}
	resize(&q.colours, nq * 4)
	resize(&q.shaded, nq)
	quads := oksl.Fragment_Quads {
		count        = nq,
		origin       = q.origin[:],
		front_facing = q.facing[:],
		helpers      = q.helpers[:],
		varyings     = q.values[:],
	}
	oksl.run_fragment(&sh.pipeline.fragment.program, block(&s.blocks, draw, .Fragment), quads, q.colours[:], q.shaded[:], &w.vm)
	blend := .Blended in s.materials[draw].flags
	for qi in 0 ..< nq {
		ref := q.tri[qi]
		t := s.chunks[ref.chunk].tris[ref.tri]
		area := edge(t.p[0].xy, t.p[1].xy, t.p[2].xy)
		for l in q.shaded[qi] {
			x, y := int(q.origin[qi].x) + l & 1, int(q.origin[qi].y) + l >> 1
			write_samples(s, t, x, y, area, q.covered[qi][l], pack_raw(q.colours[qi * 4 + l], blend), blend)
		}
	}
	return taken
}

// The samples of pixel (x, y) inside the triangle.
@(private = "file")
coverage :: #force_inline proc "contextless" (t: Tri, x, y: int, area: f32) -> (covered: Sample_Mask) {
	for k in 0 ..< SAMPLES {
		p := Vec2{f32(x) + SAMPLE_OFFSETS[k].x, f32(y) + SAMPLE_OFFSETS[k].y}
		if inside(t, p, area) {
			covered += {k}
		}
	}
	return
}

// A pixel's colour into its covered samples that pass the depth test:
// blended over them (depth kept), or replacing them (depth written).
@(private = "file")
write_samples :: #force_inline proc "contextless" (s: ^Software, t: Tri, x, y: int, area: f32, covered: Sample_Mask, col: u32, blend: bool) {
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

// A ShaderMaterial's output as written (no conversion) to RGBA8,
// premultiplied when it is blended (as the GPU's SRC_ALPHA blend does).
@(private = "file")
pack_raw :: proc "contextless" (c: [4]f32, blend: bool) -> u32 {
	a := clamp(c.a, 0, 1)
	m: f32 = a if blend else 1
	out: u32
	for ch in 0 ..< 3 {
		out |= u32(math.round(clamp(c[ch], 0, 1) * m * 255)) << (8 * u32(ch))
	}
	return out | u32(math.round(a * 255)) << 24
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
