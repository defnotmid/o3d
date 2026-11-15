package o3d

// The frame's passes after its light on the CPU: shaders/post.frag's
// programs (shading.odin's POST_PASSES), each run by oksl over its target
// in bands of rows across a pool's threads, as the GPU draws them
// (o3d_sdlgpu's post_draw): the output, bloom's chain, the depth of
// field's phases and the motion blur's.

import "core:sync"
import "core:thread"

import "deps:oksl"

// What a post pass reads under a name (oksl.channel_image: 4, 2 or 1
// floats a texel, `width x height`), bound to its channel: bloom read
// bilinearly, the rest nearest, as the GPU's samplers.
Post_Images :: [Pass_Map]oksl.Channel_Image

// Rows a band of a post pass runs (oksl's batches are 8 x 8).
@(private = "file")
POST_ROWS :: 32

// A thread's working memory for post passes: oksl's registers, and the
// light of the rows it packs into a frame's pixels (STAGE_ROWS at a time).
Post_Worker :: struct {
	vm:   oksl.Scratch,
	band: [dynamic][4]f32,
}

post_workers_destroy :: proc(workers: ^[dynamic]Post_Worker) {
	for &w in workers {
		oksl.scratch_destroy(&w.vm)
		delete(w.band)
	}
	delete(workers^)
}

@(private = "file")
Post_Job :: struct {
	x:             ^Pass_Program,
	block:         [size_of(Post_Block)]u8,
	images:        [oksl.SAMPLER_COUNT]oksl.Channel_Image,
	targets:       [][][4]f32,
	pixels:        []u8,
	width, height: int,
	workers:       []Post_Worker,
}

// Post pass `pass` over `width x height` into `targets` (one per output
// location, `width x height` texels each), reading `maps` (one not given
// reads 0), its block from `block`. `workers` grows to one per thread.
post_run :: proc(p: ^Programs, workers: ^[dynamic]Post_Worker, pass: Pass, block: ^Post_Block, maps: Post_Images, targets: [][][4]f32, width, height: int, pool: ^thread.Pool) {
	post_bands(p, workers, pass, block, maps, {targets = targets, width = width, height = height}, pool)
}

// Post pass `pass`'s light (sRGB-encoded, premultiplied) into `pixels`
// (RGBA8, `width x height`), as post_run reads; the light stays in its
// thread's Post_Worker, STAGE_ROWS at a time, until it is packed.
post_run_pixels :: proc(p: ^Programs, workers: ^[dynamic]Post_Worker, pass: Pass, block: ^Post_Block, maps: Post_Images, pixels: []u8, width, height: int, pool: ^thread.Pool) {
	post_bands(p, workers, pass, block, maps, {pixels = pixels[:width * height * 4], width = width, height = height}, pool)
}

@(private = "file")
post_bands :: proc(p: ^Programs, workers: ^[dynamic]Post_Worker, pass: Pass, block: ^Post_Block, maps: Post_Images, job: Post_Job, pool: ^thread.Pool) {
	job := job
	x := pass_get(p, pass)
	if x == nil || job.width <= 0 || job.height <= 0 {
		return
	}
	job.x = x
	block_fill(&x.block, {block, nil}, job.block[:])
	for k, c in x.maps {
		job.images[c] = maps[k]
		job.images[c].filter = .Linear if k == .Bloom else .Nearest
	}
	threads := 1
	if pool != nil {
		sync.guard(&pool.mutex)
		threads += len(pool.threads)
	}
	if len(workers) < threads {
		resize(workers, threads)
	}
	job.workers = workers[:]
	parallel(pool, (job.height + POST_ROWS - 1) / POST_ROWS, &job, proc(data: rawptr, band, worker: int) {
		j := (^Post_Job)(data)
		w := &j.workers[worker]
		y := band * POST_ROWS
		h := min(POST_ROWS, j.height - y)
		program, block := &j.x.pipeline.fragment.program, j.block[:j.x.block.size]
		if j.pixels == nil {
			oksl.run_targets(program, block, j.width, j.height, j.targets, &w.vm, {y = y, w = j.width, h = h, channel_images = j.images[:]})
			return
		}
		table_grow_non_zero(&w.band, j.width * STAGE_ROWS)
		for r0 := y; r0 < y + h; r0 += STAGE_ROWS {
			n := j.width * min(STAGE_ROWS, y + h - r0)
			light := [1][][4]f32{w.band[:n]}
			oksl.run_targets(program, block, j.width, j.height, light[:], &w.vm, {y = r0, w = j.width, h = n / j.width, channel_images = j.images[:], region_targets = true})
			post_pixels(w.band[:n], j.pixels[r0 * j.width * 4:][:n * 4])
		}
	})
}

// Pixels (RGBA8, premultiplied) from a post pass's light (sRGB-encoded,
// premultiplied), as the GPU's UNORM target rounds it.
@(private = "file")
post_pixels :: proc(light: [][4]f32, pixels: []u8) {
	for c, i in light {
		for k in 0 ..< 4 {
			pixels[i * 4 + k] = u8(clamp(c[k], 0, 1) * 255 + 0.5)
		}
	}
}
