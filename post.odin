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

@(private = "file")
Post_Job :: struct {
	x:             ^Pass_Program,
	block:         [size_of(Post_Block)]u8,
	images:        [oksl.SAMPLER_COUNT]oksl.Channel_Image,
	targets:       [][][4]f32,
	width, height: int,
	scratches:     []oksl.Scratch,
}

// Post pass `pass` over `width x height` into `targets` (one per output
// location, `width x height` texels each), reading `maps` (one not given
// reads 0), its block from `block`. `scratches` grows to one per worker.
post_run :: proc(p: ^Programs, scratches: ^[dynamic]oksl.Scratch, pass: Pass, block: ^Post_Block, maps: Post_Images, targets: [][][4]f32, width, height: int, pool: ^thread.Pool) {
	x := pass_get(p, pass)
	if x == nil || width <= 0 || height <= 0 {
		return
	}
	job := Post_Job{x = x, targets = targets, width = width, height = height}
	block_fill(&x.block, {block, nil}, job.block[:])
	for k, c in x.maps {
		job.images[c] = maps[k]
		job.images[c].filter = .Linear if k == .Bloom else .Nearest
	}
	workers := 1
	if pool != nil {
		sync.guard(&pool.mutex)
		workers += len(pool.threads)
	}
	if len(scratches) < workers {
		resize(scratches, workers)
	}
	job.scratches = scratches[:]
	parallel(pool, (height + POST_ROWS - 1) / POST_ROWS, &job, proc(data: rawptr, band, worker: int) {
		j := (^Post_Job)(data)
		y := band * POST_ROWS
		oksl.run_targets(&j.x.pipeline.fragment.program, j.block[:j.x.block.size], j.width, j.height, j.targets, &j.scratches[worker], {y = y, w = j.width, h = min(POST_ROWS, j.height - y), channel_images = j.images[:]})
	})
}

// The frame's pixels (RGBA8, premultiplied) from its post passes' last
// light (sRGB-encoded, premultiplied), as the GPU's UNORM target rounds it.
post_pixels :: proc(light: [][4]f32, pixels: []u8) {
	for c, i in light {
		for k in 0 ..< 4 {
			pixels[i * 4 + k] = u8(clamp(c[k], 0, 1) * 255 + 0.5)
		}
	}
}
