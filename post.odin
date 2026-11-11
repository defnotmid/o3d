package o3d

// The frame's passes after its light on the CPU: shaders/post.frag's
// programs (shading.odin's POST_PASSES), each run by oksl over its target
// in bands of rows across a pool's threads, as the GPU draws them
// (o3d_sdlgpu's post_draw): the output, bloom's chain, the depth of
// field's phases and the motion blur's.

import "core:sync"
import "core:thread"

import "deps:oksl"

// What a post pass reads under a name: `width x height` texels of four,
// two or one floats.
Post_Image :: struct {
	f4:            [][4]f32,
	f2:            [][2]f32,
	f1:            []f32,
	width, height: int,
}

Post_Images :: [Pass_Map]Post_Image

// Rows a band of a post pass runs (oksl's batches are 8 x 8).
@(private = "file")
POST_ROWS :: 32

@(private = "file")
Post_Job :: struct {
	x:             ^Pass_Program,
	block:         [size_of(Post_Block)]u8,
	maps:          Post_Images,
	sizes:         [oksl.SAMPLER_COUNT][3]i32,
	targets:       [][][4]f32,
	width, height: int,
	scratches:     []oksl.Scratch,
}

// Post pass `pass` over `width x height` into `targets` (one per output
// location, `width x height` texels each), reading `maps` (bloom
// bilinearly, the rest nearest, as the GPU's samplers; one not given reads
// 0), its block from `block`. `scratches` grows to one per worker.
post_run :: proc(p: ^Programs, scratches: ^[dynamic]oksl.Scratch, pass: Pass, block: ^Post_Block, maps: Post_Images, targets: [][][4]f32, width, height: int, pool: ^thread.Pool) {
	x := pass_get(p, pass)
	if x == nil || width <= 0 || height <= 0 {
		return
	}
	job := Post_Job{x = x, maps = maps, targets = targets, width = width, height = height}
	block_fill(&x.block, {block, nil}, job.block[:])
	for k, c in x.maps {
		img := maps[k]
		job.sizes[c] = {i32(img.width), i32(img.height), 1}
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
		oksl.run_targets(&j.x.pipeline.fragment.program, j.block[:j.x.block.size], j.width, j.height, j.targets, &j.scratches[worker], {y = y, w = j.width, h = min(POST_ROWS, j.height - y), sampler = sample_post_map, user = j, channel_sizes = j.sizes[:]})
	})
}

@(private = "file")
sample_post_map :: proc(user: rawptr, channel: int, uv: [2]f32, lod: f32, layer: int) -> [4]f32 {
	j := (^Post_Job)(user)
	k := j.x.maps[channel]
	img := j.maps[k]
	w, h := img.width, img.height
	if w <= 0 || h <= 0 {
		return 0
	}
	at :: #force_inline proc "contextless" (img: Post_Image, i: int) -> [4]f32 {
		switch {
		case img.f4 != nil:
			return img.f4[i]
		case img.f2 != nil:
			return {img.f2[i].x, img.f2[i].y, 0, 1}
		case img.f1 != nil:
			return {img.f1[i], 0, 0, 1}
		}
		return 0
	}
	if k != .Bloom {
		x := clamp(int(uv.x * f32(w)), 0, w - 1)
		y := clamp(int(uv.y * f32(h)), 0, h - 1)
		return at(img, y * w + x)
	}
	// Bilinear, clamped at the edges (centres at +0.5).
	s := uv.x * f32(w) - 0.5
	t := uv.y * f32(h) - 0.5
	x0, y0 := int(ffloor(s)), int(ffloor(t))
	fx, fy := s - f32(x0), t - f32(y0)
	x1, y1 := clamp(x0 + 1, 0, w - 1), clamp(y0 + 1, 0, h - 1)
	x0, y0 = clamp(x0, 0, w - 1), clamp(y0, 0, h - 1)
	top := at(img, y0 * w + x0) * (1 - fx) + at(img, y0 * w + x1) * fx
	bottom := at(img, y1 * w + x0) * (1 - fx) + at(img, y1 * w + x1) * fx
	return top * (1 - fy) + bottom * fy
}

@(private = "file")
ffloor :: #force_inline proc "contextless" (v: f32) -> f32 {
	i := f32(int(v))
	return i - 1 if i > v else i
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
