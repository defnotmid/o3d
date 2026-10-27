package o3d

import "core:math"
import "core:thread"

// Depth of field on the software renderer (`Scene.depth_of_field`;
// shaders/post.slang's fs_focus_* do the same on the GPU): once the frame's
// pass has resolved its light into `Software.frame`, each pixel's key and
// blur from the frame's depth, the largest blur per tile and near it, a
// half-size copy, then each row gathered and written as the frame's pixels.

@(private)
Focus_Tables :: struct {
	keys, blurs:  [dynamic]f32, // per pixel
	tiles, near:  [dynamic]f32, // per tile: its largest blur, the largest reaching it
	half:         [dynamic][4]f32, // the light at half size (each a 2x2 block's mean)
	half_keys:    [dynamic]f32,
}

@(private)
focus_tables_destroy :: proc(t: ^Focus_Tables) {
	delete(t.keys)
	delete(t.blurs)
	delete(t.tiles)
	delete(t.near)
	delete(t.half)
	delete(t.half_keys)
}

@(private = "file")
Focus_Job :: struct {
	s:                Software_View,
	f:                Focus,
	camera:           Camera,
	width, height:    int,
	tiles_w, tiles_h: int,
	reach:            int, // tiles
	half_w, half_h:   int,
	tone_mapping:     Tone_Mapping,
	tone_exposure:    f32,
}

// What the phases read and write of the renderer.
@(private = "file")
Software_View :: struct {
	frame:  [][4]f32,
	depth:  []f32,
	pixels: []u8,
	t:      ^Focus_Tables,
}

// The frame's pixels from its light (`s.frame`) and depth (`s.depth_map`),
// blurred by `f`, tone mapped and encoded.
@(private)
depth_of_field :: proc(s: ^Software, f: Focus, camera: Camera, tone_mapping: Tone_Mapping, tone_exposure: f32, pool: ^thread.Pool) {
	w, h := s.width, s.height
	job := Focus_Job {
		s             = {s.frame[:w * h], s.depth_map[:w * h], s.pixels, &s.focus},
		f             = f,
		camera        = camera,
		width         = w,
		height        = h,
		tiles_w       = (w + FOCUS_TILE - 1) / FOCUS_TILE,
		tiles_h       = (h + FOCUS_TILE - 1) / FOCUS_TILE,
		reach         = int(math.ceil(f.max_blur / FOCUS_TILE)),
		half_w        = (w + 1) / 2,
		half_h        = (h + 1) / 2,
		tone_mapping  = tone_mapping,
		tone_exposure = tone_exposure,
	}
	t := &s.focus
	table_grow_non_zero(&t.keys, w * h)
	table_grow_non_zero(&t.blurs, w * h)
	table_grow_non_zero(&t.tiles, job.tiles_w * job.tiles_h)
	table_grow_non_zero(&t.near, job.tiles_w * job.tiles_h)
	table_grow_non_zero(&t.half, job.half_w * job.half_h)
	table_grow_non_zero(&t.half_keys, job.half_w * job.half_h)
	parallel(pool, job.tiles_h, &job, focus_tiles)
	parallel(pool, job.tiles_h, &job, focus_near)
	parallel(pool, job.half_h, &job, focus_half)
	parallel(pool, h, &job, focus_gather)
}

// A row of tiles: its pixels' keys and blurs, and each tile's largest.
@(private = "file")
focus_tiles :: proc(data: rawptr, ty, _: int) {
	job := (^Focus_Job)(data)
	t := job.s.t
	w := job.width
	row := t.tiles[ty * job.tiles_w:][:job.tiles_w]
	for &m in row {
		m = 0
	}
	for y in ty * FOCUS_TILE ..< min((ty + 1) * FOCUS_TILE, job.height) {
		for x in 0 ..< w {
			i := y * w + x
			key := focus_key(job.camera, job.s.depth[i])
			blur := focus_blur(job.f, key)
			t.keys[i], t.blurs[i] = key, blur
			row[x / FOCUS_TILE] = max(row[x / FOCUS_TILE], blur)
		}
	}
}

// A row of tiles: the largest blur of the tiles within reach.
@(private = "file")
focus_near :: proc(data: rawptr, ty, _: int) {
	job := (^Focus_Job)(data)
	t := job.s.t
	tw := job.tiles_w
	for tx in 0 ..< tw {
		m: f32
		for ny in max(ty - job.reach, 0) ..< min(ty + job.reach + 1, job.tiles_h) {
			for nx in max(tx - job.reach, 0) ..< min(tx + job.reach + 1, tw) {
				m = max(m, t.tiles[ny * tw + nx])
			}
		}
		t.near[ty * tw + tx] = m
	}
}

// A row of the half-size copy: each 2x2 block's light and key averaged
// (the last row and column repeated on odd sizes).
@(private = "file")
focus_half :: proc(data: rawptr, hy, _: int) {
	job := (^Focus_Job)(data)
	t := job.s.t
	w := job.width
	for hx in 0 ..< job.half_w {
		light: [4]f32
		key: f32
		for dy in 0 ..< 2 {
			for dx in 0 ..< 2 {
				i := min(hy * 2 + dy, job.height - 1) * w + min(hx * 2 + dx, w - 1)
				light += job.s.frame[i]
				key += t.keys[i]
			}
		}
		t.half[hy * job.half_w + hx] = light * 0.25
		t.half_keys[hy * job.half_w + hx] = key * 0.25
	}
}

// A row's pixels: each the mean of samples on a golden-angle spiral out to
// the largest blur near it, a sample counting where its own blur reaches
// back to the pixel (one behind it no more than twice the pixel's), then
// tone mapped and encoded.
@(private = "file")
focus_gather :: proc(data: rawptr, y, _: int) {
	job := (^Focus_Job)(data)
	t := job.s.t
	w := job.width
	for x in 0 ..< w {
		i := y * w + x
		reach := t.near[(y / FOCUS_TILE) * job.tiles_w + x / FOCUS_TILE]
		sum := job.s.frame[i]
		if reach >= 0.5 {
			centre_key, centre_blur := t.keys[i], t.blurs[i]
			count: f32 = 1
			radius: f32 = 0.8
			dir := Vec2{1, 0}
			for radius < reach {
				o := dir * radius
				light: [4]f32
				key, blur: f32
				if radius < FOCUS_INNER {
					sx := clamp(floor_int(f32(x) + o.x + 0.5), 0, w - 1)
					sy := clamp(floor_int(f32(y) + o.y + 0.5), 0, job.height - 1)
					j := sy * w + sx
					light, key, blur = job.s.frame[j], t.keys[j], t.blurs[j]
				} else {
					// The 2x2 block the sample point is in.
					sx := clamp(floor_int((f32(x) + 0.5 + o.x) * 0.5), 0, job.half_w - 1)
					sy := clamp(floor_int((f32(y) + 0.5 + o.y) * 0.5), 0, job.half_h - 1)
					light, key = t.half[sy * job.half_w + sx], t.half_keys[sy * job.half_w + sx]
					blur = focus_blur(job.f, key)
				}
				if key < centre_key {
					blur = min(blur, centre_blur * 2)
				}
				m := clamp(blur - radius + 0.5, 0, 1)
				sum += (sum / count) * (1 - m) + light * m
				count += 1
				// Past the inner radius the samples are 2x2 means: half the
				// density does.
				radius += (1.6 if radius < FOCUS_INNER else 3.2) / radius
				// Turned by the golden angle.
				dir = {dir.x * -0.73736887 - dir.y * 0.67549029, dir.x * 0.67549029 + dir.y * -0.73736887}
			}
			sum /= count
		}
		out := output(sum, job.tone_mapping, job.tone_exposure)
		copy(job.s.pixels[i * 4:][:4], out[:])
	}
}

// floor(v) as an int for a pixel coordinate: truncated, one less below
// zero (math.floor goes through modf, a tenth of the gather's time).
@(private = "file")
floor_int :: #force_inline proc "contextless" (v: f32) -> int {
	i := int(v)
	return i - 1 if f32(i) > v else i
}
