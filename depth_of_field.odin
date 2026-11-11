package o3d

import "core:math"
import "core:thread"

// Depth of field on the software renderer (`focus_of`), post.frag's focus
// passes as the GPU runs them: once the frame's pass has resolved its light
// into `Software.frame` (and blurred it along its motion), the largest blur
// per tile and near it from the frame's depth, a half-size copy, then the
// gather, written as the frame's pixels.

// The tables: the tiles' rows' largest blur, the tiles', the largest near
// each, the half-size copy's light and keys.
@(private)
Focus_Tables :: struct {
	rows, tiles, near: [dynamic][4]f32,
	half, half_keys:   [dynamic][4]f32,
}

@(private)
focus_tables_destroy :: proc(t: ^Focus_Tables) {
	delete(t.rows)
	delete(t.tiles)
	delete(t.near)
	delete(t.half)
	delete(t.half_keys)
}

// The frame's pixels from its light (`frame`: `s.frame`, or that blurred
// along its motion) and depth (`s.depth_map`), blurred by `f`, the bloom
// added, tone mapped and encoded: post.frag's focus passes, as the GPU's.
@(private)
depth_of_field :: proc(s: ^Software, frame: [][4]f32, f: Focus, camera: Camera, tone_mapping: Tone_Mapping, tone_exposure: f32, bloom: Bloom_Output, pool: ^thread.Pool) {
	w, h := s.width, s.height
	n := w * h
	tiles_w, tiles_h := (w + FOCUS_TILE - 1) / FOCUS_TILE, (h + FOCUS_TILE - 1) / FOCUS_TILE
	half_w, half_h := (w + 1) / 2, (h + 1) / 2
	near := camera.near if camera.near > 0 else 0.1
	far := camera.far if camera.far > near else near * 1e4
	block := Post_Block {
		tone   = {f32(tone_mapping), tone_exposure if tone_exposure > 0 else 1, f32(w), f32(h)},
		lens   = {f.focus, f.blur, f.max_blur, math.ceil(f.max_blur / FOCUS_TILE)},
		camera = {near, far, 1 if camera.orthographic else 0, 0},
		frame  = {f32(w), f32(h), f32(tiles_w), f32(tiles_h)},
		bloom  = {0, 0, bloom.gain, 0},
	}
	t := &s.focus
	table_grow_non_zero(&t.rows, tiles_w * h)
	table_grow_non_zero(&t.tiles, tiles_w * tiles_h)
	table_grow_non_zero(&t.near, tiles_w * tiles_h)
	table_grow_non_zero(&t.half, half_w * half_h)
	table_grow_non_zero(&t.half_keys, half_w * half_h)
	table_grow_non_zero(&s.post_light, n)
	light := Post_Image{f4 = frame[:n], width = w, height = h}
	depth := Post_Image{f1 = s.depth_map[:n], width = w, height = h}
	p, sc := &s.programs, &s.post_scratch
	rows := [1][][4]f32{t.rows[:]}
	post_run(p, sc, .Focus_Rows, &block, #partial {.Depth = depth}, rows[:], tiles_w, h, pool)
	tiles := [1][][4]f32{t.tiles[:]}
	post_run(p, sc, .Focus_Tiles, &block, #partial {.Tiles = {f4 = t.rows[:], width = tiles_w, height = h}}, tiles[:], tiles_w, tiles_h, pool)
	near_tiles := [1][][4]f32{t.near[:]}
	post_run(p, sc, .Focus_Near, &block, #partial {.Tiles = {f4 = t.tiles[:], width = tiles_w, height = tiles_h}}, near_tiles[:], tiles_w, tiles_h, pool)
	halves := [2][][4]f32{t.half[:], t.half_keys[:]}
	post_run(p, sc, .Focus_Half, &block, #partial {.Light = light, .Depth = depth}, halves[:], half_w, half_h, pool)
	maps := #partial Post_Images {
		.Light     = light,
		.Bloom     = {f4 = bloom.glow, width = bloom.width, height = bloom.height},
		.Depth     = depth,
		.Tiles     = {f4 = t.near[:], width = tiles_w, height = tiles_h},
		.Half      = {f4 = t.half[:], width = half_w, height = half_h},
		.Half_Keys = {f4 = t.half_keys[:], width = half_w, height = half_h},
	}
	out := [1][][4]f32{s.post_light[:n]}
	post_run(p, sc, .Focus_Gather, &block, maps, out[:], w, h, pool)
	post_pixels(s.post_light[:n], s.pixels)
}
