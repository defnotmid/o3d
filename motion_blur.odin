package o3d

import "core:math"
import "core:thread"

import "deps:oksl"

// Motion blur over a frame's light, McGuire, Hennessy, Bukowski and Osman's
// reconstruction filter ("A Reconstruction Filter for Plausible Motion
// Blur", 2012): each pixel's velocity (its motion over the shutter's first
// half, in pixels) cut to `max_blur`, the longest per tile of `max_blur`
// pixels and the longest about each tile, then each pixel gathered across
// its tiles' longest, post.frag's MOTION_TAPS points from one end to the other, each
// weighted by whether it lies in front (its nearness) and whether its own
// motion or the pixel's reaches across them: a fast foreground smears over
// a still background and not the reverse. The same filter runs on the GPU
// (the same post.frag passes); o3d's own frames take it before their
// depth of field.

// The filter's parameters.
Motion_Blur :: struct {
	// The longest half streak, pixels: longer velocities are cut to it, and
	// the tiles are as wide.
	max_blur:    f32,
	// Nearnesses (larger is nearer) closer than this are one surface.
	depth_scale: f32,
}

// The filter's tables, the caller's, kept warm across frames: its tiles'
// rows' longest velocity, the tiles', the longest about each, and the
// programs and scratch it runs them with (post.odin).
Motion_Tables :: struct {
	rows, tiles, around: [dynamic][4]f32,
	programs:            Programs,
	scratches:           [dynamic]oksl.Scratch,
}

motion_tables_destroy :: proc(t: ^Motion_Tables) {
	delete(t.rows)
	delete(t.tiles)
	delete(t.around)
	programs_destroy(&t.programs)
	for &sc in t.scratches {
		oksl.scratch_destroy(&sc)
	}
	delete(t.scratches)
	t^ = {}
}

// o3d's frames give -ln(distance) as nearness, so points within 2% of each
// other's distance are one surface.
MOTION_DEPTH_SCALE :: 0.02
// The nearness o3d's frames give where nothing is drawn.
MOTION_FAR :: -1e4

// The filter's parameters for an o3d frame `height` pixels high through
// camera `c`: its longest half streak (`Camera.max_blur`) in pixels, and
// o3d's nearness (-ln(distance)) compared at MOTION_DEPTH_SCALE.
motion_params :: proc "contextless" (c: Camera, height: int) -> Motion_Blur {
	return {max_blur = (c.max_blur if c.max_blur > 0 else MOTION_MAX_BLUR) * f32(height), depth_scale = MOTION_DEPTH_SCALE}
}

// `color` (premultiplied light, rows from the top) blurred along `velocity`
// into `out` (not `color`), each `width x height`, `nearness` ordering what
// overlaps; `pool` shares the rows. post.frag's motion passes, as the GPU's
// `record_motion_blur`.
motion_blur :: proc(t: ^Motion_Tables, color: [][4]f32, velocity: [][2]f32, nearness: []f32, width, height: int, params: Motion_Blur, out: [][4]f32, pool: ^thread.Pool = nil) {
	tile := max(int(math.ceil(params.max_blur)), 1)
	tiles_w, tiles_h := (width + tile - 1) / tile, (height + tile - 1) / tile
	block := Post_Block {
		tone        = {0, 0, f32(width), f32(height)},
		frame       = {f32(width), f32(height), 0, 0},
		motion      = {max(params.max_blur, 0), params.depth_scale if params.depth_scale > 0 else MOTION_DEPTH_SCALE, f32(tile), 0},
		motion_size = {f32(width), f32(height), f32(tiles_w), f32(tiles_h)},
	}
	table_grow_non_zero(&t.rows, tiles_w * height)
	table_grow_non_zero(&t.tiles, tiles_w * tiles_h)
	table_grow_non_zero(&t.around, tiles_w * tiles_h)
	n := width * height
	vel := Post_Image{f2 = velocity[:n], width = width, height = height}
	rows := [1][][4]f32{t.rows[:]}
	post_run(&t.programs, &t.scratches, .Motion_Rows, &block, #partial {.Velocity = vel}, rows[:], tiles_w, height, pool)
	tiles := [1][][4]f32{t.tiles[:]}
	post_run(&t.programs, &t.scratches, .Motion_Tiles, &block, #partial {.Motion = {f4 = t.rows[:], width = tiles_w, height = height}}, tiles[:], tiles_w, tiles_h, pool)
	around := [1][][4]f32{t.around[:]}
	post_run(&t.programs, &t.scratches, .Motion_Around, &block, #partial {.Motion = {f4 = t.tiles[:], width = tiles_w, height = tiles_h}}, around[:], tiles_w, tiles_h, pool)
	gathered := [1][][4]f32{out[:n]}
	maps := #partial Post_Images {
		.Light    = {f4 = color[:n], width = width, height = height},
		.Velocity = vel,
		.Nearness = {f1 = nearness[:n], width = width, height = height},
		.Motion   = {f4 = t.around[:], width = tiles_w, height = tiles_h},
	}
	post_run(&t.programs, &t.scratches, .Motion_Gather, &block, maps, gathered[:], width, height, pool)
}
