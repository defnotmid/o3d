package o3d

import "core:thread"

import "deps:oksl"

// Bloom over a frame's light, before its tone mapping, as ovg's `.Bloom` filter
// over a 2D layer: the same numbers give the same glow. Each pixel's light above
// the threshold (per channel, so it works past white, and nothing pops as light
// crosses it), halved level by level, then a Gaussian of `radius` pixels on the
// level where that is a few texels, which the frame's light gets a share of
// where it is tone mapped, so no pass at the frame's size is added. The same
// chain runs on the GPU (the same post.frag passes), after the frame's
// motion blur.

// A scene's bloom; a strength or radius of 0 is none.
Bloom :: struct {
	// Light above this glows, in the scene's linear light.
	threshold: f32,
	// How much of the light above it is added, spread out.
	strength:  f32,
	// The Gaussian's sigma, pixels.
	radius:    f32,
}

BLOOM_MAX_LEVELS :: 8
// The level is the first where the Gaussian's sigma is under this many texels.
BLOOM_LEVEL_SIGMA :: 4
// The most sigma a level takes when a small frame has no smaller one.
BLOOM_MOST_SIGMA :: 32

// A frame's chain: its levels (the first at half the frame's size), the
// Gaussian's sigma on the last in its texels, the threshold, the share the
// frame gets.
Bloom_Chain :: struct {
	levels:    int,
	sigma:     f32,
	threshold: f32,
	gain:      f32,
}

// The chain for a `width x height` frame.
bloom_of :: proc "contextless" (b: Bloom, width, height: int) -> (c: Bloom_Chain, ok: bool) {
	if !(b.strength > 0) || !(b.radius > 0) {
		return
	}
	c.levels = 1
	c.sigma = b.radius / 2
	for c.levels < BLOOM_MAX_LEVELS && c.sigma >= BLOOM_LEVEL_SIGMA && (2 << uint(c.levels)) < min(width, height) {
		c.levels += 1
		c.sigma /= 2
	}
	c.sigma = min(c.sigma, BLOOM_MOST_SIGMA)
	c.threshold = max(b.threshold, 0)
	c.gain = b.strength
	return c, true
}

// Level `k`'s size for a `width x height` frame.
bloom_level_size :: #force_inline proc "contextless" (width, height, k: int) -> (w, h: int) {
	shift := uint(k + 1)
	round := (1 << shift) - 1
	return max((width + round) >> shift, 1), max((height + round) >> shift, 1)
}

// The chain's levels, and the Gaussian's first axis on the last, the caller's,
// kept warm across frames.
Bloom_Tables :: struct {
	levels: [BLOOM_MAX_LEVELS][dynamic][4]f32,
	across: [dynamic][4]f32,
}

bloom_tables_destroy :: proc(t: ^Bloom_Tables) {
	for &l in t.levels {
		delete(l)
	}
	delete(t.across)
	t^ = {}
}

// What the frame's light gets: the chain's last level, blurred (its glow)
// and the share of it added.
Bloom_Output :: struct {
	glow:          [][4]f32,
	width, height: int,
	gain:          f32,
}

// The chain over `frame` (`width x height` light): post.frag's bloom passes,
// as the GPU's `record_bloom`.
bloom_run :: proc(t: ^Bloom_Tables, c: Bloom_Chain, frame: [][4]f32, width, height: int, programs: ^Programs, scratches: ^[dynamic]oksl.Scratch, pool: ^thread.Pool) -> Bloom_Output {
	block := Post_Block {
		tone  = {0, 0, f32(width), f32(height)},
		frame = {f32(width), f32(height), 0, 0},
		bloom = {c.threshold, c.sigma, 0, 0},
	}
	w, h := bloom_level_size(width, height, 0)
	table_grow_non_zero(&t.levels[0], w * h)
	out := [1][][4]f32{t.levels[0][:w * h]}
	post_run(programs, scratches, .Bloom_Bright, &block, #partial {.Light = {f4 = frame[:width * height], width = width, height = height}}, out[:], w, h, pool)
	for k in 1 ..< c.levels {
		from := Post_Image{f4 = t.levels[k - 1][:w * h], width = w, height = h}
		w, h = bloom_level_size(width, height, k)
		table_grow_non_zero(&t.levels[k], w * h)
		block.bloom_size = {0, 0, f32(w), f32(h)}
		out = {t.levels[k][:w * h]}
		post_run(programs, scratches, .Bloom_Down, &block, #partial {.Bloom = from}, out[:], w, h, pool)
	}
	last := t.levels[c.levels - 1][:w * h]
	table_grow_non_zero(&t.across, w * h)
	block.bloom_size = {1, 0, f32(w), f32(h)}
	out = {t.across[:w * h]}
	post_run(programs, scratches, .Bloom_Blur, &block, #partial {.Bloom = {f4 = last, width = w, height = h}}, out[:], w, h, pool)
	block.bloom_size = {0, 1, f32(w), f32(h)}
	out = {last}
	post_run(programs, scratches, .Bloom_Blur, &block, #partial {.Bloom = {f4 = t.across[:w * h], width = w, height = h}}, out[:], w, h, pool)
	return {last, w, h, c.gain}
}
