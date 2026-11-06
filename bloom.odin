package o3d

import "core:math"
import "core:thread"

// Bloom over a frame's light, before its tone mapping, as ovg's `.Bloom` filter
// over a 2D layer: the same numbers give the same glow. Each pixel's light above
// the threshold (per channel, so it works past white, and nothing pops as light
// crosses it), halved level by level, then a Gaussian of `radius` pixels on the
// level where that is a few texels, which the frame's light gets a share of
// where it is tone mapped, so no pass at the frame's size is added. The same
// chain runs on the GPU (shaders/post.slang's fs_bloom_*), after the frame's
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
// A pixel counts for no more light than this: one highlight far past white, on
// lacquer, would flood the glow.
BLOOM_MOST_LIGHT :: 16

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

// What the frame's light gets, read by `bloom_light` for each of its pixels.
Bloom_Output :: struct {
	glow:          [][4]f32,
	width, height: int,
	gain:          f32,
}

@(private = "file")
Bloom_Job :: struct {
	chain:          Bloom_Chain,
	from, out:      [][4]f32,
	from_w, from_h: int,
	out_w, out_h:   int,
	dir:            [2]int,
}

// A level at (u, v), as a linear clamped sampler reads it.
@(private = "file")
level_at :: #force_inline proc "contextless" (level: [][4]f32, w, h: int, u, v: f32) -> [4]f32 {
	fx := u * f32(w) - 0.5
	fy := v * f32(h) - 0.5
	x0 := math.floor(fx)
	y0 := math.floor(fy)
	tx := fx - x0
	ty := fy - y0
	ax := clamp(int(x0), 0, w - 1)
	ay := clamp(int(y0), 0, h - 1)
	bx := clamp(int(x0) + 1, 0, w - 1)
	by := clamp(int(y0) + 1, 0, h - 1)
	top := level[ay * w + ax] * (1 - tx) + level[ay * w + bx] * tx
	bottom := level[by * w + ax] * (1 - tx) + level[by * w + bx] * tx
	return top * (1 - ty) + bottom * ty
}

// The light above the threshold of the four pixels under each of the first
// level's (post.slang's fs_bloom_bright).
@(private = "file")
bloom_bright :: proc(data: rawptr, y, _: int) {
	j := (^Bloom_Job)(data)
	for x in 0 ..< j.out_w {
		px, py := x * 2, y * 2
		qx, qy := min(px + 1, j.from_w - 1), min(py + 1, j.from_h - 1)
		sum: [3]f32
		for q in ([4]int{py * j.from_w + px, py * j.from_w + qx, qy * j.from_w + px, qy * j.from_w + qx}) {
			l := j.from[q].rgb
			l *= min(1, BLOOM_MOST_LIGHT / max(l.r, l.g, l.b, 1e-4))
			sum += [3]f32{max(l.r - j.chain.threshold, 0), max(l.g - j.chain.threshold, 0), max(l.b - j.chain.threshold, 0)}
		}
		sum *= 0.25
		j.out[y * j.out_w + x] = {sum.r, sum.g, sum.b, 1}
	}
}

// A level at half the size of the one before (fs_bloom_down).
@(private = "file")
bloom_down :: proc(data: rawptr, y, _: int) {
	j := (^Bloom_Job)(data)
	v := (f32(y) + 0.5) / f32(j.out_h)
	for x in 0 ..< j.out_w {
		j.out[y * j.out_w + x] = level_at(j.from, j.from_w, j.from_h, (f32(x) + 0.5) / f32(j.out_w), v)
	}
}

// The Gaussian along one axis; past the level's edge there is nothing, as in
// ovg's (fs_bloom_blur).
@(private = "file")
bloom_blur :: proc(data: rawptr, y, _: int) {
	j := (^Bloom_Job)(data)
	sigma := j.chain.sigma
	reach := int(math.ceil(3 * sigma))
	g1 := math.exp(-0.5 / (sigma * sigma))
	g2 := g1 * g1
	g0 := 1 / (math.sqrt(f32(2 * math.PI)) * sigma)
	for x in 0 ..< j.out_w {
		sum := j.from[y * j.out_w + x] * g0
		g, step := g0 * g1, g1 * g2
		for i in 1 ..= reach {
			for side in ([2]int{-i, i}) {
				sx, sy := x + side * j.dir.x, y + side * j.dir.y
				if sx >= 0 && sy >= 0 && sx < j.out_w && sy < j.out_h {
					sum += j.from[sy * j.out_w + sx] * g
				}
			}
			g *= step
			step *= g2
		}
		j.out[y * j.out_w + x] = sum
	}
}

// Runs the chain over a frame's light, into `t`.
@(private)
bloom_run :: proc(t: ^Bloom_Tables, c: Bloom_Chain, frame: [][4]f32, width, height: int, pool: ^thread.Pool) -> Bloom_Output {
	w, h := bloom_level_size(width, height, 0)
	table_grow_non_zero(&t.levels[0], w * h)
	job := Bloom_Job{chain = c, from = frame, out = t.levels[0][:w * h], from_w = width, from_h = height, out_w = w, out_h = h}
	parallel(pool, h, &job, bloom_bright)
	for k in 1 ..< c.levels {
		job.from, job.from_w, job.from_h = job.out, job.out_w, job.out_h
		job.out_w, job.out_h = bloom_level_size(width, height, k)
		table_grow_non_zero(&t.levels[k], job.out_w * job.out_h)
		job.out = t.levels[k][:job.out_w * job.out_h]
		parallel(pool, job.out_h, &job, bloom_down)
	}
	last := job.out
	n := job.out_w * job.out_h
	table_grow_non_zero(&t.across, n)
	job.from, job.from_w, job.from_h = last, job.out_w, job.out_h
	job.out, job.dir = t.across[:n], {1, 0}
	parallel(pool, job.out_h, &job, bloom_blur)
	job.from, job.out, job.dir = t.across[:n], last, {0, 1}
	parallel(pool, job.out_h, &job, bloom_blur)
	return {last, job.out_w, job.out_h, c.gain}
}

// A pixel's light with the bloom added (post.slang's `bloomed`); light over
// nothing gives the pixel as much coverage.
@(private)
bloom_light :: #force_inline proc "contextless" (light: [4]f32, b: Bloom_Output, x, y, width, height: int) -> [4]f32 {
	if b.gain <= 0 {
		return light
	}
	s := level_at(b.glow, b.width, b.height, (f32(x) + 0.5) / f32(width), (f32(y) + 0.5) / f32(height)) * b.gain
	return {light.r + s.r, light.g + s.g, light.b + s.b, max(light.a, min(max(s.r, s.g, s.b), 1))}
}
