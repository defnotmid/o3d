package o3d

import "core:math"
import "core:thread"

// Motion blur over a frame's light, McGuire, Hennessy, Bukowski and Osman's
// reconstruction filter ("A Reconstruction Filter for Plausible Motion
// Blur", 2012): each pixel's velocity (its motion over the shutter's first
// half, in pixels) cut to `max_blur`, the longest per tile of `max_blur`
// pixels and the longest about each tile, then each pixel gathered across
// its tiles' longest, MOTION_TAPS points from one end to the other, each
// weighted by whether it lies in front (its nearness) and whether its own
// motion or the pixel's reaches across them: a fast foreground smears over
// a still background and not the reverse. The same filter runs on the GPU
// (shaders/post.slang's fs_motion_*); o3d's own frames take it before their
// depth of field.

// The filter's parameters.
Motion_Blur :: struct {
	// The longest half streak, pixels: longer velocities are cut to it, and
	// the tiles are as wide.
	max_blur:    f32,
	// Nearnesses (larger is nearer) closer than this are one surface.
	depth_scale: f32,
}

// The filter's tables, the caller's, kept warm across frames.
Motion_Tables :: struct {
	tiles, around: [dynamic][2]f32, // per tile: its longest velocity, the longest about it
}

motion_tables_destroy :: proc(t: ^Motion_Tables) {
	delete(t.tiles)
	delete(t.around)
	t^ = {}
}

MOTION_TAPS :: 15
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

@(private = "file")
Motion_Job :: struct {
	t:                ^Motion_Tables,
	color:            [][4]f32,
	velocity:         [][2]f32,
	nearness:         []f32,
	out:              [][4]f32,
	width, height:    int,
	tile:             int,
	tiles_w, tiles_h: int,
	max_blur, scale:  f32,
}

// `color` (premultiplied light, rows from the top) blurred along `velocity`
// into `out` (not `color`), each `width x height`, `nearness` ordering what
// overlaps; `pool` shares the rows.
motion_blur :: proc(t: ^Motion_Tables, color: [][4]f32, velocity: [][2]f32, nearness: []f32, width, height: int, params: Motion_Blur, out: [][4]f32, pool: ^thread.Pool = nil) {
	tile := max(int(math.ceil(params.max_blur)), 1)
	job := Motion_Job {
		t        = t,
		color    = color[:width * height],
		velocity = velocity[:width * height],
		nearness = nearness[:width * height],
		out      = out[:width * height],
		width    = width,
		height   = height,
		tile     = tile,
		tiles_w  = (width + tile - 1) / tile,
		tiles_h  = (height + tile - 1) / tile,
		max_blur = max(params.max_blur, 0),
		scale    = params.depth_scale if params.depth_scale > 0 else MOTION_DEPTH_SCALE,
	}
	table_grow_non_zero(&t.tiles, job.tiles_w * job.tiles_h)
	table_grow_non_zero(&t.around, job.tiles_w * job.tiles_h)
	parallel(pool, job.tiles_h, &job, motion_tiles)
	parallel(pool, job.tiles_h, &job, motion_around)
	parallel(pool, height, &job, motion_gather)
}

// A velocity cut to `max_blur` long.
@(private = "file")
motion_cut :: #force_inline proc "contextless" (v: [2]f32, max_blur: f32) -> [2]f32 {
	l := math.sqrt(v.x * v.x + v.y * v.y)
	return v * (max_blur / l) if l > max_blur else v
}

@(private = "file")
length2 :: #force_inline proc "contextless" (v: [2]f32) -> f32 {
	return v.x * v.x + v.y * v.y
}

// A row of tiles: each tile's longest velocity.
@(private = "file")
motion_tiles :: proc(data: rawptr, ty, _: int) {
	job := (^Motion_Job)(data)
	row := job.t.tiles[ty * job.tiles_w:][:job.tiles_w]
	for &v in row {
		v = 0
	}
	w := job.width
	for y in ty * job.tile ..< min((ty + 1) * job.tile, job.height) {
		for x in 0 ..< w {
			v := motion_cut(job.velocity[y * w + x], job.max_blur)
			m := &row[x / job.tile]
			if length2(v) > length2(m^) {
				m^ = v
			}
		}
	}
}

// A row of tiles: the longest velocity of each tile and the eight about it.
@(private = "file")
motion_around :: proc(data: rawptr, ty, _: int) {
	job := (^Motion_Job)(data)
	tw, th := job.tiles_w, job.tiles_h
	t := job.t
	for tx in 0 ..< tw {
		m: [2]f32
		for ny in max(ty - 1, 0) ..= min(ty + 1, th - 1) {
			for nx in max(tx - 1, 0) ..= min(tx + 1, tw - 1) {
				v := t.tiles[ny * tw + nx]
				if length2(v) > length2(m) {
					m = v
				}
			}
		}
		t.around[ty * tw + tx] = m
	}
}

// A row's pixels gathered.
@(private = "file")
motion_gather :: proc(data: rawptr, y, _: int) {
	job := (^Motion_Job)(data)
	w, h := job.width, job.height
	for x in 0 ..< w {
		i := y * w + x
		vn := job.t.around[(y / job.tile) * job.tiles_w + x / job.tile]
		if length2(vn) < 0.25 {
			job.out[i] = job.color[i]
			continue
		}
		lx := max(math.sqrt(length2(motion_cut(job.velocity[i], job.max_blur))), 0.5)
		ln := math.sqrt(length2(vn))
		nx := job.nearness[i]
		// What passes in front of the pixel, each tap its share of the
		// shutter there (a surface moving `l` covers the pixel for its
		// stretch of taps, `ln / l` of them each); and what it shows
		// otherwise, its own and what its surface or theirs carries across.
		cover, own: [4]f32
		covered, owned: f32 = 0, 1
		own = job.color[i]
		jitter := motion_jitter(x, y) - 0.5
		for k in 0 ..< MOTION_TAPS {
			t := -1 + 2 * (f32(k) + jitter + 1) / f32(MOTION_TAPS + 1)
			px := clamp(int(math.floor(f32(x) + 0.5 + vn.x * t)), 0, w - 1)
			py := clamp(int(math.floor(f32(y) + 0.5 + vn.y * t)), 0, h - 1)
			j := py * w + px
			d := math.sqrt(length2({f32(px - x), f32(py - y)}))
			ly := max(math.sqrt(length2(motion_cut(job.velocity[j], job.max_blur))), 0.5)
			ny := job.nearness[j]
			front := clamp(1 - (nx - ny) / job.scale, 0, 1) // it in front of the pixel, or level
			level := clamp(1 - (ny - nx) / job.scale, 0, 1) // it behind, or level
			c := front * (1 - level) * motion_reach(d, ly) * ln / (ly * MOTION_TAPS)
			cover += job.color[j] * c
			covered += c
			o := level * max(motion_reach(d, lx), motion_reach(d, ly))
			own += job.color[j] * o
			owned += o
		}
		share := min(covered, 1)
		front_light := cover / covered if covered > 0 else 0
		job.out[i] = front_light * share + own / owned * (1 - share)
	}
}

// Whether motion `l` long reaches `d` away (fading over a pixel): still
// (`l` 0.5), only the pixel itself.
@(private = "file")
motion_reach :: #force_inline proc "contextless" (d, l: f32) -> f32 {
	return clamp(l - d + 0.5, 0, 1)
}

// Interleaved gradient noise (Jimenez 2014) at a pixel, in [0, 1): where
// along the streak its taps start, the same on the GPU.
@(private = "file")
motion_jitter :: #force_inline proc "contextless" (x, y: int) -> f32 {
	f :: #force_inline proc "contextless" (v: f32) -> f32 {
		return v - math.floor(v)
	}
	return f(52.9829189 * f(0.06711056 * f32(x) + 0.00583715 * f32(y)))
}
