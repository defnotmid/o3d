package o3d

import "core:math"
import "core:slice"

import "deps:oksl"

// Phase 3 of the software renderer: each band clears its rows, lays the
// background into its samples (its pass program, as the GPU draws it
// first), rasterizes its triangles in drawing order in 2x2 quads, each
// draw's quads shaded by its program (a variant of shaders/o3d.frag, or a
// ShaderMaterial's), into its samples' light (half floats, premultiplied by
// coverage, blended there, as the GPU's frame), and resolves its pixels:
// the 4 samples averaged into the frame's light, which post.odin's passes
// then tone map and encode.

@(private)
draw_band :: proc(data: rawptr, band, wi: int) {
	job := (^Frame_Job)(data)
	s := job.s
	width := job.width
	y0 := band * BAND_ROWS
	y1 := min(y0 + BAND_ROWS, job.height)
	// The second part of a frame (after its snapshot) draws over the first.
	bg_filled := job.keep || job.bg_tex == 0
	if !job.keep {
		slice.fill(s.depth[y0 * width * SAMPLES:y1 * width * SAMPLES], 0)
		if job.runs != nil {
			slice.fill(job.runs[y0 * width * SAMPLES:y1 * width * SAMPLES], 0)
		}
		if job.bg_tex == 0 {
			slice.fill(s.light[y0 * width * SAMPLES:y1 * width * SAMPLES], job.clear)
		}
	}
	refs := s.band_tris[s.band_start[band]:s.band_start[band + 1]]
	for r := 0; r < len(refs); {
		t := s.chunks[refs[r].chunk].tris[refs[r].tri] // gathered from the columns once
		if !bg_filled && !replaces(blend_of(&s.materials[t.draw])) {
			fill_band_bg(job, s, &s.workers[wi], width, y0, y1)
			bg_filled = true
		}
		r += draw_quads(job, &s.workers[wi], refs[r:], y0, y1)
	}
	if !bg_filled {
		fill_band_bg(job, s, &s.workers[wi], width, y0, y1)
	}
	// The pixels: their samples' light averaged, kept (the snapshot's or
	// the reflection's level 0), or tone mapped and encoded as the frame's.
	if job.capture_into != nil {
		for i in y0 * width ..< y1 * width {
			job.capture_into[i] = to_halves(pixel_light(s, i))
		}
		if job.capture_runs != nil {
			// As the GPU resolves its runs and copies them into its second
			// layer's red and green, alpha 1.
			for i in y0 * width ..< y1 * width {
				sum: [2]f32
				for r in job.runs[i * SAMPLES:][:SAMPLES] {
					sum += {from_half(r.x), from_half(r.y)}
				}
				sum *= 1.0 / SAMPLES
				job.capture_runs[i] = {to_half(sum.x), to_half(sum.y), 0, to_half(1)}
			}
		}
		return
	}
	// The frame's light, for its post passes.
	for i in y0 * width ..< y1 * width {
		job.light_into[i] = pixel_light(s, i)
	}
}

// Fills uncovered samples in [y0, y1) with the equirectangular background
// (deferred until the first blended draw or band resolve so fully opaque
// pixels skip sampling the background).
@(private = "file")
fill_band_bg :: proc(job: ^Frame_Job, s: ^Software, w: ^Worker, width, y0, y1: int) {
	// Compiled by software_render, before the bands run.
	x := &s.programs.passes[.Background]
	if x.pipeline == nil {
		return
	}
	tex, _ := texture_get(job.textures, job.bg_tex)
	bb := Background_Block {
		inv_view_proj = job.inv_vp,
		frame         = {f32(width), f32(job.height), 0, 0},
		background    = {job.bg_intensity, job.bg_lod, 1 if tex.hdr else 0, 0},
	}
	block: [256]u8 = ---
	block_fill(&x.block, {&bb, nil}, block[:])
	// Its map repeating around and clamped at its poles (none reads 0).
	maps: [1]oksl.Channel_Image
	maps[0], _ = texture_image(s, job.textures, job.bg_tex, {.Repeat, .Clamp}, &w.levels[0])
	table_grow_non_zero(&w.rows[0], width * STAGE_ROWS)
	for r0 := y0; r0 < y1; r0 += STAGE_ROWS {
		r1 := min(r0 + STAGE_ROWS, y1)
		targets := [1][][4]f32{w.rows[0][:]}
		oksl.run_targets(&x.pipeline.fragment.program, block[:x.block.size], width, job.height, targets[:], &w.vm, {y = r0, w = width, h = r1 - r0, channel_images = maps[:], region_targets = true})
		for y in r0 ..< r1 {
			for x in 0 ..< width {
				base := (y * width + x) * SAMPLES
				px := to_halves(w.rows[0][(y - r0) * width + x])
				for k in 0 ..< SAMPLES {
					if s.depth[base + k] == 0 {
						s.light[base + k] = px
					}
				}
			}
		}
	}
}

// A pixel's light: its samples' averaged (premultiplied; alpha its
// coverage). Most pixels' four are one light (inside a triangle, or where
// nothing is drawn): read once, the same mean.
@(private = "file")
pixel_light :: #force_inline proc "contextless" (s: ^Software, i: int) -> [4]f32 {
	px := s.light[i * SAMPLES:][:SAMPLES]
	first := transmute(u64)px[0]
	if transmute(u64)px[1] == first && transmute(u64)px[2] == first && transmute(u64)px[3] == first {
		return from_halves(px[0])
	}
	sum: [4]f32
	for p in px {
		sum += from_halves(p)
	}
	return sum * (1.0 / SAMPLES)
}

// A depth pass's band: a shadow map's (the nearest caster's depth per
// texel) or the frame's (its depth, and with motion blur each pixel's
// velocity and nearness: where nothing is drawn, the far plane moving only
// as the camera did). One sample a pixel, at its centre.
draw_depth_band :: proc(data: rawptr, band, wi: int) {
	job := (^Frame_Job)(data)
	s := job.s
	smap := job.depth_target
	size := job.width
	y0 := band * BAND_ROWS
	y1 := min(y0 + BAND_ROWS, job.height)
	slice.fill(smap[y0 * size:y1 * size], 0)
	if x := &s.programs.passes[.Velocity_Far]; x.pipeline != nil && job.velocity != nil {
		far := Far_Block{opened_far = job.opened_far, frame = {f32(size), f32(job.height), 0, 0}}
		block: [256]u8 = ---
		block_fill(&x.block, {&far, nil}, block[:])
		w := &s.workers[wi]
		table_grow_non_zero(&w.rows[0], size * STAGE_ROWS)
		table_grow_non_zero(&w.rows[1], size * STAGE_ROWS)
		for r0 := y0; r0 < y1; r0 += STAGE_ROWS {
			n := size * (min(r0 + STAGE_ROWS, y1) - r0)
			targets := [3][][4]f32{nil, w.rows[0][:n], w.rows[1][:n]}
			oksl.run_targets(&x.pipeline.fragment.program, block[:x.block.size], size, job.height, targets[:], &w.vm, {y = r0, w = size, h = n / size, region_targets = true})
			for i in 0 ..< n {
				job.velocity[r0 * size + i] = w.rows[0][i].xy
				job.nearness[r0 * size + i] = w.rows[1][i].x
			}
		}
	}
	refs := s.band_tris[s.band_start[band]:s.band_start[band + 1]]
	for r := 0; r < len(refs); {
		r += draw_depth_quads(job, &s.workers[wi], refs[r:], y0, y1)
	}
}

// A run of one draw's triangles in a depth pass, in quads (the lanes whose
// centres are outside are helpers), through its depth fragment program.
@(private = "file")
draw_depth_quads :: proc(job: ^Frame_Job, w: ^Worker, refs: []Tri_Ref, y0, y1: int) -> int {
	s := job.s
	width := job.width
	q := &w.quads
	draw := -1
	taken := 0
	for ref in refs {
		t := s.chunks[ref.chunk].tris[ref.tri]
		if draw >= 0 && t.draw != draw {
			break
		}
		draw = t.draw
		taken += 1
		area := edge(t.p[0].xy, t.p[1].xy, t.p[2].xy)
		if area == 0 {
			continue
		}
		span := tri_span(t, width)
		for qy := max(t.min_y, y0) &~ 1; qy < min(t.max_y, y1); qy += 2 {
			lo, hi := row_span(span, f32(qy), f32(qy + 2), 2)
			for qx := lo &~ 1; qx < hi; qx += 2 {
				helpers: oksl.Quad_Lanes
				for l in 0 ..< 4 {
					x, y := qx + l & 1, qy + l >> 1
					if x >= width || y < y0 || y >= y1 || !inside(t, {f32(x) + 0.5, f32(y) + 0.5}, area) {
						helpers += {l}
					}
				}
				if card(helpers) == 4 {
					continue
				}
				append(&q.origin, [2]i32{i32(qx), i32(qy)})
				append(&q.helpers, helpers)
				append(&q.facing, t.front)
				append(&q.covered, [4]Sample_Mask{})
				append(&q.tri, ref)
				if len(q.origin) >= QUAD_BATCH {
					flush_depth_quads(job, w, draw)
				}
			}
		}
	}
	flush_depth_quads(job, w, draw)
	return taken
}

// A batch of one draw's quads in a depth pass: o3d's depth variant, or a
// ShaderMaterial's depth stage in the frame's; where it is nearer, its
// depth, and with motion blur its velocity and nearness.
@(private = "file")
flush_depth_quads :: proc(job: ^Frame_Job, w: ^Worker, draw: int) {
	s := job.s
	size := job.width
	q := &w.quads
	nq := len(q.origin)
	if nq == 0 {
		return
	}
	m := &s.materials[draw]
	moving := job.velocity != nil
	for &t in q.targets {
		table_grow_non_zero(&t, nq * 4)
	}
	table_grow_non_zero(&q.shaded, nq)
	targets := [3][][4]f32{q.targets[0][:], q.targets[1][:] if moving else nil, q.targets[2][:] if moving else nil}
	quads := oksl.Fragment_Quads {
		count        = nq,
		origin       = q.origin[:],
		front_facing = q.facing[:],
		helpers      = q.helpers[:],
		depth        = q.depths[:],
	}
	if job.own_side && m.shader != 0 {
		sh := shader_get(job.shaders, m.shader)
		quad_values(s, q, sh.depth.vertex.varying_floats, &sh.depth_corners)
		quads.varyings, quads.depth = q.values[:], q.depths[:]
		oksl.run_fragment(&sh.depth.fragment.program, block(job.depth_blocks, draw, .Fragment), quads, targets[:], q.shaded[:], &w.vm)
	} else {
		x := &s.programs.list[job.programs[draw]]
		quad_values(s, q, x.pipeline.vertex.varying_floats, &x.corners)
		quads.varyings, quads.depth = q.values[:], q.depths[:]
		images := program_maps(job, x, m, &w.levels)
		fb: [size_of(Material_Block) + size_of(Frame_Block)]u8 = ---
		block_fill(&x.fragment, {&s.material_blocks[draw], job.frame_block}, fb[:])
		buffers: [2][]u32
		for b, i in x.fragment_buffers {
			if b == .Curve_Words {
				buffers[i] = job.geos.curve_words[:]
			}
		}
		oksl.run_fragment(&x.pipeline.fragment.program, fb[:x.fragment.size], quads, targets[:], q.shaded[:], &w.vm, {channel_images = images[:], buffers = buffers[:]})
	}
	smap := job.depth_target
	for qi in 0 ..< nq {
		for l in q.shaded[qi] {
			x, y := int(q.origin[qi].x) + l & 1, int(q.origin[qi].y) + l >> 1
			j := qi * 4 + l
			i := y * size + x
			z := q.depths[j]
			if z < 0 || z > 1 || z <= smap[i] {
				continue
			}
			smap[i] = z
			if moving {
				job.velocity[i] = q.targets[1][j].xy
				job.nearness[i] = q.targets[2][j].x
			}
		}
	}
	clear(&q.origin)
	clear(&q.helpers)
	clear(&q.facing)
	clear(&q.covered)
	clear(&q.tri)
}

// A batch's lanes' varyings (in the vertex stage's record of `nf` floats,
// at their offsets there: the floats each triangle's corners keep,
// perspective-correct, helpers' extrapolated, the flat ones its first
// corner's; the rest, which the fragment does not read, left as they are)
// and depths (gl_FragCoord.z).
@(private = "file")
quad_values :: proc(s: ^Software, q: ^Quads, nf: int, keep: ^Corner_Floats) {
	nq := len(q.origin)
	table_grow_non_zero(&q.values, nf * nq * 4)
	table_grow_non_zero(&q.depths, nq * 4)
	for qi in 0 ..< nq {
		ref := q.tri[qi]
		t := s.chunks[ref.chunk].tris[ref.tri]
		area := edge(t.p[0].xy, t.p[1].xy, t.p[2].xy)
		dz := depth_plane(t, area)
		n := keep.count
		vs := s.chunks[ref.chunk].values[t.values:][:3 * n]
		for l in 0 ..< 4 {
			c := Vec2{f32(int(q.origin[qi].x) + l & 1) + 0.5, f32(int(q.origin[qi].y) + l >> 1) + 0.5}
			b0 := edge(t.p[1].xy, t.p[2].xy, c) / area
			b1 := edge(t.p[2].xy, t.p[0].xy, c) / area
			b2 := 1 - b0 - b1
			iw := max(b0 * t.inv_w[0] + b1 * t.inv_w[1] + b2 * t.inv_w[2], 1e-9)
			j := qi * 4 + l
			for f, k in keep.from[:n] {
				q.values[int(f) * nq * 4 + j] = (vs[k] * b0 + vs[n + k] * b1 + vs[2 * n + k] * b2) / iw
			}
			for k in keep.flat {
				q.values[int(keep.from[k]) * nq * 4 + j] = vs[k] // kept as is: the first corner's
			}
			q.depths[j] = dz.z0 + dz.dx * (c.x - dz.x0) + dz.dy * (c.y - dz.y0)
		}
	}
}


// Coverage and helper lanes for the 2x2 quad at even pixel (qx, qy).
@(private = "file")
quad_coverage :: #force_inline proc "contextless" (t: Tri, qx, qy, width, y1: int, area: f32) -> (lanes: [4]Sample_Mask, helpers: oksl.Quad_Lanes) {
	for l in 0 ..< 4 {
		x, y := qx + l & 1, qy + l >> 1
		if x < width && y < y1 {
			lanes[l] = coverage(t, x, y, area)
		}
		if lanes[l] == {} {
			helpers += {l}
		}
	}
	return
}

// A run of one ShaderMaterial draw's triangles from `refs` (at least the
// first): their quads in the band shaded by the fragment program in one
// batch, then written in triangle order. Returns how many it took.
@(private = "file")
draw_quads :: #force_no_inline proc(job: ^Frame_Job, w: ^Worker, refs: []Tri_Ref, y0, y1: int) -> int {
	s := job.s
	width := job.width
	q := &w.quads
	draw := -1
	taken := 0
	for ref in refs {
		t := s.chunks[ref.chunk].tris[ref.tri]
		if draw >= 0 && t.draw != draw {
			break
		}
		draw = t.draw
		taken += 1
		area := edge(t.p[0].xy, t.p[1].xy, t.p[2].xy)
		if area == 0 {
			continue
		}
		// Quads at even pixels: a band (BAND_ROWS even) holds whole quads.
		span := tri_span(t, width)
		for qy := max(t.min_y, y0) &~ 1; qy < min(t.max_y, y1); qy += 2 {
			lo, hi := row_span(span, f32(qy), f32(qy + 2), 2)
			for qx := lo &~ 1; qx < hi; qx += 2 {
				lanes, helpers := quad_coverage(t, qx, qy, width, y1, area)
				if card(helpers) == 4 {
					continue
				}
				append(&q.origin, [2]i32{i32(qx), i32(qy)})
				append(&q.helpers, helpers)
				append(&q.facing, t.front)
				append(&q.covered, lanes)
				append(&q.tri, ref)
				if len(q.origin) >= QUAD_BATCH {
					flush_quads(job, w, draw)
				}
			}
		}
	}
	flush_quads(job, w, draw)
	return taken
}

// A batch of one draw's quads shaded: a ShaderMaterial's fragment program
// (its colour straight, premultiplied here as the GPU's SRC_ALPHA blend
// does) or o3d's variant's (light premultiplied by its coverage, and the
// samples an exact shape covers).
@(private = "file")
flush_quads :: proc(job: ^Frame_Job, w: ^Worker, draw: int) {
	s := job.s
	width := job.width
	q := &w.quads
	nq := len(q.origin)
	if nq == 0 {
		return
	}
	m := &s.materials[draw]
	straight := m.shader != 0
	sh: ^Shader
	x: ^Program
	program: ^oksl.Program
	nf: int
	keep: ^Corner_Floats
	if straight {
		sh = shader_get(job.shaders, m.shader)
		program, nf, keep = &sh.pipeline.fragment.program, sh.varying_floats, &sh.corners
	} else {
		x = &s.programs.list[job.programs[draw]]
		program, nf, keep = &x.pipeline.fragment.program, x.pipeline.vertex.varying_floats, &x.corners
	}
	quad_values(s, q, nf, keep)
	for &t in q.targets[:2] {
		table_grow_non_zero(&t, nq * 4)
	}
	table_grow_non_zero(&q.masks, nq * 4)
	table_grow_non_zero(&q.shaded, nq)
	quads := oksl.Fragment_Quads {
		count        = nq,
		origin       = q.origin[:],
		front_facing = q.facing[:],
		helpers      = q.helpers[:],
		varyings     = q.values[:],
		depth        = q.depths[:],
	}
	if straight {
		images := shader_maps(job, sh, m, &w.levels)
		targets := [1][][4]f32{q.targets[0][:]}
		oksl.run_fragment(program, block(job.blocks, draw, .Fragment), quads, targets[:], q.shaded[:], &w.vm, {channel_images = images[:]}, q.masks[:])
	} else {
		images := program_maps(job, x, m, &w.levels)
		fb: [size_of(Material_Block) + size_of(Frame_Block)]u8 = ---
		block_fill(&x.fragment, {&s.material_blocks[draw], job.frame_block}, fb[:])
		buffers: [2][]u32
		for b, i in x.fragment_buffers {
			#partial switch b {
			case .Sprite_Rows:
				buffers[i] = slice.reinterpret([]u32, s.expanded.sprite_rows[:])
			case .Curve_Words:
				buffers[i] = job.geos.curve_words[:]
			}
		}
		// The light, and in the mirror's pass the run (o3d.frag's o_run).
		targets := [2][][4]f32{q.targets[0][:], q.targets[1][:] if job.runs != nil else nil}
		oksl.run_fragment(program, fb[:x.fragment.size], quads, targets[:], q.shaded[:], &w.vm, {channel_images = images[:], buffers = buffers[:]}, q.masks[:])
	}
	blend := blend_of(m)
	// The mirror's runs: o3d's draws but particles (as the GPU's targets).
	runs := job.runs != nil && !straight && x.key.entry != .Sprite
	for qi in 0 ..< nq {
		ref := q.tri[qi]
		t := s.chunks[ref.chunk].tris[ref.tri]
		dz := depth_plane(t, edge(t.p[0].xy, t.p[1].xy, t.p[2].xy))
		for l in q.shaded[qi] {
			x, y := int(q.origin[qi].x) + l & 1, int(q.origin[qi].y) + l >> 1
			j := qi * 4 + l
			covered := q.covered[qi][l] & transmute(Sample_Mask)u8(q.masks[j] & 0xF)
			c := q.targets[0][j]
			light := c
			if straight {
				a := clamp(c.a, 0, 1)
				k: f32 = 1 if replaces(blend) else a
				light = {c.r * k, c.g * k, c.b * k, a}
			}
			if runs {
				write_samples(s, width, dz, x, y, covered, light, blend, job.runs, q.targets[1][j].xy, true)
			} else {
				write_samples(s, width, dz, x, y, covered, light, blend, nil, {}, false)
			}
		}
	}
	clear(&q.origin)
	clear(&q.helpers)
	clear(&q.facing)
	clear(&q.covered)
	clear(&q.tri)
}

// What a material channel reads without a texture: white, as the GPU's
// dummy texture.
@(private = "file")
white_texel := [1][4]u8{{255, 255, 255, 255}}

@(private = "file")
white_image :: proc "contextless" () -> oksl.Channel_Image {
	return {format = .Rgba8, pixels = slice.to_bytes(white_texel[:]), width = 1, height = 1}
}

// Texture `id` as oksl samples it, as the GPU's sampler: all its levels
// (`levels` holds those above 0 and outlives the run), linearly, each axis
// by `wrap` ({u, v}); an external one is the pixels the host bound to it,
// level 0 only. 8-bit texels stay sRGB-encoded (the shading decodes them).
// false: no texture, or an external one nothing is bound to.
@(private = "file")
texture_image :: proc "contextless" (s: ^Software, t: ^Textures, id: Texture_Id, wrap: [2]oksl.Channel_Wrap, levels: ^Image_Levels) -> (im: oksl.Channel_Image, ok: bool) {
	tex := texture_get(t, id) or_return
	im = {filter = .Linear, wrap = wrap}
	if tex.external {
		ext := bound_external(s, t, id) or_return
		im.width, im.height = ext.width, ext.height
		if ext.halves != nil {
			im.format, im.pixels = .Rgba16f, slice.to_bytes(ext.halves)
		} else {
			im.format, im.pixels = .Rgba8, slice.to_bytes(ext.pixels)
		}
		return im, true
	}
	w, h := int(tex.width), int(tex.height)
	im.width, im.height = w, h
	for l in 1 ..< int(tex.levels) {
		lw, lh := max(w >> uint(l), 1), max(h >> uint(l), 1)
		levels[l - 1] = oksl.channel_level(t.hdr_texels[tex.mips[l]:][:lw * lh], lw, lh) if tex.hdr else oksl.channel_level(t.texels[tex.mips[l]:][:lw * lh], lw, lh)
	}
	im.levels = levels[:tex.levels - 1]
	if tex.hdr {
		im.format, im.pixels = .Rgba16f, slice.to_bytes(t.hdr_texels[tex.first_texel:][:w * h])
	} else {
		im.format, im.pixels = .Rgba8, slice.to_bytes(t.texels[tex.first_texel:][:w * h])
	}
	return im, true
}

// What an o3d variant's channels read for a draw, as bound images: its
// material maps repeating (white without one), the environment repeating
// around and clamped at its poles, the shadow maps and the frame's depth
// nearest, the frame map (glass's frame behind, or the reflection) clamped;
// an unbound one (no environment, shadow, depth or frame map) reads 0. The
// textures' levels are in `levels` (the worker's).
@(private = "file")
program_maps :: proc(job: ^Frame_Job, x: ^Program, m: ^Shading_Material, levels: ^[oksl.SAMPLER_COUNT]Image_Levels) -> (images: [oksl.SAMPLER_COUNT]oksl.Channel_Image) {
	s := job.s
	for k, c in x.maps {
		id: Texture_Id
		#partial switch k {
		case .Color:
			id = m.color_map
		case .Normal:
			id = m.normal_map
		case .Roughness:
			id = m.roughness_map
		case .Emissive:
			id = m.emissive_map
		case .Metalness:
			id = m.metalness_map
		case .Ao:
			id = m.ao_map
		case .Alpha:
			id = m.alpha_map
		case .Env_Radiance, .Env_Irradiance:
			// No environment reads 0, as unbound.
			id = job.env.radiance if k == .Env_Radiance else job.env.irradiance
			images[c], _ = texture_image(s, job.textures, id, {.Repeat, .Clamp}, &levels[c])
			continue
		case .Layer0:
			id = m.layer_textures[0]
		case .Layer1:
			id = m.layer_textures[1]
		case .Shadow0, .Shadow1, .Shadow2, .Shadow3:
			i := int(k) - int(Program_Map.Shadow0)
			n := job.shadows[i].size
			if n > 0 && len(s.shadow_maps[i]) == n * n {
				images[c] = oksl.channel_image(s.shadow_maps[i], n, n)
			}
			continue
		case .Frame:
			// Glass: the frame before it; a reflective draw: the reflection
			// (none through the mirror).
			if !job.mirrored {
				images[c] = job.snapshot if .Transmission in m.flags else job.reflection
			}
			continue
		case .Scene_Depth:
			if n := job.width * job.height; n > 0 && len(s.depth_map) >= n {
				images[c] = oksl.channel_image(s.depth_map[:n], job.width, job.height)
			}
			continue
		case:
			continue
		}
		bound: bool
		images[c], bound = texture_image(s, job.textures, id, {.Repeat, .Repeat}, &levels[c])
		if !bound {
			// No map, or an external one nothing is bound to: as no map.
			images[c] = white_image()
		}
	}
	return
}

// A ShaderMaterial draw's channels as bound images: its textures repeating
// (white without one), and `shadowAt`'s shadow map nearest; their levels in
// `levels` (the worker's).
@(private = "file")
shader_maps :: proc(job: ^Frame_Job, sh: ^Shader, m: ^Shading_Material, levels: ^[oksl.SAMPLER_COUNT]Image_Levels) -> (images: [SHADER_MAPS]oksl.Channel_Image) {
	s := job.s
	for id, c in m.shader_maps {
		if c == int(sh.shadow_channel) {
			if n := job.shadows[0].size; job.shadow_count > 0 && len(s.shadow_maps[0]) == n * n {
				images[c] = oksl.channel_image(s.shadow_maps[0], n, n)
			}
			continue
		}
		bound: bool
		images[c], bound = texture_image(s, job.textures, id, {.Repeat, .Repeat}, &levels[c])
		if !bound {
			images[c] = white_image()
		}
	}
	return
}

@(private = "file")
Tri_Span :: struct {
	ax, ay, slope: [3]f32,
	kind:          [3]i8, // 1: left (dy > 0), -1: right (dy < 0), 0: horizontal
	x_lo, x_hi:    int,
}

@(private = "file")
tri_span :: #force_inline proc "contextless" (t: Tri, width: int) -> (s: Tri_Span) {
	s.x_lo = clamp(int(math.floor(min(t.p[0].x, t.p[1].x, t.p[2].x))), 0, width)
	s.x_hi = clamp(int(math.ceil(max(t.p[0].x, t.p[1].x, t.p[2].x))), 0, width)
	for k in 0 ..< 3 {
		a, b := t.p[k].xy, t.p[(k + 1) % 3].xy
		dx, dy := b.x - a.x, b.y - a.y
		s.ax[k], s.ay[k] = a.x, a.y
		if dy > 1e-5 {
			s.kind[k] = 1
			s.slope[k] = dx / dy
		} else if dy < -1e-5 {
			s.kind[k] = -1
			s.slope[k] = dx / dy
		}
	}
	return
}

// Pixel X range [lo, hi) a block of width `step` (1 or 2) can touch the
// triangle in, over y in [y_min, y_max].
@(private = "file")
row_span :: #force_inline proc "contextless" (s: Tri_Span, y_min, y_max: f32, step: f32 = 1) -> (lo, hi: int) {
	lo, hi = s.x_lo, s.x_hi
	for k in 0 ..< 3 {
		if s.kind[k] == 0 {
			continue
		}
		x0 := s.ax[k] + s.slope[k] * (y_min - s.ay[k])
		x1 := s.ax[k] + s.slope[k] * (y_max - s.ay[k])
		if s.kind[k] > 0 {
			lo = max(lo, int(math.floor(min(x0, x1) - step - 0.01)))
		} else {
			hi = min(hi, int(math.ceil(max(x0, x1) + 0.01)) + 1)
		}
	}
	return
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

// A pixel's light (premultiplied) into its covered samples that pass the
// depth test: replacing them (depth written), over them, or added to them
// (light added, alpha over; depth kept), in half floats as the GPU's
// target blends. In the mirror's pass (`RUNS`), its run past the mirror
// (o3d.frag's `mirror_run_of`) and coverage into `runs` too, the run premultiplied by
// the coverage, both blended as its alpha is (the GPU's second target).
@(private = "file")
write_samples :: #force_inline proc "contextless" (s: ^Software, width: int, dz: Depth_Plane, x, y: int, covered: Sample_Mask, light: [4]f32, blend: Blend, runs: [][2]f16, run: [2]f32, $RUNS: bool) {
	px := to_halves(light)
	for k in covered {
		z := dz.z0 + dz.dx * (f32(x) + SAMPLE_OFFSETS[k].x - dz.x0) + dz.dy * (f32(y) + SAMPLE_OFFSETS[k].y - dz.y0)
		i := (y * width + x) * SAMPLES + k
		if z < 0 || z > 1 || z <= s.depth[i] {
			continue
		}
		switch blend {
		case .Opaque, .Cut:
			s.light[i] = px
			s.depth[i] = z
		case .Over:
			s.light[i] = to_halves(light + from_halves(s.light[i]) * (1 - light.a))
		case .Add:
			dst := from_halves(s.light[i])
			s.light[i] = to_halves({light.r + dst.r, light.g + dst.g, light.b + dst.b, light.a + dst.a * (1 - light.a)})
		}
		when RUNS {
			// The run premultiplied by the coverage, and the coverage.
			r := run
			if !replaces(blend) {
				r += [2]f32{from_half(runs[i].x), from_half(runs[i].y)} * (1 - light.a)
			}
			runs[i] = {to_half(r.x), to_half(r.y)}
		}
	}
}

// A triangle's screen depth as a plane: at its first corner, and per pixel
// along x and y, found once a triangle.
@(private = "file")
Depth_Plane :: struct {
	z0, dx, dy: f32,
	x0, y0:     f32,
}

@(private = "file")
depth_plane :: #force_inline proc "contextless" (t: Tri, area: f32) -> Depth_Plane {
	a, b, c := t.p[0], t.p[1], t.p[2]
	inv := 1 / area
	return {
		z0 = a.z,
		dx = ((b.y - c.y) * (a.z - c.z) + (c.y - a.y) * (b.z - c.z)) * inv,
		dy = ((c.x - b.x) * (a.z - c.z) + (a.x - c.x) * (b.z - c.z)) * inv,
		x0 = a.x,
		y0 = a.y,
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

// The pixels the host bound to external texture `id` for the software
// renders (`software_set_external`).
@(private = "file")
bound_external :: #force_inline proc "contextless" (s: ^Software, textures: ^Textures, id: Texture_Id) -> (Software_External, bool) {
	if id == 0 || int(id) > len(s.externals) || (s.externals[id - 1].pixels == nil && s.externals[id - 1].halves == nil) {
		return {}, false
	}
	tex, ok := texture_get(textures, id)
	return s.externals[id - 1], ok && tex.external
}


// The last frame's depth as distance (`software_depth`'s through its
// camera, as the GPU's `record_depth_into` gives it) into `out` (`width x
// height` RGBA halves): rgb the distance along the camera's axis
// (`view_depth`) and a 1 where something was drawn; rgb `far` and a 0 where
// nothing was.
software_depth_image :: proc(s: ^Software, far: f32, out: []f16) {
	x := pass_get(&s.programs, .Depth_Image)
	n := s.width * s.height
	if x == nil || len(s.depth_map) < n || len(out) < n * 4 {
		return
	}
	c := s.camera
	near := c.near if c.near > 0 else 0.1
	cam_far := c.far if c.far > near else near * 1e4
	params := Depth_Image_Block{params = {near, cam_far, far, 1 if c.orthographic else 0}, frame = {f32(s.width), f32(s.height), 0, 0}}
	block: [256]u8 = ---
	block_fill(&x.block, {&params, nil}, block[:])
	depth := [1]oksl.Channel_Image{oksl.channel_image(s.depth_map[:n], s.width, s.height)}
	if len(s.workers) == 0 {
		table_grow(&s.workers, 1)
	}
	w := &s.workers[0]
	for y0 := 0; y0 < s.height; y0 += STAGE_ROWS {
		rows := min(STAGE_ROWS, s.height - y0)
		table_grow_non_zero(&w.rows[0], s.width * rows)
		targets := [1][][4]f32{w.rows[0][:s.width * rows]}
		oksl.run_targets(&x.pipeline.fragment.program, block[:x.block.size], s.width, s.height, targets[:], &w.vm, {y = y0, w = s.width, h = rows, channel_images = depth[:], region_targets = true})
		for v, i in w.rows[0][:s.width * rows] {
			o := (y0 * s.width + i) * 4
			out[o], out[o + 1], out[o + 2], out[o + 3] = f16(v.r), f16(v.g), f16(v.b), f16(v.a)
		}
	}
}
