package o3d

import "core:math"
import "core:math/linalg"
import "core:slice"

import "deps:oksl"
import "deps:ovg/curves"

// Phase 3 of the software renderer: each band clears its rows, rasterizes its
// triangles in drawing order (per pixel when untextured, in 2x2 quads when
// textured, an exact shape's, or running an oksl ShaderMaterial) into its
// samples' light (half floats, premultiplied by coverage, blended there, as
// the GPU's frame), and resolves its pixels: the 4 samples averaged, tone
// mapped and sRGB-encoded once (or kept as light, for a snapshot or the
// mirror). The equirectangular background is laid into the samples first,
// as the GPU draws it first.

@(private)
draw_band :: proc(data: rawptr, band, wi: int) {
	job := (^Frame_Job)(data)
	s := job.s
	width := job.width
	y0 := band * BAND_ROWS
	y1 := min(y0 + BAND_ROWS, job.height)
	// The second part of a frame (after its snapshot) draws over the first.
	if !job.keep {
		slice.fill(s.depth[y0 * width * SAMPLES:y1 * width * SAMPLES], 0)
		if job.bg_tex == 0 {
			slice.fill(s.light[y0 * width * SAMPLES:y1 * width * SAMPLES], job.clear)
		} else {
			// The background first, as the GPU draws it: what is drawn then
			// blends over it or adds to it.
			inv_w := 2.0 / f32(width)
			inv_h := 2.0 / f32(job.height)
			for y in y0 ..< y1 {
				ndc_y := 1 - (f32(y) + 0.5) * inv_h
				// From the near plane (depth 1) to the far (0).
				p0_l := job.inv_vp * Vec4{0.5 * inv_w - 1, ndc_y, 1, 1}
				p1_l := job.inv_vp * Vec4{0.5 * inv_w - 1, ndc_y, 0, 1}
				p0_r := job.inv_vp * Vec4{1.5 * inv_w - 1, ndc_y, 1, 1}
				p1_r := job.inv_vp * Vec4{1.5 * inv_w - 1, ndc_y, 0, 1}
				ray0 := p1_l.xyz / p1_l.w - p0_l.xyz / p0_l.w
				ray_step := (p1_r.xyz / p1_r.w - p0_r.xyz / p0_r.w) - ray0
				for x in 0 ..< width {
					dir := linalg.normalize(ray0 + f32(x) * ray_step)
					c := sample_lod_linear(job.textures, job.bg_tex, equirect_uv(dir), job.bg_lod) * job.bg_intensity
					slice.fill(s.light[(y * width + x) * SAMPLES:][:SAMPLES], to_halves({c.r, c.g, c.b, 1}))
				}
			}
		}
	}
	refs := s.band_tris[s.band_start[band]:s.band_start[band + 1]]
	for r := 0; r < len(refs); {
		t := s.chunks[refs[r].chunk].tris[refs[r].tri] // gathered from the columns once
		if s.materials[t.draw].shader != 0 {
			r += draw_custom(job, &s.workers[wi], refs[r:], y0, y1)
			continue
		}
		draw_standard(job, refs[r], t, y0, y1)
		r += 1
	}
	// The pixels: their samples' light averaged, kept (the snapshot's or
	// the reflection's level 0), or tone mapped and encoded as the frame's.
	if job.capture_into != nil {
		for i in y0 * width ..< y1 * width {
			job.capture_into[i] = to_halves(pixel_light(s, i))
		}
		return
	}
	if job.light_into != nil {
		for i in y0 * width ..< y1 * width {
			job.light_into[i] = pixel_light(s, i)
		}
		return
	}
	for i in y0 * width ..< y1 * width {
		out := output(pixel_light(s, i), job.tone_mapping, job.tone_exposure)
		copy(s.pixels[i * 4:][:4], out[:])
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

// A pixel of the frame from its light (premultiplied): tone mapped, sRGB
// premultiplied RGBA8, as the GPU's output pass writes it. Light past what
// its coverage holds (added light over nothing) stays, as colour past the
// alpha (up to full), which a premultiplied over adds.
@(private)
output :: #force_inline proc "contextless" (light: [4]f32, mode: Tone_Mapping, exposure: f32) -> [4]u8 {
	a := clamp(light.a, 0, 1)
	if a <= 0 {
		return {}
	}
	c := tone_map(light.rgb / a, mode, exposure)
	k := clamp(max(c.r, c.g, c.b), 1, 1 / a)
	rgb := pack(c / k, a * k)
	return {u8(rgb), u8(rgb >> 8), u8(rgb >> 16), u8(math.round(a * 255))}
}

// What a transmissive point sees behind it (`shade`'s `behind`), its
// normal facing the eye as `shade` turns it; the layers' thickness becomes
// the refracted ray's world length.
@(private = "file")
behind_of :: #force_inline proc "contextless" (job: ^Frame_Job, world, n_in: Vec3, roughness: f32, ly: ^Shading_Layers, two_sided: bool) -> Vec4 {
	n := -n_in if two_sided && linalg.dot(n_in, job.eye - world) < 0 else n_in
	uv, lod, distance := refraction(world, n, job.eye, roughness, ly, job.view_proj, job.width)
	ly.thickness = distance
	if job.mirrored {
		// The mirror has no snapshot: the environment shows through.
		return {}
	}
	return frame_behind(job.snapshot, &job.snapshot_tex, uv, lod)
}

// What a reflective point at pixel (x, y) reads of the frame's reflection
// (none: nothing, the environment's), at its roughness and its coat's.
@(private = "file")
mirror_at :: #force_no_inline proc "contextless" (job: ^Frame_Job, m: ^Shading_Material, x, y: int, roughness, coat_roughness: f32) -> (mirror: Mirror_Light) {
	mirror.amount = m.reflection
	if job.reflection == nil {
		return
	}
	uv := Vec2{(f32(x) + 0.5) / f32(job.width), (f32(y) + 0.5) / f32(job.height)}
	mirror.spec = reflection_at(job.reflection, &job.reflection_tex, uv, reflection_lod(roughness, job.reflection_last))
	if .Clearcoat in m.flags {
		mirror.coat = reflection_at(job.reflection, &job.reflection_tex, uv, reflection_lod(coat_roughness, job.reflection_last))
	}
	return
}

// Whether a point is behind the mirror's plane (`job.clip`, zero in the
// other passes).
@(private = "file")
clipped :: #force_inline proc "contextless" (job: ^Frame_Job, p: Vec3) -> bool {
	return linalg.dot(job.clip.xyz, p) + job.clip.w < 0
}

// A shaded point's light as the frame keeps it: premultiplied by its
// material's alpha, coverage that alpha times transmission's (`through`:
// what shows behind glass is in `c` already, as the GPU writes it).
@(private = "file")
light_of :: #force_inline proc "contextless" (c: Vec3, alpha, through: f32) -> [4]f32 {
	a := clamp(alpha, 0, 1)
	return {c.r * a, c.g * a, c.b * a, a * through}
}

// A band of the shadow map: its rows cleared to the far plane (0), then each
// triangle's depth at the texel centres it covers, nearest (greatest) kept.
@(private)
draw_depth_band :: proc(data: rawptr, band, _: int) {
	job := (^Frame_Job)(data)
	s := job.s
	smap := job.depth_target
	size := job.width
	y0 := band * BAND_ROWS
	y1 := min(y0 + BAND_ROWS, job.height)
	slice.fill(smap[y0 * size:y1 * size], 0)
	for ref in s.band_tris[s.band_start[band]:s.band_start[band + 1]] {
		t := s.chunks[ref.chunk].tris[ref.tri]
		area := edge(t.p[0].xy, t.p[1].xy, t.p[2].xy)
		if area == 0 {
			continue
		}
		m := &s.materials[t.draw]
		masked, _ := shadow_mask(m)
		curved := .Curves in m.flags
		nf := shadow_floats(masked, curved)
		uv, rc: [3]Vec2 = ---, ---
		vs := s.chunks[ref.chunk].values[t.values:][:3 * nf]
		for k in 0 ..< 3 {
			if masked {
				uv[k] = (^Vec2)(&vs[k * nf])^
			}
			if curved {
				rc[k] = (^Vec2)(&vs[k * nf + nf - SHADOW_MASK_FLOATS])^
			}
		}
		words: []u32
		even_odd: bool
		if curved {
			words, even_odd = curve_entry_words(job.geos, t.curve)
		}
		x_lo := clamp(int(math.floor(min(t.p[0].x, t.p[1].x, t.p[2].x))), 0, size)
		x_hi := clamp(int(math.ceil(max(t.p[0].x, t.p[1].x, t.p[2].x))), 0, size)
		for y in max(t.min_y, y0) ..< min(t.max_y, y1) {
			for x in x_lo ..< x_hi {
				c := Vec2{f32(x) + 0.5, f32(y) + 0.5}
				if !inside(t, c, area) {
					continue
				}
				b0 := edge(t.p[1].xy, t.p[2].xy, c) / area
				b1 := edge(t.p[2].xy, t.p[0].xy, c) / area
				b2 := 1 - b0 - b1
				z := b0 * t.p[0].z + b1 * t.p[1].z + b2 * t.p[2].z
				i := y * size + x
				if z >= 0 && z <= 1 && z > smap[i] {
					iw := b0 * t.inv_w[0] + b1 * t.inv_w[1] + b2 * t.inv_w[2]
					if curved && !curves_inside(words, (rc[0] * b0 + rc[1] * b1 + rc[2] * b2) / iw, even_odd) {
						continue
					}
					if masked {
						raw := (uv[0] * b0 + uv[1] * b1 + uv[2] * b2) / iw
						at :: #force_inline proc "contextless" (xf: [2]Vec3, raw: Vec2) -> Vec2 {
							return {xf[0].x * raw.x + xf[0].y * raw.y + xf[0].z, xf[1].x * raw.x + xf[1].y * raw.y + xf[1].z}
						}
						a := m.opacity
						if .Color_Map in m.flags {
							a *= sample_map_top(job, m.color_map, at(m.uv_transforms[.Color], raw)).a
						}
						if .Alpha_Map in m.flags {
							a *= sample_map_top(job, m.alpha_map, at(m.uv_transforms[.Alpha], raw)).g
						}
						if a < m.alpha_cutoff {
							continue
						}
					}
					smap[i] = z
				}
			}
		}
	}
}

// Transformed UV and screen derivatives for `slot` (selecting `uv0` or `uv1`).
@(private = "file")
slot_uv :: #force_inline proc "contextless" (m: ^Shading_Material, slot: Texture_Slot, uv0, uv1, d0x, d0y, d1x, d1y: Vec2) -> (uv, ddx, ddy: Vec2) {
	raw, dx, dy := uv0, d0x, d0y
	if ((m.uv_sets >> u32(slot)) & 1) != 0 {
		raw, dx, dy = uv1, d1x, d1y
	}
	return transform_uv(m.uv_transforms[slot], raw, dx, dy)
}

// A uv and its derivatives through a slot's transform.
@(private = "file")
transform_uv :: #force_inline proc "contextless" (xf: [2]Vec3, raw, dx, dy: Vec2) -> (uv, ddx, ddy: Vec2) {
	uv = {
		xf[0].x * raw.x + xf[0].y * raw.y + xf[0].z,
		xf[1].x * raw.x + xf[1].y * raw.y + xf[1].z,
	}
	ddx = {
		xf[0].x * dx.x + xf[0].y * dx.y,
		xf[1].x * dx.x + xf[1].y * dx.y,
	}
	ddy = {
		xf[0].x * dy.x + xf[0].y * dy.y,
		xf[1].x * dy.x + xf[1].y * dy.y,
	}
	return
}

// A standard-material triangle's rows in the band: per pixel when untextured,
// or in 2x2 quads when textured so uv and world derivatives give the mip level
// and the normal map's tangent frame (and an exact shape's coordinate its
// pixel footprint).
@(private = "file")
draw_standard :: #force_no_inline proc(job: ^Frame_Job, ref: Tri_Ref, t: Tri, y0, y1: int) {
	t := t // a local the pixel loop keeps in registers (a parameter is a reference)
	s := job.s
	width := job.width
	m := &s.materials[t.draw]
	blend := .Blended in m.flags
	receive := .Receive_Shadow in m.flags && job.shadow_count > 0
	area := edge(t.p[0].xy, t.p[1].xy, t.p[2].xy)
	if area == 0 {
		return
	}
	if .Shadow_Only in m.flags {
		draw_shadow_only(job, ref, t, m, y0, y1, area)
		return
	}
	if m.flags & (TEXTURE_FLAGS + {.Curves, .Sprite, .Vertex_Colors}) != {} {
		draw_standard_textured(job, ref, t, m, y0, y1, area, blend, receive)
		return
	}
	if .Alpha_Mask in m.flags && m.opacity < m.alpha_cutoff {
		return
	}
	out_alpha := m.opacity if blend else 1
	dz := depth_plane(t, area)
	x_lo := clamp(int(math.floor(min(t.p[0].x, t.p[1].x, t.p[2].x))), 0, width)
	x_hi := clamp(int(math.ceil(max(t.p[0].x, t.p[1].x, t.p[2].x))), 0, width)
	// The mirror's pass clips; a reflective draw reads the reflection.
	clipping := job.clip != {}
	reflective := .Reflection in m.flags
	// World position and normal over w per corner.
	vs := s.chunks[ref.chunk].values[t.values:][:3 * UNTEX_FLOATS]
	world, normal: [3]Vec3
	for k in 0 ..< 3 {
		world[k] = (^Vec3)(&vs[k * UNTEX_FLOATS])^
		normal[k] = (^Vec3)(&vs[k * UNTEX_FLOATS + 3])^
	}
	flat_n: Vec3
	if .Flat_Shading in m.flags {
		w0 := world[0] / t.inv_w[0]
		flat_n = linalg.normalize(linalg.cross(world[1] / t.inv_w[1] - w0, world[2] / t.inv_w[2] - w0))
	}
	for y in max(t.min_y, y0) ..< min(t.max_y, y1) {
		for x in x_lo ..< x_hi {
			covered := coverage(t, x, y, area)
			if covered == {} {
				continue
			}
			// Shade once, at the pixel centre, perspective-correct.
			c := Vec2{f32(x) + 0.5, f32(y) + 0.5}
			b0 := edge(t.p[1].xy, t.p[2].xy, c) / area
			b1 := edge(t.p[2].xy, t.p[0].xy, c) / area
			b2 := 1 - b0 - b1
			iw := b0 * t.inv_w[0] + b1 * t.inv_w[1] + b2 * t.inv_w[2]
			p := (world[0] * b0 + world[1] * b1 + world[2] * b2) / iw
			if clipping && clipped(job, p) {
				continue
			}
			n := flat_n
			if .Flat_Shading not_in m.flags {
				n = linalg.normalize((normal[0] * b0 + normal[1] * b1 + normal[2] * b2) / iw)
			}
			shadows := [MAX_SHADOWS]f32{1, 1, 1, 1}
			if receive {
				for si in 0 ..< job.shadow_count {
					shadows[si] = shadow_factor(s.shadow_maps[si], &job.shadows[si], p, n)
				}
			}
			// The layers as they are, or (transmission) a copy taking the
			// ray's length.
			ly := &m.layers
			local: Shading_Layers = ---
			behind: Vec4
			if .Transmission in m.flags {
				local = m.layers
				ly = &local
				behind = behind_of(job, p, n, m.roughness, ly, .Two_Sided in m.flags)
			}
			mirror: Mirror_Light = ---
			mp: ^Mirror_Light
			if reflective {
				mirror = mirror_at(job, m, x, y, m.roughness, m.layers.clearcoat_roughness)
				mp = &mirror
			}
			a: f32 = 1
			lit: Vec3
			if m.flags & LAYER_FLAGS != {} {
				a, lit = shade_layered(surface(m), p, n, job.eye, job.lights, 1, job.env, 1, shadows, m.emissive, ly, n, behind, mp)
			} else {
				lit = shade(surface(m), p, n, job.eye, job.lights, 1, job.env, 1, shadows, m.emissive, m.layers.f0, mp)
			}
			lit = fogged(lit, p, job.eye, job.fog, job.fog_color)
			write_samples(s, width, dz, x, y, covered, light_of(lit, out_alpha, a), blend_of(m))
		}
	}
}

// A shadow-only triangle: its colour at its opacity times the light its
// shadows take (1 minus the product of their factors), blended.
@(private = "file")
draw_shadow_only :: proc(job: ^Frame_Job, ref: Tri_Ref, t: Tri, m: ^Shading_Material, y0, y1: int, area: f32) {
	s := job.s
	width := job.width
	dz := depth_plane(t, area)
	x_lo := clamp(int(math.floor(min(t.p[0].x, t.p[1].x, t.p[2].x))), 0, width)
	x_hi := clamp(int(math.ceil(max(t.p[0].x, t.p[1].x, t.p[2].x))), 0, width)
	vs := s.chunks[ref.chunk].values[t.values:][:3 * UNTEX_FLOATS]
	world, normal: [3]Vec3
	for k in 0 ..< 3 {
		world[k] = (^Vec3)(&vs[k * UNTEX_FLOATS])^
		normal[k] = (^Vec3)(&vs[k * UNTEX_FLOATS + 3])^
	}
	for y in max(t.min_y, y0) ..< min(t.max_y, y1) {
		for x in x_lo ..< x_hi {
			covered := coverage(t, x, y, area)
			if covered == {} {
				continue
			}
			c := Vec2{f32(x) + 0.5, f32(y) + 0.5}
			b0 := edge(t.p[1].xy, t.p[2].xy, c) / area
			b1 := edge(t.p[2].xy, t.p[0].xy, c) / area
			b2 := 1 - b0 - b1
			iw := b0 * t.inv_w[0] + b1 * t.inv_w[1] + b2 * t.inv_w[2]
			p := (world[0] * b0 + world[1] * b1 + world[2] * b2) / iw
			n := linalg.normalize((normal[0] * b0 + normal[1] * b1 + normal[2] * b2) / iw)
			mask: f32 = 1
			for si in 0 ..< job.shadow_count {
				mask *= shadow_factor(s.shadow_maps[si], &job.shadows[si], p, n)
			}
			lit := fogged(m.base, p, job.eye, job.fog, job.fog_color)
			write_samples(s, width, dz, x, y, covered, light_of(lit, m.opacity * (1 - mask), 1), blend_of(m))
		}
	}
}

@(private = "file")
draw_standard_textured :: #force_inline proc(job: ^Frame_Job, ref: Tri_Ref, t: Tri, m: ^Shading_Material, y0, y1: int, area: f32, blend, receive: bool) {
	s := job.s
	width := job.width
	dz := depth_plane(t, area)
	coloured := .Vertex_Colors in m.flags
	nf := STANDARD_FLOATS + (COLOR_FLOATS if coloured else 0)
	vs := s.chunks[ref.chunk].values[t.values:][:3 * nf]
	world, normal: [3]Vec3
	uv, uv1: [3]Vec2
	vcolor: [3]Vec4
	for k in 0 ..< 3 {
		base := k * nf
		world[k] = (^Vec3)(&vs[base])^
		normal[k] = (^Vec3)(&vs[base + 3])^
		uv[k] = (^Vec2)(&vs[base + 6])^
		uv1[k] = (^Vec2)(&vs[base + 8])^
		if coloured {
			vcolor[k] = (^Vec4)(&vs[base + STANDARD_FLOATS])^
		}
	}
	flat_n: Vec3
	if .Flat_Shading in m.flags {
		w0 := world[0] / t.inv_w[0]
		flat_n = linalg.normalize(linalg.cross(world[1] / t.inv_w[1] - w0, world[2] / t.inv_w[2] - w0))
	}
	curved := .Curves in m.flags
	words: []u32
	even_odd: bool
	if curved {
		words, even_odd = curve_entry_words(job.geos, t.curve)
	}
	// Triplanar: its maps projected in its frame.
	triplanar := .Triplanar in m.flags
	frame: Projection_Frame = ---
	if triplanar {
		frame = s.frames[t.draw]
	}
	// A particle: its row's colour and opacity; soft against the frame's depth.
	sprite := .Sprite in m.flags
	row := s.expanded.sprite_rows[t.curve] if sprite && int(t.curve) < len(s.expanded.sprite_rows) else {color = 1}
	soft := sprite && m.softness > 0 && len(s.depth_map) == s.width * s.height
	// Probed: faded by the share of its disc unhidden (the same for the quad).
	if sprite && row.probe.w > 0 && len(s.depth_map) == s.width * s.height {
		row.color.a *= sprite_visibility(row.probe, s.depth_map[:], s.width, s.height)
	}
	inv_area := 1 / area
	x_lo := clamp(int(math.floor(min(t.p[0].x, t.p[1].x, t.p[2].x))), 0, width) &~ 1
	x_hi := clamp(int(math.ceil(max(t.p[0].x, t.p[1].x, t.p[2].x))), 0, width)
	clipping := job.clip != {}
	reflective := .Reflection in m.flags
	for qy := max(t.min_y, y0) &~ 1; qy < min(t.max_y, y1); qy += 2 {
		for qx := x_lo; qx < x_hi; qx += 2 {
			lanes, helpers := quad_coverage(t, qx, qy, width, y1, area)
			if card(helpers) == 4 {
				continue
			}
			worlds, norms: [4]Vec3 = ---, ---
			uvs, uv1s: [4]Vec2 = ---, ---
			colors: [4]Vec4 = ---
			projected: [4]Vec3 = ---
			for l in 0 ..< 4 {
				c := Vec2{f32(qx + l & 1) + 0.5, f32(qy + l >> 1) + 0.5}
				b0 := edge(t.p[1].xy, t.p[2].xy, c) * inv_area
				b1 := edge(t.p[2].xy, t.p[0].xy, c) * inv_area
				b2 := 1 - b0 - b1
				inv_iw := 1 / (b0 * t.inv_w[0] + b1 * t.inv_w[1] + b2 * t.inv_w[2])
				worlds[l] = (world[0] * b0 + world[1] * b1 + world[2] * b2) * inv_iw
				uvs[l] = (uv[0] * b0 + uv[1] * b1 + uv[2] * b2) * inv_iw
				uv1s[l] = (uv1[0] * b0 + uv1[1] * b1 + uv1[2] * b2) * inv_iw
				if coloured {
					colors[l] = (vcolor[0] * b0 + vcolor[1] * b1 + vcolor[2] * b2) * inv_iw
				}
				if triplanar {
					projected[l] = (frame.to_local * Vec4{worlds[l].x, worlds[l].y, worlds[l].z, 1}).xyz * m.projection_scale
				}
				if l not_in helpers && .Flat_Shading not_in m.flags {
					norms[l] = linalg.normalize((normal[0] * b0 + normal[1] * b1 + normal[2] * b2) * inv_iw)
				}
			}
			for l in 0 ..< 4 {
				if lanes[l] == {} || clipping && clipped(job, worlds[l]) {
					continue
				}
				x, y := qx + l & 1, qy + l >> 1
				d0x, d0y := quad_deriv(uvs, l)
				d1x, d1y := quad_deriv(uv1s, l)
				// Triplanar: where the lane reads (nil: by uv).
				lane: Triplanar_Lane = ---
				tl: ^Triplanar_Lane
				if triplanar {
					project_lane(&lane, projected, l, frame.normal_to_local * (flat_n if .Flat_Shading in m.flags else norms[l]), m.projection_sharpness)
					tl = &lane
				}
				pm := surface(m)
				emissive := m.emissive
				opacity := m.opacity
				if coloured {
					pm.base *= colors[l].rgb
					if blend || .Alpha_Mask in m.flags {
						opacity *= colors[l].a
					}
				}
				if .Color_Map in m.flags {
					tex := read_map(job, m, tl, .Color, m.color_map, uvs[l], uv1s[l], d0x, d0y, d1x, d1y)
					if .Premultiplied in m.flags {
						tex.rgb = linalg.clamp(tex.rgb / tex.a, 0, 1) if tex.a > 0 else 0
					}
					pm.base *= linear(tex.rgb)
					if blend || .Alpha_Mask in m.flags {
						opacity *= tex.a
					}
				}
				if .Alpha_Map in m.flags {
					opacity *= read_map(job, m, tl, .Alpha, m.alpha_map, uvs[l], uv1s[l], d0x, d0y, d1x, d1y).g
				}
				if sprite {
					pm.base *= linear(row.color.rgb)
					opacity *= clamp(row.color.a, 0, 1)
					if soft {
						// Faded by how far in front of the frame's depth it is.
						clip := job.view_proj * Vec4{worlds[l].x, worlds[l].y, worlds[l].z, 1}
						in_front := view_depth(job.scene.camera, s.depth_map[y * s.width + x]) - view_depth(job.scene.camera, clip.z / clip.w)
						opacity *= clamp(in_front / m.softness, 0, 1)
					}
				}
				if .Alpha_Mask in m.flags {
					if opacity < m.alpha_cutoff {
						continue
					}
					if !blend {
						opacity = 1
					}
				}
				covered := lanes[l]
				if curved {
					if blend {
						cov := curve_coverage(words, uv1s[l], d1x, d1y, even_odd)
						if cov <= 0 {
							continue
						}
						opacity *= cov
					} else {
						covered &= curve_samples(words, uv1s[l], d1x, d1y, even_odd)
						if covered == {} {
							continue
						}
					}
				}
				if .Roughness_Map in m.flags {
					pm.roughness *= read_map(job, m, tl, .Roughness, m.roughness_map, uvs[l], uv1s[l], d0x, d0y, d1x, d1y).g
				}
				if .Metalness_Map in m.flags {
					pm.metalness *= read_map(job, m, tl, .Metalness, m.metalness_map, uvs[l], uv1s[l], d0x, d0y, d1x, d1y).b
				}
				if .Emissive_Map in m.flags {
					emissive *= linear(read_map(job, m, tl, .Emissive, m.emissive_map, uvs[l], uv1s[l], d0x, d0y, d1x, d1y).rgb)
				}
				ao: f32 = 1
				if .Ao_Map in m.flags {
					ao = (read_map(job, m, tl, .Ao, m.ao_map, uvs[l], uv1s[l], d0x, d0y, d1x, d1y).r - 1) * m.ao_intensity + 1
				}
				n := flat_n if .Flat_Shading in m.flags else norms[l]
				shadows := [MAX_SHADOWS]f32{1, 1, 1, 1}
				if receive {
					for si in 0 ..< job.shadow_count {
						shadows[si] = shadow_factor(s.shadow_maps[si], &job.shadows[si], worlds[l], n)
					}
				}
				// The layers' maps, and the coat's normal: the surface's
				// before its normal map (facing the eye when two-sided), or
				// the coat's normal map's.
				ly := &m.layers
				local: Shading_Layers = ---
				if m.flags & {.Layer_Maps, .Transmission} != {} {
					local = m.layers
					ly = &local
					if .Layer_Maps in m.flags {
						read_layers(job, m, tl, ly, uvs[l], uv1s[l], d0x, d0y, d1x, d1y)
					}
				}
				if .Two_Sided in pm.flags && linalg.dot(n, job.eye - worlds[l]) < 0 {
					n = -n
				}
				pm.flags -= {.Two_Sided}
				coat_n := n
				if .Normal_Map in m.flags || .Clearcoat_Normal_Map in m.flags {
					dp_dx, dp_dy := quad_deriv(worlds, l)
					if .Normal_Map in m.flags {
						n = map_normal(job, m, tl, frame, .Normal, m.normal_map, m.normal_scale, n, dp_dx, dp_dy, uvs[l], uv1s[l], d0x, d0y, d1x, d1y)
					}
					if .Clearcoat_Normal_Map in m.flags {
						coat_n = map_normal(job, m, tl, frame, .Clearcoat_Normal, m.layer_maps[.Clearcoat_Normal], ly.coat_normal_scale, coat_n, dp_dx, dp_dy, uvs[l], uv1s[l], d0x, d0y, d1x, d1y)
					}
				}
				behind: Vec4
				if .Transmission in m.flags {
					behind = behind_of(job, worlds[l], n, pm.roughness, ly, false)
				}
				mirror: Mirror_Light = ---
				mp: ^Mirror_Light
				if reflective {
					mirror = mirror_at(job, m, x, y, pm.roughness, ly.clearcoat_roughness)
					mp = &mirror
				}
				a: f32 = 1
				lit: Vec3
				if m.flags & LAYER_FLAGS != {} {
					a, lit = shade_layered(pm, worlds[l], n, job.eye, job.lights, 1, job.env, ao, shadows, emissive, ly, coat_n, behind, mp)
				} else {
					lit = shade(pm, worlds[l], n, job.eye, job.lights, 1, job.env, ao, shadows, emissive, m.layers.f0, mp)
				}
				lit = fogged(lit, worlds[l], job.eye, job.fog, job.fog_color)
				write_samples(s, width, dz, x, y, covered, light_of(lit, opacity, a), blend_of(m))
			}
		}
	}
}

// Where a triplanar lane reads its maps: its position projected along x,
// y and z (as (z, y), (x, z), (x, y)), their derivatives and weights.
@(private = "file")
Triplanar_Lane :: struct {
	proj, proj_dx, proj_dy: [3]Vec2,
	weights:                Vec3,
}

// A lane's triplanar projections from its quad's projected positions, the
// weights from its normal in the projection's frame: |n| to the
// sharpness, summing to 1 (shaders/o3d.slang's `project_lane`).
@(private = "file")
project_lane :: #force_inline proc "contextless" (at: ^Triplanar_Lane, projected: [4]Vec3, l: int, normal: Vec3, sharpness: f32) {
	p := projected[l]
	dx, dy := quad_deriv(projected, l)
	at.proj = {p.zy, p.xz, p.xy}
	at.proj_dx = {dx.zy, dx.xz, dx.xy}
	at.proj_dy = {dy.zy, dy.xz, dy.xy}
	n := linalg.normalize(normal)
	w := Vec3{math.pow(abs(n.x), sharpness), math.pow(abs(n.y), sharpness), math.pow(abs(n.z), sharpness)}
	at.weights = w / max(w.x + w.y + w.z, 1e-20)
}

// A material map at a lane: by its uv (`at` nil), or its three
// projections' samples weighted (a projection weighing nothing is not read).
@(private = "file")
read_map :: #force_inline proc "contextless" (job: ^Frame_Job, m: ^Shading_Material, at: ^Triplanar_Lane, slot: Texture_Slot, id: Texture_Id, uv0, uv1, d0x, d0y, d1x, d1y: Vec2) -> Vec4 {
	if at == nil {
		uv, dx, dy := slot_uv(m, slot, uv0, uv1, d0x, d0y, d1x, d1y)
		return sample_map(job, id, uv, dx, dy)
	}
	return read_triplanar(job, m, at, slot, id)
}

// Out of line: the uv path stays small where it is inlined.
@(private = "file")
read_triplanar :: #force_no_inline proc "contextless" (job: ^Frame_Job, m: ^Shading_Material, at: ^Triplanar_Lane, slot: Texture_Slot, id: Texture_Id) -> Vec4 {
	acc: Vec4
	for a in 0 ..< 3 {
		if at.weights[a] > 0 {
			acc += at.weights[a] * read_projection(job, m, at, slot, id, a)
		}
	}
	return acc
}

@(private = "file")
read_projection :: #force_inline proc "contextless" (job: ^Frame_Job, m: ^Shading_Material, at: ^Triplanar_Lane, slot: Texture_Slot, id: Texture_Id, axis: int) -> Vec4 {
	uv, dx, dy := transform_uv(m.uv_transforms[slot], at.proj[axis], at.proj_dx[axis], at.proj_dy[axis])
	return sample_map(job, id, uv, dx, dy)
}

// A normal map's normal (slot `slot`, its tangent x and y times `scale`)
// about `n`: by the uv's tangent frame from screen derivatives, or
// triplanar.
@(private = "file")
map_normal :: #force_inline proc "contextless" (job: ^Frame_Job, m: ^Shading_Material, at: ^Triplanar_Lane, frame: Projection_Frame, slot: Texture_Slot, id: Texture_Id, scale: Vec2, n, dp_dx, dp_dy: Vec3, uv0, uv1, d0x, d0y, d1x, d1y: Vec2) -> Vec3 {
	if at != nil {
		return triplanar_normal(job, m, at, frame, slot, id, scale, n)
	}
	uv, dx, dy := slot_uv(m, slot, uv0, uv1, d0x, d0y, d1x, d1y)
	map_n := sample_map(job, id, uv, dx, dy).rgb * 2 - 1
	map_n.xy *= scale
	return perturb_normal(n, dp_dx, dp_dy, dx, dy, map_n)
}

// A triplanar normal map's normal: each projection's tangent normal laid
// on the surface normal's axes and blended by the weights (a whiteout
// blend), in the projection's frame, then back in the world's.
@(private = "file")
triplanar_normal :: proc "contextless" (job: ^Frame_Job, m: ^Shading_Material, at: ^Triplanar_Lane, frame: Projection_Frame, slot: Texture_Slot, id: Texture_Id, scale: Vec2, n: Vec3) -> Vec3 {
	nl := linalg.normalize(frame.normal_to_local * n)
	tn: [3]Vec3
	for a in 0 ..< 3 {
		if at.weights[a] > 0 {
			tn[a] = read_projection(job, m, at, slot, id, a).rgb * 2 - 1
			tn[a].xy *= scale
		}
	}
	bx := Vec3{tn[0].x + nl.z, tn[0].y + nl.y, abs(tn[0].z) * nl.x}
	by := Vec3{tn[1].x + nl.x, tn[1].y + nl.z, abs(tn[1].z) * nl.y}
	bz := Vec3{tn[2].x + nl.x, tn[2].y + nl.y, abs(tn[2].z) * nl.z}
	w := at.weights
	blended := bx.zyx * w.x + by.xzy * w.y + bz * w.z
	return linalg.normalize(frame.normal_to_world * linalg.normalize(blended))
}

// The layers' multiplying maps at a lane, into `ly`: clear coat (red) and
// its roughness (green), sheen colour (rgb, sRGB) and roughness (alpha),
// transmission (red) and thickness (green).
@(private = "file")
read_layers :: proc "contextless" (job: ^Frame_Job, m: ^Shading_Material, at: ^Triplanar_Lane, ly: ^Shading_Layers, uv0, uv1, d0x, d0y, d1x, d1y: Vec2) {
	read :: #force_inline proc "contextless" (job: ^Frame_Job, m: ^Shading_Material, at: ^Triplanar_Lane, r: Layer_Map, uv0, uv1, d0x, d0y, d1x, d1y: Vec2) -> Vec4 {
		return read_map(job, m, at, layer_slot(r), m.layer_maps[r], uv0, uv1, d0x, d0y, d1x, d1y)
	}
	if m.layer_maps[.Clearcoat] != 0 {
		ly.clearcoat *= read(job, m, at, .Clearcoat, uv0, uv1, d0x, d0y, d1x, d1y).r
	}
	if m.layer_maps[.Clearcoat_Roughness] != 0 {
		ly.clearcoat_roughness *= read(job, m, at, .Clearcoat_Roughness, uv0, uv1, d0x, d0y, d1x, d1y).g
	}
	if m.layer_maps[.Sheen_Color] != 0 {
		ly.sheen *= linear(read(job, m, at, .Sheen_Color, uv0, uv1, d0x, d0y, d1x, d1y).rgb)
	}
	if m.layer_maps[.Sheen_Roughness] != 0 {
		ly.sheen_roughness *= read(job, m, at, .Sheen_Roughness, uv0, uv1, d0x, d0y, d1x, d1y).a
	}
	if m.layer_maps[.Transmission] != 0 {
		ly.transmission *= read(job, m, at, .Transmission, uv0, uv1, d0x, d0y, d1x, d1y).r
	}
	if m.layer_maps[.Thickness] != 0 {
		ly.thickness *= read(job, m, at, .Thickness, uv0, uv1, d0x, d0y, d1x, d1y).g
	}
}

// The area of the pixel at shape coordinate `rc` the shape covers, the
// pixel's footprint the parallelogram its screen derivatives span
// (shaders/o3d.slang's `fs_curves` likewise): exact under rotation, skew
// and grazing views, so never faded to the shape's mean (`curves.fade`),
// which it reaches by itself once the footprint holds the whole shape.
@(private = "file")
curve_coverage :: #force_inline proc "contextless" (words: []u32, rc, dx, dy: Vec2, even_odd: bool) -> f32 {
	if len(words) == 0 {
		return 0
	}
	return curves.coverage_affine(words, rc, dx, dy, even_odd)
}

// The pixel's samples inside the shape: the shape coordinate at each sample
// position from the centre's and its derivatives.
@(private = "file")
curve_samples :: #force_inline proc "contextless" (words: []u32, rc, dx, dy: Vec2, even_odd: bool) -> (inside: Sample_Mask) {
	if len(words) == 0 {
		return
	}
	for k in 0 ..< SAMPLES {
		o := SAMPLE_OFFSETS[k] - 0.5
		if curves_inside(words, rc + dx * o.x + dy * o.y, even_odd) {
			inside += {k}
		}
	}
	return
}


// Screen-space horizontal and vertical finite differences across a 2x2 quad.
@(private = "file")
quad_deriv :: #force_inline proc "contextless" (v: [4]$T, lane: int) -> (dx, dy: T) {
	row := lane & 2
	col := lane & 1
	return v[row + 1] - v[row], v[2 + col] - v[col]
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
draw_custom :: #force_no_inline proc(job: ^Frame_Job, w: ^Worker, refs: []Tri_Ref, y0, y1: int) -> int {
	s := job.s
	width := job.width
	q := &w.quads
	clear(&q.origin)
	clear(&q.helpers)
	clear(&q.facing)
	clear(&q.covered)
	clear(&q.tri)
	draw := -1
	taken := 0
	for ref in refs {
		t := s.chunks[ref.chunk].tris[ref.tri]
		if draw >= 0 && (t.draw != draw || len(q.origin) >= QUAD_BATCH) {
			break
		}
		draw = t.draw
		taken += 1
		area := edge(t.p[0].xy, t.p[1].xy, t.p[2].xy)
		if area == 0 {
			continue
		}
		// Quads at even pixels: a band (BAND_ROWS even) holds whole quads.
		x_lo := clamp(int(math.floor(min(t.p[0].x, t.p[1].x, t.p[2].x))), 0, width) &~ 1
		x_hi := clamp(int(math.ceil(max(t.p[0].x, t.p[1].x, t.p[2].x))), 0, width)
		for qy := max(t.min_y, y0) &~ 1; qy < min(t.max_y, y1); qy += 2 {
			for qx := x_lo; qx < x_hi; qx += 2 {
				lanes, helpers := quad_coverage(t, qx, qy, width, y1, area)
				if card(helpers) == 4 {
					continue
				}
				append(&q.origin, [2]i32{i32(qx), i32(qy)})
				append(&q.helpers, helpers)
				append(&q.facing, t.front)
				append(&q.covered, lanes)
				append(&q.tri, ref)
			}
		}
	}
	nq := len(q.origin)
	if nq == 0 {
		return taken
	}
	sh := shader_get(job.shaders, s.materials[draw].shader)
	nf := sh.varying_floats
	// Each lane's varyings, perspective-correct (helpers' extrapolated).
	table_grow_non_zero(&q.values, nf * nq * 4)
	for qi in 0 ..< nq {
		ref := q.tri[qi]
		t := s.chunks[ref.chunk].tris[ref.tri]
		area := edge(t.p[0].xy, t.p[1].xy, t.p[2].xy)
		vs := s.chunks[ref.chunk].values[t.values:][:3 * nf]
		for l in 0 ..< 4 {
			c := Vec2{f32(int(q.origin[qi].x) + l & 1) + 0.5, f32(int(q.origin[qi].y) + l >> 1) + 0.5}
			b0 := edge(t.p[1].xy, t.p[2].xy, c) / area
			b1 := edge(t.p[2].xy, t.p[0].xy, c) / area
			b2 := 1 - b0 - b1
			iw := b0 * t.inv_w[0] + b1 * t.inv_w[1] + b2 * t.inv_w[2]
			j := qi * 4 + l
			for f in 0 ..< nf {
				q.values[f * nq * 4 + j] = (vs[f] * b0 + vs[nf + f] * b1 + vs[2 * nf + f] * b2) / iw
			}
			for f in shader_flat(sh) {
				q.values[int(f) * nq * 4 + j] = vs[f] // kept as is: the first corner's
			}
		}
	}
	table_grow_non_zero(&q.colours, nq * 4)
	table_grow_non_zero(&q.shaded, nq)
	quads := oksl.Fragment_Quads {
		count        = nq,
		origin       = q.origin[:],
		front_facing = q.facing[:],
		helpers      = q.helpers[:],
		varyings     = q.values[:],
	}
	maps := Shader_Maps{s = s, textures = job.textures, ids = s.materials[draw].shader_maps, shadow_channel = int(sh.shadow_channel)}
	if job.shadow_count > 0 {
		maps.shadow, maps.shadow_size = s.shadow_maps[0], job.shadows[0].size
	}
	sizes: [SHADER_MAPS][2]i32
	for id, c in maps.ids {
		if c == maps.shadow_channel {
			sizes[c] = {i32(maps.shadow_size), i32(maps.shadow_size)}
		} else if ext, bound := bound_external(s, job.textures, id); bound {
			sizes[c] = {i32(ext.width), i32(ext.height)}
		} else if tex, ok := texture_get(job.textures, id); ok && !tex.external {
			sizes[c] = {i32(tex.width), i32(tex.height)}
		}
	}
	oksl.run_fragment(&sh.pipeline.fragment.program, block(job.blocks, draw, .Fragment), quads, q.colours[:], q.shaded[:], &w.vm, {sampler = sample_shader_map, user = &maps, channel_sizes = sizes[:]})
	blend := blend_of(&s.materials[draw])
	for qi in 0 ..< nq {
		ref := q.tri[qi]
		t := s.chunks[ref.chunk].tris[ref.tri]
		dz := depth_plane(t, edge(t.p[0].xy, t.p[1].xy, t.p[2].xy))
		for l in q.shaded[qi] {
			x, y := int(q.origin[qi].x) + l & 1, int(q.origin[qi].y) + l >> 1
			// Its output is light (three.js's in a render target), premultiplied
			// when blended, as the GPU's SRC_ALPHA blend multiplies it.
			c := q.colours[qi * 4 + l]
			a := clamp(c.a, 0, 1)
			k: f32 = a if blend != .Opaque else 1
			write_samples(s, width, dz, x, y, q.covered[qi][l], {c.r * k, c.g * k, c.b * k, a}, blend)
		}
	}
	return taken
}

// A ShaderMaterial draw's textures, for its sampler, and `shadowAt`'s
// shadow map (its channel; -1: none).
@(private = "file")
Shader_Maps :: struct {
	s:              ^Software,
	textures:       ^Textures,
	ids:            [SHADER_MAPS]Texture_Id,
	shadow_channel: int,
	shadow:         []f32,
	shadow_size:    int,
}

// oksl's sampler: a channel's texture at uv (running down the image, as
// the prelude flips three.js's) and the level oksl picked, repeating as
// the GPU's sampler; white without one.
@(private = "file")
sample_shader_map :: proc(user: rawptr, channel: int, uv: [2]f32, lod: f32) -> [4]f32 {
	maps := (^Shader_Maps)(user)
	if channel < 0 || channel >= SHADER_MAPS {
		return 1
	}
	if channel == maps.shadow_channel {
		// The texel, rows from the top, nearest (the GPU's shadow sampler).
		size := maps.shadow_size
		if size == 0 || len(maps.shadow) < size * size {
			return 0
		}
		x := clamp(int(math.floor(uv.x * f32(size))), 0, size - 1)
		y := clamp(int(math.floor(uv.y * f32(size))), 0, size - 1)
		return maps.shadow[y * size + x]
	}
	if _, ok := texture_get(maps.textures, maps.ids[channel]); !ok {
		return 1
	}
	if ext, ok := bound_external(maps.s, maps.textures, maps.ids[channel]); ok {
		return sample_mip(ext.pixels, ext.width, ext.height, {uv.x, 1 - uv.y})
	}
	return sample_lod(maps.textures, maps.ids[channel], {uv.x, 1 - uv.y}, lod, clamp_v = false)
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
// target blends.
@(private = "file")
write_samples :: #force_inline proc "contextless" (s: ^Software, width: int, dz: Depth_Plane, x, y: int, covered: Sample_Mask, light: [4]f32, blend: Blend) {
	px := to_halves(light)
	for k in covered {
		z := dz.z0 + dz.dx * (f32(x) + SAMPLE_OFFSETS[k].x - dz.x0) + dz.dy * (f32(y) + SAMPLE_OFFSETS[k].y - dz.y0)
		i := (y * width + x) * SAMPLES + k
		if z < 0 || z > 1 || z <= s.depth[i] {
			continue
		}
		switch blend {
		case .Opaque:
			s.light[i] = px
			s.depth[i] = z
		case .Over:
			s.light[i] = to_halves(light + from_halves(s.light[i]) * (1 - light.a))
		case .Add:
			dst := from_halves(s.light[i])
			s.light[i] = to_halves({light.r + dst.r, light.g + dst.g, light.b + dst.b, light.a + dst.a * (1 - light.a)})
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

// Linear colour to sRGB premultiplied RGBA8.
@(private = "file")
pack :: #force_inline proc "contextless" (c: Vec3, alpha: f32) -> u32 {
	// Clamped as the GPU's 8-bit target clamps: out of range, a channel
	// would spill into the next.
	k := clamp(alpha, 0, 1)
	r := u32(math.round(linear_to_srgb(c.r) * k * 255))
	g := u32(math.round(linear_to_srgb(c.g) * k * 255))
	b := u32(math.round(linear_to_srgb(c.b) * k * 255))
	a := u32(math.round(k * 255))
	return r | g << 8 | b << 16 | a << 24
}

// The pixels the host bound to external texture `id` for the software
// renders (`software_set_external`).
@(private = "file")
bound_external :: #force_inline proc "contextless" (s: ^Software, textures: ^Textures, id: Texture_Id) -> (Software_External, bool) {
	if id == 0 || int(id) > len(s.externals) || s.externals[id - 1].pixels == nil {
		return {}, false
	}
	tex, ok := texture_get(textures, id)
	return s.externals[id - 1], ok && tex.external
}

// A map at `uv` as `sample` reads it; an external texture from the pixels
// bound to it, at their level 0, bilinear (as the GPU reads an unmipped
// one).
@(private = "file")
sample_map :: #force_inline proc "contextless" (job: ^Frame_Job, id: Texture_Id, uv, duv_dx, duv_dy: Vec2) -> Vec4 {
	if ext, ok := bound_external(job.s, job.textures, id); ok {
		return sample_mip(ext.pixels, ext.width, ext.height, uv)
	}
	return sample(job.textures, id, uv, duv_dx, duv_dy)
}

// `sample_top`'s, as `sample_map` reads externals.
@(private = "file")
sample_map_top :: #force_inline proc "contextless" (job: ^Frame_Job, id: Texture_Id, uv: Vec2) -> Vec4 {
	if ext, ok := bound_external(job.s, job.textures, id); ok {
		return sample_mip(ext.pixels, ext.width, ext.height, uv)
	}
	return sample_top(job.textures, id, uv)
}
