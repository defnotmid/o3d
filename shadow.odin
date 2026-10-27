package o3d

import "core:math"
import "core:math/linalg"

// Shadows, as three.js's light shadows with their PCF shadow map: each
// directional or spot light with `cast_shadow` (up to MAX_SHADOWS, among the
// first MAX_LIGHTS) draws the depth of the draws with `.Cast_Shadow` along
// its light into a square map, a directional light's through an
// orthographic camera fitted to what the frame shows (`fit_shadows`), a
// spot light's through its cone; a draw with `.Receive_Shadow` scales that
// light's direct light by `shadow_factor`.
// Casters are drawn with the side opposite their material's (three.js's
// default shadowSide), which keeps closed meshes from shadowing their own
// lit faces, and with their plain geometry (a ShaderMaterial's vertex
// shader is not run, as three.js's depth material does not). ShaderMaterials
// do not receive shadows.
//
// Both renderers draw the map the same way (depth at texel centres, nearest
// wins) and filter it with `shadow_factor`'s arithmetic (shaders/o3d.slang
// has it again; the tests compare them).

// A light's shadow settings; zero fields take their defaults.
Light_Shadow :: struct {
	// Texels per side (mapSize; 0: 512).
	map_size:    int,
	// A directional light's camera: its square view, world units across
	// about the light's target, and its depth range from the light's
	// position (0: 0.5 and 500). With no height it is fitted to the frame
	// (`fit_shadows`): the light's position and target give only its
	// direction. A spot light's camera is its cone, out to `far` (0: the
	// light's distance, else 500) from `near`.
	height:      f32,
	near, far:   f32,
	// The point compared moves this far away from the light, world units
	// (slightly negative, toward the light, reduces acne).
	bias:        f32,
	// The point compared moves this far along its normal, world units
	// (normalBias).
	normal_bias: f32,
}

MAX_SHADOWS :: 4
SHADOW_MAP_SIZE :: 512
SHADOW_HEIGHT :: 10
SHADOW_NEAR :: 0.5
SHADOW_FAR :: 500

// A frame's shadow: the light's index in `Scene.lights` (-1: none), its
// camera and settings resolved, and where its light comes from (toward it,
// w 0, a directional light's; its position, w 1, a spot light's).
Shadow :: struct {
	light:             int,
	view_proj:         Mat4,
	size:              int,
	bias, normal_bias: f32,
	from:              Vec4,
}

// A fitted camera's sizes: this many to a doubling, so a little change in
// what it holds keeps its texels where they were.
SHADOW_FIT_STEPS :: 8

@(private = "file")
make_light_shadow :: proc "contextless" (l: Light, index: int) -> (sh: Shadow) {
	s := l.shadow
	near := s.near if s.near > 0 else SHADOW_NEAR
	view := look_at(l.position, l.target)
	vp: Mat4
	from := Vec4{l.position.x, l.position.y, l.position.z, 1}
	if l.kind == .Spot {
		angle := clamp(l.angle if l.angle > 0 else math.PI / 3, 1e-3, math.PI * 0.499)
		fov := 2 * angle * math.DEG_PER_RAD
		far_def := l.distance if l.distance > near else max(SHADOW_FAR, near * 2)
		far := s.far if s.far > near else far_def
		vp = perspective(fov, 1, near, far) * view
	} else {
		far := s.far if s.far > near else max(SHADOW_FAR, near * 2)
		vp = orthographic(s.height if s.height > 0 else SHADOW_HEIGHT, 1, near, far) * view
		from = {view[2, 0], view[2, 1], view[2, 2], 0}
	}
	return {
		light       = index,
		view_proj   = vp,
		size        = s.map_size if s.map_size > 0 else SHADOW_MAP_SIZE,
		bias        = s.bias,
		normal_bias = s.normal_bias,
		from        = from,
	}
}

// Up to `MAX_SHADOWS` shadow-casting directional or spot lights among the
// first `MAX_LIGHTS` in `scene.lights`; the directional ones with no
// `height` fitted to the receivers `views` see (`fit_shadows`).
frame_shadows :: proc(scene: ^Scene, geos: ^Geometries, shaders: ^Shaders, views: []Mat4, out: ^[MAX_SHADOWS]Shadow, exp: ^Expanded_Draws = nil) -> int {
	n := 0
	fitted: [MAX_SHADOWS]bool
	for l, i in scene.lights[:min(len(scene.lights), MAX_LIGHTS)] {
		if (l.kind == .Directional || l.kind == .Spot) && l.cast_shadow {
			out[n] = make_light_shadow(l, i)
			fitted[n] = l.kind == .Directional && l.shadow.height <= 0
			n += 1
			if n == MAX_SHADOWS {
				break
			}
		}
	}
	fit_shadows(scene, geos, shaders, views, out[:n], fitted[:n], exp)
	return n
}

// Light-space bounds (x and y across the light, z back toward it).
@(private = "file")
Box :: struct {
	lo, hi: Vec3,
}

@(private = "file")
BOX_EMPTY :: Box{{max(f32), max(f32), max(f32)}, {min(f32), min(f32), min(f32)}}

@(private = "file")
box_union :: #force_inline proc "contextless" (a, b: Box) -> Box {
	return {linalg.min(a.lo, b.lo), linalg.max(a.hi, b.hi)}
}

@(private = "file")
box_meet :: #force_inline proc "contextless" (a, b: Box) -> Box {
	return {linalg.max(a.lo, b.lo), linalg.min(a.hi, b.hi)}
}

// The fitted directional shadows' cameras: across the light, the receivers
// a view sees (their bounds within its frustum) where a caster is over
// them, nothing else (a point no caster is over is lit wherever the map
// ends); along it, from the nearest caster to the farthest such receiver.
// Its size is rounded up to a step (SHADOW_FIT_STEPS) and its corner kept
// on the texel grid, so its texels hold still while the camera moves
// within a step. Receivers are the draws with `.Receive_Shadow`, the shadow
// catchers and the ShaderMaterials that read the shadow (`shadowAt`);
// bounds are the draws' spheres (`draw_sphere`).
@(private = "file")
fit_shadows :: proc(scene: ^Scene, geos: ^Geometries, shaders: ^Shaders, all_views: []Mat4, shadows: []Shadow, fitted: []bool, exp: ^Expanded_Draws) {
	views := all_views[:min(len(all_views), 2)]
	turns: [MAX_SHADOWS]Mat4
	frusta: [MAX_SHADOWS][2]Box
	receivers, casters: [MAX_SHADOWS]Box
	any_fitted := false
	for sh, k in shadows {
		if !fitted[k] {
			continue
		}
		any_fitted = true
		turns[k] = look_at({}, -sh.from.xyz)
		receivers[k], casters[k] = BOX_EMPTY, BOX_EMPTY
		for vp, v in views {
			inv := linalg.inverse(vp)
			f := BOX_EMPTY
			for c in 0 ..< 8 {
				w := inv * Vec4{-1 if c & 1 == 0 else 1, -1 if c & 2 == 0 else 1, 0 if c & 4 == 0 else 1, 1}
				p := (turns[k] * Vec4{w.x / w.w, w.y / w.w, w.z / w.w, 0}).xyz
				f = box_union(f, {p, p})
			}
			frusta[k][v] = f
		}
	}
	if !any_fitted {
		return
	}
	n := len(scene.draws)
	models := scene.draws.model[:n]
	geometries := scene.draws.geometry[:n]
	for m, i in scene_materials(scene) {
		receives := m.flags & {.Receive_Shadow, .Shadow_Only} != {}
		if m.shader != 0 {
			s := shader_get(shaders, m.shader)
			receives = s != nil && s.shadow_channel >= 0
		}
		casts := .Cast_Shadow in m.flags
		if !receives && !casts {
			continue
		}
		centre, radius := draw_sphere(geos, geometries[i], models[i], scene_draw_joints(scene, i), scene_draw_morphs(scene, i), exp) or_continue
		seen: [2]bool
		if receives {
			for vp, v in views {
				seen[v] = sphere_visible(vp, centre, radius)
			}
		}
		for k in 0 ..< len(shadows) {
			if !fitted[k] {
				continue
			}
			p := (turns[k] * Vec4{centre.x, centre.y, centre.z, 0}).xyz
			b := Box{p - radius, p + radius}
			for v in 0 ..< len(views) {
				if seen[v] {
					receivers[k] = box_union(receivers[k], box_meet(b, frusta[k][v]))
				}
			}
			if casts {
				casters[k] = box_union(casters[k], b)
			}
		}
	}
	for &sh, k in shadows {
		if !fitted[k] {
			continue
		}
		r, c := receivers[k], casters[k]
		lo := linalg.max(r.lo.xy, c.lo.xy)
		hi := linalg.min(r.hi.xy, c.hi.xy)
		near, far: f32 = -1, 1
		if lo.x > hi.x || lo.y > hi.y || r.lo.z > r.hi.z {
			// Nothing in view is under a caster: any camera will do.
			lo, hi = -1, 1
		} else {
			pad := max((max(c.hi.z, r.hi.z) - r.lo.z) * 0.01, 1e-3)
			near, far = -max(c.hi.z, r.hi.z) - pad, -r.lo.z + pad
		}
		texels := f32(sh.size)
		extent := max(hi.x - lo.x, hi.y - lo.y, 1e-4) * texels / (texels - 1)
		extent = math.pow(2, math.ceil(math.log2(extent) * SHADOW_FIT_STEPS) / SHADOW_FIT_STEPS)
		texel := extent / texels
		corner := linalg.floor(lo / texel) * texel
		proj := Mat4 {
			2 / extent, 0, 0, -2 * corner.x / extent - 1,
			0, 2 / extent, 0, -2 * corner.y / extent - 1,
			0, 0, 1 / (far - near), far / (far - near),
			0, 0, 0, 1,
		}
		sh.view_proj = proj * turns[k]
	}
}

// The draws that cast into the map: `.Cast_Shadow`, with geometry (and a
// compiled shader if they ask for one), their bounding sphere in the
// light's view; in the order listed.
shadow_casters :: proc(scene: ^Scene, geos: ^Geometries, shaders: ^Shaders, sh: ^Shadow, order: ^[dynamic]int, exp: ^Expanded_Draws = nil) {
	clear(order)
	n := len(scene.draws)
	models := scene.draws.model[:n]
	geometries := scene.draws.geometry[:n]
	for m, i in scene_materials(scene) {
		if .Cast_Shadow not_in m.flags || (m.shader != 0 && shader_get(shaders, m.shader) == nil) {
			continue
		}
		if draw_seen(geos, geometries[i], models[i], sh.view_proj, scene_draw_joints(scene, i), scene_draw_morphs(scene, i), exp) {
			append(order, i)
		}
	}
}

// The draws a frame's depth (`Scene.depth`) holds: the opaque ones that can
// be seen (blended ones let what is behind show; ShaderMaterials with their
// plain geometry, as shadows draw them).
depth_draws :: proc(scene: ^Scene, geos: ^Geometries, shaders: ^Shaders, view_proj: Mat4, order: ^[dynamic]int, exp: ^Expanded_Draws = nil) {
	clear(order)
	n := len(scene.draws)
	models := scene.draws.model[:n]
	geometries := scene.draws.geometry[:n]
	for m, i in scene_materials(scene) {
		if blended(m) || (m.shader != 0 && shader_get(shaders, m.shader) == nil) {
			continue
		}
		if draw_seen(geos, geometries[i], models[i], view_proj, scene_draw_joints(scene, i), scene_draw_morphs(scene, i), exp) {
			append(order, i)
		}
	}
}

// The side a caster is drawn with: the other one (three.js's shadowSide).
shadow_side :: proc "contextless" (side: Side) -> Side {
	switch side {
	case .Front:
		return .Back
	case .Back:
		return .Front
	case .Double:
	}
	return .Double
}

// How lit a point (world position, normal) is by the shadow light, 0 to 1:
// three.js's PCF filter, a 3x3 grid of bilinear depth comparisons one texel
// apart, which weighs a 4x4 block of texels by (1 - f, 1, 1, f) per axis.
// A texel is lit when the point (moved along its normal by the normal bias,
// then away from the light by the bias) is no farther than it (depths
// reversed: no less); texels past the edge repeat the edge; outside the
// camera is lit.
shadow_factor :: proc "contextless" (depths: []f32, sh: ^Shadow, world, normal: Vec3) -> f32 {
	p := world + normal * sh.normal_bias
	toward := sh.from.xyz if sh.from.w == 0 else linalg.normalize0(sh.from.xyz - p)
	p -= toward * sh.bias
	c := sh.view_proj * Vec4{p.x, p.y, p.z, 1}
	if c.w <= 1e-6 {
		return 1
	}
	inv_w := 1 / c.w
	u := c.x * inv_w * 0.5 + 0.5
	v := 0.5 - c.y * inv_w * 0.5 // rows from the top, as the map is drawn
	z := c.z * inv_w
	if u < 0 || u > 1 || v < 0 || v > 1 || z < 0 || z > 1 {
		return 1
	}
	size := sh.size
	s := u * f32(size) - 0.5
	t := v * f32(size) - 0.5
	x0, y0 := math.floor(s), math.floor(t)
	fx, fy := s - x0, t - y0
	wx := [4]f32{1 - fx, 1, 1, fx}
	wy := [4]f32{1 - fy, 1, 1, fy}
	lit: f32
	for j in 0 ..< 4 {
		row := clamp(int(y0) - 1 + j, 0, size - 1)
		acc: f32
		for i in 0 ..< 4 {
			col := clamp(int(x0) - 1 + i, 0, size - 1)
			acc += wx[i] * (1 if z >= depths[row * size + col] else 0)
		}
		lit += wy[j] * acc
	}
	return lit / 9
}

@(private)
mat_scale :: #force_inline proc "contextless" (m: Mat4) -> f32 {
	return max(
		linalg.length(Vec3{m[0, 0], m[1, 0], m[2, 0]}),
		linalg.length(Vec3{m[0, 1], m[1, 1], m[2, 1]}),
		linalg.length(Vec3{m[0, 2], m[1, 2], m[2, 2]}),
	)
}

// Whether a draw can be seen through `view_proj`: it has geometry and its
// bounding sphere (covering its morph deltas and posed joints, and scaled
// with the draw) is at least partly inside.
draw_seen :: proc(geos: ^Geometries, id: Geometry_Id, model, view_proj: Mat4, joints: []Mat4 = nil, morphs: []f32 = nil, exp: ^Expanded_Draws = nil) -> bool {
	centre, radius := draw_sphere(geos, id, model, joints, morphs, exp) or_return
	return sphere_visible(view_proj, centre, radius)
}

// A draw's bounds in the world: its geometry's sphere about its origin,
// grown by its morphs, posed by its joints, placed by its model; not ok
// with nothing to draw.
draw_sphere :: proc(geos: ^Geometries, id: Geometry_Id, model: Mat4, joints: []Mat4 = nil, morphs: []f32 = nil, exp: ^Expanded_Draws = nil) -> (centre: Vec3, radius: f32, ok: bool) {
	geo := geometry_get_any(geos, exp, id) or_return
	if geo.index_count == 0 {
		return
	}
	r := geo.radius
	if len(morphs) > 0 && geo.morph_radius > 0 {
		weight_sum: f32
		for w in morphs[:min(len(morphs), int(geo.morph_targets))] {
			weight_sum += abs(w)
		}
		r += weight_sum * geo.morph_radius
	}
	if len(joints) > 0 {
		posed: f32
		for j in joints {
			posed = max(posed, linalg.length(Vec3{j[0, 3], j[1, 3], j[2, 3]}) + mat_scale(j) * r)
		}
		r = posed
	}
	return {model[0, 3], model[1, 3], model[2, 3]}, r * mat_scale(model), true
}

