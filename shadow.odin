package o3d

import "core:math"
import "core:math/linalg"

// Shadows, as three.js's DirectionalLightShadow with its PCF shadow map: the
// first directional light with `cast_shadow` (among the first MAX_LIGHTS)
// draws the depth of the draws with `.Cast_Shadow` from its position toward
// its target through an orthographic camera into a square map; a draw with
// `.Receive_Shadow` scales that light's direct light by `shadow_factor`.
// Casters are drawn with the side opposite their material's (three.js's
// default shadowSide), which keeps closed meshes from shadowing their own
// lit faces, and with their plain geometry (a ShaderMaterial's vertex
// shader is not run, as three.js's depth material does not). ShaderMaterials
// do not receive shadows.
//
// Both renderers draw the map the same way (depth at texel centres, nearest
// wins) and filter it with `shadow_factor`'s arithmetic (shaders/o3d.slang
// has it again; the tests compare them).

// three.js's shadow settings; zero fields take its defaults.
Light_Shadow :: struct {
	// Texels per side (mapSize; 0: 512).
	map_size:    int,
	// The camera's square view, world units across (0: 10, three.js's
	// -5..5), and its depth range (0: 0.5 and 500).
	height:      f32,
	near, far:   f32,
	// Added to the depth compared (bias; slightly negative reduces acne).
	bias:        f32,
	// The point compared moves this far along its normal (normalBias).
	normal_bias: f32,
}

MAX_SHADOWS :: 4
SHADOW_MAP_SIZE :: 512
SHADOW_HEIGHT :: 10
SHADOW_NEAR :: 0.5
SHADOW_FAR :: 500

// A frame's shadow: the light's index in `Scene.lights` (-1: none), its
// camera and settings resolved.
Shadow :: struct {
	light:             int,
	view_proj:         Mat4,
	size:              int,
	bias, normal_bias: f32,
}

// The first light that casts a shadow (-1: none).
shadow_light :: proc "contextless" (lights: []Light) -> int {
	for l, i in lights[:min(len(lights), MAX_LIGHTS)] {
		if (l.kind == .Directional || l.kind == .Spot) && l.cast_shadow {
			return i
		}
	}
	return -1
}

@(private = "file")
make_light_shadow :: proc "contextless" (l: Light, index: int) -> (sh: Shadow) {
	s := l.shadow
	near := s.near if s.near > 0 else SHADOW_NEAR
	view := look_at(l.position, l.target)
	vp: Mat4
	if l.kind == .Spot {
		angle := clamp(l.angle if l.angle > 0 else math.PI / 3, 1e-3, math.PI * 0.499)
		fov := 2 * angle * math.DEG_PER_RAD
		far_def := l.distance if l.distance > near else max(SHADOW_FAR, near * 2)
		far := s.far if s.far > near else far_def
		vp = perspective(fov, 1, near, far) * view
	} else {
		far := s.far if s.far > near else max(SHADOW_FAR, near * 2)
		vp = orthographic(s.height if s.height > 0 else SHADOW_HEIGHT, 1, near, far) * view
	}
	return {
		light       = index,
		view_proj   = vp,
		size        = s.map_size if s.map_size > 0 else SHADOW_MAP_SIZE,
		bias        = s.bias,
		normal_bias = s.normal_bias,
	}
}

frame_shadow :: proc "contextless" (scene: ^Scene) -> (sh: Shadow) {
	sh.light = shadow_light(scene.lights[:])
	if sh.light < 0 {
		return
	}
	return make_light_shadow(scene.lights[sh.light], sh.light)
}

// Up to `MAX_SHADOWS` shadow-casting directional or spot lights among the
// first `MAX_LIGHTS` in `scene.lights`.
frame_shadows :: proc "contextless" (scene: ^Scene, out: ^[MAX_SHADOWS]Shadow) -> int {
	n := 0
	for l, i in scene.lights[:min(len(scene.lights), MAX_LIGHTS)] {
		if (l.kind == .Directional || l.kind == .Spot) && l.cast_shadow {
			out[n] = make_light_shadow(l, i)
			n += 1
			if n == MAX_SHADOWS {
				break
			}
		}
	}
	return n
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
// A texel is lit when the point's depth (plus the bias) is no farther than
// it; texels past the edge repeat the edge; outside the camera is lit.
shadow_factor :: proc "contextless" (depths: []f32, sh: ^Shadow, world, normal: Vec3) -> f32 {
	p := world + normal * sh.normal_bias
	c := sh.view_proj * Vec4{p.x, p.y, p.z, 1}
	if c.w <= 0 {
		return 1
	}
	inv_w := 1 / c.w
	u := c.x * inv_w * 0.5 + 0.5
	v := 0.5 - c.y * inv_w * 0.5 // rows from the top, as the map is drawn
	z := c.z * inv_w + sh.bias
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
			acc += wx[i] * (1 if z <= depths[row * size + col] else 0)
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
	geo := geometry_get_any(geos, exp, id) or_return
	if geo.index_count == 0 {
		return false
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
	return sphere_visible(view_proj, {model[0, 3], model[1, 3], model[2, 3]}, r * mat_scale(model))
}

