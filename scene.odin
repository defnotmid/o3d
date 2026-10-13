package o3d

import "core:math"
import "core:math/linalg"

// A frame's scene: one camera, the lights, and draws (geometry, its world
// transform, its material). Filled from scratch every frame; the draws keep
// their order (transparent ones are drawn after the opaque ones, farthest
// first).

Camera :: struct {
	position:     Vec3,
	target:       Vec3,
	up:           Vec3, // zero means +Y
	// PerspectiveCamera's vertical field of view, degrees; near and far.
	fov:          f32,
	near, far:    f32,
	// OrthographicCamera instead: the view's height in world units.
	orthographic: bool,
	height:       f32,
}

CAMERA_DEFAULT :: Camera{position = {0, 0, 5}, fov = 50, near = 0.1, far = 2000}

Light_Kind :: enum u8 {
	Ambient, // AmbientLight(color, intensity)
	Hemisphere, // HemisphereLight(skyColor, groundColor, intensity)
	Directional, // DirectionalLight(color, intensity): from `position` toward `target`
	Point, // PointLight(color, intensity, distance, decay)
}

Light :: struct {
	kind:      Light_Kind,
	color:     Vec3, // sRGB (the sky's for a hemisphere light)
	ground:    Vec3, // a hemisphere light's ground colour, sRGB
	intensity: f32,
	position:  Vec3,
	target:    Vec3,
	// Point lights: the range (0: none) and the falloff exponent (2 is
	// physical, three.js's default).
	distance:  f32,
	decay:     f32,
}

Side :: enum u8 {
	Front,
	Back,
	Double,
}

// MeshStandardMaterial (or MeshBasicMaterial with `basic`).
Material :: struct {
	color:              Vec3, // sRGB
	metalness:          f32,
	roughness:          f32,
	emissive:           Vec3, // sRGB
	emissive_intensity: f32,
	opacity:            f32,
	// Blended over what is behind (opacity < 1 needs it, as in three.js).
	transparent:        bool,
	// Unlit: the colour as is.
	basic:              bool,
	side:               Side,
	flat_shading:       bool,
}

MATERIAL_DEFAULT :: Material{color = {1, 1, 1}, metalness = 0, roughness = 1, emissive_intensity = 1, opacity = 1}

Draw :: struct {
	geometry: Geometry_Id,
	model:    Mat4,
	material: Material,
}

Scene :: struct {
	camera: Camera,
	lights: [dynamic]Light,
	draws:  [dynamic]Draw,
}

scene_clear :: proc(s: ^Scene) {
	s.camera = CAMERA_DEFAULT
	clear(&s.lights)
	clear(&s.draws)
}

scene_destroy :: proc(s: ^Scene) {
	delete(s.lights)
	delete(s.draws)
	s^ = {}
}

// The camera's view and projection for a `width x height` target.
camera_matrices :: proc "contextless" (c: Camera, width, height: int) -> (view, proj: Mat4) {
	up := c.up if c.up != {} else Vec3{0, 1, 0}
	view = look_at(c.position, c.target, up)
	aspect := f32(max(width, 1)) / f32(max(height, 1))
	near := c.near if c.near > 0 else 0.1
	far := c.far if c.far > near else near * 1e4
	if c.orthographic {
		proj = orthographic(c.height if c.height > 0 else 2, aspect, near, far)
	} else {
		proj = perspective(c.fov if c.fov > 0 else 50, aspect, near, far)
	}
	return
}

// The order to draw in: the opaque draws as listed, then the blended ones
// (transparent with opacity under 1) farthest from the camera first.
draw_order :: proc(scene: ^Scene, view: Mat4, order: ^[dynamic]int) {
	clear(order)
	for d, i in scene.draws {
		if !blended(d.material) {
			append(order, i)
		}
	}
	first := len(order)
	for d, i in scene.draws {
		if blended(d.material) {
			append(order, i)
		}
	}
	depth :: proc(d: Draw, view: Mat4) -> f32 {
		return (view * d.model * Vec4{0, 0, 0, 1}).z
	}
	back := order[first:]
	for i in 1 ..< len(back) {
		for j := i; j > 0 && depth(scene.draws[back[j]], view) < depth(scene.draws[back[j - 1]], view); j -= 1 {
			back[j], back[j - 1] = back[j - 1], back[j]
		}
	}
}

blended :: proc "contextless" (m: Material) -> bool {
	return m.transparent && m.opacity < 1
}

// ---- shading ----------------------------------------------------------------------------

// The lights in shading form: linear colours times intensity, directions
// normalized. Both renderers shade with these (the GPU gets them packed).
Shading_Light :: struct {
	kind:     Light_Kind,
	color:    Vec3, // linear, times intensity
	ground:   Vec3, // linear, times intensity (hemisphere)
	// Directional: the direction toward the light; point: its position.
	vector:   Vec3,
	distance: f32,
	decay:    f32,
}

MAX_LIGHTS :: 8

shading_lights :: proc(lights: []Light, out: ^[MAX_LIGHTS]Shading_Light) -> int {
	n := 0
	for l in lights {
		if n == MAX_LIGHTS {
			break
		}
		sl := Shading_Light{kind = l.kind, color = linear(l.color) * l.intensity, ground = linear(l.ground) * l.intensity, distance = l.distance, decay = l.decay}
		switch l.kind {
		case .Directional:
			d := l.position - l.target
			sl.vector = linalg.normalize(d) if linalg.length(d) > 1e-9 else Vec3{0, 1, 0}
		case .Point:
			sl.vector = l.position
		case .Ambient, .Hemisphere:
		}
		out[n] = sl
		n += 1
	}
	return n
}

// The colour (linear) leaving a point of `m` at `world` with normal `n`
// toward the eye: three.js's MeshStandardMaterial (Lambert diffuse, GGX
// specular with Smith correlated visibility and Schlick Fresnel), direct
// lights plus ambient and hemisphere light; no environment map.
shade :: proc "contextless" (m: ^Material, base: Vec3, world, n_in, eye: Vec3, lights: []Shading_Light) -> Vec3 {
	if m.basic {
		return base
	}
	n := n_in
	v := linalg.normalize(eye - world)
	if linalg.dot(n, v) < 0 && m.side != .Front {
		n = -n
	}
	diffuse := base * (1 - m.metalness)
	f0 := linalg.lerp(Vec3{0.04, 0.04, 0.04}, base, m.metalness)
	roughness := clamp(m.roughness, 0.0525, 1)
	alpha := roughness * roughness
	nv := clamp(linalg.dot(n, v), 1e-4, 1)
	out: Vec3
	for &l in lights {
		switch l.kind {
		case .Ambient:
			out += l.color * diffuse / math.PI
		case .Hemisphere:
			w := 0.5 * n.y + 0.5
			out += linalg.lerp(l.ground, l.color, w) * diffuse / math.PI
		case .Directional, .Point:
			ldir := l.vector
			light := l.color
			if l.kind == .Point {
				d := l.vector - world
				dist := linalg.length(d)
				ldir = d / max(dist, 1e-6)
				decay := l.decay if l.decay > 0 else 2
				falloff := 1 / max(math.pow(dist, decay), 0.01)
				if l.distance > 0 {
					t := clamp(1 - math.pow(dist / l.distance, 4), 0, 1)
					falloff *= t * t
				}
				light *= falloff
			}
			nl := clamp(linalg.dot(n, ldir), 0, 1)
			if nl <= 0 {
				continue
			}
			irradiance := light * nl
			h := linalg.normalize(ldir + v)
			nh := clamp(linalg.dot(n, h), 0, 1)
			vh := clamp(linalg.dot(v, h), 0, 1)
			// Schlick Fresnel, Smith-GGX correlated visibility, GGX distribution.
			fresnel := f0 + (1 - f0) * math.pow(1 - vh, 5)
			a2 := alpha * alpha
			gv := nl * math.sqrt(a2 + (1 - a2) * nv * nv)
			gl := nv * math.sqrt(a2 + (1 - a2) * nl * nl)
			vis := 0.5 / max(gv + gl, 1e-6)
			dd := nh * nh * (a2 - 1) + 1
			dist := a2 / (math.PI * dd * dd)
			out += irradiance * (diffuse / math.PI + fresnel * (vis * dist))
		}
	}
	return out
}
