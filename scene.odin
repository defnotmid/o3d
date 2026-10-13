package o3d

import "core:math"
import "core:math/linalg"

// A frame's scene: one camera, the lights, and draws (geometry, its world
// transform, its material). Filled from scratch every frame; the draws keep
// their order (transparent ones are drawn after the opaque ones, farthest
// first). The draws are a structure of arrays: the per-frame passes
// (`draw_order`, `shading_materials`) read one or two of its columns.

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

// The values are the shader's LIGHT_* constants (o3d_test checks).
Light_Kind :: enum u8 {
	Ambient     = 0, // AmbientLight(color, intensity)
	Hemisphere  = 1, // HemisphereLight(skyColor, groundColor, intensity)
	Directional = 2, // DirectionalLight(color, intensity): from `position` toward `target`
	Point       = 3, // PointLight(color, intensity, distance, decay)
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

// three.js's boolean material properties.
Material_Flag :: enum u8 {
	// Unlit: the colour as is (MeshBasicMaterial).
	Basic,
	// One normal per triangle (flatShading).
	Flat_Shading,
	// Blended over what is behind (opacity < 1 needs it, as in three.js).
	Transparent,
}

Material_Flags :: bit_set[Material_Flag; u8]

// MeshStandardMaterial (or MeshBasicMaterial with `.Basic`), or a
// ShaderMaterial with `shader` (shader.odin): then only `side`,
// `.Transparent` and `uniforms` apply.
Material :: struct {
	color:              Vec3, // sRGB
	metalness:          f32,
	roughness:          f32,
	emissive:           Vec3, // sRGB
	emissive_intensity: f32,
	opacity:            f32,
	side:               Side,
	flags:              Material_Flags,
	shader:             Shader_Id,
	// The shader's uniforms, from `scene_uniforms`.
	uniforms:           Uniform_Range,
}

// A ShaderMaterial uniform, three.js's `uniforms: { name: { value } }`: its
// float components (vectors; matrices column by column; arrays element by
// element; ints and bools converted). The name is borrowed until the frame
// is drawn.
Uniform :: struct {
	name:  string,
	value: []f32,
}

Uniform_Range :: struct {
	first, count: u32,
}

MATERIAL_DEFAULT :: Material{color = {1, 1, 1}, metalness = 0, roughness = 1, emissive_intensity = 1, opacity = 1}

Draw :: struct {
	geometry: Geometry_Id,
	model:    Mat4,
	material: Material,
}

Scene :: struct {
	camera:         Camera,
	lights:         [dynamic]Light,
	draws:          #soa[dynamic]Draw,
	// ShaderMaterials' uniforms: each a name and its floats'
	// uniform_floats[first:][:count].
	uniforms:       [dynamic]struct {
		name:         string,
		first, count: u32,
	},
	uniform_floats: [dynamic]f32,
}

scene_clear :: proc(s: ^Scene) {
	s.camera = CAMERA_DEFAULT
	clear(&s.lights)
	clear(&s.draws)
	clear(&s.uniforms)
	clear(&s.uniform_floats)
}

// Keeps uniforms for a material (`Material.uniforms`) this frame.
scene_uniforms :: proc(s: ^Scene, values: ..Uniform) -> Uniform_Range {
	r := Uniform_Range{u32(len(s.uniforms)), u32(len(values))}
	for v in values {
		append(&s.uniforms, type_of(s.uniforms[0]){v.name, u32(len(s.uniform_floats)), u32(len(v.value))})
		append(&s.uniform_floats, ..v.value)
	}
	return r
}

// The draws' materials, one column of the table.
scene_materials :: proc "contextless" (s: ^Scene) -> []Material {
	return s.draws.material[:len(s.draws)]
}

scene_destroy :: proc(s: ^Scene) {
	delete(s.lights)
	delete(s.draws)
	delete(s.uniforms)
	delete(s.uniform_floats)
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

// The order to draw in: the draws that can be seen (with geometry, and a
// compiled shader if they ask for one, the bounding sphere at least partly
// in the view), the opaque ones as listed, then the blended ones farthest
// from the camera first.
draw_order :: proc(scene: ^Scene, geos: ^Geometries, shaders: ^Shaders, view, proj: Mat4, order: ^[dynamic]int) {
	clear(order)
	n := len(scene.draws)
	materials := scene_materials(scene)
	models := scene.draws.model[:n]
	geometries := scene.draws.geometry[:n]
	view_proj := proj * view
	seen :: proc(geos: ^Geometries, id: Geometry_Id, model, view_proj: Mat4) -> bool {
		geo := geometry_get(geos, id) or_return
		if geo.index_count == 0 {
			return false
		}
		scale := max(linalg.length(Vec3{model[0, 0], model[1, 0], model[2, 0]}), linalg.length(Vec3{model[0, 1], model[1, 1], model[2, 1]}), linalg.length(Vec3{model[0, 2], model[1, 2], model[2, 2]}))
		return sphere_visible(view_proj, {model[0, 3], model[1, 3], model[2, 3]}, geo.radius * scale)
	}
	for m, i in materials {
		if !blended(m) && (m.shader == 0 || shader_get(shaders, m.shader) != nil) && seen(geos, geometries[i], models[i], view_proj) {
			append(order, i)
		}
	}
	first := len(order)
	for m, i in materials {
		if blended(m) && (m.shader == 0 || shader_get(shaders, m.shader) != nil) && seen(geos, geometries[i], models[i], view_proj) {
			append(order, i)
		}
	}
	depth :: proc(model, view: Mat4) -> f32 {
		return (view * model * Vec4{0, 0, 0, 1}).z
	}
	back := order[first:]
	for i in 1 ..< len(back) {
		for j := i; j > 0 && depth(models[back[j]], view) < depth(models[back[j - 1]], view); j -= 1 {
			back[j], back[j - 1] = back[j - 1], back[j]
		}
	}
}

// Whether a sphere (world centre and radius) is at least partly inside the
// view volume of `view_proj` (-w <= x, y <= w, 0 <= z <= w), tested against
// its six planes.
sphere_visible :: proc "contextless" (view_proj: Mat4, centre: Vec3, radius: f32) -> bool {
	row :: #force_inline proc "contextless" (m: Mat4, r: int) -> Vec4 {
		return {m[r, 0], m[r, 1], m[r, 2], m[r, 3]}
	}
	x, y, z, w := row(view_proj, 0), row(view_proj, 1), row(view_proj, 2), row(view_proj, 3)
	for plane in ([6]Vec4{w + x, w - x, w + y, w - y, z, w - z}) {
		if linalg.dot(plane.xyz, centre) + plane.w < -radius * linalg.length(plane.xyz) {
			return false
		}
	}
	return true
}

// Blended over what is behind: transparent, with opacity under 1 (a
// ShaderMaterial's alpha is its own).
blended :: proc "contextless" (m: Material) -> bool {
	return .Transparent in m.flags && (m.opacity < 1 || m.shader != 0)
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

// The materials in shading form, one per draw: linear colours, the opacity
// that is drawn and the flags both renderers' shading reads. Both build it
// once per frame (the GPU's uniforms are these, packed).
Shading_Material :: struct {
	base:      Vec3, // linear
	emissive:  Vec3, // linear, times intensity
	opacity:   f32, // 1 unless blended
	metalness: f32,
	roughness: f32,
	flags:     Shading_Flags,
	shader:    Shader_Id, // 0: the standard material
}

// Bit values are the shader's FLAG_* constants (o3d_test checks).
Shading_Flag :: enum u8 {
	Basic,
	Flat_Shading,
	// Lit from both sides: the side is not .Front.
	Two_Sided,
	Blended,
}

Shading_Flags :: bit_set[Shading_Flag; u32]

// From the draws' material column: `scene_materials(scene)`.
shading_materials :: proc(materials: []Material, out: ^[dynamic]Shading_Material) {
	resize(out, len(materials))
	for m, i in materials {
		f: Shading_Flags
		if .Basic in m.flags {
			f += {.Basic}
		}
		if .Flat_Shading in m.flags {
			f += {.Flat_Shading}
		}
		if m.side != .Front {
			f += {.Two_Sided}
		}
		if blended(m) {
			f += {.Blended}
		}
		out[i] = {
			base      = linear(m.color),
			emissive  = linear(m.emissive) * m.emissive_intensity,
			opacity   = clamp(m.opacity, 0, 1) if .Blended in f else 1,
			metalness = m.metalness,
			roughness = m.roughness,
			flags     = f,
			shader    = m.shader,
		}
	}
}

// The colour (linear) leaving a point of `m` at `world` with normal `n`
// toward the eye: three.js's MeshStandardMaterial (Lambert diffuse, GGX
// specular with Smith correlated visibility and Schlick Fresnel), direct
// lights plus ambient and hemisphere light; no environment map.
shade :: proc "contextless" (m: ^Shading_Material, world, n_in, eye: Vec3, lights: []Shading_Light) -> Vec3 {
	base := m.base
	if .Basic in m.flags {
		return base
	}
	n := n_in
	v := linalg.normalize(eye - world)
	if linalg.dot(n, v) < 0 && .Two_Sided in m.flags {
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
