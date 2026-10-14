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
	distance:    f32,
	decay:       f32,
	// Directional lights: casts a shadow (three.js's castShadow; the first
	// such light does, shadow.odin) with these settings.
	cast_shadow: bool,
	shadow:      Light_Shadow,
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
	// Drawn into the shadow map, and darkened where it is shadowed: three.js
	// puts castShadow and receiveShadow on the mesh, and a draw's material
	// is its own.
	Cast_Shadow,
	Receive_Shadow,
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
	color_map:          Texture_Id,
	normal_map:         Texture_Id,
	roughness_map:      Texture_Id, // its green channel (three.js's roughnessMap)
	emissive_map:       Texture_Id,
	metalness_map:      Texture_Id, // its blue channel (metalnessMap; glTF packs both in one)
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

MAX_JOINTS :: 64

Joint_Range :: struct {
	first, count: u32,
}

MATERIAL_DEFAULT :: Material{color = {1, 1, 1}, metalness = 0, roughness = 1, emissive_intensity = 1, opacity = 1}

Draw :: struct {
	geometry: Geometry_Id,
	model:    Mat4,
	material: Material,
}

Scene :: struct {
	camera:                Camera,
	lights:                [dynamic]Light,
	draws:                 #soa[dynamic]Draw,
	// Equirectangular environment map (image-based lighting) and background.
	environment:           Texture_Id,
	environment_intensity: f32, // 0 means 1
	background:            Texture_Id,
	background_intensity:  f32, // 0 means 1
	background_blurriness: f32, // 0..1
	// ShaderMaterials' uniforms: each a name and its floats'
	// uniform_floats[first:][:count].
	uniforms:              [dynamic]struct {
		name:         string,
		first, count: u32,
	},
	uniform_floats:        [dynamic]f32,
	// Skinned draws' joint matrices: draw i's are joints[draw_joints[i].first:][:draw_joints[i].count].
	draw_joints:           [dynamic]Joint_Range,
	joints:                [dynamic]Mat4,
}

scene_clear :: proc(s: ^Scene) {
	s.camera = CAMERA_DEFAULT
	s.environment = 0
	s.environment_intensity = 0
	s.background = 0
	s.background_intensity = 0
	s.background_blurriness = 0
	clear(&s.lights)
	clear(&s.draws)
	clear(&s.uniforms)
	clear(&s.uniform_floats)
	clear(&s.draw_joints)
	clear(&s.joints)
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

// Keeps joint matrices in `s.joints` this frame.
scene_joints :: proc(s: ^Scene, matrices: []Mat4) -> Joint_Range {
	if len(matrices) == 0 {
		return {}
	}
	first := len(s.joints)
	table_grow_non_zero(&s.joints, first + len(matrices))
	copy(s.joints[first:], matrices)
	return {u32(first), u32(len(matrices))}
}

// Associates `r` with draw `di` for this frame.
scene_set_draw_joints :: proc(s: ^Scene, di: int, r: Joint_Range) {
	if r.count == 0 {
		return
	}
	if len(s.draw_joints) <= di {
		old := len(s.draw_joints)
		table_grow_non_zero(&s.draw_joints, di + 1)
		for i in old ..< di {
			s.draw_joints[i] = {}
		}
	}
	s.draw_joints[di] = r
}

// Draw `di`'s joint matrices (nil when unskinned).
scene_draw_joints :: #force_inline proc "contextless" (s: ^Scene, di: int) -> []Mat4 {
	if di < 0 || di >= len(s.draw_joints) {
		return nil
	}
	r := s.draw_joints[di]
	if r.count == 0 || int(r.first + r.count) > len(s.joints) {
		return nil
	}
	return s.joints[r.first:][:r.count]
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
	delete(s.draw_joints)
	delete(s.joints)
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
	for m, i in materials {
		if !blended(m) && (m.shader == 0 || shader_get(shaders, m.shader) != nil) && draw_seen(geos, geometries[i], models[i], view_proj, scene_draw_joints(scene, i)) {
			append(order, i)
		}
	}
	first := len(order)
	for m, i in materials {
		if blended(m) && (m.shader == 0 || shader_get(shaders, m.shader) != nil) && draw_seen(geos, geometries[i], models[i], view_proj, scene_draw_joints(scene, i)) {
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

// Blended over what is behind: transparent, with opacity under 1 or alpha
// from elsewhere (a colour map's, a ShaderMaterial's), as three.js blends
// any transparent material.
blended :: proc "contextless" (m: Material) -> bool {
	return .Transparent in m.flags && (m.opacity < 1 || m.color_map != 0 || m.shader != 0)
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
	// The shadow light: its direct light is scaled by the shadow factor.
	shadowed: bool,
}

MAX_LIGHTS :: 8

shading_lights :: proc(lights: []Light, out: ^[MAX_LIGHTS]Shading_Light) -> int {
	n := 0
	shadow := shadow_light(lights)
	for l, i in lights {
		if n == MAX_LIGHTS {
			break
		}
		sl := Shading_Light{kind = l.kind, color = linear(l.color) * l.intensity, ground = linear(l.ground) * l.intensity, distance = l.distance, decay = l.decay, shadowed = i == shadow}
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
	base:          Vec3, // linear
	emissive:      Vec3, // linear, times intensity
	opacity:       f32, // 1 unless blended
	metalness:     f32,
	roughness:     f32,
	flags:         Shading_Flags,
	color_map:     Texture_Id,
	normal_map:    Texture_Id,
	roughness_map: Texture_Id,
	emissive_map:  Texture_Id,
	metalness_map: Texture_Id,
	shader:        Shader_Id, // 0: the standard material
}

// Bit values are the shader's FLAG_* constants (o3d_test checks).
Shading_Flag :: enum u8 {
	Basic,
	Flat_Shading,
	// Lit from both sides: the side is not .Front.
	Two_Sided,
	Blended,
	Receive_Shadow,
	Color_Map,
	Normal_Map,
	Roughness_Map,
	Emissive_Map,
	Metalness_Map,
}

Shading_Flags :: bit_set[Shading_Flag; u32]

TEXTURE_FLAGS :: Shading_Flags{.Color_Map, .Normal_Map, .Roughness_Map, .Emissive_Map, .Metalness_Map}

// From the draws' material column: `scene_materials(scene)`.
shading_materials :: proc(materials: []Material, out: ^[dynamic]Shading_Material, textures: ^Textures = nil) {
	table_grow_non_zero(out, len(materials))
	has_tex :: #force_inline proc "contextless" (t: ^Textures, id: Texture_Id) -> bool {
		_, ok := texture_get(t, id)
		return ok
	}
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
		if .Receive_Shadow in m.flags {
			f += {.Receive_Shadow}
		}
		if has_tex(textures, m.color_map) {
			f += {.Color_Map}
		}
		if has_tex(textures, m.normal_map) && .Basic not_in m.flags {
			f += {.Normal_Map}
		}
		if has_tex(textures, m.roughness_map) && .Basic not_in m.flags {
			f += {.Roughness_Map}
		}
		if has_tex(textures, m.emissive_map) {
			f += {.Emissive_Map}
		}
		if has_tex(textures, m.metalness_map) && .Basic not_in m.flags {
			f += {.Metalness_Map}
		}
		out[i] = {
			base          = linear(m.color),
			emissive      = linear(m.emissive) * m.emissive_intensity,
			opacity       = clamp(m.opacity, 0, 1) if .Blended in f else 1,
			metalness     = m.metalness,
			roughness     = m.roughness,
			flags         = f,
			color_map     = m.color_map,
			normal_map    = m.normal_map,
			roughness_map = m.roughness_map,
			emissive_map  = m.emissive_map,
			metalness_map = m.metalness_map,
			shader        = m.shader,
		}
	}
}

// The colour (linear) leaving a point of `m` at `world` with normal `n`
// toward the eye: three.js's MeshStandardMaterial (Lambert diffuse, GGX
// specular with Smith correlated visibility and Schlick Fresnel), direct
// lights plus ambient, hemisphere and equirectangular environment lighting.
// `shadow` scales the shadow light's direct light (shadow_factor; 1: lit).
shade :: proc "contextless" (m: ^Shading_Material, world, n_in, eye: Vec3, lights: []Shading_Light, shadow: f32 = 1, env: Shading_Environment = {}) -> Vec3 {
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
			if l.shadowed {
				light *= shadow
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
	if env.radiance != 0 {
		r := 2 * linalg.dot(n, v) * n - v
		irr := linear(sample_lod(env.textures, env.irradiance, equirect_uv(n), 0).rgb)
		rad := linear(sample_lod(env.textures, env.radiance, equirect_uv(r), roughness * env.max_lod).rgb)
		c0 :: Vec4{-1, -0.0275, -0.572, 0.022}
		c1 :: Vec4{1, 0.0425, 1.04, -0.04}
		rv := roughness * c0 + c1
		a004 := min(rv.x * rv.x, linalg.exp2(-9.28 * nv)) * rv.x + rv.y
		fab := Vec2{-1.04, 1.04} * a004 + rv.zw
		spec := f0 * fab.x + fab.y
		out += env.intensity * (irr * diffuse * (1 - spec) + rad * spec)
	}
	return out
}
