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
	Spot        = 4, // SpotLight(color, intensity, distance, angle, penumbra, decay): from `position` toward `target`
}

Light :: struct {
	kind:        Light_Kind,
	color:       Vec3, // sRGB (the sky's for a hemisphere light)
	ground:      Vec3, // a hemisphere light's ground colour, sRGB
	intensity:   f32,
	position:    Vec3,
	target:      Vec3,
	// Point and spot lights: the range (0: none) and the falloff exponent
	// (2 is physical, three.js's default).
	distance:    f32,
	decay:       f32,
	// Spot lights: cone half-angle in radians (0: PI / 3, max PI / 2) and
	// soft edge fraction (0..1; 0 is sharp).
	angle:       f32,
	penumbra:    f32,
	// Directional and spot lights: casts a shadow (three.js's castShadow;
	// up to MAX_SHADOWS such lights do, shadow.odin) with these settings.
	cast_shadow: bool,
	shadow:      Light_Shadow,
}

// three.js's Fog (linear from `near` to `far`) and FogExp2 (exponential-squared
// with `density`), blending drawn surfaces toward `color` by view distance.
Fog_Kind :: enum u8 {
	None   = 0,
	Linear = 1, // Fog(color, near, far)
	Exp2   = 2, // FogExp2(color, density)
}

Fog :: struct {
	kind:      Fog_Kind,
	color:     Vec3, // sRGB
	near, far: f32, // Linear (far <= near defaults to 1..1000)
	density:   f32, // Exp2 (0 defaults to 0.00025)
}

Side :: enum u8 {
	Front,
	Back,
	Double,
}

// three.js's boolean material properties.
Material_Flag :: enum u8 {
	// Unlit: the colour as is (MeshBasicMaterial, LineBasicMaterial, PointsMaterial).
	Basic,
	// Cel-shaded diffuse bands + stepped specular highlight (MeshToonMaterial).
	Toon,
	// One normal per triangle (flatShading).
	Flat_Shading,
	// Blended over what is behind (opacity < 1 needs it, as in three.js).
	Transparent,
	// Drawn into the shadow map, and darkened where it is shadowed: three.js
	// puts castShadow and receiveShadow on the mesh, and a draw's material
	// is its own.
	Cast_Shadow,
	Receive_Shadow,
	// Discards fragments with alpha below `alpha_cutoff` (glTF alphaMode: MASK,
	// three.js alphaTest), including in the shadow pass.
	Alpha_Mask,
	// LineSegments / Points: `size` is in screen pixels rather than world
	// units (three.js's `sizeAttenuation: false`).
	No_Size_Attenuation,
}

Material_Flags :: bit_set[Material_Flag; u16]

Texture_Slot :: enum u8 {
	Color     = 0,
	Normal    = 1,
	Roughness = 2,
	Emissive  = 3,
	Metalness = 4,
	Ao        = 5,
}

// UV set (0 or 1) and KHR_texture_transform for a texture slot; zero is the
// identity transform on UV set 0 (`scale = {0, 0}` means `{1, 1}`).
Texture_Transform :: struct {
	offset:    Vec2,
	rotation:  f32,
	scale:     Vec2,
	tex_coord: u8,
}

// MeshStandardMaterial (or MeshBasicMaterial with `.Basic`, MeshToonMaterial
// with `.Toon`), or a ShaderMaterial with `shader` (shader.odin): then only
// `side`, `.Transparent` and `uniforms` apply.
Material :: struct {
	color:               Vec3, // sRGB
	metalness:           f32,
	roughness:           f32,
	emissive:            Vec3, // sRGB
	emissive_intensity:  f32,
	opacity:             f32,
	alpha_cutoff:        f32,
	ao_map_intensity:    f32,
	// Line width (`.Lines`) or point size (`.Points`); 0 takes defaults
	// (0.03 world / 2 px for lines, 0.08 world / 4 px for points).
	size:                f32,
	// Inverted-hull cartoon outline thickness in world units (> 0 appends a
	// back-face extruded shell draw) and sRGB colour (default black).
	outline_width:       f32,
	outline_color:       Vec3,
	mode:                Draw_Mode,
	side:                Side,
	flags:               Material_Flags,
	color_map:           Texture_Id,
	normal_map:          Texture_Id,
	roughness_map:       Texture_Id, // its green channel (three.js's roughnessMap)
	emissive_map:        Texture_Id,
	metalness_map:       Texture_Id, // its blue channel (metalnessMap; glTF packs both in one)
	ao_map:              Texture_Id, // its red channel (three.js's aoMap; indirect light only)
	color_transform:     Texture_Transform,
	normal_transform:    Texture_Transform,
	roughness_transform: Texture_Transform,
	emissive_transform:  Texture_Transform,
	metalness_transform: Texture_Transform,
	ao_transform:        Texture_Transform,
	shader:              Shader_Id,
	// The shader's uniforms, from `scene_uniforms`.
	uniforms:            Uniform_Range,
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

Morph_Range :: struct {
	first, count: u32,
}

// One instance of an instanced draw (three.js's InstancedMesh): its local
// transform (multiplied on the right of `Draw.model`) and optional sRGB tint
// (`color != {}` or `has_color` multiplies `Material.color` in linear space).
Instance :: struct {
	transform: Mat4,
	color:     Vec3,
	has_color: bool,
}

Instance_Range :: struct {
	first, count: u32,
}

MATERIAL_DEFAULT :: Material{
	color              = {1, 1, 1},
	metalness          = 0,
	roughness          = 1,
	emissive_intensity = 1,
	opacity            = 1,
	alpha_cutoff       = 0.5,
	ao_map_intensity   = 1,
}

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
	// Tone mapping (default .None) and exposure (0 means 1), plus scene fog.
	tone_mapping:          Tone_Mapping,
	tone_mapping_exposure: f32,
	fog:                   Fog,
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
	// Morphed draws' target weights: draw i's are morph_weights[draw_morphs[i].first:][:draw_morphs[i].count].
	draw_morphs:           [dynamic]Morph_Range,
	// Which instance each draw is (0-based), filled by expand_draws; a draw
	// past its end is instance 0.
	draw_instance_ids:     [dynamic]i32,
	morph_weights:         [dynamic]f32,
	// Instanced draws' instances: draw i's are instances[draw_instances[i].first:][:draw_instances[i].count].
	draw_instances:        [dynamic]Instance_Range,
	instances:             [dynamic]Instance,
	// Per-frame line/point geometries (SCENE_GEO_BIT + index).
	path_geos:             [dynamic]Geometry,
	path_vertices:         [dynamic]Vertex,
	path_indices:          [dynamic]u32,
}

scene_clear :: proc(s: ^Scene) {
	s.camera = CAMERA_DEFAULT
	s.environment = 0
	s.environment_intensity = 0
	s.background = 0
	s.background_intensity = 0
	s.background_blurriness = 0
	s.tone_mapping = .None
	s.tone_mapping_exposure = 0
	s.fog = {}
	clear(&s.lights)
	clear(&s.draws)
	clear(&s.uniforms)
	clear(&s.uniform_floats)
	clear(&s.draw_joints)
	clear(&s.joints)
	clear(&s.draw_morphs)
	clear(&s.draw_instance_ids)
	clear(&s.morph_weights)
	clear(&s.draw_instances)
	clear(&s.instances)
	clear(&s.path_geos)
	clear(&s.path_vertices)
	clear(&s.path_indices)
}

// Per-frame LineSegments geometry in `s` from pairs of endpoints `(positions[2*i], positions[2*i+1])`.
scene_line_segments :: proc(s: ^Scene, positions: []Vec3) -> Geometry_Id {
	n := (len(positions) / 2) * 2
	if n < 2 {
		return 0
	}
	first_v := u32(len(s.path_vertices))
	first_i := u32(len(s.path_indices))
	table_grow(&s.path_vertices, int(first_v) + n)
	table_grow_non_zero(&s.path_indices, int(first_i) + n)
	r: f32
	for i in 0 ..< n {
		p := positions[i]
		s.path_vertices[int(first_v) + i] = {position = p, normal = {0, 0, 1}}
		s.path_indices[int(first_i) + i] = u32(i)
		r = max(r, linalg.length(p))
	}
	id := SCENE_GEO_BIT + Geometry_Id(len(s.path_geos))
	append(&s.path_geos, Geometry{
		first_vertex = first_v,
		vertex_count = u32(n),
		first_index  = first_i,
		index_count  = u32(n),
		mode         = .Lines,
		radius       = r,
	})
	return id
}

// Per-frame connected polyline geometry in `s` as LineSegments.
scene_polyline :: proc(s: ^Scene, positions: []Vec3, closed := false) -> Geometry_Id {
	n := len(positions)
	if n < 2 {
		return 0
	}
	segs := n if closed else n - 1
	first_v := u32(len(s.path_vertices))
	first_i := u32(len(s.path_indices))
	table_grow(&s.path_vertices, int(first_v) + n)
	table_grow_non_zero(&s.path_indices, int(first_i) + segs * 2)
	r: f32
	for p, i in positions {
		s.path_vertices[int(first_v) + i] = {position = p, normal = {0, 0, 1}}
		r = max(r, linalg.length(p))
	}
	for i in 0 ..< segs {
		s.path_indices[int(first_i) + i * 2] = u32(i)
		s.path_indices[int(first_i) + i * 2 + 1] = u32((i + 1) % n)
	}
	id := SCENE_GEO_BIT + Geometry_Id(len(s.path_geos))
	append(&s.path_geos, Geometry{
		first_vertex = first_v,
		vertex_count = u32(n),
		first_index  = first_i,
		index_count  = u32(segs * 2),
		mode         = .Lines,
		radius       = r,
	})
	return id
}

// Per-frame TubeGeometry in `s`: a tube of `radius` swept along `points`
// (`radial_segments` around, its frame carried along the curve without
// twisting), capped unless `closed`. Triangles, lit and shaded like any mesh.
scene_tube :: proc(s: ^Scene, points: []Vec3, radius: f32 = 0.08, radial_segments := 12, closed := false) -> Geometry_Id {
	n := len(points)
	if n < 2 {
		return 0
	}
	rs := max(radial_segments, 3)
	r := max(radius, 1e-5)
	rings := n + (1 if closed else 0)
	caps := 0 if closed else 2
	first_v := u32(len(s.path_vertices))
	first_i := u32(len(s.path_indices))
	n_verts := rings * (rs + 1) + caps
	n_idx := (rings - 1) * rs * 6 + caps * rs * 3
	table_grow_non_zero(&s.path_vertices, int(first_v) + n_verts)
	table_grow_non_zero(&s.path_indices, int(first_i) + n_idx)
	verts := s.path_vertices[first_v:][:n_verts]
	idx := s.path_indices[first_i:][:n_idx]
	tangent :: proc(points: []Vec3, i: int, closed: bool) -> Vec3 {
		n := len(points)
		prev := points[(i - 1 + n) % n] if closed else points[max(i - 1, 0)]
		next := points[(i + 1) % n] if closed else points[min(i + 1, n - 1)]
		d := next - prev
		l := linalg.length(d)
		return d / l if l > 1e-6 else {0, 1, 0}
	}
	t0 := tangent(points, 0, closed)
	normal := linalg.normalize(linalg.cross(Vec3{0, 1, 0} if abs(t0.y) < 0.9 else Vec3{1, 0, 0}, t0))
	rad: f32
	for i in 0 ..< rings {
		p := points[i % n]
		t := tangent(points, i % n, closed)
		// The previous ring's normal, turned onto this tangent's plane.
		if proj := normal - t * linalg.dot(normal, t); linalg.length(proj) > 1e-5 {
			normal = linalg.normalize(proj)
		}
		binormal := linalg.cross(t, normal)
		for k in 0 ..= rs {
			u := f32(k) / f32(rs)
			nv := normal * math.cos(u * 2 * math.PI) + binormal * math.sin(u * 2 * math.PI)
			verts[i * (rs + 1) + k] = {position = p + nv * r, normal = nv, uv = {u, f32(i) / f32(rings - 1)}}
		}
		rad = max(rad, linalg.length(p) + r)
	}
	w := 0
	for i in 0 ..< rings - 1 {
		row0, row1 := u32(i * (rs + 1)), u32((i + 1) * (rs + 1))
		for k in 0 ..< rs {
			a, b := row0 + u32(k), row0 + u32(k + 1)
			c, d := row1 + u32(k), row1 + u32(k + 1)
			idx[w], idx[w + 1], idx[w + 2], idx[w + 3], idx[w + 4], idx[w + 5] = a, b, c, b, d, c
			w += 6
		}
	}
	if !closed {
		start, end := u32(rings * (rs + 1)), u32(rings * (rs + 1) + 1)
		last := u32((rings - 1) * (rs + 1))
		verts[start] = {position = points[0], normal = -t0, uv = {0.5, 0}}
		verts[end] = {position = points[n - 1], normal = tangent(points, n - 1, false), uv = {0.5, 1}}
		for k in 0 ..< rs {
			idx[w], idx[w + 1], idx[w + 2] = start, u32(k + 1), u32(k)
			idx[w + 3], idx[w + 4], idx[w + 5] = end, last + u32(k), last + u32(k + 1)
			w += 6
		}
	}
	id := SCENE_GEO_BIT + Geometry_Id(len(s.path_geos))
	append(&s.path_geos, Geometry{
		first_vertex = first_v,
		vertex_count = u32(n_verts),
		first_index  = first_i,
		index_count  = u32(n_idx),
		mode         = .Triangles,
		radius       = rad,
	})
	return id
}

// Per-frame Points geometry in `s` from 3D positions.
scene_points :: proc(s: ^Scene, positions: []Vec3) -> Geometry_Id {
	n := len(positions)
	if n == 0 {
		return 0
	}
	first_v := u32(len(s.path_vertices))
	first_i := u32(len(s.path_indices))
	table_grow(&s.path_vertices, int(first_v) + n)
	table_grow_non_zero(&s.path_indices, int(first_i) + n)
	r: f32
	for p, i in positions {
		s.path_vertices[int(first_v) + i] = {position = p, normal = {0, 0, 1}}
		s.path_indices[int(first_i) + i] = u32(i)
		r = max(r, linalg.length(p))
	}
	id := SCENE_GEO_BIT + Geometry_Id(len(s.path_geos))
	append(&s.path_geos, Geometry{
		first_vertex = first_v,
		vertex_count = u32(n),
		first_index  = first_i,
		index_count  = u32(n),
		mode         = .Points,
		radius       = r,
	})
	return id
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

// Keeps morph target weights in `s.morph_weights` this frame.
scene_morphs :: proc(s: ^Scene, weights: []f32) -> Morph_Range {
	has_nonzero := false
	for w in weights {
		if w != 0 {
			has_nonzero = true
			break
		}
	}
	if !has_nonzero {
		return {}
	}
	first := len(s.morph_weights)
	table_grow_non_zero(&s.morph_weights, first + len(weights))
	copy(s.morph_weights[first:], weights)
	return {u32(first), u32(len(weights))}
}

// Associates morph weights `r` with draw `di` for this frame.
scene_set_draw_morphs :: proc(s: ^Scene, di: int, r: Morph_Range) {
	if r.count == 0 {
		return
	}
	if len(s.draw_morphs) <= di {
		old := len(s.draw_morphs)
		table_grow_non_zero(&s.draw_morphs, di + 1)
		for i in old ..< di {
			s.draw_morphs[i] = {}
		}
	}
	s.draw_morphs[di] = r
}

// Draw `di`'s morph target weights (nil when unmorphed).
scene_draw_morphs :: #force_inline proc "contextless" (s: ^Scene, di: int) -> []f32 {
	if di < 0 || di >= len(s.draw_morphs) {
		return nil
	}
	r := s.draw_morphs[di]
	if r.count == 0 || int(r.first + r.count) > len(s.morph_weights) {
		return nil
	}
	return s.morph_weights[r.first:][:r.count]
}

// Keeps instance transforms and colours in `s.instances` this frame.
scene_instances :: proc(s: ^Scene, items: []Instance) -> Instance_Range {
	if len(items) == 0 {
		return {}
	}
	first := len(s.instances)
	table_grow_non_zero(&s.instances, first + len(items))
	copy(s.instances[first:], items)
	return {u32(first), u32(len(items))}
}

// Associates instances `r` with draw `di` for this frame.
scene_set_draw_instances :: proc(s: ^Scene, di: int, r: Instance_Range) {
	if r.count == 0 {
		return
	}
	if len(s.draw_instances) <= di {
		old := len(s.draw_instances)
		table_grow_non_zero(&s.draw_instances, di + 1)
		for i in old ..< di {
			s.draw_instances[i] = {}
		}
	}
	s.draw_instances[di] = r
}

// Draw `di`'s instances (nil when not instanced).
scene_draw_instances :: #force_inline proc "contextless" (s: ^Scene, di: int) -> []Instance {
	if di < 0 || di >= len(s.draw_instances) {
		return nil
	}
	r := s.draw_instances[di]
	if r.count == 0 || int(r.first + r.count) > len(s.instances) {
		return nil
	}
	return s.instances[r.first:][:r.count]
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
	delete(s.draw_morphs)
	delete(s.draw_instance_ids)
	delete(s.morph_weights)
	delete(s.draw_instances)
	delete(s.instances)
	delete(s.path_geos)
	delete(s.path_vertices)
	delete(s.path_indices)
	s^ = {}
}

// Per-frame expanded draws and camera-facing quad geometries for instanced
// draws, LineSegments, and Points, shared by the software renderer and
// o3d/sdlgpu (zero warm-frame allocations).
Expanded_Draws :: struct {
	scene:       Scene,
	geos:        [dynamic]Geometry,
	vertices:    [dynamic]Vertex,
	indices:     [dynamic]u32,
	posed_verts: [dynamic]Vertex,
}

expanded_draws_destroy :: proc(e: ^Expanded_Draws) {
	scene_destroy(&e.scene)
	delete(e.geos)
	delete(e.vertices)
	delete(e.indices)
	delete(e.posed_verts)
	e^ = {}
}

geometry_get_any :: proc "contextless" (g: ^Geometries, exp: ^Expanded_Draws, id: Geometry_Id) -> (Geometry, bool) {
	if id >= DYNAMIC_GEO_BIT {
		idx := int(id - DYNAMIC_GEO_BIT)
		if exp == nil || idx < 0 || idx >= len(exp.geos) {
			return {}, false
		}
		return exp.geos[idx], true
	}
	return geometry_get(g, id)
}

geometry_mesh_any :: proc "contextless" (g: ^Geometries, exp: ^Expanded_Draws, id: Geometry_Id) -> (vertices: []Vertex, indices: []u32, ok: bool) {
	if id >= DYNAMIC_GEO_BIT {
		geo := geometry_get_any(g, exp, id) or_return
		return exp.vertices[geo.first_vertex:][:geo.vertex_count], exp.indices[geo.first_index:][:geo.index_count], true
	}
	return geometry_mesh(g, id)
}

@(private = "file")
scene_geometry_get :: proc "contextless" (scene: ^Scene, g: ^Geometries, id: Geometry_Id) -> (Geometry, bool) {
	if id >= SCENE_GEO_BIT && id < DYNAMIC_GEO_BIT {
		idx := int(id - SCENE_GEO_BIT)
		if idx < 0 || idx >= len(scene.path_geos) {
			return {}, false
		}
		return scene.path_geos[idx], true
	}
	return geometry_get(g, id)
}

@(private = "file")
scene_geometry_mesh :: proc "contextless" (scene: ^Scene, g: ^Geometries, id: Geometry_Id) -> (vertices: []Vertex, indices: []u32, ok: bool) {
	if id >= SCENE_GEO_BIT && id < DYNAMIC_GEO_BIT {
		geo := scene_geometry_get(scene, g, id) or_return
		return scene.path_vertices[geo.first_vertex:][:geo.vertex_count], scene.path_indices[geo.first_index:][:geo.index_count], true
	}
	return geometry_mesh(g, id)
}

@(private = "file")
effective_draw_mode :: #force_inline proc "contextless" (m: Material, geo: Geometry) -> Draw_Mode {
	return m.mode if m.mode != .Triangles else geo.mode
}

// Expands instanced draws into individual draws, inverted-hull outlines
// (`outline_width > 0`) into back-face extruded shells, `.Lines` / `.Points`
// draws into camera-facing quad triangles, and `.Curves` draws' quads grown
// by half a pixel for the view (`dilate_curve_quads`) in `exp`, returning
// `scene` unchanged when no draw uses instancing, outlines, lines, points
// or curves.
expand_draws :: proc(exp: ^Expanded_Draws, scene: ^Scene, geos: ^Geometries, view, proj: Mat4, width, height: int) -> ^Scene {
	clear(&exp.geos)
	clear(&exp.vertices)
	clear(&exp.indices)
	need_expand := len(scene.instances) > 0 || len(scene.path_geos) > 0
	if !need_expand {
		for i in 0 ..< len(scene.draws) {
			d := &scene.draws[i]
			if geo, ok := scene_geometry_get(scene, geos, d.geometry); ok {
				if effective_draw_mode(d.material, geo) != .Triangles || (d.material.outline_width > 0 && d.material.side == .Front) {
					need_expand = true
					break
				}
			}
		}
	}
	if !need_expand {
		return scene
	}
	out := &exp.scene
	scene_clear(out)
	out.camera = scene.camera
	out.environment = scene.environment
	out.environment_intensity = scene.environment_intensity
	out.background = scene.background
	out.background_intensity = scene.background_intensity
	out.background_blurriness = scene.background_blurriness
	out.tone_mapping = scene.tone_mapping
	out.tone_mapping_exposure = scene.tone_mapping_exposure
	out.fog = scene.fog
	append(&out.lights, ..scene.lights[:])
	append(&out.uniforms, ..scene.uniforms[:])
	append(&out.uniform_floats, ..scene.uniform_floats[:])
	append(&out.joints, ..scene.joints[:])
	append(&out.morph_weights, ..scene.morph_weights[:])

	view_proj := proj * view
	cam_right := Vec3{view[0, 0], view[0, 1], view[0, 2]}
	cam_up := Vec3{view[1, 0], view[1, 1], view[1, 2]}
	cam_back := Vec3{view[2, 0], view[2, 1], view[2, 2]} // points toward camera in view space
	scale_px := 2.0 / (f32(max(height, 1)) * max(proj[1, 1], 1e-6))

	for di in 0 ..< len(scene.draws) {
		d := scene.draws[di]
		geo, ok := scene_geometry_get(scene, geos, d.geometry)
		if !ok {
			continue
		}
		j_range := scene.draw_joints[di] if di < len(scene.draw_joints) else Joint_Range{}
		m_range := scene.draw_morphs[di] if di < len(scene.draw_morphs) else Morph_Range{}
		insts := scene_draw_instances(scene, di)
		mode := effective_draw_mode(d.material, geo)
		if mode == .Curves && geo.mode != .Curves {
			// Asked of geometry without curves: its triangles as they are.
			mode = .Triangles
			d.material.mode = .Triangles
		}
		if mode == .Curves {
			// The cover quads grown for this view, per instance.
			src_v, src_i, _ := scene_geometry_mesh(scene, geos, d.geometry)
			default_inst := [1]Instance{{transform = linalg.MATRIX4F32_IDENTITY}}
			run_insts := insts if len(insts) > 0 else default_inst[:]
			base_lin := linear(d.material.color)
			for inst in run_insts {
				M := d.model * inst.transform if len(insts) > 0 else d.model
				first_v := u32(len(exp.vertices))
				first_i := u32(len(exp.indices))
				table_grow_non_zero(&exp.vertices, int(first_v) + len(src_v))
				dst := exp.vertices[first_v:]
				dilate_curve_quads(src_v, dst, view_proj * M, width, height)
				r_max: f32
				for v in dst {
					r_max = max(r_max, linalg.length(v.position))
				}
				append(&exp.indices, ..src_i)
				dyn := geo
				dyn.first_vertex, dyn.first_index = first_v, first_i
				dyn.skinned, dyn.morph_targets, dyn.radius = false, 0, r_max
				append(&exp.geos, dyn)
				mat := d.material
				mat.mode = .Curves
				if len(insts) > 0 && (inst.has_color || inst.color != {}) {
					mat.color = linear_to_srgb_vec3(base_lin * linear(inst.color))
				}
				append(&out.draws, Draw{geometry = DYNAMIC_GEO_BIT + Geometry_Id(len(exp.geos) - 1), model = M, material = mat})
			}
			continue
		}
		if mode == .Triangles {
			src_geo := d.geometry
			if d.geometry >= SCENE_GEO_BIT {
				// A per-frame mesh (scene_tube): into the expanded geometry the
				// renderers draw.
				src_v, src_i, _ := scene_geometry_mesh(scene, geos, d.geometry)
				first_v := u32(len(exp.vertices))
				first_i := u32(len(exp.indices))
				append(&exp.vertices, ..src_v)
				append(&exp.indices, ..src_i)
				d.geometry = DYNAMIC_GEO_BIT + Geometry_Id(len(exp.geos))
				append(&exp.geos, Geometry{first_vertex = first_v, vertex_count = u32(len(src_v)), first_index = first_i, index_count = u32(len(src_i)), radius = geo.radius})
			}
			outline_geo_id := Geometry_Id(0)
			if d.material.outline_width > 0 && d.material.side == .Front {
				raw_verts, raw_idx, m_ok := scene_geometry_mesh(scene, geos, src_geo)
				if m_ok && len(raw_verts) > 0 && len(raw_idx) > 0 {
					skin := geometry_skin(geos, d.geometry) if d.geometry < SCENE_GEO_BIT else nil
					joints := scene_draw_joints(scene, di)
					skinned := len(skin) > 0 && len(joints) > 0
					active_buf: [MAX_MORPH_TARGETS]Active_Morph = ---
					na := active_morphs(geo, scene_draw_morphs(scene, di), &active_buf) if d.geometry < SCENE_GEO_BIT else 0
					morphed := na > 0 && geos != nil && len(geos.morph_deltas) > 0
					deltas := geos.morph_deltas[:] if geos != nil else nil
					m_scale := max(mat_scale(d.model), 1e-4)
					local_off := d.material.outline_width / m_scale
					first_v := u32(len(exp.vertices))
					first_i := u32(len(exp.indices))
					table_grow_non_zero(&exp.vertices, int(first_v) + len(raw_verts))
					table_grow_non_zero(&exp.indices, int(first_i) + len(raw_idx))
					r_max: f32
					for vtx, vi in raw_verts {
						v := vtx
						if morphed {
							v.position, v.normal = morph_vertex(v, u32(vi), deltas, active_buf[:na])
						}
						if skinned {
							v.position, v.normal = skin_vertex(v, skin[vi], joints)
						}
						nlen := linalg.length(v.normal)
						n_dir := v.normal / nlen if nlen > 1e-6 else Vec3{0, 1, 0}
						v.position += n_dir * local_off
						exp.vertices[int(first_v) + vi] = v
						r_max = max(r_max, linalg.length(v.position))
					}
					copy(exp.indices[first_i:], raw_idx)
					outline_geo_id = DYNAMIC_GEO_BIT + Geometry_Id(len(exp.geos))
					append(&exp.geos, Geometry{
						first_vertex = first_v,
						vertex_count = u32(len(raw_verts)),
						first_index  = first_i,
						index_count  = u32(len(raw_idx)),
						radius       = r_max,
					})
				}
			}
			outline_mat := MATERIAL_DEFAULT
			outline_mat.color = d.material.outline_color
			outline_mat.opacity = d.material.opacity
			outline_mat.side = .Back
			outline_mat.flags = {.Basic} + ({.Transparent} if .Transparent in d.material.flags else {})
			if len(insts) == 0 {
				ei := len(out.draws)
				append(&out.draws, d)
				scene_set_draw_joints(out, ei, j_range)
				scene_set_draw_morphs(out, ei, m_range)
				if outline_geo_id != 0 {
					append(&out.draws, Draw{geometry = outline_geo_id, model = d.model, material = outline_mat})
				}
			} else {
				base_lin := linear(d.material.color)
				for inst, k in insts {
					idraw := d
					idraw.model = d.model * inst.transform
					if inst.has_color || inst.color != {} {
						idraw.material.color = linear_to_srgb_vec3(base_lin * linear(inst.color))
					}
					ei := len(out.draws)
					append(&out.draws, idraw)
					table_grow(&out.draw_instance_ids, ei + 1)
					out.draw_instance_ids[ei] = i32(k)
					scene_set_draw_joints(out, ei, j_range)
					scene_set_draw_morphs(out, ei, m_range)
					if outline_geo_id != 0 {
						append(&out.draws, Draw{geometry = outline_geo_id, model = idraw.model, material = outline_mat})
					}
				}
			}
			continue
		}

		// .Lines or .Points: pose local vertices (morph, then skin) once per draw.
		raw_verts, raw_idx, _ := scene_geometry_mesh(scene, geos, d.geometry)
		skin := geometry_skin(geos, d.geometry) if d.geometry < SCENE_GEO_BIT else nil
		joints := scene_draw_joints(scene, di)
		skinned := len(skin) > 0 && len(joints) > 0
		active_buf: [MAX_MORPH_TARGETS]Active_Morph = ---
		na := active_morphs(geo, scene_draw_morphs(scene, di), &active_buf) if d.geometry < SCENE_GEO_BIT else 0
		morphed := na > 0 && geos != nil && len(geos.morph_deltas) > 0
		src_verts := raw_verts
		if morphed || skinned {
			table_grow_non_zero(&exp.posed_verts, len(raw_verts))
			deltas := geos.morph_deltas[:] if geos != nil else nil
			for vtx, vi in raw_verts {
				v := vtx
				if morphed {
					v.position, v.normal = morph_vertex(v, u32(vi), deltas, active_buf[:na])
				}
				if skinned {
					v.position, v.normal = skin_vertex(v, skin[vi], joints)
				}
				exp.posed_verts[vi] = v
			}
			src_verts = exp.posed_verts[:len(raw_verts)]
		}

		default_inst := [1]Instance{{transform = linalg.MATRIX4F32_IDENTITY}}
		run_insts := insts if len(insts) > 0 else default_inst[:]
		base_lin := linear(d.material.color)
		px_space := .No_Size_Attenuation in d.material.flags

		for inst in run_insts {
			M := d.model * inst.transform if len(insts) > 0 else d.model
			m_scale := mat_scale(M)
			first_v := u32(len(exp.vertices))
			first_i := u32(len(exp.indices))
			r_max: f32
			if mode == .Points {
				raw_s := d.material.size if d.material.size > 0 else (4.0 if px_space else 0.08)
				for v in src_verts {
					p := (M * Vec4{v.position.x, v.position.y, v.position.z, 1}).xyz
					half: f32
					if px_space {
						w_clip := (view_proj * Vec4{p.x, p.y, p.z, 1}).w if !scene.camera.orthographic else 1
						half = 0.5 * raw_s * max(w_clip, 1e-4) * scale_px
					} else {
						half = 0.5 * raw_s * m_scale
					}
					rx := cam_right * half
					uy := cam_up * half
					q0 := p - rx - uy
					q1 := p + rx - uy
					q2 := p + rx + uy
					q3 := p - rx + uy
					base_u := u32(len(exp.vertices)) - first_v
					append(
						&exp.vertices,
						Vertex{q0, cam_back, {0, 0}, {}},
						Vertex{q1, cam_back, {1, 0}, {}},
						Vertex{q2, cam_back, {1, 1}, {}},
						Vertex{q3, cam_back, {0, 1}, {}},
					)
					append(&exp.indices, base_u + 0, base_u + 1, base_u + 2, base_u + 0, base_u + 2, base_u + 3)
					r_max = max(r_max, linalg.length(p) + half * 1.5)
				}
			} else {
				raw_w := d.material.size if d.material.size > 0 else (2.0 if px_space else 0.03)
				emit_seg :: proc(exp: ^Expanded_Draws, first_v: u32, p0, p1, cam_back, cam_up, cam_right: Vec3, view_proj: Mat4, ortho, px_space: bool, raw_w, m_scale, scale_px: f32, r_max: ^f32) {
					seg := p1 - p0
					seg_len := linalg.length(seg)
					if seg_len < 1e-7 {
						return
					}
					dir := seg / seg_len
					side_v := linalg.cross(dir, cam_back)
					if linalg.length(side_v) < 1e-5 {
						side_v = linalg.cross(dir, cam_up)
						if linalg.length(side_v) < 1e-5 {
							side_v = cam_right
						}
					}
					side_dir := linalg.normalize(side_v)
					n := linalg.normalize(linalg.cross(side_dir, dir))
					h0, h1: f32
					if px_space {
						w0 := (view_proj * Vec4{p0.x, p0.y, p0.z, 1}).w if !ortho else 1
						w1 := (view_proj * Vec4{p1.x, p1.y, p1.z, 1}).w if !ortho else 1
						h0 = 0.5 * raw_w * max(w0, 1e-4) * scale_px
						h1 = 0.5 * raw_w * max(w1, 1e-4) * scale_px
					} else {
						h0 = 0.5 * raw_w * m_scale
						h1 = h0
					}
					q0 := p0 - side_dir * h0
					q1 := p0 + side_dir * h0
					q2 := p1 + side_dir * h1
					q3 := p1 - side_dir * h1
					base_u := u32(len(exp.vertices)) - first_v
					append(
						&exp.vertices,
						Vertex{q0, n, {0, 0}, {}},
						Vertex{q1, n, {1, 0}, {}},
						Vertex{q2, n, {1, 1}, {}},
						Vertex{q3, n, {0, 1}, {}},
					)
					append(&exp.indices, base_u + 0, base_u + 1, base_u + 2, base_u + 0, base_u + 2, base_u + 3)
					r_max^ = max(r_max^, linalg.length(p0) + h0, linalg.length(p1) + h1)
				}
				if geo.mode == .Lines {
					for s := 0; s + 1 < len(raw_idx); s += 2 {
						ia, ib := raw_idx[s], raw_idx[s + 1]
						if int(ia) >= len(src_verts) || int(ib) >= len(src_verts) {
							continue
						}
						p0 := (M * Vec4{src_verts[ia].position.x, src_verts[ia].position.y, src_verts[ia].position.z, 1}).xyz
						p1 := (M * Vec4{src_verts[ib].position.x, src_verts[ib].position.y, src_verts[ib].position.z, 1}).xyz
						emit_seg(exp, first_v, p0, p1, cam_back, cam_up, cam_right, view_proj, scene.camera.orthographic, px_space, raw_w, m_scale, scale_px, &r_max)
					}
				} else {
					for t := 0; t + 2 < len(raw_idx); t += 3 {
						i0, i1, i2 := raw_idx[t], raw_idx[t + 1], raw_idx[t + 2]
						if int(i0) >= len(src_verts) || int(i1) >= len(src_verts) || int(i2) >= len(src_verts) {
							continue
						}
						p0 := (M * Vec4{src_verts[i0].position.x, src_verts[i0].position.y, src_verts[i0].position.z, 1}).xyz
						p1 := (M * Vec4{src_verts[i1].position.x, src_verts[i1].position.y, src_verts[i1].position.z, 1}).xyz
						p2 := (M * Vec4{src_verts[i2].position.x, src_verts[i2].position.y, src_verts[i2].position.z, 1}).xyz
						emit_seg(exp, first_v, p0, p1, cam_back, cam_up, cam_right, view_proj, scene.camera.orthographic, px_space, raw_w, m_scale, scale_px, &r_max)
						emit_seg(exp, first_v, p1, p2, cam_back, cam_up, cam_right, view_proj, scene.camera.orthographic, px_space, raw_w, m_scale, scale_px, &r_max)
						emit_seg(exp, first_v, p2, p0, cam_back, cam_up, cam_right, view_proj, scene.camera.orthographic, px_space, raw_w, m_scale, scale_px, &r_max)
					}
				}
			}
			v_count := u32(len(exp.vertices)) - first_v
			i_count := u32(len(exp.indices)) - first_i
			if i_count == 0 {
				continue
			}
			dyn_id := DYNAMIC_GEO_BIT + Geometry_Id(len(exp.geos))
			append(&exp.geos, Geometry{
				first_vertex = first_v,
				vertex_count = v_count,
				first_index  = first_i,
				index_count  = i_count,
				radius       = r_max,
			})
			mat := d.material
			mat.mode = .Triangles
			mat.side = .Double
			if len(insts) > 0 && (inst.has_color || inst.color != {}) {
				mat.color = linear_to_srgb_vec3(base_lin * linear(inst.color))
			}
			append(&out.draws, Draw{geometry = dyn_id, model = linalg.MATRIX4F32_IDENTITY, material = mat})
		}
	}
	return out
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
draw_order :: proc(scene: ^Scene, geos: ^Geometries, shaders: ^Shaders, view, proj: Mat4, order: ^[dynamic]int, exp: ^Expanded_Draws = nil) {
	clear(order)
	n := len(scene.draws)
	materials := scene_materials(scene)
	models := scene.draws.model[:n]
	geometries := scene.draws.geometry[:n]
	view_proj := proj * view
	for m, i in materials {
		if !blended(m) && (m.shader == 0 || shader_get(shaders, m.shader) != nil) && draw_seen(geos, geometries[i], models[i], view_proj, scene_draw_joints(scene, i), scene_draw_morphs(scene, i), exp) {
			append(order, i)
		}
	}
	first := len(order)
	for m, i in materials {
		if blended(m) && (m.shader == 0 || shader_get(shaders, m.shader) != nil) && draw_seen(geos, geometries[i], models[i], view_proj, scene_draw_joints(scene, i), scene_draw_morphs(scene, i), exp) {
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
// from elsewhere (a colour map's, a ShaderMaterial's, an exact shape's
// coverage), as three.js blends any transparent material.
blended :: proc "contextless" (m: Material) -> bool {
	return .Transparent in m.flags && (m.opacity < 1 || m.color_map != 0 || m.shader != 0 || m.mode == .Curves)
}

// ---- shading ----------------------------------------------------------------------------

// The lights in shading form: linear colours times intensity, directions
// normalized. Both renderers shade with these (the GPU gets them packed).
Shading_Light :: struct {
	kind:      Light_Kind,
	color:     Vec3, // linear, times intensity
	// Hemisphere: ground linear times intensity; Spot: unit axis toward the light.
	ground:    Vec3,
	// Directional: the direction toward the light; point/spot: its position.
	vector:    Vec3,
	distance:  f32,
	decay:     f32,
	cos_outer: f32,
	cos_inner: f32,
	// 0: not shadowed; 1..MAX_SHADOWS: shadow slot index + 1.
	shadowed:  u32,
}

MAX_LIGHTS :: 8

shading_lights :: proc(lights: []Light, out: ^[MAX_LIGHTS]Shading_Light) -> int {
	n := 0
	shadow_count: u32 = 0
	for l in lights {
		if n == MAX_LIGHTS {
			break
		}
		shadowed: u32 = 0
		if (l.kind == .Directional || l.kind == .Spot) && l.cast_shadow && shadow_count < MAX_SHADOWS {
			shadow_count += 1
			shadowed = shadow_count
		}
		sl := Shading_Light{
			kind     = l.kind,
			color    = linear(l.color) * l.intensity,
			ground   = linear(l.ground) * l.intensity,
			distance = l.distance,
			decay    = l.decay,
			shadowed = shadowed,
		}
		switch l.kind {
		case .Directional:
			d := l.position - l.target
			sl.vector = linalg.normalize(d) if linalg.length(d) > 1e-9 else Vec3{0, 1, 0}
		case .Point:
			sl.vector = l.position
		case .Spot:
			sl.vector = l.position
			d := l.position - l.target
			sl.ground = linalg.normalize(d) if linalg.length(d) > 1e-9 else Vec3{0, 1, 0}
			angle := clamp(l.angle if l.angle > 0 else math.PI / 3, 1e-4, math.PI / 2)
			penumbra := clamp(l.penumbra, 0, 1)
			sl.cos_outer = math.cos(angle)
			sl.cos_inner = math.cos(angle * (1 - penumbra))
		case .Ambient, .Hemisphere:
		}
		out[n] = sl
		n += 1
	}
	return n
}

// Fog blend factor 0..1 at Euclidean camera distance `dist` (`shaders/o3d.slang`).
fog_factor :: #force_inline proc "contextless" (fog: Fog, dist: f32) -> f32 {
	switch fog.kind {
	case .None:
		return 0
	case .Linear:
		near := fog.near if fog.far > fog.near else 1
		far := fog.far if fog.far > fog.near else 1000
		return clamp((dist - near) / max(far - near, 1e-6), 0, 1)
	case .Exp2:
		d := fog.density if fog.density > 0 else 0.00025
		dd := d * dist
		return clamp(1 - math.exp(-dd * dd), 0, 1)
	}
	return 0
}

// Applies tone mapping and fog in linear space before sRGB encoding, matching
// `fs_main` in shaders/o3d.slang.
finish_color :: #force_inline proc "contextless" (c, world, eye: Vec3, tone_mapping: Tone_Mapping, exposure: f32, fog: Fog, fog_color_linear: Vec3) -> Vec3 {
	out := tone_map(c, tone_mapping, exposure)
	if fog.kind != .None {
		out = linalg.lerp(out, fog_color_linear, fog_factor(fog, linalg.length(world - eye)))
	}
	return out
}

// The materials in shading form, one per draw: linear colours, the opacity
// that is drawn and the flags both renderers' shading reads. Both build it
// once per frame (the GPU's uniforms are these, packed).
Shading_Material :: struct {
	base:          Vec3, // linear
	emissive:      Vec3, // linear, times intensity
	opacity:       f32, // 1 unless blended or alpha-masked
	alpha_cutoff:  f32,
	metalness:     f32,
	roughness:     f32,
	ao_intensity:  f32,
	flags:         Shading_Flags,
	uv_sets:       u32, // bit per Texture_Slot: 1 when reading uv1
	uv_transforms: [Texture_Slot][2]Vec3,
	color_map:     Texture_Id,
	normal_map:    Texture_Id,
	roughness_map: Texture_Id,
	emissive_map:  Texture_Id,
	metalness_map: Texture_Id,
	ao_map:        Texture_Id,
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
	Ao_Map,
	Alpha_Mask,
	Toon,
	// An exact shape (`Draw_Mode.Curves`): coverage from its curves, by
	// area when blended, by sample otherwise.
	Curves,
}

Shading_Flags :: bit_set[Shading_Flag; u32]

TEXTURE_FLAGS :: Shading_Flags{.Color_Map, .Normal_Map, .Roughness_Map, .Emissive_Map, .Metalness_Map, .Ao_Map}

// 2x3 affine matrix in o3d's UV space (v = 1 at top) for KHR_texture_transform
// (defined in glTF's top-left UV space, v_gltf = 1 - v_o3d).
uv_transform :: proc "contextless" (t: Texture_Transform) -> [2]Vec3 {
	sx := t.scale.x if t.scale != {} else 1
	sy := t.scale.y if t.scale != {} else 1
	c, s := math.cos(t.rotation), math.sin(t.rotation)
	return {
		{c * sx, -s * sy, t.offset.x + s * sy},
		{s * sx,  c * sy, 1 - t.offset.y - c * sy},
	}
}

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
		if .Toon in m.flags {
			f += {.Toon}
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
		if .Alpha_Mask in m.flags {
			f += {.Alpha_Mask}
		}
		if m.mode == .Curves && m.shader == 0 {
			f += {.Curves}
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
		if has_tex(textures, m.ao_map) && .Basic not_in m.flags {
			f += {.Ao_Map}
		}
		tforms := [Texture_Slot]Texture_Transform{
			.Color     = m.color_transform,
			.Normal    = m.normal_transform,
			.Roughness = m.roughness_transform,
			.Emissive  = m.emissive_transform,
			.Metalness = m.metalness_transform,
			.Ao        = m.ao_transform,
		}
		uv_sets: u32
		uv_xforms: [Texture_Slot][2]Vec3
		for slot in Texture_Slot {
			tf := tforms[slot]
			if tf.tex_coord != 0 {
				uv_sets |= 1 << u32(slot)
			}
			uv_xforms[slot] = uv_transform(tf)
		}
		out[i] = {
			base          = linear(m.color),
			emissive      = linear(m.emissive) * m.emissive_intensity,
			opacity       = clamp(m.opacity, 0, 1) if (.Blended in f || .Alpha_Mask in f) else 1,
			alpha_cutoff  = m.alpha_cutoff,
			metalness     = m.metalness,
			roughness     = m.roughness,
			ao_intensity  = m.ao_map_intensity,
			flags         = f,
			uv_sets       = uv_sets,
			uv_transforms = uv_xforms,
			color_map     = m.color_map,
			normal_map    = m.normal_map,
			roughness_map = m.roughness_map,
			emissive_map  = m.emissive_map,
			metalness_map = m.metalness_map,
			ao_map        = m.ao_map,
			shader        = m.shader,
		}
	}
}

@(private = "file")
toon_step :: #force_inline proc "contextless" (edge0, edge1, x: f32) -> f32 {
	t := clamp((x - edge0) / (edge1 - edge0), 0, 1)
	return t * t * (3 - 2 * t)
}

// The colour (linear) leaving a point of `m` at `world` with normal `n`
// toward the eye: three.js's MeshStandardMaterial (Lambert diffuse, GGX
// specular with Smith correlated visibility and Schlick Fresnel) or
// MeshToonMaterial (`.Toon`: 3-band quantized cel-shading + stepped highlight),
// direct lights plus ambient, hemisphere and equirectangular environment lighting.
// `shadow` / `shadows` scale each shadow light's direct light (`shadow_factor`;
// 1: lit); `ao` scales indirect light (ambient, hemisphere, environment).
shade :: proc "contextless" (m: ^Shading_Material, world, n_in, eye: Vec3, lights: []Shading_Light, shadow: f32 = 1, env: Shading_Environment = {}, ao: f32 = 1, shadows: [MAX_SHADOWS]f32 = {1, 1, 1, 1}) -> Vec3 {
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
			out += l.color * diffuse * (ao / math.PI)
		case .Hemisphere:
			w := 0.5 * n.y + 0.5
			out += linalg.lerp(l.ground, l.color, w) * diffuse * (ao / math.PI)
		case .Directional, .Point, .Spot:
			ldir := l.vector
			light := l.color
			if l.kind == .Point || l.kind == .Spot {
				d := l.vector - world
				dist := linalg.length(d)
				ldir = d / max(dist, 1e-6)
				decay := l.decay if l.decay > 0 else 2
				falloff := 1 / max(math.pow(dist, decay), 0.01)
				if l.distance > 0 {
					t := clamp(1 - math.pow(dist / l.distance, 4), 0, 1)
					falloff *= t * t
				}
				if l.kind == .Spot {
					angle_cos := linalg.dot(ldir, l.ground)
					spot_t := clamp((angle_cos - l.cos_outer) / max(l.cos_inner - l.cos_outer, 1e-5), 0, 1)
					falloff *= spot_t * spot_t * (3 - 2 * spot_t)
				}
				if falloff <= 0 {
					continue
				}
				light *= falloff
			}
			if l.shadowed > 0 {
				light *= shadow * shadows[min(int(l.shadowed - 1), MAX_SHADOWS - 1)]
			}
			if .Toon in m.flags {
				raw_nl := clamp(linalg.dot(n, ldir), 0, 1)
				toon_nl := 0.22 + 0.46 * toon_step(0.12, 0.18, raw_nl) + 0.32 * toon_step(0.50, 0.58, raw_nl)
				spec_term := Vec3{}
				if raw_nl > 0.12 && roughness < 0.85 {
					h := linalg.normalize(ldir + v)
					nh := clamp(linalg.dot(n, h), 0, 1)
					thresh := 1.0 - (1.0 - roughness) * 0.18
					spec_s := toon_step(thresh - 0.02, thresh + 0.02, nh) * (1.0 - roughness) * 0.55
					spec_term = linalg.lerp(Vec3{1, 1, 1}, base, m.metalness) * spec_s
				}
				out += light * (diffuse * (toon_nl / math.PI) + spec_term)
				continue
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
		out += (ao * env.intensity) * (irr * diffuse * (1 - spec) + rad * spec)
	}
	return out
}

