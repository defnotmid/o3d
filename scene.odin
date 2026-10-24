package o3d

import "core:math"
import "core:math/linalg"
import "core:slice"

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
	// Only the shadows it receives (three.js's ShadowMaterial): `color` at
	// `opacity` times the light its shadows take, blended over what is
	// behind; it receives shadows, is lit by nothing and reads no maps.
	Shadow_Only,
	// Added to what is behind (three.js's AdditiveBlending: sparks,
	// flames, glows, atmospheres), not blended over it: any material,
	// drawn with the transparent ones (depth tested, not written), its
	// colour times its opacity added.
	Additive,
	// The `color_map`'s colour is premultiplied by its alpha (a canvas's,
	// a GPU-rendered image's): divided by it after filtering, both
	// renderers alike.
	Premultiplied_Color_Map,
	// Its geometry's vertex colours (`Vertex.color`) times its colour and
	// their alpha times its opacity (three.js's vertexColors; glTF's
	// COLOR_0): interpolated in linear light, lines and points too.
	Vertex_Colors,
}

Material_Flags :: bit_set[Material_Flag; u16]

Texture_Slot :: enum u8 {
	Color               = 0,
	Normal              = 1,
	Roughness           = 2,
	Emissive            = 3,
	Metalness           = 4,
	Ao                  = 5,
	Alpha               = 6,
	Clearcoat           = 7,
	Clearcoat_Roughness = 8,
	Clearcoat_Normal    = 9,
	Sheen_Color         = 10,
	Sheen_Roughness     = 11,
	Transmission        = 12,
	Thickness           = 13,
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
	// Particles: how far before an opaque surface they fade out (world
	// units; soft particles, by the frame's depth; 0: a hard edge).
	softness:            f32,
	// Particles: moved this far toward the camera along the ray to each
	// (world units; negative, away), as they look the same size in the
	// same place: a label pinned to a surface is drawn in front of it.
	depth_offset:        f32,
	// Particles: each faded by the share of a disc this many pixels in
	// radius about its position that the frame's depth leaves unhidden
	// (three.js's Lensflare: a sun's flare fades as a hull passes in front
	// of it; off the frame counts as hidden); 0: not faded. With a
	// `depth_offset` past the camera's distance the quads are drawn over
	// everything, as a flare's are, still probed at their positions.
	occlusion_radius:    f32,
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
	alpha_map:           Texture_Id, // its green channel times opacity (three.js's alphaMap; with .Transparent or .Alpha_Mask)
	// The normal map's tangent x and y scale (three.js's normalScale; 0:
	// 1, a negative y for a map with green down).
	normal_scale:        Vec2,
	color_transform:     Texture_Transform,
	normal_transform:    Texture_Transform,
	roughness_transform: Texture_Transform,
	emissive_transform:  Texture_Transform,
	metalness_transform: Texture_Transform,
	ao_transform:        Texture_Transform,
	alpha_transform:     Texture_Transform,
	// Where the maps are read: by the vertices' uvs, or projected along the
	// three axes and blended by the normal (triplanar mapping, for meshes
	// without uvs) from the object's position (the pattern moves with it)
	// or the world's. The slots' transforms apply to each projection.
	projection:          Map_Projection,
	// Triplanar: texture repeats per unit (0: 1), and how sharply the
	// projections hand over as the normal turns (the weights' power; 0: 4).
	projection_scale:    f32,
	projection_sharpness: f32,
	// MeshPhysicalMaterial's (glTF's KHR_materials_ior, _clearcoat,
	// _sheen): the index of refraction (0: 1.5), which sets a dielectric's
	// reflectance; a clear coat over the surface (0 to 1: none to full,
	// GGX with its own roughness, F0 0.04); sheen, a cloth's or velvet's
	// rim (sRGB, black: none; Charlie distribution with its roughness).
	ior:                 f32,
	clearcoat:           f32,
	clearcoat_roughness: f32,
	sheen_color:         Vec3, // sRGB
	sheen_roughness:     f32,
	// The layers' maps, three.js's: clear coat in red and its roughness in
	// green (times the factors), the coat's own normal map (and its scale,
	// 0: 1; without it the coat follows the surface's normal, not
	// `normal_map`), sheen colour in rgb (sRGB) and its roughness in alpha,
	// transmission in red, thickness in green. Each with its transform.
	clearcoat_map:                 Texture_Id,
	clearcoat_roughness_map:       Texture_Id,
	clearcoat_normal_map:          Texture_Id,
	clearcoat_normal_scale:        Vec2,
	sheen_color_map:               Texture_Id,
	sheen_roughness_map:           Texture_Id,
	transmission_map:              Texture_Id,
	thickness_map:                 Texture_Id,
	clearcoat_transform:           Texture_Transform,
	clearcoat_roughness_transform: Texture_Transform,
	clearcoat_normal_transform:    Texture_Transform,
	sheen_color_transform:         Texture_Transform,
	sheen_roughness_transform:     Texture_Transform,
	transmission_transform:        Texture_Transform,
	thickness_transform:           Texture_Transform,
	// Transmission (KHR_materials_transmission, _volume): how much light
	// passes through (0 to 1), seen through the surface as the frame drawn
	// before it (the opaque draws), refracted by the ior through
	// `thickness` (in the model's units; 0: thin), blurred by roughness,
	// tinted by the colour and absorbed toward `attenuation_color` (sRGB)
	// over `attenuation_distance` (0: none). three.js's
	// MeshPhysicalMaterial: transmissive draws come after the other opaque
	// ones, which they see; they do not see each other or transparent draws.
	transmission:        f32,
	thickness:           f32,
	attenuation_color:   Vec3,
	attenuation_distance: f32,
	shader:              Shader_Id,
	// The shader's uniforms, from `scene_uniforms`.
	uniforms:            Uniform_Range,
	// The shader's `uniform sampler2D`s by channel (`shader_map` finds a
	// sampler's by name): mipmapped, repeating; 0 samples white.
	shader_maps:         [SHADER_MAPS]Texture_Id,
}

Map_Projection :: enum u8 {
	Uv,
	Object,
	World,
}

// Where triplanar mapping reads a draw's positions and normals: its
// object's frame (`Map_Projection.Object`), or the world's (identity).
Projection_Frame :: struct {
	to_local:        Mat4, // world positions to the projection's
	normal_to_local: Mat3,
	normal_to_world: Mat3,
}

PROJECTION_FRAME_WORLD :: Projection_Frame{linalg.MATRIX4F32_IDENTITY, linalg.MATRIX3F32_IDENTITY, linalg.MATRIX3F32_IDENTITY}

// A draw's projection frame (both renderers).
projection_frame :: proc "contextless" (projection: Map_Projection, model: Mat4) -> Projection_Frame {
	if projection != .Object {
		return PROJECTION_FRAME_WORLD
	}
	inv := linalg.inverse(model)
	m3 := Mat3(model)
	return {to_local = inv, normal_to_local = linalg.transpose(m3), normal_to_world = linalg.transpose(Mat3(inv))}
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
	// What the frame is cleared to under everything (three.js's
	// scene.background as a Color, or the renderer's clear colour): sRGB
	// and alpha, written as is (not tone-mapped or fogged). The default,
	// alpha 0, leaves the frame transparent where nothing is drawn, to
	// composite over; alpha 1 makes it opaque. A `background` texture
	// covers it.
	background_color:      Vec4,
	// Tone mapping (default .None) and exposure (0 means 1), plus scene fog.
	tone_mapping:          Tone_Mapping,
	tone_mapping_exposure: f32,
	fog:                   Fog,
	// Also draw the frame's depth (`software_depth`, `o3d_sdlgpu.depth` /
	// `read_depth`): per pixel, the depth both renderers test (reversed: 1
	// near, 0 far, 0 where nothing is) of the nearest opaque draw at its centre, as
	// `camera_project` gives a point's. For post effects (depth of field,
	// haze) and to know whether something hides a point (a flare, a label).
	depth:                 bool,
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
	// Per-frame particles (`scene_particles`), by row.
	particles:             [dynamic]Particle,
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
	s.background_color = {}
	s.tone_mapping = .None
	s.tone_mapping_exposure = 0
	s.fog = {}
	s.depth = false
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
	clear(&s.particles)
	clear(&s.path_vertices)
	clear(&s.path_indices)
}

// Per-frame LineSegments geometry in `s` from pairs of endpoints
// `(positions[2*i], positions[2*i+1])`, and their colours (sRGB and alpha,
// one a position; for a `.Vertex_Colors` material) when given.
scene_line_segments :: proc(s: ^Scene, positions: []Vec3, colors: []Vec4 = nil) -> Geometry_Id {
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
		s.path_vertices[int(first_v) + i] = {position = p, normal = {0, 0, 1}, color = vertex_color(colors[i]) if i < len(colors) else {}}
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

// Per-frame connected polyline geometry in `s` as LineSegments, and its
// points' colours (sRGB and alpha; for a `.Vertex_Colors` material: a
// gradient along it) when given.
scene_polyline :: proc(s: ^Scene, positions: []Vec3, closed := false, colors: []Vec4 = nil) -> Geometry_Id {
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
		s.path_vertices[int(first_v) + i] = {position = p, normal = {0, 0, 1}, color = vertex_color(colors[i]) if i < len(colors) else {}}
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

// A particle: a camera-facing quad `size` (width, height: world units, or
// pixels with `.No_Size_Attenuation`; three.js's Sprite.scale) turned
// `rotation` radians in the view plane, its colour (sRGB) and opacity
// (clamped to 0..1) times its material's (and its `color_map`'s, the
// sprite). `offset` is the quad's centre from `position` in its width and
// height, x right and y up in the quad's turned frame: {0, 0} centred on
// it, {0, 0.5} its bottom edge's middle on it (a label's anchor), turning
// about it.
Particle :: struct {
	position: Vec3,
	size:     Vec2,
	color:    Vec4,
	rotation: f32,
	offset:   Vec2,
}

// Per-frame particles in `s`: draw the geometry with a transparent (or
// `.Additive`) material, its `color_map` the sprite, `softness` to fade
// them into what they meet. Drawn unlit, farthest first when blended.
scene_particles :: proc(s: ^Scene, particles: []Particle) -> Geometry_Id {
	if len(particles) == 0 {
		return 0
	}
	first := u32(len(s.particles))
	append(&s.particles, ..particles)
	r: f32
	for p in particles {
		r = max(r, linalg.length(p.position) + linalg.length(p.size) * (0.5 + linalg.length(p.offset)))
	}
	id := SCENE_GEO_BIT + Geometry_Id(len(s.path_geos))
	append(&s.path_geos, Geometry{vertex_count = u32(len(particles)), sprite_table = first, mode = .Sprites, radius = r})
	return id
}

// Per-frame Points geometry in `s` from 3D positions, and their colours
// (sRGB and alpha; for a `.Vertex_Colors` material) when given. Points of
// their own sizes are particles (`scene_particles`).
scene_points :: proc(s: ^Scene, positions: []Vec3, colors: []Vec4 = nil) -> Geometry_Id {
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
		s.path_vertices[int(first_v) + i] = {position = p, normal = {0, 0, 1}, color = vertex_color(colors[i]) if i < len(colors) else {}}
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
	delete(s.particles)
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
	// The drawn particle quads' rows, one a quad.
	sprite_rows: [dynamic]Sprite_Row,
	// Scratch: particles by distance, to draw the farthest first.
	sprite_keys: [dynamic]Sprite_Key,
}

// A drawn particle quad's colour (sRGB, times the material's) and opacity,
// and where its visibility is probed (`Material.occlusion_radius`): the
// particle's position on the frame (pixels from the top left), its depth,
// and the disc's radius (0: not probed).
Sprite_Row :: struct {
	color: Vec4,
	probe: Vec4,
}

@(private)
Sprite_Key :: struct {
	distance: f32,
	row:      u32,
}

expanded_draws_destroy :: proc(e: ^Expanded_Draws) {
	scene_destroy(&e.scene)
	delete(e.geos)
	delete(e.vertices)
	delete(e.indices)
	delete(e.posed_verts)
	delete(e.sprite_rows)
	delete(e.sprite_keys)
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

// A particle draw's quads (per instance), the farthest first unless they
// add, each with its row in `exp.sprite_rows`.
@(private = "file")
expand_particles :: proc(exp: ^Expanded_Draws, scene: ^Scene, d: Draw, geo: Geometry, insts: []Instance, view_proj: Mat4, cam_right, cam_up, cam_back: Vec3, scale_px: f32, frame: Vec2, draws: ^#soa[dynamic]Draw) {
	rows := scene.particles[geo.sprite_table:][:geo.vertex_count]
	px_space := .No_Size_Attenuation in d.material.flags
	offset := d.material.depth_offset
	eye := scene.camera.position
	near := scene.camera.near if scene.camera.near > 0 else 0.1
	probe_radius := max(d.material.occlusion_radius, 0)
	default_inst := [1]Instance{{transform = linalg.MATRIX4F32_IDENTITY}}
	run_insts := insts if len(insts) > 0 else default_inst[:]
	for inst in run_insts {
		M := d.model * inst.transform if len(insts) > 0 else d.model
		m_scale := mat_scale(M)
		// World positions, and the order to draw them in.
		clear(&exp.sprite_keys)
		for p, i in rows {
			w := (M * Vec4{p.position.x, p.position.y, p.position.z, 1}).xyz
			clip := view_proj * Vec4{w.x, w.y, w.z, 1}
			append(&exp.sprite_keys, Sprite_Key{distance = clip.w if !scene.camera.orthographic else -clip.z, row = u32(i)})
		}
		if .Additive not_in d.material.flags {
			slice.stable_sort_by(exp.sprite_keys[:], proc(a, b: Sprite_Key) -> bool {return a.distance > b.distance})
		}
		first_v := u32(len(exp.vertices))
		first_i := u32(len(exp.indices))
		first_row := u32(len(exp.sprite_rows))
		r_max: f32
		for key in exp.sprite_keys {
			p := rows[key.row]
			w := (M * Vec4{p.position.x, p.position.y, p.position.z, 1}).xyz
			half: Vec2
			if px_space {
				w_clip := (view_proj * Vec4{w.x, w.y, w.z, 1}).w if !scene.camera.orthographic else 1
				half = 0.5 * p.size * max(w_clip, 1e-4) * scale_px
			} else {
				half = 0.5 * p.size * m_scale
			}
			c, sn := math.cos(p.rotation), math.sin(p.rotation)
			rx := (cam_right * c + cam_up * sn) * half.x
			uy := (cam_up * c - cam_right * sn) * half.y
			centre := w + (rx * p.offset.x + uy * p.offset.y) * 2
			corners := [4]Vec3{centre - rx - uy, centre + rx - uy, centre + rx + uy, centre - rx + uy}
			if offset != 0 {
				if scene.camera.orthographic {
					for &v in corners {
						v += cam_back * offset
					}
				} else {
					// Scaled about the eye, so the same on screen; not past
					// the near plane.
					along := linalg.dot(eye - w, cam_back)
					k := max(1 - offset / max(linalg.length(w - eye), 1e-6), near * 1.001 / max(along, 1e-6))
					for &v in corners {
						v = eye + (v - eye) * k
					}
				}
			}
			base := u32(len(exp.vertices)) - first_v
			append(&exp.vertices, Vertex{position = corners[0], normal = cam_back, uv = {0, 0}}, Vertex{position = corners[1], normal = cam_back, uv = {1, 0}}, Vertex{position = corners[2], normal = cam_back, uv = {1, 1}}, Vertex{position = corners[3], normal = cam_back, uv = {0, 1}})
			append(&exp.indices, base, base + 1, base + 2, base, base + 2, base + 3)
			row := Sprite_Row{color = p.color}
			if probe_radius > 0 {
				// Behind the camera: off the frame.
				row.probe = {-1e6, -1e6, -1, probe_radius}
				clip := view_proj * Vec4{w.x, w.y, w.z, 1}
				if clip.w > 0 {
					row.probe.xyz = {(clip.x / clip.w * 0.5 + 0.5) * frame.x, (0.5 - clip.y / clip.w * 0.5) * frame.y, clip.z / clip.w}
				}
			}
			append(&exp.sprite_rows, row)
			for v in corners {
				r_max = max(r_max, linalg.length(v))
			}
		}
		append(&exp.geos, Geometry{first_vertex = first_v, vertex_count = u32(len(exp.vertices)) - first_v, first_index = first_i, index_count = u32(len(exp.indices)) - first_i, mode = .Sprites, sprite_table = first_row, radius = r_max})
		// Unlit, always blended (each particle's own opacity) or added.
		mat := d.material
		mat.mode = .Sprites
		mat.flags += {.Basic, .Transparent}
		mat.flags -= {.Vertex_Colors}
		if len(insts) > 0 && (inst.has_color || inst.color != {}) {
			mat.color = linear_to_srgb_vec3(linear(mat.color) * linear(inst.color))
		}
		append(draws, Draw{geometry = DYNAMIC_GEO_BIT + Geometry_Id(len(exp.geos) - 1), model = linalg.MATRIX4F32_IDENTITY, material = mat})
	}
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
	clear(&exp.sprite_rows)
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
	out.background_color = scene.background_color
	out.tone_mapping = scene.tone_mapping
	out.tone_mapping_exposure = scene.tone_mapping_exposure
	out.fog = scene.fog
	out.depth = scene.depth
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
		if mode == .Sprites {
			if geo.mode == .Sprites && d.geometry >= SCENE_GEO_BIT {
				expand_particles(exp, scene, d, geo, insts, view_proj, cam_right, cam_up, cam_back, scale_px, {f32(width), f32(height)}, &out.draws)
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
					c := v.color
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
						Vertex{q0, cam_back, {0, 0}, {}, c},
						Vertex{q1, cam_back, {1, 0}, {}, c},
						Vertex{q2, cam_back, {1, 1}, {}, c},
						Vertex{q3, cam_back, {0, 1}, {}, c},
					)
					append(&exp.indices, base_u + 0, base_u + 1, base_u + 2, base_u + 0, base_u + 2, base_u + 3)
					r_max = max(r_max, linalg.length(p) + half * 1.5)
				}
			} else {
				raw_w := d.material.size if d.material.size > 0 else (2.0 if px_space else 0.03)
				emit_seg :: proc(exp: ^Expanded_Draws, first_v: u32, p0, p1: Vec3, c0, c1: [4]u8, cam_back, cam_up, cam_right: Vec3, view_proj: Mat4, ortho, px_space: bool, raw_w, m_scale, scale_px: f32, r_max: ^f32) {
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
						Vertex{q0, n, {0, 0}, {}, c0},
						Vertex{q1, n, {1, 0}, {}, c0},
						Vertex{q2, n, {1, 1}, {}, c1},
						Vertex{q3, n, {0, 1}, {}, c1},
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
						emit_seg(exp, first_v, p0, p1, src_verts[ia].color, src_verts[ib].color, cam_back, cam_up, cam_right, view_proj, scene.camera.orthographic, px_space, raw_w, m_scale, scale_px, &r_max)
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
						emit_seg(exp, first_v, p0, p1, src_verts[i0].color, src_verts[i1].color, cam_back, cam_up, cam_right, view_proj, scene.camera.orthographic, px_space, raw_w, m_scale, scale_px, &r_max)
						emit_seg(exp, first_v, p1, p2, src_verts[i1].color, src_verts[i2].color, cam_back, cam_up, cam_right, view_proj, scene.camera.orthographic, px_space, raw_w, m_scale, scale_px, &r_max)
						emit_seg(exp, first_v, p2, p0, src_verts[i2].color, src_verts[i0].color, cam_back, cam_up, cam_right, view_proj, scene.camera.orthographic, px_space, raw_w, m_scale, scale_px, &r_max)
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

// Marks the draws of particle quads (`.Sprites` geometry) in their shading
// materials, for both renderers' sprite paths.
sprite_materials :: proc "contextless" (scene: ^Scene, geos: ^Geometries, exp: ^Expanded_Draws, materials: []Shading_Material) {
	for g, i in scene.draws.geometry[:len(scene.draws)] {
		if geo, ok := geometry_get_any(geos, exp, g); ok && geo.mode == .Sprites && i < len(materials) {
			materials[i].flags += {.Sprite}
		}
	}
}

// Whether a frame needs its depth drawn: asked for, soft particles or
// probed ones.
needs_depth :: proc "contextless" (scene: ^Scene) -> bool {
	if scene.depth {
		return true
	}
	for m in scene.draws.material[:len(scene.draws)] {
		if m.softness > 0 || m.occlusion_radius > 0 {
			return true
		}
	}
	return false
}

// Points on a disc of radius 1, spread evenly (a Vogel spiral), where a
// particle's visibility is probed (shaders/o3d.slang's alike).
@(rodata)
OCCLUSION_TAPS := [16]Vec2{
	{0.176777, 0}, {-0.225772, 0.206826}, {0.034558, -0.393771}, {0.284571, 0.371173},
	{-0.522223, -0.092374}, {0.494695, -0.314685}, {-0.165466, 0.615525}, {-0.315561, -0.607594},
	{0.684642, 0.250030}, {-0.712256, 0.294009}, {0.343355, -0.733729}, {0.253730, 0.808932},
	{-0.764746, -0.443186}, {0.897134, -0.197232}, {-0.547507, 0.778772}, {-0.126487, -0.976090},
}

// The share of a particle row's probe disc (`Sprite_Row.probe`) that the
// frame's depth (`depths`, `width x height`) leaves unhidden: a point is
// where nothing is drawn or what is drawn is farther, and hidden off the
// frame. Both renderers alike (shaders/o3d.slang's `sprite_visibility`).
@(private)
sprite_visibility :: proc "contextless" (probe: Vec4, depths: []f32, width, height: int) -> f32 {
	seen := 0
	for o in OCCLUSION_TAPS {
		p := probe.xy + o * probe.w
		if p.x < 0 || p.y < 0 || p.x >= f32(width) || p.y >= f32(height) {
			continue
		}
		d := depths[int(p.y) * width + int(p.x)]
		if d <= 0 || probe.z > d {
			seen += 1
		}
	}
	return f32(seen) / len(OCCLUSION_TAPS)
}

// A depth value (the frame's depth, `camera_project`'s) as distance along
// the camera's axis: what depth of field, haze and soft particles compare.
view_depth :: proc "contextless" (c: Camera, depth: f32) -> f32 {
	near := c.near if c.near > 0 else 0.1
	far := c.far if c.far > near else near * 1e4
	if c.orthographic {
		return far - depth * (far - near)
	}
	return near * far / (near + depth * (far - near))
}

// Where world point `p` falls on a `width x height` frame of camera `c`, as
// both renderers map it: `pixel` from the top left (pixel centres at .5),
// `depth` the depth they test (reversed: 1 at near, 0 at far), and whether it is in
// front of the camera (behind it, `pixel` means nothing). Whether something
// nearer hides it is the frame's depth's to say.
camera_project :: proc "contextless" (c: Camera, width, height: int, p: Vec3) -> (pixel: Vec2, depth: f32, in_front: bool) {
	view, proj := camera_matrices(c, width, height)
	clip := proj * view * Vec4{p.x, p.y, p.z, 1}
	if clip.w <= 0 {
		return {}, 0, false
	}
	iw := 1 / clip.w
	pixel = {(clip.x * iw * 0.5 + 0.5) * f32(width), (0.5 - clip.y * iw * 0.5) * f32(height)}
	return pixel, clip.z * iw, true
}

// Whether what the frame's depth holds (`depths`: software_depth's or
// read_depth's, `width x height`) hides world point `p` from camera `c`:
// something there is nearer than the point by more than `bias` (world
// units along the camera's axis; for a point a little inside a surface).
// The four texels around the point are read and the farthest kept, so a
// point on a drawn surface (a label pinned to a part) is not hidden by its
// own surface's depth, sampled at pixel centres. Off the frame or behind
// the camera: hidden (not seen).
point_hidden :: proc "contextless" (c: Camera, depths: []f32, width, height: int, p: Vec3, bias: f32 = 0) -> bool {
	pixel, depth, in_front := camera_project(c, width, height, p)
	if !in_front || len(depths) < width * height || pixel.x < 0 || pixel.y < 0 || pixel.x >= f32(width) || pixel.y >= f32(height) {
		return true
	}
	x0 := clamp(int(math.floor(pixel.x - 0.5)), 0, width - 1)
	y0 := clamp(int(math.floor(pixel.y - 0.5)), 0, height - 1)
	x1, y1 := min(x0 + 1, width - 1), min(y0 + 1, height - 1)
	farthest := min(depths[y0 * width + x0], depths[y0 * width + x1], depths[y1 * width + x0], depths[y1 * width + x1])
	point := view_depth(c, depth)
	return view_depth(c, farthest) + bias < point * (1 - 1e-5)
}

// The frame's depth (`depths`: software_depth's, `width x height`) as the
// host's RGBA16F image (`out`, 4 halves per pixel, rows from the top), as
// `o3d_sdlgpu.record_depth_into` draws it: rgb the distance along camera
// `c`'s axis (`view_depth`, world units) and a 1 where something was drawn;
// rgb `far` and a 0 where nothing was.
depth_image :: proc "contextless" (c: Camera, depths: []f32, width, height: int, far: f32, out: []f16) {
	n := min(width * height, len(depths), len(out) / 4)
	for i in 0 ..< n {
		d := depths[i]
		v := f16(view_depth(c, d)) if d > 0 else f16(far)
		out[i * 4], out[i * 4 + 1], out[i * 4 + 2], out[i * 4 + 3] = v, v, v, 1 if d > 0 else 0
	}
}

// The order to draw in: the draws that can be seen (with geometry, and a
// compiled shader if they ask for one, the bounding sphere at least partly
// in the view), the opaque ones as listed, then the transmissive ones
// (blended or not), then the other blended ones, each of those two
// farthest from the camera first, as three.js orders them. `snapshot` is
// where the frame is kept for the transmissive draws to see through: the
// first transmissive one's place (len(order): none).
draw_order :: proc(scene: ^Scene, geos: ^Geometries, shaders: ^Shaders, view, proj: Mat4, order: ^[dynamic]int, exp: ^Expanded_Draws = nil) -> (snapshot: int) {
	clear(order)
	n := len(scene.draws)
	materials := scene_materials(scene)
	models := scene.draws.model[:n]
	geometries := scene.draws.geometry[:n]
	view_proj := proj * view
	seen :: #force_inline proc(scene: ^Scene, geos: ^Geometries, shaders: ^Shaders, m: Material, i: int, geometries: []Geometry_Id, models: []Mat4, view_proj: Mat4, exp: ^Expanded_Draws) -> bool {
		return (m.shader == 0 || shader_get(shaders, m.shader) != nil) && draw_seen(geos, geometries[i], models[i], view_proj, scene_draw_joints(scene, i), scene_draw_morphs(scene, i), exp)
	}
	for m, i in materials {
		if !blended(m) && !transmissive(m) && seen(scene, geos, shaders, m, i, geometries, models, view_proj, exp) {
			append(order, i)
		}
	}
	snapshot = len(order)
	for m, i in materials {
		if transmissive(m) && seen(scene, geos, shaders, m, i, geometries, models, view_proj, exp) {
			append(order, i)
		}
	}
	first := len(order)
	for m, i in materials {
		if blended(m) && !transmissive(m) && seen(scene, geos, shaders, m, i, geometries, models, view_proj, exp) {
			append(order, i)
		}
	}
	if snapshot == first {
		snapshot = len(order)
	}
	// The view z of the model's origin: view's row 2 by its translation.
	depth :: #force_inline proc "contextless" (model, view: Mat4) -> f32 {
		return view[2, 0] * model[0, 3] + view[2, 1] * model[1, 3] + view[2, 2] * model[2, 3] + view[2, 3]
	}
	for span in ([2][]int{order[min(snapshot, first):first], order[first:]}) {
		for i in 1 ..< len(span) {
			for j := i; j > 0 && depth(models[span[j]], view) < depth(models[span[j - 1]], view); j -= 1 {
				span[j], span[j - 1] = span[j - 1], span[j]
			}
		}
	}
	return
}

// Whether a material sees the frame behind it (`Material.transmission`,
// the standard material's; not particles').
transmissive :: #force_inline proc "contextless" (m: Material) -> bool {
	return m.transmission > 0 && m.shader == 0 && m.mode != .Sprites && .Basic not_in m.flags && .Toon not_in m.flags && .Shadow_Only not_in m.flags
}

// Each transmissive draw's model axis scales (three.js scales the
// refracted ray by them).
transmission_scale :: proc "contextless" (materials: []Shading_Material, models: []Mat4) {
	for &m, i in materials {
		if .Transmission in m.flags {
			md := models[i]
			m.layers.scale = {
				linalg.length(Vec3{md[0, 0], md[1, 0], md[2, 0]}),
				linalg.length(Vec3{md[0, 1], md[1, 1], md[2, 1]}),
				linalg.length(Vec3{md[0, 2], md[1, 2], md[2, 2]}),
			}
		}
	}
}

// Where a transmissive point sees the frame behind it (three.js's
// getIBLVolumeRefraction): the view refracted into the surface by the ior,
// out the thickness later (each axis by the model's scale), projected to
// the frame (uv, v up); the level to read there (blurrier as it is
// rougher and refracts more), and how far the ray went (the absorption's
// distance).
refraction :: #force_inline proc "contextless" (world, n, eye: Vec3, roughness: f32, ly: ^Shading_Layers, view_proj: Mat4, width: int) -> (uv: Vec2, lod, distance: f32) {
	v := linalg.normalize(eye - world)
	ray := linalg.refract(-v, n, 1 / ly.ior)
	if l := linalg.length(ray); l > 0 {
		ray *= ly.thickness / l
	}
	ray *= ly.scale
	c := view_proj * Vec4{world.x + ray.x, world.y + ray.y, world.z + ray.z, 1}
	w := c.w if abs(c.w) > 1e-6 else 1e-6
	uv = linalg.clamp(c.xy / w * 0.5 + 0.5, 0, 1)
	lod = math.log2(f32(max(width, 1))) * clamp(roughness, 0.0525, 1) * clamp(ly.ior * 2 - 2, 0, 1)
	return uv, lod, linalg.length(ray)
}

// The frame behind a transmissive point: the snapshot (texture 1 of `t`,
// linear and premultiplied, not tone mapped) at `uv` (v up), clamped at
// its edges, by three.js's textureBicubic: a cubic B-spline at mips
// floor(lod) and floor(lod) + 1, four bilinear taps each, mixed by the
// fraction.
frame_behind :: proc "contextless" (t: ^Textures, uv: Vec2, lod: f32) -> Vec4 {
	tex, ok := texture_get(t, 1)
	if !ok || !tex.hdr {
		return {}
	}
	l := clamp(lod, 0, f32(tex.levels - 1))
	f := math.floor(l)
	c := frame_bicubic(t, &tex, int(f), uv)
	if l > f {
		c = linalg.lerp(c, frame_bicubic(t, &tex, int(f) + 1, uv), l - f)
	}
	return c
}

@(private = "file")
frame_bicubic :: proc "contextless" (t: ^Textures, tex: ^Texture, level: int, uv: Vec2) -> Vec4 {
	w, h := int(max(tex.width >> u32(level), 1)), int(max(tex.height >> u32(level), 1))
	mip := t.hdr_texels[tex.mips[level]:][:w * h]
	// In texels from the top left (row 0 is the frame's top).
	st := Vec2{uv.x * f32(w), (1 - uv.y) * f32(h)} + 0.5
	i := linalg.floor(st)
	f := st - i
	g0, g1, h0, h1 := cubic_weights(f)
	// Each tap a bilinear read at texel coordinate q (centres at +0.5).
	tap :: #force_inline proc "contextless" (mip: [][4]f16, w, h: int, q: Vec2) -> Vec4 {
		p := q - 0.5
		p0 := linalg.floor(p)
		wt := linalg.array_cast(linalg.clamp(linalg.array_cast(linalg.round((p - p0) * f32(BILINEAR_SCALE)), i32), 0, BILINEAR_SCALE), f32) * (1.0 / BILINEAR_SCALE)
		x0, y0 := int(p0.x), int(p0.y)
		x1, y1 := clamp(x0 + 1, 0, w - 1), clamp(y0 + 1, 0, h - 1)
		x0, y0 = clamp(x0, 0, w - 1), clamp(y0, 0, h - 1)
		top := linalg.lerp(linalg.array_cast(mip[y0 * w + x0], f32), linalg.array_cast(mip[y0 * w + x1], f32), wt.x)
		bottom := linalg.lerp(linalg.array_cast(mip[y1 * w + x0], f32), linalg.array_cast(mip[y1 * w + x1], f32), wt.x)
		return linalg.lerp(top, bottom, wt.y)
	}
	return g0.y * (g0.x * tap(mip, w, h, i + {h0.x, h0.y} - 0.5) + g1.x * tap(mip, w, h, i + {h1.x, h0.y} - 0.5)) +
		g1.y * (g0.x * tap(mip, w, h, i + {h0.x, h1.y} - 0.5) + g1.x * tap(mip, w, h, i + {h1.x, h1.y} - 0.5))
}

// The cubic B-spline's two amplitudes and offsets per axis (three.js's g0,
// g1, h0, h1).
@(private = "file")
cubic_weights :: #force_inline proc "contextless" (a: Vec2) -> (g0, g1, h0, h1: Vec2) {
	w0 := (1.0 / 6) * (a * (a * (-a + 3) - 3) + 1)
	w1 := (1.0 / 6) * (a * a * (3 * a - 6) + 4)
	w2 := (1.0 / 6) * (a * (a * (-3 * a + 3) + 3) + 1)
	w3 := (1.0 / 6) * (a * a * a)
	g0, g1 = w0 + w1, w2 + w3
	h0, h1 = -1 + w1 / g0, 1 + w3 / g1
	return
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
// from elsewhere (a colour map's, its vertices', a ShaderMaterial's, an
// exact shape's coverage), as three.js blends any transparent material.
blended :: proc "contextless" (m: Material) -> bool {
	return .Shadow_Only in m.flags || .Additive in m.flags || .Transparent in m.flags && (m.opacity < 1 || m.color_map != 0 || m.alpha_map != 0 || .Vertex_Colors in m.flags || m.shader != 0 || m.mode == .Curves || m.mode == .Sprites)
}

// How a draw's colour meets what is drawn before it: replacing it (its
// depth written), blended over it, or added to it (three.js's
// AdditiveBlending, `.Additive`); blended and added draws test the depth
// without writing it, in the transparent pass. Both renderers alike.
Blend :: enum u8 {
	Opaque,
	Over,
	Add,
}

// A shading material's (`shading_materials`').
blend_of :: #force_inline proc "contextless" (m: ^Shading_Material) -> Blend {
	if .Additive in m.flags {
		return .Add
	}
	return .Over if .Blended in m.flags else .Opaque
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

// The frame's clear (`Scene.background_color`) as both renderers write
// it: premultiplied sRGB bytes.
background_clear :: proc "contextless" (scene: ^Scene) -> [4]u8 {
	c := linalg.clamp(scene.background_color, 0, 1)
	return {u8(math.round(c.r * c.a * 255)), u8(math.round(c.g * c.a * 255)), u8(math.round(c.b * c.a * 255)), u8(math.round(c.a * 255))}
}

// The clear in the transmission snapshot: the background colour's linear
// light, premultiplied, and its alpha.
background_clear_light :: proc "contextless" (scene: ^Scene) -> Vec4 {
	c := linalg.clamp(scene.background_color, 0, 1)
	return {srgb_to_linear(c.r) * c.a, srgb_to_linear(c.g) * c.a, srgb_to_linear(c.b) * c.a, c.a}
}

// The fog's near, far and density, defaults in place of unset ones.
fog_terms :: #force_inline proc "contextless" (fog: Fog) -> (near, far, density: f32) {
	if fog.far > fog.near {
		near, far = fog.near, fog.far
	} else {
		near, far = 1, 1000
	}
	density = fog.density if fog.density > 0 else 0.00025
	return
}

// Fog blend factor 0..1 at Euclidean camera distance `dist` (`shaders/o3d.slang`).
fog_factor :: #force_inline proc "contextless" (fog: Fog, dist: f32) -> f32 {
	near, far, density := fog_terms(fog)
	switch fog.kind {
	case .None:
		return 0
	case .Linear:
		return clamp((dist - near) / max(far - near, 1e-6), 0, 1)
	case .Exp2:
		dd := density * dist
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
	softness:      f32, // particles: fade distance before the frame's depth
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
	alpha_map:     Texture_Id,
	normal_scale:  Vec2,
	// The layers' maps (0: none, its layer off, or past the textures
	// bound); for the GPU, each map's texture (4 bits a Layer_Map: 0 none,
	// else 1 + its index in layer_textures).
	layer_maps:    [Layer_Map]Texture_Id,
	layer_slots:   u32,
	layer_textures: [LAYER_TEXTURES]Texture_Id,
	shader:        Shader_Id, // 0: the standard material
	shader_maps:   [SHADER_MAPS]Texture_Id,
	// Triplanar (`.Triplanar`): repeats per unit and the weights' power.
	projection_scale, projection_sharpness: f32,
	layers:        Shading_Layers,
}

// The physical material's terms (`Material.ior`, `.clearcoat`, `.sheen_*`),
// clamped as three.js does; the clear coat and sheen are read when their
// flags are set.
Shading_Layers :: struct {
	f0:                  f32, // a dielectric's reflectance from the ior (0.04 at 1.5)
	// The factors, times their maps at a point (`shade` clamps them as
	// three.js does after the maps: the coat to 1, its roughness from
	// 0.0525).
	clearcoat:           f32,
	clearcoat_roughness: f32,
	sheen_roughness:     f32, // clamped to 0.07 before its map, as three.js
	sheen:               Vec3, // linear
	coat_normal_scale:   Vec2,
	transmission:        f32,
	// In the model's units; at a point, the refracted ray's length in the
	// world's (`refraction`).
	thickness:           f32,
	// The draw's model's axis scales (`transmission_scale`).
	scale:               Vec3,
	ior:                 f32,
	// Absorption per world unit (log of the attenuation colour over its
	// distance; 0: none).
	attenuation:         Vec3,
}

SHADING_LAYERS_NONE :: Shading_Layers{f0 = 0.04, ior = 1.5, scale = 1}


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
	Alpha_Map,
	Shadow_Only,
	// A particle quad (`.Sprites` geometry): its row's colour and opacity,
	// faded by `softness` against the frame's depth.
	Sprite,
	Additive,
	// The colour map's colour is premultiplied by its alpha.
	Premultiplied,
	Vertex_Colors,
	// Maps projected along three axes (`Material.projection`).
	Triplanar,
	Clearcoat,
	Sheen,
	Transmission,
	// The layers' multiplying maps (`clearcoat_map` .. `thickness_map`) are
	// read: those of the layers on (white without a map).
	Layer_Maps,
	Clearcoat_Normal_Map,
}

Shading_Flags :: bit_set[Shading_Flag; u32]

// Whether a material's shadow is cut by its alpha (a colour or alpha map
// with .Alpha_Mask, read by uv: a triplanar one casts whole), and the uv
// set its mask reads: the colour slot's, or the alpha slot's without a
// colour map (both renderers alike).
@(private)
shadow_mask :: #force_inline proc "contextless" (m: ^Shading_Material) -> (masked, uv1: bool) {
	masked = .Alpha_Mask in m.flags && m.flags & {.Color_Map, .Alpha_Map} != {} && .Triplanar not_in m.flags
	slot := Texture_Slot.Color if .Color_Map in m.flags else .Alpha
	return masked, (m.uv_sets >> u32(slot)) & 1 != 0
}

// The physical layers' maps (Material.clearcoat_map ..), as Texture_Slot
// names them, and how many distinct textures a material's may use.
Layer_Map :: enum u8 {
	Clearcoat,
	Clearcoat_Roughness,
	Clearcoat_Normal,
	Sheen_Color,
	Sheen_Roughness,
	Transmission,
	Thickness,
}

LAYER_TEXTURES :: 2

// A layer map's texture slot (its transform and uv set).
layer_slot :: #force_inline proc "contextless" (r: Layer_Map) -> Texture_Slot {
	return Texture_Slot(u8(Texture_Slot.Clearcoat) + u8(r))
}

TEXTURE_FLAGS :: Shading_Flags{.Color_Map, .Normal_Map, .Roughness_Map, .Emissive_Map, .Metalness_Map, .Ao_Map, .Alpha_Map, .Layer_Maps, .Clearcoat_Normal_Map}

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
			if .Premultiplied_Color_Map in m.flags {
				f += {.Premultiplied}
			}
		}
		if .Vertex_Colors in m.flags && m.shader == 0 {
			f += {.Vertex_Colors}
		}
		if m.projection != .Uv && m.shader == 0 {
			f += {.Triplanar}
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
		if has_tex(textures, m.alpha_map) && (.Transparent in m.flags || .Alpha_Mask in m.flags) {
			f += {.Alpha_Map}
		}
		tforms := [Texture_Slot]Texture_Transform{
			.Color               = m.color_transform,
			.Normal              = m.normal_transform,
			.Roughness           = m.roughness_transform,
			.Emissive            = m.emissive_transform,
			.Metalness           = m.metalness_transform,
			.Ao                  = m.ao_transform,
			.Alpha               = m.alpha_transform,
			.Clearcoat           = m.clearcoat_transform,
			.Clearcoat_Roughness = m.clearcoat_roughness_transform,
			.Clearcoat_Normal    = m.clearcoat_normal_transform,
			.Sheen_Color         = m.sheen_color_transform,
			.Sheen_Roughness     = m.sheen_roughness_transform,
			.Transmission        = m.transmission_transform,
			.Thickness           = m.thickness_transform,
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
		if .Additive in m.flags {
			f += {.Additive}
		}
		ior := m.ior if m.ior > 0 else 1.5
		r := (ior - 1) / (ior + 1)
		layers := Shading_Layers {
			f0                  = r * r,
			clearcoat           = clamp(m.clearcoat, 0, 1),
			clearcoat_roughness = clamp(m.clearcoat_roughness, 0, 1),
			sheen_roughness     = clamp(m.sheen_roughness, 0.07, 1),
			sheen               = linear(m.sheen_color),
			transmission        = clamp(m.transmission, 0, 1),
			thickness           = max(m.thickness, 0),
			scale               = 1,
			ior                 = ior,
		}
		if m.attenuation_distance > 0 {
			c := linear(m.attenuation_color)
			layers.attenuation = {math.ln(max(c.r, 1e-4)), math.ln(max(c.g, 1e-4)), math.ln(max(c.b, 1e-4))} / m.attenuation_distance
		}
		if .Basic not_in f && .Toon not_in f && m.shader == 0 {
			if layers.clearcoat > 0 {
				f += {.Clearcoat}
			}
			if layers.sheen != {} {
				f += {.Sheen}
			}
			if transmissive(m) {
				f += {.Transmission}
			}
		}
		// The layers' maps, of the layers on, in at most LAYER_TEXTURES
		// distinct textures (the GPU binds that many): a map in another is
		// dropped, on both renderers.
		layer_map :: #force_inline proc(t: ^Textures, id: Texture_Id, on: bool) -> Texture_Id {
			_, ok := texture_get(t, id)
			return id if ok && on else 0
		}
		coat, sheen, trans := .Clearcoat in f, .Sheen in f, .Transmission in f
		maps := [Layer_Map]Texture_Id {
			.Clearcoat           = layer_map(textures, m.clearcoat_map, coat),
			.Clearcoat_Roughness = layer_map(textures, m.clearcoat_roughness_map, coat),
			.Clearcoat_Normal    = layer_map(textures, m.clearcoat_normal_map, coat),
			.Sheen_Color         = layer_map(textures, m.sheen_color_map, sheen),
			.Sheen_Roughness     = layer_map(textures, m.sheen_roughness_map, sheen),
			.Transmission        = layer_map(textures, m.transmission_map, trans),
			.Thickness           = layer_map(textures, m.thickness_map, trans),
		}
		layer_textures: [LAYER_TEXTURES]Texture_Id
		layer_slots: u32
		bound := 0
		for &id, r in maps {
			if id == 0 {
				continue
			}
			k := 0
			for k < bound && layer_textures[k] != id {
				k += 1
			}
			if k == bound {
				if bound == LAYER_TEXTURES {
					id = 0
					continue
				}
				layer_textures[k] = id
				bound += 1
			}
			layer_slots |= u32(k + 1) << (4 * u32(r))
			if r != .Clearcoat_Normal {
				f += {.Layer_Maps}
			}
		}
		if maps[.Clearcoat_Normal] != 0 {
			f += {.Clearcoat_Normal_Map}
		}
		layers.coat_normal_scale = m.clearcoat_normal_scale if m.clearcoat_normal_scale != {} else 1
		if .Shadow_Only in m.flags && m.shader == 0 {
			f = (f - TEXTURE_FLAGS - {.Basic, .Toon, .Alpha_Mask, .Curves, .Vertex_Colors, .Clearcoat, .Sheen, .Transmission}) + {.Shadow_Only, .Receive_Shadow}
		}
		out[i] = {
			base          = linear(m.color),
			emissive      = linear(m.emissive) * m.emissive_intensity if .Shadow_Only not_in f else {},
			opacity       = clamp(m.opacity, 0, 1) if (.Blended in f || .Alpha_Mask in f) else 1,
			alpha_cutoff  = m.alpha_cutoff,
			softness      = m.softness,
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
			alpha_map     = m.alpha_map,
			normal_scale  = m.normal_scale if m.normal_scale != {} else 1,
			layer_maps    = maps,
			layer_slots   = layer_slots,
			layer_textures = layer_textures,
			shader        = m.shader,
			shader_maps   = m.shader_maps,
			projection_scale     = m.projection_scale if m.projection_scale > 0 else 1,
			projection_sharpness = m.projection_sharpness if m.projection_sharpness > 0 else 4,
			layers               = layers,
		}
	}
}

@(private = "file")
toon_step :: #force_inline proc "contextless" (edge0, edge1, x: f32) -> f32 {
	t := clamp((x - edge0) / (edge1 - edge0), 0, 1)
	return t * t * (3 - 2 * t)
}

// What `shade` reads of a material at a point: its own, or as its maps
// change it there.
Surface :: struct {
	base:      Vec3, // linear
	metalness: f32,
	roughness: f32,
	flags:     Shading_Flags,
}

surface :: #force_inline proc "contextless" (m: ^Shading_Material) -> Surface {
	return {m.base, m.metalness, m.roughness, m.flags}
}

// The colour (linear) leaving a point of `m` at `world` with normal `n`
// toward the eye: three.js's MeshStandardMaterial (Lambert diffuse, GGX
// specular with Smith correlated visibility and Schlick Fresnel) or
// MeshToonMaterial (`.Toon`: 3-band quantized cel-shading + stepped highlight),
// direct lights plus ambient, hemisphere and equirectangular environment lighting,
// plus `emissive`.
// `shadow` / `shadows` scale each shadow light's direct light (`shadow_factor`;
// 1: lit); `ao` scales indirect light (ambient, hemisphere, environment).
//
// MeshPhysicalMaterial's layers (`layers`: the material's as its maps
// change them at the point; nil: none), composed as three.js r181 does:
// the dielectric F0 from the ior; transmission (`.Transmission`) mixing the
// diffuse light toward the frame `behind` (linear, premultiplied:
// `frame_behind` at `refraction`, whose distance is `layers.thickness`)
// absorbed and less what the surface reflects, its alpha into `alpha`;
// sheen, the rest scaled by 1 - 0.157 max(sheen); and a clear coat over
// everything (emissive too), lit about its own normal `coat_n` (the
// surface's before its normal map, or the coat's normal map's).
shade :: proc "contextless" (m: Surface, world, n_in, eye: Vec3, lights: []Shading_Light, shadow: f32 = 1, env: Shading_Environment = {}, ao: f32 = 1, shadows: [MAX_SHADOWS]f32 = {1, 1, 1, 1}, emissive: Vec3 = {}, f0: f32 = 0.04) -> Vec3 {
	_, out := shade_point(m, world, n_in, eye, lights, shadow, env, ao, shadows, emissive, nil, {}, {}, f0, false)
	return out
}

// `shade` with the layers (any of .Clearcoat, .Sheen, .Transmission), and
// transmission's alpha.
shade_layered :: proc "contextless" (m: Surface, world, n_in, eye: Vec3, lights: []Shading_Light, shadow: f32, env: Shading_Environment, ao: f32, shadows: [MAX_SHADOWS]f32, emissive: Vec3, layers: ^Shading_Layers, coat_n: Vec3, behind: Vec4) -> (alpha: f32, out: Vec3) {
	return shade_point(m, world, n_in, eye, lights, shadow, env, ao, shadows, emissive, layers, coat_n, behind, layers.f0, true)
}

LAYER_FLAGS :: Shading_Flags{.Clearcoat, .Sheen, .Transmission}

// Both, one source: LAYERED compiles the layers in.
@(private = "file")
shade_point :: #force_inline proc "contextless" (m: Surface, world, n_in, eye: Vec3, lights: []Shading_Light, shadow: f32, env: Shading_Environment, ao: f32, shadows: [MAX_SHADOWS]f32, emissive: Vec3, layers: ^Shading_Layers, coat_n: Vec3, behind: Vec4, dielectric_f0: f32, $LAYERED: bool) -> (alpha: f32, out: Vec3) {
	base := m.base
	alpha = 1
	if .Basic in m.flags {
		return 1, base + emissive
	}
	n := n_in
	v := linalg.normalize(eye - world)
	if linalg.dot(n, v) < 0 && .Two_Sided in m.flags {
		n = -n
	}
	diffuse := base * (1 - m.metalness)
	when LAYERED {
		ly := layers
	}
	f0 := linalg.lerp(Vec3(dielectric_f0), base, m.metalness)
	roughness := clamp(m.roughness, 0.0525, 1)
	alpha_r := roughness * roughness
	nv := clamp(linalg.dot(n, v), 1e-4, 1)
	when LAYERED {
		coat := .Clearcoat in m.flags
		cn := coat_n
		cc, cr, cnv: f32
		sheen, coat_l: Vec3
	}
	when LAYERED do if coat {
		if linalg.dot(cn, v) < 0 && .Two_Sided in m.flags {
			cn = -cn
		}
		cc = clamp(ly.clearcoat, 0, 1)
		cr = clamp(ly.clearcoat_roughness, 0.0525, 1)
		cnv = clamp(linalg.dot(cn, v), 1e-4, 1)
	}
	// Diffuse and specular apart only for transmission (it mixes the
	// diffuse); one sum otherwise.
	out_d: Vec3
	when LAYERED {
		spec_sum: Vec3
		out_s := &spec_sum
	} else {
		out_s := &out_d
	}
	for &l in lights {
		switch l.kind {
		case .Ambient:
			out_d += l.color * diffuse * (ao / math.PI)
		case .Hemisphere:
			w := 0.5 * n.y + 0.5
			out_d += linalg.lerp(l.ground, l.color, w) * diffuse * (ao / math.PI)
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
				out_d += light * diffuse * (toon_nl / math.PI)
				out_s^ += light * spec_term
				continue
			}
			when LAYERED do if coat {
				// The coat by its own normal, whatever the surface's.
				if cnl := clamp(linalg.dot(cn, ldir), 0, 1); cnl > 0 {
					ch := linalg.normalize(ldir + v)
					coat_l += light * cnl * ggx(0.04, cr * cr, cnl, cnv, clamp(linalg.dot(cn, ch), 0, 1), clamp(linalg.dot(v, ch), 0, 1))
				}
			}
			nl := clamp(linalg.dot(n, ldir), 0, 1)
			if nl <= 0 {
				continue
			}
			irradiance := light * nl
			h := linalg.normalize(ldir + v)
			nh := clamp(linalg.dot(n, h), 0, 1)
			vh := clamp(linalg.dot(v, h), 0, 1)
			out_d += irradiance * diffuse / math.PI
			out_s^ += irradiance * ggx(f0, alpha_r, nl, nv, nh, vh)
			when LAYERED do if .Sheen in m.flags {
				sheen += irradiance * ly.sheen * (charlie(ly.sheen_roughness, nh) * neubelt(nv, nl))
			}
		}
	}
	if env.radiance != 0 {
		r := 2 * linalg.dot(n, v) * n - v
		irr := sample_lod_linear(env.textures, env.irradiance, equirect_uv(n), 0)
		rad := sample_lod_linear(env.textures, env.radiance, equirect_uv(r), roughness * env.max_lod)
		spec := env_brdf(f0, roughness, nv)
		k := ao * env.intensity
		out_d += k * irr * diffuse * (1 - spec)
		out_s^ += k * rad * spec
		when LAYERED do if .Sheen in m.flags {
			sheen += k * irr * ly.sheen * ibl_sheen(ly.sheen_roughness, nv)
		}
		when LAYERED do if coat {
			cr_dir := 2 * linalg.dot(cn, v) * cn - v
			coat_l += k * sample_lod_linear(env.textures, env.radiance, equirect_uv(cr_dir), cr * env.max_lod) * env_brdf(0.04, cr, cnv)
		}
	}
	when LAYERED do if .Transmission in m.flags {
		// The frame behind, absorbed over the ray and less what the surface
		// reflects, takes transmission's share of the diffuse light.
		t := diffuse * linalg.exp(ly.attenuation * ly.thickness)
		through := (1 - env_brdf(f0, roughness, nv)) * t * behind.rgb
		out_d = linalg.lerp(out_d, through, ly.transmission)
		alpha = linalg.lerp(f32(1), 1 - (1 - behind.a) * (t.r + t.g + t.b) / 3, ly.transmission)
	}
	when LAYERED {
		out = out_d + spec_sum + emissive
	} else {
		out = out_d + emissive
	}
	when LAYERED do if .Sheen in m.flags {
		out = out * (1 - 0.157 * max(ly.sheen.r, ly.sheen.g, ly.sheen.b)) + sheen
	}
	when LAYERED do if coat {
		out = out * (1 - cc * (0.04 + 0.96 * math.pow(1 - cnv, 5))) + coat_l * cc
	}
	return
}

// GGX specular (Schlick Fresnel to 1, Smith-GGX correlated visibility,
// GGX distribution) of reflectance `f0` and alpha (roughness squared).
@(private = "file")
ggx :: #force_inline proc "contextless" (f0: Vec3, alpha, nl, nv, nh, vh: f32) -> Vec3 {
	fresnel := f0 + (1 - f0) * math.pow(1 - vh, 5)
	a2 := alpha * alpha
	gv := nl * math.sqrt(a2 + (1 - a2) * nv * nv)
	gl := nv * math.sqrt(a2 + (1 - a2) * nl * nl)
	vis := 0.5 / max(gv + gl, 1e-6)
	dd := nh * nh * (a2 - 1) + 1
	return fresnel * (vis * a2 / (math.PI * dd * dd))
}

// Karis's analytic environment BRDF (three.js's EnvironmentBRDF, F90 1).
@(private = "file")
env_brdf :: #force_inline proc "contextless" (f0: Vec3, roughness, nv: f32) -> Vec3 {
	c0 :: Vec4{-1, -0.0275, -0.572, 0.022}
	c1 :: Vec4{1, 0.0425, 1.04, -0.04}
	rv := roughness * c0 + c1
	a004 := min(rv.x * rv.x, linalg.exp2(-9.28 * nv)) * rv.x + rv.y
	fab := Vec2{-1.04, 1.04} * a004 + rv.zw
	return f0 * fab.x + fab.y
}

// Sheen: the Charlie distribution and Neubelt's visibility (three.js's
// D_Charlie, V_Neubelt), and its fit for environment light (IBLSheenBRDF,
// times pi: o3d's irradiance maps hold three.js's irradiance over pi).
@(private = "file")
charlie :: #force_inline proc "contextless" (roughness, nh: f32) -> f32 {
	inv := 1 / (roughness * roughness)
	sin2h := max(1 - nh * nh, 0.0078125)
	return (2 + inv) * math.pow(sin2h, inv * 0.5) / (2 * math.PI)
}

@(private = "file")
neubelt :: #force_inline proc "contextless" (nv, nl: f32) -> f32 {
	return clamp(1 / (4 * (nl + nv - nl * nv)), 0, 1)
}

@(private = "file")
ibl_sheen :: #force_inline proc "contextless" (roughness, nv: f32) -> f32 {
	r2 := roughness * roughness
	a := -339.2 * r2 + 161.4 * roughness - 25.9 if roughness < 0.25 else -8.48 * r2 + 14.3 * roughness - 9.95
	b := 44.0 * r2 - 23.7 * roughness + 3.26 if roughness < 0.25 else 1.97 * r2 - 3.27 * roughness + 0.72
	dg := math.exp(a * nv + b) + (0 if roughness < 0.25 else 0.1 * (roughness - 0.25))
	return clamp(dg / math.PI, 0, 1) * math.PI
}

