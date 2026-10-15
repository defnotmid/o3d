package o3d

import "core:math"
import "core:math/linalg"

// Geometry: indexed triangles with a normal and uv per vertex, made once and
// referred to by id. Every geometry's vertices and indices live in two
// shared arrays (`Geometries.vertices`, `.indices`); a geometry is its
// ranges in them, its indices counting from its first vertex. The
// primitives take three.js's geometry parameters (BoxGeometry,
// SphereGeometry, PlaneGeometry, CylinderGeometry, TorusGeometry), are made
// by the `PRIMITIVES` table from their `Geometry_Key`, and the same key gives
// the same id, so a frame may ask for `box(1, 1, 1)` every time without
// making it again.
//
// Geometry is kept for the life of `Geometries` (never freed, so ids stay
// valid and the GPU uploads each mesh once). A shape that changes over time
// changes through the draw's transform, as three.js animates mesh.scale
// rather than geometry parameters: `sphere(geos, 1)` scaled by `compose`,
// not `sphere(geos, radius(t))`, which makes a new mesh every frame it
// differs. `geometries_made` lets a host notice that: it should stop
// growing once every shape has been seen.

Vertex :: struct {
	position: Vec3,
	normal:   Vec3,
	uv:       Vec2,
	uv1:      Vec2,
}

// A skinned vertex's four joint indices (into the draw's joint matrices) and
// weights (normalized to sum to 1), kept in `Geometries.skin_vertices` so
// unskinned meshes stay as they are.
Skin_Vertex :: struct {
	joints:  [4]u16,
	weights: Vec4,
}

MAX_MORPH_TARGETS :: 8

// A morph target's vertex delta (kept in `Geometries.morph_deltas`, target by
// target, `vertex_count` deltas per target, so meshes without morphs pay
// nothing). Padded to 16-byte vectors so the GPU's storage buffer layout
// matches across Vulkan and Metal.
Morph_Delta :: struct {
	position: [4]f32,
	normal:   [4]f32,
}
#assert(size_of(Morph_Delta) == 32)

// A non-zero morph target active on a draw: `base` is the target's first
// delta in `Geometries.morph_deltas` (`first_morph + target * vertex_count`).
Active_Morph :: struct {
	base:   u32,
	weight: f32,
}

// Selects up to `MAX_MORPH_TARGETS` non-zero morph targets for `geo` from
// `weights` (the strongest when more than `MAX_MORPH_TARGETS` are non-zero,
// kept in target order), shared by the software vertex stage and the GPU.
active_morphs :: proc "contextless" (geo: Geometry, weights: []f32, out: ^[MAX_MORPH_TARGETS]Active_Morph) -> int {
	limit := min(int(geo.morph_targets), len(weights))
	if limit <= 0 || geo.vertex_count == 0 {
		return 0
	}
	n := 0
	for t in 0 ..< limit {
		w := weights[t]
		if w == 0 {
			continue
		}
		base := geo.first_morph + u32(t) * geo.vertex_count
		if n < MAX_MORPH_TARGETS {
			out[n] = {base, w}
			n += 1
			continue
		}
		min_idx := 0
		for k in 1 ..< MAX_MORPH_TARGETS {
			if abs(out[k].weight) < abs(out[min_idx].weight) {
				min_idx = k
			}
		}
		if abs(w) > abs(out[min_idx].weight) {
			for k in min_idx ..< MAX_MORPH_TARGETS - 1 {
				out[k] = out[k + 1]
			}
			out[MAX_MORPH_TARGETS - 1] = {base, w}
		}
	}
	return n
}

// Applies morph target deltas `active` to `v` (vertex `vi` within its
// geometry) in local space, before skinning, the same math as `vs_main` and
// `vs_skinned` in shaders/o3d.slang.
morph_vertex :: #force_inline proc "contextless" (v: Vertex, vi: u32, deltas: []Morph_Delta, active: []Active_Morph) -> (pos, norm: Vec3) {
	pos, norm = v.position, v.normal
	for a in active {
		d := deltas[a.base + vi]
		pos += Vec3{d.position.x, d.position.y, d.position.z} * a.weight
		norm += Vec3{d.normal.x, d.normal.y, d.normal.z} * a.weight
	}
	return
}

// Applies linear blend skinning from `joints` to `v` in the draw's local
// space, the same math as `vs_skinned` in shaders/o3d.slang.
skin_vertex :: #force_inline proc "contextless" (v: Vertex, sv: Skin_Vertex, joints: []Mat4) -> (pos, norm: Vec3) {
	last := u16(len(joints) - 1)
	j0 := min(sv.joints[0], last)
	j1 := min(sv.joints[1], last)
	j2 := min(sv.joints[2], last)
	j3 := min(sv.joints[3], last)
	skin := Mat4(sv.weights.x) * joints[j0] +
	        Mat4(sv.weights.y) * joints[j1] +
	        Mat4(sv.weights.z) * joints[j2] +
	        Mat4(sv.weights.w) * joints[j3]
	pos = (skin * Vec4{v.position.x, v.position.y, v.position.z, 1}).xyz
	norm = (skin * Vec4{v.normal.x, v.normal.y, v.normal.z, 0}).xyz
	return
}

// How a geometry or draw is rasterized: indexed triangles (Mesh), line
// segments (LineSegments: pairs of vertices or mesh edges expanded into
// camera-facing ribbons), or points (Points: vertices expanded into
// camera-facing quads).
Draw_Mode :: enum u8 {
	Triangles = 0,
	Lines     = 1,
	Points    = 2,
}

Geometry :: struct {
	first_vertex, vertex_count: u32,
	first_index, index_count:   u32,
	first_skin:                 u32,
	first_morph, morph_targets: u32,
	skinned:                    bool,
	mode:                       Draw_Mode,
	// Distance from the origin to the farthest vertex (culling, sorting),
	// and the largest morph position delta length across targets.
	radius, morph_radius:       f32,
}

// 0 is no geometry; ids with SCENE_GEO_BIT refer to per-frame line/point
// geometries in `Scene`; ids with DYNAMIC_GEO_BIT refer to per-frame expanded
// line/point quad geometries in `Expanded_Draws`.
Geometry_Id :: distinct u32

SCENE_GEO_BIT :: Geometry_Id(1 << 29)
DYNAMIC_GEO_BIT :: Geometry_Id(1 << 30)

Primitive :: enum u8 {
	None, // made from vertices by the caller
	Box,
	Sphere,
	Plane,
	Cylinder,
	Torus,
}

PARAMS :: 7

// A primitive and its parameters, in its constructor's order (counts as
// floats, booleans as 0 or 1).
Geometry_Key :: struct {
	kind:   Primitive,
	params: [PARAMS]f32,
}

// Appends a primitive's vertices and indices (counting from 0).
Builder :: #type proc(p: [PARAMS]f32, verts: ^[dynamic]Vertex, idx: ^[dynamic]u32)

PRIMITIVES := [Primitive]Builder {
	.None     = nil,
	.Box      = build_box,
	.Sphere   = build_sphere,
	.Plane    = build_plane,
	.Cylinder = build_cylinder,
	.Torus    = build_torus,
}

Geometries :: struct {
	generation:    Generation, // of these contents, for the GPU's mirror
	vertices:      [dynamic]Vertex,
	skin_vertices: [dynamic]Skin_Vertex,
	morph_deltas:  [dynamic]Morph_Delta,
	indices:       [dynamic]u32,
	list:          [dynamic]Geometry,
	keys:          map[Geometry_Key]Geometry_Id,
	// A primitive is built here, then appended to the shared arrays.
	scratch:       struct {
		vertices: [dynamic]Vertex,
		indices:  [dynamic]u32,
	},
}

geometries_destroy :: proc(g: ^Geometries) {
	delete(g.vertices)
	delete(g.skin_vertices)
	delete(g.morph_deltas)
	delete(g.indices)
	delete(g.list)
	delete(g.keys)
	delete(g.scratch.vertices)
	delete(g.scratch.indices)
	g^ = {}
}

geometry_get :: proc "contextless" (g: ^Geometries, id: Geometry_Id) -> (Geometry, bool) {
	if id == 0 || id >= SCENE_GEO_BIT || int(id) > len(g.list) {
		return {}, false
	}
	return g.list[id - 1], true
}

// A geometry's vertices and its indices into them.
geometry_mesh :: proc "contextless" (g: ^Geometries, id: Geometry_Id) -> (vertices: []Vertex, indices: []u32, ok: bool) {
	geo := geometry_get(g, id) or_return
	return g.vertices[geo.first_vertex:][:geo.vertex_count], g.indices[geo.first_index:][:geo.index_count], true
}

// A geometry's skinning vertices (nil when unskinned).
geometry_skin :: proc "contextless" (g: ^Geometries, id: Geometry_Id) -> []Skin_Vertex {
	geo, ok := geometry_get(g, id)
	if !ok || !geo.skinned {
		return nil
	}
	return g.skin_vertices[geo.first_skin:][:geo.vertex_count]
}

// Adds geometry made by the caller (copied; indices count from 0).
// How many geometries have been made (each kept): a host compares it across
// frames to find a scene that makes geometry every frame.
geometries_made :: proc "contextless" (g: ^Geometries) -> int {
	return len(g.list)
}

geometry_add :: proc(g: ^Geometries, vertices: []Vertex, indices: []u32, skin: []Skin_Vertex = nil, morph_deltas: []Morph_Delta = nil, mode: Draw_Mode = .Triangles) -> Geometry_Id {
	generation_take(&g.generation)
	r: f32
	for v in vertices {
		r = max(r, linalg.length(v.position))
	}
	skinned := len(skin) == len(vertices) && len(skin) > 0
	first_skin := u32(len(g.skin_vertices)) if skinned else 0
	morph_targets: u32
	first_morph: u32
	morph_r: f32
	if len(vertices) > 0 && len(morph_deltas) > 0 && len(morph_deltas) % len(vertices) == 0 {
		morph_targets = u32(len(morph_deltas) / len(vertices))
		first_morph = u32(len(g.morph_deltas))
		for d in morph_deltas {
			morph_r = max(morph_r, linalg.length(Vec3{d.position.x, d.position.y, d.position.z}))
		}
	}
	append(&g.list, Geometry{
		first_vertex  = u32(len(g.vertices)),
		vertex_count  = u32(len(vertices)),
		first_index   = u32(len(g.indices)),
		index_count   = u32(len(indices)),
		first_skin    = first_skin,
		first_morph   = first_morph,
		morph_targets = morph_targets,
		skinned       = skinned,
		mode          = mode,
		radius        = r,
		morph_radius  = morph_r,
	})
	append(&g.vertices, ..vertices)
	if skinned {
		append(&g.skin_vertices, ..skin)
	}
	if morph_targets > 0 {
		append(&g.morph_deltas, ..morph_deltas)
	}
	append(&g.indices, ..indices)
	return Geometry_Id(len(g.list))
}

// LineSegments geometry from pairs of endpoints `(positions[2*i], positions[2*i+1])`.
line_segments :: proc(g: ^Geometries, positions: []Vec3) -> Geometry_Id {
	n := (len(positions) / 2) * 2
	clear(&g.scratch.vertices)
	clear(&g.scratch.indices)
	table_grow(&g.scratch.vertices, n)
	table_grow_non_zero(&g.scratch.indices, n)
	for i in 0 ..< n {
		g.scratch.vertices[i] = {position = positions[i], normal = {0, 0, 1}}
		g.scratch.indices[i] = u32(i)
	}
	return geometry_add(g, g.scratch.vertices[:], g.scratch.indices[:], mode = .Lines)
}

// Connected polyline geometry `(positions[0]->positions[1]->...)` as LineSegments.
polyline :: proc(g: ^Geometries, positions: []Vec3, closed := false) -> Geometry_Id {
	n := len(positions)
	if n < 2 {
		return 0
	}
	segs := n if closed else n - 1
	clear(&g.scratch.vertices)
	clear(&g.scratch.indices)
	table_grow(&g.scratch.vertices, n)
	table_grow_non_zero(&g.scratch.indices, segs * 2)
	for p, i in positions {
		g.scratch.vertices[i] = {position = p, normal = {0, 0, 1}}
	}
	for i in 0 ..< segs {
		g.scratch.indices[i * 2] = u32(i)
		g.scratch.indices[i * 2 + 1] = u32((i + 1) % n)
	}
	return geometry_add(g, g.scratch.vertices[:], g.scratch.indices[:], mode = .Lines)
}

// WireframeGeometry(src): the edges of a triangle mesh as LineSegments.
wireframe :: proc(g: ^Geometries, src: Geometry_Id) -> Geometry_Id {
	verts, idx, ok := geometry_mesh(g, src)
	if !ok || len(idx) < 3 {
		return 0
	}
	clear(&g.scratch.vertices)
	clear(&g.scratch.indices)
	append(&g.scratch.vertices, ..verts)
	add_edge :: proc(indices: ^[dynamic]u32, u, v: u32) {
		lo, hi := min(u, v), max(u, v)
		for i := 0; i + 1 < len(indices^); i += 2 {
			if indices[i] == lo && indices[i + 1] == hi {
				return
			}
		}
		append(indices, lo, hi)
	}
	for t := 0; t + 2 < len(idx); t += 3 {
		a, b, c := idx[t], idx[t + 1], idx[t + 2]
		add_edge(&g.scratch.indices, a, b)
		add_edge(&g.scratch.indices, b, c)
		add_edge(&g.scratch.indices, c, a)
	}
	return geometry_add(g, g.scratch.vertices[:], g.scratch.indices[:], mode = .Lines)
}

// Points geometry from 3D positions.
points :: proc(g: ^Geometries, positions: []Vec3) -> Geometry_Id {
	n := len(positions)
	clear(&g.scratch.vertices)
	clear(&g.scratch.indices)
	table_grow(&g.scratch.vertices, n)
	table_grow_non_zero(&g.scratch.indices, n)
	for p, i in positions {
		g.scratch.vertices[i] = {position = p, normal = {0, 0, 1}}
		g.scratch.indices[i] = u32(i)
	}
	return geometry_add(g, g.scratch.vertices[:], g.scratch.indices[:], mode = .Points)
}

// The geometry for a primitive's key, made the first time it is asked for.
primitive :: proc(g: ^Geometries, key: Geometry_Key) -> Geometry_Id {
	if id, ok := g.keys[key]; ok {
		return id
	}
	build := PRIMITIVES[key.kind]
	if build == nil {
		return 0
	}
	clear(&g.scratch.vertices)
	clear(&g.scratch.indices)
	build(key.params, &g.scratch.vertices, &g.scratch.indices)
	id := geometry_add(g, g.scratch.vertices[:], g.scratch.indices[:])
	g.keys[key] = id
	return id
}

// BoxGeometry(width, height, depth, widthSegments, heightSegments, depthSegments).
box :: proc(g: ^Geometries, width: f32 = 1, height: f32 = 1, depth: f32 = 1, width_segments := 1, height_segments := 1, depth_segments := 1) -> Geometry_Id {
	return primitive(g, {.Box, {width, height, depth, f32(max(width_segments, 1)), f32(max(height_segments, 1)), f32(max(depth_segments, 1)), 0}})
}

// SphereGeometry(radius, widthSegments, heightSegments).
sphere :: proc(g: ^Geometries, radius: f32 = 1, width_segments := 32, height_segments := 16) -> Geometry_Id {
	return primitive(g, {.Sphere, {radius, f32(max(width_segments, 3)), f32(max(height_segments, 2)), 0, 0, 0, 0}})
}

// PlaneGeometry(width, height, widthSegments, heightSegments): in the XY
// plane, facing +Z.
plane :: proc(g: ^Geometries, width: f32 = 1, height: f32 = 1, width_segments := 1, height_segments := 1) -> Geometry_Id {
	return primitive(g, {.Plane, {width, height, f32(max(width_segments, 1)), f32(max(height_segments, 1)), 0, 0, 0}})
}

// CylinderGeometry(radiusTop, radiusBottom, height, radialSegments,
// heightSegments, openEnded): along Y, centred; radiusTop 0 is a cone.
cylinder :: proc(g: ^Geometries, radius_top: f32 = 1, radius_bottom: f32 = 1, height: f32 = 1, radial_segments := 32, height_segments := 1, open_ended := false) -> Geometry_Id {
	return primitive(g, {.Cylinder, {radius_top, radius_bottom, height, f32(max(radial_segments, 3)), f32(max(height_segments, 1)), 1 if open_ended else 0, 0}})
}

// TorusGeometry(radius, tube, radialSegments, tubularSegments): in the XY
// plane around Z.
torus :: proc(g: ^Geometries, radius: f32 = 1, tube: f32 = 0.4, radial_segments := 12, tubular_segments := 48) -> Geometry_Id {
	return primitive(g, {.Torus, {radius, tube, f32(max(radial_segments, 3)), f32(max(tubular_segments, 3)), 0, 0, 0}})
}

// ---- builders (PRIMITIVES) --------------------------------------------------------------

@(private = "file")
build_box :: proc(p: [PARAMS]f32, verts: ^[dynamic]Vertex, idx: ^[dynamic]u32) {
	w, h, d := p[0], p[1], p[2]
	sx, sy, sz := int(p[3]), int(p[4]), int(p[5])
	// One face: axes u, v across it, `w` the outward axis (three.js's
	// buildPlane: px, nx, py, ny, pz, nz).
	face :: proc(verts: ^[dynamic]Vertex, idx: ^[dynamic]u32, u, v, w: int, udir, vdir: f32, width, height, depth: f32, gx, gy: int) {
		base := u32(len(verts))
		for iy in 0 ..= gy {
			y := f32(iy) * height / f32(gy) - height / 2
			for ix in 0 ..= gx {
				x := f32(ix) * width / f32(gx) - width / 2
				p: Vec3
				p[u] = x * udir
				p[v] = y * vdir
				p[w] = depth / 2
				n: Vec3
				n[w] = 1 if depth > 0 else -1
				append(verts, Vertex{p, n, {f32(ix) / f32(gx), 1 - f32(iy) / f32(gy)}, {}})
			}
		}
		for iy in 0 ..< gy {
			for ix in 0 ..< gx {
				a := base + u32(ix + (gx + 1) * iy)
				b := base + u32(ix + (gx + 1) * (iy + 1))
				c := base + u32(ix + 1 + (gx + 1) * (iy + 1))
				e := base + u32(ix + 1 + (gx + 1) * iy)
				append(idx, a, b, e, b, c, e)
			}
		}
	}
	face(verts, idx, 2, 1, 0, -1, -1, d, h, w, sz, sy) // +x
	face(verts, idx, 2, 1, 0, 1, -1, d, h, -w, sz, sy) // -x
	face(verts, idx, 0, 2, 1, 1, 1, w, d, h, sx, sz) // +y
	face(verts, idx, 0, 2, 1, 1, -1, w, d, -h, sx, sz) // -y
	face(verts, idx, 0, 1, 2, 1, -1, w, h, d, sx, sy) // +z
	face(verts, idx, 0, 1, 2, -1, -1, w, h, -d, sx, sy) // -z
}

@(private = "file")
build_sphere :: proc(p: [PARAMS]f32, verts: ^[dynamic]Vertex, idx: ^[dynamic]u32) {
	r, ws, hs := p[0], int(p[1]), int(p[2])
	for iy in 0 ..= hs {
		v := f32(iy) / f32(hs)
		for ix in 0 ..= ws {
			u := f32(ix) / f32(ws)
			// three.js: x = -cos(phi) sin(theta), y = cos(theta), z = sin(phi) sin(theta).
			phi, theta := u * 2 * math.PI, v * math.PI
			n := Vec3{-math.cos(phi) * math.sin(theta), math.cos(theta), math.sin(phi) * math.sin(theta)}
			append(verts, Vertex{n * r, n, {u, 1 - v}, {}})
		}
	}
	for iy in 0 ..< hs {
		for ix in 0 ..< ws {
			a := u32(iy * (ws + 1) + ix + 1)
			b := u32(iy * (ws + 1) + ix)
			c := u32((iy + 1) * (ws + 1) + ix)
			d := u32((iy + 1) * (ws + 1) + ix + 1)
			if iy != 0 {
				append(idx, a, b, d)
			}
			if iy != hs - 1 {
				append(idx, b, c, d)
			}
		}
	}
}

@(private = "file")
build_plane :: proc(p: [PARAMS]f32, verts: ^[dynamic]Vertex, idx: ^[dynamic]u32) {
	w, h, gx, gy := p[0], p[1], int(p[2]), int(p[3])
	for iy in 0 ..= gy {
		y := f32(iy) * h / f32(gy) - h / 2
		for ix in 0 ..= gx {
			x := f32(ix) * w / f32(gx) - w / 2
			append(verts, Vertex{{x, -y, 0}, {0, 0, 1}, {f32(ix) / f32(gx), 1 - f32(iy) / f32(gy)}, {}})
		}
	}
	for iy in 0 ..< gy {
		for ix in 0 ..< gx {
			a := u32(ix + (gx + 1) * iy)
			b := u32(ix + (gx + 1) * (iy + 1))
			c := u32(ix + 1 + (gx + 1) * (iy + 1))
			d := u32(ix + 1 + (gx + 1) * iy)
			append(idx, a, b, d, b, c, d)
		}
	}
}

@(private = "file")
build_cylinder :: proc(p: [PARAMS]f32, verts: ^[dynamic]Vertex, idx: ^[dynamic]u32) {
	rt, rb, h := p[0], p[1], p[2]
	rs, hs, open := int(p[3]), int(p[4]), p[5] != 0
	slope := (rb - rt) / h
	for iy in 0 ..= hs {
		v := f32(iy) / f32(hs)
		r := v * (rb - rt) + rt
		for ix in 0 ..= rs {
			u := f32(ix) / f32(rs)
			theta := u * 2 * math.PI
			s, c := math.sin(theta), math.cos(theta)
			n := linalg.normalize(Vec3{s, slope, c})
			append(verts, Vertex{{r * s, -v * h + h / 2, r * c}, n, {u, 1 - v}, {}})
		}
	}
	for ix in 0 ..< rs {
		for iy in 0 ..< hs {
			a := u32(iy * (rs + 1) + ix)
			b := u32((iy + 1) * (rs + 1) + ix)
			c := u32((iy + 1) * (rs + 1) + ix + 1)
			d := u32(iy * (rs + 1) + ix + 1)
			append(idx, a, b, d, b, c, d)
		}
	}
	if open {
		return
	}
	cap :: proc(verts: ^[dynamic]Vertex, idx: ^[dynamic]u32, top: bool, r, h: f32, rs: int) {
		if r <= 0 {
			return
		}
		sign: f32 = 1 if top else -1
		centre := u32(len(verts))
		append(verts, Vertex{{0, h / 2 * sign, 0}, {0, sign, 0}, {0.5, 0.5}, {}})
		for ix in 0 ..= rs {
			theta := f32(ix) / f32(rs) * 2 * math.PI
			s, c := math.sin(theta), math.cos(theta)
			append(verts, Vertex{{r * s, h / 2 * sign, r * c}, {0, sign, 0}, {c * 0.5 + 0.5, s * 0.5 * sign + 0.5}, {}})
		}
		for ix in 0 ..< rs {
			i := centre + 1 + u32(ix)
			if top {
				append(idx, i, i + 1, centre)
			} else {
				append(idx, i + 1, i, centre)
			}
		}
	}
	cap(verts, idx, true, rt, h, rs)
	cap(verts, idx, false, rb, h, rs)
}

@(private = "file")
build_torus :: proc(p: [PARAMS]f32, verts: ^[dynamic]Vertex, idx: ^[dynamic]u32) {
	R, r, rs, ts := p[0], p[1], int(p[2]), int(p[3])
	for j in 0 ..= rs {
		for i in 0 ..= ts {
			u := f32(i) / f32(ts) * 2 * math.PI
			v := f32(j) / f32(rs) * 2 * math.PI
			pos := Vec3{(R + r * math.cos(v)) * math.cos(u), (R + r * math.cos(v)) * math.sin(u), r * math.sin(v)}
			centre := Vec3{R * math.cos(u), R * math.sin(u), 0}
			append(verts, Vertex{pos, linalg.normalize(pos - centre), {f32(i) / f32(ts), f32(j) / f32(rs)}, {}})
		}
	}
	for j in 1 ..= rs {
		for i in 1 ..= ts {
			a := u32((ts + 1) * j + i - 1)
			b := u32((ts + 1) * (j - 1) + i - 1)
			c := u32((ts + 1) * (j - 1) + i)
			d := u32((ts + 1) * j + i)
			append(idx, a, b, d, b, c, d)
		}
	}
}
