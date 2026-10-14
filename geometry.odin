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
}

Geometry :: struct {
	first_vertex, vertex_count: u32,
	first_index, index_count:   u32,
	// Distance from the origin to the farthest vertex (culling, sorting).
	radius:                     f32,
}

// 0 is no geometry.
Geometry_Id :: distinct u32

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
	vertices: [dynamic]Vertex,
	indices:  [dynamic]u32,
	list:     [dynamic]Geometry,
	keys:     map[Geometry_Key]Geometry_Id,
	// A primitive is built here, then appended to the shared arrays.
	scratch:  struct {
		vertices: [dynamic]Vertex,
		indices:  [dynamic]u32,
	},
}

geometries_destroy :: proc(g: ^Geometries) {
	delete(g.vertices)
	delete(g.indices)
	delete(g.list)
	delete(g.keys)
	delete(g.scratch.vertices)
	delete(g.scratch.indices)
	g^ = {}
}

geometry_get :: proc "contextless" (g: ^Geometries, id: Geometry_Id) -> (Geometry, bool) {
	if id == 0 || int(id) > len(g.list) {
		return {}, false
	}
	return g.list[id - 1], true
}

// A geometry's vertices and its indices into them.
geometry_mesh :: proc "contextless" (g: ^Geometries, id: Geometry_Id) -> (vertices: []Vertex, indices: []u32, ok: bool) {
	geo := geometry_get(g, id) or_return
	return g.vertices[geo.first_vertex:][:geo.vertex_count], g.indices[geo.first_index:][:geo.index_count], true
}

// Adds geometry made by the caller (copied; indices count from 0).
// How many geometries have been made (each kept): a host compares it across
// frames to find a scene that makes geometry every frame.
geometries_made :: proc "contextless" (g: ^Geometries) -> int {
	return len(g.list)
}

geometry_add :: proc(g: ^Geometries, vertices: []Vertex, indices: []u32) -> Geometry_Id {
	r: f32
	for v in vertices {
		r = max(r, linalg.length(v.position))
	}
	append(&g.list, Geometry{u32(len(g.vertices)), u32(len(vertices)), u32(len(g.indices)), u32(len(indices)), r})
	append(&g.vertices, ..vertices)
	append(&g.indices, ..indices)
	return Geometry_Id(len(g.list))
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
				append(verts, Vertex{p, n, {f32(ix) / f32(gx), 1 - f32(iy) / f32(gy)}})
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
			append(verts, Vertex{n * r, n, {u, 1 - v}})
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
			append(verts, Vertex{{x, -y, 0}, {0, 0, 1}, {f32(ix) / f32(gx), 1 - f32(iy) / f32(gy)}})
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
			append(verts, Vertex{{r * s, -v * h + h / 2, r * c}, n, {u, 1 - v}})
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
		append(verts, Vertex{{0, h / 2 * sign, 0}, {0, sign, 0}, {0.5, 0.5}})
		for ix in 0 ..= rs {
			theta := f32(ix) / f32(rs) * 2 * math.PI
			s, c := math.sin(theta), math.cos(theta)
			append(verts, Vertex{{r * s, h / 2 * sign, r * c}, {0, sign, 0}, {c * 0.5 + 0.5, s * 0.5 * sign + 0.5}})
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
			append(verts, Vertex{pos, linalg.normalize(pos - centre), {f32(i) / f32(ts), f32(j) / f32(rs)}})
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
