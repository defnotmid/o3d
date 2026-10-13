package o3d

import "core:math"
import "core:math/linalg"

// Geometry: indexed triangles with a normal and uv per vertex, made once and
// referred to by id. The primitives take three.js's geometry parameters
// (BoxGeometry, SphereGeometry, PlaneGeometry, CylinderGeometry,
// TorusGeometry) and the same parameters give the same id, so a frame may
// ask for `box(1, 1, 1)` every time without making it again.

Vertex :: struct {
	position: Vec3,
	normal:   Vec3,
	uv:       Vec2,
}

Geometry :: struct {
	vertices: []Vertex,
	indices:  []u32,
	// Distance from the origin to the farthest vertex (culling, sorting).
	radius:   f32,
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

Geometry_Key :: struct {
	kind:   Primitive,
	params: [7]f32,
}

Geometries :: struct {
	list: [dynamic]Geometry,
	keys: map[Geometry_Key]Geometry_Id,
}

geometries_destroy :: proc(g: ^Geometries) {
	for geo in g.list {
		delete(geo.vertices)
		delete(geo.indices)
	}
	delete(g.list)
	delete(g.keys)
	g^ = {}
}

geometry_get :: proc "contextless" (g: ^Geometries, id: Geometry_Id) -> (^Geometry, bool) {
	if id == 0 || int(id) > len(g.list) {
		return nil, false
	}
	return &g.list[id - 1], true
}

// Adds geometry made by the caller (the slices are taken over).
geometry_add :: proc(g: ^Geometries, vertices: []Vertex, indices: []u32) -> Geometry_Id {
	r: f32
	for v in vertices {
		r = max(r, linalg.length(v.position))
	}
	append(&g.list, Geometry{vertices, indices, r})
	return Geometry_Id(len(g.list))
}

@(private = "file")
keyed :: proc(g: ^Geometries, key: Geometry_Key, make_it: proc(key: Geometry_Key) -> ([]Vertex, []u32)) -> Geometry_Id {
	if id, ok := g.keys[key]; ok {
		return id
	}
	v, i := make_it(key)
	id := geometry_add(g, v, i)
	g.keys[key] = id
	return id
}

// BoxGeometry(width, height, depth, widthSegments, heightSegments, depthSegments).
box :: proc(g: ^Geometries, width: f32 = 1, height: f32 = 1, depth: f32 = 1, width_segments := 1, height_segments := 1, depth_segments := 1) -> Geometry_Id {
	key := Geometry_Key{.Box, {width, height, depth, f32(max(width_segments, 1)), f32(max(height_segments, 1)), f32(max(depth_segments, 1)), 0}}
	return keyed(g, key, proc(k: Geometry_Key) -> ([]Vertex, []u32) {
		w, h, d := k.params[0], k.params[1], k.params[2]
		sx, sy, sz := int(k.params[3]), int(k.params[4]), int(k.params[5])
		verts := make([dynamic]Vertex)
		idx := make([dynamic]u32)
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
		face(&verts, &idx, 2, 1, 0, -1, -1, d, h, w, sz, sy) // +x
		face(&verts, &idx, 2, 1, 0, 1, -1, d, h, -w, sz, sy) // -x
		face(&verts, &idx, 0, 2, 1, 1, 1, w, d, h, sx, sz) // +y
		face(&verts, &idx, 0, 2, 1, 1, -1, w, d, -h, sx, sz) // -y
		face(&verts, &idx, 0, 1, 2, 1, -1, w, h, d, sx, sy) // +z
		face(&verts, &idx, 0, 1, 2, -1, -1, w, h, -d, sx, sy) // -z
		return verts[:], idx[:]
	})
}

// SphereGeometry(radius, widthSegments, heightSegments).
sphere :: proc(g: ^Geometries, radius: f32 = 1, width_segments := 32, height_segments := 16) -> Geometry_Id {
	key := Geometry_Key{.Sphere, {radius, f32(max(width_segments, 3)), f32(max(height_segments, 2)), 0, 0, 0, 0}}
	return keyed(g, key, proc(k: Geometry_Key) -> ([]Vertex, []u32) {
		r, ws, hs := k.params[0], int(k.params[1]), int(k.params[2])
		verts := make([dynamic]Vertex)
		idx := make([dynamic]u32)
		for iy in 0 ..= hs {
			v := f32(iy) / f32(hs)
			for ix in 0 ..= ws {
				u := f32(ix) / f32(ws)
				// three.js: x = -cos(phi) sin(theta), y = cos(theta), z = sin(phi) sin(theta).
				phi, theta := u * 2 * math.PI, v * math.PI
				n := Vec3{-math.cos(phi) * math.sin(theta), math.cos(theta), math.sin(phi) * math.sin(theta)}
				append(&verts, Vertex{n * r, n, {u, 1 - v}})
			}
		}
		for iy in 0 ..< hs {
			for ix in 0 ..< ws {
				a := u32(iy * (ws + 1) + ix + 1)
				b := u32(iy * (ws + 1) + ix)
				c := u32((iy + 1) * (ws + 1) + ix)
				d := u32((iy + 1) * (ws + 1) + ix + 1)
				if iy != 0 {
					append(&idx, a, b, d)
				}
				if iy != hs - 1 {
					append(&idx, b, c, d)
				}
			}
		}
		return verts[:], idx[:]
	})
}

// PlaneGeometry(width, height, widthSegments, heightSegments): in the XY
// plane, facing +Z.
plane :: proc(g: ^Geometries, width: f32 = 1, height: f32 = 1, width_segments := 1, height_segments := 1) -> Geometry_Id {
	key := Geometry_Key{.Plane, {width, height, f32(max(width_segments, 1)), f32(max(height_segments, 1)), 0, 0, 0}}
	return keyed(g, key, proc(k: Geometry_Key) -> ([]Vertex, []u32) {
		w, h, gx, gy := k.params[0], k.params[1], int(k.params[2]), int(k.params[3])
		verts := make([dynamic]Vertex)
		idx := make([dynamic]u32)
		for iy in 0 ..= gy {
			y := f32(iy) * h / f32(gy) - h / 2
			for ix in 0 ..= gx {
				x := f32(ix) * w / f32(gx) - w / 2
				append(&verts, Vertex{{x, -y, 0}, {0, 0, 1}, {f32(ix) / f32(gx), 1 - f32(iy) / f32(gy)}})
			}
		}
		for iy in 0 ..< gy {
			for ix in 0 ..< gx {
				a := u32(ix + (gx + 1) * iy)
				b := u32(ix + (gx + 1) * (iy + 1))
				c := u32(ix + 1 + (gx + 1) * (iy + 1))
				d := u32(ix + 1 + (gx + 1) * iy)
				append(&idx, a, b, d, b, c, d)
			}
		}
		return verts[:], idx[:]
	})
}

// CylinderGeometry(radiusTop, radiusBottom, height, radialSegments,
// heightSegments, openEnded): along Y, centred; radiusTop 0 is a cone.
cylinder :: proc(g: ^Geometries, radius_top: f32 = 1, radius_bottom: f32 = 1, height: f32 = 1, radial_segments := 32, height_segments := 1, open_ended := false) -> Geometry_Id {
	key := Geometry_Key{.Cylinder, {radius_top, radius_bottom, height, f32(max(radial_segments, 3)), f32(max(height_segments, 1)), 1 if open_ended else 0, 0}}
	return keyed(g, key, proc(k: Geometry_Key) -> ([]Vertex, []u32) {
		rt, rb, h := k.params[0], k.params[1], k.params[2]
		rs, hs, open := int(k.params[3]), int(k.params[4]), k.params[5] != 0
		verts := make([dynamic]Vertex)
		idx := make([dynamic]u32)
		slope := (rb - rt) / h
		for iy in 0 ..= hs {
			v := f32(iy) / f32(hs)
			r := v * (rb - rt) + rt
			for ix in 0 ..= rs {
				u := f32(ix) / f32(rs)
				theta := u * 2 * math.PI
				s, c := math.sin(theta), math.cos(theta)
				n := linalg.normalize(Vec3{s, slope, c})
				append(&verts, Vertex{{r * s, -v * h + h / 2, r * c}, n, {u, 1 - v}})
			}
		}
		for ix in 0 ..< rs {
			for iy in 0 ..< hs {
				a := u32(iy * (rs + 1) + ix)
				b := u32((iy + 1) * (rs + 1) + ix)
				c := u32((iy + 1) * (rs + 1) + ix + 1)
				d := u32(iy * (rs + 1) + ix + 1)
				append(&idx, a, b, d, b, c, d)
			}
		}
		if !open {
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
			cap(&verts, &idx, true, rt, h, rs)
			cap(&verts, &idx, false, rb, h, rs)
		}
		return verts[:], idx[:]
	})
}

// TorusGeometry(radius, tube, radialSegments, tubularSegments): in the XY
// plane around Z.
torus :: proc(g: ^Geometries, radius: f32 = 1, tube: f32 = 0.4, radial_segments := 12, tubular_segments := 48) -> Geometry_Id {
	key := Geometry_Key{.Torus, {radius, tube, f32(max(radial_segments, 3)), f32(max(tubular_segments, 3)), 0, 0, 0}}
	return keyed(g, key, proc(k: Geometry_Key) -> ([]Vertex, []u32) {
		R, r, rs, ts := k.params[0], k.params[1], int(k.params[2]), int(k.params[3])
		verts := make([dynamic]Vertex)
		idx := make([dynamic]u32)
		for j in 0 ..= rs {
			for i in 0 ..= ts {
				u := f32(i) / f32(ts) * 2 * math.PI
				v := f32(j) / f32(rs) * 2 * math.PI
				p := Vec3{(R + r * math.cos(v)) * math.cos(u), (R + r * math.cos(v)) * math.sin(u), r * math.sin(v)}
				centre := Vec3{R * math.cos(u), R * math.sin(u), 0}
				append(&verts, Vertex{p, linalg.normalize(p - centre), {f32(i) / f32(ts), f32(j) / f32(rs)}})
			}
		}
		for j in 1 ..= rs {
			for i in 1 ..= ts {
				a := u32((ts + 1) * j + i - 1)
				b := u32((ts + 1) * (j - 1) + i - 1)
				c := u32((ts + 1) * (j - 1) + i)
				d := u32((ts + 1) * j + i)
				append(&idx, a, b, d, b, c, d)
			}
		}
		return verts[:], idx[:]
	})
}
