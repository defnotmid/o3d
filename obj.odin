package o3d

// Wavefront .obj + .mtl (Material Template Library) scene import and unified
// `model_load` dispatcher (.glb / .gltf / .3mf / .obj).
//
// Populates the same flat `Model`, `Geometries`, and `Textures` tables as
// `gltf_load` and `threemf_load`, so `gltf_draw(&scene, &model, ...)` renders
// `.obj` models through both the software and GPU renderers with zero changes
// to caller code.
//
// Supported OBJ directives:
//   - `v x y z [r g b]`      vertex position (and optional sRGB vertex colour)
//   - `vt u v`               texture coordinates (OBJ v=1 at top matches o3d)
//   - `vn nx ny nz`          vertex normal
//   - `f v1[/vt1[/vn1]] ...` triangle or N-gon (fan-triangulated; 1-based positive
//                            or negative relative indices)
//   - `o name` / `g name`    object / group boundaries (each becomes a Model_Node)
//   - `mtllib file.mtl`      loads companion `.mtl` relative to `base_dir`
//   - `usemtl name`          splits primitives within an object/group by material
//
// Supported MTL directives:
//   - `newmtl name`
//   - `Kd r g b`             base colour (`Material.color`)
//   - `Ke r g b`             emissive colour (`Material.emissive`)
//   - `Ks r g b` / `Ns s`    specular / shininess -> roughness heuristic when `Pr` absent
//   - `Pr r` / `Pm m`        PBR roughness (`Material.roughness`) and metalness (`Material.metalness`)
//   - `d a` / `Tr t`         opacity (`d` or `1 - Tr`; `< 1` enables `.Blended`)
//   - `map_Kd path`          base colour texture (`Material.color_map`)
//   - `map_Ke path`          emissive texture (`Material.emissive_map`)
//   - `map_Bump` / `bump`    normal map (`Material.normal_map`)
//   - `map_Pr` / `map_Ns`    roughness map (`Material.roughness_map`)
//   - `map_Pm`               metalness map (`Material.metalness_map`)
//   - `map_d`                opacity map (`Material.alpha_map`, its green channel; transparent)
//
// Normals: when faces omit `vn`, area-weighted crease-angle smooth normals
// (60° threshold, matching `threemf.odin`) are computed automatically so smooth
// curved surfaces and sharp hard edges both render cleanly.

import "core:math"
import "core:math/linalg"
import "core:os"
import "core:strconv"
import "core:strings"

Obj_Error :: enum u8 {
	None,
	Empty,
	File_Not_Found,
	Io,
	Invalid_Obj,
	No_Geometry,
}

OBJ_ERROR_TEXT := [Obj_Error]string {
	.None           = "",
	.Empty          = "empty buffer",
	.File_Not_Found = "file not found",
	.Io             = "I/O error reading file",
	.Invalid_Obj    = "invalid Wavefront OBJ data",
	.No_Geometry    = "OBJ contains no valid triangle faces",
}

// Dispatches by file extension (`.glb`, `.gltf`, `.3mf`, `.obj`, case-insensitive)
// and loads the model into flat `Model`, `Geometries`, and `Textures` tables.
model_load :: proc(path: string, geos: ^Geometries, textures: ^Textures = nil) -> (m: Model, ok: bool) {
	dot := strings.last_index_byte(path, '.')
	if dot < 0 {
		return {}, false
	}
	ext := path[dot:]
	if strings.equal_fold(ext, ".glb") || strings.equal_fold(ext, ".gltf") {
		err: Gltf_Error
		m, err = gltf_load(path, geos, textures)
		return m, err == .None
	}
	if strings.equal_fold(ext, ".3mf") {
		err: Threemf_Error
		m, err = threemf_load(path, geos, textures)
		return m, err == .None
	}
	if strings.equal_fold(ext, ".obj") {
		err: Obj_Error
		m, err = obj_load(path, geos, textures)
		return m, err == .None
	}
	return {}, false
}

// Loads a `.obj` file (and any referenced `.mtl` and textures in its directory)
// from disk into flat `Model`, `Geometries`, and `Textures` tables.
obj_load :: proc(path: string, geos: ^Geometries, textures: ^Textures = nil) -> (m: Model, err: Obj_Error) {
	if !os.exists(path) {
		return {}, .File_Not_Found
	}
	bytes, read_err := os.read_entire_file(path, context.temp_allocator)
	if read_err != nil {
		return {}, .Io
	}
	slash := max(strings.last_index_byte(path, '/'), strings.last_index_byte(path, '\\'))
	base_dir := path[:slash] if slash >= 0 else ""
	return obj_parse(bytes, geos, textures, base_dir)
}

// Parses a `.obj` byte slice in memory (resolving `mtllib` and texture paths
// relative to `base_dir` when non-empty) into flat `Model`, `Geometries`, and
// `Textures` tables.
obj_parse :: proc(bytes: []u8, geos: ^Geometries, textures: ^Textures = nil, base_dir := "") -> (m: Model, err: Obj_Error) {
	if len(bytes) == 0 {
		return {}, .Empty
	}

	positions := make([dynamic]Vec3, 0, 256, context.temp_allocator)
	vcolors := make([dynamic]Vec3, 0, 256, context.temp_allocator)
	has_vcolor := make([dynamic]bool, 0, 256, context.temp_allocator)
	uvs := make([dynamic]Vec2, 0, 256, context.temp_allocator)
	normals := make([dynamic]Vec3, 0, 256, context.temp_allocator)
	mtls := make([dynamic]Obj_Mtl, 0, 16, context.temp_allocator)
	groups := make([dynamic]Obj_Group, 0, 8, context.temp_allocator)

	cur_group := -1
	cur_mtl := -1

	src := string(bytes)
	for len(src) > 0 {
		nl := strings.index_byte(src, '\n')
		line := src[:nl] if nl >= 0 else src
		src = src[nl + 1:] if nl >= 0 else ""

		if hash := strings.index_byte(line, '#'); hash >= 0 {
			line = line[:hash]
		}
		line = strings.trim_space(line)
		if len(line) == 0 {
			continue
		}

		kw, rest := obj_next_token(line)
		switch kw {
		case "v":
			x, ok_x := obj_next_f32(&rest)
			y, ok_y := obj_next_f32(&rest)
			z, ok_z := obj_next_f32(&rest)
			if !ok_x || !ok_y || !ok_z {
				return {}, .Invalid_Obj
			}
			append(&positions, Vec3{x, y, z})
			if r, ok_r := obj_next_f32(&rest); ok_r {
				g, ok_g := obj_next_f32(&rest)
				b, ok_b := obj_next_f32(&rest)
				if ok_g && ok_b {
					append(&vcolors, Vec3{clamp(r, 0, 1), clamp(g, 0, 1), clamp(b, 0, 1)})
					append(&has_vcolor, true)
				} else {
					append(&vcolors, Vec3{1, 1, 1})
					append(&has_vcolor, false)
				}
			} else {
				append(&vcolors, Vec3{1, 1, 1})
				append(&has_vcolor, false)
			}
		case "vt":
			u, ok_u := obj_next_f32(&rest)
			v, ok_v := obj_next_f32(&rest)
			if !ok_u {
				return {}, .Invalid_Obj
			}
			append(&uvs, Vec2{u, v if ok_v else 0})
		case "vn":
			nx, ok_x := obj_next_f32(&rest)
			ny, ok_y := obj_next_f32(&rest)
			nz, ok_z := obj_next_f32(&rest)
			if !ok_x || !ok_y || !ok_z {
				return {}, .Invalid_Obj
			}
			n := Vec3{nx, ny, nz}
			append(&normals, linalg.normalize(n) if linalg.length2(n) > 1e-12 else Vec3{0, 1, 0})
		case "o", "g":
			name := strings.trim_space(rest)
			cur_group = len(groups)
			append(&groups, Obj_Group{
				name = name,
				subs = make([dynamic]Obj_Submesh, 0, 4, context.temp_allocator),
			})
		case "mtllib":
			mtl_rel := strings.trim_space(rest)
			if len(mtl_rel) > 0 {
				mtl_path := strings.concatenate({base_dir, "/", mtl_rel}, context.temp_allocator) if len(base_dir) > 0 else mtl_rel
				if os.exists(mtl_path) {
					if mtl_bytes, merr := os.read_entire_file(mtl_path, context.temp_allocator); merr == nil {
						obj_parse_mtl(string(mtl_bytes), &mtls, textures, base_dir)
					}
				}
			}
		case "usemtl":
			mname := strings.trim_space(rest)
			cur_mtl = -1
			for mtl, mi in mtls {
				if mtl.name == mname {
					cur_mtl = mi
					break
				}
			}
		case "f":
			if len(positions) == 0 {
				return {}, .Invalid_Obj
			}
			corners: [64]Obj_Corner
			ccount := 0
			for len(rest) > 0 && ccount < len(corners) {
				tok: string
				tok, rest = obj_next_token(rest)
				if len(tok) == 0 {
					break
				}
				c, ok := obj_parse_corner(tok, len(positions), len(uvs), len(normals))
				if !ok {
					return {}, .Invalid_Obj
				}
				corners[ccount] = c
				ccount += 1
			}
			if ccount < 3 {
				return {}, .Invalid_Obj
			}
			if cur_group < 0 {
				cur_group = len(groups)
				append(&groups, Obj_Group{
					name = "",
					subs = make([dynamic]Obj_Submesh, 0, 4, context.temp_allocator),
				})
			}
			sub := obj_get_submesh(&groups[cur_group], cur_mtl)
			for i := 1; i + 1 < ccount; i += 1 {
				append(&sub.corners, corners[0], corners[i], corners[i + 1])
			}
		}
	}

	total_tris := 0
	for &g in groups {
		for &s in g.subs {
			total_tris += len(s.corners) / 3
		}
	}
	if total_tris == 0 {
		return {}, .No_Geometry
	}

	for mtl in mtls {
		append(&m.material_names, strings.clone(mtl.name))
	}
	ident := linalg.MATRIX4F32_IDENTITY
	for &g in groups {
		first_prim := u32(len(m.primitives))
		prim_count := u32(0)
		for &s in g.subs {
			if len(s.corners) < 3 {
				continue
			}
			base_mat := mtls[s.mtl_idx].mat if s.mtl_idx >= 0 && s.mtl_idx < len(mtls) else MATERIAL_DEFAULT
			geo_id, out_mat := obj_build_primitive(geos, textures, positions[:], vcolors[:], has_vcolor[:], uvs[:], normals[:], s.corners[:], base_mat)
			append(&m.primitives, Model_Primitive{
				geometry    = geo_id,
				material    = out_mat,
				material_id = Model_Material_Id(s.mtl_idx + 1) if s.mtl_idx >= 0 && s.mtl_idx < len(mtls) else 0,
			})
			prim_count += 1
		}
		if prim_count == 0 {
			continue
		}
		mesh_idx := i32(len(m.meshes))
		append(&m.meshes, Model_Mesh{
			first_primitive = first_prim,
			primitive_count = prim_count,
		})
		append(&m.nodes, Model_Node{
			name     = strings.clone(g.name),
			parent   = -1,
			mesh     = mesh_idx,
			skin     = -1,
			light    = -1,
			rotation = linalg.QUATERNIONF32_IDENTITY,
			scale    = {1, 1, 1},
			local    = ident,
		})
	}

	if len(m.primitives) == 0 {
		model_destroy(&m)
		return {}, .No_Geometry
	}
	return m, .None
}

// ---- internal types & helpers -----------------------------------------------------------

@(private = "file")
Obj_Corner :: struct {
	v, vt, vn: i32, // 0-based index into positions / uvs / normals (-1 if absent)
}

@(private = "file")
Obj_Submesh :: struct {
	mtl_idx: int,
	corners: [dynamic]Obj_Corner,
}

@(private = "file")
Obj_Group :: struct {
	name: string,
	subs: [dynamic]Obj_Submesh,
}

@(private = "file")
Obj_Mtl :: struct {
	name: string,
	mat:  Material,
}

@(private = "file")
obj_get_submesh :: proc(g: ^Obj_Group, mtl_idx: int) -> ^Obj_Submesh {
	for &s in g.subs {
		if s.mtl_idx == mtl_idx {
			return &s
		}
	}
	append(&g.subs, Obj_Submesh{
		mtl_idx = mtl_idx,
		corners = make([dynamic]Obj_Corner, 0, 192, context.temp_allocator),
	})
	return &g.subs[len(g.subs) - 1]
}

@(private = "file")
obj_next_token :: proc "contextless" (s: string) -> (tok, rest: string) {
	i := 0
	for i < len(s) && (s[i] == ' ' || s[i] == '\t' || s[i] == '\r') {
		i += 1
	}
	start := i
	for i < len(s) && s[i] != ' ' && s[i] != '\t' && s[i] != '\r' {
		i += 1
	}
	return s[start:i], s[i:]
}

@(private = "file")
obj_next_f32 :: proc(rest: ^string) -> (val: f32, ok: bool) {
	tok, rem := obj_next_token(rest^)
	if len(tok) == 0 {
		return 0, false
	}
	v, parse_ok := strconv.parse_f32(tok)
	if !parse_ok {
		return 0, false
	}
	rest^ = rem
	return v, true
}

@(private = "file")
obj_resolve_index :: proc "contextless" (raw: int, count: int) -> (idx: i32, ok: bool) {
	if raw > 0 {
		if raw > count {
			return -1, false
		}
		return i32(raw - 1), true
	}
	if raw < 0 {
		resolved := count + raw
		if resolved < 0 || resolved >= count {
			return -1, false
		}
		return i32(resolved), true
	}
	return -1, false
}

@(private = "file")
obj_parse_corner :: proc(tok: string, v_count, vt_count, vn_count: int) -> (c: Obj_Corner, ok: bool) {
	c = {-1, -1, -1}
	s1 := strings.index_byte(tok, '/')
	if s1 < 0 {
		v_raw, v_ok := strconv.parse_int(tok)
		if !v_ok {
			return c, false
		}
		c.v, ok = obj_resolve_index(v_raw, v_count)
		return c, ok
	}
	v_raw, v_ok := strconv.parse_int(tok[:s1])
	if !v_ok {
		return c, false
	}
	if c.v, ok = obj_resolve_index(v_raw, v_count); !ok {
		return c, false
	}
	after1 := tok[s1 + 1:]
	s2 := strings.index_byte(after1, '/')
	vt_str := after1[:s2] if s2 >= 0 else after1
	if len(vt_str) > 0 {
		vt_raw, vt_ok := strconv.parse_int(vt_str)
		if !vt_ok {
			return c, false
		}
		if c.vt, ok = obj_resolve_index(vt_raw, vt_count); !ok {
			return c, false
		}
	}
	if s2 >= 0 {
		vn_str := after1[s2 + 1:]
		if len(vn_str) > 0 {
			vn_raw, vn_ok := strconv.parse_int(vn_str)
			if !vn_ok {
				return c, false
			}
			if c.vn, ok = obj_resolve_index(vn_raw, vn_count); !ok {
				return c, false
			}
		}
	}
	return c, true
}

@(private = "file")
obj_parse_mtl :: proc(src_in: string, mtls: ^[dynamic]Obj_Mtl, textures: ^Textures, base_dir: string) {
	cur := -1
	has_pr := false
	src := src_in
	for len(src) > 0 {
		nl := strings.index_byte(src, '\n')
		line := src[:nl] if nl >= 0 else src
		src = src[nl + 1:] if nl >= 0 else ""

		if hash := strings.index_byte(line, '#'); hash >= 0 {
			line = line[:hash]
		}
		line = strings.trim_space(line)
		if len(line) == 0 {
			continue
		}

		kw, rest := obj_next_token(line)
		if kw == "newmtl" {
			name := strings.trim_space(rest)
			cur = len(mtls^)
			has_pr = false
			def_mat := MATERIAL_DEFAULT
			def_mat.roughness = 0.8
			append(mtls, Obj_Mtl{
				name = strings.clone(name, context.temp_allocator),
				mat  = def_mat,
			})
			continue
		}
		if cur < 0 {
			continue
		}
		m := &mtls[cur].mat
		switch kw {
		case "Kd":
			r, ok_r := obj_next_f32(&rest)
			g, ok_g := obj_next_f32(&rest)
			b, ok_b := obj_next_f32(&rest)
			if ok_r && ok_g && ok_b {
				m.color = {clamp(r, 0, 1), clamp(g, 0, 1), clamp(b, 0, 1)}
			}
		case "Ke":
			r, ok_r := obj_next_f32(&rest)
			g, ok_g := obj_next_f32(&rest)
			b, ok_b := obj_next_f32(&rest)
			if ok_r && ok_g && ok_b {
				m.emissive = {clamp(r, 0, 1), clamp(g, 0, 1), clamp(b, 0, 1)}
				if r > 0 || g > 0 || b > 0 {
					m.emissive_intensity = 1
				}
			}
		case "Pr":
			if pr, ok := obj_next_f32(&rest); ok {
				m.roughness = clamp(pr, 0, 1)
				has_pr = true
			}
		case "Pm":
			if pm, ok := obj_next_f32(&rest); ok {
				m.metalness = clamp(pm, 0, 1)
			}
		case "Ns":
			if !has_pr {
				if ns, ok := obj_next_f32(&rest); ok {
					// Convert Blinn-Phong exponent (0..1000) to PBR roughness.
					m.roughness = clamp(math.sqrt(2.0 / (clamp(ns, 0, 1000) + 2.0)), 0.05, 1.0)
				}
			}
		case "d":
			if d, ok := obj_next_f32(&rest); ok {
				m.opacity = clamp(d, 0, 1)
				if m.opacity < 1 {
					m.flags += {.Transparent}
				}
			}
		case "Tr":
			if tr, ok := obj_next_f32(&rest); ok {
				m.opacity = clamp(1 - tr, 0, 1)
				if m.opacity < 1 {
					m.flags += {.Transparent}
				}
			}
		case "map_Kd":
			if tid := obj_load_tex(textures, base_dir, rest); tid != 0 {
				m.color_map = tid
			}
		case "map_Ke":
			if tid := obj_load_tex(textures, base_dir, rest); tid != 0 {
				m.emissive_map = tid
				if m.emissive == ({}) {
					m.emissive = {1, 1, 1}
				}
				if m.emissive_intensity == 0 {
					m.emissive_intensity = 1
				}
			}
		case "map_Bump", "bump", "norm":
			if tid := obj_load_tex(textures, base_dir, rest); tid != 0 {
				m.normal_map = tid
			}
		case "map_Pr", "map_Ns":
			if tid := obj_load_tex(textures, base_dir, rest); tid != 0 {
				m.roughness_map = tid
			}
		case "map_d":
			if tid := obj_load_tex(textures, base_dir, rest); tid != 0 {
				m.alpha_map = tid
				m.flags += {.Transparent}
			}
		case "map_Pm":
			if tid := obj_load_tex(textures, base_dir, rest); tid != 0 {
				m.metalness_map = tid
			}
		}
	}
}

@(private = "file")
obj_load_tex :: proc(textures: ^Textures, base_dir, raw_rest: string) -> Texture_Id {
	if textures == nil {
		return 0
	}
	// Skip optional MTL texture flags like `-bm 1.0` or `-s 1 1 1` and take the final path token.
	rem := strings.trim_space(raw_rest)
	if len(rem) == 0 {
		return 0
	}
	last_tok := rem
	for len(rem) > 0 {
		tok: string
		tok, rem = obj_next_token(rem)
		if len(tok) > 0 {
			last_tok = tok
		}
	}
	path := strings.concatenate({base_dir, "/", last_tok}, context.temp_allocator) if len(base_dir) > 0 else last_tok
	tid, ok := texture_load(textures, path)
	return tid if ok else 0
}

@(private = "file")
obj_build_primitive :: proc(
	geos: ^Geometries,
	textures: ^Textures,
	positions: []Vec3,
	vcolors: []Vec3,
	has_vcolor: []bool,
	uvs: []Vec2,
	normals: []Vec3,
	corners: []Obj_Corner,
	base_mat: Material,
) -> (
	geo_id: Geometry_Id,
	mat: Material,
) {
	mat = base_mat
	if mat.color == ({}) && mat.opacity == 0 && mat.roughness == 0 {
		mat = MATERIAL_DEFAULT
		mat.roughness = 0.8
	}

	c_count := len(corners)
	t_count := c_count / 3

	// Compute per-face normals and crease-angle smooth normals (60° threshold)
	// across shared position indices for any corner without an explicit `vn`.
	face_normals := make([]Vec3, t_count, context.temp_allocator)
	need_smooth := false
	for t in 0 ..< t_count {
		c0, c1, c2 := corners[t * 3], corners[t * 3 + 1], corners[t * 3 + 2]
		if c0.vn < 0 || c1.vn < 0 || c2.vn < 0 {
			need_smooth = true
		}
		p0, p1, p2 := positions[c0.v], positions[c1.v], positions[c2.v]
		fn := linalg.cross(p1 - p0, p2 - p0)
		face_normals[t] = fn if linalg.length2(fn) > 1e-20 else Vec3{0, 1, 0}
	}

	smooth_normals: []Vec3
	if need_smooth {
		smooth_normals = make([]Vec3, c_count, context.temp_allocator)
		v_count := len(positions)
		head := make([]i32, v_count, context.temp_allocator)
		for &h in head {
			h = -1
		}
		next_corner := make([]i32, c_count, context.temp_allocator)
		for ci in 0 ..< c_count {
			vi := corners[ci].v
			next_corner[ci] = head[vi]
			head[vi] = i32(ci)
		}
		face_unit := make([]Vec3, t_count, context.temp_allocator)
		for t in 0 ..< t_count {
			fn := face_normals[t]
			face_unit[t] = linalg.normalize(fn) if linalg.length2(fn) > 1e-20 else Vec3{0, 1, 0}
		}
		COS_CREASE :: f32(0.5) // cos(60°)
		for t in 0 ..< t_count {
			u_self := face_unit[t]
			for k in 0 ..< 3 {
				ci := t * 3 + k
				vi := corners[ci].v
				sum := Vec3{}
				it := head[vi]
				for it >= 0 {
					other_t := int(it) / 3
					if linalg.dot(u_self, face_unit[other_t]) >= COS_CREASE {
						sum += face_normals[other_t]
					}
					it = next_corner[it]
				}
				smooth_normals[ci] = linalg.normalize(sum) if linalg.length2(sum) > 1e-20 else u_self
			}
		}
	}

	// Check if corners use per-vertex `v x y z r g b` colours when no colour texture is bound.
	any_vcolor := false
	if mat.color_map == 0 {
		for c in corners {
			if has_vcolor[c.v] {
				any_vcolor = true
				break
			}
		}
	}

	palette := make([dynamic][4]u8, 0, 16, context.temp_allocator)
	corner_pal := make([]int, c_count if any_vcolor else 0, context.temp_allocator)
	if any_vcolor {
		for c, ci in corners {
			vc := vcolors[c.v] * mat.color
			rgba := [4]u8{
				u8(clamp(vc.r * 255.0 + 0.5, 0, 255)),
				u8(clamp(vc.g * 255.0 + 0.5, 0, 255)),
				u8(clamp(vc.b * 255.0 + 0.5, 0, 255)),
				255,
			}
			found := -1
			for p, pi in palette {
				if p == rgba {
					found = pi
					break
				}
			}
			if found < 0 {
				found = len(palette)
				append(&palette, rgba)
			}
			corner_pal[ci] = found
		}
		if len(palette) == 1 {
			p := palette[0]
			mat.color = {f32(p[0]) / 255.0, f32(p[1]) / 255.0, f32(p[2]) / 255.0}
			any_vcolor = false
		} else if len(palette) > 1 && textures != nil {
			pw := len(palette)
			pixels := make([]u8, pw * 4, context.temp_allocator)
			for p, pi in palette {
				pixels[pi * 4 + 0] = p[0]
				pixels[pi * 4 + 1] = p[1]
				pixels[pi * 4 + 2] = p[2]
				pixels[pi * 4 + 3] = p[3]
			}
			mat.color = {1, 1, 1}
			mat.color_map = texture_add(textures, pixels, pw, 1)
		}
	}

	out_verts := make([]Vertex, c_count, context.temp_allocator)
	out_indices := make([]u32, c_count, context.temp_allocator)
	pw_f := f32(max(len(palette), 1))
	for c, ci in corners {
		n := normals[c.vn] if c.vn >= 0 else smooth_normals[ci]
		uv := uvs[c.vt] if c.vt >= 0 else Vec2{}
		if any_vcolor && mat.color_map != 0 {
			uv = {(f32(corner_pal[ci]) + 0.5) / pw_f, 0.5}
		}
		out_verts[ci] = Vertex{
			position = positions[c.v],
			normal   = n,
			uv       = uv,
		}
		out_indices[ci] = u32(ci)
	}

	geo_id = geometry_add(geos, out_verts, out_indices)
	return geo_id, mat
}
