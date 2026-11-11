package o3d

import "core:strings"
import "deps:oksl"
import "core:testing"

// Every entry, geometry and material flag, each flag alone and in the
// combinations that change the shading's shape, under frames with no light,
// with every kind of light and shadows, an environment and fog: each
// variant compiles, and its blocks' fields fit their structs.
@(test)
test_program_variants :: proc(t: ^testing.T) {
	p: Programs
	defer programs_destroy(&p)
	lights := []Shading_Light{{kind = .Ambient}, {kind = .Hemisphere}, {kind = .Directional, shadowed = 1}, {kind = .Point}, {kind = .Spot, shadowed = 2}}
	frames := [?]Frame_Shape {
		{},
		frame_shape(lights, 2, {radiance = 1, intensity = 1}, true, .Exp2),
		frame_shape(lights[:3], 1, {radiance = 1, intensity = 1}, false, .Linear),
	}
	combos := [?]Shading_Flags {
		{},
		{.Basic},
		{.Shadow_Only, .Blended},
		{.Toon},
		{.Flat_Shading, .Two_Sided},
		{.Color_Map, .Normal_Map, .Roughness_Map, .Metalness_Map, .Ao_Map, .Emissive_Map, .Alpha_Map, .Alpha_Mask},
		{.Color_Map, .Premultiplied, .Blended, .Vertex_Colors},
		{.Triplanar, .Color_Map, .Normal_Map},
		{.Clearcoat, .Sheen, .Layer_Maps, .Clearcoat_Normal_Map},
		{.Transmission, .Layer_Maps},
		{.Reflection, .Clearcoat, .Receive_Shadow},
		{.Additive, .Blended},
	}
	compiled := 0
	for frame in frames {
		for flags in combos {
			for entry in Program_Entry {
				f := flags
				geometries: []Program_Geometry
				switch entry {
				case .Sprite:
					f += {.Sprite, .Blended}
					geometries = {.Quads}
				case .Curves, .Curves_Mirror:
					f += {.Curves}
					geometries = {.Quads}
				case .Main, .Cut, .Mirror:
					geometries = {.Mesh, .Skinned}
				case .Shadow, .Velocity:
					continue // depth_key's, below
				}
				for g in geometries {
					// Layer maps in both textures and none; uv1 for a few slots.
					key := Program_Key{flags = f, uv_sets = 0b101, layer_slots = 0x0211021, entry = entry, geometry = g, frame = frame}
					id := program_get(&p, key)
					testing.expectf(t, id >= 0, "%v does not compile", key)
					compiled += 1
				}
			}
		}
	}
	// The depth passes: masked or not, exact shapes, each geometry.
	for flags in ([?]Shading_Flags{{}, {.Alpha_Mask, .Color_Map}, {.Alpha_Mask, .Alpha_Map}, {.Curves}}) {
		for velocity in ([2]bool{false, true}) {
			for skinned in ([2]bool{false, true}) {
				m := Shading_Material{flags = flags, uv_sets = 1}
				key := depth_key(&m, skinned, velocity)
				testing.expectf(t, program_get(&p, key) >= 0, "%v does not compile", key)
				compiled += 1
			}
		}
	}
	for pass in Pass {
		testing.expectf(t, pass_get(&p, pass) != nil, "the %v pass does not compile", pass)
	}
	testing.expect(t, compiled > 300)
}

// An opaque draw's shader never discards: on a tile-based GPU that costs
// hidden-surface removal even when no fragment is ever cut. A cut draw's
// does (a masked material's cutoff, the mirror's plane).
@(test)
test_opaque_variant_never_discards :: proc(t: ^testing.T) {
	p: Programs
	defer programs_destroy(&p)
	flags := Shading_Flags{.Color_Map, .Alpha_Mask}
	main := program_get(&p, {flags = flags, entry = .Main})
	cut := program_get(&p, {flags = flags, entry = .Cut})
	if !testing.expect(t, main >= 0 && cut >= 0) {
		return
	}
	testing.expect(t, !strings.contains(string(p.list[main].pipeline.fragment.msl), "discard_fragment"))
	testing.expect(t, strings.contains(string(p.list[cut].pipeline.fragment.msl), "discard_fragment"))
}

// One fragment of `m`'s variant (no maps) lit by `lights`, at `world` with
// `normal`, seen from `eye`, through oksl's CPU runner: its light,
// premultiplied (a check of the shading's formulas alone).
shade_fragment :: proc(p: ^Programs, m: ^Shading_Material, lights: []Shading_Light, world, normal, eye: Vec3) -> Vec4 {
	id := program_get(p, program_key(m, false, false, frame_shape(lights, 0, {}, false, .None)))
	if id < 0 {
		return {}
	}
	x := &p.list[id]
	mb := material_block(m)
	fb: Frame_Block
	frame_block(&fb, {near = 0.1, far = 100}, eye, 1, 1, 1, lights, nil, {}, {})
	block: [size_of(Material_Block) + size_of(Frame_Block)]u8
	block_fill(&x.fragment, {&mb, &fb}, block[:])
	// One quad, every lane the same point: its record by the inputs' names.
	module := x.pipeline.fragment
	values := make([]f32, module.varying_floats * 4, context.temp_allocator)
	for in_ in module.stage_inputs {
		v: [4]f32
		switch in_.name {
		case "v_world":
			v.xyz = world
		case "v_normal":
			v.xyz = normal
		case "v_color":
			v = 1
		}
		for k in 0 ..< int(in_.type.rows) {
			for l in 0 ..< 4 {
				values[(in_.offset + k) * 4 + l] = v[k]
			}
		}
	}
	out: [4][4]f32
	shaded: [1]oksl.Quad_Lanes
	scratch: oksl.Scratch
	defer oksl.scratch_destroy(&scratch)
	origin := [1][2]i32{}
	targets := [1][][4]f32{out[:]}
	oksl.run_fragment(&module.program, block[:x.fragment.size], {count = 1, origin = origin[:], varyings = values}, targets[:], shaded[:], &scratch)
	return out[0]
}

// `c` through the Output pass (shaders/post.frag's tone map) as one texel
// of full coverage, tone mapped by `mode` and `exposure`, its sRGB encoding
// decoded.
tone_mapped :: proc(p: ^Programs, c: Vec3, mode: Tone_Mapping, exposure: f32) -> Vec3 {
	scratches: [dynamic]oksl.Scratch
	defer {
		for &sc in scratches {
			oksl.scratch_destroy(&sc)
		}
		delete(scratches)
	}
	light := [1][4]f32{{c.r, c.g, c.b, 1}}
	out: [1][4]f32
	block := Post_Block {
		tone  = {f32(mode), exposure, 1, 1},
		frame = {1, 1, 0, 0},
	}
	maps := #partial Post_Images {
		.Light = {f4 = light[:], width = 1, height = 1},
	}
	targets := [1][][4]f32{out[:]}
	post_run(p, &scratches, .Output, &block, maps, targets[:], 1, 1, nil)
	return {srgb_to_linear_exact(out[0].r), srgb_to_linear_exact(out[0].g), srgb_to_linear_exact(out[0].b)}
}
