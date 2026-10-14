package o3d

import "core:fmt"
import "core:mem"
import "core:strings"

import "deps:oksl"

// Custom materials, three.js's ShaderMaterial: a vertex and a fragment
// shader in GLSL ES (three.js's GLSL 1 style with `attribute`, `varying`
// and `gl_FragColor`, or GLSL 3 with `in` / `out`), compiled once by oksl to
// SPIR-V and MSL for the GPU and to programs the software renderer runs,
// and referred to by id like geometry. Each stage gets three.js's prefix as
// an oksl prelude, made from the tables below: the built-in uniforms
// (`BUILTINS`), the vertex attributes (`VERTEX_ATTRIBUTES`) and the GLSL 1
// defines. A draw's uniforms are filled each frame: the built-ins at the
// offsets found when the shader was compiled, the material's own by name
// (`scene_uniforms`). The output is written as the shader leaves it (no
// colour conversion, as three.js's ShaderMaterial); a transparent material
// is blended over what is behind.

// 0 is the standard material.
Shader_Id :: distinct u32

Shader_Stage :: enum u8 {
	Vertex,
	Fragment,
}

// The uniforms three.js declares for a ShaderMaterial, filled per draw;
// `fragment` ones are declared in both stages.
Builtin :: enum u8 {
	Model_Matrix,
	Model_View_Matrix,
	Projection_Matrix,
	View_Matrix,
	Normal_Matrix, // of the model-view matrix, as three.js's
	Camera_Position,
	Is_Orthographic,
}

@(rodata)
BUILTINS := [Builtin]struct {
	name, type: string,
	fragment:   bool,
} {
	.Model_Matrix      = {"modelMatrix", "mat4", false},
	.Model_View_Matrix = {"modelViewMatrix", "mat4", false},
	.Projection_Matrix = {"projectionMatrix", "mat4", false},
	.View_Matrix       = {"viewMatrix", "mat4", true},
	.Normal_Matrix     = {"normalMatrix", "mat3", false},
	.Camera_Position   = {"cameraPosition", "vec3", true},
	.Is_Orthographic   = {"isOrthographic", "bool", true},
}

// The vertex attributes a shader may read, where they are in a `Vertex`.
Vertex_Attribute :: enum u8 {
	Position,
	Normal,
	Uv,
}

@(rodata)
VERTEX_ATTRIBUTES := [Vertex_Attribute]struct {
	name, type: string,
	offset:     uintptr,
	components: int,
} {
	.Position = {"position", "vec3", offset_of(Vertex, position), 3},
	.Normal   = {"normal", "vec3", offset_of(Vertex, normal), 3},
	.Uv       = {"uv", "vec2", offset_of(Vertex, uv), 2},
}

// Varying floats a shader may pass per vertex (15 vec4s and a bit, beyond
// what GLSL ES 3.0 guarantees).
MAX_VARYING_FLOATS :: 64

// A compiled shader and what its stages read, found once.
Shader :: struct {
	pipeline:       ^oksl.Pipeline, // nil: it failed to compile (`errors`)
	errors:         string,
	// Per stage, each built-in's offset in its uniform block (-1: unused).
	builtins:       [Shader_Stage][Builtin]i32,
	// Each attribute's location (-1: unused).
	attributes:     [Vertex_Attribute]i32,
	// The varyings: floats per vertex, and the ones that are flat (not
	// interpolated: the first vertex's, as the GPU takes them), listed so
	// the rasterizer's loops visit only those (`shader_flat`).
	varying_floats: int,
	flat:           [MAX_VARYING_FLOATS]u8,
	flat_count:     int,
}

Shaders :: struct {
	generation: Generation, // of these contents, for the GPU's pipelines
	list: [dynamic]Shader,
	keys: map[string]Shader_Id,
}

shaders_destroy :: proc(s: ^Shaders) {
	for &sh in s.list {
		oksl.destroy_pipeline(sh.pipeline)
		delete(sh.errors)
	}
	for k in s.keys {
		delete(k)
	}
	delete(s.list)
	delete(s.keys)
	s^ = {}
}

// The shader for these sources (`glsl3`: written in GLSL 3, three.js's
// `glslVersion: THREE.GLSL3`, declaring its own output), compiled the first
// time it is asked for. When it does not compile or reads what o3d does not
// provide, `errors` says why (`Vertex:3:7: ...`, kept with the shader) and
// draws with it are left out, by both renderers; the id is the shader's
// either way, so a broken one never stands in as the standard material.
shader :: proc(s: ^Shaders, vertex, fragment: string, glsl3 := false) -> (id: Shader_Id, errors: string) {
	key := strings.concatenate({vertex, "\x00", fragment, "\x01" if glsl3 else "\x00"}, context.temp_allocator)
	if found, ok := s.keys[key]; ok {
		return found, s.list[found - 1].errors
	}
	generation_take(&s.generation)
	append(&s.list, make_shader(vertex, fragment, glsl3))
	id = Shader_Id(len(s.list))
	s.keys[strings.clone(key)] = id
	return id, s.list[id - 1].errors
}

// The compiled shader, or nil (no such id, or it failed).
shader_get :: proc "contextless" (s: ^Shaders, id: Shader_Id) -> ^Shader {
	if s == nil || id == 0 || int(id) > len(s.list) || s.list[id - 1].pipeline == nil {
		return nil
	}
	return &s.list[id - 1]
}

// The varying floats that are flat (by offset in a vertex's record).
shader_flat :: proc "contextless" (sh: ^Shader) -> []u8 {
	return sh.flat[:sh.flat_count]
}

shader_module :: proc "contextless" (sh: ^Shader, stage: Shader_Stage) -> ^oksl.Module {
	return sh.pipeline.vertex if stage == .Vertex else sh.pipeline.fragment
}

@(private = "file")
make_shader :: proc(vertex, fragment: string, glsl3: bool) -> (sh: Shader) {
	p, errors, ok := oksl.compile_pipeline({vertex = vertex, fragment = fragment, vertex_prelude = prelude(.Vertex, glsl3), fragment_prelude = prelude(.Fragment, glsl3)})
	fail :: proc(sh: ^Shader, p: ^oksl.Pipeline, text: string) {
		oksl.destroy_pipeline(p)
		sh.pipeline = nil
		sh.errors = strings.clone(text)
	}
	if !ok {
		fail(&sh, p, oksl.format_errors(errors, context.temp_allocator))
		return
	}
	sh.pipeline = p
	sh.attributes = {.Position = -1, .Normal = -1, .Uv = -1}
	for &offsets, stage in sh.builtins {
		m := shader_module(&sh, stage)
		if m.num_samplers > 0 {
			fail(&sh, p, "o3d: textures are not supported yet\n")
			return
		}
		for b in Builtin {
			offset, _, found := oksl.uniform_offset(m, BUILTINS[b].name)
			offsets[b] = i32(offset) if found else -1
		}
	}
	inputs: for v in p.vertex.stage_inputs {
		for a in Vertex_Attribute {
			if v.name == VERTEX_ATTRIBUTES[a].name {
				sh.attributes[a] = i32(v.location)
				continue inputs
			}
		}
		fail(&sh, p, fmt.tprintf("Vertex: attribute %s is not one o3d provides (position, normal, uv)\n", v.name))
		return
	}
	if len(p.fragment.stage_outputs) == 0 {
		fail(&sh, p, "Fragment: the shader writes no colour\n")
		return
	}
	sh.varying_floats = p.vertex.varying_floats
	if sh.varying_floats > MAX_VARYING_FLOATS {
		fail(&sh, p, fmt.tprintf("Vertex: %d varying floats, more than o3d's %d\n", sh.varying_floats, MAX_VARYING_FLOATS))
		return
	}
	for v in p.vertex.stage_outputs {
		if v.flat {
			for k in 0 ..< type_floats(v.type) {
				sh.flat[sh.flat_count] = u8(v.offset + k)
				sh.flat_count += 1
			}
		}
	}
	return
}

// three.js's prefix for a ShaderMaterial stage.
@(private = "file")
prelude :: proc(stage: Shader_Stage, glsl3: bool) -> string {
	b := strings.builder_make(context.temp_allocator)
	strings.write_string(&b, "precision highp float;\nprecision highp int;\n")
	if stage == .Vertex {
		strings.write_string(&b, "#define attribute in\n#define varying out\n")
	} else {
		strings.write_string(&b, "#define varying in\n")
		if !glsl3 {
			strings.write_string(&b, "layout(location = 0) out highp vec4 pc_fragColor;\n#define gl_FragColor pc_fragColor\n")
		}
	}
	strings.write_string(&b, "#define texture2D texture\n#define texture2DLodEXT textureLod\n#define texture2DProj textureProj\n")
	for row in BUILTINS {
		if stage == .Vertex || row.fragment {
			fmt.sbprintf(&b, "uniform %s %s;\n", row.type, row.name)
		}
	}
	if stage == .Vertex {
		for row in VERTEX_ATTRIBUTES {
			fmt.sbprintf(&b, "in %s %s;\n", row.type, row.name)
		}
	}
	return strings.to_string(b)
}

@(private = "file")
type_floats :: proc "contextless" (t: oksl.Type) -> int {
	return int(t.rows) * max(int(t.cols), 1)
}

// ---- uniforms per frame -----------------------------------------------------------------

// The frame's uniform blocks: per draw and stage, where its block starts in
// `bytes` (-1: no shader, or the stage reads no uniforms). The GPU pushes
// these bytes; the software renderer runs the programs on them.
Shader_Blocks :: struct {
	bytes: [dynamic]u8,
	at:    [dynamic][Shader_Stage]i32,
}

shader_blocks_destroy :: proc(b: ^Shader_Blocks) {
	delete(b.bytes)
	delete(b.at)
	b^ = {}
}

// Fills every custom draw's blocks: the built-ins its stages read, then the
// material's uniforms by name.
shader_blocks :: proc(scene: ^Scene, shaders: ^Shaders, view, proj: Mat4, out: ^Shader_Blocks) {
	n := len(scene.draws)
	clear(&out.bytes)
	table_grow_non_zero(&out.at, n)
	materials := scene_materials(scene)
	models := scene.draws.model[:n]
	for m, i in materials {
		out.at[i] = {.Vertex = -1, .Fragment = -1}
		sh := shader_get(shaders, m.shader)
		if sh == nil {
			continue
		}
		model_view := view * models[i]
		for stage in Shader_Stage {
			mod := shader_module(sh, stage)
			if mod.block_size == 0 {
				continue
			}
			at := len(out.bytes)
			table_grow(&out.bytes, at + mod.block_size)
			block := out.bytes[at:]
			out.at[i][stage] = i32(at)
			for offset, b in sh.builtins[stage] {
				if offset < 0 {
					continue
				}
				dst := block[offset:]
				switch b {
				case .Model_Matrix:
					put(dst, models[i])
				case .Model_View_Matrix:
					put(dst, model_view)
				case .Projection_Matrix:
					put(dst, proj)
				case .View_Matrix:
					put(dst, view)
				case .Normal_Matrix:
					// std140: three columns of a vec4 each.
					nm := normal_matrix(model_view)
					for c in 0 ..< 3 {
						put(dst[c * 16:], [4]f32{nm[0, c], nm[1, c], nm[2, c], 0})
					}
				case .Camera_Position:
					put(dst, scene.camera.position)
				case .Is_Orthographic:
					put(dst, u32(1 if scene.camera.orthographic else 0))
				}
			}
			for u in scene.uniforms[m.uniforms.first:][:m.uniforms.count] {
				write_uniform(mod, block, u.name, scene.uniform_floats[u.first:][:u.count])
			}
		}
	}
}

@(private = "file")
put :: #force_inline proc(dst: []u8, value: $T) {
	value := value
	mem.copy(raw_data(dst), &value, size_of(T))
}

// A uniform's components by name; an array takes its elements in order.
// Names the stage does not read are skipped.
@(private = "file")
write_uniform :: proc(m: ^oksl.Module, block: []u8, name: string, values: []f32) {
	for info in m.uniforms {
		if info.name != name {
			continue
		}
		if info.count == 0 {
			oksl.set_uniform(m, block, name, values)
			return
		}
		elem := info.type
		elem.array = 0
		per := type_floats(elem)
		for e in 0 ..< min(int(info.count), len(values) / max(per, 1)) {
			oksl.set_uniform(m, block, name, values[e * per:][:per], e)
		}
		return
	}
}
