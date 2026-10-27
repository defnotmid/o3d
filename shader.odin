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
// is blended over what is behind. The fragment prelude declares o3d's
// functions a shader calls (`Prelude_Function`): the scene's shadows, tone
// mapping, fog and output encoding.

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
	// An instanced draw's instance: its colour (linear, as three.js's;
	// white without one) and 0-based index.
	Instance_Color,
	Instance_Id,
	// `shadowAt`'s: the first shadow-casting light's view-projection, and
	// its map's size, bias, normal bias and 1 (0: no light casts).
	Shadow_Matrix,
	Shadow_Params,
	// `toneMapping`'s: the scene's mode (`Tone_Mapping`) and exposure.
	Tone_Params,
	// `fogged`'s: the fog's linear colour; its kind (`Fog_Kind`), near, far
	// and density (`fog_terms`).
	Fog_Color,
	Fog_Params,
}

@(rodata)
BUILTINS := [Builtin]struct {
	name, type: string,
	fragment:   bool,
	// Declared only for these functions (none: always).
	functions:  bit_set[Prelude_Function],
} {
	.Model_Matrix      = {"modelMatrix", "mat4", false, {}},
	.Model_View_Matrix = {"modelViewMatrix", "mat4", false, {}},
	.Projection_Matrix = {"projectionMatrix", "mat4", false, {}},
	.View_Matrix       = {"viewMatrix", "mat4", true, {}},
	.Normal_Matrix     = {"normalMatrix", "mat3", false, {}},
	.Camera_Position   = {"cameraPosition", "vec3", true, {}},
	.Is_Orthographic   = {"isOrthographic", "bool", true, {}},
	.Instance_Color    = {"instanceColor", "vec3", true, {}},
	.Instance_Id       = {"instanceId", "float", true, {}},
	.Shadow_Matrix     = {"o3dShadowMatrix", "mat4", true, {.Shadow_At}},
	.Shadow_Params     = {"o3dShadowParams", "vec4", true, {.Shadow_At}},
	.Tone_Params       = {"o3dToneParams", "vec4", true, {.Tone_Mapping}},
	.Fog_Color         = {"o3dFogColor", "vec3", true, {.Fogged}},
	.Fog_Params        = {"o3dFogParams", "vec4", true, {.Fogged}},
}

// The functions the fragment prelude declares for a shader that calls them
// (its source names them), with the built-ins they read, so a
// ShaderMaterial can finish its colour as the standard material does:
//
//	gl_FragColor = linearToOutputTexel(vec4(fogged(toneMapping(c), worldPosition), a));
//
// (o3d's standard order: tone mapping, fog in linear, then sRGB; three.js
// fogs after encoding). three.js's names where it has the function.
Prelude_Function :: enum u8 {
	// `float shadowAt(vec3 worldPosition, vec3 worldNormal)` (SHADOW_PRELUDE).
	Shadow_At,
	// `vec3 toneMapping(vec3 color)`: the scene's tone mapping and exposure
	// (`tone_map`), linear to linear; as is without tone mapping.
	Tone_Mapping,
	// `vec3 fogged(vec3 color, vec3 worldPosition)`: towards the scene's
	// fog's colour by the distance from the camera (`fog_factor`), linear.
	Fogged,
	// `vec4 linearToOutputTexel(vec4 value)`: linear to sRGB, as o3d writes
	// the standard material (clamped to 0..1), alpha as is.
	Linear_To_Output,
}

@(rodata)
PRELUDE_FUNCTIONS := [Prelude_Function]struct {
	name, glsl: string,
} {
	.Shadow_At        = {"shadowAt", SHADOW_PRELUDE},
	.Tone_Mapping     = {"toneMapping", TONE_PRELUDE},
	.Fogged           = {"fogged", FOG_PRELUDE},
	.Linear_To_Output = {"linearToOutputTexel", OUTPUT_PRELUDE},
}

// The shadow map's sampler `shadowAt` reads (a channel of its own).
SHADOW_SAMPLER :: "o3dShadowMap"

// A fragment shader that calls `shadowAt(worldPosition, worldNormal)` gets
// it: how lit the point is by the first shadow-casting light, 0 to 1, as
// the standard material's shadows (shadow_factor: the normal bias, the
// bias, three.js's PCF over 4 x 4 texels), so a ShaderMaterial receives
// shadows (three.js's getShadow with lights: true). Its map takes the
// first sampler channel; `shader_map` finds the shader's own by name.
@(private = "file")
SHADOW_PRELUDE :: `uniform sampler2D o3dShadowMap;
float shadowAt(vec3 worldPosition, vec3 worldNormal) {
	if (o3dShadowParams.w == 0.0) {
		return 1.0;
	}
	vec4 c = o3dShadowMatrix * vec4(worldPosition + normalize(worldNormal) * o3dShadowParams.z, 1.0);
	if (c.w <= 1e-6) {
		return 1.0;
	}
	float u = c.x / c.w * 0.5 + 0.5;
	float v = 0.5 - c.y / c.w * 0.5;
	float z = c.z / c.w - o3dShadowParams.y;
	if (u < 0.0 || u > 1.0 || v < 0.0 || v > 1.0 || z < 0.0 || z > 1.0) {
		return 1.0;
	}
	float size = o3dShadowParams.x;
	float s = u * size - 0.5;
	float t = v * size - 0.5;
	float x0 = floor(s);
	float y0 = floor(t);
	float fx = s - x0;
	float fy = t - y0;
	float lit = 0.0;
	for (int j = 0; j < 4; j++) {
		float row = clamp(y0 - 1.0 + float(j), 0.0, size - 1.0);
		float wy = j == 0 ? 1.0 - fy : (j == 3 ? fy : 1.0);
		float acc = 0.0;
		for (int i = 0; i < 4; i++) {
			float col = clamp(x0 - 1.0 + float(i), 0.0, size - 1.0);
			float wx = i == 0 ? 1.0 - fx : (i == 3 ? fx : 1.0);
			// The texel's centre, rows from the top (the macro flips v).
			float d = textureLod(o3dShadowMap, vec2((col + 0.5) / size, 1.0 - (row + 0.5) / size), 0.0).r;
			acc += wx * (z >= d ? 1.0 : 0.0);
		}
		lit += wy * acc;
	}
	return lit / 9.0;
}
`

// tone_map's modes, as o3d.odin's and shaders/common.slang's.
@(private = "file")
TONE_PRELUDE :: `vec3 toneMapping(vec3 color) {
	float mode = o3dToneParams.x;
	if (mode == 0.0) {
		return color;
	}
	vec3 v = color * o3dToneParams.y;
	if (mode == 1.0) {
		return clamp(v, 0.0, 1.0);
	}
	if (mode == 2.0) {
		return clamp(v / (vec3(1.0) + v), 0.0, 1.0);
	}
	if (mode == 3.0) {
		vec3 x = max(v - vec3(0.004), vec3(0.0));
		vec3 u = (x * (6.2 * x + vec3(0.5))) / (x * (6.2 * x + vec3(1.7)) + vec3(0.06));
		return pow(u, vec3(2.2));
	}
	if (mode == 4.0) {
		v *= 1.0 / 0.6;
		v = vec3(dot(vec3(0.59719, 0.35458, 0.04823), v), dot(vec3(0.07600, 0.90834, 0.01566), v), dot(vec3(0.02840, 0.13383, 0.83777), v));
		vec3 a = v * (v + vec3(0.0245786)) - vec3(0.000090537);
		vec3 b = v * (0.983729 * v + vec3(0.4329510)) + vec3(0.238081);
		v = a / b;
		v = vec3(dot(vec3(1.60475, -0.53108, -0.07367), v), dot(vec3(-0.10208, 1.10813, -0.00605), v), dot(vec3(-0.00327, -0.07276, 1.07602), v));
		return clamp(v, 0.0, 1.0);
	}
	if (mode == 5.0) {
		// Linear Rec.709 -> Rec.2020 -> AgX's inset, the curve, its outset.
		vec3 r = vec3(dot(vec3(0.6274039, 0.3292830, 0.0433131), v), dot(vec3(0.0690973, 0.9195404, 0.0113623), v), dot(vec3(0.0163914, 0.0880133, 0.8955953), v));
		r = max(r, vec3(1e-10));
		vec3 inset = vec3(dot(vec3(0.85662715, 0.09512124, 0.04825161), r), dot(vec3(0.13731897, 0.76124199, 0.10143904), r), dot(vec3(0.11189821, 0.07679942, 0.81130237), r));
		inset = max(inset, vec3(1e-10));
		vec3 x = clamp((log2(inset) + vec3(12.47393)) * (1.0 / 16.5), 0.0, 1.0);
		vec3 x2 = x * x;
		vec3 x4 = x2 * x2;
		x = 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - vec3(0.00232);
		vec3 o = vec3(dot(vec3(1.12710058, -0.11060664, -0.01649394), x), dot(vec3(-0.14132976, 1.15782370, -0.01649394), x), dot(vec3(-0.14132976, -0.11060664, 1.25193641), x));
		o = pow(max(o, vec3(0.0)), vec3(2.2));
		return clamp(vec3(dot(vec3(1.6604910, -0.5876411, -0.0728499), o), dot(vec3(-0.1245505, 1.1328999, -0.0083494), o), dot(vec3(-0.0181508, -0.1005789, 1.1187297), o)), 0.0, 1.0);
	}
	// Khronos PBR Neutral.
	float lo = min(v.r, min(v.g, v.b));
	float offset = lo < 0.08 ? lo - 6.25 * lo * lo : 0.04;
	v -= vec3(offset);
	float peak = max(v.r, max(v.g, v.b));
	if (peak < 0.76) {
		return clamp(v, 0.0, 1.0);
	}
	float newPeak = 1.0 - 0.24 * 0.24 / (peak + 0.24 - 0.76);
	v *= newPeak / peak;
	float g = 1.0 - 1.0 / (0.15 * (peak - newPeak) + 1.0);
	return clamp(mix(v, vec3(newPeak), g), 0.0, 1.0);
}
`

// fog_factor's, the terms resolved (fog_terms).
@(private = "file")
FOG_PRELUDE :: `vec3 fogged(vec3 color, vec3 worldPosition) {
	float dist = length(worldPosition - cameraPosition);
	float f = 0.0;
	if (o3dFogParams.x == 1.0) {
		f = clamp((dist - o3dFogParams.y) / max(o3dFogParams.z - o3dFogParams.y, 1e-6), 0.0, 1.0);
	} else if (o3dFogParams.x == 2.0) {
		float d = o3dFogParams.w * dist;
		f = clamp(1.0 - exp(-d * d), 0.0, 1.0);
	}
	return mix(color, o3dFogColor, f);
}
`

// linear_to_srgb's.
@(private = "file")
OUTPUT_PRELUDE :: `float o3dSrgb(float c) {
	c = clamp(c, 0.0, 1.0);
	return c <= 0.0031308 ? c * 12.92 : 1.055 * pow(c, 1.0 / 2.4) - 0.055;
}
vec4 linearToOutputTexel(vec4 value) {
	return vec4(o3dSrgb(value.r), o3dSrgb(value.g), o3dSrgb(value.b), value.a);
}
`

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

// A ShaderMaterial's textures: `Material.shader_maps`, one per oksl
// channel (its `uniform sampler2D`s in declaration order).
SHADER_MAPS :: oksl.CHANNEL_COUNT

// Why a shader cannot draw: oksl's diagnostics, or what o3d does not
// provide. Values (a code and its operands); text only at the edge
// (`shader_error_message`, `format_shader_errors`).
Shader_Error_Code :: enum u8 {
	None,
	// oksl's diagnostic (`compile`: its position, code and operands).
	Compile,
	// The vertex stage samples a texture (`name`: the sampler).
	Vertex_Sampler,
	// A vertex attribute o3d does not provide (`name`).
	Unknown_Attribute,
	// The fragment stage writes no colour.
	No_Colour,
	// `n` varying floats, more than `m` (MAX_VARYING_FLOATS).
	Too_Many_Varyings,
}

Shader_Error :: struct {
	code:    Shader_Error_Code,
	stage:   Shader_Stage,
	compile: oksl.Error,
	name:    string,
	n, m:    i64,
}

// A shader error's text, without its stage and position.
shader_error_message :: proc(e: Shader_Error, allocator := context.allocator) -> string {
	switch e.code {
	case .None:
		return ""
	case .Compile:
		return oksl.error_message(e.compile, allocator)
	case .Vertex_Sampler:
		return fmt.aprintf("sampler %s: o3d samples textures in fragment shaders only", e.name, allocator = allocator)
	case .Unknown_Attribute:
		return fmt.aprintf("attribute %s is not one o3d provides (position, normal, uv)", e.name, allocator = allocator)
	case .No_Colour:
		return fmt.aprintf("the shader writes no colour", allocator = allocator)
	case .Too_Many_Varyings:
		return fmt.aprintf("%d varying floats, more than o3d's %d", e.n, e.m, allocator = allocator)
	}
	return ""
}

// "Stage:line:col: message" lines (o3d's own: "Stage: message"), for printing.
format_shader_errors :: proc(errors: []Shader_Error, allocator := context.allocator) -> string {
	b := strings.builder_make(allocator)
	for e in errors {
		if e.code == .Compile {
			strings.write_string(&b, oksl.format_errors({e.compile}, context.temp_allocator))
			continue
		}
		fmt.sbprintf(&b, "%v: %s\n", e.stage, shader_error_message(e, context.temp_allocator))
	}
	return strings.to_string(b)
}

// A compiled shader and what its stages read, found once.
Shader :: struct {
	// Kept when it failed too: `errors`' operands are its strings.
	pipeline:       ^oksl.Pipeline,
	// Why it cannot draw (none: it can).
	errors:         []Shader_Error,
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
	// `shadowAt`'s map's channel (-1: the shader does not call it).
	shadow_channel: i32,
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
// provide, `errors` says why (values kept with the shader: oksl's
// diagnostics with their stage and position, or o3d's; text by
// `format_shader_errors`) and
// draws with it are left out, by both renderers; the id is the shader's
// either way, so a broken one never stands in as the standard material.
shader :: proc(s: ^Shaders, vertex, fragment: string, glsl3 := false) -> (id: Shader_Id, errors: []Shader_Error) {
	key := shader_key(vertex, fragment, glsl3)
	if found, ok := s.keys[key]; ok {
		return found, s.list[found - 1].errors
	}
	generation_take(&s.generation)
	append(&s.list, make_shader(vertex, fragment, glsl3))
	id = Shader_Id(len(s.list))
	s.keys[strings.clone(key)] = id
	return id, s.list[id - 1].errors
}

// The shader for these sources if `shader` has made it, compiled or not
// (`errors` says why not), without making it: only reads `s`, so a host
// may look up under a shared lock and make under an exclusive one.
shader_find :: proc(s: ^Shaders, vertex, fragment: string, glsl3 := false) -> (id: Shader_Id, errors: []Shader_Error, found: bool) {
	id, found = s.keys[shader_key(vertex, fragment, glsl3)]
	if !found {
		return 0, nil, false
	}
	return id, s.list[id - 1].errors, true
}

@(private = "file")
shader_key :: proc(vertex, fragment: string, glsl3: bool) -> string {
	return strings.concatenate({vertex, "\x00", fragment, "\x01" if glsl3 else "\x00"}, context.temp_allocator)
}

// The compiled shader, or nil (no such id, or it failed).
shader_get :: proc "contextless" (s: ^Shaders, id: Shader_Id) -> ^Shader {
	if s == nil || id == 0 || int(id) > len(s.list) || len(s.list[id - 1].errors) > 0 {
		return nil
	}
	return &s.list[id - 1]
}

// The channel of the shader's `uniform sampler2D` named `name`: its index
// in `Material.shader_maps` (-1: no such sampler). Found once, kept.
shader_map :: proc "contextless" (s: ^Shaders, id: Shader_Id, name: string) -> int {
	sh := shader_get(s, id)
	if sh == nil || sh.pipeline == nil {
		return -1
	}
	for n, i in sh.pipeline.fragment.channel_names {
		if n == name && n != "" {
			return i
		}
	}
	return -1
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
	functions: bit_set[Prelude_Function]
	for f in Prelude_Function {
		if strings.contains(fragment, PRELUDE_FUNCTIONS[f].name) {
			functions += {f}
		}
	}
	p, errors, ok := oksl.compile_pipeline({vertex = vertex, fragment = fragment, vertex_prelude = prelude(.Vertex, glsl3, {}), fragment_prelude = prelude(.Fragment, glsl3, functions)})
	sh.pipeline = p
	fail :: proc(sh: ^Shader, e: Shader_Error) {
		sh.errors = make([]Shader_Error, 1)
		sh.errors[0] = e
	}
	if !ok {
		sh.errors = make([]Shader_Error, len(errors))
		for e, i in errors {
			stage := Shader_Stage.Vertex if e.pos.src == .Vertex || e.pos.src == .Vertex_Prelude else .Fragment
			sh.errors[i] = {code = .Compile, stage = stage, compile = e}
		}
		return
	}
	sh.attributes = {.Position = -1, .Normal = -1, .Uv = -1}
	sh.shadow_channel = -1
	for n, i in p.fragment.channel_names {
		if n == SHADOW_SAMPLER {
			sh.shadow_channel = i32(i)
		}
	}
	for &offsets, stage in sh.builtins {
		m := shader_module(&sh, stage)
		if stage == .Vertex && m.num_samplers > 0 {
			fail(&sh, {code = .Vertex_Sampler, stage = .Vertex, name = m.channel_names[0]})
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
		fail(&sh, {code = .Unknown_Attribute, stage = .Vertex, name = v.name})
		return
	}
	if len(p.fragment.stage_outputs) == 0 {
		fail(&sh, {code = .No_Colour, stage = .Fragment})
		return
	}
	sh.varying_floats = p.vertex.varying_floats
	if sh.varying_floats > MAX_VARYING_FLOATS {
		fail(&sh, {code = .Too_Many_Varyings, stage = .Vertex, n = i64(sh.varying_floats), m = MAX_VARYING_FLOATS})
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

// three.js's prefix for a ShaderMaterial stage, with the functions the
// shader calls.
@(private = "file")
prelude :: proc(stage: Shader_Stage, glsl3: bool, functions: bit_set[Prelude_Function]) -> string {
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
	if stage == .Fragment {
		// three.js's uv runs up the image (flipY); o3d's images and the
		// GPU's sampling run down it.
		strings.write_string(&b, "#define texture(s, c) texture(s, vec2((c).x, 1.0 - (c).y))\n")
		strings.write_string(&b, "#define textureLod(s, c, l) textureLod(s, vec2((c).x, 1.0 - (c).y), l)\n")
		strings.write_string(&b, "#define texelFetch(s, p, l) texelFetch(s, ivec2((p).x, textureSize(s, l).y - 1 - (p).y), l)\n")
	}
	for row in BUILTINS {
		if (stage == .Vertex || row.fragment) && (row.functions == {} || row.functions & functions != {}) {
			fmt.sbprintf(&b, "uniform %s %s;\n", row.type, row.name)
		}
	}
	for f in functions {
		strings.write_string(&b, PRELUDE_FUNCTIONS[f].glsl)
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

// Fills every custom draw's blocks: the built-ins its stages read (the
// first of the frame's `shadows` for `shadowAt`), then the material's
// uniforms by name. `eye` (the camera's position when nil) is
// cameraPosition: the mirror's pass draws through its image.
shader_blocks :: proc(scene: ^Scene, shaders: ^Shaders, view, proj: Mat4, shadows: []Shadow, out: ^Shader_Blocks, eye: Maybe(Vec3) = nil) {
	camera_position := eye.? or_else scene.camera.position
	fog_near, fog_far, fog_density := fog_terms(scene.fog)
	tone_params := [4]f32{f32(scene.tone_mapping), scene.tone_mapping_exposure if scene.tone_mapping_exposure > 0 else 1, 0, 0}
	fog_params := [4]f32{f32(scene.fog.kind), fog_near, fog_far, fog_density}
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
					put(dst, camera_position)
				case .Is_Orthographic:
					put(dst, u32(1 if scene.camera.orthographic else 0))
				case .Instance_Color:
					put(dst, linear(m.color))
				case .Instance_Id:
					put(dst, f32(scene.draw_instance_ids[i] if i < len(scene.draw_instance_ids) else 0))
				case .Shadow_Matrix:
					put(dst, shadows[0].view_proj if len(shadows) > 0 else Mat4{})
				case .Shadow_Params:
					if len(shadows) > 0 {
						put(dst, [4]f32{f32(shadows[0].size), shadows[0].bias, shadows[0].normal_bias, 1})
					} else {
						put(dst, [4]f32{})
					}
				case .Tone_Params:
					put(dst, tone_params)
				case .Fog_Color:
					put(dst, linear(scene.fog.color))
				case .Fog_Params:
					put(dst, fog_params)
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
