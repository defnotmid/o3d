// o3d's fragment stage for its own draws (oksl GLSL, compiled at load per
// variant, shading.odin): three.js's MeshStandardMaterial and
// MeshPhysicalMaterial lit by the frame's lights, shadows and environment,
// fogged, as light premultiplied by its coverage. The frame is tone mapped
// and encoded after, a pixel at a time.
//
// The prelude (fragment_prelude) fixes the variant's shape: ENTRY (below),
// the material's flags (FLAG_<NAME>), the uv set and transform rows of each
// map's slot (UV_<SLOT> its varying, DUV_<SLOT> its derivatives,
// ROW0_<SLOT>, ROW1_<SLOT>), the uv sets the variant reads (UV0, UV1),
// where each layer map is (LAYER_<MAP>), and the frame's lights (O3D_LIGHTS), shadow maps
// (SHADOW_MAPS, RECEIVE_SHADOWS, SHADOW_ONLY_MASK), environment (ENV,
// ENV_HDR) and fog (FOG). It declares the uniforms (Shading_Material's and
// Shading_Frame's fields as u_<field>), O3D_COMMON and, for exact shapes,
// ocurves' curves.GLSL.
//
// ENTRY:
//   0  a draw that replaces what is behind it (no discard)
//   1  one that may cut: masked, blended or added
//   2  the mirror's pass (cut; its run into a second target)
//   3  a particle quad: its row's colour, opacity and smear
//   4  an exact shape: its coverage from its curves
//   5  an exact shape in the mirror's pass

#define TRANSMIT (ENTRY != 3)
#define CUT (ENTRY != 0)

#if TRANSMIT && defined(FLAG_TRANSMISSION)
#define GLASS 1
#endif
#if TRANSMIT && defined(FLAG_REFLECTION)
#define REFLECTIVE 1
#endif

// Linked to o3d.vert's outputs by name.
in vec3 v_world;
in vec3 v_normal;
#ifdef UV0
in vec2 v_uv;
#endif
#ifdef UV1
in vec2 v_uv1;
#endif
in vec4 v_color;
#if ENTRY >= 3
flat in uint v_quad;
#endif

layout(location = 0) out vec4 o_light;
#if ENTRY == 2 || ENTRY == 5
layout(location = 1) out vec4 o_run;
#endif

#ifdef FLAG_COLOR_MAP
uniform sampler2D colorMap;
#endif
#if defined(LIT) && defined(FLAG_NORMAL_MAP)
uniform sampler2D normalMap;
#endif
#if defined(LIT) && defined(FLAG_ROUGHNESS_MAP)
uniform sampler2D roughnessMap;
#endif
#ifdef FLAG_EMISSIVE_MAP
uniform sampler2D emissiveMap;
#endif
#if defined(LIT) && defined(FLAG_METALNESS_MAP)
uniform sampler2D metalnessMap;
#endif
#if defined(LIT) && defined(FLAG_AO_MAP)
uniform sampler2D aoMap;
#endif
#ifdef FLAG_ALPHA_MAP
uniform sampler2D alphaMap;
#endif
#if defined(LIT) && (defined(ENV) || defined(REFLECTIVE))
#ifdef ENV
uniform sampler2D envRadianceMap;
uniform sampler2D envIrradianceMap;
#endif
#endif
#if SHADOW_MAPS > 0
uniform sampler2D shadowMap0;
#endif
#if SHADOW_MAPS > 1
uniform sampler2D shadowMap1;
#endif
#if SHADOW_MAPS > 2
uniform sampler2D shadowMap2;
#endif
#if SHADOW_MAPS > 3
uniform sampler2D shadowMap3;
#endif
#if defined(LIT) && LAYER_TEXTURES > 0
uniform sampler2D layerMap0;
#endif
#if defined(LIT) && LAYER_TEXTURES > 1
uniform sampler2D layerMap1;
#endif
// Glass: the frame before it (linear light, premultiplied, mipmapped:
// three.js's transmissionSamplerMap) in layer 0. A reflective draw: the
// reflection's light in layer 0, its runs (mirror_run_of's: premultiplied
// by the coverage, and the coverage) in layer 1's red and green.
#if defined(LIT) && (defined(GLASS) || defined(REFLECTIVE))
uniform sampler2DArray frameMap;
#endif

#if ENTRY == 3
// Particles: the frame's depth (soft particles), and the quads' rows
// (o3d.Sprite_Row): colour and opacity; probe (pixel x, y, depth; the
// disc's radius, 0 when not probed); smear (its way in its sprite's uv
// over the shutter, the moments averaged).
uniform sampler2D sceneDepth;

struct Sprite_Row {
	vec4 color;
	vec4 probe;
	vec4 motion;
};

layout(std430) readonly buffer Sprite_Rows {
	Sprite_Row rows[];
} spriteRows;
#endif

#if ENTRY >= 4
// Exact shapes' curves (Geometries.curve_words), read by curves.GLSL.
layout(std430) readonly buffer Curve_Words {
	uint words[];
} curveWords;

uint curves_word(uint i) {
	return curveWords.words[i];
}
#endif

float linear_to_srgb(float c) {
	c = clamp(c, 0.0, 1.0);
	return c <= 0.0031308 ? c * 12.92 : 1.055 * pow(c, 1.0 / 2.4) - 0.055;
}

// The fragment's derivatives, taken where every lane runs.
vec3 g_dp_dx;
vec3 g_dp_dy;
#ifdef UV0
vec2 g_d0x;
vec2 g_d0y;
#endif
#ifdef UV1
vec2 g_d1x;
vec2 g_d1y;
#endif

// A map at uv with its derivatives (raster.odin's sample: the level from
// them).
vec4 sample_map(sampler2D tex, vec2 uv, vec2 dx, vec2 dy) {
	return textureGrad(tex, vec2(uv.x, 1.0 - uv.y), vec2(dx.x, -dx.y), vec2(dy.x, -dy.y));
}

void transform_uv(vec4 r0, vec4 r1, vec2 raw, vec2 rdx, vec2 rdy, out vec2 uv, out vec2 dx, out vec2 dy) {
	uv = vec2(dot(r0.xyz, vec3(raw, 1.0)), dot(r1.xyz, vec3(raw, 1.0)));
	dx = vec2(dot(r0.xy, rdx), dot(r1.xy, rdx));
	dy = vec2(dot(r0.xy, rdy), dot(r1.xy, rdy));
}

vec3 to_local_n(vec3 n) {
	return vec3(dot(u_normal_to_local[0].xyz, n), dot(u_normal_to_local[1].xyz, n), dot(u_normal_to_local[2].xyz, n));
}

#ifdef FLAG_TRIPLANAR
// The fragment's position projected along x, y and z (as (z, y), (x, z),
// (x, y)), their derivatives and weights (o3d's project_lane).
vec2 g_proj[3];
vec2 g_pdx[3];
vec2 g_pdy[3];
vec3 g_weights;

void project_lane(vec3 world, vec3 normal) {
	vec4 w4 = vec4(world, 1.0);
	vec3 p = vec3(dot(u_to_local[0], w4), dot(u_to_local[1], w4), dot(u_to_local[2], w4)) * u_triplanar.x;
	vec3 dx = dFdx(p);
	vec3 dy = dFdy(p);
	g_proj[0] = p.zy;
	g_proj[1] = p.xz;
	g_proj[2] = p.xy;
	g_pdx[0] = dx.zy;
	g_pdx[1] = dx.xz;
	g_pdx[2] = dx.xy;
	g_pdy[0] = dy.zy;
	g_pdy[1] = dy.xz;
	g_pdy[2] = dy.xy;
	vec3 n = normalize(to_local_n(normal));
	vec3 w = pow(abs(n), vec3(u_triplanar.y));
	g_weights = w / max(w.x + w.y + w.z, 1e-20);
}

vec4 read_projection(sampler2D tex, vec4 r0, vec4 r1, vec2 p, vec2 pdx, vec2 pdy) {
	vec2 uv;
	vec2 dx;
	vec2 dy;
	transform_uv(r0, r1, p, pdx, pdy, uv, dx, dy);
	return sample_map(tex, uv, dx, dy);
}

vec4 read_map(sampler2D tex, vec4 r0, vec4 r1) {
	vec4 acc = vec4(0.0);
	if (g_weights.x > 0.0) {
		acc += g_weights.x * read_projection(tex, r0, r1, g_proj[0], g_pdx[0], g_pdy[0]);
	}
	if (g_weights.y > 0.0) {
		acc += g_weights.y * read_projection(tex, r0, r1, g_proj[1], g_pdx[1], g_pdy[1]);
	}
	if (g_weights.z > 0.0) {
		acc += g_weights.z * read_projection(tex, r0, r1, g_proj[2], g_pdx[2], g_pdy[2]);
	}
	return acc;
}

// o3d's triplanar_normal: a whiteout blend in the projection's frame, then
// back in the world's.
vec3 triplanar_normal(sampler2D tex, vec4 r0, vec4 r1, vec2 scale, vec3 n) {
	vec3 nl = normalize(to_local_n(n));
	vec3 tn0 = vec3(0.0);
	vec3 tn1 = vec3(0.0);
	vec3 tn2 = vec3(0.0);
	if (g_weights.x > 0.0) {
		tn0 = read_projection(tex, r0, r1, g_proj[0], g_pdx[0], g_pdy[0]).rgb * 2.0 - 1.0;
		tn0.xy *= scale;
	}
	if (g_weights.y > 0.0) {
		tn1 = read_projection(tex, r0, r1, g_proj[1], g_pdx[1], g_pdy[1]).rgb * 2.0 - 1.0;
		tn1.xy *= scale;
	}
	if (g_weights.z > 0.0) {
		tn2 = read_projection(tex, r0, r1, g_proj[2], g_pdx[2], g_pdy[2]).rgb * 2.0 - 1.0;
		tn2.xy *= scale;
	}
	vec3 bx = vec3(tn0.x + nl.z, tn0.y + nl.y, abs(tn0.z) * nl.x);
	vec3 by = vec3(tn1.x + nl.x, tn1.y + nl.z, abs(tn1.z) * nl.y);
	vec3 bz = vec3(tn2.x + nl.x, tn2.y + nl.y, abs(tn2.z) * nl.z);
	vec3 b = normalize(bx.zyx * g_weights.x + by.xzy * g_weights.y + bz * g_weights.z);
	return normalize(u_to_local[0].xyz * b.x + u_to_local[1].xyz * b.y + u_to_local[2].xyz * b.z);
}
#else
vec4 read_map(sampler2D tex, vec4 r0, vec4 r1, vec2 raw, vec2 rdx, vec2 rdy) {
	vec2 uv;
	vec2 dx;
	vec2 dy;
	transform_uv(r0, r1, raw, rdx, rdy, uv, dx, dy);
	return sample_map(tex, uv, dx, dy);
}
#endif

// A slot's map: its transform rows (literal: oksl's program reads a literal
// index in place) and, read by uv, its set.
#ifdef FLAG_TRIPLANAR
#define MAP(tex, slot) read_map(tex, u_uv_transforms[ROW0_##slot], u_uv_transforms[ROW1_##slot])
#else
#define MAP(tex, slot) read_map(tex, u_uv_transforms[ROW0_##slot], u_uv_transforms[ROW1_##slot], UV_##slot, DUV_##slot)
#endif

vec3 perturb_normal(vec3 n, vec3 dp_dx, vec3 dp_dy, vec2 duv_dx, vec2 duv_dy, vec3 map_n) {
	vec3 q1perp = cross(n, dp_dy);
	vec3 q0perp = cross(dp_dx, n);
	vec3 t = q1perp * duv_dx.x + q0perp * duv_dy.x;
	vec3 b = q1perp * duv_dx.y + q0perp * duv_dy.y;
	float det = max(dot(t, t), dot(b, b));
	float scale = det > 0.0 ? inversesqrt(det) : 0.0;
	return normalize(t * (map_n.x * scale) + b * (map_n.y * scale) + n * map_n.z);
}

// raster.odin's map_normal: a normal map's normal (its tangent x and y
// times `scale`) about n.
#ifdef FLAG_TRIPLANAR
#define NORMAL_MAP(tex, slot, scale, n) triplanar_normal(tex, u_uv_transforms[ROW0_##slot], u_uv_transforms[ROW1_##slot], scale, n)
#else
vec3 map_normal(sampler2D tex, vec4 r0, vec4 r1, vec2 raw, vec2 rdx, vec2 rdy, vec2 scale, vec3 n) {
	vec2 uv;
	vec2 dx;
	vec2 dy;
	transform_uv(r0, r1, raw, rdx, rdy, uv, dx, dy);
	vec3 map_n = sample_map(tex, uv, dx, dy).rgb * 2.0 - 1.0;
	map_n.xy *= scale;
	return perturb_normal(n, g_dp_dx, g_dp_dy, dx, dy, map_n);
}

#define NORMAL_MAP(tex, slot, scale, n) map_normal(tex, u_uv_transforms[ROW0_##slot], u_uv_transforms[ROW1_##slot], UV_##slot, DUV_##slot, scale, n)
#endif

#if SHADOW_MAPS > 0
// shadow.odin's shadow_factor: a 3x3 grid of bilinear depth comparisons (a
// 4x4 block of texels weighed 1 - f, 1, 1, f per axis, edges repeated),
// outside the light's camera lit.
float shadow_factor(sampler2D map, mat4 m, vec4 params, vec4 from, vec3 world, vec3 normal) {
	if (params.x <= 0.0) {
		return 1.0;
	}
	vec3 p = world + normal * params.z;
	vec3 toward = from.xyz - p * from.w;
	float len2 = dot(toward, toward);
	p -= (len2 > 0.0 ? toward * inversesqrt(len2) : vec3(0.0)) * params.y;
	vec4 c = m * vec4(p, 1.0);
	if (c.w <= 1e-6) {
		return 1.0;
	}
	float inv_w = 1.0 / c.w;
	float u = c.x * inv_w * 0.5 + 0.5;
	float v = 0.5 - c.y * inv_w * 0.5;
	float z = c.z * inv_w;
	if (u < 0.0 || u > 1.0 || v < 0.0 || v > 1.0 || z < 0.0 || z > 1.0) {
		return 1.0;
	}
	int size = int(params.x);
	float s = u * float(size) - 0.5;
	float t = v * float(size) - 0.5;
	float x0 = floor(s);
	float y0 = floor(t);
	float fx = s - x0;
	float fy = t - y0;
	float lit = 0.0;
	for (int j = 0; j < 4; j++) {
		int row = clamp(int(y0) - 1 + j, 0, size - 1);
		float wy = j == 0 ? 1.0 - fy : (j == 3 ? fy : 1.0);
		float acc = 0.0;
		for (int i = 0; i < 4; i++) {
			int col = clamp(int(x0) - 1 + i, 0, size - 1);
			float wx = i == 0 ? 1.0 - fx : (i == 3 ? fx : 1.0);
			float depth = textureLod(map, (vec2(float(col), float(row)) + 0.5) / float(size), 0.0).r;
			acc += wx * (z >= depth ? 1.0 : 0.0);
		}
		lit += wy * acc;
	}
	return lit / 9.0;
}
#endif

#ifdef LIT
vec3 ggx(vec3 f0, float alpha, float nl, float nv, float nh, float vh) {
	vec3 fresnel = f0 + (1.0 - f0) * pow(1.0 - vh, 5.0);
	float a2 = alpha * alpha;
	float gv = nl * sqrt(a2 + (1.0 - a2) * nv * nv);
	float gl = nv * sqrt(a2 + (1.0 - a2) * nl * nl);
	float vis = 0.5 / max(gv + gl, 1e-6);
	float dd = nh * nh * (a2 - 1.0) + 1.0;
	return fresnel * (vis * a2 / (PI * dd * dd));
}

vec3 env_brdf(vec3 f0, float roughness, float nv) {
	vec4 c0 = vec4(-1.0, -0.0275, -0.572, 0.022);
	vec4 c1 = vec4(1.0, 0.0425, 1.04, -0.04);
	vec4 rv = roughness * c0 + c1;
	float a004 = min(rv.x * rv.x, exp2(-9.28 * nv)) * rv.x + rv.y;
	vec2 fab = vec2(-1.04, 1.04) * a004 + rv.zw;
	return f0 * fab.x + fab.y;
}

float charlie(float roughness, float nh) {
	float inv = 1.0 / (roughness * roughness);
	float sin2h = max(1.0 - nh * nh, 0.0078125);
	return (2.0 + inv) * pow(sin2h, inv * 0.5) / (2.0 * PI);
}

float neubelt(float nv, float nl) {
	return clamp(1.0 / (4.0 * (nl + nv - nl * nv)), 0.0, 1.0);
}

float ibl_sheen(float roughness, float nv) {
	float r2 = roughness * roughness;
	float a = roughness < 0.25 ? -339.2 * r2 + 161.4 * roughness - 25.9 : -8.48 * r2 + 14.3 * roughness - 9.95;
	float b = roughness < 0.25 ? 44.0 * r2 - 23.7 * roughness + 3.26 : 1.97 * r2 - 3.27 * roughness + 0.72;
	float dg = exp(a * nv + b) + (roughness < 0.25 ? 0.0 : 0.1 * (roughness - 0.25));
	return clamp(dg / PI, 0.0, 1.0) * PI;
}

#ifdef ENV
vec3 env_radiance(vec3 r, float lod) {
	vec3 rad = textureLod(envRadianceMap, equirect_uv(r), lod).rgb;
#ifdef ENV_HDR
	return rad;
#else
	return linear_rgb(rad);
#endif
}
#endif

#ifdef GLASS
// scene.odin's refraction: where the fragment sees the frame behind it (uv,
// v up) and the level there; the ray's world length.
vec2 refraction(vec3 world, vec3 n, float roughness, float thickness, out float lod, out float distance) {
	vec3 v = normalize(u_eye.xyz - world);
	float ior = u_transmission.z;
	vec3 ray = refract(-v, n, 1.0 / ior);
	float rl = length(ray);
	if (rl > 0.0) {
		ray *= thickness / rl;
	}
	ray *= u_scale.xyz;
	vec4 cp = u_view_proj * vec4(world + ray, 1.0);
	float w = abs(cp.w) > 1e-6 ? cp.w : 1e-6;
	lod = log2(max(u_frame.x, 1.0)) * clamp(roughness, 0.0525, 1.0) * clamp(ior * 2.0 - 2.0, 0.0, 1.0);
	distance = length(ray);
	return clamp(cp.xy / w * 0.5 + 0.5, 0.0, 1.0);
}

// scene.odin's frame_behind: the frame by three.js's textureBicubic (a
// cubic B-spline at two levels, four bilinear taps each).
vec4 frame_bicubic(vec2 uv, int level) {
	vec2 size = max(floor(u_frame.xy / float(1 << level)), vec2(1.0));
	vec2 st = vec2(uv.x, 1.0 - uv.y) * size + 0.5;
	vec2 i = floor(st);
	vec2 a = st - i;
	vec2 w0 = (1.0 / 6.0) * (a * (a * (-a + 3.0) - 3.0) + 1.0);
	vec2 w1 = (1.0 / 6.0) * (a * a * (3.0 * a - 6.0) + 4.0);
	vec2 w2 = (1.0 / 6.0) * (a * (a * (-3.0 * a + 3.0) + 3.0) + 1.0);
	vec2 w3 = (1.0 / 6.0) * (a * a * a);
	vec2 g0 = w0 + w1;
	vec2 g1 = w2 + w3;
	vec2 h0 = -1.0 + w1 / g0;
	vec2 h1 = 1.0 + w3 / g1;
	float lod = float(level);
	vec4 t00 = textureLod(frameMap, vec3((i + vec2(h0.x, h0.y) - 0.5) / size, 0.0), lod);
	vec4 t10 = textureLod(frameMap, vec3((i + vec2(h1.x, h0.y) - 0.5) / size, 0.0), lod);
	vec4 t01 = textureLod(frameMap, vec3((i + vec2(h0.x, h1.y) - 0.5) / size, 0.0), lod);
	vec4 t11 = textureLod(frameMap, vec3((i + vec2(h1.x, h1.y) - 0.5) / size, 0.0), lod);
	return g0.y * (g0.x * t00 + g1.x * t10) + g1.y * (g0.x * t01 + g1.x * t11);
}

vec4 frame_behind(vec2 uv, float lod) {
	float top = floor(log2(max(u_frame.x, u_frame.y)));
	float l = clamp(lod, 0.0, top);
	float f = floor(l);
	vec4 c = frame_bicubic(uv, int(f));
	if (l > f) {
		c = mix(c, frame_bicubic(uv, int(f) + 1), l - f);
	}
	return c;
}
#endif

#ifdef REFLECTIVE
const float REFLECTION_MISS = 1e9;

// scene.odin's reflection_lod: u_reflect is the reflection's last level and
// its focal (Mirror_View.focal); u_camera.z whether the camera is
// orthographic.
float reflection_lod(float roughness, float run, float d) {
	float alpha = clamp(roughness, 0.0, 1.0) * clamp(roughness, 0.0, 1.0);
	float k = u_camera.z != 0.0 ? run : run / (d + run);
	float blur = 1.29 * alpha * u_reflect.y * k;
	return clamp(log2(max(2.0 * blur, 1.0)), 0.0, u_reflect.x);
}

// scene.odin's reflected_run: how far the mirrored ray runs past the
// fragment to what it meets.
float reflected_run(vec2 uv, float far) {
	vec4 near = textureLod(frameMap, vec3(uv, 1.0), 0.0);
	vec4 about = textureLod(frameMap, vec3(uv, 1.0), far);
	float beyond = about.g > 0.0 ? about.r / about.g : REFLECTION_MISS;
	return near.r + (1.0 - near.g) * beyond;
}

// scene.odin's mirrored: the environment's reflected light with a mirror's
// in its place, the environment's where it covers less than all.
vec3 mirrored(vec3 env, vec4 p) {
	return p.rgb + env * (1.0 - p.a);
}
#endif

// shade's state, which the lights add to.
vec3 g_n;
vec3 g_v;
vec3 g_base;
vec3 g_diffuse;
vec3 g_f0;
vec3 g_coat_n;
vec3 g_sheen_c;
float g_metalness;
float g_roughness;
float g_alpha_r;
float g_nv;
float g_ao;
float g_cr;
float g_cnv;
float g_sheen_r;
vec4 g_shadows;
vec3 s_out_d;
vec3 s_out_s;
vec3 s_sheen;
vec3 s_coat_l;

// A directional, point or spot light's light from ldir.
void direct(vec3 light, vec3 ldir) {
#ifdef FLAG_TOON
	float raw_nl = clamp(dot(g_n, ldir), 0.0, 1.0);
	float toon_nl = 0.22 + 0.46 * smoothstep(0.12, 0.18, raw_nl) + 0.32 * smoothstep(0.50, 0.58, raw_nl);
	vec3 spec_term = vec3(0.0);
	if (raw_nl > 0.12 && g_roughness < 0.85) {
		vec3 h = normalize(ldir + g_v);
		float nh = clamp(dot(g_n, h), 0.0, 1.0);
		float thresh = 1.0 - (1.0 - g_roughness) * 0.18;
		float spec_s = smoothstep(thresh - 0.02, thresh + 0.02, nh) * (1.0 - g_roughness) * 0.55;
		spec_term = mix(vec3(1.0), g_base, g_metalness) * spec_s;
	}
	s_out_d += light * g_diffuse * (toon_nl / PI);
	s_out_s += light * spec_term;
#else
#ifdef FLAG_CLEARCOAT
	float cnl = clamp(dot(g_coat_n, ldir), 0.0, 1.0);
	if (cnl > 0.0) {
		vec3 ch = normalize(ldir + g_v);
		s_coat_l += light * cnl * ggx(vec3(0.04), g_cr * g_cr, cnl, g_cnv, clamp(dot(g_coat_n, ch), 0.0, 1.0), clamp(dot(g_v, ch), 0.0, 1.0));
	}
#endif
	float nl = clamp(dot(g_n, ldir), 0.0, 1.0);
	if (nl > 0.0) {
		vec3 irradiance = light * nl;
		vec3 h = normalize(ldir + g_v);
		float nh = clamp(dot(g_n, h), 0.0, 1.0);
		float vh = clamp(dot(g_v, h), 0.0, 1.0);
		s_out_d += irradiance * g_diffuse / PI;
		s_out_s += irradiance * ggx(g_f0, g_alpha_r, nl, g_nv, nh, vh);
#ifdef FLAG_SHEEN
		s_sheen += irradiance * g_sheen_c * (charlie(g_sheen_r, nh) * neubelt(g_nv, nl));
#endif
	}
#endif
}

// A point or spot light's light at the fragment, and its direction.
vec3 point_light(vec3 color, vec3 position, vec4 falloff, out vec3 ldir) {
	vec3 d = position - v_world;
	float dist = length(d);
	ldir = d / max(dist, 1e-6);
	float decay = falloff.y > 0.0 ? falloff.y : 2.0;
	float f = 1.0 / max(pow(dist, decay), 0.01);
	if (falloff.x > 0.0) {
		float t = clamp(1.0 - pow(dist / falloff.x, 4.0), 0.0, 1.0);
		f *= t * t;
	}
	return color * f;
}

void point_direct(vec3 color, vec3 position, vec4 falloff, float shadow) {
	vec3 ld;
	vec3 li = point_light(color, position, falloff, ld);
	direct(li * shadow, ld);
}

void spot_direct(vec4 color, vec4 ground, vec3 position, vec4 falloff, float shadow) {
	vec3 ld;
	vec3 li = point_light(color.rgb, position, falloff, ld);
	float spot = clamp((dot(ld, ground.xyz) - color.w) / max(ground.w - color.w, 1e-5), 0.0, 1.0);
	direct(li * (spot * spot * (3.0 - 2.0 * spot)) * shadow, ld);
}

// scene.odin's shade, with emissive, the layers, the coat's normal, the
// frame behind (glass) and the reflection; transmission's alpha out.
vec3 shade(vec3 base, float roughness_in, float metalness, vec3 n, float ao, vec3 emissive, float clearcoat, float clearcoat_roughness, float sheen_roughness, vec3 sheen, vec3 coat_n, float transmission, float thickness, vec4 behind, vec4 mirror, vec4 mirror_coat, out float alpha) {
	g_n = n;
	g_base = base;
	g_metalness = metalness;
	g_ao = ao;
	g_v = normalize(u_eye.xyz - v_world);
	g_diffuse = base * (1.0 - metalness);
	g_f0 = mix(vec3(u_layers.x), base, metalness);
	g_roughness = clamp(roughness_in, 0.0525, 1.0);
	g_alpha_r = g_roughness * g_roughness;
	g_nv = clamp(dot(n, g_v), 1e-4, 1.0);
	float cc = clamp(clearcoat, 0.0, 1.0);
	g_cr = clamp(clearcoat_roughness, 0.0525, 1.0);
	g_coat_n = coat_n;
	g_cnv = clamp(dot(coat_n, g_v), 1e-4, 1.0);
	g_sheen_c = sheen;
	g_sheen_r = sheen_roughness;
	s_out_d = vec3(0.0);
	s_out_s = vec3(0.0);
	s_sheen = vec3(0.0);
	s_coat_l = vec3(0.0);
	O3D_LIGHTS
#if defined(ENV) || defined(REFLECTIVE)
	vec3 spec = env_brdf(g_f0, g_roughness, g_nv);
	float env_k = u_env.x * u_sheen.w;
	float k = g_ao * env_k;
	vec3 rad = vec3(0.0);
#ifdef ENV
	vec3 r = 2.0 * dot(g_n, g_v) * g_n - g_v;
	vec3 irr = textureLod(envIrradianceMap, equirect_uv(g_n), 0.0).rgb;
#ifndef ENV_HDR
	irr = linear_rgb(irr);
#endif
	rad = env_radiance(r, g_roughness * u_env.y);
	s_out_d += k * irr * g_diffuse * (1.0 - spec);
#ifdef FLAG_SHEEN
	s_sheen += k * irr * g_sheen_c * ibl_sheen(g_sheen_r, g_nv);
#endif
#endif
#ifdef REFLECTIVE
	s_out_s += g_ao * mirrored(env_k * rad, mirror) * spec;
#else
	s_out_s += k * rad * spec;
#endif
#ifdef FLAG_CLEARCOAT
	vec3 coat_rad = vec3(0.0);
#ifdef ENV
	vec3 cr_dir = 2.0 * dot(g_coat_n, g_v) * g_coat_n - g_v;
	coat_rad = env_radiance(cr_dir, g_cr * u_env.y);
#endif
#ifdef REFLECTIVE
	s_coat_l += g_ao * mirrored(env_k * coat_rad, mirror_coat) * env_brdf(vec3(0.04), g_cr, g_cnv);
#else
	s_coat_l += k * coat_rad * env_brdf(vec3(0.04), g_cr, g_cnv);
#endif
#endif
#endif
	alpha = 1.0;
#ifdef FLAG_TRANSMISSION
	vec3 t = g_diffuse * exp(u_attenuation.rgb * thickness);
	vec3 through = (1.0 - env_brdf(g_f0, g_roughness, g_nv)) * t * behind.rgb;
	s_out_d = mix(s_out_d, through, transmission);
	alpha = mix(1.0, 1.0 - (1.0 - behind.a) * (t.r + t.g + t.b) / 3.0, transmission);
#endif
	vec3 o = s_out_d + s_out_s + emissive;
#ifdef FLAG_SHEEN
	o = o * (1.0 - 0.157 * max(g_sheen_c.r, max(g_sheen_c.g, g_sheen_c.b))) + s_sheen;
#endif
#ifdef FLAG_CLEARCOAT
	o = o * (1.0 - cc * (0.04 + 0.96 * pow(1.0 - g_cnv, 5.0))) + s_coat_l * cc;
#endif
	return o;
}
#endif

#if ENTRY == 3
// Points on a disc of radius 1, a Vogel spiral.
const vec2 OCCLUSION_TAPS[16] = vec2[16](
	vec2(0.176777, 0.0), vec2(-0.225772, 0.206826), vec2(0.034558, -0.393771), vec2(0.284571, 0.371173),
	vec2(-0.522223, -0.092374), vec2(0.494695, -0.314685), vec2(-0.165466, 0.615525), vec2(-0.315561, -0.607594),
	vec2(0.684642, 0.250030), vec2(-0.712256, 0.294009), vec2(0.343355, -0.733729), vec2(0.253730, 0.808932),
	vec2(-0.764746, -0.443186), vec2(0.897134, -0.197232), vec2(-0.547507, 0.778772), vec2(-0.126487, -0.976090));

// A view depth from a depth value (o3d.view_depth).
float view_depth(float depth) {
	float near = u_camera.x;
	float far = u_camera.y;
	if (u_camera.z != 0.0) {
		return far - depth * (far - near);
	}
	return near * far / (near + depth * (far - near));
}

// The share of a probe's disc the frame's depth leaves unhidden.
float sprite_visibility(vec4 probe) {
	vec2 size = u_frame.xy;
	float seen = 0.0;
	for (int k = 0; k < 16; k++) {
		vec2 p = probe.xy + OCCLUSION_TAPS[k] * probe.w;
		if (p.x < 0.0 || p.y < 0.0 || p.x >= size.x || p.y >= size.y) {
			continue;
		}
		float d = textureLod(sceneDepth, (floor(p) + 0.5) / size, 0.0).r;
		if (d <= 0.0 || probe.z > d) {
			seen += 1.0;
		}
	}
	return seen / 16.0;
}

// raster.odin's smeared_map: a smeared particle's sprite at the moments of
// the shutter, its light averaged, back as a straight colour and coverage.
vec4 smeared_map(vec4 motion) {
	int taps = int(motion.z);
	vec4 acc = vec4(0.0);
	for (int k = 0; k < taps; k++) {
		vec2 uv = v_uv - motion.xy * ((float(k) + 0.5) / float(taps) - 0.5);
		if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
			continue;
		}
		vec4 s = vec4(1.0);
#ifdef FLAG_COLOR_MAP
		vec2 tuv;
		vec2 dx;
		vec2 dy;
		transform_uv(u_uv_transforms[ROW0_COLOR], u_uv_transforms[ROW1_COLOR], uv, g_d0x, g_d0y, tuv, dx, dy);
		s = sample_map(colorMap, tuv, dx, dy);
#ifdef FLAG_PREMULTIPLIED
		s.rgb = s.a > 0.0 ? clamp(s.rgb / s.a, 0.0, 1.0) : vec3(0.0);
#endif
#endif
		acc += vec4(linear_rgb(s.rgb) * s.a, s.a);
	}
	if (acc.a <= 0.0) {
		return vec4(0.0);
	}
	vec3 c = acc.rgb / acc.a;
	return vec4(linear_to_srgb(c.r), linear_to_srgb(c.g), linear_to_srgb(c.b), acc.a / float(taps));
}
#endif

#if ENTRY == 2 || ENTRY == 5
// The mirror's pass: how far the mirror's ray runs past the mirror to the
// fragment (u_eye is the eye's image, u_clip_plane the mirror)
// premultiplied by its coverage, and the coverage.
vec4 mirror_run_of(vec3 world, vec4 color) {
	vec3 n = u_clip_plane.xyz;
	float h = max(dot(n, world) + u_clip_plane.w, 0.0);
	float below = -(dot(n, u_eye.xyz) + u_clip_plane.w);
	float run = length(world - u_eye.xyz) * h / max(h + below, 1e-6);
	return vec4(run * color.a, color.a, 0.0, color.a);
}
#endif

// The material's light at the fragment, premultiplied by its alpha, and
// its coverage (that alpha times transmission's: glass leaves room for
// what shows behind, whose light is in it already). `tint` (sRGB colour,
// opacity) multiplies the material's: a particle's row; `smear` is a
// smeared particle's (Sprite_Row.motion).
vec4 surface(vec4 tint, vec4 smear) {
	vec3 base = u_color.rgb * linear_rgb(tint.rgb);
	float a = u_color.a * clamp(tint.a, 0.0, 1.0);
#ifdef FLAG_TRIPLANAR
#ifdef FLAG_FLAT_SHADING
	project_lane(v_world, cross(g_dp_dx, g_dp_dy));
#else
	project_lane(v_world, v_normal);
#endif
#endif
	// The mirror's pass leaves out what is behind its plane (after the
	// derivatives; a zero plane in the other passes).
#if CUT
	if (dot(u_clip_plane.xyz, v_world) + u_clip_plane.w < 0.0) {
		discard;
	}
#endif
#ifdef FLAG_VERTEX_COLORS
	base *= v_color.rgb;
#if defined(FLAG_BLENDED) || defined(FLAG_ALPHA_MASK)
	a *= v_color.a;
#endif
#endif
#if defined(FLAG_COLOR_MAP) || ENTRY == 3
	bool mapped = false;
	vec4 tex = vec4(1.0);
#if ENTRY == 3
	if (smear.z > 1.0) {
		tex = smeared_map(smear);
		mapped = true;
	}
#endif
#ifdef FLAG_COLOR_MAP
	if (!mapped) {
		tex = MAP(colorMap, COLOR);
#ifdef FLAG_PREMULTIPLIED
		tex.rgb = tex.a > 0.0 ? clamp(tex.rgb / tex.a, 0.0, 1.0) : vec3(0.0);
#endif
		mapped = true;
	}
#endif
	if (mapped) {
		base *= linear_rgb(tex.rgb);
#if defined(FLAG_BLENDED) || defined(FLAG_ALPHA_MASK)
		a *= tex.a;
#endif
	}
#endif
#ifdef FLAG_ALPHA_MAP
	a *= MAP(alphaMap, ALPHA).g;
#endif
#ifdef FLAG_ALPHA_MASK
#if CUT
	if (a < u_surface.w) {
		discard;
	}
#endif
#ifndef FLAG_BLENDED
	a = 1.0;
#endif
#endif
	vec3 emissive = u_emissive.rgb;
#ifdef FLAG_EMISSIVE_MAP
	emissive *= linear_rgb(MAP(emissiveMap, EMISSIVE).rgb);
#endif
	vec3 c;
	float ta = 1.0;
#if defined(FLAG_SHADOW_ONLY)
	// Only the light its shadows take: 1 minus their factors' product.
	vec3 sn = normalize(v_normal);
	float mask = 1.0;
	SHADOW_ONLY_MASK
	a *= 1.0 - mask;
	c = base;
#elif defined(FLAG_BASIC)
	c = base + emissive;
#else
	float roughness = u_surface.y;
#ifdef FLAG_ROUGHNESS_MAP
	roughness *= MAP(roughnessMap, ROUGHNESS).g;
#endif
	float metalness = u_surface.x;
#ifdef FLAG_METALNESS_MAP
	metalness *= MAP(metalnessMap, METALNESS).b;
#endif
	float ao = 1.0;
#ifdef FLAG_AO_MAP
	ao = (MAP(aoMap, AO).r - 1.0) * u_surface.z + 1.0;
#endif
	float ly_clearcoat = u_layers.y;
	float ly_cr = u_layers.z;
	float ly_sheen_r = u_layers.w;
	vec3 ly_sheen = u_sheen.rgb;
	float ly_transmission = u_transmission.x;
	float ly_thickness = u_transmission.y;
#ifdef FLAG_LAYER_MAPS
	ly_clearcoat *= LAYER_CLEARCOAT.r;
	ly_cr *= LAYER_CLEARCOAT_ROUGHNESS.g;
	ly_sheen *= linear_rgb(LAYER_SHEEN_COLOR.rgb);
	ly_sheen_r *= LAYER_SHEEN_ROUGHNESS.a;
	ly_transmission *= LAYER_TRANSMISSION.r;
	ly_thickness *= LAYER_THICKNESS.g;
#endif
	vec3 n = normalize(v_normal);
	// The triangle's normal, facing the eye (screen derivatives run with y
	// down). A two-sided surface's normal turns to this side: a smooth
	// normal can lean away from the eye at a silhouette and still be on the
	// eye's side of its triangle.
	vec3 face = cross(g_dp_dx, g_dp_dy);
	if (dot(face, u_eye.xyz - v_world) < 0.0) {
		face = -face;
	}
#ifdef FLAG_FLAT_SHADING
	n = normalize(face);
#endif
	g_shadows = vec4(1.0);
#ifdef FLAG_RECEIVE_SHADOW
	RECEIVE_SHADOWS
#endif
#ifdef FLAG_TWO_SIDED
	if (dot(n, face) < 0.0) {
		n = -n;
	}
#endif
	// The coat's normal: the surface's before its normal map, or the coat's
	// normal map's.
	vec3 coat_n = n;
#ifdef FLAG_NORMAL_MAP
	n = NORMAL_MAP(normalMap, NORMAL, u_normal_scales.xy, n);
#endif
#if defined(FLAG_CLEARCOAT_NORMAL_MAP) && LAYER_TEX_CLEARCOAT_NORMAL == 1
	coat_n = NORMAL_MAP(layerMap0, CLEARCOAT_NORMAL, u_normal_scales.zw, coat_n);
#elif defined(FLAG_CLEARCOAT_NORMAL_MAP) && LAYER_TEX_CLEARCOAT_NORMAL == 2
	coat_n = NORMAL_MAP(layerMap1, CLEARCOAT_NORMAL, u_normal_scales.zw, coat_n);
#endif
	vec4 behind = vec4(0.0);
#ifdef GLASS
	float lod;
	float distance;
	vec2 fuv = refraction(v_world, n, roughness, ly_thickness, lod, distance);
	ly_thickness = distance;
	behind = frame_behind(fuv, lod);
#endif
	// The reflection where the fragment is on the screen, at its roughness
	// and its coat's.
	vec4 mirror = vec4(0.0);
	vec4 mirror_coat = vec4(0.0);
#ifdef REFLECTIVE
	vec2 ruv = gl_FragCoord.xy / u_frame.xy;
	float d = length(v_world - u_eye.xyz);
#ifdef FLAG_CLEARCOAT
	float run = reflected_run(ruv, reflection_lod(max(roughness, ly_cr), REFLECTION_MISS, d));
	mirror_coat = textureLod(frameMap, vec3(ruv, 0.0), reflection_lod(ly_cr, run, d));
#else
	float run = reflected_run(ruv, reflection_lod(roughness, REFLECTION_MISS, d));
#endif
	mirror = textureLod(frameMap, vec3(ruv, 0.0), reflection_lod(roughness, run, d));
#endif
	c = shade(base, roughness, metalness, n, ao, emissive, ly_clearcoat, ly_cr, ly_sheen_r, ly_sheen, coat_n, ly_transmission, ly_thickness, behind, mirror, mirror_coat, ta);
#endif
	float fog = 0.0;
#if FOG == 1
	float fog_dist = length(v_world - u_eye.xyz);
	fog = clamp((fog_dist - u_fog.x) / max(u_fog.y - u_fog.x, 1e-6), 0.0, 1.0);
#elif FOG == 2
	float fog_d = u_fog.z * length(v_world - u_eye.xyz);
	fog = clamp(1.0 - exp(-fog_d * fog_d), 0.0, 1.0);
#endif
	c = mix(c, u_fog_color.rgb, fog);
	return vec4(c * a, a * ta);
}

#if ENTRY >= 4
// Sample positions in a pixel (the standard 4x pattern, software.odin's
// SAMPLE_OFFSETS).
const vec2 SAMPLE_OFFSETS[4] = vec2[4](vec2(0.375, 0.125), vec2(0.875, 0.375), vec2(0.125, 0.625), vec2(0.625, 0.875));

// A `.Curves` quad's table entry: its set's first word, the top bit its fill
// rule (curve.odin's CURVE_EVEN_ODD).
const uint CURVE_EVEN_ODD = 0x80000000u;

int curve_entry_rule(uint entry) {
	return (entry & CURVE_EVEN_ODD) != 0u ? CURVES_EVEN_ODD : CURVES_NON_ZERO;
}
#endif

void main() {
	g_dp_dx = dFdx(v_world);
	g_dp_dy = dFdy(v_world);
#ifdef UV0
	g_d0x = dFdx(v_uv);
	g_d0y = dFdy(v_uv);
#endif
#ifdef UV1
	g_d1x = dFdx(v_uv1);
	g_d1y = dFdy(v_uv1);
#endif
#if ENTRY == 3
	// A particle quad: its row's colour and opacity, faded by the share of
	// its probe's disc unhidden and by how far in front of the frame's depth
	// it is (soft particles).
	Sprite_Row r = spriteRows.rows[u_table.x + v_quad];
	vec4 row = r.color;
	if (r.probe.w > 0.0) {
		row.a *= sprite_visibility(r.probe);
	}
	if (u_particle.x > 0.0) {
		float behind = view_depth(textureLod(sceneDepth, gl_FragCoord.xy / u_frame.xy, 0.0).r);
		row.a *= clamp((behind - view_depth(gl_FragCoord.z)) / u_particle.x, 0.0, 1.0);
	}
	o_light = surface(row, r.motion);
#elif ENTRY >= 4
	// An exact shape: uv1 is its shape coordinate in its quad's curve set
	// (the quad's entry in the draw's table). Blended, the pixel's covered
	// area scales its light (raster.odin's curve_coverage); opaque, the
	// samples inside the shape are written (the coordinate at each from the
	// centre's and its derivatives, raster.odin's curve_samples).
	uint entry = curves_word(u_table.x + v_quad);
	uint s = entry & ~CURVE_EVEN_ODD;
	int rule = curve_entry_rule(entry);
	vec4 color = surface(vec4(1.0), vec4(0.0));
#ifdef FLAG_BLENDED
	float cov = curves_coverage_affine(s, v_uv1, g_d1x, g_d1y, rule);
	if (cov <= 0.0) {
		discard;
	}
	o_light = color * cov;
#else
	int mask = 0;
	for (int k = 0; k < 4; k++) {
		vec2 off = SAMPLE_OFFSETS[k] - 0.5;
		if (curves_fill_inside(curves_winding(s, v_uv1 + g_d1x * off.x + g_d1y * off.y), rule)) {
			mask |= 1 << k;
		}
	}
	if (mask == 0) {
		discard;
	}
	gl_SampleMask[0] = mask;
	o_light = color;
#endif
#if ENTRY == 5
	o_run = mirror_run_of(v_world, o_light);
#endif
#else
	o_light = surface(vec4(1.0), vec4(0.0));
#if ENTRY == 2
	o_run = mirror_run_of(v_world, o_light);
#endif
#endif
}
