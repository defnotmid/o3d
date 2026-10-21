// o3d's vertex and fragment shaders for SDL_GPU, compiled by build.sh to
// SPIR-V (Vulkan) and MSL (Metal). The shading must stay the software
// renderer's (scene.odin `shade`): three.js's MeshStandardMaterial with
// equirectangular environment lighting and background.
//
// SDL_GPU binding convention: vertex uniforms set 1, fragment uniforms
// set 3.
//
// Exact shapes (curve.odin) draw with fs_curves, which takes coverage from
// their curves by ovg's curves.slang (each pixel's exact covered area), as raster.odin
// does with ovg's curves package.
import common;
#include "curves.slang"

static const int MAX_LIGHTS = 8;
static const int MAX_SHADOWS = 4;
static const int MAX_JOINTS = 64;
static const int MAX_MORPH_TARGETS = 8;
// Light_Kind's values and Shading_Flag's bits (o3d_test checks).
static const uint LIGHT_AMBIENT = 0;
static const uint LIGHT_HEMISPHERE = 1;
static const uint LIGHT_DIRECTIONAL = 2;
static const uint LIGHT_POINT = 3;
static const uint LIGHT_SPOT = 4;

static const int FOG_NONE = 0;
static const int FOG_LINEAR = 1;
static const int FOG_EXP2 = 2;

static const uint FLAG_BASIC = 1;
static const uint FLAG_FLAT_SHADING = 2;
static const uint FLAG_TWO_SIDED = 4;
static const uint FLAG_BLENDED = 8;
static const uint FLAG_RECEIVE_SHADOW = 16;
static const uint FLAG_COLOR_MAP = 32;
static const uint FLAG_NORMAL_MAP = 64;
static const uint FLAG_ROUGHNESS_MAP = 128;
static const uint FLAG_EMISSIVE_MAP = 256;
static const uint FLAG_METALNESS_MAP = 512;
static const uint FLAG_AO_MAP = 1024;
static const uint FLAG_ALPHA_MASK = 2048;
static const uint FLAG_TOON = 4096;
static const uint FLAG_CURVES = 8192;
static const uint FLAG_ALPHA_MAP = 16384;
static const uint FLAG_SHADOW_ONLY = 32768;
static const uint FLAG_SPRITE = 65536;
static const uint FLAG_ADDITIVE = 131072;
static const uint FLAG_PREMULTIPLIED = 262144;

static const uint SLOT_COLOR = 0;
static const uint SLOT_NORMAL = 1;
static const uint SLOT_ROUGHNESS = 2;
static const uint SLOT_EMISSIVE = 3;
static const uint SLOT_METALNESS = 4;
static const uint SLOT_AO = 5;
static const uint SLOT_ALPHA = 6;

struct Morph_Delta {
    float4 position;
    float4 normal;
};

struct Draw_VS {
    float4x4 mvp;
    float4x4 model;
    float4x4 normal;
    uint4 morph_bases[2];
    float4 morph_weights[2];
    uint4 morph_info;
};

// Shading_Material: color is linear rgb, a = opacity (1 unless blended or
// alpha-masked); emissive is linear rgb times intensity; flags are FLAG_* bits.
// curve_table: an exact shape's or glyph run's table of its quads' entries
// in curveWords (each its curve set's first word, bit 31 for even-odd).
struct Material {
    float4 color;
    float4 emissive;
    float metalness;
    float roughness;
    float ao_intensity;
    uint flags;
    float alpha_cutoff;
    uint uv_sets;
    uint curve_table; // an exact shape's or glyph run's quad table, or a particle draw's first row
    float softness;   // particles: fade distance before the frame's depth
    float4 uv_transforms[14];
};

// Shading_Light: color, ground linear times intensity (or spot axis + cos_inner/outer);
// vector the direction toward a directional light or a point/spot light's position;
// kind a LIGHT_*; shadowed: 1-based shadow map index (0: unshadowed).
struct Light {
    float4 color;
    float4 ground;
    float4 vector;
    uint kind;
    float distance;
    float decay;
    uint shadowed;
};

// shadowMatrix: the shadow lights' view-projections; shadowParams: map size
// (0: no shadow), bias, normal bias; envParams: intensity (0: no env),
// max radiance mip level, tone mapping exposure, fog density.
struct Lighting {
    float4 eye;
    int4 count;
    float4x4 shadowMatrix[MAX_SHADOWS];
    float4 shadowParams[MAX_SHADOWS];
    float4 envParams;
    // x: the environment's maps are linear (HDR), not sRGB-encoded;
    // y: the camera is orthographic.
    uint4 envFlags;
    // The frame's width and height, the camera's near and far.
    float4 frameParams;
    float4 fogColor;
    float4 fogParams;
    Light lights[MAX_LIGHTS];
};

struct Joints_VS {
    float4x4 mats[MAX_JOINTS];
};

[[vk::binding(0, 1)]] ConstantBuffer<Draw_VS> draw : register(b0, space1);
[[vk::binding(1, 1)]] ConstantBuffer<Joints_VS> joints : register(b1, space1);
[[vk::binding(0, 0)]] StructuredBuffer<Morph_Delta> morphDeltas : register(t1, space0);
[[vk::binding(0, 0)]] StructuredBuffer<Morph_Delta> skinnedMorphDeltas : register(t2, space1);
[[vk::binding(0, 3)]] ConstantBuffer<Material> material : register(b0, space3);
[[vk::binding(1, 3)]] ConstantBuffer<Lighting> lighting : register(b1, space3);
[[vk::binding(0, 2)]] Sampler2D<float4> colorMap : register(t0, space2);
// The shadow maps: depth per texel (R32_FLOAT), read texel by texel.
[[vk::binding(1, 2)]] Sampler2D<float> shadowMap : register(t1, space2);
[[vk::binding(2, 2)]] Sampler2D<float4> normalMap : register(t2, space2);
[[vk::binding(3, 2)]] Sampler2D<float4> roughnessMap : register(t3, space2);
[[vk::binding(4, 2)]] Sampler2D<float4> emissiveMap : register(t4, space2);
[[vk::binding(5, 2)]] Sampler2D<float4> envRadianceMap : register(t5, space2);
[[vk::binding(6, 2)]] Sampler2D<float4> envIrradianceMap : register(t6, space2);
[[vk::binding(7, 2)]] Sampler2D<float4> metalnessMap : register(t7, space2);
[[vk::binding(8, 2)]] Sampler2D<float4> aoMap : register(t8, space2);
[[vk::binding(9, 2)]] Sampler2D<float> shadowMap1 : register(t9, space2);
[[vk::binding(10, 2)]] Sampler2D<float> shadowMap2 : register(t10, space2);
[[vk::binding(11, 2)]] Sampler2D<float> shadowMap3 : register(t11, space2);
[[vk::binding(12, 2)]] Sampler2D<float4> alphaMap : register(t12, space2);
// The shadow pass's alpha map, after its colour map.
[[vk::binding(1, 2)]] Sampler2D<float4> shadowAlphaMap : register(t1, space2) : register(s1, space2);
// Particles: the frame's depth (soft particles) after the main pass's
// samplers, and the rows after it (SPIR-V; on Metal after the uniform
// buffers, as curveWords).
[[vk::binding(13, 2)]] Sampler2D<float> sceneDepth : register(t13, space2) : register(s13, space2);
[[vk::binding(14, 2)]] StructuredBuffer<float4> spriteRows : register(t2, space2);
// Exact shapes' curves (Geometries.curve_words): after the samplers in
// SPIR-V's set 2, after the uniform buffers on Metal (slang numbers the
// buffer from the register; no DXIL for these entries): fs_curves' at
// [[buffer(2)]], fs_shadow_curves' at [[buffer(1)]].
[[vk::binding(13, 2)]] StructuredBuffer<uint> curveWords : register(t2, space2);
[[vk::binding(2, 2)]] StructuredBuffer<uint> shadowCurveWords : register(t1, space2);

struct VS_In {
    float3 position : TEXCOORD0;
    float3 normal : TEXCOORD1;
    float2 uv : TEXCOORD2;
    float2 uv1 : TEXCOORD3;
};

struct VS_Skinned_In {
    float3 position : TEXCOORD0;
    float3 normal : TEXCOORD1;
    float2 uv : TEXCOORD2;
    float2 uv1 : TEXCOORD3;
    uint4 joints : TEXCOORD4;
    float4 weights : TEXCOORD5;
};

struct VS_Out {
    float4 position : SV_Position;
    float3 world : TEXCOORD0;
    float3 normal : TEXCOORD1;
    float2 uv : TEXCOORD2;
    float2 uv1 : TEXCOORD3;
};

[shader("vertex")]
VS_Out vs_main(VS_In i, uint vid : SV_VertexID) {
    float3 pos = i.position;
    float3 norm = i.normal;
    for (uint k = 0; k < draw.morph_info.x; k++) {
        uint base = draw.morph_bases[k >> 2][k & 3];
        float w = draw.morph_weights[k >> 2][k & 3];
        Morph_Delta d = morphDeltas[base + vid];
        pos += d.position.xyz * w;
        norm += d.normal.xyz * w;
    }
    VS_Out o;
    o.position = mul(draw.mvp, float4(pos, 1));
    o.world = mul(draw.model, float4(pos, 1)).xyz;
    o.normal = mul(draw.normal, float4(norm, 0)).xyz;
    o.uv = i.uv;
    o.uv1 = i.uv1;
    return o;
}

// An exact shape's or glyph run's vertex: as vs_main's (no morphs), and its
// quad (vertices 4q..4q+3, the draw counting from 0) for the fragment to
// find its curve set by.
struct VS_Curves_Out {
    float4 position : SV_Position;
    float3 world : TEXCOORD0;
    float3 normal : TEXCOORD1;
    float2 uv : TEXCOORD2;
    float2 uv1 : TEXCOORD3;
    nointerpolation uint quad : TEXCOORD4;
};

VS_Out curves_surface_in(VS_Curves_Out i) {
    VS_Out o;
    o.position = i.position;
    o.world = i.world;
    o.normal = i.normal;
    o.uv = i.uv;
    o.uv1 = i.uv1;
    return o;
}

[shader("vertex")]
VS_Curves_Out vs_curves(VS_In i, uint vid : SV_VertexID) {
    VS_Curves_Out o;
    o.position = mul(draw.mvp, float4(i.position, 1));
    o.world = mul(draw.model, float4(i.position, 1)).xyz;
    o.normal = mul(draw.normal, float4(i.normal, 0)).xyz;
    o.uv = i.uv;
    o.uv1 = i.uv1;
    o.quad = vid / 4;
    return o;
}

[shader("vertex")]
VS_Out vs_skinned(VS_Skinned_In i, uint vid : SV_VertexID) {
    float3 mpos = i.position;
    float3 mnorm = i.normal;
    for (uint k = 0; k < draw.morph_info.x; k++) {
        uint base = draw.morph_bases[k >> 2][k & 3];
        float w = draw.morph_weights[k >> 2][k & 3];
        Morph_Delta d = skinnedMorphDeltas[base + vid];
        mpos += d.position.xyz * w;
        mnorm += d.normal.xyz * w;
    }
    float4x4 skin = joints.mats[i.joints.x] * i.weights.x +
                    joints.mats[i.joints.y] * i.weights.y +
                    joints.mats[i.joints.z] * i.weights.z +
                    joints.mats[i.joints.w] * i.weights.w;
    float3 pos = mul(skin, float4(mpos, 1)).xyz;
    float3 norm = mul(skin, float4(mnorm, 0)).xyz;
    VS_Out o;
    o.position = mul(draw.mvp, float4(pos, 1));
    o.world = mul(draw.model, float4(pos, 1)).xyz;
    o.normal = mul(draw.normal, float4(norm, 0)).xyz;
    o.uv = i.uv;
    o.uv1 = i.uv1;
    return o;
}

void slot_uv(uint slot, float2 uv0, float2 uv1, float2 d0x, float2 d0y, float2 d1x, float2 d1y, out float2 uv, out float2 dx, out float2 dy) {
    bool use1 = ((material.uv_sets >> slot) & 1) != 0;
    float2 raw = use1 ? uv1 : uv0;
    float2 rdx = use1 ? d1x : d0x;
    float2 rdy = use1 ? d1y : d0y;
    float3 r0 = material.uv_transforms[slot * 2 + 0].xyz;
    float3 r1 = material.uv_transforms[slot * 2 + 1].xyz;
    uv = float2(dot(r0, float3(raw, 1)), dot(r1, float3(raw, 1)));
    dx = float2(dot(r0.xy, rdx), dot(r1.xy, rdx));
    dy = float2(dot(r0.xy, rdy), dot(r1.xy, rdy));
}

float4 sample_map(Sampler2D<float4> tex, float2 uv, float2 dx, float2 dy) {
    return tex.SampleGrad(float2(uv.x, 1 - uv.y), float2(dx.x, -dx.y), float2(dy.x, -dy.y));
}

float3 perturb_normal(float3 n, float3 dp_dx, float3 dp_dy, float2 duv_dx, float2 duv_dy, float3 map_n) {
    float3 q1perp = cross(n, dp_dy);
    float3 q0perp = cross(dp_dx, n);
    float3 t = q1perp * duv_dx.x + q0perp * duv_dy.x;
    float3 b = q1perp * duv_dx.y + q0perp * duv_dy.y;
    float det = max(dot(t, t), dot(b, b));
    float scale = det > 0 ? rsqrt(det) : 0;
    return normalize(t * (map_n.x * scale) + b * (map_n.y * scale) + n * map_n.z);
}

float sample_shadow_map(int idx, float2 uv) {
    if (idx == 0) return shadowMap.SampleLevel(uv, 0);
    if (idx == 1) return shadowMap1.SampleLevel(uv, 0);
    if (idx == 2) return shadowMap2.SampleLevel(uv, 0);
    return shadowMap3.SampleLevel(uv, 0);
}

// shadow.odin's shadow_factor: a 3x3 grid of bilinear depth comparisons,
// a 4x4 block of texels weighed (1 - f, 1, 1, f) per axis, edges repeated,
// outside the light's camera lit.
float shadow_factor(int idx, float3 world, float3 normal) {
    float4 params = lighting.shadowParams[idx];
    if (params.x <= 0) {
        return 1;
    }
    float3 p = world + normal * params.z;
    float4 c = mul(lighting.shadowMatrix[idx], float4(p, 1));
    if (c.w <= 1e-6) {
        return 1;
    }
    float inv_w = 1 / c.w;
    float u = c.x * inv_w * 0.5 + 0.5;
    float v = 0.5 - c.y * inv_w * 0.5;
    float z = c.z * inv_w + params.y;
    if (u < 0 || u > 1 || v < 0 || v > 1 || z < 0 || z > 1) {
        return 1;
    }
    int size = int(params.x);
    float s = u * float(size) - 0.5;
    float t = v * float(size) - 0.5;
    float x0 = floor(s);
    float y0 = floor(t);
    float fx = s - x0;
    float fy = t - y0;
    float wx[4] = { 1 - fx, 1, 1, fx };
    float wy[4] = { 1 - fy, 1, 1, fy };
    float lit = 0;
    for (int j = 0; j < 4; j++) {
        int row = clamp(int(y0) - 1 + j, 0, size - 1);
        float acc = 0;
        for (int i = 0; i < 4; i++) {
            int col = clamp(int(x0) - 1 + i, 0, size - 1);
            // The texel itself: its centre through the nearest sampler (Load
            // on a combined texture-sampler miscompiles to MSL).
            float depth = sample_shadow_map(idx, (float2(col, row) + 0.5) / float(size));
            acc += wx[i] * (z <= depth ? 1.0 : 0.0);
        }
        lit += wy[j] * acc;
    }
    return lit / 9;
}

float3 shade(float3 base, float roughness_in, float metalness, float3 world, float3 n, float3 eye, float4 shadows, float ao, uint flags) {
    float3 v = normalize(eye - world);
    float3 diffuse = base * (1 - metalness);
    float3 f0 = lerp(float3(0.04), base, metalness);
    float roughness = clamp(roughness_in, 0.0525, 1);
    float alpha = roughness * roughness;
    float nv = clamp(dot(n, v), 1e-4, 1);
    float3 o = float3(0);
    for (int k = 0; k < lighting.count.x; k++) {
        Light l = lighting.lights[k];
        if (l.kind == LIGHT_AMBIENT) {
            o += l.color.rgb * diffuse * (ao / PI);
            continue;
        }
        if (l.kind == LIGHT_HEMISPHERE) {
            float w = 0.5 * n.y + 0.5;
            o += lerp(l.ground.rgb, l.color.rgb, w) * diffuse * (ao / PI);
            continue;
        }
        float3 ldir = l.vector.xyz;
        float3 light = l.color.rgb;
        if (l.kind == LIGHT_POINT || l.kind == LIGHT_SPOT) {
            float3 d = l.vector.xyz - world;
            float dist = length(d);
            ldir = d / max(dist, 1e-6);
            float decay = l.decay > 0 ? l.decay : 2;
            float falloff = 1 / max(pow(dist, decay), 0.01);
            if (l.distance > 0) {
                float t = saturate(1 - pow(dist / l.distance, 4));
                falloff *= t * t;
            }
            if (l.kind == LIGHT_SPOT) {
                float cos_angle = dot(ldir, l.ground.xyz);
                float denom = max(l.ground.w - l.color.w, 1e-6);
                float spot = saturate((cos_angle - l.color.w) / denom);
                falloff *= spot * spot * (3 - 2 * spot);
            }
            light *= falloff;
        }
        if (l.shadowed > 0 && l.shadowed <= uint(MAX_SHADOWS)) {
            light *= shadows[l.shadowed - 1];
        }
        if ((flags & FLAG_TOON) != 0) {
            float raw_nl = saturate(dot(n, ldir));
            float toon_nl = 0.22 + 0.46 * smoothstep(0.12, 0.18, raw_nl) + 0.32 * smoothstep(0.50, 0.58, raw_nl);
            float3 spec_term = float3(0);
            if (raw_nl > 0.12 && roughness < 0.85) {
                float3 h = normalize(ldir + v);
                float nh = saturate(dot(n, h));
                float thresh = 1.0 - (1.0 - roughness) * 0.18;
                float spec_s = smoothstep(thresh - 0.02, thresh + 0.02, nh) * (1.0 - roughness) * 0.55;
                spec_term = lerp(float3(1), base, metalness) * spec_s;
            }
            o += light * (diffuse * (toon_nl / PI) + spec_term);
            continue;
        }
        float nl = saturate(dot(n, ldir));
        if (nl <= 0) {
            continue;
        }
        float3 h = normalize(ldir + v);
        float nh = saturate(dot(n, h));
        float vh = saturate(dot(v, h));
        float3 fresnel = f0 + (1 - f0) * pow(1 - vh, 5);
        float a2 = alpha * alpha;
        float gv = nl * sqrt(a2 + (1 - a2) * nv * nv);
        float gl = nv * sqrt(a2 + (1 - a2) * nl * nl);
        float vis = 0.5 / max(gv + gl, 1e-6);
        float dd = nh * nh * (a2 - 1) + 1;
        float dist_term = a2 / (PI * dd * dd);
        o += light * nl * (diffuse / PI + fresnel * (vis * dist_term));
    }
    if (lighting.envParams.x > 0) {
        float3 r = 2 * dot(n, v) * n - v;
        float3 irr = envIrradianceMap.SampleLevel(equirect_uv(n), 0).rgb;
        float3 rad = envRadianceMap.SampleLevel(equirect_uv(r), roughness * lighting.envParams.y).rgb;
        if (lighting.envFlags.x == 0u) {
            irr = linear_rgb(irr);
            rad = linear_rgb(rad);
        }
        float4 c0 = float4(-1, -0.0275, -0.572, 0.022);
        float4 c1 = float4(1, 0.0425, 1.04, -0.04);
        float4 rv = roughness * c0 + c1;
        float a004 = min(rv.x * rv.x, exp2(-9.28 * nv)) * rv.x + rv.y;
        float2 fab = float2(-1.04, 1.04) * a004 + rv.zw;
        float3 spec = f0 * fab.x + fab.y;
        o += (ao * lighting.envParams.x) * (irr * diffuse * (1 - spec) + rad * spec);
    }
    return o;
}

// The standard material's colour at a fragment (premultiplied sRGB), as
// fs_main writes it.
// `tint` (sRGB colour, opacity) multiplies the material's: a particle's row.
float4 surface(VS_Out i, bool front, float4 tint = float4(1)) {
    uint flags = material.flags;
    float3 base = material.color.rgb * linear_rgb(tint.rgb);
    float a = material.color.a * saturate(tint.a);
    float3 dp_dx = ddx(i.world);
    float3 dp_dy = ddy(i.world);
    float2 d0x = ddx(i.uv);
    float2 d0y = ddy(i.uv);
    float2 d1x = ddx(i.uv1);
    float2 d1y = ddy(i.uv1);
    if ((flags & FLAG_COLOR_MAP) != 0) {
        float2 uv, dx, dy;
        slot_uv(SLOT_COLOR, i.uv, i.uv1, d0x, d0y, d1x, d1y, uv, dx, dy);
        float4 tex = sample_map(colorMap, uv, dx, dy);
        if ((flags & FLAG_PREMULTIPLIED) != 0) {
            tex.rgb = tex.a > 0 ? saturate(tex.rgb / tex.a) : float3(0);
        }
        base *= linear_rgb(tex.rgb);
        if ((flags & (FLAG_BLENDED | FLAG_ALPHA_MASK)) != 0) {
            a *= tex.a;
        }
    }
    if ((flags & FLAG_ALPHA_MAP) != 0) {
        float2 uv, dx, dy;
        slot_uv(SLOT_ALPHA, i.uv, i.uv1, d0x, d0y, d1x, d1y, uv, dx, dy);
        a *= sample_map(alphaMap, uv, dx, dy).g;
    }
    if ((flags & FLAG_ALPHA_MASK) != 0) {
        if (a < material.alpha_cutoff) {
            discard;
        }
        if ((flags & FLAG_BLENDED) == 0) {
            a = 1;
        }
    }
    float3 c;
    if ((flags & FLAG_SHADOW_ONLY) != 0) {
        // Only the light its shadows take: 1 minus their factors' product.
        float3 sn = normalize(i.normal);
        float mask = 1;
        for (int si = 0; si < lighting.count.w; si++) {
            mask *= shadow_factor(si, i.world, sn);
        }
        a *= 1 - mask;
        c = base;
    } else if ((flags & FLAG_BASIC) != 0) {
        c = base;
    } else {
        float roughness = material.roughness;
        if ((flags & FLAG_ROUGHNESS_MAP) != 0) {
            float2 uv, dx, dy;
            slot_uv(SLOT_ROUGHNESS, i.uv, i.uv1, d0x, d0y, d1x, d1y, uv, dx, dy);
            roughness *= sample_map(roughnessMap, uv, dx, dy).g;
        }
        float metalness = material.metalness;
        if ((flags & FLAG_METALNESS_MAP) != 0) {
            float2 uv, dx, dy;
            slot_uv(SLOT_METALNESS, i.uv, i.uv1, d0x, d0y, d1x, d1y, uv, dx, dy);
            metalness *= sample_map(metalnessMap, uv, dx, dy).b;
        }
        float ao = 1;
        if ((flags & FLAG_AO_MAP) != 0) {
            float2 uv, dx, dy;
            slot_uv(SLOT_AO, i.uv, i.uv1, d0x, d0y, d1x, d1y, uv, dx, dy);
            ao = (sample_map(aoMap, uv, dx, dy).r - 1) * material.ao_intensity + 1;
        }
        float3 n = normalize(i.normal);
        if ((flags & FLAG_FLAT_SHADING) != 0) {
            n = normalize(cross(dp_dx, dp_dy));
            // Screen derivatives run with y down: face the eye.
            if (dot(n, lighting.eye.xyz - i.world) < 0) {
                n = -n;
            }
        }
        float4 shadows = float4(1);
        if ((flags & FLAG_RECEIVE_SHADOW) != 0) {
            for (int si = 0; si < lighting.count.w; si++) {
                shadows[si] = shadow_factor(si, i.world, n);
            }
        }
        if ((flags & FLAG_TWO_SIDED) != 0 && !front) {
            n = -n;
        }
        if ((flags & FLAG_NORMAL_MAP) != 0) {
            float2 uv, dx, dy;
            slot_uv(SLOT_NORMAL, i.uv, i.uv1, d0x, d0y, d1x, d1y, uv, dx, dy);
            float3 map_n = sample_map(normalMap, uv, dx, dy).rgb * 2 - 1;
            n = perturb_normal(n, dp_dx, dp_dy, dx, dy, map_n);
        }
        c = shade(base, roughness, metalness, i.world, n, lighting.eye.xyz, shadows, ao, flags);
    }
    float3 emissive = material.emissive.rgb;
    if ((flags & FLAG_EMISSIVE_MAP) != 0) {
        float2 uv, dx, dy;
        slot_uv(SLOT_EMISSIVE, i.uv, i.uv1, d0x, d0y, d1x, d1y, uv, dx, dy);
        emissive *= linear_rgb(sample_map(emissiveMap, uv, dx, dy).rgb);
    }
    c += emissive;
    c = tone_map(c, lighting.count.y, lighting.envParams.z);
    if (lighting.count.z != FOG_NONE) {
        float dist = length(i.world - lighting.eye.xyz);
        float f = 0;
        if (lighting.count.z == FOG_LINEAR) {
            f = saturate((dist - lighting.fogParams.x) / max(lighting.fogParams.y - lighting.fogParams.x, 1e-6));
        } else if (lighting.count.z == FOG_EXP2) {
            float d = lighting.envParams.w * dist;
            f = saturate(1 - exp(-d * d));
        }
        c = lerp(c, lighting.fogColor.rgb, f);
    }
    return float4(linear_to_srgb(c.r) * a, linear_to_srgb(c.g) * a, linear_to_srgb(c.b) * a, a);
}

[shader("fragment")]
float4 fs_main(VS_Out i, bool front : SV_IsFrontFace) : SV_Target {
    return surface(i, front);
}

// A view depth from a depth value (o3d.view_depth).
float view_depth(float depth) {
    float near = lighting.frameParams.z;
    float far = lighting.frameParams.w;
    if (lighting.envFlags.y != 0u) {
        return near + depth * (far - near);
    }
    return near * far / (far - depth * (far - near));
}

// A particle quad: its row's colour and opacity, faded by how far in front
// of the frame's depth it is (soft particles).
[shader("fragment")]
float4 fs_sprite(VS_Curves_Out i, bool front : SV_IsFrontFace) : SV_Target {
    float4 row = spriteRows[material.curve_table + i.quad];
    if (material.softness > 0) {
        float2 uv = i.position.xy / lighting.frameParams.xy;
        float behind = view_depth(sceneDepth.SampleLevel(uv, 0));
        row.a *= saturate((behind - view_depth(i.position.z)) / material.softness);
    }
    return surface(curves_surface_in(i), front, row);
}

// Sample positions in a pixel (the standard 4x MSAA pattern, software.odin's
// SAMPLE_OFFSETS), by sample index.
static const float2 SAMPLE_OFFSETS[4] = { float2(0.375, 0.125), float2(0.875, 0.375), float2(0.125, 0.625), float2(0.625, 0.875) };

bool curves_inside(StructuredBuffer<uint> words, uint s, float2 rc, bool even_odd) {
    int w = curves_winding(words, s, rc);
    return even_odd ? (w & 1) != 0 : w != 0;
}

static const uint CURVE_EVEN_ODD = 0x80000000u;

// Fine derivatives (within the 2x2 quad's row or column) where the target
// names them; Metal's dfdx has no coarse kind to choose.
float2 fine_ddx(float2 v) {
    __target_switch {
    case metal:
        return ddx(v);
    default:
        return ddx_fine(v);
    }
}

float2 fine_ddy(float2 v) {
    __target_switch {
    case metal:
        return ddy(v);
    default:
        return ddy_fine(v);
    }
}

struct Curves_Out {
    float4 color : SV_Target;
    uint mask : SV_Coverage;
};

// An exact shape's fragment: uv1 is its shape coordinate in its quad's
// curve set (the quad's entry in the draw's table). Blended, the
// pixel's covered area scales its colour (raster.odin's `curve_coverage`);
// opaque, the samples inside the shape are
// written (the coordinate at each from the centre's and its derivatives,
// raster.odin's `curve_samples`). Derivatives are fine (per row and column
// of the 2x2 quad, as the software renderer's) and taken before any
// discard.
[shader("fragment")]
Curves_Out fs_curves(VS_Curves_Out i, bool front : SV_IsFrontFace) {
    float2 dx = fine_ddx(i.uv1);
    float2 dy = fine_ddy(i.uv1);
    uint entry = curveWords[material.curve_table + i.quad];
    uint s = entry & ~CURVE_EVEN_ODD;
    bool even_odd = (entry & CURVE_EVEN_ODD) != 0u;
    float4 color = surface(curves_surface_in(i), front);
    Curves_Out o;
    if ((material.flags & FLAG_BLENDED) != 0) {
        float cov = curves_coverage_affine(curveWords, s, i.uv1, dx, dy, even_odd);
        if (cov <= 0.0) {
            discard;
        }
        o.color = color * cov;
        o.mask = 0xFu;
    } else {
        uint mask = 0;
        for (uint k = 0; k < 4; k++) {
            float2 off = SAMPLE_OFFSETS[k] - 0.5;
            if (curves_inside(curveWords, s, i.uv1 + dx * off.x + dy * off.y, even_odd)) {
                mask |= 1u << k;
            }
        }
        if (mask == 0u) {
            discard;
        }
        o.color = color;
        o.mask = mask;
    }
    return o;
}

// The shadow pass: the depth of the nearest caster per texel.
float shadow_depth(VS_Out i) {
    if ((material.flags & FLAG_ALPHA_MASK) != 0) {
        // Both maps from one uv set: the colour slot's, or the alpha slot's
        // without a colour map (as the software renderer's shadow pass).
        uint mask_slot = (material.flags & FLAG_COLOR_MAP) != 0 ? SLOT_COLOR : SLOT_ALPHA;
        float2 raw = ((material.uv_sets >> mask_slot) & 1) != 0 ? i.uv1 : i.uv;
        float a = material.color.a;
        if ((material.flags & FLAG_COLOR_MAP) != 0) {
            float3 r0 = material.uv_transforms[SLOT_COLOR * 2 + 0].xyz;
            float3 r1 = material.uv_transforms[SLOT_COLOR * 2 + 1].xyz;
            float2 uv = float2(dot(r0, float3(raw, 1)), 1 - dot(r1, float3(raw, 1)));
            a *= colorMap.SampleLevel(uv, 0).a;
        }
        if ((material.flags & FLAG_ALPHA_MAP) != 0) {
            float3 r0 = material.uv_transforms[SLOT_ALPHA * 2 + 0].xyz;
            float3 r1 = material.uv_transforms[SLOT_ALPHA * 2 + 1].xyz;
            float2 uv = float2(dot(r0, float3(raw, 1)), 1 - dot(r1, float3(raw, 1)));
            a *= shadowAlphaMap.SampleLevel(uv, 0).g;
        }
        if (a < material.alpha_cutoff) {
            discard;
        }
    }
    return i.position.z;
}

[shader("fragment")]
float fs_shadow(VS_Out i) : SV_Target {
    return shadow_depth(i);
}

// An exact shape's shadow: the texels inside its outline.
[shader("fragment")]
float fs_shadow_curves(VS_Curves_Out i) : SV_Target {
    uint entry = shadowCurveWords[material.curve_table + i.quad];
    if (!curves_inside(shadowCurveWords, entry & ~CURVE_EVEN_ODD, i.uv1, (entry & CURVE_EVEN_ODD) != 0u)) {
        discard;
    }
    return shadow_depth(curves_surface_in(i));
}
