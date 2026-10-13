// o3d's vertex and fragment shaders for SDL_GPU, compiled by build.sh to
// SPIR-V (Vulkan) and MSL (Metal). The shading must stay the software
// renderer's (scene.odin `shade`): three.js's MeshStandardMaterial without
// an environment map.
//
// SDL_GPU binding convention: vertex uniforms set 1, fragment uniforms
// set 3.

static const float PI = 3.14159265358979323846;
static const int MAX_LIGHTS = 8;
// Light_Kind's values and Shading_Flag's bits (o3d_test checks).
static const uint LIGHT_AMBIENT = 0;
static const uint LIGHT_HEMISPHERE = 1;
static const uint LIGHT_DIRECTIONAL = 2;
static const uint LIGHT_POINT = 3;
static const uint FLAG_BASIC = 1;
static const uint FLAG_FLAT_SHADING = 2;
static const uint FLAG_TWO_SIDED = 4;
static const uint FLAG_BLENDED = 8;
static const uint FLAG_RECEIVE_SHADOW = 16;

struct Draw_VS {
    float4x4 mvp;
    float4x4 model;
    float4x4 normal;
};

// Shading_Material: color is linear rgb, a = opacity (1 when not blended);
// emissive is linear rgb times intensity; flags are FLAG_* bits.
struct Material {
    float4 color;
    float4 emissive;
    float metalness;
    float roughness;
    uint flags;
    uint pad;
};

// Shading_Light: color, ground linear times intensity; vector the direction
// toward a directional light or a point light's position; kind a LIGHT_*;
// shadowed: the shadow light (its direct light scaled by the shadow).
struct Light {
    float4 color;
    float4 ground;
    float4 vector;
    uint kind;
    float distance;
    float decay;
    uint shadowed;
};

// shadowMatrix: the shadow light's view-projection; shadowParams: map size
// (0: no shadow), bias, normal bias.
struct Lighting {
    float4 eye;
    int4 count;
    float4x4 shadowMatrix;
    float4 shadowParams;
    Light lights[MAX_LIGHTS];
};

[[vk::binding(0, 1)]] ConstantBuffer<Draw_VS> draw : register(b0, space1);
[[vk::binding(0, 3)]] ConstantBuffer<Material> material : register(b0, space3);
[[vk::binding(1, 3)]] ConstantBuffer<Lighting> lighting : register(b1, space3);
// The shadow map: depth per texel (R32_FLOAT), read texel by texel.
[[vk::binding(0, 2)]] Sampler2D<float> shadowMap : register(t0, space2);

struct VS_In {
    float3 position : TEXCOORD0;
    float3 normal : TEXCOORD1;
    float2 uv : TEXCOORD2;
};

struct VS_Out {
    float4 position : SV_Position;
    float3 world : TEXCOORD0;
    float3 normal : TEXCOORD1;
    float2 uv : TEXCOORD2;
};

[shader("vertex")]
VS_Out vs_main(VS_In i) {
    VS_Out o;
    o.position = mul(draw.mvp, float4(i.position, 1));
    o.world = mul(draw.model, float4(i.position, 1)).xyz;
    o.normal = mul(draw.normal, float4(i.normal, 0)).xyz;
    o.uv = i.uv;
    return o;
}

float linear_to_srgb(float c) {
    c = saturate(c);
    return c <= 0.0031308 ? c * 12.92 : 1.055 * pow(c, 1.0 / 2.4) - 0.055;
}

// shadow.odin's shadow_factor: a 3x3 grid of bilinear depth comparisons,
// a 4x4 block of texels weighed (1 - f, 1, 1, f) per axis, edges repeated,
// outside the light's camera lit.
float shadow_factor(float3 world, float3 normal) {
    float3 p = world + normal * lighting.shadowParams.z;
    float4 c = mul(lighting.shadowMatrix, float4(p, 1));
    float u = c.x * 0.5 + 0.5;
    float v = 0.5 - c.y * 0.5;
    float z = c.z + lighting.shadowParams.y;
    if (u < 0 || u > 1 || v < 0 || v > 1 || z > 1) {
        return 1;
    }
    int size = int(lighting.shadowParams.x);
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
            acc += wx[i] * (z <= shadowMap.Load(int3(col, row, 0)) ? 1.0 : 0.0);
        }
        lit += wy[j] * acc;
    }
    return lit / 9;
}

float3 shade(float3 base, float3 world, float3 n, float3 eye, float shadow) {
    float metalness = material.metalness;
    float3 v = normalize(eye - world);
    float3 diffuse = base * (1 - metalness);
    float3 f0 = lerp(float3(0.04), base, metalness);
    float roughness = clamp(material.roughness, 0.0525, 1);
    float alpha = roughness * roughness;
    float nv = clamp(dot(n, v), 1e-4, 1);
    float3 o = float3(0);
    for (int k = 0; k < lighting.count.x; k++) {
        Light l = lighting.lights[k];
        if (l.kind == LIGHT_AMBIENT) {
            o += l.color.rgb * diffuse / PI;
            continue;
        }
        if (l.kind == LIGHT_HEMISPHERE) {
            float w = 0.5 * n.y + 0.5;
            o += lerp(l.ground.rgb, l.color.rgb, w) * diffuse / PI;
            continue;
        }
        float3 ldir = l.vector.xyz;
        float3 light = l.color.rgb;
        if (l.kind == LIGHT_POINT) {
            float3 d = l.vector.xyz - world;
            float dist = length(d);
            ldir = d / max(dist, 1e-6);
            float decay = l.decay > 0 ? l.decay : 2;
            float falloff = 1 / max(pow(dist, decay), 0.01);
            if (l.distance > 0) {
                float t = saturate(1 - pow(dist / l.distance, 4));
                falloff *= t * t;
            }
            light *= falloff;
        }
        if (l.shadowed != 0) {
            light *= shadow;
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
    return o;
}

[shader("fragment")]
float4 fs_main(VS_Out i, bool front : SV_IsFrontFace) : SV_Target {
    uint flags = material.flags;
    float3 base = material.color.rgb;
    float3 c;
    if ((flags & FLAG_BASIC) != 0) {
        c = base;
    } else {
        float3 n = normalize(i.normal);
        if ((flags & FLAG_FLAT_SHADING) != 0) {
            n = normalize(cross(ddx(i.world), ddy(i.world)));
            // Screen derivatives run with y down: face the eye.
            if (dot(n, lighting.eye.xyz - i.world) < 0) {
                n = -n;
            }
        }
        float shadow = 1;
        if ((flags & FLAG_RECEIVE_SHADOW) != 0 && lighting.shadowParams.x > 0) {
            shadow = shadow_factor(i.world, n);
        }
        if ((flags & FLAG_TWO_SIDED) != 0 && !front) {
            n = -n;
        }
        c = shade(base, i.world, n, lighting.eye.xyz, shadow);
    }
    c += material.emissive.rgb;
    float a = material.color.a;
    return float4(linear_to_srgb(c.r) * a, linear_to_srgb(c.g) * a, linear_to_srgb(c.b) * a, a);
}

// The shadow pass: the depth of the nearest caster per texel.
[shader("fragment")]
float fs_shadow(float4 position : SV_Position) : SV_Target {
    return position.z;
}
