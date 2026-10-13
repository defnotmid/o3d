// o3d's vertex and fragment shaders for SDL_GPU, compiled by build.sh to
// SPIR-V (Vulkan) and MSL (Metal). The shading must stay the software
// renderer's (scene.odin `shade`): three.js's MeshStandardMaterial without
// an environment map.
//
// SDL_GPU binding convention: vertex uniforms set 1, fragment uniforms
// set 3.

static const float PI = 3.14159265358979323846;
static const int MAX_LIGHTS = 8;
static const int AMBIENT = 0;
static const int HEMISPHERE = 1;
static const int DIRECTIONAL = 2;
static const int POINT = 3;

struct Draw_VS {
    float4x4 mvp;
    float4x4 model;
    float4x4 normal;
};

// color: linear rgb, a = opacity (1 when not blended); emissive: linear
// rgb times intensity; params: metalness, roughness, flags (1 basic,
// 2 flat shading, 4 lit from both sides).
struct Material {
    float4 color;
    float4 emissive;
    float4 params;
};

// color, ground: linear times intensity; vector: the direction toward a
// directional light or a point light's position; params: kind, distance,
// decay.
struct Light {
    float4 color;
    float4 ground;
    float4 vector;
    float4 params;
};

struct Lighting {
    float4 eye;
    int4 count;
    Light lights[MAX_LIGHTS];
};

[[vk::binding(0, 1)]] ConstantBuffer<Draw_VS> draw : register(b0, space1);
[[vk::binding(0, 3)]] ConstantBuffer<Material> material : register(b0, space3);
[[vk::binding(1, 3)]] ConstantBuffer<Lighting> lighting : register(b1, space3);

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

float3 shade(float3 base, float3 world, float3 n, float3 eye) {
    float metalness = material.params.x;
    float3 v = normalize(eye - world);
    float3 diffuse = base * (1 - metalness);
    float3 f0 = lerp(float3(0.04), base, metalness);
    float roughness = clamp(material.params.y, 0.0525, 1);
    float alpha = roughness * roughness;
    float nv = clamp(dot(n, v), 1e-4, 1);
    float3 o = float3(0);
    for (int k = 0; k < lighting.count.x; k++) {
        Light l = lighting.lights[k];
        int kind = int(l.params.x);
        if (kind == AMBIENT) {
            o += l.color.rgb * diffuse / PI;
            continue;
        }
        if (kind == HEMISPHERE) {
            float w = 0.5 * n.y + 0.5;
            o += lerp(l.ground.rgb, l.color.rgb, w) * diffuse / PI;
            continue;
        }
        float3 ldir = l.vector.xyz;
        float3 light = l.color.rgb;
        if (kind == POINT) {
            float3 d = l.vector.xyz - world;
            float dist = length(d);
            ldir = d / max(dist, 1e-6);
            float decay = l.params.z > 0 ? l.params.z : 2;
            float falloff = 1 / max(pow(dist, decay), 0.01);
            if (l.params.y > 0) {
                float t = saturate(1 - pow(dist / l.params.y, 4));
                falloff *= t * t;
            }
            light *= falloff;
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
    int flags = int(material.params.z);
    float3 base = material.color.rgb;
    float3 c;
    if ((flags & 1) != 0) {
        c = base;
    } else {
        float3 n = normalize(i.normal);
        if ((flags & 2) != 0) {
            n = normalize(cross(ddx(i.world), ddy(i.world)));
            // Screen derivatives run with y down: face the eye.
            if (dot(n, lighting.eye.xyz - i.world) < 0) {
                n = -n;
            }
        }
        if ((flags & 4) != 0 && !front) {
            n = -n;
        }
        c = shade(base, i.world, n, lighting.eye.xyz);
    }
    c += material.emissive.rgb;
    float a = material.color.a;
    return float4(linear_to_srgb(c.r) * a, linear_to_srgb(c.g) * a, linear_to_srgb(c.b) * a, a);
}
