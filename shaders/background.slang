// Fullscreen equirectangular background pass for SDL_GPU (its light), compiled by
// build.sh to SPIR-V (Vulkan) and MSL (Metal). Drawn with 3 vertices via
// SV_VertexID (no vertex buffer), unprojecting the camera ray per corner and
// sampling the background texture (or its prefiltered radiance pyramid when
// blurred) at the requested LOD.
import common;

struct Background_VS {
    float4x4 inv_vp;
};

struct Background_FS {
    float4 params; // x = intensity, y = lod
    uint4 flags; // x: the map is linear (HDR), not sRGB-encoded
};

[[vk::binding(0, 1)]] ConstantBuffer<Background_VS> bg_vs : register(b0, space1);
[[vk::binding(0, 3)]] ConstantBuffer<Background_FS> bg_fs : register(b0, space3);
[[vk::binding(0, 2)]] Sampler2D<float4> bgMap : register(t0, space2);

struct VS_Out {
    float4 position : SV_Position;
    float3 ray : TEXCOORD0;
};

[shader("vertex")]
VS_Out vs_main(uint vid : SV_VertexID) {
    float2 ndc = float2(vid == 1 ? 3.0 : -1.0, vid == 2 ? 3.0 : -1.0);
    // From the near plane (depth 1, reversed) to the far (0).
    float4 p0 = mul(bg_vs.inv_vp, float4(ndc, 1, 1));
    float4 p1 = mul(bg_vs.inv_vp, float4(ndc, 0, 1));
    VS_Out o;
    o.position = float4(ndc, 0, 1);
    o.ray = p1.xyz / p1.w - p0.xyz / p0.w;
    return o;
}

[shader("fragment")]
float4 fs_main(VS_Out i) : SV_Target {
    float3 dir = normalize(i.ray);
    float3 bg = bgMap.SampleLevel(equirect_uv(dir), bg_fs.params.y).rgb;
    if (bg_fs.flags.x == 0u) {
        bg = linear_rgb(bg);
    }
    // Light, as the draws over it write theirs.
    return float4(bg * bg_fs.params.x, 1);
}

