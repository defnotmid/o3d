// The frame's depth map as distance (o3d_sdlgpu.record_depth_into): a
// fullscreen pass into the host's RGBA16F target, compiled by build.sh to
// SPIR-V (Vulkan) and MSL (Metal). Drawn with 3 vertices via SV_VertexID.
struct Depth_FS {
    float4 params; // x = near, y = far (the camera's), z = far for empty pixels, w = orthographic
    float4 size; // xy: the frame's width and height
};

[[vk::binding(0, 3)]] ConstantBuffer<Depth_FS> depth_fs : register(b0, space3);
[[vk::binding(0, 2)]] Sampler2D<float> depthMap : register(t0, space2);

[shader("vertex")]
float4 vs_main(uint vid : SV_VertexID) : SV_Position {
    return float4(vid == 1 ? 3.0 : -1.0, vid == 2 ? 3.0 : -1.0, 0, 1);
}

// rgb the distance along the camera's axis (o3d.view_depth), a 1 where
// something was drawn; rgb the given far and a 0 where nothing was.
[shader("fragment")]
float4 fs_main(float4 position : SV_Position) : SV_Target {
    float d = depthMap.SampleLevel(position.xy / depth_fs.size.xy, 0);
    if (d <= 0) {
        return float4(depth_fs.params.zzz, 0);
    }
    float near = depth_fs.params.x;
    float far = depth_fs.params.y;
    float v = depth_fs.params.w != 0 ? far - d * (far - near) : near * far / (near + d * (far - near));
    return float4(v, v, v, 1);
}
