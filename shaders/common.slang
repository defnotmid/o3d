// Shared colour-space and equirectangular projection helpers for o3d's
// Slang shaders.

static const float PI = 3.14159265358979323846;

float srgb_to_linear(float c) {
    return c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4);
}

float3 linear_rgb(float3 c) {
    return float3(srgb_to_linear(c.r), srgb_to_linear(c.g), srgb_to_linear(c.b));
}

float linear_to_srgb(float c) {
    c = saturate(c);
    return c <= 0.0031308 ? c * 12.92 : 1.055 * pow(c, 1.0 / 2.4) - 0.055;
}

// Maps unit direction `d` to top-down texture coordinates matching sample_mip.
float2 equirect_uv(float3 d) {
    float u = atan2(d.z, d.x) * (0.5 / PI) + 0.5;
    float v = 0.5 - asin(clamp(d.y, -1.0, 1.0)) * (1.0 / PI);
    return float2(u, v);
}
