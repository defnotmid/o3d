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

static const uint TONE_NONE = 0;
static const uint TONE_LINEAR = 1;
static const uint TONE_REINHARD = 2;
static const uint TONE_CINEON = 3;
static const uint TONE_ACES_FILMIC = 4;
static const uint TONE_AGX = 5;
static const uint TONE_NEUTRAL = 6;

float3 tone_map(float3 c, uint mode, float exposure) {
    if (mode == TONE_NONE) {
        return c;
    }
    float exp_val = exposure > 0.0 ? exposure : 1.0;
    float3 v = c * exp_val;
    if (mode == TONE_LINEAR) {
        return saturate(v);
    }
    if (mode == TONE_REINHARD) {
        return saturate(v / (float3(1.0) + v));
    }
    if (mode == TONE_CINEON) {
        float3 x = max(v - float3(0.004), float3(0.0));
        float3 u = (x * (6.2 * x + float3(0.5))) / (x * (6.2 * x + float3(1.7)) + float3(0.06));
        return pow(u, float3(2.2));
    }
    if (mode == TONE_ACES_FILMIC) {
        v *= 1.0 / 0.6;
        v = float3(
            0.59719 * v.r + 0.35458 * v.g + 0.04823 * v.b,
            0.07600 * v.r + 0.90834 * v.g + 0.01566 * v.b,
            0.02840 * v.r + 0.13383 * v.g + 0.83777 * v.b
        );
        float3 a = v * (v + float3(0.0245786)) - float3(0.000090537);
        float3 b = v * (0.983729 * v + float3(0.4329510)) + float3(0.238081);
        v = a / b;
        v = float3(
             1.60475 * v.r - 0.53108 * v.g - 0.07367 * v.b,
            -0.10208 * v.r + 1.10813 * v.g - 0.00605 * v.b,
            -0.00327 * v.r - 0.07276 * v.g + 1.07602 * v.b
        );
        return saturate(v);
    }
    if (mode == TONE_AGX) {
        float3 r2020 = float3(
            0.6274039 * v.r + 0.3292830 * v.g + 0.0433131 * v.b,
            0.0690973 * v.r + 0.9195404 * v.g + 0.0113623 * v.b,
            0.0163914 * v.r + 0.0880133 * v.g + 0.8955953 * v.b
        );
        r2020 = max(r2020, float3(1e-10));
        float3 inset = float3(
            0.85662715 * r2020.r + 0.09512124 * r2020.g + 0.04825161 * r2020.b,
            0.13731897 * r2020.r + 0.76124199 * r2020.g + 0.10143904 * r2020.b,
            0.11189821 * r2020.r + 0.07679942 * r2020.g + 0.81130237 * r2020.b
        );
        inset = max(inset, float3(1e-10));
        float3 ev = log2(inset);
        float3 x = saturate((ev - float3(-12.47393)) * (1.0 / 16.5));
        float3 x2 = x * x;
        float3 x4 = x2 * x2;
        x = 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - float3(0.00232);
        float3 outset = float3(
             1.12710058 * x.r - 0.11060664 * x.g - 0.01649394 * x.b,
            -0.14132976 * x.r + 1.15782370 * x.g - 0.01649394 * x.b,
            -0.14132976 * x.r - 0.11060664 * x.g + 1.25193641 * x.b
        );
        outset = max(outset, float3(0.0));
        float3 lin2020 = pow(outset, float3(2.2));
        float3 out709 = float3(
             1.6604910 * lin2020.r - 0.5876411 * lin2020.g - 0.0728499 * lin2020.b,
            -0.1245505 * lin2020.r + 1.1328999 * lin2020.g - 0.0083494 * lin2020.b,
            -0.0181508 * lin2020.r - 0.1005789 * lin2020.g + 1.1187297 * lin2020.b
        );
        return saturate(out709);
    }
    if (mode == TONE_NEUTRAL) {
        const float START_COMPRESSION = 0.8 - 0.04;
        const float DESATURATION = 0.15;
        float x = min(v.r, min(v.g, v.b));
        float offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
        float3 p = v - float3(offset);
        float peak = max(p.r, max(p.g, p.b));
        if (peak < START_COMPRESSION) {
            return saturate(p);
        }
        const float d = 1.0 - START_COMPRESSION;
        float new_peak = 1.0 - d * d / (peak + d - START_COMPRESSION);
        p *= new_peak / peak;
        float g = 1.0 - 1.0 / (DESATURATION * (peak - new_peak) + 1.0);
        return saturate(lerp(p, float3(new_peak), g));
    }
    return c;
}

