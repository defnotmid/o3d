// The frame's passes after its light for SDL_GPU, compiled by build.sh to
// SPIR-V (Vulkan) and MSL (Metal), each drawn with 3 vertices via
// SV_VertexID: its output (its light, premultiplied, tone mapped and
// sRGB-encoded once a pixel, as raster.odin's `output`), its motion blur
// (motion_blur.odin's phases: the longest velocity per tile and about it,
// then the gather, written as light) and its depth of field
// (depth_of_field.odin's phases: the largest blur per tile and near it, a
// half-size copy, then the gather, written as the output).
import common;

// tone: the tone mapping, exposure, the light texture's size; lens: the
// focus, blur and its most (pixels), how many tiles a blur reaches; camera:
// near, far, orthographic; frame: its size, its tiles across and down;
// motion: the longest half streak (pixels), nearness's scale, the tiles'
// size (pixels); motion_size: the velocity and nearness maps' size, the
// motion tiles across and down.
struct Post_FS {
    float4 tone;
    float4 lens;
    float4 camera;
    float4 frame;
    float4 motion;
    float4 motion_size;
};

[[vk::binding(0, 3)]] ConstantBuffer<Post_FS> post : register(b0, space3);
// The frame's light (at the targets' size); its depth (reversed, 0 where
// nothing is drawn); a tile table (the tiles' largest blurs, or the largest
// near them); the half-size copy's light and keys.
[[vk::binding(0, 2)]] Sampler2D<float4> lightMap : register(t0, space2);
[[vk::binding(1, 2)]] Sampler2D<float> depthMap : register(t1, space2);
[[vk::binding(2, 2)]] Sampler2D<float> tileMap : register(t2, space2);
[[vk::binding(3, 2)]] Sampler2D<float4> halfMap : register(t3, space2);
[[vk::binding(4, 2)]] Sampler2D<float> halfKeyMap : register(t4, space2);
// Motion blur: each pixel's velocity and nearness; a motion tile table (the
// tiles' longest velocity, or the longest about them).
[[vk::binding(5, 2)]] Sampler2D<float2> velocityMap : register(t5, space2);
[[vk::binding(6, 2)]] Sampler2D<float> nearnessMap : register(t6, space2);
[[vk::binding(7, 2)]] Sampler2D<float2> motionMap : register(t7, space2);

static const int FOCUS_TILE = 16;
static const float FOCUS_INNER = 6;

[shader("vertex")]
float4 vs_main(uint vid : SV_VertexID) : SV_Position {
    return float4(vid == 1 ? 3.0 : -1.0, vid == 2 ? 3.0 : -1.0, 0, 1);
}

// Light past what its coverage holds (added light over nothing) stays, as
// colour past the alpha (up to full), which a premultiplied over adds.
float4 output(float4 light) {
    float a = saturate(light.a);
    if (a <= 0) {
        return float4(0);
    }
    float3 c = tone_map(light.rgb / a, uint(post.tone.x), post.tone.y);
    float k = clamp(max(c.r, max(c.g, c.b)), 1, 1 / a);
    c /= k;
    float m = a * k;
    return float4(linear_to_srgb(c.r) * m, linear_to_srgb(c.g) * m, linear_to_srgb(c.b) * m, a);
}

float4 light_at(int2 p) {
    return lightMap.SampleLevel((float2(p) + 0.5) / post.tone.zw, 0);
}

// o3d.focus_key and focus_blur.
float key_at(int2 p) {
    float d = depthMap.SampleLevel((float2(p) + 0.5) / post.frame.xy, 0);
    if (d <= 0) {
        return 0;
    }
    float near = post.camera.x;
    float far = post.camera.y;
    float v = post.camera.z != 0 ? far - d * (far - near) : near * far / (near + d * (far - near));
    return 1 / v;
}

float focus_blur(float key) {
    return min(post.lens.y * abs(1 - post.lens.x * key), post.lens.z);
}

[shader("fragment")]
float4 fs_output(float4 position : SV_Position) : SV_Target {
    return output(light_at(int2(position.xy)));
}

// A tile's largest blur.
[shader("fragment")]
float fs_focus_tiles(float4 position : SV_Position) : SV_Target {
    int2 t = int2(position.xy);
    int2 size = int2(post.frame.xy);
    float m = 0;
    for (int y = t.y * FOCUS_TILE; y < min((t.y + 1) * FOCUS_TILE, size.y); y++) {
        for (int x = t.x * FOCUS_TILE; x < min((t.x + 1) * FOCUS_TILE, size.x); x++) {
            m = max(m, focus_blur(key_at(int2(x, y))));
        }
    }
    return m;
}

// The largest blur of the tiles within reach.
[shader("fragment")]
float fs_focus_near(float4 position : SV_Position) : SV_Target {
    int2 t = int2(position.xy);
    int2 tiles = int2(post.frame.zw);
    int reach = int(post.lens.w);
    float m = 0;
    for (int y = max(t.y - reach, 0); y < min(t.y + reach + 1, tiles.y); y++) {
        for (int x = max(t.x - reach, 0); x < min(t.x + reach + 1, tiles.x); x++) {
            m = max(m, tileMap.SampleLevel((float2(x, y) + 0.5) / float2(tiles), 0));
        }
    }
    return m;
}

// The half-size copy: each 2x2 block's light and key averaged (the last
// row and column repeated on odd sizes).
struct Half_Out {
    float4 light : SV_Target0;
    float key : SV_Target1;
};

[shader("fragment")]
Half_Out fs_focus_half(float4 position : SV_Position) {
    int2 h = int2(position.xy);
    int2 size = int2(post.frame.xy);
    Half_Out o;
    o.light = float4(0);
    o.key = 0;
    for (int dy = 0; dy < 2; dy++) {
        for (int dx = 0; dx < 2; dx++) {
            int2 p = min(h * 2 + int2(dx, dy), size - 1);
            o.light += light_at(p);
            o.key += key_at(p);
        }
    }
    o.light *= 0.25;
    o.key *= 0.25;
    return o;
}

// A pixel: the mean of samples on a golden-angle spiral out to the largest
// blur near it, a sample counting where its own blur reaches back to the
// pixel (one behind it no more than twice the pixel's), as the output.
[shader("fragment")]
float4 fs_focus(float4 position : SV_Position) : SV_Target {
    int2 p = int2(position.xy);
    int2 size = int2(post.frame.xy);
    int2 half_size = (size + 1) / 2;
    float reach = tileMap.SampleLevel((float2(p / FOCUS_TILE) + 0.5) / post.frame.zw, 0);
    float4 sum = light_at(p);
    if (reach >= 0.5) {
        float centre_key = key_at(p);
        float centre_blur = focus_blur(centre_key);
        float count = 1;
        float radius = 0.8;
        float2 dir = float2(1, 0);
        while (radius < reach) {
            float2 o = dir * radius;
            float4 light;
            float key;
            if (radius < FOCUS_INNER) {
                int2 s = clamp(int2(floor(float2(p) + o + 0.5)), int2(0), size - 1);
                light = light_at(s);
                key = key_at(s);
            } else {
                int2 s = clamp(int2(floor((float2(p) + 0.5 + o) * 0.5)), int2(0), half_size - 1);
                float2 uv = (float2(s) + 0.5) / float2(half_size);
                light = halfMap.SampleLevel(uv, 0);
                key = halfKeyMap.SampleLevel(uv, 0);
            }
            float blur = focus_blur(key);
            if (key < centre_key) {
                blur = min(blur, centre_blur * 2);
            }
            float m = saturate(blur - radius + 0.5);
            sum += (sum / count) * (1 - m) + light * m;
            count += 1;
            radius += (radius < FOCUS_INNER ? 1.6 : 3.2) / radius;
            dir = float2(dir.x * -0.73736887 - dir.y * 0.67549029, dir.x * 0.67549029 + dir.y * -0.73736887);
        }
        sum /= count;
    }
    return output(sum);
}

// Motion blur (motion_blur.odin's, which says how).
static const int MOTION_TAPS = 15;

float2 motion_cut(float2 v) {
    float l = length(v);
    return l > post.motion.x ? v * (post.motion.x / l) : v;
}

float2 velocity_at(int2 p) {
    return velocityMap.SampleLevel((float2(p) + 0.5) / post.motion_size.xy, 0);
}

float nearness_at(int2 p) {
    return nearnessMap.SampleLevel((float2(p) + 0.5) / post.motion_size.xy, 0);
}

float2 motion_tile(int2 t) {
    return motionMap.SampleLevel((float2(t) + 0.5) / post.motion_size.zw, 0);
}

float motion_reach(float d, float l) {
    return saturate(l - d + 0.5);
}

float motion_jitter(int2 p) {
    return frac(52.9829189 * frac(0.06711056 * float(p.x) + 0.00583715 * float(p.y)));
}

// A tile's row's longest velocity (the tiles across by the frame's rows),
// then a tile's: the longest of its rows'.
[shader("fragment")]
float2 fs_motion_rows(float4 position : SV_Position) : SV_Target {
    int2 t = int2(position.xy);
    int k = int(post.motion.z);
    float2 m = float2(0);
    for (int x = t.x * k; x < min((t.x + 1) * k, int(post.frame.x)); x++) {
        float2 v = motion_cut(velocity_at(int2(x, t.y)));
        if (dot(v, v) > dot(m, m)) {
            m = v;
        }
    }
    return m;
}

[shader("fragment")]
float2 fs_motion_tiles(float4 position : SV_Position) : SV_Target {
    int2 t = int2(position.xy);
    int k = int(post.motion.z);
    float2 m = float2(0);
    for (int y = t.y * k; y < min((t.y + 1) * k, int(post.frame.y)); y++) {
        float2 v = motionMap.SampleLevel((float2(t.x, y) + 0.5) / float2(post.motion_size.z, post.frame.y), 0);
        if (dot(v, v) > dot(m, m)) {
            m = v;
        }
    }
    return m;
}

// The longest velocity of a tile and the eight about it.
[shader("fragment")]
float2 fs_motion_around(float4 position : SV_Position) : SV_Target {
    int2 t = int2(position.xy);
    int2 tiles = int2(post.motion_size.zw);
    float2 m = float2(0);
    for (int y = max(t.y - 1, 0); y < min(t.y + 2, tiles.y); y++) {
        for (int x = max(t.x - 1, 0); x < min(t.x + 2, tiles.x); x++) {
            float2 v = motion_tile(int2(x, y));
            if (dot(v, v) > dot(m, m)) {
                m = v;
            }
        }
    }
    return m;
}

// A pixel gathered across its tiles' longest motion, as light.
[shader("fragment")]
float4 fs_motion(float4 position : SV_Position) : SV_Target {
    int2 p = int2(position.xy);
    int2 size = int2(post.frame.xy);
    float4 centre = light_at(p);
    float2 vn = motion_tile(p / int(post.motion.z));
    if (dot(vn, vn) < 0.25) {
        return centre;
    }
    float lx = max(length(motion_cut(velocity_at(p))), 0.5);
    float ln = length(vn);
    float nx = nearness_at(p);
    float4 cover = float4(0);
    float4 own = centre;
    float covered = 0;
    float owned = 1;
    float jitter = motion_jitter(p) - 0.5;
    for (int k = 0; k < MOTION_TAPS; k++) {
        float t = -1 + 2 * (float(k) + jitter + 1) / float(MOTION_TAPS + 1);
        int2 s = clamp(int2(floor(float2(p) + 0.5 + vn * t)), int2(0), size - 1);
        float d = length(float2(s - p));
        float ly = max(length(motion_cut(velocity_at(s))), 0.5);
        float ny = nearness_at(s);
        float front = saturate(1 - (nx - ny) / post.motion.y);
        float level = saturate(1 - (ny - nx) / post.motion.y);
        float4 light = light_at(s);
        float c = front * (1 - level) * motion_reach(d, ly) * ln / (ly * MOTION_TAPS);
        cover += light * c;
        covered += c;
        float o = level * max(motion_reach(d, lx), motion_reach(d, ly));
        own += light * o;
        owned += o;
    }
    float share = min(covered, 1);
    float4 front_light = covered > 0 ? cover / covered : float4(0);
    return front_light * share + own / owned * (1 - share);
}
