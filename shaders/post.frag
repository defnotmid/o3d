// The frame's passes after its light (oksl GLSL, shading.odin's post
// passes; the prelude defines one PASS_<NAME>, and o3d's FOCUS_TILE and
// Tone_Mapping's values as TONE_<MODE>), both renderers'
// fullscreen passes: its output (its light, premultiplied, tone mapped and
// sRGB-encoded once a pixel); its motion blur (the longest velocity per
// tile and about it, then the gather, written as light); its bloom (a chain
// the output adds) and its depth of field (the largest blur per tile and
// near it, a half-size copy, then the gather, written as the output).
//
// Post_Block's fields: u_tone: the tone mapping, exposure, the light
// texture's size; u_lens: the focus, blur and its most (pixels), how many
// tiles a blur reaches; u_camera: near, far, orthographic; u_frame: its
// size, its tiles across and down; u_motion: the longest half streak
// (pixels), nearness's scale, the tiles' size (pixels); u_motion_size: the
// velocity and nearness maps' size, the motion tiles across and down;
// u_bloom: the threshold, the Gaussian's sigma on the chain's last level,
// the share of it the output adds; u_bloom_size: the Gaussian's axis, the
// size of the level drawn.

// The frame's light (at the light texture's size); its depth (reversed, 0
// where nothing is drawn); a tile table (the tiles' largest blurs, or the
// largest near them); the half-size copy's light and keys; bloom, read
// linearly (the level a pass reads: the output's, the chain's last); motion
// blur: each pixel's velocity and nearness, a motion tile table (the tiles'
// longest velocity, or the longest about them). A pass declares the ones it
// reads (the renderers bind them by name).
#if defined(PASS_OUTPUT) || defined(PASS_BLOOM_BRIGHT) || defined(PASS_FOCUS_HALF) || defined(PASS_FOCUS_GATHER) || defined(PASS_MOTION_GATHER)
uniform sampler2D lightMap;
#endif
#if defined(PASS_OUTPUT) || defined(PASS_BLOOM_DOWN) || defined(PASS_BLOOM_BLUR) || defined(PASS_FOCUS_GATHER)
uniform sampler2D bloomMap;
#endif
#if defined(PASS_FOCUS_ROWS) || defined(PASS_FOCUS_HALF) || defined(PASS_FOCUS_GATHER)
uniform sampler2D depthMap;
#endif
#if defined(PASS_FOCUS_TILES) || defined(PASS_FOCUS_NEAR) || defined(PASS_FOCUS_GATHER)
uniform sampler2D tileMap;
#endif
#ifdef PASS_FOCUS_GATHER
uniform sampler2D halfMap;
uniform sampler2D halfKeyMap;
#endif
#if defined(PASS_MOTION_ROWS) || defined(PASS_MOTION_GATHER)
uniform sampler2D velocityMap;
#endif
#ifdef PASS_MOTION_GATHER
uniform sampler2D nearnessMap;
#endif
#if defined(PASS_MOTION_TILES) || defined(PASS_MOTION_AROUND) || defined(PASS_MOTION_GATHER)
uniform sampler2D motionMap;
#endif

#ifdef PASS_FOCUS_HALF
layout(location = 0) out vec4 o_light;
layout(location = 1) out float o_key;
#elif defined(PASS_FOCUS_ROWS) || defined(PASS_FOCUS_TILES) || defined(PASS_FOCUS_NEAR)
layout(location = 0) out float o_blur;
#elif defined(PASS_MOTION_ROWS) || defined(PASS_MOTION_TILES) || defined(PASS_MOTION_AROUND)
layout(location = 0) out vec2 o_motion;
#else
layout(location = 0) out vec4 o_light;
#endif

// The radius past which the depth of field's gather reads the half-size
// copy, in pixels.
const float FOCUS_INNER = 6.0;
// A pixel counts for no more light than this in bloom's chain: one
// highlight far past white, on lacquer, would flood the glow.
const float BLOOM_MOST_LIGHT = 16.0;
// The points a pixel's motion gather takes along its tiles' longest
// velocity, one end to the other.
const int MOTION_TAPS = 15;

ivec2 pixel() {
	return ivec2(gl_FragCoord.xy);
}

#if defined(PASS_OUTPUT) || defined(PASS_FOCUS_GATHER)
float linear_to_srgb(float c) {
	c = clamp(c, 0.0, 1.0);
	return c <= 0.0031308 ? c * 12.92 : 1.055 * pow(c, 1.0 / 2.4) - 0.055;
}

// three.js's tone mappings.
vec3 tone_map(vec3 c, int mode, float exposure) {
	if (mode == TONE_NONE) {
		return c;
	}
	vec3 v = c * (exposure > 0.0 ? exposure : 1.0);
	if (mode == TONE_LINEAR) {
		return clamp(v, 0.0, 1.0);
	}
	if (mode == TONE_REINHARD) {
		return clamp(v / (vec3(1.0) + v), 0.0, 1.0);
	}
	if (mode == TONE_CINEON) {
		vec3 x = max(v - vec3(0.004), vec3(0.0));
		vec3 u = (x * (6.2 * x + vec3(0.5))) / (x * (6.2 * x + vec3(1.7)) + vec3(0.06));
		return pow(u, vec3(2.2));
	}
	if (mode == TONE_ACES_FILMIC) {
		v *= 1.0 / 0.6;
		v = vec3(0.59719 * v.r + 0.35458 * v.g + 0.04823 * v.b, 0.07600 * v.r + 0.90834 * v.g + 0.01566 * v.b, 0.02840 * v.r + 0.13383 * v.g + 0.83777 * v.b);
		vec3 a = v * (v + vec3(0.0245786)) - vec3(0.000090537);
		vec3 b = v * (0.983729 * v + vec3(0.4329510)) + vec3(0.238081);
		v = a / b;
		v = vec3(1.60475 * v.r - 0.53108 * v.g - 0.07367 * v.b, -0.10208 * v.r + 1.10813 * v.g - 0.00605 * v.b, -0.00327 * v.r - 0.07276 * v.g + 1.07602 * v.b);
		return clamp(v, 0.0, 1.0);
	}
	if (mode == TONE_AGX) {
		vec3 r2020 = vec3(0.6274039 * v.r + 0.3292830 * v.g + 0.0433131 * v.b, 0.0690973 * v.r + 0.9195404 * v.g + 0.0113623 * v.b, 0.0163914 * v.r + 0.0880133 * v.g + 0.8955953 * v.b);
		r2020 = max(r2020, vec3(1e-10));
		vec3 inset = vec3(0.85662715 * r2020.r + 0.09512124 * r2020.g + 0.04825161 * r2020.b, 0.13731897 * r2020.r + 0.76124199 * r2020.g + 0.10143904 * r2020.b, 0.11189821 * r2020.r + 0.07679942 * r2020.g + 0.81130237 * r2020.b);
		inset = max(inset, vec3(1e-10));
		vec3 ev = log2(inset);
		vec3 x = clamp((ev - vec3(-12.47393)) * (1.0 / 16.5), 0.0, 1.0);
		vec3 x2 = x * x;
		vec3 x4 = x2 * x2;
		x = 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - vec3(0.00232);
		vec3 outset = vec3(1.12710058 * x.r - 0.11060664 * x.g - 0.01649394 * x.b, -0.14132976 * x.r + 1.15782370 * x.g - 0.01649394 * x.b, -0.14132976 * x.r - 0.11060664 * x.g + 1.25193641 * x.b);
		outset = max(outset, vec3(0.0));
		vec3 lin2020 = pow(outset, vec3(2.2));
		vec3 out709 = vec3(1.6604910 * lin2020.r - 0.5876411 * lin2020.g - 0.0728499 * lin2020.b, -0.1245505 * lin2020.r + 1.1328999 * lin2020.g - 0.0083494 * lin2020.b, -0.0181508 * lin2020.r - 0.1005789 * lin2020.g + 1.1187297 * lin2020.b);
		return clamp(out709, 0.0, 1.0);
	}
	if (mode == TONE_NEUTRAL) {
		const float START_COMPRESSION = 0.8 - 0.04;
		const float DESATURATION = 0.15;
		float x = min(v.r, min(v.g, v.b));
		float offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
		vec3 p = v - vec3(offset);
		float peak = max(p.r, max(p.g, p.b));
		if (peak < START_COMPRESSION) {
			return clamp(p, 0.0, 1.0);
		}
		const float d = 1.0 - START_COMPRESSION;
		float new_peak = 1.0 - d * d / (peak + d - START_COMPRESSION);
		p *= new_peak / peak;
		float g = 1.0 - 1.0 / (DESATURATION * (peak - new_peak) + 1.0);
		return clamp(mix(p, vec3(new_peak), g), 0.0, 1.0);
	}
	return c;
}

// Light past what its coverage holds (added light over nothing) stays, as
// colour past the alpha (up to full), which a premultiplied over adds.
vec4 output_of(vec4 light) {
	float a = clamp(light.a, 0.0, 1.0);
	if (a <= 0.0) {
		return vec4(0.0);
	}
	vec3 c = tone_map(light.rgb / a, int(u_tone.x), u_tone.y);
	float k = clamp(max(c.r, max(c.g, c.b)), 1.0, 1.0 / a);
	c /= k;
	float m = a * k;
	return vec4(linear_to_srgb(c.r) * m, linear_to_srgb(c.g) * m, linear_to_srgb(c.b) * m, a);
}

// A pixel's light with the bloom added; light over nothing gives the pixel
// as much coverage.
vec4 bloomed(vec4 light, ivec2 p) {
	if (u_bloom.z <= 0.0) {
		return light;
	}
	vec3 s = textureLod(bloomMap, (vec2(p) + 0.5) / u_frame.xy, 0.0).rgb * u_bloom.z;
	return vec4(light.rgb + s, max(light.a, min(max(s.r, max(s.g, s.b)), 1.0)));
}
#endif

#if defined(PASS_OUTPUT) || defined(PASS_BLOOM_BRIGHT) || defined(PASS_FOCUS_HALF) || defined(PASS_FOCUS_GATHER) || defined(PASS_MOTION_GATHER)
vec4 light_at(ivec2 p) {
	return textureLod(lightMap, (vec2(p) + 0.5) / u_tone.zw, 0.0);
}
#endif

#if defined(PASS_FOCUS_ROWS) || defined(PASS_FOCUS_HALF) || defined(PASS_FOCUS_GATHER)
// A pixel's key from its depth (reversed, 0 where nothing is): 1 / its
// distance along the camera's axis (o3d.view_depth), 0 where nothing is.
float key_at(ivec2 p) {
	float d = textureLod(depthMap, (vec2(p) + 0.5) / u_frame.xy, 0.0).r;
	if (d <= 0.0) {
		return 0.0;
	}
	float near = u_camera.x;
	float far = u_camera.y;
	float v = u_camera.z != 0.0 ? far - d * (far - near) : near * far / (near + d * (far - near));
	return 1.0 / v;
}

// A pixel's blur radius from its key.
float focus_blur(float key) {
	return min(u_lens.y * abs(1.0 - u_lens.x * key), u_lens.z);
}
#endif

#if defined(PASS_MOTION_ROWS) || defined(PASS_MOTION_GATHER)
vec2 motion_cut(vec2 v) {
	float l = length(v);
	return l > u_motion.x ? v * (u_motion.x / l) : v;
}

vec2 velocity_at(ivec2 p) {
	return textureLod(velocityMap, (vec2(p) + 0.5) / u_motion_size.xy, 0.0).xy;
}
#endif

#if defined(PASS_MOTION_AROUND) || defined(PASS_MOTION_GATHER)
vec2 motion_tile(ivec2 t) {
	return textureLod(motionMap, (vec2(t) + 0.5) / u_motion_size.zw, 0.0).xy;
}
#endif

#ifdef PASS_MOTION_GATHER
float nearness_at(ivec2 p) {
	return textureLod(nearnessMap, (vec2(p) + 0.5) / u_motion_size.xy, 0.0).r;
}

float motion_reach(float d, float l) {
	return clamp(l - d + 0.5, 0.0, 1.0);
}

float motion_jitter(ivec2 p) {
	return fract(52.9829189 * fract(0.06711056 * float(p.x) + 0.00583715 * float(p.y)));
}
#endif

#ifdef PASS_BLOOM_BRIGHT
vec3 bloom_above(vec3 l) {
	l *= min(1.0, BLOOM_MOST_LIGHT / max(max(l.r, max(l.g, l.b)), 1e-4));
	return max(l - u_bloom.x, vec3(0.0));
}
#endif

void main() {
	ivec2 p = pixel();
#if defined(PASS_OUTPUT)
	o_light = output_of(bloomed(light_at(p), p));
#elif defined(PASS_BLOOM_BRIGHT)
	// The light above the threshold of the four pixels under each of the
	// chain's first level's.
	ivec2 a = p * 2;
	ivec2 b = min(a + 1, ivec2(u_frame.xy) - 1);
	vec3 c = bloom_above(light_at(a).rgb) + bloom_above(light_at(ivec2(b.x, a.y)).rgb) + bloom_above(light_at(ivec2(a.x, b.y)).rgb) + bloom_above(light_at(b).rgb);
	o_light = vec4(c * 0.25, 1.0);
#elif defined(PASS_BLOOM_DOWN)
	// A level at half the size of the one before.
	o_light = textureLod(bloomMap, gl_FragCoord.xy / u_bloom_size.zw, 0.0);
#elif defined(PASS_BLOOM_BLUR)
	// The Gaussian along one axis (u_bloom_size.xy) of the last level; past
	// its edge there is nothing.
	ivec2 size = ivec2(u_bloom_size.zw);
	ivec2 dir = ivec2(u_bloom_size.xy);
	float sigma = u_bloom.y;
	int reach = int(ceil(3.0 * sigma));
	float g1 = exp(-0.5 / (sigma * sigma));
	float g2 = g1 * g1;
	float g0 = 1.0 / (sqrt(2.0 * 3.14159265358979) * sigma);
	vec3 sum = textureLod(bloomMap, (vec2(p) + 0.5) / vec2(size), 0.0).rgb * g0;
	float g = g0 * g1;
	float step = g1 * g2;
	for (int i = 1; i <= reach; i++) {
		ivec2 a = p - dir * i;
		ivec2 b = p + dir * i;
		if (a.x >= 0 && a.y >= 0) {
			sum += textureLod(bloomMap, (vec2(a) + 0.5) / vec2(size), 0.0).rgb * g;
		}
		if (b.x < size.x && b.y < size.y) {
			sum += textureLod(bloomMap, (vec2(b) + 0.5) / vec2(size), 0.0).rgb * g;
		}
		g *= step;
		step *= g2;
	}
	o_light = vec4(sum, 1.0);
#elif defined(PASS_FOCUS_ROWS)
	// A tile's row's largest blur (the tiles across by the frame's rows).
	float m = 0.0;
	for (int x = p.x * FOCUS_TILE; x < min((p.x + 1) * FOCUS_TILE, int(u_frame.x)); x++) {
		m = max(m, focus_blur(key_at(ivec2(x, p.y))));
	}
	o_blur = m;
#elif defined(PASS_FOCUS_TILES)
	// A tile's: the largest of its rows'.
	float m = 0.0;
	for (int y = p.y * FOCUS_TILE; y < min((p.y + 1) * FOCUS_TILE, int(u_frame.y)); y++) {
		m = max(m, textureLod(tileMap, (vec2(float(p.x), float(y)) + 0.5) / vec2(u_frame.z, u_frame.y), 0.0).r);
	}
	o_blur = m;
#elif defined(PASS_FOCUS_NEAR)
	// The largest blur of the tiles within reach.
	ivec2 tiles = ivec2(u_frame.zw);
	int reach = int(u_lens.w);
	float m = 0.0;
	for (int y = max(p.y - reach, 0); y < min(p.y + reach + 1, tiles.y); y++) {
		for (int x = max(p.x - reach, 0); x < min(p.x + reach + 1, tiles.x); x++) {
			m = max(m, textureLod(tileMap, (vec2(float(x), float(y)) + 0.5) / vec2(tiles), 0.0).r);
		}
	}
	o_blur = m;
#elif defined(PASS_FOCUS_HALF)
	// The half-size copy: each 2x2 block's light and key averaged (the last
	// row and column repeated on odd sizes).
	ivec2 size = ivec2(u_frame.xy);
	vec4 light = vec4(0.0);
	float key = 0.0;
	for (int dy = 0; dy < 2; dy++) {
		for (int dx = 0; dx < 2; dx++) {
			ivec2 q = min(p * 2 + ivec2(dx, dy), size - 1);
			light += light_at(q);
			key += key_at(q);
		}
	}
	o_light = light * 0.25;
	o_key = key * 0.25;
#elif defined(PASS_FOCUS_GATHER)
	// A pixel: the mean of samples on a golden-angle spiral out to the
	// largest blur near it, a sample counting where its own blur reaches back
	// to the pixel (one behind it no more than twice the pixel's), as the
	// output.
	ivec2 size = ivec2(u_frame.xy);
	ivec2 half_size = (size + 1) / 2;
	float reach = textureLod(tileMap, (vec2(p / FOCUS_TILE) + 0.5) / u_frame.zw, 0.0).r;
	vec4 sum = light_at(p);
	if (reach >= 0.5) {
		float centre_key = key_at(p);
		float centre_blur = focus_blur(centre_key);
		float count = 1.0;
		float radius = 0.8;
		vec2 dir = vec2(1.0, 0.0);
		while (radius < reach) {
			vec2 o = dir * radius;
			vec4 light;
			float key;
			if (radius < FOCUS_INNER) {
				ivec2 s = clamp(ivec2(floor(vec2(p) + o + 0.5)), ivec2(0), size - 1);
				light = light_at(s);
				key = key_at(s);
			} else {
				ivec2 s = clamp(ivec2(floor((vec2(p) + 0.5 + o) * 0.5)), ivec2(0), half_size - 1);
				vec2 uv = (vec2(s) + 0.5) / vec2(half_size);
				light = textureLod(halfMap, uv, 0.0);
				key = textureLod(halfKeyMap, uv, 0.0).r;
			}
			float blur = focus_blur(key);
			if (key < centre_key) {
				blur = min(blur, centre_blur * 2.0);
			}
			float m = clamp(blur - radius + 0.5, 0.0, 1.0);
			sum += (sum / count) * (1.0 - m) + light * m;
			count += 1.0;
			radius += (radius < FOCUS_INNER ? 1.6 : 3.2) / radius;
			dir = vec2(dir.x * -0.73736887 - dir.y * 0.67549029, dir.x * 0.67549029 + dir.y * -0.73736887);
		}
		sum /= count;
	}
	o_light = output_of(bloomed(sum, p));
#elif defined(PASS_MOTION_ROWS)
	// A tile's row's longest velocity (the tiles across by the frame's rows).
	int k = int(u_motion.z);
	vec2 m = vec2(0.0);
	for (int x = p.x * k; x < min((p.x + 1) * k, int(u_frame.x)); x++) {
		vec2 v = motion_cut(velocity_at(ivec2(x, p.y)));
		if (dot(v, v) > dot(m, m)) {
			m = v;
		}
	}
	o_motion = m;
#elif defined(PASS_MOTION_TILES)
	// A tile's: the longest of its rows'.
	int k = int(u_motion.z);
	vec2 m = vec2(0.0);
	for (int y = p.y * k; y < min((p.y + 1) * k, int(u_frame.y)); y++) {
		vec2 v = textureLod(motionMap, (vec2(float(p.x), float(y)) + 0.5) / vec2(u_motion_size.z, u_frame.y), 0.0).xy;
		if (dot(v, v) > dot(m, m)) {
			m = v;
		}
	}
	o_motion = m;
#elif defined(PASS_MOTION_AROUND)
	// The longest velocity of a tile and the eight about it.
	ivec2 tiles = ivec2(u_motion_size.zw);
	vec2 m = vec2(0.0);
	for (int y = max(p.y - 1, 0); y < min(p.y + 2, tiles.y); y++) {
		for (int x = max(p.x - 1, 0); x < min(p.x + 2, tiles.x); x++) {
			vec2 v = motion_tile(ivec2(x, y));
			if (dot(v, v) > dot(m, m)) {
				m = v;
			}
		}
	}
	o_motion = m;
#elif defined(PASS_MOTION_GATHER)
	// A pixel gathered across its tiles' longest motion, as light
	// (motion_blur.odin says how).
	ivec2 size = ivec2(u_frame.xy);
	vec4 centre = light_at(p);
	vec2 vn = motion_tile(p / int(u_motion.z));
	if (dot(vn, vn) < 0.25) {
		o_light = centre;
		return;
	}
	float lx = max(length(motion_cut(velocity_at(p))), 0.5);
	float ln = length(vn);
	float nx = nearness_at(p);
	vec4 cover = vec4(0.0);
	vec4 own = centre;
	float covered = 0.0;
	float owned = 1.0;
	float jitter = motion_jitter(p) - 0.5;
	for (int k = 0; k < MOTION_TAPS; k++) {
		float t = -1.0 + 2.0 * (float(k) + jitter + 1.0) / float(MOTION_TAPS + 1);
		ivec2 s = clamp(ivec2(floor(vec2(p) + 0.5 + vn * t)), ivec2(0), size - 1);
		float d = length(vec2(s - p));
		float ly = max(length(motion_cut(velocity_at(s))), 0.5);
		float ny = nearness_at(s);
		float front = clamp(1.0 - (nx - ny) / u_motion.y, 0.0, 1.0);
		float level = clamp(1.0 - (ny - nx) / u_motion.y, 0.0, 1.0);
		vec4 light = light_at(s);
		float c = front * (1.0 - level) * motion_reach(d, ly) * ln / (ly * float(MOTION_TAPS));
		cover += light * c;
		covered += c;
		float o = level * max(motion_reach(d, lx), motion_reach(d, ly));
		own += light * o;
		owned += o;
	}
	float share = min(covered, 1.0);
	vec4 front_light = covered > 0.0 ? cover / covered : vec4(0.0);
	o_light = front_light * share + own / owned * (1.0 - share);
#endif
}
