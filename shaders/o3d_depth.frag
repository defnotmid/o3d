// o3d's depth passes for its own draws (oksl GLSL, compiled at load per
// variant, shading.odin): a shadow map's (ENTRY 6: the nearest caster's
// depth) and the frame's with motion blur (ENTRY 7: its depth, each pixel's
// velocity, its place now less where it was at the shutter's opening, in
// pixels, and its nearness, -ln of its distance), each as both renderers
// draw them. An exact shape covers the pixels inside its outline; a masked
// material (MASKED) the ones its maps leave at its cutoff or more. The
// prelude (fragment_prelude) declares the uniforms (Material_Block's and
// Frame_Block's fields as u_<field>), O3D_COMMON and, for exact shapes,
// ocurves' curves.GLSL.

// A uv set's varying only when the variant reads it (the prelude's UV0,
// UV1).
#ifdef UV0
in vec2 v_uv;
#endif
#ifdef UV1
in vec2 v_uv1;
#endif
#ifdef FLAG_CURVES
flat in uint v_quad;
#endif
#if ENTRY == 7
in vec3 v_opened;
#endif

layout(location = 0) out float o_depth;
#if ENTRY == 7
layout(location = 1) out vec2 o_velocity;
layout(location = 2) out float o_nearness;
#endif

#ifdef MASKED
#ifdef FLAG_COLOR_MAP
uniform sampler2D colorMap;
#endif
#ifdef FLAG_ALPHA_MAP
uniform sampler2D alphaMap;
#endif
#endif

#ifdef FLAG_CURVES
layout(std430) readonly buffer Curve_Words {
	uint words[];
} curveWords;

uint curves_word(uint i) {
	return curveWords.words[i];
}

const uint CURVE_EVEN_ODD = 0x80000000u;

int curve_entry_rule(uint entry) {
	return (entry & CURVE_EVEN_ODD) != 0u ? CURVES_EVEN_ODD : CURVES_NON_ZERO;
}
#endif

#ifdef MASKED
// A map's level 0 at a slot's uv (the shadow passes read no derivatives).
vec4 mask_map(sampler2D tex, vec4 r0, vec4 r1, vec2 raw) {
	return textureLod(tex, vec2(dot(r0.xyz, vec3(raw, 1.0)), 1.0 - dot(r1.xyz, vec3(raw, 1.0))), 0.0);
}
#endif

void main() {
#ifdef FLAG_CURVES
	uint entry = curves_word(u_table.x + v_quad);
	if (!curves_fill_inside(curves_winding(entry & ~CURVE_EVEN_ODD, v_uv1), curve_entry_rule(entry))) {
		discard;
	}
#endif
#ifdef MASKED
	// Both maps from one uv set: the colour slot's, or the alpha slot's
	// without a colour map.
#ifdef FLAG_COLOR_MAP
	vec2 raw = UV_COLOR;
#else
	vec2 raw = UV_ALPHA;
#endif
	float a = u_color.a;
#ifdef FLAG_COLOR_MAP
	a *= mask_map(colorMap, u_uv_transforms[ROW0_COLOR], u_uv_transforms[ROW1_COLOR], raw).a;
#endif
#ifdef FLAG_ALPHA_MAP
	a *= mask_map(alphaMap, u_uv_transforms[ROW0_ALPHA], u_uv_transforms[ROW1_ALPHA], raw).g;
#endif
	if (a < u_surface.w) {
		discard;
	}
#endif
	float z = gl_FragCoord.z;
	o_depth = z;
#if ENTRY == 7
	vec2 c = gl_FragCoord.xy;
	o_velocity = vec2(0.0);
	if (v_opened.z > 1e-6) {
		vec2 ndc = v_opened.xy / v_opened.z;
		o_velocity = c - vec2((ndc.x * 0.5 + 0.5) * u_frame.x, (0.5 - ndc.y * 0.5) * u_frame.y);
	}
	float near = u_camera.x;
	float far = u_camera.y;
	float d = u_camera.z != 0.0 ? far - z * (far - near) : near * far / (near + z * (far - near));
	o_nearness = -log(d);
#endif
}
