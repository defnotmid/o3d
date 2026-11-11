// The frame's background where nothing is drawn (shading.odin's
// Background pass): its equirectangular map, or its prefiltered radiance
// when blurred, along each pixel's view ray at u_background.y's level,
// times u_background.x, as light (u_background.z: the map is linear, not
// sRGB-encoded).
uniform sampler2D bgMap;

layout(location = 0) out vec4 o_light;

void main() {
	vec2 ndc = vec2(gl_FragCoord.x / u_frame.x * 2.0 - 1.0, 1.0 - gl_FragCoord.y / u_frame.y * 2.0);
	// From the near plane (depth 1, reversed) to the far (0).
	vec4 p0 = u_inv_view_proj * vec4(ndc, 1.0, 1.0);
	vec4 p1 = u_inv_view_proj * vec4(ndc, 0.0, 1.0);
	vec3 dir = normalize(p1.xyz / p1.w - p0.xyz / p0.w);
	vec3 bg = textureLod(bgMap, equirect_uv(dir), u_background.y).rgb;
	if (u_background.z == 0.0) {
		bg = linear_rgb(bg);
	}
	o_light = vec4(bg * u_background.x, 1.0);
}
