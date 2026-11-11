// The frame's depth pass where nothing is drawn (shading.odin's
// Velocity_Far pass, first, under the draws): the far plane, moving only as
// the camera did (u_opened_far: from a pixel's clip position on the far
// plane now to where it was at the shutter's opening), its nearness
// o3d's MOTION_FAR (the prelude defines it).
layout(location = 0) out float o_depth;
layout(location = 1) out vec2 o_velocity;
layout(location = 2) out float o_nearness;

void main() {
	vec2 c = gl_FragCoord.xy;
	vec2 ndc = vec2(c.x / u_frame.x * 2.0 - 1.0, 1.0 - c.y / u_frame.y * 2.0);
	vec3 o = (u_opened_far * vec4(ndc, 0.0, 1.0)).xyw;
	o_depth = 0.0;
	o_velocity = vec2(0.0);
	if (o.z > 1e-6) {
		vec2 n = o.xy / o.z;
		o_velocity = c - vec2((n.x * 0.5 + 0.5) * u_frame.x, (0.5 - n.y * 0.5) * u_frame.y);
	}
	o_nearness = MOTION_FAR;
}
