// o3d's vertex stage for its own draws (oksl GLSL, compiled at load per
// variant, shading.odin): a mesh's vertex morphed by its active targets
// (GEOMETRY 0), a skinned one's also through its joints (1), or an exact
// shape's or particle's quad corner, which passes its quad on (2); for the
// colour pass (PASS 0), a shadow map's (1), or the frame's depth with
// motion blur (2), which also places it as at the shutter's opening (its
// model-view-projection, morph targets and joints then). The prelude
// (vertex_prelude) defines GEOMETRY and PASS and declares the uniforms
// (Draw_Block's fields as u_<field>) and O3D_COMMON.

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;
layout(location = 3) in vec2 uv1;
// sRGB bytes and alpha (o3d.Vertex.color), normalized.
layout(location = 4) in vec4 color;
#if GEOMETRY == 1
layout(location = 5) in uvec4 joints;
layout(location = 6) in vec4 weights;
#endif

#if PASS == 0
out vec3 v_world;
out vec3 v_normal;
// Linear, as it is interpolated.
out vec4 v_color;
#endif
out vec2 v_uv;
out vec2 v_uv1;
#if PASS == 2
// Clip x, y and w at the shutter's opening.
out vec3 v_opened;
#endif
#if GEOMETRY == 2
flat out uint v_quad;
#endif

#if GEOMETRY != 2
// o3d.Morph_Delta: the draw's targets' deltas, vertex by vertex (the draw's
// vertices counted from 0: gl_VertexID).
struct Morph_Delta {
	vec4 position;
	vec4 normal;
};

layout(std430) readonly buffer Morph_Deltas {
	Morph_Delta deltas[];
} morphDeltas;
#endif

#if GEOMETRY == 1
// The frame's joint table (o3d.joint_table): the draw's joints now from
// u_skin.x, at the shutter's opening from u_skin.y.
// A joint's matrix by its columns (o3d.Mat4's layout).
struct Joint {
	vec4 c0;
	vec4 c1;
	vec4 c2;
	vec4 c3;
};

layout(std430) readonly buffer Skin_Joints {
	Joint mats[];
} skinJoints;

mat4 joint_matrix(uint i) {
	Joint j = skinJoints.mats[i];
	return mat4(j.c0, j.c1, j.c2, j.c3);
}

mat4 skin_matrix(uint base) {
	return joint_matrix(base + joints.x) * weights.x + joint_matrix(base + joints.y) * weights.y + joint_matrix(base + joints.z) * weights.z + joint_matrix(base + joints.w) * weights.w;
}
#endif

#if PASS == 2
// The vertex as at the shutter's opening: its targets then, its joints then.
vec3 opened_position() {
	vec3 pos = position;
#if GEOMETRY != 2
	for (uint k = 0u; k < u_opened_morph_info.x; k++) {
		pos += morphDeltas.deltas[u_opened_morph_bases[k >> 2u][k & 3u] + uint(gl_VertexID)].position.xyz * u_opened_morph_weights[k >> 2u][k & 3u];
	}
#endif
#if GEOMETRY == 1
	pos = (skin_matrix(u_skin.y) * vec4(pos, 1.0)).xyz;
#endif
	return pos;
}
#endif

void main() {
	vec3 pos = position;
	vec3 norm = normal;
#if GEOMETRY != 2
	for (uint k = 0u; k < u_morph_info.x; k++) {
		uint base = u_morph_bases[k >> 2u][k & 3u];
		float w = u_morph_weights[k >> 2u][k & 3u];
		Morph_Delta d = morphDeltas.deltas[base + uint(gl_VertexID)];
		pos += d.position.xyz * w;
		norm += d.normal.xyz * w;
	}
#endif
#if GEOMETRY == 1
	mat4 skin = skin_matrix(u_skin.x);
	pos = (skin * vec4(pos, 1.0)).xyz;
	norm = (skin * vec4(norm, 0.0)).xyz;
#endif
	gl_Position = u_mvp * vec4(pos, 1.0);
#if PASS == 0
	v_world = (u_model * vec4(pos, 1.0)).xyz;
	v_normal = (u_normal * vec4(norm, 0.0)).xyz;
	v_color = vec4(linear_rgb(color.rgb), color.a);
#endif
#if PASS == 2
	v_opened = (u_opened_mvp * vec4(opened_position(), 1.0)).xyw;
#endif
	v_uv = uv;
	v_uv1 = uv1;
#if GEOMETRY == 2
	v_quad = uint(gl_VertexID) / 4u;
#endif
}
