// o3d's fullscreen passes' vertex stage (oksl GLSL, shading.odin): a
// triangle over the whole target from 3 vertices, no buffer. The fragment
// stages work from gl_FragCoord.
void main() {
	gl_Position = vec4(gl_VertexID == 1 ? 3.0 : -1.0, gl_VertexID == 2 ? 3.0 : -1.0, 0.0, 1.0);
}
