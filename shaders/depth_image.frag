// The frame's depth as distance (shading.odin's Depth_Image pass): rgb the
// distance along the camera's axis (o3d.view_depth) and a 1 where
// something was drawn; rgb u_params.z and a 0 where nothing was. u_params:
// the camera's near and far, the far for empty pixels, 1 when
// orthographic; u_frame: the frame's width and height.
uniform sampler2D depthMap;

layout(location = 0) out vec4 o_depth;

void main() {
	float d = textureLod(depthMap, gl_FragCoord.xy / u_frame.xy, 0.0).r;
	if (d <= 0.0) {
		o_depth = vec4(u_params.zzz, 0.0);
		return;
	}
	float near = u_params.x;
	float far = u_params.y;
	float v = u_params.w != 0.0 ? far - d * (far - near) : near * far / (near + d * (far - near));
	o_depth = vec4(v, v, v, 1.0);
}
