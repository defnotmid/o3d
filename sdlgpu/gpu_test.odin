package o3d_sdlgpu

import "core:log"
import "core:os"
import "core:testing"

// A device for a GPU test. Without one the test fails, so the GPU is never
// silently left unchecked, unless O3D_NO_GPU is set (CI without a GPU, or
// macOS, whose Metal driver needs the main thread the test runner does not
// give: tools/gpu_check runs the same cases there), which skips it.
test_gpu :: proc(t: ^testing.T, g: ^GPU) -> bool {
	if init(g) == nil {
		return true
	}
	buf: [16]u8
	if os.get_env(buf[:], "O3D_NO_GPU") != "" {
		log.warn("no GPU device: skipped (O3D_NO_GPU)")
		return false
	}
	testing.fail_now(t, "no GPU device (set O3D_NO_GPU=1 to skip the GPU tests)")
}

// A PARITY_CASES case drawn the same by the GPU and the software renderer.
@(private = "file")
expect_parity :: proc(t: ^testing.T, name: string) {
	index := -1
	for c, i in PARITY_CASES {
		if c.name == name {
			index = i
		}
	}
	if index < 0 {
		testing.fail_now(t, "no such parity case")
	}
	g: GPU
	if !test_gpu(t, &g) {
		return
	}
	defer destroy(&g)
	psnr, gpu_px, cpu_px, err := parity_check(&g, PARITY_CASES[index])
	defer delete(gpu_px)
	defer delete(cpu_px)
	testing.expectf(t, err == "", "%s", err)
	testing.expectf(t, err != "" || psnr > PARITY_MIN_PSNR, "%s: PSNR %.1f dB", name, psnr)
}

// Standard materials and every kind of light.
@(test)
test_gpu_matches_software :: proc(t: ^testing.T) {expect_parity(t, "standard")}

// ShaderMaterials: oksl's SPIR-V / MSL against its programs over the
// software renderer's triangles and quads.
@(test)
test_custom_matches_software :: proc(t: ^testing.T) {expect_parity(t, "custom")}

// The shadow map, the PCF filter and the shadowed light.
@(test)
test_shadow_matches_software :: proc(t: ^testing.T) {expect_parity(t, "shadow")}

// Colour, normal, roughness and emissive maps with mipmaps.
@(test)
test_textures_match_software :: proc(t: ^testing.T) {expect_parity(t, "textures")}

// Environment lighting and a blurred background.
@(test)
test_environment_matches_software :: proc(t: ^testing.T) {expect_parity(t, "environment")}

// An animated glTF model.
@(test)
test_gltf_matches_software :: proc(t: ^testing.T) {expect_parity(t, "gltf")}

// One GPU through every case in turn, as tools/gpu_check and a host that
// remakes its tables use it: each case's new tables (ids and sizes reused)
// replace what the GPU mirrored, so nothing stale is drawn or read.
@(test)
test_gpu_reused_across_tables :: proc(t: ^testing.T) {
	g: GPU
	if !test_gpu(t, &g) {
		return
	}
	defer destroy(&g)
	for c in PARITY_CASES {
		psnr, gpu_px, cpu_px, err := parity_check(&g, c)
		delete(gpu_px)
		delete(cpu_px)
		testing.expectf(t, err == "" && psnr > PARITY_MIN_PSNR, "%s after the others: %s PSNR %.1f dB", c.name, err, psnr)
	}
}
