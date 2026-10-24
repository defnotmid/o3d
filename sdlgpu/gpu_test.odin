package o3d_sdlgpu

import "core:log"
import "core:os"
import "core:testing"

import sdl "vendor:sdl3"

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

// Morph targets and morph + skinning.
@(test)
test_morph_matches_software :: proc(t: ^testing.T) {expect_parity(t, "morph")}

// InstancedMesh, LineSegments, Points, tone mapping, and fog.
@(test)
test_manim_matches_software :: proc(t: ^testing.T) {expect_parity(t, "manim")}

// CapsuleGeometry, a per-frame tube, and instanceColor / instanceId.
@(test)
test_shapes_matches_software :: proc(t: ^testing.T) {expect_parity(t, "shapes")}

// ExtrudeGeometry (bevels, smoothed walls, caps and sides as two
// materials) and ShapeGeometry, from shapes_from_contours.
@(test)
test_extrude_matches_software :: proc(t: ^testing.T) {expect_parity(t, "extrude")}

// Exact shapes: coverage by area (transparent) and by sample (opaque), the
// even-odd rule, a slanted and a small one, a colour map.
@(test)
test_curves_match_software :: proc(t: ^testing.T) {expect_parity(t, "curves")}

// Exact shapes cast their outlines' shadows.
@(test)
test_curve_shadow_matches_software :: proc(t: ^testing.T) {expect_parity(t, "curve_shadow")}

// Glyph runs: quads of shared curve sets, overlapping, mirrored, grazing,
// shadowed and colour-mapped.
@(test)
test_text_matches_software :: proc(t: ^testing.T) {expect_parity(t, "text")}

// Labels as particles: behind a cut-out net, pinned before a wall by
// depth_offset, anchored by their bottom edge; a premultiplied panel.
@(test)
test_labels_match_software :: proc(t: ^testing.T) {expect_parity(t, "labels")}

// Additive meshes, a ShaderMaterial and lines over a planet.
@(test)
test_additive_match_software :: proc(t: ^testing.T) {expect_parity(t, "additive")}

// Lens flares faded by their probes, drawn over everything.
@(test)
test_flare_match_software :: proc(t: ^testing.T) {expect_parity(t, "flare")}

// Vertex colours on a lit mesh, a fading panel, lines and points.
@(test)
test_vertex_colors_match_software :: proc(t: ^testing.T) {expect_parity(t, "vertex_colors")}

// Layers 0.005 apart 200 units out with far 40000 (reversed depth).
@(test)
test_far_layers_match_software :: proc(t: ^testing.T) {expect_parity(t, "far_layers")}

// Triplanar mapping in object and world frames on meshes without uvs.
@(test)
test_triplanar_match_software :: proc(t: ^testing.T) {expect_parity(t, "triplanar")}
@(test)
test_physical_match_software :: proc(t: ^testing.T) {expect_parity(t, "physical")}
@(test)
test_transmission_match_software :: proc(t: ^testing.T) {expect_parity(t, "transmission")}

// ShaderMaterials receiving shadows through shadowAt.
@(test)
test_custom_shadow_match_software :: proc(t: ^testing.T) {expect_parity(t, "custom_shadow")}

// ShaderMaterials finished with the scene's tone mapping and fog.
@(test)
test_custom_finish_match_software :: proc(t: ^testing.T) {expect_parity(t, "custom_finish")}

// Additive draws over an opaque clear colour.
@(test)
test_background_color_match_software :: proc(t: ^testing.T) {expect_parity(t, "background_color")}

// A baked Sky as the background and the environment.
@(test)
test_sky_match_software :: proc(t: ^testing.T) {expect_parity(t, "sky")}

// The frame's depth as distance into the host's image, as o3d.depth_image
// makes the software renderer's.
@(test)
test_depth_image :: proc(t: ^testing.T) {
	g: GPU
	if !test_gpu(t, &g) {
		return
	}
	defer destroy(&g)
	err := depth_image_check(&g)
	testing.expectf(t, err == "", "%s", err)
}

// A host's GPU texture bound to an external texture draws as the same
// pixels uploaded; unbound, or bound for another table, as no map.
@(test)
test_external_texture :: proc(t: ^testing.T) {
	g: GPU
	if !test_gpu(t, &g) {
		return
	}
	defer destroy(&g)
	err := external_check(&g)
	testing.expectf(t, err == "", "%s", err)
}

// Jobs prepared together, then recorded in another order into one command
// buffer: the frames `render` draws, nothing made while recording.
@(test)
test_jobs_prepared_then_recorded :: proc(t: ^testing.T) {
	g: GPU
	if !test_gpu(t, &g) {
		return
	}
	defer destroy(&g)
	err := jobs_check(&g)
	testing.expectf(t, err == "", "%s", err)
}

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

// Frames of other sizes reuse the targets (they only grow): a small frame
// drawn after a large one is the same bytes as drawn on a fresh GPU, and
// neither it nor a second small one makes new targets.
@(test)
test_targets_grow_only :: proc(t: ^testing.T) {
	fresh, reused: GPU
	if !test_gpu(t, &fresh) {
		return
	}
	defer destroy(&fresh)
	if !test_gpu(t, &reused) {
		return
	}
	defer destroy(&reused)
	a: Parity_Assets
	defer parity_assets_destroy(&a)
	test_scene(&a.scene, &a.geos)
	frame :: proc(g: ^GPU, a: ^Parity_Assets, w, h: u32) -> []u8 {
		out := target(g, w, h)
		defer sdl.ReleaseGPUTexture(g.device, out)
		render(g, &a.scene, &a.geos, nil, out, w, h)
		px, _ := read(g, out, w, h)
		return px
	}
	want := frame(&fresh, &a, 160, 120)
	defer delete(want)
	delete(frame(&reused, &a, 320, 240))
	targets := reused.color
	got := frame(&reused, &a, 160, 120)
	defer delete(got)
	delete(frame(&reused, &a, 200, 100))
	testing.expect(t, reused.color == targets && reused.capacity == {320, 240}, "smaller frames reuse the targets")
	differ := 0
	for i in 0 ..< len(want) {
		differ += 1 if got[i] != want[i] else 0
	}
	testing.expectf(t, differ == 0, "%d bytes differ from a fresh GPU's frame", differ)
}
