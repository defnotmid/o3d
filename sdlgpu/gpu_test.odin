package o3d_sdlgpu

import "core:fmt"
import "core:math"
import "core:testing"

import o3d ".."

// A scene with every kind of light and material the shading covers.
test_scene :: proc(scene: ^o3d.Scene, geos: ^o3d.Geometries) {
	o3d.scene_clear(scene)
	scene.camera = {position = {2.5, 2, 5}, target = {0, 0.5, 0}, fov = 45, near = 0.1, far = 50}
	append(&scene.lights, o3d.Light{kind = .Hemisphere, color = {0.8, 0.9, 1}, ground = {0.3, 0.2, 0.25}, intensity = 1.5})
	append(&scene.lights, o3d.Light{kind = .Directional, color = {1, 0.95, 0.9}, intensity = 2.5, position = {3, 5, 2}})
	append(&scene.lights, o3d.Light{kind = .Point, color = {0.3, 0.6, 1}, intensity = 8, position = {-1.5, 1, 1.5}, distance = 6, decay = 2})
	append(&scene.lights, o3d.Light{kind = .Ambient, color = {1, 1, 1}, intensity = 0.2})
	red := o3d.MATERIAL_DEFAULT
	red.color, red.roughness = {1, 0.35, 0.3}, 0.35
	metal := o3d.MATERIAL_DEFAULT
	metal.color, metal.metalness, metal.roughness = {0.9, 0.8, 0.5}, 1, 0.3
	floor := o3d.MATERIAL_DEFAULT
	floor.color, floor.roughness = {0.4, 0.42, 0.5}, 0.9
	glass := o3d.MATERIAL_DEFAULT
	glass.color, glass.opacity, glass.flags, glass.side = {0.4, 0.8, 1}, 0.45, {.Transparent}, .Double
	glow := o3d.MATERIAL_DEFAULT
	glow.color, glow.emissive, glow.emissive_intensity, glow.flags = {0.1, 0.1, 0.1}, {1, 0.6, 0.1}, 1.5, {.Flat_Shading}
	append(&scene.draws, o3d.Draw{o3d.plane(geos, 6, 6), o3d.compose({0, 0, 0}, {-math.PI / 2, 0, 0}), floor})
	append(&scene.draws, o3d.Draw{o3d.sphere(geos, 0.7, 48, 24), o3d.compose({-0.6, 0.7, 0}), red})
	append(&scene.draws, o3d.Draw{o3d.box(geos, 0.8, 0.8, 0.8), o3d.compose({1, 0.4, -0.4}, {0, 0.6, 0}), metal})
	append(&scene.draws, o3d.Draw{o3d.torus(geos, 0.45, 0.15, 12, 32), o3d.compose({0.4, 1.4, 0.6}, {0.5, 0.3, 0}), glow})
	append(&scene.draws, o3d.Draw{o3d.plane(geos, 1.4, 1.2), o3d.compose({0.2, 0.6, 1.4}, {0, 0.3, 0}), glass})
}

// The GPU and the software renderer draw the same frame: close everywhere
// (anti-aliasing and float order differ at the edges only).
@(test)
test_gpu_matches_software :: proc(t: ^testing.T) {
	g: GPU
	if init(&g) != nil {
		fmt.println("no GPU device: skipped")
		return
	}
	defer destroy(&g)
	geos: o3d.Geometries
	defer o3d.geometries_destroy(&geos)
	scene: o3d.Scene
	defer o3d.scene_destroy(&scene)
	test_scene(&scene, &geos)
	W, H :: 160, 120
	out := target(&g, W, H)
	testing.expect(t, render(&g, &scene, &geos, out, W, H) == nil)
	gpu, _ := read(&g, out, W, H)
	defer delete(gpu)
	sw: o3d.Software
	defer o3d.software_destroy(&sw)
	cpu := o3d.software_render(&sw, &scene, &geos, W, H)
	// PSNR over the colour channels.
	sum: f64
	for i in 0 ..< len(cpu) {
		d := f64(cpu[i]) - f64(gpu[i])
		sum += d * d
	}
	mse := sum / f64(len(cpu))
	psnr := 10 * math.log10(255 * 255 / max(mse, 1e-9))
	testing.expectf(t, psnr > 32, "PSNR %.1f dB", psnr)
	// Something was drawn.
	opaque := 0
	for i in 0 ..< W * H {
		opaque += 1 if gpu[i * 4 + 3] == 255 else 0
	}
	testing.expectf(t, opaque > W * H / 3, "%d opaque pixels", opaque)
}
