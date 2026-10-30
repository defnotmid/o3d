// Renders the test scene with the GPU and the software renderer to PNGs:
//
//     odin run demo -collection:deps=.. -- out_dir   (deps: the directory holding o3d, oksl, ocurves and oicut)
package o3d_demo

import "core:fmt"
import "core:os"
import image "deps:oicut/image"

import o3d ".."
import gpu "../sdlgpu"

main :: proc() {
	dir := os.args[1] if len(os.args) > 1 else "."
	W, H :: 960, 540
	geos: o3d.Geometries
	defer o3d.geometries_destroy(&geos)
	scene: o3d.Scene
	defer o3d.scene_destroy(&scene)
	gpu.test_scene(&scene, &geos)

	sw: o3d.Software
	defer o3d.software_destroy(&sw)
	cpu := o3d.software_render(&sw, &scene, &geos, nil, W, H)
	save(dir, "software.png", cpu, W, H)

	g: gpu.GPU
	if gpu.init(&g) != nil {
		fmt.eprintln("no GPU device")
		return
	}
	defer gpu.destroy(&g)
	out := gpu.target(&g, W, H)
	gpu.render(&g, &scene, &geos, nil, out, W, H)
	px, _ := gpu.read(&g, out, W, H)
	save(dir, "gpu.png", px, W, H)
}

save :: proc(dir, name: string, px: []u8, w, h: int) {
	path := fmt.tprintf("%s/%s", dir, name)
	image.write_png(path, px, w, h)
	fmt.println(path)
}
