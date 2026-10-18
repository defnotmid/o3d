// Draws every GPU-vs-software parity case (o3d/sdlgpu PARITY_CASES) with
// both renderers on the main thread and compares them by PSNR, then the
// external texture check (sdlgpu external_check): the check
// for backends `odin test` cannot reach, such as Metal, whose SDL driver
// needs the main thread. Exits 1 if a case fails or no GPU device opens.
//
//     odin run tools/gpu_check -collection:deps=../.. [-- out_dir]
//
// With `out_dir`, each case's frames are saved as <case>.gpu.png and
// <case>.software.png.
package o3d_gpu_check

import "core:fmt"
import "core:os"
import "core:strings"
import stbi "vendor:stb/image"

import gpu "../../sdlgpu"

main :: proc() {
	dir := os.args[1] if len(os.args) > 1 else ""
	g: gpu.GPU
	if err := gpu.init(&g); err != nil {
		fmt.eprintln("gpu_check: no GPU device:", err)
		os.exit(1)
	}
	defer gpu.destroy(&g)
	failed := 0
	for c in gpu.PARITY_CASES {
		psnr, gpu_px, cpu_px, err := gpu.parity_check(&g, c)
		defer delete(gpu_px)
		defer delete(cpu_px)
		switch {
		case err != "":
			fmt.printf("%-12s FAIL  %s\n", c.name, err)
			failed += 1
		case psnr <= gpu.PARITY_MIN_PSNR:
			fmt.printf("%-12s FAIL  PSNR %.1f dB (needs more than %d)\n", c.name, psnr, gpu.PARITY_MIN_PSNR)
			failed += 1
		case:
			fmt.printf("%-12s ok    PSNR %.1f dB\n", c.name, psnr)
		}
		if dir != "" && err == "" {
			for px, i in ([2][]u8{gpu_px, cpu_px}) {
				path := strings.clone_to_cstring(fmt.tprintf("%s/%s.%s.png", dir, c.name, "gpu" if i == 0 else "software"), context.temp_allocator)
				stbi.write_png(path, i32(c.width), i32(c.height), 4, raw_data(px), i32(c.width * 4))
			}
		}
	}
	if err := gpu.external_check(&g); err != "" {
		fmt.printf("%-12s FAIL  %s\n", "external", err)
		failed += 1
	} else {
		fmt.printf("%-12s ok    bound = uploaded, unbound = no map\n", "external")
	}
	fmt.printf("%d of %d cases match\n", len(gpu.PARITY_CASES) + 1 - failed, len(gpu.PARITY_CASES) + 1)
	if failed > 0 {
		os.exit(1)
	}
}
