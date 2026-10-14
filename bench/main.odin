package o3d_bench

import "base:runtime"
import "core:fmt"
import "core:math"
import "core:mem"
import "core:slice"
import "core:thread"
import "core:time"

import o3d ".."

Allocation_Guard :: struct {
	saved_allocator:      runtime.Allocator,
	saved_temp_allocator: runtime.Allocator,
}

@(deferred_out = allocation_guard_end)
no_allocations :: proc() -> Allocation_Guard {
	guard := Allocation_Guard {
		saved_allocator      = context.allocator,
		saved_temp_allocator = context.temp_allocator,
	}
	panic_alloc := mem.panic_allocator()
	context.allocator = panic_alloc
	context.temp_allocator = panic_alloc
	return guard
}

allocation_guard_end :: proc(guard: Allocation_Guard) {
	context.allocator = guard.saved_allocator
	context.temp_allocator = guard.saved_temp_allocator
}

fnv1a64 :: proc(bytes: []u8) -> u64 {
	h: u64 = 0xcbf29ce484222325
	for b in bytes {
		h = (h ~ u64(b)) * 0x100000001b3
	}
	return h
}

build_small_scene :: proc(scene: ^o3d.Scene, geos: ^o3d.Geometries, textures: ^o3d.Textures = nil) {
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

	if textures != nil {
		TW, TH :: 64, 64
		color_px: [TW * TH * 4]u8
		normal_px: [TW * TH * 4]u8
		rough_px: [TW * TH * 4]u8
		emissive_px: [TW * TH * 4]u8
		for y in 0 ..< TH {
			for x in 0 ..< TW {
				i := (y * TW + x) * 4
				u := f32(x) / f32(TW - 1)
				v := f32(y) / f32(TH - 1)
				color_px[i + 0] = u8(clamp(0.25 + 0.65 * (0.5 + 0.5 * math.cos(u * math.TAU * 2)), 0, 1) * 255)
				color_px[i + 1] = u8(clamp(0.25 + 0.65 * (0.5 + 0.5 * math.sin(v * math.TAU * 2)), 0, 1) * 255)
				color_px[i + 2] = u8(clamp(0.35 + 0.50 * u * v, 0, 1) * 255)
				color_px[i + 3] = 255

				nx := 0.25 * math.sin(u * math.TAU * 2)
				ny := 0.25 * math.cos(v * math.TAU * 2)
				nz := math.sqrt(max(1 - nx * nx - ny * ny, 0))
				normal_px[i + 0] = u8(clamp((nx * 0.5 + 0.5) * 255, 0, 255))
				normal_px[i + 1] = u8(clamp((ny * 0.5 + 0.5) * 255, 0, 255))
				normal_px[i + 2] = u8(clamp((nz * 0.5 + 0.5) * 255, 0, 255))
				normal_px[i + 3] = 255

				rough := u8(clamp((0.35 + 0.55 * (0.5 + 0.5 * math.sin((u + v) * math.TAU))) * 255, 0, 255))
				rough_px[i + 0] = rough
				rough_px[i + 1] = rough
				rough_px[i + 2] = rough
				rough_px[i + 3] = 255

				emissive_px[i + 0] = u8(clamp(0.4 * u * 255, 0, 255))
				emissive_px[i + 1] = u8(clamp(0.2 * v * 255, 0, 255))
				emissive_px[i + 2] = u8(clamp(0.5 * (1 - u) * 255, 0, 255))
				emissive_px[i + 3] = 255
			}
		}
		color_id := o3d.texture_add(textures, color_px[:], TW, TH)
		normal_id := o3d.texture_add(textures, normal_px[:], TW, TH)
		rough_id := o3d.texture_add(textures, rough_px[:], TW, TH)
		emissive_id := o3d.texture_add(textures, emissive_px[:], TW, TH)

		floor.color_map = color_id
		floor.roughness_map = rough_id
		red.color_map = color_id
		red.normal_map = normal_id
		red.roughness_map = rough_id
		metal.normal_map = normal_id
		metal.emissive = {1, 1, 1}
		metal.emissive_intensity = 0.6
		metal.emissive_map = emissive_id
	}

	append(&scene.draws, o3d.Draw{o3d.plane(geos, 6, 6), o3d.compose({0, 0, 0}, {-math.PI / 2, 0, 0}), floor})
	append(&scene.draws, o3d.Draw{o3d.sphere(geos, 0.7, 48, 24), o3d.compose({-0.6, 0.7, 0}), red})
	append(&scene.draws, o3d.Draw{o3d.box(geos, 0.8, 0.8, 0.8), o3d.compose({1, 0.4, -0.4}, {0, 0.6, 0}), metal})
	append(&scene.draws, o3d.Draw{o3d.torus(geos, 0.45, 0.15, 12, 32), o3d.compose({0.4, 1.4, 0.6}, {0.5, 0.3, 0}), glow})
	append(&scene.draws, o3d.Draw{o3d.plane(geos, 1.4, 1.2), o3d.compose({0.2, 0.6, 1.4}, {0, 0.3, 0}), glass})
}

run_case :: proc(
	label: string,
	sw: ^o3d.Software,
	scene: ^o3d.Scene,
	geos: ^o3d.Geometries,
	textures: ^o3d.Textures,
	pool: ^thread.Pool,
	width, height: int,
	warmup, iters: int,
) {
	samples_us := make([]f64, iters)
	defer delete(samples_us)

	// Warmup allocates all internal scratch tables on Software and Scene.
	pixels: []u8
	for _ in 0 ..< warmup {
		pixels = o3d.software_render(sw, scene, geos, nil, width, height, pool, textures)
	}
	digest := fnv1a64(pixels)

	// Measured iterations enforce zero heap/temp allocations on single-threaded runs.
	for i in 0 ..< iters {
		t0 := time.tick_now()
		if pool == nil {
			no_allocations()
			pixels = o3d.software_render(sw, scene, geos, nil, width, height, nil, textures)
		} else {
			pixels = o3d.software_render(sw, scene, geos, nil, width, height, pool, textures)
		}
		dt := time.tick_since(t0)
		samples_us[i] = time.duration_microseconds(dt)
	}

	slice.sort(samples_us)
	p50 := samples_us[iters / 2]
	p95 := samples_us[(iters * 95) / 100]
	p99 := samples_us[(iters * 99) / 100]
	max_us := samples_us[iters - 1]
	fmt.printf(
		"%-28s  %dx%-4d  p50=%7.1f us  p95=%7.1f us  p99=%7.1f us  max=%7.1f us  digest=%016x\n",
		label,
		width,
		height,
		p50,
		p95,
		p99,
		max_us,
		digest,
	)
}

main :: proc() {
	W, H :: 320, 240
	WARMUP :: 8
	ITERS :: 64

	sw: o3d.Software
	defer o3d.software_destroy(&sw)

	pool: thread.Pool
	thread.pool_init(&pool, context.allocator, 4)
	defer thread.pool_destroy(&pool)
	thread.pool_start(&pool)
	defer thread.pool_join(&pool)

	{
		geos: o3d.Geometries
		defer o3d.geometries_destroy(&geos)
		scene: o3d.Scene
		defer o3d.scene_destroy(&scene)
		build_small_scene(&scene, &geos, nil)

		run_case("small_untextured (1T, 0-alloc)", &sw, &scene, &geos, nil, nil, W, H, WARMUP, ITERS)
		run_case("small_untextured (pool)", &sw, &scene, &geos, nil, &pool, W, H, WARMUP, ITERS)
	}

	{
		geos: o3d.Geometries
		defer o3d.geometries_destroy(&geos)
		textures: o3d.Textures
		defer o3d.textures_destroy(&textures)
		scene: o3d.Scene
		defer o3d.scene_destroy(&scene)
		build_small_scene(&scene, &geos, &textures)

		run_case("small_textured   (1T, 0-alloc)", &sw, &scene, &geos, &textures, nil, W, H, WARMUP, ITERS)
		run_case("small_textured   (pool)", &sw, &scene, &geos, &textures, &pool, W, H, WARMUP, ITERS)
	}
}
