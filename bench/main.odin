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

		EW, EH :: 64, 32
		env_px: [EW * EH * 4]u8
		for y in 0 ..< EH {
			v := 1 - (f32(y) + 0.5) / f32(EH)
			for x in 0 ..< EW {
				u := (f32(x) + 0.5) / f32(EW)
				dir := o3d.equirect_dir({u, v})
				sky := clamp(dir.y * 0.5 + 0.5, 0, 1)
				col := o3d.Vec3{0.55, 0.3, 0.15} * (1 - sky) + o3d.Vec3{0.2, 0.5, 0.95} * sky
				i := (y * EW + x) * 4
				env_px[i + 0] = u8(col.r * 255)
				env_px[i + 1] = u8(col.g * 255)
				env_px[i + 2] = u8(col.b * 255)
				env_px[i + 3] = 255
			}
		}
		env_id := o3d.texture_add(&textures, env_px[:], EW, EH)
		scene.environment = env_id
		scene.background = env_id

		run_case("small_env        (1T, 0-alloc)", &sw, &scene, &geos, &textures, nil, W, H, WARMUP, ITERS)
		run_case("small_env        (pool)", &sw, &scene, &geos, &textures, &pool, W, H, WARMUP, ITERS)
	}

	{
		Skinned_Bin :: struct #packed {
			pos:     [6][3]f32,
			norm:    [6][3]f32,
			joints:  [6][4]u16,
			weights: [6][4]f32,
			idx:     [12]u16,
			ibm:     [2]o3d.Mat4,
			times:   [2]f32,
			rot:     [2][4]f32,
		}
		s45 := math.sqrt_f32(0.5)
		ibm0 := o3d.Mat4(1)
		ibm1 := o3d.Mat4(1)
		ibm1[1, 3] = -1
		b := Skinned_Bin {
			pos     = {{-0.3, 0, 0}, {0.3, 0, 0}, {-0.3, 1, 0}, {0.3, 1, 0}, {-0.3, 2, 0}, {0.3, 2, 0}},
			norm    = {{0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}},
			joints  = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 1, 0, 0}, {0, 1, 0, 0}, {1, 0, 0, 0}, {1, 0, 0, 0}},
			weights = {{1, 0, 0, 0}, {1, 0, 0, 0}, {0.5, 0.5, 0, 0}, {0.5, 0.5, 0, 0}, {1, 0, 0, 0}, {1, 0, 0, 0}},
			idx     = {0, 1, 3, 0, 3, 2, 2, 3, 5, 2, 5, 4},
			ibm     = {ibm0, ibm1},
			times   = {0, 1},
			rot     = {{0, 0, 0, 1}, {0, 0, s45, s45}},
		}
		// Build a minimal GLB in memory (12-byte header + JSON chunk + BIN chunk).
		json_str := `{"asset":{"version":"2.0"},"buffers":[{"byteLength":480}],"bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":72},{"buffer":0,"byteOffset":72,"byteLength":72},{"buffer":0,"byteOffset":144,"byteLength":48},{"buffer":0,"byteOffset":192,"byteLength":96},{"buffer":0,"byteOffset":288,"byteLength":24},{"buffer":0,"byteOffset":312,"byteLength":128},{"buffer":0,"byteOffset":440,"byteLength":8},{"buffer":0,"byteOffset":448,"byteLength":32}],"accessors":[{"bufferView":0,"componentType":5126,"count":6,"type":"VEC3"},{"bufferView":1,"componentType":5126,"count":6,"type":"VEC3"},{"bufferView":2,"componentType":5123,"count":6,"type":"VEC4"},{"bufferView":3,"componentType":5126,"count":6,"type":"VEC4"},{"bufferView":4,"componentType":5123,"count":12,"type":"SCALAR"},{"bufferView":5,"componentType":5126,"count":2,"type":"MAT4"},{"bufferView":6,"componentType":5126,"count":2,"type":"SCALAR"},{"bufferView":7,"componentType":5126,"count":2,"type":"VEC4"}],"materials":[{"pbrMetallicRoughness":{"baseColorFactor":[0.9,0.4,0.2,1],"metallicFactor":0.1,"roughnessFactor":0.4},"doubleSided":true}],"meshes":[{"primitives":[{"attributes":{"POSITION":0,"NORMAL":1,"JOINTS_0":2,"WEIGHTS_0":3},"indices":4,"material":0}]}],"skins":[{"inverseBindMatrices":5,"joints":[1,2]}],"nodes":[{"mesh":0,"skin":0},{"children":[2]},{"translation":[0,1,0]}],"scenes":[{"nodes":[0,1]}],"scene":0,"animations":[{"name":"bend","samplers":[{"input":6,"output":7,"interpolation":"LINEAR"}],"channels":[{"sampler":0,"target":{"node":2,"path":"rotation"}}]}]}`
		jlen := (len(json_str) + 3) & ~int(3)
		blen := size_of(Skinned_Bin)
		total := 12 + 8 + jlen + 8 + blen
		glb := make([]u8, total)
		defer delete(glb)
		(^u32)(&glb[0])^ = 0x46546C67 // "glTF"
		(^u32)(&glb[4])^ = 2
		(^u32)(&glb[8])^ = u32(total)
		(^u32)(&glb[12])^ = u32(jlen)
		(^u32)(&glb[16])^ = 0x4E4F534A // "JSON"
		for i in 0 ..< jlen {
			glb[20 + i] = json_str[i] if i < len(json_str) else ' '
		}
		boff := 20 + jlen
		(^u32)(&glb[boff])^ = u32(blen)
		(^u32)(&glb[boff + 4])^ = 0x004E4942 // "BIN\0"
		mem.copy(&glb[boff + 8], &b, blen)

		geos: o3d.Geometries
		defer o3d.geometries_destroy(&geos)
		model, _ := o3d.gltf_parse(glb, &geos)
		defer o3d.model_destroy(&model)
		scene: o3d.Scene
		defer o3d.scene_destroy(&scene)

		samples_us := make([]f64, ITERS)
		defer delete(samples_us)
		pixels: []u8
		for w in 0 ..< WARMUP {
			o3d.scene_clear(&scene)
			scene.camera = {position = {0.4, 1.4, 3.8}, target = {-0.2, 0.8, 0}, fov = 45, near = 0.1, far = 30}
			append(&scene.lights, o3d.Light{kind = .Directional, color = {1, 0.95, 0.85}, intensity = 2.6, position = {2, 4, 3}})
			o3d.gltf_draw(&scene, &model, anim = 0, time = f32(w) * 0.1)
			pixels = o3d.software_render(&sw, &scene, &geos, nil, W, H, nil, nil)
		}
		for i in 0 ..< ITERS {
			t0 := time.tick_now()
			{
				no_allocations()
				o3d.scene_clear(&scene)
				scene.camera = {position = {0.4, 1.4, 3.8}, target = {-0.2, 0.8, 0}, fov = 45, near = 0.1, far = 30}
				append(&scene.lights, o3d.Light{kind = .Directional, color = {1, 0.95, 0.85}, intensity = 2.6, position = {2, 4, 3}})
				o3d.gltf_draw(&scene, &model, anim = 0, time = 0.65)
				pixels = o3d.software_render(&sw, &scene, &geos, nil, W, H, nil, nil)
			}
			samples_us[i] = time.duration_microseconds(time.tick_since(t0))
		}
		slice.sort(samples_us)
		fmt.printf(
			"%-28s  %dx%-4d  p50=%7.1f us  p95=%7.1f us  p99=%7.1f us  max=%7.1f us  digest=%016x\n",
			"small_skinned    (1T, 0-alloc)",
			W,
			H,
			samples_us[ITERS / 2],
			samples_us[(ITERS * 95) / 100],
			samples_us[(ITERS * 99) / 100],
			samples_us[ITERS - 1],
			fnv1a64(pixels),
		)
	}
}
