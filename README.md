# o3d

3D for Odin in three.js's vocabulary, for rendering video frames: every
frame is described from scratch (a camera, lights, geometry with materials),
so a frame is a pure function of whatever drives it. Drawn on the GPU
through SDL_GPU (Vulkan, Metal), with a software renderer that draws the
same pixels on the CPU and is the reference the GPU is tested against.
MIT licensed ([LICENSE](LICENSE)); what it carries (oksl, cgltf,
stb_image, SDL 3, three.js's models, Odin's packages) is in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Conventions (three.js's)

What is known about three.js carries over:

- Right-handed and Y up; the camera looks down -Z.
- Rotations are Euler angles in radians, applied in X, Y, Z order; a
  transform is position, rotation and scale (`compose`), like Object3D.
- The camera's field of view is vertical, in degrees (PerspectiveCamera);
  an orthographic camera takes the view's height.
- Colours are sRGB; lighting is linear; the output is sRGB with no tone
  mapping.
- Materials are MeshStandardMaterial (`color`, `metalness`, `roughness`,
  `emissive`, `emissive_intensity`, `ao_map_intensity`, `opacity`,
  `alpha_cutoff`, `side`, texture maps `color_map`, `normal_map`,
  `roughness_map` (green), `metalness_map` (blue; glTF packs both in one
  texture), `emissive_map`, `ao_map` (red, applied to indirect light only as
  three.js's `aoMap` does), per-slot UV transforms `*_transform` (`offset`,
  `rotation`, `scale`, `tex_coord` 0 or 1 selecting `uv` or `uv1`), and the
  booleans as a bit set: `flags = {.Transparent, .Alpha_Mask, .Flat_Shading}`)
  or MeshBasicMaterial (`.Basic`), or ShaderMaterial: your own vertex and
  fragment shaders in GLSL ES (three.js's GLSL 1 style with `attribute`,
  `varying` and `gl_FragColor`, or GLSL 3 with `glsl3 = true`), with
  three.js's built-in uniforms (`modelMatrix`, `modelViewMatrix`,
  `projectionMatrix`, `viewMatrix`, `normalMatrix`, `cameraPosition`,
  `isOrthographic`) and attributes (`position`, `normal`, `uv`), compiled by
  [oksl](https://github.com/defnotmid/oksl) on both renderers. The output is written as the shader
  leaves it; `.Transparent` blends it, `side` culls as for any material.
  `projectionMatrix` maps depth to 0..1 (SDL_GPU's clip space) rather than
  WebGL's -1..1, which matters only to shaders that build their own. On a
  morphed or skinned mesh, `position` and `normal` arrive already morphed and
  posed by the skin's joints.
- Lights are AmbientLight, HemisphereLight, DirectionalLight (from
  `position` toward `target`) and PointLight (`distance`, `decay`), in
  three.js's units: a white directional light of intensity 1 lights a white
  surface facing it to 1/pi of white, so scenes use intensities around 2 to 3.
- Shadows are DirectionalLightShadow's with PCF: a directional light with
  `cast_shadow` (the first) and its `shadow` settings (`map_size`,
  `height` of its square orthographic view, `near`, `far`, `bias`,
  `normal_bias`; zero takes three.js's 512, 10, 0.5, 500), draws with
  `.Cast_Shadow` and `.Receive_Shadow` in their material's flags (three.js
  has them on the mesh; a draw's material is its own). Casters draw their
  other side (three.js's default shadowSide) and morphed/skinned geometry,
  discarding shadow texels below `alpha_cutoff` when `.Alpha_Mask` is set
  (cut-out shadows); ShaderMaterials cast (including morphed and skinned
  meshes) but do not receive.
- Geometry takes the three.js parameters: `box(w, h, d, segments...)`,
  `sphere(radius, widthSegments, heightSegments)`, `plane`, `cylinder`
  (radiusTop 0 is a cone), `torus`. The same parameters give the same id,
  so a frame may ask for `box(1, 1, 1)` every time; `primitive(key)` makes
  one from its `Geometry_Key` alone. Geometry is kept (never freed), so a
  shape changes over time through the draw's transform, as three.js
  animates `mesh.scale`: `sphere(geos, 1)` scaled by `compose`, not
  `sphere(geos, radius(t))`, which makes a new mesh every frame it differs.
  `geometries_made` lets a host check that it stops growing.

The output is premultiplied RGBA, transparent where nothing was drawn, to
composite over anything.

## Use

```odin
geos: o3d.Geometries                       // kept: geometry is made once
textures: o3d.Textures                     // kept: textures and mipmaps are made once
albedo, _ := o3d.texture_load(&textures, "brick.png")
scene: o3d.Scene                           // filled every frame
o3d.scene_clear(&scene)
scene.camera = {position = {0, 2, 6}, target = {0, 0.5, 0}, fov = 40, near = 0.1, far = 100}
append(&scene.lights, o3d.Light{kind = .Hemisphere, color = {0.8, 0.9, 1}, ground = {0.2, 0.15, 0.2}, intensity = 1.5})
append(&scene.lights, o3d.Light{kind = .Directional, color = {1, 1, 1}, intensity = 2.5, position = {3, 5, 2},
	cast_shadow = true, shadow = {map_size = 1024, bias = -0.0005}})
m := o3d.MATERIAL_DEFAULT
m.color, m.roughness, m.color_map = {1, 0.35, 0.3}, 0.4, albedo
m.flags = {.Cast_Shadow, .Receive_Shadow}
append(&scene.draws, o3d.Draw{o3d.box(&geos, 1, 1, 1), o3d.compose({0, 0.5, 0}, {0, t, 0}), m})

// glTF 2.0 / GLB: loaded once into flat tables, drawn per frame at time t
duck, err := o3d.gltf_load("duck.glb", &geos, &textures)   // err: .None, or why (GLTF_ERROR_TEXT[err])
o3d.gltf_draw(&scene, &duck, o3d.compose({-1.5, 0, 0}), anim = 0, time = t)
// Morph target overrides by target name or 0-based index (applied after animation):
// o3d.gltf_draw(&scene, &face, anim = 0, time = t, morphs = {{name = "Smile", weight = 0.8}, {index = 1, weight = 0.5}})
// From several threads, each with its own pose (the model is then only read):
// o3d.gltf_draw(&scene, &duck, anim = 0, time = t, pose = &thread_pose)

// ShaderMaterial: compiled once (same sources, same id), uniforms per frame
shaders: o3d.Shaders                       // kept, like geometry
wave, errors := o3d.shader(&shaders, vertex_glsl, fragment_glsl)   // errors: "Fragment:3:7: ..."
w := o3d.MATERIAL_DEFAULT
w.shader = wave
w.uniforms = o3d.scene_uniforms(&scene, {"uTime", {t}}, {"uColor", {0.3, 0.8, 0.4}})
append(&scene.draws, o3d.Draw{o3d.sphere(&geos, 1), o3d.compose(), w})

// GPU (o3d/sdlgpu): borrows the host's SDL_GPU device, or makes a headless one
g: o3d_sdlgpu.GPU
o3d_sdlgpu.init(&g, device)
out := o3d_sdlgpu.target(&g, 1920, 1080)
o3d_sdlgpu.render(&g, &scene, &geos, &shaders, out, 1920, 1080, cmd, &textures)   // or nil cmd: submit and wait

// CPU: the same pixels, rows banded on the caller's thread pool
sw: o3d.Software
pixels := o3d.software_render(&sw, &scene, &geos, &shaders, 1920, 1080, pool, &textures)
```

o3d imports oksl as `deps:oksl`: clone
[oksl](https://github.com/defnotmid/oksl) next to o3d, and builds pass
`-collection:deps=..` (the directory holding both). `vendor:cgltf` needs its library built once
per Odin install (`$(odin root)/vendor/cgltf/src/build_cgltf.sh`).

- `odin test . -collection:deps=..` and `odin test sdlgpu
  -collection:deps=..`: the CPU and GPU tests. A GPU test fails without a
  device unless `O3D_NO_GPU=1` (CI without a GPU; macOS, below).
- `odin run tools/gpu_check -collection:deps=.. [-- out_dir]`: every
  GPU-vs-software parity case (`PARITY_CASES`) on the main thread, by PSNR,
  optionally saving both frames. On macOS this is the GPU check: SDL's
  Metal driver needs the main thread, which `odin test` does not give.
- `odin run bench -collection:deps=.. -o:speed`: the zero-allocation
  software benchmark; `odin run demo -collection:deps=.. -- out_dir`: a
  test scene both ways to PNGs.

## Design

- **Data:** the scene is flat tables (lights, draws: geometry id, model
  matrix, material, plus optional joint and morph-weight ranges) refilled
  each frame, the draws a structure of arrays (`#soa`) since the per-frame
  passes read a column or two; geometry is the only retained state: one
  array of interleaved position / normal / uv / uv1 vertices (40 bytes), one
  of 32-bit indices, a separate array of `Skin_Vertex` (4 joint indices and
  normalized weights) for skinned meshes, and a separate array of
  `Morph_Delta` (position and normal deltas, 32 bytes) for morphed meshes so
  meshes without skinning or morph targets pay nothing. The GPU mirrors the
  arrays in vertex, index, and storage buffers, uploading what is new at the
  start of a frame.
- **glTF (`gltf.odin`):** `gltf_load` and `gltf_parse` (`vendor:cgltf`)
  unpack meshes (`POSITION`, `NORMAL`, `TEXCOORD_0`, `TEXCOORD_1`,
  `JOINTS_0`, `WEIGHTS_0` with weights normalized, including normalized and
  non-normalized integer attributes from `KHR_mesh_quantization`, plus morph
  target `POSITION` and `NORMAL` deltas — `TANGENT` deltas are ignored as
  tangents are derived from screen derivatives) into `Geometries`, mesh
  default `weights` and `extras.targetNames` into `Model`, images into
  `Textures`, metallic-roughness materials (`alphaMode`: `OPAQUE`, `MASK`
  with `alphaCutoff` default 0.5, `BLEND`; `occlusionTexture` red channel +
  `strength`; `KHR_materials_unlit`; `KHR_materials_emissive_strength`;
  `KHR_texture_transform` `offset`, `rotation`, `scale` and `texCoord` 0 or
  1 per texture reference) into `Material`, the node hierarchy into a
  topologically sorted `#soa` array (`parent < i`), skins (`joints` and
  `inverseBindMatrices`, up to `MAX_JOINTS = 64`) into flat arrays, and
  keyframe animations (`STEP`, `LINEAR`, `CUBICSPLINE`, including morph
  `"weights"` channels) into flat time/value arrays. `gltf_draw` evaluates
  the hierarchy, animation, morph weights (with optional `Morph_Weight`
  overrides by target name or 0-based index applied after animation), and
  joint matrices (`inv_mesh * world[joint] * ibm`) at time `t` with zero
  heap allocations on warm frames, posing into the caller's `Model_Pose`
  (one per thread lets threads pose one model at once) or the model's own.
  Morphs (up to `MAX_MORPH_TARGETS = 8` active non-zero targets per draw) are
  applied before skinning in the per-chunk vertex stage on the software
  renderer and in `vs_main` / `vs_skinned` (`shaders/o3d.slang`, reading the
  `Morph_Delta` storage buffer via `SV_VertexID`) on the GPU, including the
  shadow pass and before a `ShaderMaterial`'s vertex stage. A failed load
  returns a `Gltf_Error` with `GLTF_ERROR_TEXT` for messages. If
  `extensionsRequired` names an extension o3d does not implement, loading
  fails with `Gltf_Error.Unsupported_Extension` and records the extension
  name in `Model.unsupported_extension`. Optional extensions listed only in
  `extensionsUsed` (`KHR_materials_ior`, `KHR_materials_transmission`,
  `KHR_materials_volume`, `KHR_materials_clearcoat`, `KHR_materials_sheen`,
  `KHR_materials_specular`, `KHR_materials_iridescence`,
  `KHR_materials_anisotropy`, `KHR_lights_punctual`, `EXT_texture_webp`, …)
  are ignored, as the spec allows. Remaining unsupported required extensions
  (rejected with `.Unsupported_Extension`): `KHR_draco_mesh_compression`,
  `EXT_meshopt_compression`, `KHR_texture_basisu`, `EXT_texture_webp`, and
  any other extension outside `KHR_materials_unlit`,
  `KHR_materials_emissive_strength`, `KHR_texture_transform`, and
  `KHR_mesh_quantization`. Also not imported: glTF cameras and lights.
- **Tables:** primitives are made by `PRIMITIVES`, indexed by `Primitive`.
  Each frame, `shading_lights` and `shading_materials` turn the lights and
  the draws' materials into their shading form (linear colours, the
  opacity drawn, 2x3 UV transform matrices in o3d UV space, UV set bits,
  `Shading_Flags`) once; both renderers read those, and the GPU's uniforms
  are them, packed. Boolean sets are bit sets (`Material_Flags`,
  `Shading_Flags`, `Texture_Slots`, a pixel's covered samples).
- **Shader contract:** `Light_Kind`'s values and `Shading_Flag`'s bits are
  the shader's `LIGHT_*` and `FLAG_*` constants; a test reads
  `shaders/o3d.slang` and checks them.
- **Shading** is one model in two places, `shade` in scene.odin and
  `shaders/o3d.slang`: Lambert diffuse plus GGX specular with Smith
  correlated visibility and Schlick Fresnel, plus equirectangular
  environment lighting (`scene.environment`, `scene.environment_intensity`)
  and background (`scene.background`, `scene.background_intensity`,
  `scene.background_blurriness`), as three.js's MeshStandardMaterial
  computes it, with `ao_map` (`1 + (r - 1) * ao_map_intensity`) multiplying
  indirect light only (ambient, hemisphere, environment irradiance and
  radiance). When a texture is used as `scene.environment` (or blurred
  `scene.background`), the texture has derived maps, as it has mipmaps: a
  GGX-prefiltered radiance mip chain (roughness 0..1) and a
  cosine-convolved diffuse irradiance map in `Textures`, sampled alike by
  both renderers. `environment_prefilter(textures, id)` makes them; it is
  heavy, so a host calls it at load (a render that meets a texture without
  them makes them then, in frame time); `texture_environment` looks them up.
  Changing one means changing the other; the tests compare the two
  renderers.
- **GPU targets:** the colour (multisampled), resolve and depth targets
  only grow, to the largest frame drawn; a frame renders into their
  top-left width x height (viewport and scissor) and is copied to the
  caller's texture, so views of several sizes, or one whose size animates,
  make no targets per frame.
- **Anti-aliasing:** 4x MSAA on the GPU; the software renderer keeps four
  samples per pixel at the same positions, shades once per pixel per
  triangle, and resolves the same way.
- **Software phases,** each spread over the caller's pool: chunks of the
  drawing order (up to 8192 triangles, so one large mesh splits too) are
  transformed, clipped and counted per band of 32 rows; after a prefix sum
  each chunk lists its triangles in the bands, so a band's list keeps the
  drawing order; then each band clears its rows, draws its list and
  resolves its pixels. A frame is the same bytes with or without a pool
  (tested). Triangles are a structure of arrays, as counting and listing
  read only their row spans. Vertices stay interleaved: the GPU reads them
  as one buffer and the rasterizer gathers whole vertices by index.
- **Shadows** (shadow.odin) are drawn first: the casters' depth from the
  light (`shadow_casters`, culled to its view), one sample per texel
  centre, nearest kept (discarding texels below `alpha_cutoff` for
  `.Alpha_Mask` casters with a `color_map`); the software renderer runs the
  same phases in a depth-only mode at the map's size, the GPU renders into
  an R32_FLOAT colour target (sampleable on every backend) with a depth
  buffer. A receiver's pixel scales the shadow light's direct light by
  `shadow_factor`: a 3x3 grid of bilinear depth comparisons (a 4x4 block
  of texels weighed 1 - f, 1, 1, f per axis), edges repeated, outside the
  light's view lit. The shader has the same arithmetic (`shadow_factor` in
  `shaders/o3d.slang`); `FLAG_RECEIVE_SHADOW` and `FLAG_ALPHA_MASK` join the
  shader contract.
- **Order:** draws whose bounding sphere (the geometry's radius plus
  `(sum |w_t|) * morph_radius`, scaled with the draw, or covering the posed
  joints for a skinned draw) is outside the view, or whose shader failed,
  are skipped; then opaque and alpha-masked draws as listed, then blended
  ones (transparent with opacity under 1, or a transparent ShaderMaterial)
  farthest first, without writing depth (`draw_order`, shared by both
  renderers).
- **Shaders:** the standard material's are Slang compiled offline
  (`shaders/build.sh`) to committed SPIR-V and MSL, as ovg's are.
  ShaderMaterials are compiled at run time by oksl, which is plain Odin in
  the binary (no external toolchain), to SPIR-V and MSL and to programs
  its CPU interpreter runs. Their table (`Shaders`) keeps each compiled
  pair by id with what it reads, found once: the built-ins' block offsets
  per stage, the attributes' locations, the varyings' floats and the flat
  ones as a list. `BUILTINS` and `VERTEX_ATTRIBUTES` make three.js's
  prelude for each stage and say what to fill; `shader_blocks` writes every
  custom draw's uniform bytes once a frame, which the GPU pushes and the
  software renderer runs on. A shader that fails is kept with its errors
  and its draws left out by both renderers.
- **Software ShaderMaterials:** a chunk runs its draw's vertex stage once
  over the vertices it uses (the standard material's too: world position
  and normal), and a triangle's corners carry those floats, divided by w
  except flat ones. A band shades a run of one custom draw's triangles as
  2x2 quads (helper lanes for derivatives) in one oksl call, then writes
  them through the same coverage, depth and blending as the standard
  material, in triangle order.
- Errors are values (`Error`); no threads of its own: the software
  renderer runs its phases on the caller's `thread.Pool`.

## Next

Integrated in onimite (`view3d{}`, v0.0.3). What is still missing, found by
using it, most important first.

For models people download:

1. **Compressed models and remaining glTF extensions:** decoders for Draco
   (`KHR_draco_mesh_compression`), meshopt (`EXT_meshopt_compression`),
   KTX2 / Basis textures (`KHR_texture_basisu`) and WebP textures
   (`EXT_texture_webp`), which today fail cleanly with
   `Gltf_Error.Unsupported_Extension` when required; and optional PBR
   extensions (`KHR_materials_transmission`, `KHR_materials_ior`,
   `KHR_materials_volume`, `KHR_materials_clearcoat`, `KHR_materials_sheen`,
   `KHR_materials_specular`, `KHR_lights_punctual`), which today are ignored
   when listed in `extensionsUsed`.

For typical three.js scenes:

2. **Tone mapping and exposure** (three.js's `ACESFilmicToneMapping` /
   `AgXToneMapping`, `toneMappingExposure`): most three.js scenes use one,
   and lit models look flat or blown out without it.
3. **SpotLight**, and shadows from more than the first directional light
   (point and spot light shadows).
4. **Fog** (`Fog`, `FogExp2`).

For onimite:

5. **Textures that change every frame:** a video clip or a group's 2D
   drawing mapped onto a surface. Today a texture is added once
   (`texture_add`); this needs an update path the host fills per frame
   (on the GPU: a texture the host renders into).
6. **ShaderMaterial textures** (small here): `uniform sampler2D` bound to
   `Textures`; `texelFetch` / `textureSize` and CPU implicit-LOD
   `texture()` wait on oksl (deferred there).
7. **Instancing, lines and points:** `InstancedMesh` for many copies of one
   mesh, `LineSegments` and `Points` for data visualisation.

Metal: v0.0.1's standard shader did not compile there (a texel `Load` on
a combined texture-sampler, which Slang emits wrongly for MSL; the shadow
map is now sampled at texel centres). Run `tools/gpu_check` on macOS to
confirm every case before relying on it.
