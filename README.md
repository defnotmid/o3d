# o3d

<img src="assets/logo.svg" width="128" alt="o3d: Oxa, a pixel-art axolotl whose gills are the X, Y and Z axes">

3D for Odin in three.js's vocabulary, for rendering video frames: every
frame is described from scratch (a camera, lights, geometry with materials),
so a frame is a pure function of whatever drives it. Drawn on the GPU
through SDL_GPU (Vulkan, Metal), with a software renderer that draws the
same pixels on the CPU and is the reference the GPU is tested against.

The mascot, Oxa, is drawn by `tools/mascot/oxa.py`: `assets/logo.svg`
keeps its gills, eyes and mouths as layers a script switches by id, each
colour one traced path, plus a hidden `silhouette` (all of Oxa as one
outline with its hole) to extrude, morph or collide.

MIT licensed ([LICENSE](LICENSE)); what it carries (oksl, ovg's curves,
odecs' image package, SDL 3, three.js's models, Odin's
packages) is in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Conventions (three.js's)

What is known about three.js carries over:

- Right-handed and Y up; the camera looks down -Z.
- Rotations are Euler angles in radians, applied in X, Y, Z order; a
  transform is position, rotation and scale (`compose`), like Object3D.
- The camera's field of view is vertical, in degrees (PerspectiveCamera);
  an orthographic camera takes the view's height.
- Colours are sRGB; lighting is linear; the output is tone-mapped
  (`scene.tone_mapping`: `.None` default, `.Linear`, `.Reinhard`, `.Cineon`,
  `.ACES_Filmic`, `.AgX`, `.Neutral`, scaled by `scene.tone_mapping_exposure`,
  default 1), blended toward `scene.fog` (`.Linear` between `near` and `far`,
  or `.Exp2` with `density`, by Euclidean camera distance on drawn fragments),
  and encoded to sRGB.
- Materials are MeshStandardMaterial (`color`, `metalness`, `roughness`,
  `emissive`, `emissive_intensity`, `ao_map_intensity`, `opacity`,
  `alpha_cutoff`, `size`, `mode`, `side`, texture maps `color_map`,
  `normal_map`, `roughness_map` (green), `metalness_map` (blue; glTF packs
  both in one texture), `emissive_map`, `ao_map` (red, applied to indirect
  light only as three.js's `aoMap` does), `alpha_map` (green, times opacity,
  as three.js's `alphaMap`; with `.Transparent` or `.Alpha_Mask`, shadows
  cut too), per-slot UV transforms `*_transform`
  (`offset`, `rotation`, `scale`, `tex_coord` 0 or 1 selecting `uv` or `uv1`),
  and the booleans as a bit set: `flags = {.Transparent, .Alpha_Mask,
  .Flat_Shading, .No_Size_Attenuation}`), ShadowMaterial (`.Shadow_Only`:
  only the shadows it receives, `color` at `opacity` times the light they
  take, blended over what is behind: shadows on a backdrop or a video
  without a visible ground), MeshBasicMaterial (`.Basic`),
  MeshToonMaterial (`.Toon`: banded diffuse and a stepped highlight; any
  front-sided material also takes `outline_width` and `outline_color`, an
  inverted-hull cartoon outline), or
  ShaderMaterial: your own vertex and fragment shaders in GLSL ES (three.js's
  GLSL 1 style with `attribute`, `varying` and `gl_FragColor`, or GLSL 3 with
  `glsl3 = true`), with three.js's built-in uniforms (`modelMatrix`,
  `modelViewMatrix`, `projectionMatrix`, `viewMatrix`, `normalMatrix`,
  `cameraPosition`, `isOrthographic`, plus `instanceColor` and `instanceId`:
  an instanced draw's instance colour, white without one, and 0-based index)
  and attributes (`position`, `normal`, `uv`), compiled by [oksl](https://github.com/defnotmid/oksl) on both
  renderers. Fragment shaders sample textures as three.js's do: `uniform
  sampler2D`s bound to `Material.shader_maps` by channel (`shader_map`
  finds a sampler's by name, once), `texture` / `texture2D` at the level
  its derivatives pick (the full mip chain, repeating), `textureLod`,
  `texelFetch` and `textureSize`, uv up the image as with three.js's flipY;
  an unbound sampler reads white. The output is written as the shader
  leaves it; `.Transparent` blends it, `side` culls as for any material. `projectionMatrix` maps depth to
  0..1 (SDL_GPU's clip space) rather than WebGL's -1..1, which matters only to
  shaders that build their own. On a morphed or skinned mesh, `position` and
  `normal` arrive already morphed and posed by the skin's joints.
- Lights are AmbientLight, HemisphereLight, DirectionalLight (from
  `position` toward `target`), PointLight (`distance`, `decay`), and SpotLight
  (`position`, `target`, `distance`, `decay`, outer half-angle `angle` default
  `pi/3`, `penumbra` 0..1 default 0), in three.js's units: a white directional
  light of intensity 1 lights a white surface facing it to 1/pi of white, so
  scenes use intensities around 2 to 3.
- Shadows are DirectionalLightShadow's (orthographic) and SpotLightShadow's
  (perspective, `fov = 2 * angle`) with PCF, for up to `MAX_SHADOWS = 4`
  shadow-casting directional or spot lights per frame: a light with
  `cast_shadow` and its `shadow` settings (`map_size`, `height` of a
  directional light's square orthographic view, `near`, `far`, `bias`,
  `normal_bias`; zero takes three.js's 512, 10, 0.5, 500 or the spot light's
  `distance`), draws with `.Cast_Shadow` and `.Receive_Shadow` in their
  material's flags (three.js has them on the mesh; a draw's material is its
  own). Casters draw their other side (three.js's default shadowSide) and
  morphed/skinned geometry, discarding shadow texels below `alpha_cutoff` when
  `.Alpha_Mask` is set (cut-out shadows); ShaderMaterials cast (including
  morphed and skinned meshes) but do not receive.
- Geometry takes the three.js parameters: `box(w, h, d, segments...)`,
  `sphere(radius, widthSegments, heightSegments)`, `plane`, `cylinder`
  (radiusTop 0 is a cone), `torus`, `capsule(radius, height, capSegments,
  radialSegments)`, plus `scene_tube(scene, points, radius, radialSegments,
  closed)` (a per-frame lit tube through the points), `line_segments(positions)` (pairs of
  endpoints), `polyline(positions, closed)` (connected path),
  `wireframe(src)` (edges of a triangle mesh), and `points(positions)` (point
  clouds). Outlines become geometry as three.js's Shape, ShapeGeometry and
  ExtrudeGeometry: a `Shape` is an outline and its holes as points (curves
  flattened by the caller), `shapes_from_contours(contours, fill_rule)` is
  ShapePath.toShapes (a font's or SVG's closed contours grouped into shapes
  with holes, nonzero or even-odd), `triangulate_shape(contour, holes)` is
  ShapeUtils.triangulateShape (earcut), `shape_geometry(shapes)` the flat
  faces, and `extrude_geometry(shapes, options)` the solid (`depth`,
  `steps`, `bevel_enabled`, `bevel_thickness`, `bevel_size`, `bevel_offset`,
  `bevel_segments` from `EXTRUDE_DEFAULT`, WorldUVGenerator's UVs), with its
  two material groups as geometries of their own (`caps`, `sides`;
  `geometry_group` is addGroup). three.js's extrusions are flat-faceted;
  `crease_angle` smooths the walls and bevels instead, caps kept flat.
  `Extruded.folded` counts the contours a bevel turned inside out (an
  inward `bevel_offset` over half the thinnest feature, a `bevel_size`
  over half a hole, pixels touching at a corner under an outward bevel):
  made as three.js makes them, but a host can warn. A `Path` keeps the
  curves (three.js's Path / ShapePath: `path_move_to`, `path_line_to`,
  `path_quadratic_curve_to`, `path_bezier_curve_to`, `path_close`);
  `path_contours(path, curve_segments)` flattens it as getPoints does, for
  the above, and `exact_shape_geometry(path, fill_rule)` is ShapeGeometry
  with exact edges: one quad whose pixels take their coverage from the
  curves (the exact area each pixel covers, by ovg's curves package), so the outline has
  no facets at any distance. Any standard material draws it: a
  `.Transparent` one blends each pixel by the area it covers (the exact
  edge), an opaque one writes the samples inside the curves (4x MSAA's
  edge, exact curves, no sorting); shadows take the outline. Text is the
  same with a quad per glyph: `curve_set_add(path)` (or
  `curve_set_add_words` for an ovg font's glyph shapes) keeps a glyph's
  curves once, and `glyph_run_geometry(glyphs, scale)` places them as the
  caller's layout says (positions in run units; a negative y scale takes
  a font's y-down units Y up), one geometry and one draw a run. Any triangle geometry can also be drawn as wireframe lines or
  vertex points by setting `material.mode = .Lines` or `.Points`. Line width
  and point size are controlled by `material.size` in world units (perspective
  foreshortened) or in screen pixels when `.No_Size_Attenuation` is in
  `material.flags`. Any draw can be instanced (`InstancedMesh`) via
  `scene_instances` and `scene_set_draw_instances` with per-instance
  `transform` and optional `color`. The same primitive parameters give the same
  id, so a frame may ask for `box(1, 1, 1)` every time; `primitive(key)` makes
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
sky, _ := o3d.texture_load(&textures, "sky.hdr")          // Radiance HDR: linear half floats, past 1
scene: o3d.Scene                           // filled every frame
o3d.scene_clear(&scene)
scene.camera = {position = {0, 2, 6}, target = {0, 0.5, 0}, fov = 40, near = 0.1, far = 100}
scene.tone_mapping, scene.tone_mapping_exposure = .ACES_Filmic, 1.0
scene.fog = {kind = .Exp2, color = {0.05, 0.06, 0.08}, density = 0.04}
append(&scene.lights, o3d.Light{kind = .Hemisphere, color = {0.8, 0.9, 1}, ground = {0.2, 0.15, 0.2}, intensity = 1.5})
append(&scene.lights, o3d.Light{kind = .Directional, color = {1, 1, 1}, intensity = 2.5, position = {3, 5, 2},
	cast_shadow = true, shadow = {map_size = 1024, bias = -0.0005}})
m := o3d.MATERIAL_DEFAULT
m.color, m.roughness, m.color_map = {1, 0.35, 0.3}, 0.4, albedo
m.flags = {.Cast_Shadow, .Receive_Shadow}
append(&scene.draws, o3d.Draw{o3d.box(&geos, 1, 1, 1), o3d.compose({0, 0.5, 0}, {0, t, 0}), m})

// glTF 2.0 / GLB, USDZ / USDC, 3MF and OBJ (model_load picks by extension): loaded once into flat tables, drawn per frame at time t
duck, err := o3d.gltf_load("duck.glb", &geos, &textures)   // err: .None, or why (GLTF_ERROR_TEXT[err])
part, _ := o3d.threemf_load("part.3mf", &geos, &textures)  // err: .None, or why (THREEMF_ERROR_TEXT[err])
mesh, _ := o3d.obj_load("mesh.obj", &geos, &textures)      // err: .None, or why (OBJ_ERROR_TEXT[err])
dock, _ := o3d.usd_load("dock.usdz", &geos, &textures)     // err: .None, or why (USD_ERROR_TEXT[err])
o3d.gltf_draw(&scene, &duck, o3d.compose({-1.5, 0, 0}), anim = 0, time = t)
o3d.gltf_draw(&scene, &part, o3d.compose({1.5, 0, 0}))
// Morph target overrides by target name or 0-based index (applied after animation):
// o3d.gltf_draw(&scene, &face, anim = 0, time = t, morphs = {{name = "Smile", weight = 0.8}, {index = 1, weight = 0.5}})
// From several threads, each with its own pose (the model is then only read):
// o3d.gltf_draw(&scene, &duck, anim = 0, time = t, pose = &thread_pose)

// Parts of one model by durable id (names looked up once): rows by node and
// by material, a node's row holding for the nodes under it
rover, _ := o3d.gltf_load("rover.glb", &geos, &textures)
wheel, body := o3d.model_node(&rover, "Wheel_L1"), o3d.model_node(&rover, "Body")
net := o3d.model_material(&rover, "SteelNet")
parts: o3d.Model_Parts
o3d.model_parts_init(&parts, &rover)
parts.nodes[body - 1].remove = {.Cast_Shadow}                        // the body and everything under it
parts.materials[net - 1].hidden = true                               // every primitive of that material
parts.transforms[wheel - 1] = {set = {.Rotation}, rotation = spin}   // the wheel's own rotation; its children follow
o3d.gltf_draw(&scene, &rover, flags = {.Cast_Shadow}, parts = &parts)
hub := o3d.model_node_world(&rover, wheel)                           // where the wheel went, to attach other draws
lo, hi, _ := o3d.model_bounds(&rover, &geos)                         // its rest-pose box, for placing and framing

// Where a world point lands on the frame (a lens flare, a label): pixel, depth, in front
px, depth, in_front := o3d.camera_project(scene.camera, 1920, 1080, sun_position)
// Particles, per frame: rows of position, size, rotation, colour and opacity (the emitter is the host's)
puffs := o3d.scene_particles(&scene, smoke_rows)                    // []o3d.Particle
smoke := o3d.MATERIAL_DEFAULT
smoke.color_map, smoke.softness = puff_sprite, 0.3                   // the sprite; fade 0.3 before what they meet
append(&scene.draws, o3d.Draw{puffs, o3d.compose(), smoke})          // drawn unlit, farthest first
// sparks: the same with flags = {.Additive}

// The frame's depth, to hide it behind what is nearer and for post effects
scene.depth = true                                                   // then render as usual
hidden := o3d.point_hidden(scene.camera, o3d.software_depth(&sw), 1920, 1080, sun_position)   // GPU: read_depth(&g, o3d_sdlgpu.depth(&g), ...)

// Text and logos: outlines (flattened to points) grouped into shapes, extruded once, drawn per frame
shapes := o3d.shapes_from_contours(glyph_contours)          // ShapePath.toShapes; holes found by containment
opts := o3d.EXTRUDE_DEFAULT
opts.depth, opts.bevel_segments, opts.crease_angle = 0.4, 4, math.PI / 4
letter := o3d.extrude_geometry(&geos, shapes, opts)         // kept: .geometry, or .caps and .sides
o3d.shapes_destroy(shapes)
append(&scene.draws, o3d.Draw{letter.caps, o3d.compose({0, 1, 0}, {0, t, 0}), face}, o3d.Draw{letter.sides, o3d.compose({0, 1, 0}, {0, t, 0}), rim})

// Flat outlines with exact edges: the curves kept, one quad, coverage per pixel
logo: o3d.Path
o3d.path_move_to(&logo, {0, -1})
o3d.path_bezier_curve_to(&logo, {-1.6, -0.1}, {-0.9, 1.2}, {0, 0.5})
o3d.path_bezier_curve_to(&logo, {0.9, 1.2}, {1.6, -0.1}, {0, -1})
heart := o3d.exact_shape_geometry(&geos, &logo)           // kept, like any geometry
o3d.path_destroy(&logo)
ink := o3d.MATERIAL_DEFAULT
ink.flags = {.Transparent}                                  // coverage by area (opaque: by sample)
append(&scene.draws, o3d.Draw{heart, o3d.compose({0, 2, 0}), ink})

// Text: each glyph's curves kept once (here an ovg font's), placed by the layout
glyph_a := o3d.curve_set_add_words(&geos, a.data, a.origin, a.extent)   // a: ovg.font_glyph_shape(font, gi)
run := o3d.glyph_run_geometry(&geos, []o3d.Glyph{{glyph_a, {0, 0}}, {glyph_a, {0.6, 0}}}, {1.0 / 1000, -1.0 / 1000})   // font units, y down
append(&scene.draws, o3d.Draw{run, o3d.compose({0, 3, 0}), ink})

// ShaderMaterial: compiled once (same sources, same id), uniforms per frame
shaders: o3d.Shaders                       // kept, like geometry
wave, errors := o3d.shader(&shaders, vertex_glsl, fragment_glsl)   // errors: "Fragment:3:7: ..."
w := o3d.MATERIAL_DEFAULT
w.shader = wave
w.uniforms = o3d.scene_uniforms(&scene, {"uTime", {t}}, {"uColor", {0.3, 0.8, 0.4}})
w.shader_maps[o3d.shader_map(&shaders, wave, "map")] = earth   // its `uniform sampler2D map`
append(&scene.draws, o3d.Draw{o3d.sphere(&geos, 1), o3d.compose(), w})

// GPU (o3d/sdlgpu): borrows the host's SDL_GPU device, or makes a headless one
g: o3d_sdlgpu.GPU
o3d_sdlgpu.init(&g, device)
out := o3d_sdlgpu.target(&g, 1920, 1080)
o3d_sdlgpu.render(&g, &scene, &geos, &shaders, out, 1920, 1080, cmd, &textures)   // or nil cmd: submit and wait

// A texture whose pixels change every frame (a video) and are already on the GPU:
clip := o3d.texture_add_external(&textures, 1280, 720)   // once; a map like any other
o3d_sdlgpu.texture_set_external(&g, &textures, clip, frame_texture)   // before each render that draws it

// Recorded at a point of the host's own command stream, where nothing may be made:
job: o3d_sdlgpu.Job                         // kept per view, reused every frame
o3d_sdlgpu.prepare(&g, &job, &scene, &geos, &shaders, 1920, 1080, &textures)   // makes and uploads (submits its own copies)
o3d_sdlgpu.record(&g, &job, out, cmd)      // later, in the host's command buffer: passes only

// CPU: the same pixels, rows banded on the caller's thread pool
sw: o3d.Software
pixels := o3d.software_render(&sw, &scene, &geos, &shaders, 1920, 1080, pool, &textures)
```

o3d imports oksl as `deps:oksl`, ovg's core-only curves package as
`deps:ovg/curves` and odecs' image package (texture decoding, PNG writing)
as `deps:odecs/image`: clone [oksl](https://github.com/defnotmid/oksl),
[ovg](https://github.com/defnotmid/ovg) and
[odecs](https://github.com/defnotmid/odecs) next to o3d, and builds pass
`-collection:deps=..` (the directory holding them). `shaders/build.sh`
includes ovg's `shaders/curves.slang` from there too.

- `odin test . -collection:deps=..` and `odin test sdlgpu
  -collection:deps=..`: the CPU and GPU tests. A GPU test fails without a
  device unless `O3D_NO_GPU=1` (CI without a GPU; macOS, below).
- `odin run tools/gpu_check -collection:deps=.. [-- out_dir]`: every
  GPU-vs-software parity case (`PARITY_CASES`) on the main thread, by PSNR,
  optionally saving both frames, then the external texture and prepared
  jobs checks. On macOS this is the GPU check: SDL's
  Metal driver needs the main thread, which `odin test` does not give.
- `odin run bench -collection:deps=.. -o:speed`: the zero-allocation
  software benchmark; `odin run demo -collection:deps=.. -- out_dir`: a
  test scene both ways to PNGs.

## Design

- **Data:** the scene is flat tables (lights, draws: geometry id, model
  matrix, material, plus optional joint, morph-weight, and instance ranges)
  refilled each frame, the draws a structure of arrays (`#soa`) since the
  per-frame passes read a column or two; geometry is the only retained state:
  one array of interleaved position / normal / uv / uv1 vertices (40 bytes),
  one of 32-bit indices, a separate array of `Skin_Vertex` (4 joint indices
  and normalized weights) for skinned meshes, and a separate array of
  `Morph_Delta` (position and normal deltas, 32 bytes) for morphed meshes so
  meshes without skinning or morph targets pay nothing. Before rendering a
  frame, `expand_draws` expands instanced draws (`scene.instances`) and turns
  `.Lines` / `.Points` draws into camera-facing quad triangles in reusable
  scratch buffers (`Expanded_Draws`, zero warm-frame allocations), returning
  `scene` unchanged when no draw uses instancing, lines, or points. The GPU
  mirrors the static geometry arrays in vertex, index, and storage buffers
  (uploading only what is new) and uploads any expanded line/point vertices
  into reusable dynamic vertex/index buffers.
- **glTF (`gltf_read.odin`, `gltf.odin`):** `gltf_read` reads a `.gltf`
  (external, data-URI or GLB buffers) or `.glb` into a `Gltf_Document`, the
  file's tables as flat arrays referring to each other by index, its JSON
  parsed by `core:encoding/json` into a scratch arena, its buffers loaded,
  and the whole checked by the rules cgltf's validation used (accessors and
  views inside their data, indices and sparse indices in range, one count
  per primitive, morph weights matching targets, no node with two parents
  or in a cycle); `gltf_accessor_floats` / `gltf_accessor_uints` unpack an
  accessor in one sweep per component type (strides, normalized integers,
  padded matrix columns, sparse values). Files may come from anyone, so a
  `Limits` table (`limits.odin`, as ovg's) bounds JSON and hierarchy depth,
  node and accessor counts, buffer bytes, unpacked values and image pixels:
  a `Max` limit fails the file (`Gltf_Error.Limit`), a `Total` one skips
  the primitive, channel or image it would pass, each with a line in
  `Model.warnings`. From the document `gltf_load` and `gltf_parse`
  keep node and material names (`Model_Node.name`, `Model.material_names`,
  each primitive's `material_id`; OBJ's `o`/`g` and `newmtl` names and
  3MF's object names likewise), found once by `model_node` /
  `model_material` as durable ids, and
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
  1 per texture reference) into `Material`, punctual lights
  (`KHR_lights_punctual`: directional, point, and spot lights with `color`,
  `intensity`, `range`, and cone angles `innerConeAngle` / `outerConeAngle`)
  into `Model.lights`, the node hierarchy into a topologically sorted `#soa`
  array (`parent < i`), skins (`joints` and `inverseBindMatrices`, up to
  `MAX_JOINTS = 64`) into flat arrays, and keyframe animations (`STEP`,
  `LINEAR`, `CUBICSPLINE`, including morph `"weights"` channels) into flat
  time/value arrays. `gltf_draw` evaluates the hierarchy, animation, morph
  weights (with optional `Morph_Weight` overrides by target name or 0-based
  index applied after animation), node punctual lights (transformed by the
  node's world matrix with directional/spot lights aiming along local `-Z`,
  appended to `scene.lights`), and joint matrices (`inv_mesh * world[joint] *
  ibm`) at time `t` with zero heap allocations on warm frames, posing into
  the caller's `Model_Pose` (one per thread lets threads pose one model at
  once) or the model's own. `Model_Parts` are the caller's tables by node
  and material id: node rows (hidden, flags added or removed) fold down
  the hierarchy in the same parent-first sweep as the world matrices,
  transform rows replace a node's translation, rotation or scale (a
  matrix node's decomposed first), and material rows hide, re-flag or
  replace every primitive of their material. Morphs (up to
  `MAX_MORPH_TARGETS = 8` active non-zero targets per draw) are applied before skinning in the per-chunk
  vertex stage on the software renderer and in `vs_main` / `vs_skinned`
  (`shaders/o3d.slang`, reading the `Morph_Delta` storage buffer via
  `SV_VertexID`) on the GPU, including the shadow pass and before a
  `ShaderMaterial`'s vertex stage. A failed load returns a `Gltf_Error` with
  `GLTF_ERROR_TEXT` for messages. If `extensionsRequired` names an extension
  o3d does not implement, loading fails with
  `Gltf_Error.Unsupported_Extension` and records the extension name in
  `Model.unsupported_extension`. Optional extensions listed only in
  `extensionsUsed` (`KHR_materials_ior`, `KHR_materials_transmission`,
  `KHR_materials_volume`, `KHR_materials_clearcoat`, `KHR_materials_sheen`,
  `KHR_materials_specular`, `KHR_materials_iridescence`,
  `KHR_materials_anisotropy`, `EXT_texture_webp`, …) are drawn without, as
  the spec allows, each with a line in `Model.warnings` naming the
  materials that have it. Remaining unsupported required extensions (rejected with
  `.Unsupported_Extension`): `KHR_draco_mesh_compression`,
  `EXT_meshopt_compression`, `KHR_texture_basisu`, `EXT_texture_webp`, and
  any other extension outside `KHR_materials_unlit`,
  `KHR_materials_emissive_strength`, `KHR_texture_transform`,
  `KHR_mesh_quantization`, and `KHR_lights_punctual`. Also not imported: glTF
  cameras.
- **3MF (`threemf.odin`):** `threemf_load` and `threemf_parse` read `.3mf`
  (3D Manufacturing Format) ZIP archives (`3D/3dmodel.model` via `_rels/.rels`,
  plus production-extension sub-model `.model` parts) into the same flat
  `Model`, `Geometries` and `Textures` tables as `gltf_load`, so `gltf_draw`
  and `model_destroy` work on 3MF models with no special cases. 3MF Z-up
  vertices and 3x4 affine `<component>` / `<item>` transforms are converted
  into o3d's Y-up space via the +1-determinant basis change `(x, y, z) ->
  (x, z, -y)` (`B * M * B^-1`), sub-models with a different `<model unit>`
  scale into the root model's unit, and `<basematerials>` (`displaycolor`),
  `<m:colorgroup>` (`color`), `<m:pbmetallicdisplayproperties>`
  (`metallicness`, `roughness`), and `<m:texture2d>` + `<m:texture2dgroup>`
  (`u`, `v`) map to `Material` and `Textures` (splitting multi-material meshes
  into one `Model_Primitive` per distinct material). Unsupported entries in
  `requiredextensions` fail with `Threemf_Error.Unsupported_Extension` and
  record the namespace URI in `Model.unsupported_extension`. Faces share
  smooth normals within a 60° crease angle; sharper edges stay hard.
- **OBJ (`obj.odin`):** `obj_load` and `obj_parse` read Wavefront `.obj`
  (`v` with optional vertex colours, `vt`, `vn`, `f` with N-gons fanned and
  negative indices, `o` / `g` as nodes, `usemtl` splitting primitives) and its
  `mtllib` `.mtl` (`Kd`, `Ke`, `Ks`/`Ns` or `Pr`/`Pm`, `d`/`Tr`, and the
  `map_Kd`, `map_Ke`, `map_Bump`, `map_Pr`/`map_Ns`, `map_Pm`, `map_d`
  textures) relative to the file, into the same tables. Faces without `vn`
  get the 60° crease-angle normals. `model_load` picks glTF, USD, 3MF or OBJ
  by extension.
- **USD (`usd_read.odin`, `usd.odin`):** `usd_load` and `usd_parse` read a
  `.usdz` (its first `.usdc` layer, its textures from the archive) or a
  binary `.usdc` (text `.usda` is not read). `usd_read_layer` reads the
  Crate file's tables as OpenUSD's `crateFile.cpp` lays them out (version
  0.4.0 on: tokens, strings, fields, field sets, paths and specs, LZ4 and
  USD's integer coding, its own LZ4 block decoder) into flat arrays by
  index, values read on demand by typed getters into the caller's buffers.
  `usd_parse` composes the layer as USD does within one layer: a prim's
  sources are its own specs then, weaker, what it references internally
  (each reference expanded in turn); children merge by name; attributes
  take the strongest opinion (a default, else the first time sample). From
  the composed prims: Xform/Scope nodes with their `xformOpOrder` ops;
  Meshes fanned into triangles with `normals` / `primvars:normals` and the
  materials' UV primvar by their interpolation and `:indices`, smooth
  normals where there are none, `orientation`, `doubleSided`, and a
  primitive per `materialBind` GeomSubset; inherited `material:binding` to
  UsdPreviewSurface materials (values or UsdUVTexture inputs through
  UsdTransform2d, a channel other than the one o3d's slot reads copied
  once into a grey texture; `opacity` textures as `alpha_map`,
  `opacityThreshold` as `.Alpha_Mask`); `primvars:displayColor`; and the
  stage's `metersPerUnit` and `upAxis` as a root node to meters, Y up.
  Instances' meshes are built once and share geometry. References to other
  files, UsdSkel, lights, cameras and other prim types are not read, each
  named in `Model.warnings`; the `Limits` table bounds it as glTF's. Checked
  against OpenUSD (pxr) on 115 files: every composed mesh's triangles,
  world bounds and materials.
- **Shapes (`shape.odin`):** three.js's code ported: earcut (mapbox/earcut
  as three.js carries it, nodes in a pool linked by index),
  ShapePath.toShapes' containment (largest first, an interior point tested
  against the innermost enclosing contour, fill-rule windings), and
  ExtrudeGeometry's layers (back bevel rings, `steps + 1` wall rings, front
  bevel rings; getBevelVec's offsets). They run in f64 as three.js does in
  JavaScript, so the geometry is three.js's, triangle for triangle (checked
  against three.js r186 to 1e-6, but for a wall's UV on an edge at exactly
  45°, where x and y tie). Holes are always rewound against the outline
  (three.js only does so when it reverses the outline). The output is
  indexed, corners with equal position, normal and UV shared, the caps'
  triangles first so each group is one index range. With `crease_angle`,
  a wall corner's normal sums the wall faces there within the angle,
  weighted by their corner angles. Folds are found per shape at each bevel
  ring's offset: an edge reversed against the contour's, a winding
  flipped, or a proper crossing (a sweep over the edges) that the
  unbevelled contours do not have.
- **Tables:** primitives are made by `PRIMITIVES`, indexed by `Primitive`.
  Each frame, `shading_lights` and `shading_materials` turn the lights and
  the draws' materials into their shading form (linear colours, spot cone
  cosines and axis, the opacity drawn, 2x3 UV transform matrices in o3d UV
  space, UV set bits, `Shading_Flags`) once; both renderers read those, and
  the GPU's uniforms are them, packed. Boolean sets are bit sets
  (`Material_Flags`, `Shading_Flags`, `Texture_Slots`, a pixel's covered
  samples).
- **Shader contract:** `Light_Kind`'s values and `Shading_Flag`'s bits are
  the shader's `LIGHT_*` and `FLAG_*` constants; a test reads
  `shaders/o3d.slang` and checks them.
- **Shading** is one model in two places, `shade` / `finish_color` in
  `scene.odin` (`tone_map` in `o3d.odin`) and `shaders/o3d.slang`
  (`shaders/common.slang`): Lambert diffuse plus GGX specular with Smith
  correlated visibility and Schlick Fresnel, plus equirectangular
  environment lighting (`scene.environment`, `scene.environment_intensity`)
  and background (`scene.background`, `scene.background_intensity`,
  `scene.background_blurriness`), as three.js's MeshStandardMaterial
  computes it, with `ao_map` (`1 + (r - 1) * ao_map_intensity`) multiplying
  indirect light only (ambient, hemisphere, environment irradiance and
  radiance), followed by `tone_map` (`None`, `Linear`, `Reinhard`, `Cineon`,
  `ACES_Filmic`, `AgX`, `Neutral` with `tone_mapping_exposure`, applied to
  both drawn materials and the equirectangular background) and `fog_factor`
  (`.Linear` or `.Exp2` blending toward `linear(fog.color)` before sRGB
  encoding). When a texture is used as `scene.environment` (or blurred
  `scene.background`), the texture has derived maps, as it has mipmaps: a
  GGX-prefiltered radiance mip chain (roughness 0..1) and a
  cosine-convolved diffuse irradiance map in `Textures`, sampled alike by
  both renderers. `environment_prefilter(textures, id)` makes them; it is
  heavy, so a host calls it at load (a render that meets a texture without
  them makes them then, in frame time); `texture_environment` looks them up.
  An HDR environment or background (a Radiance `.hdr` through
  `texture_decode` / `texture_load`, by odecs, or `texture_add_hdr` from
  linear floats) keeps its range: its texels are linear half floats
  (`Textures.hdr_texels`, `Texture.hdr`), its mipmaps and derived maps are
  made in float, both renderers read it linear (RGBA16F on the GPU), so a
  sun far past 1 lights as one and tone mapping takes it down. Standard
  material maps read an HDR texture's values as they would encoded colour;
  HDR is for environments, backgrounds and ShaderMaterials.
  Changing one means changing the other; the tests compare the two
  renderers.
- **Particles:** `scene_particles` keeps a frame's particles as rows
  (position, size in world units or pixels, rotation in the view plane,
  sRGB colour and opacity); expansion turns each into a camera-facing quad
  (farthest first unless `.Additive`) with a row in the frame's sprite
  table, which both renderers read per quad as exact shapes read their
  curve tables (the GPU through the curves' vertex shader and
  `fs_sprite`). The material's `color_map` is the sprite and its colour and
  opacity multiply the row's; drawn unlit, blended or added
  (`.Additive`), fogged and tone mapped; `softness` fades a particle over
  that distance before the frame's depth (soft particles), which is then
  drawn for them. The emitter (where particles are, frame by frame) is the
  host's.
- **Frame depth:** with `Scene.depth` both renderers also draw the
  frame's depth through their shadow pass, from the camera at the frame's
  size: per pixel, the depth they test (0 near, 1 far, 1 where nothing is,
  `camera_project`'s depth for a point) of the nearest opaque draw at the
  pixel's centre (blended draws let what is behind show; ShaderMaterials
  with their plain geometry, alpha masks cut). `software_depth` gives the
  software renderer's; `o3d_sdlgpu.depth` / `job_depth` a sampleable R32F
  texture for the host's own passes and `read_depth` its values;
  `point_hidden` says whether it hides a world point (the four texels
  around it, the farthest kept, so a point on a drawn surface is not hidden
  by it; `bias` for points a little inside one). gpu_check
  compares the two (all but 0 to 2 texels in some 30000 within 3e-6).
- **External textures:** `texture_add_external` adds a `Textures` entry
  with no texels, for pixels the host already has on the GPU and changes
  every frame (a video's frames). `o3d_sdlgpu.texture_set_external` binds
  the host's texture to it for the renders that follow: same device,
  `.SAMPLER` usage, RGBA with sRGB-encoded bytes in a UNORM format (as an
  image's), sampled at its own mip levels (none unless the host makes
  them). The binding is for that table only (a remade table's reused ids
  sample white), the host's texture is never released by o3d, and the host
  unbinds (nil) before releasing it. The software renderer has no such
  pixels: it samples an external texture white (the material's own colour).
- **Prepared jobs:** `render` is `prepare` then `record` with the GPU's
  own `Job`. A host that records 3D at a point of its own command stream
  (a 2D canvas's ordered pass, where creating a GPU resource may submit)
  calls them itself: `prepare` makes everything the scene's render needs
  (frame targets, the job's shadow maps and per-frame buffers, the tables'
  mirrors, ShaderMaterials' pipelines) and uploads what is new in a
  command buffer it submits; `record` records the shadow passes, the main
  pass and the copy into the host's command buffer, making nothing. A
  frame's several views each prepare a job first and record later, in any
  order, once per prepare. Jobs share the GPU's mirrors of the tables (the
  jobs prepared together draw from the same tables; a prepare on other
  tables starts the mirrors over, and an earlier job then records
  `.Not_Prepared`) and its frame targets (grown to the largest job); each
  job has its own shadow maps and per-frame buffers.
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
- **Shadows** (shadow.odin) are drawn first for up to `MAX_SHADOWS = 4`
  shadow-casting directional (orthographic) or spot (perspective) lights
  (`frame_shadows`): the casters' depth from each light (`shadow_casters`,
  culled to its view), one sample per texel centre, nearest kept (discarding
  texels below `alpha_cutoff` for `.Alpha_Mask` casters with a `color_map`);
  the software renderer runs the same phases in a depth-only mode at each
  map's size, the GPU renders into R32_FLOAT colour targets (sampleable on
  every backend) with depth buffers. A receiver's pixel scales each shadow
  light's direct light by `shadow_factor` (with perspective divide `c.xyz /
  c.w`): a 3x3 grid of bilinear depth comparisons (a 4x4 block of texels
  weighed 1 - f, 1, 1, f per axis), edges repeated, outside the light's view
  lit. The shader has the same arithmetic (`shadow_factor` in
  `shaders/o3d.slang`); `FLAG_RECEIVE_SHADOW` and `FLAG_ALPHA_MASK` join the
  shader contract.
- **Order:** draws whose bounding sphere (the geometry's radius plus
  `(sum |w_t|) * morph_radius`, scaled with the draw, or covering the posed
  joints for a skinned draw) is outside the view, or whose shader failed,
  are skipped; then opaque and alpha-masked draws as listed, then blended
  ones (transparent with opacity under 1, a transparent ShaderMaterial, or
  a transparent exact shape) farthest first, without writing depth
  (`draw_order`, shared by both renderers).
- **Exact shapes and text (`curve.odin`):** a path's contours go to
  ovg's curves package as they are (cubics split into quadratics within a
  tolerance), which builds the shape's block of 32-bit words (monotone
  pieces in a grid of cells, each cell with its backdrop winding, so a
  pixel's cost is the pieces near it; and a mean coverage), or an ovg
  glyph's block is copied: a curve set.
  `Geometries.curve_words` keeps every set's block, and every `.Curves`
  geometry's table of its quads' sets (a word a quad: the set's first
  word, the top bit for even-odd), and the GPU mirrors it as one storage
  buffer. The geometry is a `.Curves` quad per glyph (an exact shape is a
  run of one) over its set's bounds, `uv1` the shape coordinate; the
  fragment finds its quad's entry by the quad (`vs_curves` passes vertex
  index / 4 flat, the draw's vertices bound from its first so they count
  from 0, as morphed draws are; software keeps each triangle's entry).
  Each frame `expand_draws` grows each quad
  by half a pixel's footprint around each edge for the view (from the
  model-view-projection's derivative at each corner), so edge pixels just
  outside the bounds are drawn. A pixel's footprint is the shape
  coordinate's derivatives within its 2x2 quad (fine derivatives on the
  GPU, the quad's differences in software): a blended draw's alpha is
  `curves.coverage_affine`, the exact area of the footprint's
  parallelogram inside the shape (Green's theorem over the curves crossing
  it, so exact at grazing angles too; never faded to the shape's mean,
  which it already is once the footprint holds the whole shape); an
  opaque draw's
  sample mask (`SV_Coverage`) is the winding
  at the four sample positions, extrapolated from the centre. Both
  renderers call the same procedures (`curves.odin`, `curves.slang`), so
  they agree as the other cases do. The shadow pass keeps texels whose
  centre is inside. Coverage by sample writes depth and needs no sorting;
  coverage by area blends, so it is drawn with the blended draws. Shapes are flat in their
  XY plane (the quad grows along x and y); a ShaderMaterial draws the bare
  quad.
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
  material, in triangle order. Its samplers read the draw's
  `shader_maps` at the level oksl picks from the quad's derivatives.
- Errors are values (`Error`); no threads of its own: the software
  renderer runs its phases on the caller's `thread.Pool`.

## Next

Integrated in onimite (`view3d{}`). What is still missing, found by using it,
most important first.

For models people download:

1. **Compressed models and remaining glTF extensions:** decoders for Draco
   (`KHR_draco_mesh_compression`), meshopt (`EXT_meshopt_compression`),
   KTX2 / Basis textures (`KHR_texture_basisu`) and WebP textures
   (`EXT_texture_webp`), which today fail cleanly with
   `Gltf_Error.Unsupported_Extension` when required; and optional PBR
   extensions (`KHR_materials_transmission`, `KHR_materials_ior`,
   `KHR_materials_volume`, `KHR_materials_clearcoat`, `KHR_materials_sheen`,
   `KHR_materials_specular`), which today are ignored when listed in
   `extensionsUsed`.

For typical three.js scenes:

2. **Point light shadows:** omnidirectional cube/dual-paraboloid shadow maps
   (directional and spot light shadows are supported, up to 4 per frame).

For onimite:

3. **Textures that change every frame:** a video clip or a group's 2D
   drawing mapped onto a surface. Today a texture is added once
   (`texture_add`); this needs an update path the host fills per frame
   (on the GPU: a texture the host renders into).
4. **Exact extrusion caps**, optional: extrusions' caps as exact shapes
   (unbevelled first; a bevelled cap is an offset outline). Opaque caps
   draw by sample, which equals the mesh, so this waits on a need.

Metal: checked on an Apple Silicon Mac (macOS 27), last with exact shapes:
`tools/gpu_check` matches the software renderer in all 14 parity cases (53
to 87 dB PSNR) and passes the external texture and prepared jobs checks, and onimite's notarized macOS release draws its 3D examples
on Metal as on the CPU. (v0.0.1's standard shader did not compile there:
a texel `Load` on a combined texture-sampler, which Slang emits wrongly
for MSL; the shadow map is now sampled at texel centres.)

