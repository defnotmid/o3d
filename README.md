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

MIT licensed ([LICENSE](LICENSE)); what it carries (oksl, ocurves,
odecs' image package, SDL 3, three.js's models, Odin's
packages) is in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Conventions (three.js's)

What is known about three.js carries over:

- Right-handed and Y up; the camera looks down -Z.
- Rotations are Euler angles in radians, applied in X, Y, Z order; a
  transform is position, rotation and scale (`compose`), like Object3D.
- The camera's field of view is vertical, in degrees (PerspectiveCamera);
  an orthographic camera takes the view's height. A perspective camera has
  a thin lens: `f_stop` (0: a pinhole), `focus` (metres along its axis) and
  `film_height` (millimetres, 24 by default; with `fov` the focal length),
  so a point at depth z blurs over a circle (f / f_stop) * f / (focus - f) *
  |1 - focus / z| across on the film. One view gathers that blur from its
  depth (`focus_of`); `lens`, a point on the unit aperture disc, draws the
  view sharp through that point instead (the focus plane kept), so a host
  averaging views over the aperture gets the exact depth of field. Its
  `shutter` (seconds, centred on the frame's moment) smears particles along
  their `velocity`, and blurs the frame along its motion: given where
  things were at the shutter's opening (the camera's pose,
  `camera.opened`; a draw's model, instances, joints, morph weights and
  ShaderMaterial uniforms, `scene_set_draw_opened`, `_opened_instances`,
  `_opened_joints`, `_opened_morphs`, `_opened_uniforms`; a glTF model's,
  `gltf_draw`'s `opened`; none: where it is now), each pixel streaks from
  there through now and on as far again, up to `max_blur` of the frame's
  height (`MOTION_MAX_BLUR`, 0.04). The frame's depth pass writes each
  pixel's velocity and nearness (a ShaderMaterial's by its own vertex
  stage, run once more at the opening by oksl's taps), and a
  reconstruction filter (`motion_blur.odin`) gathers the light along it
  before the depth of field. The filter is public for any frame of light:
  `motion_blur` on spans (software) and `o3d_sdlgpu.record_motion_blur` on
  textures, with `Motion_Blur{max_blur, depth_scale}` and the caller's
  `Motion_Tables`.
- Colours are sRGB; lighting is linear. The frame is drawn in light, as
  three.js's EffectComposer draws into a HalfFloat target: each fragment's
  light, blended toward `scene.fog` (`.Linear` between `near` and `far`, or
  `.Exp2` with `density`, by Euclidean camera distance), premultiplied by
  its coverage, blended and added there (half floats, 4x MSAA), resolved,
  then once a pixel through the camera's depth of field (a gather in
  tiles of the largest blur near, at most a sixteenth of the frame's
  height), tone-mapped
  (`scene.tone_mapping`: `.None` default, `.Linear`, `.Reinhard`,
  `.Cineon`, `.ACES_Filmic`, `.AgX`, `.Neutral`, scaled by
  `scene.tone_mapping_exposure`, default 1) and encoded to sRGB (three.js's
  OutputPass). Additive glows add as light; the background colour, the fog
  and ShaderMaterials' output are light, tone-mapped with the rest.
- Materials are MeshStandardMaterial (`color`, `metalness`, `roughness`,
  `emissive`, `emissive_intensity`, `ao_map_intensity`, `env_intensity`
  (three.js's envMapIntensity: the environment's light on it as a share of
  the scene's, 0 meaning 1), `opacity`,
  `alpha_cutoff`, `size`, `mode`, `side`, texture maps `color_map`,
  `normal_map` (with `normal_scale`), `roughness_map` (green), `metalness_map` (blue; glTF packs
  both in one texture), `emissive_map`, `ao_map` (red, applied to indirect
  light only as three.js's `aoMap` does), `alpha_map` (green, times opacity,
  as three.js's `alphaMap`; with `.Transparent` or `.Alpha_Mask`, shadows
  cut too), per-slot UV transforms `*_transform`
  (`offset`, `rotation`, `scale`, `tex_coord` 0 or 1 selecting `uv` or `uv1`),
  and the booleans as a bit set: `flags = {.Transparent, .Alpha_Mask,
  .Flat_Shading, .No_Size_Attenuation}`) with MeshPhysicalMaterial's
  layers, composed as three.js r181 does: `ior` (a dielectric's
  reflectance), `clearcoat` and `clearcoat_roughness` (a second GGX lobe
  over everything, emissive too, lit about the surface's normal before its
  normal map, or `clearcoat_normal_map`'s), `sheen_color` and
  `sheen_roughness` (cloth's Charlie lobe), and `transmission` with
  `thickness`, `attenuation_color` and `attenuation_distance` (glass: the
  opaque draws are kept as a linear, untone-mapped snapshot with mipmaps,
  seen through the glass refracted by the ior out the thickness (each axis
  by the model's scale), read with three.js's bicubic filter, blurred by
  roughness, absorbed over the ray and mixed into the diffuse light before
  tone mapping; transmissive draws come after the opaque ones, farthest
  first, and see neither each other, the back faces of double-sided ones,
  nor transparent draws; over the empty background they leave it
  see-through, where three.js shows half-white). Their maps, three.js's:
  `clearcoat_map` (red), `clearcoat_roughness_map` (green),
  `clearcoat_normal_map` (with `clearcoat_normal_scale`),
  `sheen_color_map` (rgb, sRGB), `sheen_roughness_map` (alpha),
  `transmission_map` (red), `thickness_map` (green), each with its
  transform; a material's layer maps may use two distinct textures (the
  GPU binds two; a map in a third is dropped on both renderers — none of
  the 150 Khronos sample models uses more than two). A ShaderMaterial
  behind glass is seen as it draws into the frame (its own tone mapping
  included). Planar reflections (three.js's Reflector, physically):
  `Scene.reflector` is a plane
  (`point`, `normal` toward the side it mirrors, `scale` the reflection's
  share of the frame's size, 0.5 by default), and a material flagged
  `.Reflector` is a surface of it (as three.js's Reflector is the mesh
  that shows it): the scene mirrored there takes the place of the
  environment map's reflection, all of it, where the mirror shows
  something (where it shows nothing the environment does; no environment
  is needed). How strong and how sharp the reflection is belongs to the
  material, as the environment's does: its Fresnel (metalness, ior, the
  clear coat's layer) and its roughness, not a separate amount; a dimmer
  reflection is a rougher or less metallic surface. The roughness blurs it
  as far as the mirrored ray runs to what it meets: GGX's lobe (alpha = roughness^2) spreads over about 1.29 alpha
  times that run, seen from the eye that run further away, and the mip
  level (a full chain) is the one whose texels are that blur across. What
  touches the plane reflects sharp and what is far from it blurs most
  (contact hardening); the clear coat blurs by its own roughness. The
  mirror is drawn first, through the mirrored camera at its size, in
  linear light (blended draws blended there), each sample's run beside
  its light (premultiplied by coverage, a second target), leaving out
  what is behind the plane (not for ShaderMaterials, which write no runs
  either), particles, shadow catchers and the surfaces that show it.
  Those read it where they are on the screen: the run where their ray
  meets something, else the coverage-weighted runs about it at the level
  an endless run would take, so the edge of what is mirrored blurs as it
  does. They belong on the
  plane (a floor, a table top, a mirror). One reflector a scene; it is
  not drawn when the camera is behind it or nothing that shows it is in
  view; transmission takes a `.Reflector` material's place of reflection,
  and glass in the mirror shows the environment behind it. ShadowMaterial (`.Shadow_Only`:
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
  an instanced draw's instance colour, linear as three.js's, white without
  one, and 0-based index)
  and attributes (`position`, `normal`, `uv`), compiled by [oksl](https://github.com/defnotmid/oksl) on both
  renderers. Fragment shaders sample textures as three.js's do: `uniform
  sampler2D`s bound to `Material.shader_maps` by channel (`shader_map`
  finds a sampler's by name, once), `texture` / `texture2D` at the level
  its derivatives pick (the full mip chain, repeating), `textureLod`,
  `texelFetch` and `textureSize`, uv up the image as with three.js's flipY;
  an unbound sampler reads white. The output is written as the shader
  leaves it, as light; `.Transparent` blends it, `side` culls as for any material. `projectionMatrix` maps depth to
  0..1 (SDL_GPU's clip space) rather than WebGL's -1..1, which matters only to
  shaders that build their own. On a morphed or skinned mesh, `position` and
  `normal` arrive already morphed and posed by the skin's joints.
- Lights are AmbientLight, HemisphereLight, DirectionalLight (from
  `position` toward `target`), PointLight (`distance`, `decay`), and SpotLight
  (`position`, `target`, `distance`, `decay`, outer half-angle `angle` default
  `pi/3`, `penumbra` 0..1 default 0), in three.js's units: a white directional
  light of intensity 1 lights a white surface facing it to 1/pi of white, so
  scenes use intensities around 2 to 3.
- Shadows are three.js's light shadows with PCF, for up to `MAX_SHADOWS = 4`
  shadow-casting directional or spot lights per frame: a light with
  `cast_shadow` and its `shadow` settings. A directional light's camera is
  orthographic and, with no `height`, fitted to the frame: across the
  light, the receivers in view (the mirror's too) where a caster is over
  them; along it, from the nearest caster to the farthest such receiver;
  its size rounded up to an eighth of a doubling and its corner on the
  texel grid, so its texels hold still while the camera moves within a
  step (only the light's direction counts). With a `height` it is that
  square about the light's target, `near` to `far` from its position (0.5
  and 500 by default). A spot light's is its cone (`fov = 2 * angle`, out to
  `far`, else its `distance`). `map_size` (512 by default) texels a side;
  `bias` and `normal_bias` are world units, the point compared moved away
  from the light (negative: toward it) and along its normal. Draws with
  `.Cast_Shadow` and `.Receive_Shadow` in their
  material's flags (three.js has them on the mesh; a draw's material is its
  own). Casters draw their other side (three.js's default shadowSide) and
  morphed/skinned geometry, discarding shadow texels below `alpha_cutoff` when
  `.Alpha_Mask` is set (cut-out shadows); ShaderMaterials cast (including
  morphed and skinned meshes) and receive by calling
  `shadowAt(worldPosition, worldNormal)` in their fragment shader: how lit
  the point is by the first shadow-casting light, 0 to 1, as the standard
  material's (normal bias, bias, the same PCF; three.js's getShadow). Its
  map takes the shader's first sampler channel (`shader_map` finds the
  shader's own by name). A ShaderMaterial writes light, as three.js's does
  into its EffectComposer's render target: the frame tone maps and encodes
  it with everything else. It finishes as the standard material does by
  calling `fogged(color, worldPosition)` (the scene's fog, linear);
  `toneMapping(color)` and `linearToOutputTexel(value)` are as is (a render
  target's), kept for three.js's sources; `sRGBTransferEOTF(value)` decodes
  an sRGB colour to linear (three.js's). These, like `shadowAt`, are
  declared only for a shader that names them.
- Geometry takes the three.js parameters: `box(w, h, d, segments...)`,
  `sphere(radius, widthSegments, heightSegments, phiStart, phiLength,
  thetaStart, thetaLength)`, `plane`, `circle(radius, segments,
  thetaStart, thetaLength)`, `ring(innerRadius, outerRadius,
  thetaSegments, phiSegments, thetaStart, thetaLength)`, `cylinder`
  (radiusTop 0 is a cone), `torus`, `capsule(radius, height, capSegments,
  radialSegments)`, `rounded_box(width, height, depth, segments, radius)`
  (the RoundedBoxGeometry addon), `lathe(points, segments, phiStart,
  phiLength)` (made each call: its points are no key; the others are kept
  by their parameters), all as three.js r186 makes them (to 1e-6), plus `scene_tube(scene, points, radius, radialSegments,
  closed)` (a per-frame lit tube through the points), `line_segments(positions)` (pairs of
  endpoints), `polyline(positions, closed)` (connected path),
  `wireframe(src)` (edges of a triangle mesh), and `points(positions)` (point
  clouds). Outlines become geometry as three.js's Shape, ShapeGeometry and
  ExtrudeGeometry. A path is ocurves' `curves.Contours` (lines and cubics
  in one table: `contours_begin` is moveTo, `contours_line` lineTo,
  `contours_cubic` bezierCurveTo, `contours_quad` quadraticCurveTo).
  `shapes_from_contours(&shapes, &contours, fill_rule, curve_segments)` is
  ShapePath.toShapes with getPoints: the area the contours fill by the rule
  (`curves.Fill_Rule`: `.Non_Zero` or `.Even_Odd`)
  (ocurves' simplify, so contours that cross or overlap are merged where
  three.js keeps them overlapping), as outlines with their holes,
  flattened at `curve_segments` points a curve, into one `Shapes` table
  (points, rings as ranges of them, shapes as ranges of rings).
  `triangulate_shape(&shapes, i)` is ShapeUtils.triangulateShape (earcut),
  `shape_geometry(&shapes)` the flat faces, and
  `extrude_geometry(&shapes, options)` the solid (`depth`,
  `steps`, `bevel_enabled`, `bevel_thickness`, `bevel_size`, `bevel_offset`,
  `bevel_segments`, `miter_limit` from `EXTRUDE_DEFAULT`, WorldUVGenerator's
  UVs), with its two material groups as geometries of their own (`caps`,
  `sides`; `geometry_group` is addGroup). three.js's extrusions are
  flat-faceted; `crease_angle` smooths the walls and bevels instead, caps
  kept flat. The bevel offsets the shape as its straight skeleton does
  (three.js moves each point along its corner's direction, which turns
  thin parts inside out): each edge moves its distance along its normal,
  corners stay where its lines meet unless that is more than `miter_limit`
  times the distance out (4, SVG's; then they are cut across their
  bisector), parts thinner than twice an inward offset close up, holes
  narrower than an outward one close, and parts an offset brings together
  join, so every offset makes a closed solid whose caps are the offset
  outlines. `exact_shape_geometry(&contours, fill_rule)` is ShapeGeometry
  with exact edges: one quad whose pixels take their coverage from the
  curves (the exact area each pixel covers, by ocurves), so the outline has
  no facets at any distance. Any standard material draws it: a
  `.Transparent` one blends each pixel by the area it covers (the exact
  edge), an opaque one writes the samples inside the curves (4x MSAA's
  edge, exact curves, no sorting); shadows take the outline. Text is the
  same with a quad per glyph: `curve_set_add(&contours)` (or
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
composite over anything, unless `scene.background_color` (sRGB and alpha,
three.js's `scene.background` as a Color) gives the frame a clear colour:
with alpha 1 it is opaque everywhere, so what is added (a glow, a plume)
lands on that colour rather than on nothing; it is light like the rest,
tone-mapped with it. A `scene.background` texture covers it. Light added
past a pixel's coverage (glows over nothing) stays as colour past its
alpha, which a premultiplied over adds.

## Use

```odin
geos: o3d.Geometries                       // kept: geometry is made once
textures: o3d.Textures                     // kept: textures and mipmaps are made once
albedo, _ := o3d.texture_load(&textures, "brick.png")
sky, _ := o3d.texture_load(&textures, "sky.hdr")          // Radiance HDR: linear half floats, past 1
// or three.js's Sky (Preetham), baked once at load: dawn = o3d.sky_texture(&textures, {sun = {0.6, 0.08, -1}, up = {0, 1, 0},
//     turbidity = 10, rayleigh = 3, mie_coefficient = 0.005, mie_directional_g = 0.7}, 2048)
scene: o3d.Scene                           // filled every frame
o3d.scene_clear(&scene)
scene.camera = {position = {0, 2, 6}, target = {0, 0.5, 0}, fov = 40, near = 0.1, far = 100}
scene.tone_mapping, scene.tone_mapping_exposure = .ACES_Filmic, 1.0
scene.fog = {kind = .Exp2, color = {0.05, 0.06, 0.08}, density = 0.04}
append(&scene.lights, o3d.Light{kind = .Hemisphere, color = {0.8, 0.9, 1}, ground = {0.2, 0.15, 0.2}, intensity = 1.5})
append(&scene.lights, o3d.Light{kind = .Directional, color = {1, 1, 1}, intensity = 2.5, position = {3, 5, 2},
	cast_shadow = true, shadow = {map_size = 1024, bias = -0.005}})
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
// With motion blur: also as it was at the shutter's opening (each draw's model, joints and weights then)
// o3d.gltf_draw(&scene, &duck, anim = 0, time = t, opened = o3d.Gltf_Opening{transform = then_xf, time = t - shutter / 2})

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
boxes := make([][2]o3d.Vec3, len(rover.nodes))
o3d.model_node_bounds(&rover, &geos, boxes)                          // each node's subtree's {lo, hi} in the model's space

// Where a world point lands on the frame (a lens flare, a label): pixel, depth, in front
px, depth, in_front := o3d.camera_project(scene.camera, 1920, 1080, sun_position)
// Particles, per frame: rows of position, size (width, height), rotation, colour and opacity (the emitter is the host's)
puffs := o3d.scene_particles(&scene, smoke_rows)                    // []o3d.Particle
smoke := o3d.MATERIAL_DEFAULT
smoke.color_map, smoke.softness = puff_sprite, 0.3                   // the sprite; fade 0.3 before what they meet
append(&scene.draws, o3d.Draw{puffs, o3d.compose(), smoke})          // drawn unlit, farthest first
// sparks: the same with flags = {.Additive}; .Additive works on any material (a back-face atmosphere shell, a halo shader, glowing lines)
// A lens flare: additive rows at the sun's position (ghosts moved by offset), drawn over everything and faded by how much
// of a 10 px disc about the sun the frame's depth leaves unhidden (in the render, no read back)
flare.flags, flare.depth_offset, flare.occlusion_radius = {.Additive, .No_Size_Attenuation}, 1e6, 10
// Vertex colours (three.js's vertexColors, glTF's COLOR_0): Vertex.color (sRGB + alpha bytes, o3d.vertex_color({r, g, b, a})),
// read by materials with flags += {.Vertex_Colors}; per-frame paths take them too
stars := o3d.scene_points(&scene, positions, colors)                 // colors: []Vec4, sRGB + alpha, one a point
trail := o3d.scene_polyline(&scene, positions, colors = colors)      // a gradient along it; scene_line_segments(&scene, pairs, colors)
// Labels: pixel-sized rows the panel's width x height with offset = {0, 0.5} (the bottom edge's middle on the point), the
// material's depth_offset bringing them in front of what they are pinned to; a canvas's
// premultiplied image as color_map with flags += {.Premultiplied_Color_Map}

// The frame's depth, to hide it behind what is nearer and for post effects
scene.depth = true                                                   // then render as usual
hidden := o3d.point_hidden(scene.camera, o3d.software_depth(&sw), 1920, 1080, sun_position)   // GPU: read_depth(&g, o3d_sdlgpu.depth(&g), ...)
o3d.depth_image(scene.camera, o3d.software_depth(&sw), 1920, 1080, far, rgba16f)   // distance (a 1 where drawn); GPU: record_depth_into

// Text and logos: a path's filled area as shapes (flattened), extruded once, drawn per frame
shapes: o3d.Shapes                                          // kept warm for the next path
o3d.shapes_from_contours(&shapes, &glyph_contours)          // ShapePath.toShapes: outlines and holes as filled
opts := o3d.EXTRUDE_DEFAULT
opts.depth, opts.bevel_segments, opts.crease_angle = 0.4, 4, math.PI / 4
letter := o3d.extrude_geometry(&geos, &shapes, opts)        // kept: .geometry, or .caps and .sides
o3d.shapes_destroy(&shapes)
append(&scene.draws, o3d.Draw{letter.caps, o3d.compose({0, 1, 0}, {0, t, 0}), face}, o3d.Draw{letter.sides, o3d.compose({0, 1, 0}, {0, t, 0}), rim})

// Flat outlines with exact edges: the curves kept, one quad, coverage per pixel
logo: curves.Contours                                       // import curves "deps:ocurves"
curves.contours_begin(&logo, {0, -1})
curves.contours_cubic(&logo, {-1.6, -0.1}, {-0.9, 1.2}, {0, 0.5})
curves.contours_cubic(&logo, {0.9, 1.2}, {1.6, -0.1}, {0, -1})
heart := o3d.exact_shape_geometry(&geos, &logo)           // kept, like any geometry
curves.contours_destroy(&logo)
ink := o3d.MATERIAL_DEFAULT
ink.flags = {.Transparent}                                  // coverage by area (opaque: by sample)
append(&scene.draws, o3d.Draw{heart, o3d.compose({0, 2, 0}), ink})

// Text: each glyph's curves kept once (here an ovg font's), placed by the layout
glyph_a := o3d.curve_set_add_words(&geos, a.data, a.origin, a.extent)   // a: ovg.font_glyph_shape(font, gi)
run := o3d.glyph_run_geometry(&geos, []o3d.Glyph{{glyph_a, {0, 0}}, {glyph_a, {0.6, 0}}}, {1.0 / 1000, -1.0 / 1000})   // font units, y down
append(&scene.draws, o3d.Draw{run, o3d.compose({0, 3, 0}), ink})

// ShaderMaterial: compiled once (same sources, same id), uniforms per frame
shaders: o3d.Shaders                       // kept, like geometry
wave, errors := o3d.shader(&shaders, vertex_glsl, fragment_glsl)   // errors: []Shader_Error (code, stage, oksl's with its position); text: format_shader_errors
w := o3d.MATERIAL_DEFAULT
w.shader = wave
w.uniforms = o3d.scene_uniforms(&scene, {"uTime", {t}}, {"uColor", {0.3, 0.8, 0.4}})
w.shader_maps[o3d.shader_map(&shaders, wave, "map")] = earth   // its `uniform sampler2D map`
// shadows in it: `float lit = max(dot(n, sunDir), 0.0) * shadowAt(vWorld, n);` (the first shadow light's)
// it writes light, fogged as the standard material: `gl_FragColor = vec4(fogged(c, vWorld), 1.0);`
append(&scene.draws, o3d.Draw{o3d.sphere(&geos, 1), o3d.compose(), w})

// GPU (o3d/sdlgpu): borrows the host's SDL_GPU device, or makes a headless one
g: o3d_sdlgpu.GPU
o3d_sdlgpu.init(&g, device)
out := o3d_sdlgpu.target(&g, 1920, 1080)
o3d_sdlgpu.render(&g, &scene, &geos, &shaders, out, 1920, 1080, cmd, &textures)   // or nil cmd: submit and wait

// A texture whose pixels change every frame (a video) and are already on the GPU:
clip := o3d.texture_add_external(&textures, 1280, 720)   // once; a map like any other
o3d_sdlgpu.texture_set_external(&g, &textures, clip, frame_texture)   // before each render that draws it
o3d.software_set_external(&sw, clip, frame_pixels, 1280, 720)         // the software renderer's: RGBA8 in memory
// The frame's depth as distance into the host's RGBA16F image, after record in the same command buffer
depth_img := o3d_sdlgpu.depth_image_target(&g, 1920, 1080)
o3d_sdlgpu.record_depth_into(&g, &g.job, depth_img, far, cmd)       // a scene with Scene.depth

// Recorded at a point of the host's own command stream, where nothing may be made:
job: o3d_sdlgpu.Job                         // kept per view, reused every frame
o3d_sdlgpu.prepare(&g, &job, &scene, &geos, &shaders, 1920, 1080, &textures)   // makes and uploads (submits its own copies)
o3d_sdlgpu.record(&g, &job, out, cmd)      // later, in the host's command buffer: passes only

// CPU: the same pixels, rows banded on the caller's thread pool
sw: o3d.Software
pixels := o3d.software_render(&sw, &scene, &geos, &shaders, 1920, 1080, pool, &textures)
// or into the host's own image, with no copy: o3d.software_render(&sw, &scene, &geos, &shaders, 1920, 1080, pool, &textures, out = image_pixels)
```

o3d imports oksl as `deps:oksl`, ocurves' core-only curves package as
`deps:ocurves` and odecs' image package (texture decoding, PNG writing)
as `deps:odecs/image`: clone [oksl](https://github.com/defnotmid/oksl),
[ocurves](https://github.com/defnotmid/ocurves) and
[odecs](https://github.com/defnotmid/odecs) next to o3d, and builds pass
`-collection:deps=..` (the directory holding them). `shaders/build.sh`
includes ocurves' `curves.slang` from there too, and records the curves
format it reads (`curves.FORMAT`) in `sdlgpu/curves_format.odin`:
o3d_sdlgpu does not compile against an ocurves of another format (each o3d
tag names the ocurves it needs), where its curve shaders would draw
garbage.

- `odin test . -collection:deps=..` and `odin test sdlgpu
  -collection:deps=..`: the CPU and GPU tests. A GPU test fails without a
  device unless `O3D_NO_GPU=1` (CI without a GPU; macOS, below).
- `odin run tools/gpu_check -collection:deps=.. [-- out_dir]`: every
  GPU-vs-software parity case (`PARITY_CASES`) on the main thread, by PSNR,
  optionally saving both frames, then the external texture, prepared
  jobs, depth and depth image checks. On macOS this is the GPU check: SDL's
  Metal driver needs the main thread, which `odin test` does not give.
- `odin run bench -collection:deps=.. -o:speed`: the zero-allocation
  software benchmark; `odin run demo -collection:deps=.. -- out_dir`: a
  test scene both ways to PNGs.

## Design

- **Data:** the scene is flat tables (lights, draws: geometry id, model
  matrix, material, plus optional joint, morph-weight, and instance ranges)
  refilled each frame, the draws a structure of arrays (`#soa`) since the
  per-frame passes read a column or two; geometry is the only retained state:
  one array of interleaved position / normal / uv / uv1 / colour vertices
  (44 bytes; the colour sRGB and alpha bytes, read with `.Vertex_Colors`),
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
  non-normalized integer attributes from `KHR_mesh_quantization`, and views
  compressed with `EXT_meshopt_compression` (meshoptimizer's attribute,
  triangle and index-sequence codecs and its octahedral, quaternion and
  exponential filters, as gltfpack and glTF-Transform write them; decoded
  once into a buffer of the document's), plus morph
  target `POSITION` and `NORMAL` deltas — `TANGENT` deltas are ignored as
  tangents are derived from screen derivatives) into `Geometries`, mesh
  default `weights` and `extras.targetNames` into `Model`, images into
  `Textures` (WebP ones by `EXT_texture_webp`), metallic-roughness materials (`alphaMode`: `OPAQUE`, `MASK`
  with `alphaCutoff` default 0.5, `BLEND`; `occlusionTexture` red channel +
  `strength`; `KHR_materials_unlit`; `KHR_materials_emissive_strength`;
  `KHR_texture_transform` `offset`, `rotation`, `scale` and `texCoord` 0 or
  1 per texture reference; `normalTexture.scale`; `KHR_materials_ior`,
  `_clearcoat`, `_sheen`, `_transmission` and `_volume` with their
  textures) into `Material`, punctual lights
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
  `extensionsUsed` (`KHR_materials_specular`, `KHR_materials_iridescence`,
  `KHR_materials_anisotropy`, `KHR_materials_dispersion`, …) are drawn
  without, as
  the spec allows, each with a line in `Model.warnings` naming the
  materials that have it. Remaining unsupported required extensions (rejected with
  `.Unsupported_Extension`): `KHR_draco_mesh_compression`,
  `KHR_texture_basisu`, and any other extension outside
  `KHR_materials_unlit`, `KHR_materials_emissive_strength`,
  `KHR_texture_transform`, `KHR_mesh_quantization`, `KHR_lights_punctual`,
  `EXT_meshopt_compression`, `EXT_texture_webp`, `KHR_materials_ior`,
  `KHR_materials_clearcoat`, `KHR_materials_sheen`,
  `KHR_materials_transmission` and `KHR_materials_volume` (gltfpack or glTF-Transform
  turn a Draco file into a meshopt one). A compressed view that does not
  decode fails the file with `Gltf_Error.Invalid_Compression`. Also not
  imported: glTF cameras.
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
- **Shapes (`shape.odin`):** a path's contours go through ocurves'
  `bool_run` once (one operand under its fill rule, its simplify), which
  arranges every piece, keeps the faces the rule fills and gives each
  connected region its outline (clockwise, Y up) and holes
  (counter-clockwise) as ranges of its result, each segment with the range
  of its source it covers; each is flattened from there (a curve
  `curve_segments` points over its whole parameter, so an uncut one as
  three.js flattens it) into the `Shapes` table. Then three.js's code
  ported: earcut (mapbox/earcut as three.js carries it, nodes in a pool
  linked by index) and ExtrudeGeometry's layers (back bevel, walls, front bevel). They run in
  f64 as three.js does in JavaScript, so an unbevelled extrusion is
  three.js's triangle for triangle (checked against three.js r186 to 1e-6,
  but for a wall's UV on an edge at exactly 45°, where x and y tie), and a
  bevel that changes nothing about the outline has three.js's rings and
  quads. The bevel's rings come from a wavefront (`wave_run`): each edge a
  line moving at unit speed along its normal, each vertex where its two
  lines meet; edge events (an edge shrinks to nothing: its ends become one
  vertex) and split events (a reflex vertex runs into an edge ahead: the
  loop splits, or two loops join) are taken in time order, each the
  earliest of a scan over the live edges and the reflex vertices against
  them. A ring samples every vertex; an event samples only the vertices
  whose edges it changes, so each edge's face (planar in x, y and time)
  is the zip of its two ends' sample chains between when it began and
  ended, with no T-junctions, and the faces stay one quad a ring where
  nothing happens. A corner past `miter_limit` starts as a chamfer line of
  no length along its bisector. The caps triangulate the loops at the
  first ring (outlines with the holes inside them), the walls stand on the
  loops at the last. Holes are always rewound against the outline
  (three.js only does so when it reverses the outline). The output is
  indexed, corners with equal position, normal and UV shared, the caps'
  triangles first so each group is one index range. With `crease_angle`,
  a wall corner's normal sums the wall faces there within the angle,
  weighted by their corner angles.
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
- **Shading** is one model in two places, `shade` / `fogged` in
  `scene.odin` (`tone_map` in `o3d.odin`) and `shaders/o3d.slang`
  (`shaders/common.slang`): Lambert diffuse plus GGX specular with Smith
  correlated visibility and Schlick Fresnel, plus equirectangular
  environment lighting (`scene.environment`, `scene.environment_intensity`)
  and background (`scene.background`, `scene.background_intensity`,
  `scene.background_blurriness`), as three.js's MeshStandardMaterial
  computes it, with `ao_map` (`1 + (r - 1) * ao_map_intensity`) multiplying
  indirect light only (ambient, hemisphere, environment irradiance and
  radiance) and `fog_factor` (`.Linear` or `.Exp2` blending toward
  `linear(fog.color)`), written as light; the frame's pixels then through
  `tone_map` (`None`, `Linear`, `Reinhard`, `Cineon`, `ACES_Filmic`, `AgX`,
  `Neutral` with `tone_mapping_exposure`) and sRGB encoding, once each
  (raster.odin's `output`, shaders/post.slang's). When a texture is used as `scene.environment` (or blurred
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
  HDR is for environments, backgrounds and ShaderMaterials. `sky_texture`
  bakes three.js's Sky (Preetham's daylight with the sun's disc, `Sky`,
  `SKY_DEFAULT`, `sky_radiance` per direction) into such a texture, for
  the background and, prefiltered, the environment (three.js draws it as
  a mesh and bakes it with PMREMGenerator).
  Changing one means changing the other; the tests compare the two
  renderers.
- **Motion blur (`motion_blur.odin`, post.slang's `fs_motion*`):** the
  frame's depth pass writes, beside its depth, each pixel's velocity (its
  place now less where its surface was at the shutter's opening, in
  pixels: both clip positions interpolated and divided per pixel, the
  opened one through the camera then and the draw then; nothing where it
  was behind the camera) and nearness (-ln of its distance); where nothing
  is drawn, the far plane moving with the camera. A ShaderMaterial's depth
  is its own vertex stage's, and its position at the opening the same
  stage run again by oksl's tap with its uniforms then (`Shader.depth`).
  The filter follows McGuire et al. 2012's tiles (each tile of `max_blur`
  pixels its longest velocity, then the longest about it; on the GPU the
  tile found in two passes, rows then columns) and gathers 15 taps across
  the tiles' longest motion with interleaved-gradient jitter, but weighs
  them its own way: a pixel shows what passes in front of it (each tap its
  share of the shutter there, the neighbourhood's length over its own)
  and otherwise its own surface and what its or their motion carries
  across. Against 32 subframes at 320x200 that is 38.5 / 35.6 / 29.2 dB on
  a moving box, a pan and both (McGuire's weights: 30.7 / 33.8 / 27.8; the
  sharp frame 26.3 / 18.1 / 18.2). At 1080p it costs about 7 G
  instructions a frame on one core and 0.8 ms on the GPU (Radeon), velocity
  writes included; with motion blur on, the software renderer keeps
  98 MB more (velocity, nearness, the frame's light and its blurred copy).
  Limits: motion is straight per vertex over the shutter; shading that
  changes over it does not blur; blended draws blur by the surface behind
  them (particles smear on their own); reflections and transmission
  snapshots stay sharp; what a fast object uncovers is made from its
  neighbours.
- **Particles:** `scene_particles` keeps a frame's particles as rows
  (position, size as width and height (three.js's Sprite.scale) in world
  units or pixels, rotation in the view plane,
  sRGB colour and opacity, velocity); expansion turns each into a
  camera-facing quad
  (farthest first unless `.Additive`) with a row in the frame's sprite
  table, which both renderers read per quad as exact shapes read their
  curve tables (the GPU through the curves' vertex shader and
  `fs_sprite`). The material's `color_map` is the sprite and its colour and
  opacity multiply the row's; drawn unlit, blended or added
  (`.Additive`), fogged and tone mapped; `softness` fades a particle over
  that distance before the frame's depth (soft particles), which is then
  drawn for them. A row's `offset` moves its quad from its point in its
  width and height, in its turned frame (`{0, 0.5}`: the bottom edge's middle on it, a
  label's anchor); the material's `depth_offset` moves the quads that far
  toward the camera along the ray to each (scaled about the eye, so the
  same on screen; not past the near plane), so a label pinned to a surface
  draws in front of it and still behind what is nearer. A particle moving
  while the camera's shutter is open smears: its quad is grown along its
  way on the frame and each pixel averages its sprite (nothing past its
  edges) at a moment a pixel of that way apart (up to `SMEAR_TAPS = 64`),
  its light premultiplied and linear, its coverage the share of the
  shutter it was there; the sprite only moves, it does not grow as it
  nears the camera. Particles test
  the frame's depth as everything blended does: an alpha-masked net hides
  them behind its wires, not its gaps. The emitter (where particles are,
  frame by frame) is the host's. A material's `occlusion_radius` (pixels)
  fades each particle by the share of a disc that size about its position
  that the frame's depth leaves unhidden (16 points on a Vogel spiral,
  `OCCLUSION_TAPS`; off the frame hidden), in the render on both renderers
  (three.js's Lensflare): with a `depth_offset` past the camera the quads
  draw over everything (at the near plane) yet are probed where the sun is,
  so a flare and its ghosts fade as a hull passes in front of it.
- **Triplanar mapping:** `Material.projection` `.Object` or `.World` reads
  every map (colour, normal, roughness, metalness, emissive, AO, alpha)
  projected along the three axes of the object's frame (the pattern moves
  with it) or the world's, instead of by uv: for meshes without uvs (a USD
  hull) under a normal PBR material, with shadows, environment and tone
  mapping. `projection_scale` is repeats per unit, `projection_sharpness`
  the weights' power (|n| to it, summing to 1); each slot's transform still
  applies. Normal maps blend by a whiteout blend in the projection's frame.
  Alpha masks read this way do not cut shadows (the caster is whole).
  Both renderers alike (`projection_frame`); a triplanar map reads three
  times where it weighs anything.
- **Reversed depth:** `perspective` and `orthographic` map near to depth 1
  and far to 0 (three.js's reversedDepthBuffer), for the camera and the
  lights' shadows alike: both renderers clear depth to 0 and keep the
  greater. A float's precision is densest near 0, so far surfaces keep
  theirs: layers 0.005 apart 200 units out with near 1 and far 40000 no
  longer fight (`far_layers`, `test_depth_precision`), and `near` needs no
  tuning. The frame's depth, `camera_project`'s depth and a ShaderMaterial's
  `projectionMatrix` and `gl_FragCoord.z` follow it; `view_depth`,
  `point_hidden` and `depth_image` turn it into distances as before.
- **Blending:** `blend_of(material)` is a draw's `Blend`: `.Opaque`
  (depth written), `.Over` (blended: transparent with some alpha) or `.Add`
  (`.Additive`, three.js's AdditiveBlending: the light times its opacity
  added, for any material: meshes, ShaderMaterials (SRC_ALPHA, ONE), shapes,
  lines, particles), in the frame's light (half floats, as both renderers
  keep each sample). Blended and added draws come after the opaque ones,
  depth tested, not written; the GPU's pipelines are indexed by it.
- **Vertex colours:** `.Vertex_Colors` multiplies the material's colour by
  its vertices' (`Vertex.color`, sRGB bytes made linear per vertex and
  interpolated so, as three.js does) and its opacity by their alpha (a
  transparent material then blends). glTF's COLOR_0 sets it; `scene_points`,
  `scene_line_segments` and `scene_polyline` take colours, which lines and
  points carry onto their quads (a segment's ends' colours along it).
  ShaderMaterials do not read them yet. Points of their own sizes are
  particles.
- **Premultiplied colour maps:** `.Premultiplied_Color_Map` says the
  `color_map`'s colour is already multiplied by its alpha (a canvas's, an
  image rendered on the GPU): both renderers divide it out after
  filtering (the right order for premultiplied texels), so its
  anti-aliased edges keep their colour.
- **Frame depth:** with `Scene.depth` both renderers also draw the
  frame's depth through their shadow pass, from the camera at the frame's
  size: per pixel, the depth they test (reversed: 1 near, 0 far, 0 where nothing is,
  `camera_project`'s depth for a point) of the nearest opaque draw at the
  pixel's centre (blended draws let what is behind show; ShaderMaterials
  with their plain geometry, alpha masks cut). `software_depth` gives the
  software renderer's; `o3d_sdlgpu.depth` / `job_depth` a sampleable R32F
  texture for the host's own passes and `read_depth` its values;
  `point_hidden` says whether it hides a world point (the four texels
  around it, the farthest kept, so a point on a drawn surface is not hidden
  by it; `bias` for points a little inside one). gpu_check
  compares the two (all but 0 to 2 texels in some 30000 within 3e-6).
  `o3d_sdlgpu.record_depth_into` draws it as distance into the host's
  RGBA16F image (`depth_image_target`, in the same command buffer after
  `record`, with no readback): rgb the distance along the camera's axis
  (`view_depth`, world units) and a 1 where something was drawn; rgb the
  given `far` and a 0 where nothing was. `depth_image` makes the same from
  `software_depth` and `read_depth_image` reads the GPU's back; gpu_check
  compares them (within 1e-3 relatively).
- **External textures:** `texture_add_external` adds a `Textures` entry
  with no texels, for pixels the host already has on the GPU and changes
  every frame (a video's frames). `o3d_sdlgpu.texture_set_external` binds
  the host's texture to it for the renders that follow: same device,
  `.SAMPLER` usage, RGBA with sRGB-encoded bytes in a UNORM format (as an
  image's), sampled at its own mip levels (none unless the host makes
  them). The binding is for that table only (a remade table's reused ids
  sample white), the host's texture is never released by o3d, and the host
  unbinds (nil) before releasing it. The software renderer's counterpart
  is `software_set_external`: the host's RGBA8 pixels in memory (the same
  bytes, rows from the top), read as the GPU reads an unmipped texture
  (bilinear at its level 0), kept by the host while bound; unbound, an
  external texture samples white (the material's own colour).
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
- **GPU targets:** the light (RGBA16F, multisampled), its resolve, the
  depth and the output (RGBA8) targets only grow, to the largest frame
  drawn; every pass of a frame (the mirror's, the transmission snapshot's,
  the frame's) renders into their top-left width x height (viewport and
  scissor), the output pass tone maps and encodes it, and it is copied to
  the caller's texture, so views of several sizes, or one whose size
  animates, make no targets per frame.
- **Anti-aliasing:** 4x MSAA on the GPU; the software renderer keeps four
  samples per pixel at the same positions (each a half-float light, as the
  GPU's), shades once per pixel per triangle, and resolves the same way,
  in light, before tone mapping.
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
  (`frame_shadows`, a fitted directional camera from the draws' bounding
  spheres, `fit_shadows`): the casters' depth from each light
  (`shadow_casters`, culled to its view), one sample per texel centre, nearest kept (discarding
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
  ocurves' builder as they are (`builder_add_contours`) (cubics split into quadratics within a
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
  (values: `Shader_Error`, oksl's diagnostics or o3d's own, whose operands
  live with the failed pipeline, also kept) and its draws left out by both
  renderers; `shader_find` looks one up without making it.
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
   (`KHR_draco_mesh_compression`) and KTX2 / Basis textures
   (`KHR_texture_basisu`), which today fail cleanly with
   `Gltf_Error.Unsupported_Extension` when required; and
   `KHR_materials_specular`, ignored when listed in `extensionsUsed`.

For typical three.js scenes:

2. **Point light shadows:** omnidirectional cube/dual-paraboloid shadow maps
   (directional and spot light shadows are supported, up to 4 per frame).

For onimite:

3. **Exact extrusion caps**, optional: extrusions' caps as exact shapes
   (unbevelled first; a bevelled cap is an offset outline). Opaque caps
   draw by sample, which equals the mesh, so this waits on a need.

Metal: checked on an Apple Silicon Mac (macOS 27), last with exact shapes:
`tools/gpu_check` matches the software renderer in all 14 parity cases (53
to 87 dB PSNR) and passes the external texture and prepared jobs checks, and onimite's notarized macOS release draws its 3D examples
on Metal as on the CPU. (v0.0.1's standard shader did not compile there:
a texel `Load` on a combined texture-sampler, which Slang emits wrongly
for MSL; the shadow map is now sampled at texel centres.)

