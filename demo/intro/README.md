# o3d intro

A 48 second showcase of o3d, music included: 1920x1080, 60 fps, 140 BPM.
It is an [onimite](https://github.com/defnotmid/onimite) script (Odin +
Lua video framework, which draws its 3D with o3d), not part of o3d's
build: `demo/main.odin` beside this folder is the Odin demo, and nothing
here touches it. The renders are not in the repository: make them with
onimite's release binary
([releases](https://github.com/defnotmid/onimite/releases), v0.1.0;
`onimite update` keeps it current), from o3d's folder:

```sh
sh demo/intro/fetch.sh                      # once: the glTF models, SHA-256 checked
onimite render demo/intro/software.lua --software -o demo/intro/media/parity_software.mp4 --qp 12
onimite render demo/intro/video.lua -o demo/intro/demo.mp4 --qp 22
onimite preview demo/intro/video.lua        # play, hot reload on save
```

The second command draws the Parity scene's 3D with o3d's software
renderer, on the CPU; the video's Parity scene plays that clip in its right
half beside the same view drawn on the GPU. Render the video itself on the
GPU (no `--software`), or both halves come from the CPU.

| bars | scene | what |
| --- | --- | --- |
| 1-2 | Title | Oxa, o3d's mascot, as its own SVG (`assets/logo.svg`, 32 x 32 pixel art) on a pixel grid, hopping on the chips' kick, blinking, its gills swapping poses; the chips alone |
| 3-4 | Lift | the band comes in: the same pixels as voxels (one RoundedBoxGeometry, 636 instances, each its pixel's colour), first seen exactly as the 2D was, growing their depth from the middle out as the camera swings round; the wordmark |
| 5-8 | Materials | one a coin: MeshStandardMaterial metal, a clear coat, glass (transmission, ior, thickness) refracting a small Oxa, velvet sheen, MeshToonMaterial with an outline, in a studio drawn once with `bake{}` |
| 9-12 | Light + lens | half time in the dark: a SpotLight swinging its shadow round, the gills emissive under the view's bloom, then the lens opening to f/1.4 and racking focus to the lamps behind |
| 13-16 | glTF 2.0 | the chorus: the Khronos Fox (a skin, each clip in turn), AnimatedMorphCube (morph targets), ToyCar (clearcoat, sheen, transmission) |
| 17-20 | 2D on 3D | the breakdown, chips alone again: a screen whose picture is a panel (Oxa's SVG, its parts swapped on the beat), "o3d" on the floor as a ShapeGeometry (exact coverage), Oxa's silhouette extruded with bevels |
| 21-24 | Parity | one view, the left half drawn on the GPU (SDL_GPU) and the right by the software renderer, and gpu_check's numbers |
| 25-28 | Outro | the wordmark extruded beside Oxa's voxels over the spectrum; the song lands on its tonic only here |

## Facts on screen

Every number is in `facts.lua` with its source:

| on screen | source |
| --- | --- |
| GPU = software 49 / 49, PSNR 43.8 – 71.1 dB, Radeon (Vulkan, RADV) · Apple M4 (Metal) | `tools/gpu_check`, o3d v0.1.0 with oksl v0.1.0, 2026-11-17: 49 of 49 cases match on both; the parity cases' PSNR on RADV runs from 43.8 dB (motion blur) to 71.1 dB (shapes) |

The rest of the text names three.js's types and fields as the script uses
them (`f_stop 1.4`, `ior 1.5 · thickness 1.2`, ...), the values each scene
draws with.

## Files

- `video.lua`: the video, its HUD, the kick's pulse and the impacts'
  flashes; each scene in `scenes/` calls `scene{}` itself, starting on its
  song section.
- `track.lua`: the soundtrack, an odawg `song{}`, chiptune over a live band
  after Anamanaguchi (Endless Fantasy, 2013), 140 BPM in D major on J-pop's
  royal road (Gmaj7 A7 F#m7 Bm7), the tonic held back for the last chord.
  The chips (a pulse lead, a pulse arpeggio, a triangle bass, noise drums)
  are rows of its own; the band (a kit, a fingered bass, strings) comes in
  as Oxa turns to voxels, and drops out again for the 2D-on-3D breakdown.
  In the Parity scene the hook is played by two instruments in unison, a
  chip on the left and the strings on the right. Its coin blips are the
  picture's pops, its impacts the flashes.
- `timeline.lua` reads from the song on which beat everything happens;
  `ui.lua` holds the palette (Oxa's pinks and its gills' X, Y, Z red, green
  and blue), type, springs and labels; `stage.lua` the shared 3D (Oxa's
  voxels, the studio, the floor and lights); `oxa.lua` reads Oxa's pixels
  from its SVG; `parity_view.lua` is the Parity scene's 3D, which
  `software.lua` renders alone.

Mix (`onimite audio analyze`, onimite v0.1.0): -14.4 LUFS integrated,
-2.0 dBTP true peak, no clipped samples; the scenes from -19.3 LUFS (the
breakdown) and -17.3 (the title) to -12.4 (the chorus). Each part sits in
its role's band of LU under the mix (`onimite audio parts`), with lanes
for the breakdown, the half-time verse and the tail.

## Media

- `oxa.svg`: a link to o3d's `assets/logo.svg` (drawn by
  `tools/mascot/oxa.py`).
- Fonts (SIL Open Font License 1.1, `media/OFL.txt` with each one's
  copyright), copied from ovg's demo: Bebas Neue, Space Grotesk, JetBrains
  Mono.
- `fetch.sh` downloads three models from Khronos' glTF-Sample-Assets at a
  pinned commit; they are not kept here (`media/.gitignore`):
  - Fox: model CC0 (PixelMannen), rigging and animations CC-BY 4.0
    ([@tomkranis](https://sketchfab.com/tomkranis); glTF conversion by
    @AsoboStudio and @scurest);
  - AnimatedMorphCube: CC0 (Microsoft);
  - ToyCar: CC0 (Guido Odendahl, Eric Chadwick).
- `media/instruments/sfx.json`: the onimite skill library's sound-effect
  rows (the riser and the impact).
- The soundtrack is composed by odawg from `track.lua`; the band plays
  odawg's sample sets, CC0, fetched when it plays: virtuosity's jazz kit,
  Karoryfer's fingered bass (growlybass) and VSCO 2's string sections.

Made with onimite; music by odawg.
