#!/usr/bin/env bash
# Compiles o3d.slang and background.slang to the SPIR-V (Vulkan) and MSL
# (Metal) blobs the SDL_GPU backend #loads. Only needed after editing a
# shader; the outputs are committed. DXIL (D3D12) is produced when slangc's
# dxc works; Windows otherwise runs on Vulkan.
set -euo pipefail
cd "$(dirname "$0")"

FLAGS=(-matrix-layout-column-major -O2 -I .)
UNITS=(
	o3d:vert:vs_main
	o3d:skinned_vert:vs_skinned
	o3d:frag:fs_main
	o3d:shadow:fs_shadow
	background:vert:vs_main
	background:frag:fs_main
)
for unit in "${UNITS[@]}"; do
	IFS=: read -r mod suffix entry <<<"$unit"
	slangc "$mod.slang" "${FLAGS[@]}" -entry "$entry" -target spirv -o "$mod.$suffix.spv"
	slangc "$mod.slang" "${FLAGS[@]}" -entry "$entry" -target metal -o "$mod.$suffix.msl"
	if ! slangc "$mod.slang" "${FLAGS[@]}" -entry "$entry" -target dxil -o "$mod.$suffix.dxil" 2>/dev/null; then
		rm -f "$mod.$suffix.dxil"
		echo "  note: no DXIL for $mod.$suffix (slangc's dxc failed)" >&2
	fi
	spirv-val --target-env vulkan1.2 "$mod.$suffix.spv"
	echo "  $mod.slang:$entry -> $mod.$suffix.{spv,msl}"
done

