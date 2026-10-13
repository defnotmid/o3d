#!/usr/bin/env bash
# Compiles o3d.slang to the SPIR-V (Vulkan) and MSL (Metal) blobs the
# SDL_GPU backend #loads. Only needed after editing the shader; the outputs
# are committed. DXIL (D3D12) is produced when slangc's dxc works; Windows
# otherwise runs on Vulkan.
set -euo pipefail
cd "$(dirname "$0")"

FLAGS=(-matrix-layout-column-major -O2)
for unit in vert:vs_main frag:fs_main shadow:fs_shadow; do
	IFS=: read -r suffix entry <<<"$unit"
	slangc o3d.slang "${FLAGS[@]}" -entry "$entry" -target spirv -o "o3d.$suffix.spv"
	slangc o3d.slang "${FLAGS[@]}" -entry "$entry" -target metal -o "o3d.$suffix.msl"
	if ! slangc o3d.slang "${FLAGS[@]}" -entry "$entry" -target dxil -o "o3d.$suffix.dxil" 2>/dev/null; then
		rm -f "o3d.$suffix.dxil"
		echo "  note: no DXIL for $suffix (slangc's dxc failed)" >&2
	fi
	echo "  o3d.slang:$entry -> o3d.$suffix.{spv,msl}"
done
spirv-val --target-env vulkan1.2 o3d.vert.spv
spirv-val --target-env vulkan1.2 o3d.frag.spv
spirv-val --target-env vulkan1.2 o3d.shadow.spv
