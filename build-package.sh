#!/usr/bin/env bash
# Assemble the GPU runtime that score's "support" packages ship for one platform, as
# two zips (a GitHub release asset is capped at 2 GiB, and the whole set is larger):
#
#   cuda-<platform>.zip      CUDA 13 + cuDNN 9: what onnxruntime's CUDA execution
#                            provider links against, plus NPP and nvJitLink.
#   tensorrt-<platform>.zip  TensorRT 11: the runtime librediffusion loads, with the
#                            builder resources it needs to build engines.
#
# Libraries are flat at the zip root, laid out the way NVIDIA ships them: on Linux the
# real file (libcudnn_graph.so.9.27.0) plus symbolic links under the names it is
# looked up by (its soname, libcudnn_graph.so.9). The links matter: NEEDED entries
# and cuDNN's dispatcher, which dlopen()s "libcudnn_graph.so.9.27.0" by full version,
# both resolve through the libraries' $ORIGIN runpath to this directory, never to a
# system copy of another or the same version. score's package manager restores the
# links when it extracts (zipdownloader with symlink support, score > 3.9).
#
# Usage: build-package.sh <linux-x86_64|windows-x86_64> <output-dir>
# All sources are NVIDIA's public, login-free redistributables.

set -euo pipefail

PLATFORM="$1"
OUT="$(realpath -m "$2")"
mkdir -p "$OUT"

# CUDA_REDIST pins the CUDA 13 toolkit components; CUDNN_VERSION must be a cuda13 build.
# TENSORRT_VERSION must match the tensorrt-cu13 pin in librediffusion's uv.lock: engines
# built by its Python toolchain only deserialize with the exact same TensorRT.
CUDA_REDIST="${CUDA_REDIST:-13.4.3}"
CUDNN_VERSION="${CUDNN_VERSION:-9.27.0}"
TENSORRT_VERSION="${TENSORRT_VERSION:-11.2.1.2}"
# Engine-building resources per GPU architecture, a few hundred MB each. PTX is the
# JIT fallback for every architecture not listed: Turing, Ampere consumer (3090),
# Ada (40xx) and Blackwell consumer (50xx) get native ones. Hopper and datacenter
# Blackwell (sm90, sm100) do not, to stay under the asset size limit.
TRT_BUILDER_ARCHS="${TRT_BUILDER_ARCHS:-ptx sm75 sm86 sm89 sm120}"

CUDA_BASE=https://developer.download.nvidia.com/compute/cuda/redist
CUDNN_BASE=https://developer.download.nvidia.com/compute/cudnn/redist
TRT_INDEX=https://pypi.nvidia.com

# The nvrtc builtins library carries the toolkit's major.minor in its name.
CUDA_MM="$(cut -d. -f1-2 <<<"$CUDA_REDIST")"

# What onnxruntime 1.27 (gpu_cuda13) and librediffusion need, by soname / DLL name.
# onnxruntime_providers_cuda: cublasLt cublas curand cufft cudart nvrtc cudnn.
# librediffusion: nvinfer and nppig (which needs nppc). TensorRT dlopen()s nvJitLink,
# nvrtc and the builder resources, and nvrtc dlopen()s its builtins.
case "$PLATFORM" in
  linux-x86_64)
    WHEEL_TAG=manylinux_2_28_x86_64
    CUDA_FILES="libcudart.so.13 libcublas.so.13 libcublasLt.so.13 libcufft.so.12
                libcurand.so.10 libnvrtc.so.13 libnvrtc-builtins.so.$CUDA_MM libnvJitLink.so.13
                libnppc.so.13 libnppig.so.13"
    CUDNN_FILES="libcudnn*.so.9"
    TRT_FILES="libnvinfer.so.11 libnvinfer_plugin.so.11 libnvonnxparser.so.11"
    TRT_RESOURCE="libnvinfer_builder_resource_%s.so.11*"
    ;;
  windows-x86_64)
    WHEEL_TAG=win_amd64
    CUDA_FILES="cudart64_13.dll cublas64_13.dll cublasLt64_13.dll cufft64_12.dll
                curand64_10.dll nvrtc64_130_0.dll nvrtc-builtins64_${CUDA_MM/./}.dll nvJitLink_130_0.dll
                nppc64_13.dll nppig64_13.dll"
    CUDNN_FILES="cudnn*64_9.dll"
    TRT_FILES="nvinfer_11.dll nvinfer_plugin_11.dll nvonnxparser_11.dll"
    TRT_RESOURCE="nvinfer_builder_resource_%s_11.dll"
    ;;
  *) echo "unsupported platform: $PLATFORM" >&2; exit 2 ;;
esac
for arch in $TRT_BUILDER_ARCHS; do
  # shellcheck disable=SC2059
  TRT_FILES+=" $(printf "$TRT_RESOURCE" "$arch")"
done

WORK="$(mktemp -d -p "${TMPDIR:-/var/tmp}")"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/cuda" "$WORK/tensorrt"
touch "$WORK/found"

# Copy the files matching one pattern (a name or a symlink to it) from an extracted
# tree into a stage directory: the real file once under its own name, and on Linux a
# relative symlink to it under the matched name and under its soname. Patterns that
# matched are recorded in $WORK/found. Returns 1 when none did.
take() {
  local tree="$1" stage="$2" pat="$3" found=1 real name soname alias
  while IFS= read -r -d '' f; do
    real="$(realpath "$f")"
    name="$(basename "$real")"
    [[ -e "$stage/$name" ]] || cp "$real" "$stage/$name"
    if [[ $PLATFORM == linux-* ]]; then
      # Only a soname that prefixes the file name: TensorRT's builder resources carry
      # a placeholder soname and are dlopen()ed by their full file name.
      soname="$(readelf -d "$real" | sed -n 's/.*(SONAME).*\[\(.*\)\]/\1/p')"
      [[ -n "$soname" && $name == "$soname"* ]] || soname=""
      for alias in "$(basename "$f")" "$soname"; do
        [[ -n "$alias" && $alias != "$name" ]] && ln -sfn "$name" "$stage/$alias"
      done
    fi
    found=0
  done < <(find "$tree" \( -type f -o -type l \) -name "$pat" -print0)
  ((found == 0)) && echo "$pat" >> "$WORK/found"
  return $found
}

fetch_extract() {
  local url="$1" dest="$2"
  mkdir -p "$dest"
  echo "fetch $url"
  case "$url" in
    *.tar.xz) curl -fsSL "$url" | tar -xJ -C "$dest" ;;
    *)        curl -fsSL -o "$dest.zip" "$url" && unzip -q "$dest.zip" -d "$dest" && rm -f "$dest.zip" ;;
  esac
}

require() {
  local pat
  for pat in "$@"; do
    grep -qxF "$pat" "$WORK/found" || { echo "error: $pat missing from $PLATFORM sources" >&2; exit 1; }
  done
}

# CUDA toolkit components, one at a time to bound the disk use.
manifest="$(curl -fsSL "$CUDA_BASE/redistrib_$CUDA_REDIST.json")"
for comp in cuda_cudart libcublas libcufft libcurand cuda_nvrtc libnvjitlink libnpp; do
  rel="$(jq -r --arg c "$comp" --arg p "$PLATFORM" '.[$c][$p].relative_path' <<<"$manifest")"
  fetch_extract "$CUDA_BASE/$rel" "$WORK/src"
  for pat in $CUDA_FILES; do take "$WORK/src" "$WORK/cuda" "$pat" || true; done
  rm -rf "$WORK/src"
done
# shellcheck disable=SC2086
require $CUDA_FILES

rel="$(curl -fsSL "$CUDNN_BASE/redistrib_$CUDNN_VERSION.json" \
       | jq -r --arg p "$PLATFORM" '.cudnn[$p].cuda13.relative_path')"
fetch_extract "$CUDNN_BASE/$rel" "$WORK/src"
take "$WORK/src" "$WORK/cuda" "$CUDNN_FILES"
rm -rf "$WORK/src"

# TensorRT: the tensorrt-cu13-libs wheel carries the runtime, the ONNX parser and the
# builder resources (including win_* ones on Linux, for cross-building: not taken).
wheel="$(curl -fsSL "$TRT_INDEX/tensorrt-cu13-libs/" \
         | grep -o "tensorrt_cu13_libs-$TENSORRT_VERSION-py3-none-$WHEEL_TAG\.whl" | head -1)"
[[ -n "$wheel" ]] || { echo "error: no tensorrt-cu13-libs $TENSORRT_VERSION wheel for $WHEEL_TAG" >&2; exit 1; }
fetch_extract "$TRT_INDEX/tensorrt-cu13-libs/$wheel" "$WORK/src"
for pat in $TRT_FILES; do take "$WORK/src" "$WORK/tensorrt" "$pat" || true; done
rm -rf "$WORK/src"
# shellcheck disable=SC2086
require $TRT_FILES

# Every library a staged one imports must be staged in one of the two packages or be
# a system library: this catches a version bump that adds a dependency.
unresolved=0
if [[ $PLATFORM == linux-x86_64 ]]; then
  sonames=" "
  for f in "$WORK"/cuda/* "$WORK"/tensorrt/*; do
    sonames+="$(readelf -d "$f" | sed -n 's/.*(SONAME).*\[\(.*\)\]/\1/p') "
  done
  for f in "$WORK"/cuda/* "$WORK"/tensorrt/*; do
    for dep in $(readelf -d "$f" | sed -n 's/.*(NEEDED).*\[\(.*\)\]/\1/p'); do
      case "$dep" in
        libc.so.*|libm.so.*|libdl.so.*|librt.so.*|libpthread.so.*|libstdc++.so.*|libgcc_s.so.*|ld-linux*|libcuda.so.1|libz.so.1) ;;
        *) [[ $sonames == *" $dep "* ]] \
             || { echo "unresolved: $(basename "$f") -> $dep" >&2; unresolved=1; } ;;
      esac
    done
  done
else
  OBJDUMP="$(command -v llvm-objdump || command -v objdump)"
  for f in "$WORK"/cuda/*.dll "$WORK"/tensorrt/*.dll; do
    for dep in $("$OBJDUMP" -p "$f" | sed -n 's/.*DLL Name: //p'); do
      case "${dep,,}" in
        cu*|npp*|nv*) [[ -n "$(find "$WORK/cuda" "$WORK/tensorrt" -maxdepth 1 -iname "$dep" -print -quit)" ]] \
                        || { echo "unresolved: $(basename "$f") -> $dep" >&2; unresolved=1; } ;;
      esac
    done
  done
fi
((unresolved == 0)) || exit 1

printf 'CUDA     %s (redist)\ncuDNN    %s (cuda13)\n' "$CUDA_REDIST" "$CUDNN_VERSION" > "$WORK/cuda/VERSIONS.txt"
printf 'TensorRT %s\nBuilder resources: %s\n' "$TENSORRT_VERSION" "$TRT_BUILDER_ARCHS" > "$WORK/tensorrt/VERSIONS.txt"

for pkg in cuda tensorrt; do
  zip="$OUT/$pkg-$PLATFORM.zip"
  rm -f "$zip"
  ls -l "$WORK/$pkg"
  # -y: store the symbolic links as links, not as copies of their targets.
  (cd "$WORK/$pkg" && zip -q -9 -y "$zip" ./*)
  rm -rf "${WORK:?}/$pkg"
  size=$(stat -c %s "$zip")
  echo "package: $zip ($((size / 1048576)) MB)"
  if ((size >= 2147483648)); then
    echo "error: $zip exceeds GitHub's 2 GiB release asset limit" >&2
    exit 1
  fi
done
