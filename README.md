# HIP RT SDK

**This repository is only for tutorials. The original upstream layout expects prebuilt HIP RT SDK binaries, but this branch also includes a current `CMake + CUDA/cu-bridge` path for the basic tutorials against the in-tree HIPRT project at `/data/HIPRT`.**

HIP RT is a ray tracing library in HIP. The APIs are designed to be minimal and low level, making it easy to write a ray tracing application in HIP. We designed the library and APIs to be simple to use and integrate into any existing HIP applications. Although there are a few other ray tracing APIs which, we designed HIP RT to be simpler and easier to use, so you do not need to learn many new kernel types. 

## Features

- Ray-triangle intersection
- Ray-custom primitive intersection
- Ray mask to filter geometries
- Several bounding volume hierarchy (BVH) options
- Offline BVH load and store
- BVH import
- Motion blur

## Requirement

HIP RT runs on AMD and NVIDIA® GPUs. HIP and CUDA® APIs are dynamically loaded so you do not need to have these SDKs if your have these DLLs installed with the driver package. Hardware-accelerated ray tracing only works on RDNA™ 2 GPUs (Radeon™ RX 6000 series or newer).

The supported AMD GPU architecture families are:

- Navi3x (Radeon™ RX 7000 series)
- Navi2x (Radeon™ RX 6000 series)
- Navi1x (Radeon™ RX 5000 series)

You will also need AMD Software: Adrenalin Edition 23.30 or later for Windows and ROCm 5.7 for Linux.

----


## Directories

- [hiprt](hiprt)
  - The header and the library which you need to download from [HIP RT project page](https://gpuopen.com/hiprt/). 
- [tutorials](tutorials)
  - Tutorial source code
- [contrib](contrib)
  - External dependencies


## Building the tutorials

1. First you need to clone the repository, then init and update the submodules if you didn't clone with `--recursive`:

````
git submodule init
git submodule update
````
2. Download the HIP RT SDK from [HIP RT prject page](https://gpuopen.com/hiprt/), copy the 'hiprt' directory as explained [here](./hiprt/README.md).

### Windows:
3. Run premake to generate a solution for Visual Studio 2022:
````
cd tutorials
"../tools/premake5/win/premake5.exe" vs2022
````

4. Open the solution, compile & run.

### Linux:
3. Run premake like this on Linux, which will generate a make file:
````
cd tutorials
../tools/premake5/linux64/premake5 gmake2
````

4. Compile using the make file.
````
make config=release_x64
````

5. Run a demo.
````
cd dist
export LD_LIBRARY_PATH=../../hiprt/linux64/:$LD_LIBRARY_PATH
bin/Release/19_primary_ray64
````

These tutorials are made to run on both AMD and NVIDIA by specifying the device index. 

## Current CUDA/cu-bridge Path

For the current local HIPRT mainline in `/data/HIPRT`, this branch provides a direct CMake path for the basic tutorials:

```bash
MACA_PATH=/opt/maca \
CUCC_PATH=/opt/maca/tools/cu-bridge \
CUDA_PATH=/opt/maca/tools/cu-bridge \
PATH=/opt/maca/tools/cu-bridge/tools:/opt/maca/tools/cu-bridge/bin:$PATH \
LIBRARY_PATH=/opt/maca/tools/cu-bridge/lib:/opt/mxdriver/lib:${LIBRARY_PATH:-} \
CUCC_CMAKE_ENTRY=2 \
cmake_maca -S . -B build_basic_maca -DCMAKE_BUILD_TYPE=Release

cd build_basic_maca
make_maca -j4 \
  00_context_creation64 \
  01_geom_intersection64 \
  02_scene_intersection64 \
  03_custom_intersection64 \
  04_compaction64 \
  05_global_stack64 \
  06_dynamic_stack64 \
  07_custom_bvh_import64 \
  08_ambient_occlusion64 \
  09_motion_blur_srt64 \
  10_motion_blur_matrix64 \
  11_multi_custom_intersection64 \
  13_concurrent_scene_build64 \
  14_batch_build64 \
  15_multi_level_instancing64 \
  16_fluid_simulation64 \
  17_hiprt_hip64 \
  12_cutout64
```

Run them with:

```bash
export HIPRT_PATH=/data/HIPRT
export LD_LIBRARY_PATH=/data/HIPRT/dist/bin/Release:${LD_LIBRARY_PATH:-}
cd tutorials/dist/bin/Release
./00_context_creation64
./01_geom_intersection64
./02_scene_intersection64
./03_custom_intersection64
./04_compaction64
./05_global_stack64
./06_dynamic_stack64
./07_custom_bvh_import64
./08_ambient_occlusion64
./09_motion_blur_srt64
./10_motion_blur_matrix64
./11_multi_custom_intersection64
./13_concurrent_scene_build64
./14_batch_build64
./15_multi_level_instancing64
./16_fluid_simulation64
./17_hiprt_hip64
./12_cutout64
```

Current validated status on this branch:

- `00_context_creation64`: passed
- `01_geom_intersection64`: passed and generated `01_geom_intersection.png`
- `02_scene_intersection64`: passed and generated `02_scene_intersection.png`
- `03_custom_intersection64`: passed and generated `03_custom_intersection.png`
- `04_compaction64`: passed and generated `04_compaction.png`
- `05_global_stack64`: passed and generated `05_global_stack.png`
- `06_dynamic_stack64`: passed and generated `06_dynamic_stack.png`
- `07_custom_bvh_import64`: passed and generated `07_custom_bvh_import.png`
- `08_ambient_occlusion64`: passed and generated `08_ambient_occlusion.png`
- `09_motion_blur_srt64`: passed and generated `09_motion_blur_srt.png`
- `10_motion_blur_matrix64`: passed and generated `10_motion_blur_matrix.png`
- `11_multi_custom_intersection64`: passed and generated `11_multi_custom_intersection.png`
- `13_concurrent_scene_build64`: passed and generated `13_concurrent_scene_build.png`
- `14_batch_build64`: passed and generated `14_batch_build.png`
- `15_multi_level_instancing64`: passed and generated `15_multi_level_instancing.png`
- `16_fluid_simulation64`: passed and generated `16_fluid_simulation.png`
- `17_hiprt_hip64`: passed
- `12_cutout64`: passed and generated `12_cutout.png`

This path intentionally avoids the old Orochi/premake runtime-bitcode flow for the basic tutorials and instead uses the current HIPRT CUDA runtime build API directly.

## Introduction to the HIP RT APIs

The minimum example can be found at [tutorials/00_context_creation/main.cpp](tutorials/00_context_creation/main.cpp). On AMD platforms, you need to create a HIP context and device to initialize `hiprt` using `hiprtCreateContext` by passing `hiprtContextCreationInput` object where you need to set the native API context, device, and device type (HIP or CUDA). 

After that, use `hiprtCreateGeometry` and `hiprtBuildGeometry` to create `hiprtGeometry`. Once you have finished setting up objects on the host side, you can pass the context object to your kernel. 

An example of a minimum kernel can be found at [here](tutorials/01_geom_intersection/TestKernel.h). This is a simple HIP kernel that we are passing `hiprtGeometry ` to. To perform an intersection test, simply fill in `hiprtRay`, then create a `hiprtGeomTraversalClosest` object, then call `getNextHit()`. It is that simple.

## References 
- Introducing HIP RT – a ray tracing library in HIP, [GPUOpen Blog](https://gpuopen.com/learn/introducing-hiprt/)
- HIP Ray Tracing 2.0 introduces improved custom function handling, Bitcode linking, and more, [GPUOpen Blog](https://gpuopen.com/learn/hiprt-v2-release-optimizations-and-updates/)
- Introducing HIP RT v2.1 - batch construction for small geometries, transformation query functions, and more, [GPUOpen Blog](https://gpuopen.com/learn/hiprt_2_1_batch_construction_transformation_functions/)
- Introducing HIP RT v2.2, [GPUOpen Blog](https://gpuopen.com/learn/introducing-hip-rt-v2-2/)
