# HIPRTSDK MACA 适配对比与整理

## 1. 当前基线

- 当前私有主线提交：`ebf0c7d`
- 当前私有里程碑 tag：`maca_tutorials_all_green`
- 当前本地可确认的官方共同祖先：`2442d0d`
  - `Merge pull request #49 from GPUOpen-LibrariesAndSDKs/feature/HRT-0-03000`

说明：

- 本地尝试通过网络拉取官方最新 `main` 指针时，遭遇 GitHub `HTTP2 framing layer` 错误。
- 因此本次“官方对比”以 **当前仓库里可确认的官方共同祖先 `2442d0d`** 为基线做代码审计。
- 这已经足够用于判断“哪些修改是 MACA 必要适配、哪些是私有环境约束、哪些可以继续收敛”。

## 2. 当前修改的本质

从 `2442d0d..main` 的差异看，当前修改本质上不是“给官方教程补一些零散 patch”，而是做了 3 层重构：

1. **构建系统层**
   - 新增根级 `CMakeLists.txt`
   - 让教程可以直接对接当前 `/data/HIPRT` 主线
   - 让教程可以在 `MACA + cu-bridge` 下通过 `cmake_maca + make_maca` 构建

2. **运行时抽象层**
   - `TutorialBase` 从旧的 `Orochi + orortc + runtime bitcode` 路径
   - 收敛为当前：
     - CUDA driver/runtime API
     - `hiprtBuildTraceKernels(...)`
     - 必要时外部 `cucc/nvcc -> cubin` fallback

3. **教程 kernel 层**
   - 把原来集中在 `TutorialKernels.h` 里的大量 kernel 拆成多个最小头文件
   - 目的不是“代码风格整理”
   - 而是为了降低每个教程的兼容面，避免一个大头文件里的历史代码把所有教程一起拖死

## 3. 修改分类

### 3.1 明确必要，且适合保留

这些修改是当前 MACA 路径真正需要的，删除后会直接导致教程回退到不可编译或不可运行状态：

- 根级 `CMakeLists.txt`
  - 这是当前教程从旧 premake/SDK 二进制布局切到主线 HIPRT 的基础入口

- `TutorialBase.h/.cpp`
  - 去除对旧 `Orochi/orortc` 教程路径的强依赖
  - 改为当前 CUDA/HIPRT 主线调用方式
  - 包括：
    - CUDA context/device 初始化
    - `hiprtBuildTraceKernels(...)`
    - 外部 cubin fallback

- `SceneDemo.h/.cpp`
  - 这是 `18_shadow_ray`、`19_primary_ray` 能在当前主线下工作的必要桥接层

- 各类最小 kernel 头
  - `BasicTutorialKernels.h`
  - `CustomIntersectionTutorialKernels.h`
  - `CutoutTutorialKernels.h`
  - `CornellTutorialKernels.h`
  - `CustomBvhImportTutorialKernels.h`
  - `SceneBuildTutorialKernels.h`
  - `MotionBlurTutorialKernels.h`
  - `MultiCustomTutorialKernels.h`
  - `FluidTutorialKernels.h`

这些文件的价值不是“为了拆分而拆分”，而是：

- 把教程按功能域拆开
- 避免旧 `TutorialKernels.h` 中的历史兼容代码互相污染
- 让每个教程的适配边界更清楚

### 3.2 当前必要，但更适合后续继续收敛

这些修改当前是有效的，但从长期维护看还可以继续优化：

- `TutorialBase.cpp` 中的外部 cubin fallback
  - 这是当前 `cu-bridge` 环境下绕过 runtime JIT 不稳定的有效办法
  - 但它属于“教程仓库里的宿主侧兜底逻辑”
  - 如果未来 HIPRT 主线或 cu-bridge 进一步稳定，理想状态是尽量减少这部分分叉逻辑

- `README.md` 中对 `/data/HIPRT` 的默认路径描述
  - 对当前环境是必要的
  - 但如果考虑更通用的仓库可移植性，后续可以改成更参数化的表述

- `17_hiprt_hip`
  - 当前为了保证示例“可运行”，已经被转成当前 CUDA/MACA 路径的 context creation 示例
  - 从“教程语义命名”看，它已经不再准确代表原始 HIP-only 教程意图
  - 如果要长期保留，建议后续二选一：
    - 重命名
    - 或恢复一个真正独立的 HIP-only 分支说明

### 3.3 不建议上游化，或至少不应原样上游化

这些不是“错误修改”，但更像私有环境适配，不适合原样合回官方：

- README 中直接写死 `/data/HIPRT`
- 任何默认依赖 `MACA_PATH=/opt/maca`、`CUCC_PATH=/opt/maca/tools/cu-bridge`
- 以当前私有验证环境为前提的命令示例

这类内容适合作为：

- 私有分支文档
- 或通过环境变量/脚本参数控制

而不应直接作为官方仓库默认行为

## 4. 是否存在明显不必要修改

回头看当前差异，本轮没有发现“已经确认无价值、应该直接回退”的核心代码改动。

原因是：

- 当前每一类修改最终都被至少一个实际可运行教程覆盖到了
- 不是只停留在“编译过”或“猜测需要”

更准确的结论是：

- **当前没有明显应立即回退的主路径改动**
- 但存在一些后续可以继续美化或参数化的实现方式

## 5. 提交整理建议

当前提交链已经基本按功能域展开：

1. `e99d410`
   - 基础教程适配起点
2. `4b55c4b`
   - custom intersection
3. `e652916`
   - cutout
4. `094b69b`
   - compaction / stack / ambient occlusion
5. `8f09041`
   - custom bvh import / cornell 组
6. `62b36ba`
   - scene build 组
7. `b80e4f1`
   - motion blur / multi custom / 17
8. `ba344e5`
   - fluid simulation
9. `ebf0c7d`
   - SceneDemo/OBJ 教程收尾

如果后续要准备上游化或做 patch 精简，建议进一步整理成下面 4 组：

1. `build-system`
   - 根级 CMake
   - `.gitignore`
   - 文档中新的构建入口

2. `runtime-adapter`
   - `TutorialBase.*`
   - `SceneDemo.*`

3. `kernel-split`
   - 各个最小 kernel 头

4. `tutorial-migrations`
   - 各 tutorial `main.cpp`

这样会比当前按示例逐步推进的提交链更适合代码审查。

## 6. 当前结论

可以把当前状态总结成一句话：

- **当前私有分支相对官方共同祖先 `2442d0d` 的修改总体是必要且有效的；其中“教程运行时适配层”和“最小 kernel 头拆分”是核心必要修改，环境路径和部分命名则属于私有化实现细节，后续可继续参数化和清理，但当前没有发现需要立即回退的关键逻辑改动。**
