# 3DGS 在 Lume/AGP 引擎上的渲染集成经验

> 场景：在 OpenHarmony `graphic_3d`（Lume 引擎）上以原生 Vulkan 渲染 3D Gaussian Splatting，
> 目标是与 PlayCanvas `supersplat-viewer` 的网页端渲染效果对齐。
> 本文记录遇到的问题、走过的弯路与最终方案，供后续复用与排障。

## 1. 最终架构

```
PLY / .splat 文件
   │  应用侧解码（liblumeapp.so）
   ▼
共享 GPU 数据（跨 .so，dlsym 交换指针）
   ├── gaussian SSBO        64B/gaussian（位置/scale/opacity/四元数/DC 颜色）
   ├── SH SSBO              48 floats/gaussian（f_rest，当前不参与渲染）
   ├── 排序索引 SSBO ×3      按相机深度排序的 gaussian 索引（三缓冲）
   └── 相机 UBO ×3            view/proj/viewport/cameraPos（三缓冲）
   ▼
RenderNodeGaussianSplatting（lume/LumeRender）
   ├── CPU：视锥剔除 + 尺寸剔除 + 65536 桶计数排序（远→近）
   ├── Pass 1：splat → UNORM 离屏（raw 码值空间混合，web canvas 语义）
   └── Pass 2：全屏拷贝 → sRGB backbuffer（decode 抵消硬件 encode）
```

关键设计取舍：

| 决策 | 原因 |
| --- | --- |
| 排序放 CPU（65536 桶计数排序） | 数据本就在 CPU；只在相机变化帧执行；桶粒度足够时效果等同精确排序 |
| UNORM 离屏 + decode 拷贝 | sRGB 附件硬件做线性空间混合（Jensen 暗化、视角相关色偏）；UNORM 原始码值混合才等价 web canvas |
| DC-only 颜色（SH 阶 0） | 见 §4.3；与 web 端实际表现一致 |
| 三缓冲（相机 UBO / 排序索引） | 相机运动时 CPU 覆写与 GPU 在途帧竞争，导致运动闪烁 |
| 参考视角固化 | 调试期固定启动视角，使截图对比有效 |

## 2. Lume 引擎集成要点（踩坑清单）

### 2.1 Shader 系统的 `.spv.lsb` 反射数据

Lume 的 `ShaderLoader` 要求每个 `.spv` 配套 `.spv.lsb`（自定义二进制反射格式），
否则 `ShaderModuleVk` 构造失败（`invalid vulkan shader module`）。
LumeShaderCompiler 可用但要编 75MB 三方依赖；**本地用脚本直接生成**：
参考 `lume/LumeRender/src/device/shader_reflection_data.cpp` 的二进制格式
（`'r','f','l',1` 头 + push constant / spec constant / descriptor set / input / local size 分节），
每个 shader 的 descriptor 布局按 GLSL 手写。改动 binding 时必须同步：
GLSL 声明 → `.lsb` → `.shaderpl`（pipeline layout JSON）。三者任一处不一致都会静默失败或绑定错乱。

### 2.2 RenderNode 注册与生命周期

- 新节点需在 `core_render_node_factory.cpp` 注册 `Create`/`Destroy` 与 UID。
- RNG（Render Node Graph）由应用侧 `IRenderNodeGraphManager::Create` 动态创建，
  追加在场景 RNG 之后执行（splat pass 用 LOAD/贴回 backbuffer）。
- **模型必须在首帧 TickFrame 之后加载**：`RenderSystem::Initialize()` 在首次 tick 注册事件监听，
  提前加载会让 mesh 事件永久丢失（glTF 场景的坑，splat 无此问题但保留顺序避免回归）。

### 2.3 相机与矩阵约定

- Vulkan 裁剪空间：Z ∈ [0,1]；投影矩阵 `out[5] = -f`（Y 翻转）。
- 本工程 splat 相机为自由滚转（right/up/back 基向量 + Rodrigues 旋转），
  **view 矩阵第三行为 `(back, -back·eye)`**：写成 `(-back, +back·eye)` 会整体镜像 Z，
  表现为所有 gaussian 被剔除或场景消失（曾发生，排查花了不少时间）。
- 主机端视锥剔除与 shader 内的 `z` 判定必须与投影矩阵同一套符号约定。

### 2.4 描述符集管理

- `INodeContextDescriptorSetManager::ResetAndReserve` **会重置整个池**。
  多个 binder（例如 splat pass + 拷贝 pass）必须用
  `ResetAndReserve(array_view<DescriptorCounts>)` **一次性合并预留**，
  否则后一次调用会破坏先前 binder 的集合（症状：splat 不绘制、验证层报 `sizes must match`）。
- 管线（PSO）按 render pass 兼容性惰性缓存；更换附件格式/深度状态会自动生成新管线，无需手动管理。

## 3. 深度排序与混合

### 3.1 排序策略演进

| 阶段 | 方案 | 结果 |
| --- | --- | --- |
| v1 | 每帧全量 256 桶排序 | 可用，但深堆叠区域出现色偏 |
| v2 | 4096 桶 + 相机变化阈值（0.25°）| 旋转更顺，但**背面视角出现绿色条带** |
| 最终 | **65536 桶**（切片 <1mm），纯轴向缩放跳过重排 | 绿色像素 0.214% → 0.022%（web 浏览器基准 0.013%）|

**教训**：桶内保持稳定散列（等价于按 gaussian 索引序混合）在半透明深堆叠中会累积混合顺序误差。
粗桶粒度下误差足以产生可见的暗区条带/色偏。65536 桶的计数排序仍是 O(n)，成本增加可忽略。

### 3.2 三缓冲与哨兵填充

相机运动时 CPU 每帧覆写 UBO/排序索引，而 GPU 仍有 1-2 帧在途——
读端拿到撕裂数据，表现为运动闪烁。解决：

- 相机 UBO 与排序索引各 3 份，按帧轮转（swapchain 3 图，2 帧在途 + 1 帧写入）；
- 排序索引尾部用 `0xFFFFFFFF` 哨兵填充，shader 检测到即跳过，兜底可见数量收缩时的越界读取。

### 3.3 CPU 剔除

排序遍历顺带完成：view-space z 范围剔除、视锥剔除（含世界半径 3σ 余量，避免近处大 splat 被误杀）、
屏幕尺寸剔除（<2px，匹配 web 的 `minPixelSize`）。剔除后仅绘制可见子集，旋转时通常可省 30-50% 顶点。

## 4. 色彩管线（本项目最大的坑）

### 4.1 参考实现的关键事实（PlayCanvas 源码确认）

- splat 颜色是**gamma 空间（显示就绪）值**：`0.5 + SH_C0 * f_dc`，**不做 linear 转换**；
- 混合为**码值空间 premultiplied alpha**（8bit canvas 直接混合，无 decode/encode）；
- `prepareOutputFromGamma` 在默认 tonemapping=linear 下为恒等变换；
- falloff 为 `normExp(A) = (exp(-4A) - exp(-4)) / (1 - exp(-4))`，quad 尺寸 `2√(2λ)`（≈2.83σ）；
- `minPixelSize = 2.0`、无 aaFactor（`GSPLAT_AA` 非默认）、`alphaClipForward = 1/255`；
- SH 方向 = 相机→gaussian（模型空间），基函数与 f_rest 布局（系数优先、通道交错）与 INRIA 一致。

### 4.2 走过的弯路（全部废弃）

| 尝试 | 为什么错 |
| --- | --- |
| 经验 gamma（1.5→1.8→2.0）+ 暗部提升曲线 | 单视角拟合；换视角失效，且在暗堆叠上扭曲色相（紫色问题根源）|
| 在 sRGB 附件上输出 `decode(L)` | 硬件在线性空间混合，深堆叠 Jensen 暗化，视角相关偏差 |
| 通道增益（0.995/0.942/0.76）| 仍是经验校准；UNORM 方案落地后全部删除 |
| SH 方向翻转（OUT）| 双视角趋势分析证伪（蓝色趋势方向相反）|
| aaFactor 抗锯齿补偿 | web 默认未启用；启用后小 splat alpha 被压 30-45%，阴影过暗 |
| 绿色条带归因截图 JPEG 伪影 | 手机浏览器同内容无条带 → 证实是渲染侧的排序问题 |

### 4.3 SH 阶数：从"红薯发灰"到根因

症状：正面红薯在 web 端红润，本引擎发灰发暗。

排查：对 web 截图框选红薯区域做色相分解对比（红/橙/黄分段统计），
分别在 SH3 / SH1 / DC 三个构建下测量同一区域：

| 构建 | 红薯区域 RGB | 结论 |
| --- | --- | --- |
| SH3 | (123,65,41) | 偏暗 |
| SH1 / DC | (140,90,67) | 与 web (142,89,67) 几乎一致 |

**根因**：模型 degree 2/3 的 `f_rest` 系数是训练残留噪声（均值 ≈0.03），
但高阶基函数乘数很大（C3 max 2.89），15 个系数累积出 ±0.2 的可观颜色摆动；
web viewer 实际只渲染 SH ≤ 1 阶。方案：**DC-only**（`SH_DEG1_SCALE/SH_DEG23_SCALE = 0`），
颜色与视角无关，同时解决"旋转时颜色变化明显"。

### 4.4 最终色彩链路

```text
fragment：raw 颜色(0.5+C0·f_dc) × alpha  →  UNORM 离屏（CLEAR 为 web 背景色 (18,12,6)）
拷贝 pass：decode(texel)  →  sRGB backbuffer
            （硬件 encode(decode(x)) = x，最终显示码值 == 离屏 raw 值 == web canvas）
```

所有经验校准常数已删除。正面/侧面/背面聚合指标与 web 参考差距 ≤5%，
红薯区域逐通道差 ≤2。

## 5. 性能

| 优化 | 效果 |
| --- | --- |
| 按需渲染（view 矩阵不变则跳过 RenderFrame）| 静止时 GPU 零负载，设备不再发热 |
| 视锥 + 尺寸剔除 | 旋转时顶点数减少 30-50% |
| 轴向缩放跳过重排 | 捏合缩放不再每帧排序 |
| 65536 桶排序 | 排序成本与 4K 桶基本持平（仍是两遍线性扫描）|
| 三缓冲 | 消除运动闪烁（非性能项，但避免降频抖动）|

## 6. 调试方法论（可复用）

1. **同视角对比**：固定参考视角（相机状态日志 → 硬编码为启动默认），
   用 `snapshot_display` 抓原始 framebuffer；对比图不受"重启后视角回默认"干扰。
2. **区域级统计**：全图均值会稀释局部差异。让用户框选关键物体（如红薯），
   按色相分段（红/橙/黄）统计 RGB/饱和度/亮度。
3. **色相分布**：直方图能区分"整体偏色"与"局部物体错误"。
4. **周期检测**：绿色条带通过列方向自相关确认周期性（发现其与排序桶粒度的关系）。
5. **对照组**：手机浏览器运行同一 web viewer，与原生 app 在同一设备同屏对比，
   排除显示器/截图链路的干扰——本项目的关键转折点。
6. **A/B 构建**：把可疑机制做成 shader 常数（SH 阶数、falloff 变体），
   单次部署验证一个假设。

## 7. 遗留与后续方向

- **SOG 格式加载**（`fruit_bowl.sog`）：开源精简格式，字典/调色板解压。
- **godzilla.splat / doll.glb**：在新管线（UNORM + DC + 65K 排序）下复测。
- **与 PC-web 观感差距**：已定位为手机平台差异（移动 GPU 精度、canvas 分辨率上限、
  浏览器色彩处理）；详见 [移动端与桌面的渲染差异分析](3dgs-mobile-vs-desktop.md)。
  如需进一步贴近，可实现 web viewer 的 grading 后处理
  （brightness/contrast/saturation/tint，默认关闭，saturation ≈1.06 即可补齐观感）。
- **性能深化**：多线程/ SIMD 排序；大场景（5M+）需 LOD 流式加载。
- **提交前**：GPU/Shader 行为变更按仓库规则应先执行
  `ohos-dev-graphics-stability-code-review`。

## 8. 代码索引

| 路径 | 说明 |
| --- | --- |
| `LumeApp/entry/src/main/cpp/render_main.cpp` | 应用侧：引擎初始化、相机（自由滚转 + 参考视角）、触摸交互、按需渲染 |
| `LumeApp/entry/src/main/cpp/splat_loader.h` | 标准 3DGS PLY 与压缩 .splat 解码 |
| `LumeApp/entry/src/main/cpp/gaussian_splat_data.*` | 跨 .so 共享数据（C 链接）|
| `lume/LumeRender/src/node/render_node_gaussian_splatting.*` | 核心 RenderNode：剔除/排序/双 pass/PSO |
| `lume/LumeRender/src/node/core_render_node_factory.cpp` | 节点注册 |
| `LumeApp/entry/src/main/resources/rawfile/shaders/splat/` | splat/拷贝 shader（GLSL + .spv + .lsb + .shader/.shaderpl）|
