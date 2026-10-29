# 3DGS 在 Lume/AGP 引擎上的渲染集成经验

> 场景：在 OpenHarmony `graphic_3d`（Lume 引擎）上以原生 Vulkan 渲染 3D Gaussian Splatting，
> 目标是与 PlayCanvas `supersplat-viewer` 的网页端渲染效果对齐。
> 本文记录遇到的问题、走过的弯路与最终方案，供后续复用与排障。
>
> 按时间线整理的工作内容、决策依据与完整踩坑清单见
> [3DGS 演示实现与踩坑总结](3dgs-demo-summary.md)；资产与复现条件见
> [演示资产清单](3dgs-demo-assets.md)。

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

**根因（最终结论）**：不是"高阶噪声太大"，而是 **`f_rest` 系数顺序错位**（PLY 是通道优先，
shader 读系数优先，见 7.4）。错位后高阶项被当成别的基函数累加，才出现 12–39% 的压暗。
修正转置后，同一视角下 SH 相对 DC 的平均差只有 4.84/255（p99 20），与 SOG 的 4.85 一致，
说明系数恢复正确。

**当前策略**仍按模型区分：fruit_bowl / SOG 用 DC-only（对齐 web viewer 的实际表现），
doll 用 SH1+（对齐它自己的参考 app），godzilla 无 SH 数据。

### 4.4 最终色彩链路

```text
fragment：raw 颜色(0.5+C0·f_dc) × alpha  →  UNORM 离屏（CLEAR 为 web 背景色 (18,12,6)）
拷贝 pass：decode(texel)  →  sRGB backbuffer
            （硬件 encode(decode(x)) = x，最终显示码值 == 离屏 raw 值 == web canvas）
```

所有经验校准常数已删除。正面/侧面/背面聚合指标与 web 参考差距 ≤5%，
红薯区域逐通道差 ≤2。

## 5. 压缩 splat 格式（super-splat compressed PLY / UWA glTF）

`.splat`、GLB 内嵌的 `UWA_gaussian_splatting` 载荷都基于 super-splat 的
"压缩 PLY"：`element chunk`（每 256 个 splat 一组）＋ 4 个 packed uint32。
真实打包规则（对照 `playcanvas/splat-transform` 的 `compressed-chunk.ts`
并用数据往返验证）与我们最初的假设完全不同，错一处分就会出现
"尖刺 / shredded / 位置错乱"：

| 字段 | 真实打包 | 曾经踩的坑 |
| --- | --- | --- |
| position | `(x:11 << 21) | (y:10 << 11) | z:11`，即**低位是 z** | 误以为 10-10-10、x 在低位 |
| scale | 同 position 布局，chunk 内为 **log 空间**，解码 `exp()` | 位宽/轴序错 → 椭球尺寸与方向全错 |
| rotation | `2 bit 最大分量索引 + 3×10 bit`（其余分量按升序填 30..0 位，量化 `(a·√2/2+0.5)`），**分量顺序按写入端的 (w,x,y,z)** | 误用 4×int8；分量顺序误按 (x,y,z,w) → 定向尖刺 |
| colour | `pack8888`：字节序因导出工具而异——godzilla 为 `A,B,G,R`，doll 为 `A,B,G,B?`（实测选中布局），**alpha 恒在 byte0** | 统一按 RGBA 解读 → 品红/灰蒙蒙偏色 |

**颜色还有一层**：声明了

```
declare feature_dc_min -3.32...
declare feature_dc_max  3.01...
```

的导出文件，颜色字节存的是**归一化的 f_dc 系数**而不是最终颜色：

```
f_dc  = dcMin + n * (dcMax - dcMin)        n = byte / 255
color = 0.5 + SH_C0 * f_dc                 SH_C0 = 0.28209479
```

直接按字节当颜色会压缩约 2 倍的通道差 → 主体"灰蒙蒙"。未声明该范围的文件
（如 godzilla.splat）颜色字节即最终颜色，需按文件区分。

**SH**：GLB 变体用向量量化——`element feature_rest_vq`（每 splat 一个 uint
索引）＋ `element feature_rest`（int8 码本 × 45 系数），配合
`declare feature_rest_scale / feature_rest_zero_point` 重建
`sh[m] = codebook[idx*45+m] * scale + zeroPoint`。加载层应完整解析存入共享
SH 数组，是否参与渲染由管线侧常量决定（当前 DC-only）。

**环境壳体的薄雾**：扫描场景的外围环境球（半径 ~437）常有尺度接近主体尺寸
（数十单位）的大 gaussian，近距离观看主体时它们像半透明薄纱罩在画面上。
实测剔除 r>150 的壳体后主体饱和度 0.125→0.195、对比度 0.087→0.190。
需要按产品形态决定：完全剔除、按相机距离渐隐，或保留（远景才是完整场景外观）。

## 6. 多模型运行时切换与取景

演示 app 通过 ArkTS 按钮在三个模型间运行时切换，核心约束是**资源生命周期**与**取景鲁棒性**。

### 6.1 切换路径（跨线程归口）

```text
Index.ets Button.onClick -> lumeapp.setModel("doll")
        ↓ (只写 g_pendingModelIndex，不碰 GPU)
DrawFrame() 开头：g_modelIndex = pending; LoadSplatModel();
        ↓ 同一线程先销毁旧 RNG 再建新 RNG
```

- `setModel` 只记录意图（`g_pendingModelIndex`），真正的加载在渲染线程的 `DrawFrame` 开头执行；
  ArkTS 主线程不直接创建/释放 RNG，避免与渲染线程并发访问 GPU 资源。
- `ReleaseSplatModel()` 先清 `splatRngHandle`（`RenderHandleReference` 是引用计数，
  置空即析构 RenderNode 与其 GPU buffer/PSO），再清共享全局指针，
  最后 `splatData = SplatData{}` 释放 CPU 数组——顺序反了会让 RenderNode 持有悬空指针。
- 实测稳定性：连续 9 次切换，`VmRSS` 稳定在 258 MB、fd 数 60 不变，无泄漏。
- 切换后 `g_forceRenderFrames = 30`，保证首帧就绪；否则按需渲染会一直停在旧 buffer。

### 6.2 每模型配置（`kDemoModels[]`）

| 字段 | 作用 |
| --- | --- |
| `fovYDeg` | 垂直 FOV（godzilla 参考实现用 80°，其余 60°）|
| `shDeg1` / `shDeg23` | SH 阶数开关，经 shader 参数 SSBO 下发 |
| `hasView` + `right/up/back` | 录制视角的朝向（仅朝向，不含位置）|
| `target` | 全零表示用加权质心，否则用录制目标点 |
| `insideDistance` / `farDistance` | `<=0` 表示按主体半径推导 |

### 6.3 取景必须用鲁棒统计，不能用 bounds

原始 bounds 会被**离群点**（房间墙面、环境球壳）拉爆，直接导致相机落进几何体内部：
fruit_bowl 的 99% 高斯落在 `r < 0.2`（乘 100 世界缩放后 ≈ 20），但 bounds 是 ±20（≈ ±2000）。

`ComputeSplatFraming()` 的做法（O(n)，两次遍历 + 1024 桶直方图）：

1. 用不透明度 sigmoid 为权重求质心（比 bounds 中心更贴合主体）；
2. 算到质心的距离分布，取 `r50 / r80 / r90`；
3. 取景：近景 `distance = 1.6 * r80`，远景 `= 4 * r80`，target = 质心。

实测值：

| 模型 | 质心 | r50 / r80 / r90 | 近景 / 远景 |
| --- | --- | --- | --- |
| fruit_bowl.ply | (2.01, -4.40, -0.18) | 11.8 / 14.0 / 15.3 | 24.4（推导）/ 57.76（录制参考视角）|
| godzilla.splat | (-0.12, -0.66, 0.28) | 0.99 / 2.53 / 3.82 | 4.04 / 10.1 |
| doll.glb | (1.78, -1.57, 0.74) | 4.30 / 40.9 / 220 | 42（参考取景）/ 250 |

- fruit_bowl 的远景保留用户录制视角（`target=(3.887,-1.901,0.268)`、`distance=57.76`），
  用于与 web 截图做同视角色彩对比；近景改用质心。
- doll 的 bounds 由环境球壳（半径 437）主导，r80=40.9 仍是壳层，故近景/远景都用显式值。
- 双击在近景/远景之间切换（`farDistance` / `insideDistance`），不再依赖"上次停留距离"。

## 7. SOG v2（`.sog`）加载

`.sog` 是一个 ZIP 容器（条目为 stored，未压缩），内部是 `meta.json` + 若干 WebP 数据纹理：

| 条目 | 内容 |
| --- | --- |
| `meta.json` | `count`、`means.mins/maxs`、`scales/sh0/shN.codebook`、`shN.count/bands` |
| `means_l.webp` / `means_u.webp` | 位置低/高字节，RGB 三通道组成 16-bit 量化 |
| `scales.webp` | RGB = 256 项 log-scale 码本索引 |
| `quats.webp` | RGB = smallest-three 三个分量，A = 最大分量 tag（252..255） |
| `sh0.webp` | RGB = DC 码本索引，A = 不透明度（0..255） |
| `shN_centroids.webp` | 宽 `64*shCoeffs`、高 `count/64`，RGB = SH 码本索引 |
| `shN_labels.webp` | `R | G<<8` = 每个高斯的调色板行号 |

解码要点（对照 `playcanvas/splat-transform` 的 `read-sog.ts`）：

1. 位置在文件里是 **log 空间**：`pos = sign(v)*(exp(|v|)-1)`，其中 `v = min + span*q/65535`，
   `q = lo | hi<<8`（16-bit）。与其他模型一致还要乘世界缩放 100。
2. scale 码本是 log 值：`exp(codebook[idx]) * worldScale`。
3. 旋转是 smallest-three：`a = (byte/255*2-1)/√2`，最大分量取 `+sqrt(1-sumSq)`，
   映射表 `QUAT_IDX = [1,2,3, 0,2,3, 0,1,3, 0,1,2]`，输出 (w,x,y,z)。
4. 颜色：`color = 0.5 + SH_C0 * codebook[byte]`；不透明度直接是 `byte/255`
   （与 PLY 路径一致，shader 里不再做 sigmoid）。
5. SH rest 在调色板里是 **通道优先**（前 15 个是 R 系数），本工程的 `sh` 缓冲是
   **系数优先**（`sh[j*3+c]`），所以写入时要转置：`sh[j*3+c] = codebook[centroid_c[colBase+j]]`。

### 7.1 坑：系统解码器返回预乘 alpha

用 ImageKit NDK（`OH_ImageSourceNative_CreateFromData` + `OH_PixelmapNative_ReadPixels`）
解码这些 WebP 时，得到的是 **premultiplied** RGBA，而 SOG 把 opacity / quat tag
都存在 alpha 通道里 —— 结果是 RGB 数据被乘脏（`sh0` 的 RGB 索引整体偏小），
画面颜色几乎全黑、只剩形状。

- 现象：`sh0` 原始字节 192/179/123（alpha=37），设备上读到 27/25/17，且
  `27 = floor(192*37/255)`、`25 = floor(179*37/255)`、`17 = floor(123*37/255)`。
- 修复：读回后反预乘 `v = (p*255 + a/2)/a`（`a==0`/`a==255` 跳过）。
  量化误差实测：可见高斯（alpha≥0.25）颜色误差均值 0.009，alpha≥0.1 为 0.019。
- 平台没有提供"解码为不预乘"的开关；请求 `PIXEL_FORMAT_RGB_888` 会直接失败
  （`IMAGE_UNSUPPORTED_DATA_FORMAT`），`libwebp` 也不在 SDK/系统库中。

### 7.2 验证结果（fruit_bowl.sog vs fruit_bowl.ply，同视角）

| 指标 | SOG | PLY |
| --- | --- | --- |
| 高斯数 | 750,322 | 750,322 |
| 包围盒 | (-11.21,-13.62,-15.83)~(20.16,7.19,15.75) | 同左，z 差 0.01（量化） |
| 画面均值 | (167.1, 117.5, 54.7) | (170.3, 120.8, 56.7) |
| 画面 std | (59.5, 53.3, 43.5) | (59.5, 53.8, 44.4) |
| 逐像素平均差 | 2.98/255（1.2%），p99 = 9 | — |

体积代价：13 MB vs 177 MB。SH 调色板路径也用「权重 1 开启」验证过
（画面变化 1.9%、p99 = 20，属于正常的视角相关变化），最终配置仍用 DC
以便和 PLY 参考对比。

### 7.3 SH 开关曾被写死

`gaussian_render.vert` 里曾留有一行 A/B 测试代码
`color = mix(g.color.rgb, color, 0.0);`（commit `53a61530` 引入），
它无条件把逐模型 SH 配置变成空转——把 SH 权重调到 3.0、甚至把 SH 缓冲全填 1.0，
画面都逐像素不变。现已删除，SH 由 `shScale`（binding 4）按模型生效。

### 7.4 `f_rest` 是通道优先（已修）

标准 3DGS PLY（以及 super-splat 的 chunk `shRest`）的 `f_rest_0..44` 是**通道优先**
（0..14 为 R 系数，15..29 为 G，30..44 为 B），而 shader 按**系数优先** `sh[j*3+c]` 读取。

早期三个加载器都按文件顺序原样拷贝，导致"SH3 让红薯明显变暗 12–39%"——
其实是系数错位，不是 SH 本身的问题。

现在 `StoreShRestChannelMajorToCoefficientMajor()` 统一在拷贝时转置（标准 PLY / 压缩 VQ /
内联 float 三条路径都用它）。验证方法：同一场景用两个独立解码器各算一次 SH 增量，
结果应当一致——实测 fruit_bowl 的 PLY 与 SOG 在 SH 权重 1 下相对 DC 的平均差分别为
**4.84/255** 与 **4.85/255**（p99 都是 20），互相印证。

## 8. 性能

| 优化 | 效果 |
| --- | --- |
| 按需渲染（view 矩阵不变则跳过 RenderFrame）| 静止时 GPU 零负载，设备不再发热 |
| 视锥 + 尺寸剔除 | 旋转时顶点数减少 30-50% |
| 轴向缩放跳过重排 | 捏合缩放不再每帧排序 |
| 65536 桶排序 | 排序成本与 4K 桶基本持平（仍是两遍线性扫描）|
| 三缓冲 | 消除运动闪烁（非性能项，但避免降频抖动）|
| 排序并行化（4 线程，见下）| 拖动时 1M 高斯 87 → 28 ms、750k 65 → 25 ms |

### 8.1 排序并行化（cull / 直方图 / scatter 三段）

线程池常驻（`SortPool`，4 个 worker，主线程固定承担 task 0），每帧只在
`viewChanged` 时三段各并行一次：

1. **cull**：把高斯区间切成 `tasks` 片，每片写进自己的 `sortIndices_/sortKeys_`
   切片（互不干扰，因此不需要归并/压缩），并记录每片可见数与本片深度范围。
2. **直方图**：每片独立 65536 桶直方图，布局按桶优先（`[bucket*tasks + task]`）
   以便归约时连续读；归约、前缀和、以及"每片绝对游标"三段里，除 65536 前缀和
   外都并行。
3. **scatter**：每片用自己的游标数组写共享输出，桶内顺序与单线程版本完全一致。

实测（5 万个排序的 min/avg/max，设备：14 hw threads）：

| 模型 | 可见/总数 | 单线程 | 并行 min/avg/max | 加速 |
| --- | --- | --- | --- | --- |
| doll.glb | 19–21 万 / 426,605 | 31–41 ms | 9.5 / 17.8 / 26.9 | ~2.3× |
| fruit_bowl.ply | 62 万 / 750,322 | 61–69 ms | 16 / 25 / 40 | ~2.7× |
| fruit_bowl.sog | 62 万 / 750,322 | 68 ms | 16 / 25 / 40 | ~2.7× |
| godzilla.splat | 74–77 万 / 1,067,182 | 87 ms | 19 / 28 / 38 | ~3× |

- 冷启动首帧（缺页）仍是 22–47 ms，只在切换模型时出现一次。
- 排序结果与单线程版本**逐像素一致**（bucket 分配未变，桶内跨片顺序等于原顺序）。
- avg 与 min 差距大（cull 6 → 21 ms）说明剩余瓶颈是 CPU 调度/降频，而不是算法：
  6 worker 反而更抖，因此固定 4。

### 8.2 紧凑 cull 记录（16 B）

`cullRecords_` 在模型上传时生成一次（`px, py, pz, maxScale`，16 B/高斯），
cull pass 只读它而不是完整的 64 B 顶点记录：750k 模型 11 MB、1.07M 模型 16 MB。
实测 cull 段 22 → 18 ms（约 15%），收益有限——说明该段已接近**延迟/调度**受限，
而不是带宽受限。

### 8.3 异步排序（专用线程 + 双缓冲）

排序搬到专用线程，渲染线程只用"上一次排完的结果"（与 PlayCanvas 的 worker 方案一致）：

```text
渲染线程：viewChanged ? 快照 (view/proj/height/records) -> requestId++ : 什么都不做
         每帧：adopt 最新完成结果（互斥量内 3 MB 拷贝）-> 上传本 frame slot
排序线程：等待 requestId 变化 -> ComputeSort（内部仍用 4 worker 并行）-> 发布结果
```

- 请求**合并**：排序进行中再次请求只更新快照，worker 始终用最新相机状态。
- 首个模型的首帧仍同步排序（避免第一帧顺序错乱），之后全部异步。
- 实测渲染线程 `ExecuteFrame` 成本（曾经 = 排序耗时）：

| 模型 | 同步排序时 ExecuteFrame | 异步后 avg / max | worker 排序 |
| --- | --- | --- | --- |
| doll.glb | 30–41 ms | 0.30 / 1.4 ms | 15–17 ms |
| fruit_bowl.ply | 61–69 ms | 0.36 / 1.6 ms | 19–25 ms |
| fruit_bowl.sog | 68 ms | 0.99 / 4.9 ms | 19–25 ms |
| godzilla.splat | 87 ms | 0.92 / 4.9 ms | 24 ms |

- 代价：快速旋转时排序可能落后一次（15–25 ms），表现为极短暂的前后层次错位；
  与 PlayCanvas 行为一致，肉眼几乎不可见。
- 切模型时仍有一次同步冷排序尖峰（~285 ms，含 3 MB 上传与缺页），属预期。

### 8.4 PlayCanvas 的排序做法（对照）

`playcanvas/engine` 的 `GSplatSorter` 用**独立 Web Worker**（CPU、非渲染线程）做
计数排序：桶数自适应 `2^clamp(round(log2(n/4)),10,20)`，先用 chunk 包围球做 32 桶
粗直方图按密度分配桶空间；只按距离排序并用二分裁掉相机后方的高斯；结果延迟一帧
上传（双缓冲），所以拖动时用"上一帧排序"，渲染线程从不等待。

## 9. 调试方法论（可复用）

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

## 10. 遗留与后续方向

- ~~**SOG 格式加载**~~：已完成，见第 7 节（13 MB，与 PLY 版本画面差 1.2%）。
- **多模型**：godzilla.splat / doll.glb / fruit_bowl.ply 已在 UNORM + DC/SH + 65K 排序管线下复测，
  支持运行时切换（ArkTS 按钮）。
- **与 PC-web 观感差距**：已定位为手机平台差异（移动 GPU 精度、canvas 分辨率上限、
  浏览器色彩处理）；详见 [移动端与桌面的渲染差异分析](3dgs-mobile-vs-desktop.md)。
  如需进一步贴近，可实现 web viewer 的 grading 后处理
  （brightness/contrast/saturation/tint，默认关闭，saturation ≈1.06 即可补齐观感）。
- **性能深化**：多线程/ SIMD 排序；大场景（5M+）需 LOD 流式加载。
- **提交前**：GPU/Shader 行为变更按仓库规则应先执行
  `ohos-dev-graphics-stability-code-review`。

## 11. 代码索引

| 路径 | 说明 |
| --- | --- |
| `LumeApp/entry/src/main/cpp/render_main.cpp` | 应用侧：引擎初始化、相机（自由滚转 + 鲁棒取景）、触摸交互、模型运行时切换、按需渲染 |
| `LumeApp/entry/src/main/cpp/splat_loader.h` | 标准 3DGS PLY 与压缩 .splat 解码 |
| `LumeApp/entry/src/main/cpp/sog_loader.h` | SOG v2 解码：ZIP + 迷你 JSON + ImageKit WebP（含反预乘）|
| `LumeApp/entry/src/main/cpp/gaussian_splat_data.*` | 跨 .so 共享数据（C 链接）|
| `lume/LumeRender/src/node/render_node_gaussian_splatting.*` | 核心 RenderNode：剔除/排序/双 pass/PSO |
| `lume/LumeRender/src/node/core_render_node_factory.cpp` | 节点注册 |
| `LumeApp/entry/src/main/resources/rawfile/shaders/splat/` | splat/拷贝 shader（GLSL + .spv + .lsb + .shader/.shaderpl）|

演示模型不在仓库内（体积与许可原因），清单与放置方式见
[3DGS 演示资产清单](3dgs-demo-assets.md)。
