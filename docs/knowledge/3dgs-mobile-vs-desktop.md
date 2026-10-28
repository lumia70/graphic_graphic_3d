# 3DGS：移动端与桌面的渲染差异分析

> 现象：同一模型、同一 web 渲染器（PlayCanvas supersplat-viewer），
> 手机端渲染的色彩明显不如 PC 端，缩放/旋转的流畅性也更差。
> 本文按平台机制拆解原因，并给出可验证的证据与应对方向。
>
> 相关文档：[3DGS 渲染集成经验](3dgs-rendering-integration.md)

## 1. 色彩差异的来源

### 1.1 着色器精度档位（mediump = fp16）

PlayCanvas gsplat 着色器的插值变量声明为 `mediump`：

```glsl
varying mediump vec2 gaussianUV;      // 高斯衰减求值
flat varying mediump vec4 gaussianColor;
```

- 桌面 GPU：mediump 通常被提升为 fp32，等价于全精度；
- 移动 GPU（Maleoon / Adreno / Mali）：mediump **实际是 fp16**。

后果：每片元的高斯衰减与颜色混合在半精度下计算，色阶过渡、暗部细节与
累积混合结果存在可测量的差异；深堆叠暗区尤其明显。

### 1.2 画布分辨率平台上限

viewer 的 `quality.ts` 对画布像素密度设有平台硬上限：

```ts
const canvasPixelRatio = () => {
    const maxPixelDim = platform.mobile ? 1080 : 2160;
    return Math.min(maxPixelDim / Math.min(screen.width, screen.height), window.devicePixelRatio);
};
```

- 移动端：画布短边最多 1080 px；
- 桌面端：最多 2160 px。

桌面端线性采样密度约为移动端 2 倍（像素数 4 倍），微观细节与色彩过渡更细腻。
移动端画布还可能被系统合成器缩放显示，插值会轻微降低锐度与饱和度。

### 1.3 质量档位联动降级

`quality.ts` 中移动端使用独立的资源阶梯：

```ts
const CEILING_MB = { mobile: 1100, standaloneHeadset: 1900, desktop: 2600 };
const COLOR_UPDATE_ANGLE = 0.2;                 // 颜色烘焙刷新阈值（度）
const COLOR_UPDATE_ANGLE_PERFORMANCE = 1;       // 低档位更粗
```

低质量档位会：
- 降低 SH 颜色的刷新频率（烘焙缓存按相机转角阈值更新）；
- 降低 splat 预算（更早进入 LOD / 简化数据）；
- 可能截断 SH 阶数、降低排序键精度——这与我们观测到的
  "web 端实际表现接近 SH ≤ 1 阶"一致。

### 1.4 浏览器色彩链路差异

- OHOS 浏览器与桌面 Chrome 的色彩空间处理（sRGB / P3）与合成路径不同；
- 移动端 canvas 到屏幕之间多一层浏览器/系统合成器；
- 移动面板本身的色域与出厂调校通常弱于桌面显示器
  （仅在跨设备对比时存在；同屏对比截图可排除此项）。

## 2. 流畅性差异的来源

### 2.1 GPU 填充率与带宽

750K splat 的渲染是 overdraw / fillrate 密集型负载：

- 每个 splat 的光栅化面积随缩放平方增长（捏合放大时尤为明显）；
- 移动 GPU 的填充率与显存带宽通常只有桌面 GPU 的几十分之一；
- LPDDR 带宽 vs GDDR 带宽差距进一步放大混合写入成本。

### 2.2 每帧 CPU 排序

web 渲染器在每次相机变化时于 Web Worker 中执行全量排序：

- 移动 CPU 单核性能与缓存容量显著弱于桌面；
- 缩放/旋转期间几乎每帧触发排序 → 排序时间直接进入帧预算；
- 桌面端排序耗时相对帧预算可忽略，感官上"无感"。

### 2.3 运行时开销

- 浏览器 JS 引擎的 GC、worker 消息传递、typed array 上传（排序结果每帧重传）；
- WebGL/WebGPU 命令提交经过浏览器进程边界；
- 移动端 QoS / 调度策略更激进，后台线程与主线程争抢。

### 2.4 热与 DVFS

持续满负载渲染触发移动端温控降频，长时间操作帧率逐步下降。
这也是本项目采用"按需渲染"（相机静止即停止 RenderFrame）的直接原因。

## 3. 如何验证差异归属

| 实验 | 结论指向 |
| --- | --- |
| 手机浏览器与原生 app 同屏对比 | 二者接近 → 差异来自移动平台，而非原生实现 |
| 两端截图放在同一显示器对比 | 排除显示面板因素，差异存在于像素数据本身 |
| 关后处理（`?nofx`）对比 | 排除 viewer 端后处理影响 |
| 检查 viewer `quality` / `URL 参数` | 确认是否处于低质量档位 |

本项目实测结论：手机浏览器渲染 ≈ LumeApp 原生渲染；
PC web 与二者存在系统性差异。即：**实现层已对齐，观感差距源于移动平台**。

## 4. 应对方向

1. **原生优势**：去掉浏览器/JS 层开销（GC、worker、命令提交、合成器），
   移动端原生 Vulkan 在同硬件下帧时间更稳定；本项目已实现按需渲染以抑制发热。
2. **填充率优化**：视锥 + 尺寸剔除（已实现）、LOD / splat 预算、
   低分辨率离屏 + 上采样、tile-based 或 stochastic 渲染。
3. **排序优化**：CPU 多线程/SIMD 或 GPU radix 排序；相机静止与纯轴向缩放跳过重排（已实现）。
4. **色彩观感**：如需向桌面观感靠拢，可实现 viewer 的 grading 后处理
   （saturation / contrast，默认关闭）作为可调选项，而非硬编码进基础管线。

## 5. 参考

- `playcanvas/supersplat-viewer`：`src/quality.ts`（画布上限、质量阶梯、颜色刷新阈值）、
  `src/schemas/defaults.ts`（默认设置，后处理默认关闭）。
- `playcanvas/engine`：`src/scene/gsplat/`（material、resource、instance、sorter）、
  `src/scene/shader-lib/glsl/chunks/gsplat/`（着色器，mediump 声明）。
