# 3DGS 演示资产清单（LumeApp）

演示 app 需要三个高斯泼溅模型。它们是**第三方样例资产**，体积很大（合计约 206 MB），
已被 `LumeApp/.gitignore` 排除（`*.glb`、`*.ply`、`*.splat`、`*.sog`），
新 clone 的仓库里没有这些文件，需要按本文件自行放入。

放置路径：

```text
LumeApp/entry/src/main/resources/rawfile/assets/
```

运行时由 `ExtractRawDir()` 解包到 app 沙箱
`.../files/assets/`，再按文件名加载（`kDemoModels[]` 配置见
[3DGS 渲染集成经验](3dgs-rendering-integration.md) 第 6 节）。

## 资产与用途

| 文件 | 格式 | 大小 | 高斯数 | 用途 |
| --- | --- | --- | --- | --- |
| `fruit_bowl.ply` | 标准 3DGS PLY（`x,y,z,rot_*,scale_*,f_dc_*,f_rest_*,opacity`）| 177 MB | 750,322 | 色彩标定基准：与 web viewer 同视角对比；SH 仅用 DC |
| `godzilla.splat` | super-splat 压缩 chunk PLY（`.splat`）| 17 MB | 1,067,182 | 压缩格式 + 大场景验证；无 SH 数据；FOV 80° |
| `doll.glb` | glTF 2.0 + `UWA_gaussian_splatting` 扩展 | 12 MB | 426,605 | 压缩格式 + VQ-SH + 环境球壳；FOV 60°，SH 全开 |
| `fruit_bowl.sog` | SOG v2（ZIP + WebP 调色板压缩）| 13 MB | 750,322 | `fruit_bowl.ply` 的压缩版，用于体积/解码对比 |
| `cube.glb` | 普通 glTF 立方体 | 1.3 KB | — | 早期 PBR 管线验证，3DGS 路径不使用 |

## 已知的量化特征（复现问题时的判据）

| 模型 | bounds（世界单位，已乘 PLY 世界缩放 100）| 质心 | r50 / r80 / r90 |
| --- | --- | --- | --- |
| fruit_bowl | (-11.2, -13.6, -15.8) ~ (20.2, 7.2, 15.8) | (2.01, -4.40, -0.18) | 11.8 / 14.0 / 15.3 |
| godzilla | (-50, -50, -50) ~ (50, 50, 50) | (-0.12, -0.66, 0.28) | 0.99 / 2.53 / 3.82 |
| doll | (-437.8, -437.8, -437.8) ~ (437.8, 437.8, 437.8) | (1.78, -1.57, 0.74) | 4.30 / 40.9 / 220.2 |

- `fruit_bowl.ply` 缩放 100 倍后主体半径约 15，但 bounds 是 ±20（房间墙面）；
  直接用 bounds 取景会把相机放进墙里。
- `godzilla.splat` 与 `doll.glb` 的 bounds 都由环境球壳主导（godzilla 半径约 50，
  doll 约 437），主体只占很小一部分。
- `doll.glb` 的环境球壳颜色是**灰白**；若渲染成粉色，说明颜色字节序或
  `feature_dc_min/max` 映射错了。
- `godzilla.splat` 的环境球壳是**粉色**；主体是约 ±3 的室内房间。
- `fruit_bowl.ply` 若 SH3 全开，画面中心的土豆会明显变暗——它的参考渲染是 DC 主导。

## 来源与许可

这些文件来自公开 3DGS 样例（压缩格式与 super-splat / UWA 工具链相关），
仅用于本仓库的渲染验证。

> **待补充**：每个资产的原始出处链接与许可证尚未记录。对外分发 HAP 或截图之前
> 必须补齐（`fruit_bowl.ply`、`godzilla.splat`、`doll.glb`、`fruit_bowl.sog` 分别填写来源、
> 作者、许可证、是否允许再分发）。

## 体积注意事项

`fruit_bowl.ply` 会被整包打进 HAP（约 177 MB），安装/升级明显变慢。
若需要做轻量演示包，可在 `LumeApp/.gitignore` 覆盖的本地副本上先裁剪
（例如抽稀到 20 万高斯），不要把裁剪产物提交进仓库。
