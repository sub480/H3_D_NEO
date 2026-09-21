# H3_D_NEO

面向 **ComfyUI 官方 MiniMax H3** 的多段音视频导演台。一个节点内完成时间轴编排、分段条件、MiniMax H3 采样、音视频解码、二采/放大和导出。

![H3_D_NEO 工作流截图](docs/screenshot.png)

节点类名 / 显示名：`H3_D_NEO`。底层走官方 `MiniMaxH3ImageToVideo` / `MiniMaxH3ReferenceToVideo` + `MiniMaxH3SigmaShift` + `KSampler` + AV 分离解码。

> 本项目 Fork 自 [AIMixer/ComfyUI_MiniMaxH3_Director](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director)。

## 功能概览

- **混合时间轴**：顶层模式固定为 `mixed`。添加组后，每组可单独设为 `t2v` / `i2v` / `fl2v` / `r2v` / `v2v` / `rv2v`。
- **首尾帧**：`fl2v` 可只填首帧、只填尾帧、两端都填或都不填（文生）。只填一端时不会把另一端补成同一张图。
- **参考素材组**：`r2v` / `rv2v` 每组独立管理图片 1–9、视频 1–3、音频 1–3；提示词用 `<Picture N>` / `<Video K>` / `<Audio J>`，输入 `@` 只打开本组素材菜单；图片槽支持从剪贴板直接粘贴图片。
- **源视频编辑**：`v2v` / `rv2v` 在组内上传源视频、选定范围；帧数向下对齐 MiniMax `17k+5` 网格（裁掉尾部多余帧，不补帧）；不足 5 帧的范围无效。
- **段间引导**：将上一段末尾运动和音频钉入下一段再裁掉前缀。上下文帧数 5 / 22 / 39 / 56（默认 22）。可「引导」或「引导+重绘」，以及「保完整」；单段还可「强制」引用时间轴紧邻的上一段及其缓存。
- **内置二采与放大**：同分辨率精修、像素/H3 latent 放大后再二采、或只放大 latent。高清默认空间分块。无需外接 Refine 节点。
- **选择运行与缓存**：只跑勾选组；一采缓存可复用。组卡片显示匹配/不匹配，可单组清理或清理全部（移入回收站）。
- **原生音频**：生成声音 / 使用原声 / 静音。`v2v` / `rv2v` 原声与时间轴预览同一套 24 fps 墙钟截取。
- **快照**：保存、还原、导入、导出、重命名、复制、删除当前导演台配置。
- **预览与导出**：实时 TAE 预览、分段 MP4、全部导出；可选把源画面打到独立的 `source_images` 口。
- **Director Prompt 同步/导出**：可选口接入 `minimax-h3-director-prompt/v1` JSON，工具栏「同步」刷新分组；「导出」可将当前导演台配置复制为 JSON；每组 `name` 写入卡片标题，导出时带回自定义组名，并支持传入每组的 `passMode` / `forceResample` / `continuityFromPrev` / `continuityForcePrevCache`。
- **界面语言**：工具栏可在中 / 英之间切换。

## 任务类型

顶层 `task_type` 只提供 **混合模式 (`mixed`)**。各分组类型：

| 组类型 | 说明 | UNET |
|--------|------|------|
| `t2v` | 文生音视频，无首帧/参考图 | `fl2va`（主 `model`） |
| `i2v` | 首帧图生音视频 | `fl2va` |
| `fl2v` | 首帧和/或尾帧；都不填=文生 | `fl2va` |
| `r2v` | 参考图/视频/音频生视频 | `ref2va`（建议接 `model_r2v`） |
| `v2v` | 源视频时间轴编辑，源片段绑定为 `<Video 1>` | `ref2va` |
| `rv2v` | 源视频 + 参考图/音频；源画面为 `<Video 1>` | `ref2va` |

`t2v` / `i2v` / `fl2v` 组走主 `model`（ImageToVideo / fl2va）。`r2v` / `v2v` / `rv2v` 组走可选 `model_r2v`（ReferenceToVideo / ref2va）；不接则回退到主模型。纯 ref2v 工作流仍可把 `ref2va` 接到主 `model` 口。

提示词里只有 `<Picture N>` / `<Video K>` / `<Audio J>` 会进入 tokenizer；`<Subject 1>` 只是普通文字。MiniMax H3 无反向提示词。

## 节点

当前插件只注册 **一个节点**：`H3_D_NEO`。没有独立的 Refine 节点，也没有外部 Group / Groups Combine 节点。

### 必需输入

```text
model → video_vae → audio_vae → clip
```

CLIP Loader 的 type 必须选 **`minimax`**（Qwen3-VL）。`audio_vae` 对 `r2v` / `v2v` / `rv2v` 是必需的。

### 可选输入

| 端口 | 作用 |
|------|------|
| `model_r2v` | 混合模式中 `r2v` / `v2v` / `rv2v` 组使用的 ref2va UNET；不接则回退 `model` |
| `director_prompt` | 接收 `minimax-h3-director-prompt/v1` JSON。工具栏「同步」读取上游 STRING 并刷新分组；支持每组 `durationSec`、`name`（组卡片标题）、`passMode`、`forceResample` 和 `settings.aspectRatio` 标准画幅，连接后保留同序号组的已有素材槽位。未连接时完全使用现有时间轴，执行仍以导演台时间轴为准 |
| `lora_trigger_words` | LoRA 触发词。`r2v` 追加到文末独立块 `style_tags:`；其它模式拼到每组提示词最前面 |
| `lora_trigger_words_r2v` | 仅供 `r2v` / `v2v` / `rv2v`；连接时覆盖 `lora_trigger_words`，未连接则回退 |
| `refine_model` | 二采 UNET；不接则用该段一采模型 |
| `refine_model_r2v` | 仅 `r2v` / `v2v` / `rv2v` 二采 UNET；连接时优先于 `refine_model` |
| `upscale_model` | 像素放大模型（RealESRGAN 等），仅 `upscale` + `lanczos` 时使用 |
| `live_tae_vae` | 实时预览 TinyVAE / taeh3（扫描 `models/vae_approx`）；`auto` 选用 minimax-h3/taeh3 |

一采控件（种子、步数、采样器、调度器、`shift_video` / `shift_audio`、画布、二采参数等）在节点内置面板中，不占用独立接线口。

### 输出

```text
images → audio → frame_count → source_images → images_pre_refine → refine_enabled
```

| 输出 | 说明 |
|------|------|
| `images` | 最终画面（列表）。全部导出时合并；分段导出时按段 |
| `audio` | 生成音频、源音频或静音（列表） |
| `frame_count` | 成片帧数 |
| `source_images` | 可选的源视频画面；需打开「输出原片」，且不会替换 `images`。解码失败时输出灰色占位 |
| `images_pre_refine` | 二采或放大前的一采画面 |
| `refine_enabled` | 二采勾选状态。勾选为 `true`，未勾选为 `false` |

没有 `fps`、`report` 输出口。帧率锁定 **24 fps**。

## 时间轴与分组

- 新建组默认 **5 秒**，按 24 fps 对齐到 MiniMax `17k+5` 网格（通常 **124 帧**）。可在组卡片改时长。
- 画布默认 **0.4MP、16:9（864×480）**，宽高对齐 ×32。
- **选择运行**：只采样勾选组；未勾选段不采样也不导出。
- 组可折叠、双击重命名、拖动排序；`fl2v` 可拖边缘改时长。
- 每组可设种子：默认（用一采配置）/ 随机 / 固定。一采和二采使用同一个最终种子。
- 卡片预览列可在 **采样预览** / **提示词预览** 之间切换；后者显示实际送进采样的全文（含触发词）。

### fl2v

点「添加一组」：可只写提示词，或上传首帧和/或尾帧（官方支持只传尾帧；仅首帧=图生）。开「段间引导」并勾「引用上段」时，空组用上一段末尾做运动/音频衔接。总时长 = 各组之和。

### r2v / rv2v 素材槽

每个素材组独立维护提示词和槽位，没有「公共参数」。空槽点击上传，已填槽点击预览，拖动可交换同类槽位；同一组不会重复添加同一路径。参考视频显示首帧海报；参考音频可选本地音频或视频（视频提取音轨）。`rv2v` 无参考素材时等同 `v2v`。

参考图尺寸每组可设 `match`（按输出画布面积等比缩小，只缩小不放大）、最长边 `1024` / `1280` / `1536`（先限制最长边，再使用官方 `max`）或 `max`（短边最多 2048px）。

### v2v / rv2v 源视频

- 组内上传源视频并选定范围；上传、还原、分割或拖动范围后，帧数**向下**对齐 `17k+5`（裁尾、不补帧）。
- 秒数决定生成长度；上传或替换源视频、拖动截取范围不会改写秒数。只有点击源视频右侧的秒数同步图标，才会将当前截取区间写入秒数。
- 源范围长于生成长度时只读取所需帧数；源范围短于生成长度时在采样前用最后一帧补齐。最终生成长度按 `17k+5` 对齐。
- 使用原声时只保留截取区间的音频，延长部分补静音；生成声音则覆盖完整目标时长。
- 预览条支持 **Alt + 滚轮** 缩放；播放器可拖进度、静音，并导出当前选中区间为 MP4。源图槽支持从剪贴板直接粘贴图片。
- 可追加视频到时间轴末尾。超过 ComfyUI 默认 100MB 上传限制时走分块上传。

## 段间引导

默认开启：22 帧、**引导**、重绘幅度 0.1、**保完整**关闭。第 1 组无上段，不显示「引用上段」。其后各组默认可引用上段；可在接缝处点开关硬切。

- 上下文帧数：5 / 22 / 39 / 56。
- **引导**：冻结上段尾部；**引导+重绘**：允许上段尾部参与重绘。
- **强制**：固定使用时间轴上紧邻的上一段，而不是上一段「选择运行」中的上一已选段；如果该段本次未运行，则读取其有效一采缓存。紧邻上一段没有有效缓存时直接停止并提示先运行上一段，或取消「强制」。
- **保完整**：仅对 `t2v` / `i2v` / `fl2v` / `r2v` 生效，保留对齐产生的尾部余帧，减少动作或台词被裁断；开启后成片可能略长于时间轴。`v2v` / `rv2v` 始终按源区间导出。
- 普通「引用上段」缺少上一段结果时会跳过引导并继续采样，不会中断；「强制」模式则要求紧邻上一段存在有效缓存，否则停止运行并提示处理。
- 请勿与独立的 [ComfyUI-H3-Motion-Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context) 同时安装。段间清显存常开。

## 一采、二采与放大

一采 / 二采 / 预览从输出栏打开，三个面板互斥（同时只开一个）。

### 一采默认

- `seed=42`，**25** 步，`res_multistep` + `simple`，CFG **1.0**
- Sigma shift：video **12** / audio **3**

### 二采

面板内勾选 **启用二采**（默认开；关 = 只一采）。模式：

| 模式 | 行为 |
|------|------|
| `refine` | 同分辨率二采精修 |
| `upscale` | 先放大到目标画布再二采 |
| `latent_upscale` | 只放大 H3 视频 latent，不二采 |

放大方式（`upscale`）：`h3_latent`（默认）/ `lanczos` / `nvidia_rtx_vsr`。H3 3D 权重放在 `ComfyUI/models/latent_upscale_models/`。`lanczos` 可另接 `upscale_model`。`nvidia_rtx_vsr` 需要 NVIDIA GPU 与 `nvidia-vfx`。

内置默认：`euler` + `simple`、3 步、denoise **0.35**；低噪加步默认 +1、cosine、起始 sigma 0.70。`passes` 为精修次数（`upscale` 只在第 1 次放大）。目标画布可跟随导演台、按比例+百万像素（默认 2.0MP），或自定义宽高（×32）。`skip_fl2v` 默认关；需要保护首尾帧时再勾选。

启用二采后，每组可选 **一采** / **二采**：

- **一采**：只跑第一遍；精确匹配的一采缓存则跳过
- **二采**：先一采再二采；命中一采缓存则直接二采
- 一采左侧的 **跳过缓存检查**：勾选后忽略该组一采缓存，强制重新采样；二采模式下只强制重采一采一次，不会在二采阶段重复采样
- 点状态色点可看是否匹配及差异项
- **清理**只删该组一采缓存
- 第 1 组从不钉上一段，加组或开关段间引导不会打掉组 1 的一采缓存
- 命中一采缓存时仍推送实时预览

底部运行状态例如：`一采 864×480 · 24fps` / `二采 1280×720 · 放大 · h3_latent`。

### 高清空间分块

默认开启，降低 DiT 显存：

- `refine_tile`：关 = 整幅采样
- `n_tiles`（默认 4；设 1 则整幅）
- `tile_axis`（`auto` = 较长边）
- `tile_overlap`（latent 域重叠，建议 4–8）
- 目标轴 latent 边长 ≤ `max_size_for_no_tile`（默认 64，约 480p）时自动不分块
- 可选接缝精修（默认开）

音频不切开；I2V/FL2V 关键帧随块裁切。采样逐步同步融合，并把各块 RoPE 对齐到整幅画布。

## 音频

声音下拉对 `v2v` / `rv2v` / `r2v` 以及混合模式中的对应组显示：

- **生成声音**：模型音轨（32 kHz 立体声）
- **使用原声**：`v2v` / `rv2v` 从该组源视频截取；`r2v` 使用参考音频。截取与时间轴预览同一套墙钟：`pcm_start = src_frame / 24`，不用容器 PTS
- **静音**

原声长度跟随成片画面（含段间引导裁切后的长度）。混合模式中各 `v2v` / `rv2v` 组使用自己的源视频，而不是整条全局时间轴。音频导出强制对齐 24 fps 网格。

## 预览与导出

- **预览**面板：启用开关、速度 0–1（默认 0.25；0 不显示，1 为原速 16 fps）、预览 VAE
- 视频缩略图懒加载首帧海报；二采逐步出预览；HD latent 先缩小再 TAE
- **全部导出**：合并为一个视频
- **分段导出**：每段单独落 MP4 到 `output/H3_D_NEO/segment_export/<YYYYMMDD_HHMMSS>/`，落盘后释放像素
- 「全部导出」不会静默使用未运行片段的过期缓存；请把不匹配片段加入「选择运行」，或改用分段导出

## 缓存

- 一采缓存指纹只记录提示词实际引用的图片 / 视频 / 音频槽位；未引用槽位变化不会误使该组失效
- 开启段间引导时，指纹包含 `continuity_keep_tail`；ref2v 类组使用接线后的 `lora_trigger_words_r2v`（未接则回退）
- 源视频按路径、文件大小和修改时间识别；文件被替换后旧缓存不会继续当成品
- 快照还原导致路径变化时，可按媒体内容摘要识别相同文件
- 状态扫描覆盖整条时间轴，不受「选择运行」限制
- **跳过缓存检查**的勾选状态写入时间线数据，保存/重新加载 ComfyUI 工作流时保留
- 组内「清理」只清该组一采；输出栏「清理全部缓存」清当前节点的一采、成片和音频缓存，确认后移入回收站（需要 `send2trash`）

## 快照

工具栏「快照」按需打开管理窗口（不在插件加载时创建全屏窗口，也不注册全局键盘）。Escape 仅在快照窗口内处理。

- 保存当前时间轴、素材、主要一采参数（步数、采样器、调度器、CFG、shift、seed）以及每组「跳过缓存检查」状态
- 混合模式任务下拉可在当前 mixed 工作区与已保存快照之间切换；切到快照会先缓存当前工作区
- 支持还原、重命名、复制、导入、导出、删除；还原需确认，删除移入系统回收站
- 文件在 `output/H3_D_NEO/snapshots/`，导出为 `*.mmxsnapshot.zip`
- 导入只要求文件名以 `.zip` 结尾；内容须是当前导演包格式（`pack.json` + `timeline.json`，timeline v5）
- 确认和错误显示在快照窗口内部

没有独立的「导入/导出导演包」工具栏按钮；快照 zip 内部使用同一套 pack 格式。

## 文件布局

```text
output/H3_D_NEO/snapshots/                 # 快照
output/H3_D_NEO/segment_cache/<node_id>/   # 分段一采、二采、音频缓存
output/H3_D_NEO/segment_export/<时间戳>/   # 分段 MP4
input/H3_D_NEO/uploads/                    # 上传的源视频
input/H3_D_NEO/references/                 # 参考图片、视频、音频
input/H3_D_NEO/packs/<pack-id>/            # 快照/包导入的素材
```

## 环境要求

- ComfyUI **v0.30.0 或更高**，并已包含官方 MiniMax H3 节点（[PR #15224](https://github.com/comfyanonymous/ComfyUI/pull/15224)、[PR #15228](https://github.com/comfyanonymous/ComfyUI/pull/15228)）
- 测试环境：**ComfyUI Desktop**，**未开启 Node 2.0**
- Python **3.10 或更高**
- `opencv-python-headless`：源视频读取
- `imageio-ffmpeg`：源音频提取和分段 MP4 编码
- `send2trash`：清理缓存时移入回收站（未写入 `requirements.txt`，使用「清理全部缓存」时需要）

可选：

- `nvidia-vfx`：仅 `nvidia_rtx_vsr` 放大，且需要 NVIDIA GPU  
  `pip install nvidia-vfx --extra-index-url https://pypi.nvidia.com`

## 安装

仓库：<https://github.com/sub480/H3_D_NEO>

### 手动安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/sub480/H3_D_NEO.git
pip install -r H3_D_NEO/requirements.txt
```

重启 ComfyUI。若自定义节点目录名不是 `H3_D_NEO`，以实际克隆目录为准。

### ComfyUI Manager

1. 打开 ComfyUI Manager
2. 选择 **Install via Git URL**
3. 填入 `https://github.com/sub480/H3_D_NEO.git`
4. 安装完成后重启 ComfyUI

## 模型

| 用途 | 文件名 | 目录 |
|------|--------|------|
| UNET（t2v / i2v / fl2v） | `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | `models/diffusion_models/` |
| UNET（r2v / v2v / rv2v） | `minimax_h3_ref2va_pruned_int8_convrot.safetensors` | `models/diffusion_models/` |
| CLIP | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | `models/text_encoders/` |
| Video VAE | `minimax_h3_video_vae_fp16.safetensors` | `models/vae/` |
| Audio VAE | `minimax_h3_audio_vae_fp32.safetensors` | `models/vae/` |

权重与说明：[Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)、[ComfyUI MiniMax H3 文档](https://docs.comfy.org/zh/tutorials/video/minimax/minimax-h3)。

本仓库当前不附带 `example_workflows/`。

## 快速开始

1. 确认 ComfyUI ≥ 0.30.0，已能加载官方 MiniMax H3 节点和对应模型。
2. 将本仓库放入 `ComfyUI/custom_nodes/`，安装 `requirements.txt`。
3. 添加 `H3_D_NEO`，连接 `model`、`video_vae`、`audio_vae`、`clip`。
4. 混合模式中若有 `r2v` / `v2v` / `rv2v` 组，另接 `model_r2v`（ref2va）。
5. 在导演台里添加分组、选组类型、填提示词、加素材或源视频。
6. Queue 运行。需要放大或二采时，在内置二采面板中配置。

## 相关链接

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [官方 MiniMax H3 权重](https://huggingface.co/Comfy-Org/MiniMax-H3)
- [ComfyUI MiniMax H3 文档](https://docs.comfy.org/zh/tutorials/video/minimax/minimax-h3)
- [MiniMax H3 Motion Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context)（段间引导思路参考；不要与本插件同时安装独立版）
- [H3 3D latent 放大](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler)（权重格式参考）

## 许可证

Apache-2.0，详见 [LICENSE](LICENSE)。
