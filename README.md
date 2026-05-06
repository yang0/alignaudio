# AlignAudio

将配音音频（多个 WAV 片段）逐段对齐到原始视频，自动调整视频/音频速度匹配时长。支持 **RIFE GPU 运动插帧**、**SRT 间隙检测**、**混合速度调整策略**。

## 核心流程

```
SRT 字幕时间轴
  │
  ├─ 间隙检测 (gap > 0.01s) → 插入静音音频段 + 保留原视频
  │
  └─ 逐段处理
       │
       ├─ |ratio-1| ≤ 0.20 → astretch：纯音频拉伸，视频零损耗
       │
       ├─ |ratio-1| > 0.20 → bidirectional：双向分担（视频+音频各 √ratio）
       │                       │
       │                       ├─ 减速 √ratio > 1.15 → RIFE GPU 运动插帧
       │                       │ 减速 √ratio 0.85~1.15 → setpts 帧复制
       │                       └─ 加速 √ratio < 0.85   → framerate 混合
       │
       └─ 每段合并：精确截断到公共时长（消除 PTS 缝隙）
            │
            └─ concat filter（非 demuxer）合并全部段
```

## 安装

### 1. 基础依赖

```bash
pip install -e .
# 需要 ffmpeg/ffprobe in PATH
```

### 2. RIFE GPU 插帧（可选，但推荐）

```bash
# 克隆 RIFE 模型代码
git clone --depth 1 https://github.com/hzwer/ECCV2022-RIFE.git ECCV2022-RIFE

# 下载预训练模型 v4.26 (HuggingFace)
python -c "
import urllib.request, zipfile, shutil, glob
url = 'https://huggingface.co/hzwer/RIFE/resolve/main/RIFEv4.26_0921.zip'
urllib.request.urlretrieve(url, 'model.zip')
zipfile.ZipFile('model.zip').extractall('train_log')
for f in glob.glob('train_log/RIFEv4.26_0921/*'):
    shutil.move(f, 'train_log/')
os.remove('model.zip')
"
```

### 3. 测试数据

`test/` 目录包含：
- `video_no_audio.mp4` — 原始无音轨视频
- `video_zh.srt` — 中文字幕（321 条时间戳）
- `dubbed_output/0001.wav ~ 0321.wav` — 中文配音片段

## CLI 命令

```bash
# 主流程：视频配音对齐（完整管道）
align-video <字幕.srt> <视频.mp4> <配音目录> <输出.mp4> [选项]

# 纯音频对齐
align-audio <字幕.srt> <配音目录> <输出.wav>

# 字幕时间戳调整（根据配音时长重算）
adjust-subs <字幕.srt> <配音目录> <输出.srt>
```

### 选项

| 参数 | 默认 | 说明 |
|---|---|---|
| `--workers N` | 4 | 并行线程数 |
| `--resume` | — | 从 checkpoint 断点续跑 |
| `--no-audio-stretch` | — | 禁用音频拉伸（纯视频调整） |
| `--scene-snap` | — | 将片段边界吸附到场景切换点 |

### 示例

```bash
# 30 段测试
align-video test/video_zh.srt test/video_no_audio.mp4 test/dubbed_output test/output.mp4 --workers 4

# 完整 321 段 + 断点续跑
align-video test/video_zh.srt test/video_no_audio.mp4 test/dubbed_output test/final.mp4 --workers 4
```

## 速度调整策略

### 混合策略

| speed_ratio 范围 | 变化幅度 | 策略 | 视频损耗 |
|---|---|---|---|
| 0.80 ~ 1.20 | <20% | **astretch**: 纯音频拉伸（rubberband）| **零损耗** |
| 0.85~0.80 / 1.15~1.20 | 15-20% | bidirectional: setpts + 音频拉伸 | 中 |
| > 1.15 (bidirectional) | 减速 >15% | **RIFE GPU 插帧** + 音频拉伸 | 极低 |
| < 0.85 | 加速 >15% | framerate 混合 + 音频拉伸 | 低 |

### x264 编码预设

所有编码环节统一使用 `-preset fast -crf 18`，保证画质。

## 解决的关键问题

| 问题 | 原因 | 修复 |
|---|---|---|
| 静帧（视频冻住） | concat demuxer PTS 不连续 | → **concat filter** 替代 demuxer |
| 视频片段丢失 | SRT 间有空隙未保留 | → **间隙检测**，插入静音段 |
| 音视频不同步 | `-shortest` + 采样率不一致 | → 统一 16kHz + 精确 `-t` 截断 |
| 像素模糊 | `-preset ultrafast` 反复编解码劣化 | → **`-preset fast`** |
| setpts 减速帧卡顿 | 帧复制肉眼可见 | → 双向分担 + RIFE 插帧 |
| concat 丢视频流 | mp4v + h264 混合不兼容 | → concat filter 统一重编码 |

## 犯过的错误（Don't Repeat）

1. **先猜再测**：修改前不先定位根因就开始改代码。应该先跑最小复现例 → 读 log → 定位 → 再改。
2. **一改一大片**：一次改多个变量（scale + 阈值 + concat），出问题不知道哪个是原因。应该每次只改一个变量。
3. **忽略 ffmpeg 版本差异**：系统有 ffmpeg 6.1 和 8.1 两个版本，不同版本对 concat demuxer 的严格程度不同。应该固定版本。
4. **ThreadPool 并行 + CUDA 共享隐患**：多线程共享同一个 GPU 模型时，stderr 日志交叉输出干扰调试。应该用 `threading.Lock` 保护打印。
5. **阈值改完不验证边界**：改 RIFE 阈值从 1.15→1.08 后，没验证边界段（ratio≈1.08 的段）表现如何。应该每改一次阈值就跑一次边界测试。
6. **cv2.VideoWriter 的 fps 参数不可靠**：高 fps(>50) 写入时 mp4v 编码可能不按指定 fps 输出。应该始终用 ffmpeg 校验输出时长。
7. **过早优化**：先写 gap 插入和并行调度，但基础音视频对齐还没稳定。应该先跑通基础路径，再加功能。
8. **concat demuxer ≠ concat filter**：demuxer 对输入流的编码一致性要求高，filter 更健壮。优先用 filter。
9. **Windows 路径问题**：`-c:v copy` 输出 AV1（原视频编码），与 h264 混用 concat 失败。所有段应统一编码。

## 文件结构

```
alignaudio/
├── align_video_to_dubbing.py   # 主管线（完整管道 + 混合策略）
├── align_video_dubbing.py      # 备选管线（带字幕嵌入）
├── align_dubbed_audio.py       # 纯音频对齐
├── adjust_subtitle_timestamps.py # 字幕时间戳重算
├── rife_interpolator.py        # RIFE GPU 流式插帧模块（scale=1.0）
├── preview_3seg.py             # 快速预览
├── pyproject.toml              # 包配置 + CLI 入口
├── ECCV2022-RIFE/              # RIFE 模型代码（手动克隆）
├── train_log/                  # RIFE 模型权重（手动下载）
└── test/                       # 测试数据
```

## 依赖

- **必需**: Python ≥ 3.10, ffmpeg/ffprobe, opencv-python, numpy, torch
- **可选**: RIFE GPU (PyTorch + CUDA + train_log/flownet.pkl)

## License

MIT
