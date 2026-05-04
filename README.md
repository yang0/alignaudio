# AlignAudio

将配音音频（多个 WAV 片段）逐段对齐到原始视频，自动调整视频速度匹配配音时长，支持 **RIFE GPU 运动插帧** 消除减速时的画面卡顿。

## 核心流程

```
SRT 字幕时间轴 → 逐段切视频 → 计算速度比 → 混合策略调整速度 → 合并音频 → 输出
                                                  │
                   speed_ratio > 1.15 ──→ RIFE GPU 运动插帧 (减速)
                   0.85~1.15         ──→ setpts 裸调
                   0.95~1.05         ──→ 直接复制
                   < 0.85            ──→ framerate 混合 (加速)
```

## 安装

### 1. 基础依赖

```bash
pip install alignaudio
# 或开发模式
pip install -e .
```

### 2. RIFE GPU 插帧（可选，但强烈推荐）

```bash
# 克隆 RIFE 模型代码
git clone --depth 1 https://github.com/hzwer/ECCV2022-RIFE.git ECCV2022-RIFE

# 下载预训练模型 v4.26
# 方式 A: HuggingFace (推荐)
python -c "
import urllib.request, zipfile
url = 'https://huggingface.co/hzwer/RIFE/resolve/main/RIFEv4.26_0921.zip'
urllib.request.urlretrieve(url, 'model.zip')
zipfile.ZipFile('model.zip').extractall('train_log')
import shutil, glob
for f in glob.glob('train_log/RIFEv4.26_0921/*'):
    shutil.move(f, 'train_log/')
"

# 方式 B: 百度网盘
# https://pan.baidu.com/share/init?surl=u6Q7-i4Hu4Vx9_5BJibPPA 密码: hfk3
# 解压后把 flownet.pkl / RIFE_HDv3.py / IFNet_HDv3.py / refine.py 放到 train_log/
```

## CLI 命令

安装后可使用以下命令：

```bash
# 主流程：视频配音对齐（混合策略 + RIFE 插帧）
align-video <字幕.srt> <视频.mp4> <配音目录> <输出.mp4>

# 纯音频对齐（不处理视频）
align-audio <字幕.srt> <配音目录> <输出.wav>

# 字幕时间戳调整（根据配音时长重算）
adjust-subs <字幕.srt> <配音目录> <输出.srt>
```

### 示例

```bash
# 目录结构
# test/
#   video_no_audio.mp4      ← 原始视频（无音频）
#   video_zh.srt             ← 中文字幕（321条时间戳）
#   dubbed_output/           ← 配音片段
#     0001.wav, 0002.wav, ...

align-video test/video_zh.srt test/video_no_audio.mp4 test/dubbed_output test/output.mp4
# 输出: test/output.mp4 (视频+配音音频，时长与配音总时长一致)
```

## 速度调整策略

| speed_ratio 范围 | 变化幅度 | 策略 | 原因 |
|---|---|---|---|
| 0.95 ~ 1.05 | <5% | 直接复制 | 肉眼无感，避免重编码 |
| 0.85 ~ 1.15 | 5-15% | setpts 裸调 | 帧复制/丢弃，变化小卡顿不明显 |
| > 1.15 | >15% 减速 | **RIFE GPU 插帧** | 运动补偿生成中间帧，消除画面卡顿 |
| < 0.85 | >15% 加速 | framerate 混合 | 帧混合而非粗暴丢弃 |

## 文件结构

```
alignaudio/
├── align_video_to_dubbing.py   # 主管线（混合策略 + RIFE 集成）
├── align_video_dubbing.py      # 备选管线（带字幕嵌入功能）
├── align_dubbed_audio.py       # 纯音频对齐
├── adjust_subtitle_timestamps.py # 字幕时间戳重算
├── rife_interpolator.py        # RIFE GPU 流式插帧模块
├── preview_3seg.py             # 快速预览（前3段硬编码）
├── pyproject.toml              # 包配置 + CLI 入口
├── ECCV2022-RIFE/              # RIFE 模型代码（需手动克隆）
├── train_log/                  # RIFE 模型权重（需手动下载）
└── test/                       # 测试数据
```

## 依赖

- **必需**: Python ≥ 3.10, ffmpeg/ffprobe, opencv-python, numpy, torch
- **可选**: RIFE GPU (PyTorch + CUDA + train_log/flownet.pkl), torchvision, tqdm

## License

MIT
