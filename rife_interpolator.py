"""
RIFE GPU 帧插值模块 - 流式逐帧处理，每对帧仅需 ~90MB 显存
依赖: ECCV2022-RIFE (PyTorch), train_log/flownet.pkl
"""
import sys
import subprocess
import tempfile
import time
from pathlib import Path
import json

import cv2
import torch
import numpy as np
import torch.nn.functional as F

# ── ECCV2022-RIFE 路径 ──
_RIFE_REPO = Path(__file__).resolve().parent / "ECCV2022-RIFE"
if str(_RIFE_REPO) not in sys.path:
    sys.path.insert(0, str(_RIFE_REPO))

_model = None

def _get_model():
    """懒加载 RIFE 模型（全局单例，~3GB VRAM）"""
    global _model
    if _model is None:
        from train_log.RIFE_HDv3 import Model
        _model = Model()
        _model.load_model('train_log', -1)
        _model.eval()
        _model.device()
    return _model

def _pad_to_multiple(t, multiple=64):
    _, _, h, w = t.shape
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    if ph or pw:
        return F.pad(t, [0, pw, 0, ph], mode='replicate'), ph, pw
    return t, 0, 0

def _np2t(frame):
    return torch.from_numpy(frame).permute(2,0,1).unsqueeze(0).float().cuda()/255.0

def _t2np(t, oh, ow):
    return (t[0,:,:oh,:ow].permute(1,2,0).cpu().numpy()*255).astype(np.uint8)


def rife_slowdown(input_video, speed_ratio, output_video):
    """
    RIFE GPU 运动插帧减速（流式，显存友好）

    策略: RIFE 2^passes 倍增 → 直接写入目标 FPS（无需二次 setpts）
    speed_ratio=1.5 → 2x RIFE, output_fps = 2*fps / 1.5
    speed_ratio=2.3 → 4x RIFE, output_fps = 4*fps / 2.3
    """
    t0 = time.time()
    model = _get_model()

    cap = cv2.VideoCapture(str(input_video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    multiplier = max(2, int(np.ceil(speed_ratio)))
    passes = max(1, int(np.ceil(np.log2(multiplier))))
    target_fps = fps * (2**passes) / speed_ratio

    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)

        if passes == 1:
            # 单级 2x：直接以目标 FPS 写出，无需中间文件
            _stream_2x(cap, model, fps, w, h, str(output_video), output_fps=target_fps)
            cap.release()
        else:
            # 多级倍增：中间级写临时文件，末级以目标 FPS 写出
            curr_input = tmpdir / "s0.mp4"
            _stream_2x(cap, model, fps, w, h, str(curr_input))
            cap.release()

            for level in range(1, passes):
                next_input = tmpdir / f"s{level}.mp4"
                cap2 = cv2.VideoCapture(str(curr_input))
                _stream_2x(cap2, model, fps * (2**level), w, h, str(next_input))
                cap2.release()
                time.sleep(0.1)  # Windows 文件句柄释放延迟
                curr_input.unlink(missing_ok=True)
                curr_input = next_input

            # 末级：以目标 FPS 写出
            cap_last = cv2.VideoCapture(str(curr_input))
            _stream_2x(cap_last, model, fps * (2**passes), w, h, str(output_video), output_fps=target_fps)
            cap_last.release()

    elapsed = time.time() - t0
    return (True, elapsed, nframes)


def _stream_2x(cap, model, fps, w, h, outpath, output_fps=None):
    """流式 2x 插帧：逐帧→插值→写盘，仅2帧在GPU。output_fps 给定则直接以该帧率写出。"""
    actual_fps = output_fps if output_fps is not None else fps * 2
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(outpath, fourcc, actual_fps, (w, h))

    ret, prev = cap.read()
    if not ret:
        out.release(); return
    out.write(prev)

    pt = _np2t(prev); pt, ph, pw = _pad_to_multiple(pt, 64)
    idx = 1
    while True:
        ret, cur = cap.read()
        if not ret: break
        ct = _np2t(cur); ct, _, _ = _pad_to_multiple(ct, 64)

        # FP16 自动混合精度推理 + scale=2 降内部分辨率加速
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                mid = model.inference(pt, ct, timestep=0.5, scale=1.0)

        out.write(_t2np(mid, h, w))
        out.write(cur)

        del pt; pt = ct
        idx += 1
        if idx % 60 == 0:
            torch.cuda.empty_cache()
    out.release()


# ── CLI 测试 ──
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("input"); ap.add_argument("ratio", type=float); ap.add_argument("output")
    a = ap.parse_args()
    ok, t, n = rife_slowdown(a.input, a.ratio, a.output)
    if ok:
        r = subprocess.run(['ffprobe','-v','quiet','-print_format','json','-show_format',a.output],
                          capture_output=True, text=True)
        dur = float(json.loads(r.stdout)['format']['duration'])
        print(f"OK {n}frames → {dur:.1f}s  ({t:.1f}s)")
    else:
        print(f"FAIL  ({t:.1f}s)")
