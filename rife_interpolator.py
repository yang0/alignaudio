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
    
    策略: RIFE 2^passes 倍增 → setpts 微调
    speed_ratio=1.5 → 2x RIFE + setpts=0.75
    speed_ratio=2.3 → 4x RIFE + setpts=0.575
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
    
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        curr_input = tmpdir / "s0.mp4"
        
        # RIFE 2x 逐帧流式
        _stream_2x(cap, model, fps, w, h, str(curr_input))
        cap.release()
        
        for level in range(1, passes):
            next_input = tmpdir / f"s{level}.mp4"
            cap2 = cv2.VideoCapture(str(curr_input))
            _stream_2x(cap2, model, fps * (2**level), w, h, str(next_input))
            cap2.release()
            curr_input.unlink(missing_ok=True)
            curr_input = next_input
        
        # setpts 微调: setpts=speed_ratio 直接拉伸到目标时长
        # RIFE 提供的是帧密度(60fps→120fps)，setpts 负责时长
        ratio = speed_ratio
        r = subprocess.run([
            'ffmpeg','-y','-i',str(curr_input),
            '-vf',f'setpts={ratio}*PTS',
            '-c:v','libx264','-preset','fast','-crf','18','-an',
            str(output_video)
        ], capture_output=True, text=True)
    
    elapsed = time.time() - t0
    return (r.returncode == 0, elapsed, nframes)


def _stream_2x(cap, model, fps, w, h, outpath):
    """流式 2x 插帧：逐帧→插值→写盘，仅2帧在GPU"""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(outpath, fourcc, fps*2, (w, h))
    
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
        
        with torch.no_grad():
            mid = model.inference(pt, ct, timestep=0.5)
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
