import os
import re
import subprocess
import tempfile
from pathlib import Path
import shutil
import json

# ── RIFE GPU 插帧 (PyTorch, 流式) ──
try:
    from rife_interpolator import rife_slowdown
    _rife_available = True
except Exception:
    rife_slowdown = None
    _rife_available = False

def parse_srt(srt_path):
    """解析SRT文件，返回时间戳列表 [(start_seconds, end_seconds), ...]"""
    timestamps = []
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    pattern = r'(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})'
    matches = re.findall(pattern, content)
    
    for match in matches:
        sh, sm, ss, sms, eh, em, es, ems = match
        start = int(sh) * 3600 + int(sm) * 60 + int(ss) + int(sms) / 1000
        end = int(eh) * 3600 + int(em) * 60 + int(es) + int(ems) / 1000
        timestamps.append((start, end))
    
    return timestamps

def get_audio_duration(audio_path):
    """获取音频时长"""
    cmd = [
        'ffprobe', '-v', 'quiet', '-print_format', 'json',
        '-show_format', str(audio_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    data = json.loads(result.stdout)
    return float(data['format']['duration'])

def cut_video_segment(video_path, start, end, output_path):
    """
    从视频中切出指定时间段（精确帧边界，重编码切边）
    修复: 去掉 -c:v copy，改为重编码以保证帧边界精确
    """
    duration = end - start
    cmd = [
        'ffmpeg', '-y',
        '-i', str(video_path),              # -i 在 -ss 之前 = 帧精确切割
        '-ss', str(start),                  # 输入 seek 到精确时间
        '-t', str(duration),
        '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '18',
        '-an',
        str(output_path)
    ]
    subprocess.run(cmd, capture_output=True)

def _setpts_adjust(video_segment, speed_ratio, output_path):
    """纯 setpts 调整（无插帧），用于变化 <15% 的段"""
    cmd = [
        'ffmpeg', '-y', '-i', str(video_segment),
        '-vf', f'setpts={speed_ratio}*PTS',
        '-an',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
        str(output_path)
    ]
    subprocess.run(cmd, capture_output=True)

def _rife_slowdown(video_segment, speed_ratio, output_path):
    """
    RIFE GPU 运动插帧减速 - 流式逐帧处理，显存友好
    策略: RIFE 倍增帧 → setpts=speed_ratio 拉伸时长
    """
    if not _rife_available or rife_slowdown is None:
        print("      [WARN] RIFE 模块不可用，回退 setpts")
        _setpts_adjust(video_segment, speed_ratio, output_path)
        return False
    
    ok, elapsed, nframes = rife_slowdown(video_segment, speed_ratio, output_path)
    return ok

def _framerate_speedup(video_segment, speed_ratio, output_path):
    """
    加速段：framerate 混合帧（避免 setpts 纯跳帧的生硬感）
    """
    cmd = [
        'ffmpeg', '-y', '-i', str(video_segment),
        '-vf',
        f'setpts={speed_ratio}*PTS,'
        f'framerate=fps=30:interp_start=0:interp_end=100:scene=100',
        '-an',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
        str(output_path)
    ]
    subprocess.run(cmd, capture_output=True)

def adjust_video_speed(video_segment, speed_ratio, output_path):
    """
    智能速度调整 — 混合策略：
    
    speed_ratio 范围         │ 策略                  │ 原因
    0.95 ~ 1.05              │ 直接复制              │ 肉眼无感
    0.85 ~ 1.15              │ setpts 裸调            │ 变化小，卡顿不明显
    > 1.15                   │ RIFE GPU 插帧          │ 减速明显 → 运动插帧
    < 0.85                   │ framerate 混合帧       │ 加速跳帧有生硬感
    """
    # 基本不变 → 直接复制
    if 0.95 <= speed_ratio <= 1.05:
        shutil.copy(video_segment, output_path)
        return "copy"
    
    # 轻微变化 → 纯 setpts
    if 0.85 <= speed_ratio <= 1.15:
        _setpts_adjust(video_segment, speed_ratio, output_path)
        return "setpts"
    
    # 显著减速 → RIFE GPU 插帧
    if speed_ratio > 1.15:
        if _rife_available and rife_slowdown:
            ok = _rife_slowdown(video_segment, speed_ratio, output_path)
            if ok:
                return "rife"
        # RIFE 不可用 → 回退 setpts（避免卡死的 minterpolate）
        print(f"      [WARN] RIFE 不可用, 使用 setpts 兜底")
        _setpts_adjust(video_segment, speed_ratio, output_path)
        return "setpts-fallback"
    
    # 显著加速 → framerate 混合
    _framerate_speedup(video_segment, speed_ratio, output_path)
    return "framerate"

def create_silence_video(duration, width, height, fps, output_path):
    """创建指定时长的黑屏视频"""
    cmd = [
        'ffmpeg', '-y', '-f', 'lavfi', '-i',
        f'color=c=black:s={width}x{height}:r={fps}',
        '-t', str(duration),
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
        str(output_path)
    ]
    subprocess.run(cmd, capture_output=True)

def get_video_info(video_path):
    """获取视频信息"""
    cmd = [
        'ffprobe', '-v', 'quiet', '-print_format', 'json',
        '-show_streams', str(video_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    data = json.loads(result.stdout)
    
    video_stream = None
    for stream in data['streams']:
        if stream['codec_type'] == 'video':
            video_stream = stream
            break
    
    if video_stream:
        return {
            'width': int(video_stream['width']),
            'height': int(video_stream['height']),
            'fps': eval(video_stream['r_frame_rate']),
            'duration': float(video_stream.get('duration', 0))
        }
    return None

def merge_video_audio(video_path, audio_path, output_path):
    """合并视频和音频"""
    cmd = [
        'ffmpeg', '-y',
        '-i', str(video_path),
        '-i', str(audio_path),
        '-c:v', 'copy',
        '-c:a', 'aac', '-b:a', '192k',
        '-shortest',
        str(output_path)
    ]
    subprocess.run(cmd, capture_output=True)

def combine_segments(segment_list, output_path):
    """使用concat demuxer合并多个视频片段"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        for segment in segment_list:
            f.write(f"file '{segment}'\n")
        concat_file = f.name
    
    cmd = [
        'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
        '-i', concat_file,
        '-c', 'copy',
        str(output_path)
    ]
    subprocess.run(cmd, capture_output=True)
    os.unlink(concat_file)

def process_video_with_dubbing(srt_path, video_path, dubbed_dir, output_path):
    """
    根据字幕时间戳逐段调整视频速度，匹配配音音频
    
    Args:
        srt_path: 英文字幕文件路径（提供时间戳）
        video_path: 原始视频路径（无音频）
        dubbed_dir: 配音片段目录 (0001.wav, 0002.wav, ...)
        output_path: 最终输出路径
    """
    print("[START] 开始逐段对齐视频和配音...")
    
    # 检测 RIFE 状态
    if _rife_available:
        print(f"[INFO] RIFE GPU 插帧已就绪 (PyTorch + CUDA)")
    else:
        print("[INFO] RIFE 模块不可用, 减速段将使用 setpts 兜底")
    
    # 1. 解析字幕时间戳
    timestamps = parse_srt(srt_path)
    print(f"[INFO] 解析到 {len(timestamps)} 段字幕")
    
    # 2. 获取配音片段
    dubbed_dir = Path(dubbed_dir)
    dubbed_files = sorted(dubbed_dir.glob("*.wav"))
    print(f"[INFO] 找到 {len(dubbed_files)} 个配音片段")
    
    if len(timestamps) != len(dubbed_files):
        print(f"[WARN] 数量不匹配，取最小值")
        min_count = min(len(timestamps), len(dubbed_files))
        timestamps = timestamps[:min_count]
        dubbed_files = dubbed_files[:min_count]
    
    # 3. 获取视频信息
    video_info = get_video_info(video_path)
    if not video_info:
        print("[ERROR] 无法获取视频信息")
        return False
    
    print(f"[INFO] 视频分辨率: {video_info['width']}x{video_info['height']}, FPS: {video_info['fps']}")
    
    # 4. 创建临时工作目录
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        segments_dir = tmpdir / "segments"
        segments_dir.mkdir()
        
        adjusted_segments = []
        method_counts = {"copy": 0, "setpts": 0, "rife": 0, "setpts-fallback": 0, "framerate": 0}
        
        # 5. 逐段处理
        for i, ((start, end), dubbed_file) in enumerate(zip(timestamps, dubbed_files), 1):
            video_duration = end - start
            audio_duration = get_audio_duration(dubbed_file)
            speed_ratio = audio_duration / video_duration
            
            segment_video = segments_dir / f"segment_{i:04d}.mp4"
            adjusted_video = segments_dir / f"adjusted_{i:04d}.mp4"
            
            # 切出原视频片段（重编码精确切割）
            cut_video_segment(video_path, start, end, segment_video)
            
            # 智能速度调整（混合策略）:
            #   copy:    0.95~1.05
            #   setpts:  0.85~1.15
            #   rife:    >1.15 (GPU 运动插帧)
            #   framerate: <0.85
            method = adjust_video_speed(segment_video, speed_ratio, adjusted_video)
            method_counts[method] = method_counts.get(method, 0) + 1
            
            # 合并调整后的视频和配音音频
            final_segment = segments_dir / f"final_{i:04d}.mp4"
            merge_video_audio(adjusted_video, dubbed_file, final_segment)
            
            adjusted_segments.append(str(final_segment))
            
            if i % 50 == 0:
                counts = ", ".join(f"{k}={v}" for k, v in method_counts.items())
                print(f"  进度: {i}/{len(timestamps)} | 采样率 {speed_ratio:.3f} → {method} | 累计: {counts}")
            
            # 清理中间文件（rife 在里层已清理，这里清理其余）
            segment_video.unlink(missing_ok=True)
            adjusted_video.unlink(missing_ok=True)
        
        # 6. 合并所有段
        print(f"[WORK] 策略统计: copy={method_counts['copy']}, setpts={method_counts['setpts']}, "
              f"rife={method_counts['rife']}, framerate={method_counts['framerate']}")
        print("[WORK] 合并所有调整后的片段...")
        
        concat_file = tmpdir / "concat.txt"
        with open(concat_file, 'w') as f:
            for segment in adjusted_segments:
                f.write(f"file '{segment}'\n")
        
        cmd = [
            'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
            '-i', str(concat_file),
            '-c', 'copy',
            str(output_path)
        ]
        result = subprocess.run(cmd, capture_output=True)
        
        if result.returncode == 0:
            output_info = get_video_info(output_path)
            total_audio_duration = sum(get_audio_duration(f) for f in dubbed_files)
            print(f"[OK] 处理完成!")
            print(f"   输出文件: {output_path}")
            if output_info:
                print(f"   输出时长: {output_info['duration']:.2f}秒")
            print(f"   配音总时长: {total_audio_duration:.2f}秒")
            return True
        else:
            print(f"[ERROR] 合并失败: {result.stderr[:300]}")
            return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 5:
        print("视频配音对齐工具 (混合策略版)")
        print("根据字幕时间戳逐段调整视频速度，匹配配音音频时长")
        print()
        print("用法: python align_video_to_dubbing.py <字幕.srt> <视频.mp4> <配音目录> <输出.mp4>")
        print()
        print("速度调整策略:")
        print("  0.95~1.05  → 直接复制 (肉眼无感)")
        print("  0.85~1.15  → setpts 裸调")
        print("  > 1.15     → RIFE GPU 运动插帧 (减速)")
        print("               └ 不可用时 setpts 兜底")
        print("  < 0.85     → framerate 混合帧 (加速)")
        print()
        print("RIFE GPU 依赖: ECCV2022-RIFE + PyTorch + train_log/flownet.pkl")
        print()
        print("示例:")
        print("  python align_video_to_dubbing.py test/video_zh.srt test/video_no_audio.mp4 test/dubbed_output test/final_output.mp4")
        sys.exit(1)
    
    srt_path = sys.argv[1]
    video_path = sys.argv[2]
    dubbed_dir = sys.argv[3]
    output_path = sys.argv[4]
    
    process_video_with_dubbing(srt_path, video_path, dubbed_dir, output_path)

def main():
    import sys
    if len(sys.argv) < 5:
        print("用法: align-video <字幕.srt> <视频.mp4> <配音目录> <输出.mp4>")
        print("      align-video test/video_zh.srt test/video_no_audio.mp4 test/dubbed_output test/output.mp4")
        sys.exit(1)
    process_video_with_dubbing(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
