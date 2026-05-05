import os
import re
import subprocess
import tempfile
from pathlib import Path
import shutil
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# ── RIFE GPU 插帧 (PyTorch, 流式) ──
try:
    from rife_interpolator import rife_slowdown
    _rife_available = True
except Exception:
    rife_slowdown = None
    _rife_available = False

# ── P1 配置常量 ──
AUDIO_ONLY_THRESHOLD = 0.10           # |ratio-1| 在此阈值内使用纯音频拉伸，超过则双向分担
SCENE_SNAP_RANGE = 0.5                # 场景边界吸附范围（秒）

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

def stretch_audio(input_wav, output_wav, target_duration):
    """
    Stretch/compress audio to exactly match target duration.
    Uses rubberband for quality, atempo as fallback.
    Returns: Path to stretched audio file
    """
    original = get_audio_duration(input_wav)
    tempo = original / target_duration

    # 几乎不变 → 直接复制
    if 0.98 <= tempo <= 1.02:
        shutil.copy(input_wav, output_wav)
        return output_wav

    # 优先尝试 rubberband（高质量，任意 tempo）
    result = subprocess.run(
        [
            'ffmpeg', '-y', '-i', str(input_wav),
            '-af', f'rubberband=tempo={tempo:.4f}',
            '-ar', '16000', '-ac', '1',
            str(output_wav)
        ],
        capture_output=True
    )
    if result.returncode == 0:
        return output_wav

    # 回退: atempo（链式处理超出 0.5~2.0 范围的情况）
    filters = []
    remaining = tempo
    while remaining > 2.0:
        filters.append('atempo=2.0')
        remaining /= 2.0
    while remaining < 0.5:
        filters.append('atempo=0.5')
        remaining /= 0.5
    filters.append(f'atempo={remaining:.4f}')

    subprocess.run(
        [
            'ffmpeg', '-y', '-i', str(input_wav),
            '-af', ','.join(filters),
            '-ar', '16000', '-ac', '1',
            str(output_wav)
        ],
        capture_output=True
    )
    return output_wav

def detect_scene_changes(video_path):
    """
    Detect all scene-change timestamps in video.
    Returns: sorted list of (timestamp_seconds, score) tuples
    """
    cmd = [
        'ffmpeg', '-i', str(video_path),
        '-vf', "select='gt(scene\\,0.3)',showinfo",
        '-vsync', 'vfr', '-an', '-f', 'null', '-'
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    scene_times = []
    # 从 stderr 解析 pts_time 值
    for line in result.stderr.splitlines():
        if 'pts_time:' in line:
            match = re.search(r'pts_time:([\d.]+)', line)
            if match:
                ts = float(match.group(1))
                scene_times.append(ts)

    return sorted(scene_times)

def snap_to_scene(time, scene_times, range_sec=SCENE_SNAP_RANGE):
    """
    将时间吸附到最近的场景边界（若边界在 ±range_sec 内）。
    返回吸附后的时间；若无合适边界，返回原时间。
    """
    nearest = None
    nearest_dist = float('inf')
    for st in scene_times:
        dist = abs(st - time)
        if dist <= range_sec and dist < nearest_dist:
            nearest_dist = dist
            nearest = st
    return nearest if nearest is not None else time

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
    """合并视频和音频（固定采样率 16kHz 确保 concat 兼容）"""
    cmd = [
        'ffmpeg', '-y',
        '-i', str(video_path),
        '-i', str(audio_path),
        '-c:v', 'copy',
        '-c:a', 'aac', '-b:a', '192k', '-ar', '16000',
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

def _process_one_segment(args):
    """
    处理单个片段的纯函数（线程安全）

    Args:
        args: (i, video_path, start, end, dubbed_file, segments_dir, audio_stretch)

    Returns:
        (i, method, final_path)
    """
    i, video_path, start, end, dubbed_file, segments_dir, audio_stretch = args
    video_duration = end - start
    audio_duration = get_audio_duration(dubbed_file)
    speed_ratio = audio_duration / video_duration

    segment_video = segments_dir / f"segment_{i:04d}.mp4"
    adjusted_video = segments_dir / f"adjusted_{i:04d}.mp4"
    final_segment = segments_dir / f"final_{i:04d}.mp4"

    # 如果已存在则跳过（恢复场景）
    if final_segment.exists():
        return (i, "existing", str(final_segment))

    deviation = abs(speed_ratio - 1.0)

    if not audio_stretch:
        # ── 音频拉伸完全禁用：仅调整视频 ──
        cut_video_segment(video_path, start, end, segment_video)
        method = adjust_video_speed(segment_video, speed_ratio, adjusted_video)
        merge_video_audio(adjusted_video, dubbed_file, final_segment)

        segment_video.unlink(missing_ok=True)
        adjusted_video.unlink(missing_ok=True)
        return (i, method, str(final_segment))

    if deviation <= AUDIO_ONLY_THRESHOLD:
        # ── 纯音频拉伸（视频原画质） ──
        cut_video_segment(video_path, start, end, segment_video)
        actual_vid_dur = get_audio_duration(segment_video)

        stretched_audio = segments_dir / f"stretched_{i:04d}.wav"
        stretch_audio(dubbed_file, stretched_audio, actual_vid_dur)

        shutil.copy(segment_video, adjusted_video)
        merge_video_audio(adjusted_video, stretched_audio, final_segment)

        segment_video.unlink(missing_ok=True)
        adjusted_video.unlink(missing_ok=True)
        stretched_audio.unlink(missing_ok=True)

        return (i, "astretch", str(final_segment))

    # ── 双向分担：音频和视频各走一半的差距 ──
    split_ratio = speed_ratio ** 0.5

    # Step 1: 先切视频，获取实际时长（非理论值）
    cut_video_segment(video_path, start, end, segment_video)
    actual_vid_dur = get_audio_duration(segment_video)

    # Step 2: 拉伸音频，目标匹配调整后的视频时长
    stretched_audio = segments_dir / f"stretched_{i:04d}.wav"
    stretch_audio(dubbed_file, stretched_audio, actual_vid_dur * split_ratio)

    # Step 3: 视频按 split_ratio 调整速度
    vid_method = adjust_video_speed(segment_video, split_ratio, adjusted_video)

    # Step 4: 合并
    merge_video_audio(adjusted_video, stretched_audio, final_segment)

    # 清理中间文件
    segment_video.unlink(missing_ok=True)
    adjusted_video.unlink(missing_ok=True)
    stretched_audio.unlink(missing_ok=True)

    return (i, f"bi-{vid_method}", str(final_segment))


def process_video_with_dubbing(srt_path, video_path, dubbed_dir, output_path, workers=4, resume=False, audio_stretch=True, scene_snap=False):
    """
    根据字幕时间戳逐段调整视频速度，匹配配音音频

    Args:
        srt_path: 英文字幕文件路径（提供时间戳）
        video_path: 原始视频路径（无音频）
        dubbed_dir: 配音片段目录 (0001.wav, 0002.wav, ...)
        output_path: 最终输出路径
        workers: 并行处理线程数（默认 4）
        resume: 是否从检查点恢复（默认 False）
    """
    print("[START] 开始逐段对齐视频和配音...")

    # 检测 RIFE 状态
    if _rife_available:
        print(f"[INFO] RIFE GPU 插帧已就绪 (PyTorch + CUDA)")
    else:
        print("[INFO] RIFE 模块不可用, 减速段将使用 setpts 兜底")

    # P1-2: 场景边界检测（如启用）
    scene_times = []
    if scene_snap:
        print("[INFO] 正在检测场景切换边界...")
        scene_times = detect_scene_changes(video_path)
        print(f"[INFO] 检测到 {len(scene_times)} 个场景切换点")

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

    # 4. 创建持久化工作目录
    segments_dir = Path(f"{output_path}.segments")
    segments_dir.mkdir(exist_ok=True)

    checkpoint_path = Path(f"{output_path}.checkpoint.json")
    completed = set()

    # 恢复检查点
    if resume and checkpoint_path.exists():
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
            completed = set(checkpoint.get('completed', []))
            print(f"[RESUME] 从检查点恢复，已处理 {len(completed)} 个片段")
        except Exception as e:
            print(f"[WARN] 检查点读取失败: {e}")
            completed = set()

    # 同时检查已存在的输出文件
    for i in range(1, len(timestamps) + 1):
        if (segments_dir / f"final_{i:04d}.mp4").exists():
            completed.add(i)

    if len(completed) > 0 and not resume:
        print(f"[INFO] 发现 {len(completed)} 个已存在的片段，将跳过")

    # 构建任务列表（应用场景边界吸附）
    tasks = []
    for i, ((start, end), dubbed_file) in enumerate(zip(timestamps, dubbed_files), 1):
        if i not in completed:
            orig_start, orig_end = start, end
            if scene_snap and scene_times:
                start = snap_to_scene(start, scene_times)
                end = snap_to_scene(end, scene_times)
                if start != orig_start or end != orig_end:
                    print(f"[SCENE] seg {i}: start {orig_start:.2f}→{start:.2f}, end {orig_end:.2f}→{end:.2f}")
            tasks.append((i, video_path, start, end, dubbed_file, segments_dir, audio_stretch))

    method_counts = {
        "copy": 0, "setpts": 0, "rife": 0, "setpts-fallback": 0, "framerate": 0,
        "astretch": 0,
        "bi-copy": 0, "bi-setpts": 0, "bi-rife": 0, "bi-setpts-fallback": 0, "bi-framerate": 0,
    }

    if tasks:
        print(f"[WORK] 并行处理中 (workers={workers})...")
        lock = threading.Lock()
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_process_one_segment, task): task[0] for task in tasks}

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    i, method, final_path = future.result()

                    with lock:
                        if method != "existing":
                            method_counts[method] = method_counts.get(method, 0) + 1
                        completed.add(i)

                        # 写入检查点
                        checkpoint = {
                            "output": str(output_path),
                            "completed": sorted(completed),
                            "total": len(timestamps)
                        }
                        try:
                            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                                json.dump(checkpoint, f)
                        except Exception as e:
                            print(f"\n[WARN] 检查点写入失败: {e}")

                        # 进度报告
                        elapsed = time.time() - start_time
                        done_count = len(completed)
                        avg_time = elapsed / done_count if done_count > 0 else 0

                        counts_str = " ".join(f"{k}={v}" for k, v in [
                            ("rife", method_counts.get("rife", 0) + method_counts.get("setpts-fallback", 0)
                             + method_counts.get("bi-rife", 0) + method_counts.get("bi-setpts-fallback", 0)),
                            ("setpts", method_counts.get("setpts", 0) + method_counts.get("bi-setpts", 0)),
                            ("astretch", method_counts.get("astretch", 0)),
                            ("copy", method_counts.get("copy", 0) + method_counts.get("bi-copy", 0)),
                            ("framerate", method_counts.get("framerate", 0) + method_counts.get("bi-framerate", 0))
                        ])
                        print(f"\r  完成: {done_count}/{len(timestamps)} | {counts_str} | ~{avg_time:.1f}s/段", end="", flush=True)

                except Exception as e:
                    print(f"\n[ERROR] 处理片段 {idx} 失败: {e}")
                    return False

        print()  # 换行

    # 收集所有片段（按索引排序）
    adjusted_segments = []
    for i in range(1, len(timestamps) + 1):
        final_segment = segments_dir / f"final_{i:04d}.mp4"
        if not final_segment.exists():
            print(f"[ERROR] 片段 {i} 输出文件缺失")
            return False
        adjusted_segments.append(str(final_segment))

    # 6. 合并所有段
    print(f"[WORK] 策略统计: copy={method_counts['copy']+method_counts.get('bi-copy',0)}, "
          f"setpts={method_counts['setpts']+method_counts.get('bi-setpts',0)}, "
          f"rife={method_counts['rife']+method_counts.get('bi-rife',0)}, "
          f"astretch={method_counts['astretch']}, "
          f"framerate={method_counts['framerate']+method_counts.get('bi-framerate',0)}")
    print("[WORK] 合并所有调整后的片段...")

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        for segment in adjusted_segments:
            # 使用绝对路径 + 正斜杠（concat 文件在 Temp 目录，相对路径会错）
            abs_path = str(Path(segment).resolve()).replace(chr(92), '/')
            f.write(f"file '{abs_path}'\n")
        concat_file = f.name

    cmd = [
        'ffmpeg', '-y', '-fflags', '+genpts', '-f', 'concat', '-safe', '0',
        '-i', concat_file,
        '-c', 'copy',
        str(output_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    os.unlink(concat_file)
    
    if result.returncode == 0:
        ...
        output_info = get_video_info(output_path)
        total_audio_duration = sum(get_audio_duration(f) for f in dubbed_files)
        print(f"[OK] 处理完成!")
        print(f"   输出文件: {output_path}")
        if output_info:
            print(f"   输出时长: {output_info['duration']:.2f}秒")
        print(f"   配音总时长: {total_audio_duration:.2f}秒")

        # 成功后清理持久化目录
        if segments_dir.exists():
            shutil.rmtree(segments_dir)
        if checkpoint_path.exists():
            checkpoint_path.unlink(missing_ok=True)

        return True
    else:
        print(f"[ERROR] 合并失败: {result.stderr[-600:]}")
        return False

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="视频配音对齐工具 (混合策略版 + 并行处理)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""速度调整策略:
  0.95~1.05  → 直接复制 (肉眼无感)
  0.85~1.15  → setpts 裸调
  > 1.15     → RIFE GPU 运动插帧 (减速)
               └ 不可用时 setpts 兜底
  < 0.85     → framerate 混合帧 (加速)

示例:
  python align_video_to_dubbing.py test/video_zh.srt test/video_no_audio.mp4 test/dubbed_output test/final_output.mp4 --workers 4
"""
    )
    parser.add_argument("srt", help="字幕文件路径 (.srt)")
    parser.add_argument("video", help="原始视频路径 (无音频)")
    parser.add_argument("dubbed_dir", help="配音片段目录 (0001.wav, 0002.wav, ...)")
    parser.add_argument("output", help="最终输出路径 (.mp4)")
    parser.add_argument("--workers", type=int, default=4, help="并行处理线程数 (默认 4)")
    parser.add_argument("--resume", action="store_true", help="从检查点恢复")
    parser.add_argument("--audio-stretch", action="store_true", default=True, help="小范围内拉伸音频而非调整视频速度 (默认启用)")
    parser.add_argument("--no-audio-stretch", dest="audio_stretch", action="store_false", help="禁用音频拉伸")
    parser.add_argument("--scene-snap", action="store_true", default=False, help="将片段切点吸附到场景切换边界")

    args = parser.parse_args()

    process_video_with_dubbing(
        args.srt,
        args.video,
        args.dubbed_dir,
        args.output,
        workers=args.workers,
        resume=args.resume,
        audio_stretch=args.audio_stretch,
        scene_snap=args.scene_snap
    )

def main():
    import sys
    if len(sys.argv) < 5:
        print("用法: align-video <字幕.srt> <视频.mp4> <配音目录> <输出.mp4>")
        print("      align-video test/video_zh.srt test/video_no_audio.mp4 test/dubbed_output test/output.mp4")
        sys.exit(1)

    # 解析位置参数
    srt_path = sys.argv[1]
    video_path = sys.argv[2]
    dubbed_dir = sys.argv[3]
    output_path = sys.argv[4]

    workers = 4
    resume = False
    audio_stretch = True
    scene_snap = False

    if "--workers" in sys.argv:
        idx = sys.argv.index("--workers")
        if idx + 1 < len(sys.argv):
            workers = int(sys.argv[idx + 1])

    if "--resume" in sys.argv:
        resume = True

    if "--no-audio-stretch" in sys.argv:
        audio_stretch = False
    if "--audio-stretch" in sys.argv:
        audio_stretch = True

    if "--scene-snap" in sys.argv:
        scene_snap = True

    process_video_with_dubbing(
        srt_path, video_path, dubbed_dir, output_path,
        workers=workers, resume=resume,
        audio_stretch=audio_stretch, scene_snap=scene_snap
    )
