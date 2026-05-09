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
AUDIO_ONLY_THRESHOLD = 0.20           # |ratio-1| 在此阈值内使用纯音频拉伸（视频零损耗），超过则双向分担
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
            '-ar', '24000', '-ac', '1',
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
            '-ar', '24000', '-ac', '1',
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

def cut_video_segment(video_path, start, end, output_path, fast=False):
    """
    从视频中切出指定时间段（精确帧边界，重编码切边）
    修复: 去掉 -c:v copy，改为重编码以保证帧边界精确
    fast=True: 使用 -c:v copy 快速切割（适用于视频未修改的场景）
    """
    duration = end - start
    if fast:
        # Keyframe-accurate cut, no re-encode (for astretch: video unchanged, audio stretched to match)
        cmd = [
            'ffmpeg', '-y', '-ss', str(start), '-t', str(duration),
            '-i', str(video_path), '-c:v', 'copy', '-an',
            str(output_path)
        ]
    else:
        # Frame-accurate cut with re-encode
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
        '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '18',
        str(output_path)
    ]
    subprocess.run(cmd, capture_output=True)

def _cut_and_adjust_segment(video_path, start, end, speed_ratio, output_path):
    """Single ffmpeg pass: cut segment + apply setpts or framerate."""
    duration = end - start
    if 0.85 <= speed_ratio <= 1.15:
        cmd = [
            'ffmpeg', '-y',
            '-ss', str(start),
            '-i', str(video_path), '-t', str(duration),
            '-vf', f'setpts={speed_ratio}*PTS',
            '-an',
            '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '18',
            str(output_path)
        ]
        subprocess.run(cmd, capture_output=True)
        return "setpts"
    elif speed_ratio < 0.85:
        cmd = [
            'ffmpeg', '-y',
            '-ss', str(start),
            '-i', str(video_path), '-t', str(duration),
            '-vf',
            f'setpts={speed_ratio}*PTS,'
            f'framerate=fps=30:interp_start=0:interp_end=100:scene=100',
            '-an',
            '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '18',
            str(output_path)
        ]
        subprocess.run(cmd, capture_output=True)
        return "framerate"
    else:
        raise ValueError(f"_cut_and_adjust_segment: unsupported speed_ratio {speed_ratio}")

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
        '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '18',
        str(output_path)
    ]
    subprocess.run(cmd, capture_output=True)

def adjust_video_speed(video_segment, speed_ratio, output_path):
    """
    智能速度调整 — 混合策略：
    
    speed_ratio 范围         │ 策略                  │ 原因
    0.95 ~ 1.05              │ 直接复制              │ 肉眼无感
    0.85 ~ 1.08              │ setpts 裸调            │ 变化小，卡顿不明显
    > 1.08                   │ RIFE GPU 插帧          │ 减速明显 → 运动插帧
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
        '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '18',
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
    """合并视频和音频（精确截断到公共时长，消除 PTS 缝隙）"""
    # 测量实际时长
    vid_dur = get_audio_duration(video_path)
    aud_dur = get_audio_duration(audio_path)
    min_dur = min(vid_dur, aud_dur)
    
    cmd = [
        'ffmpeg', '-y',
        '-i', str(video_path),
        '-i', str(audio_path),
        '-c:v', 'copy',
        '-c:a', 'aac', '-b:a', '192k', '-ar', '24000',
        '-t', str(min_dur),
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
        if 0.95 <= speed_ratio <= 1.05:
            # Near-1.0: fast cut (gap segments etc, re-encode to h264 for concat compatibility)
            subprocess.run(['ffmpeg', '-y', '-ss', str(start), '-i', str(video_path),
                '-t', str(end-start), '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '18', '-an',
                str(adjusted_video)], capture_output=True)
            method = "copy"
        elif speed_ratio > 1.15 and _rife_available and rife_slowdown is not None:
            # RIFE path needs separate cut step (RIFE takes segment_video as input)
            cut_video_segment(video_path, start, end, segment_video)
            method = adjust_video_speed(segment_video, speed_ratio, adjusted_video)
        else:
            # setpts (0.85~0.95, 1.05~1.15) or framerate (<0.85) — single ffmpeg pass
            method = _cut_and_adjust_segment(video_path, start, end, speed_ratio, adjusted_video)
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

    # Step 1: 先切视频
    cut_video_segment(video_path, start, end, segment_video)

    # Step 2: 视频按 split_ratio 调整速度（用原有自适应策略）
    vid_method = adjust_video_speed(segment_video, split_ratio, adjusted_video)

    # Step 3: 拉伸音频，精确匹配调整后的实际视频时长
    actual_adj_dur = get_audio_duration(adjusted_video)
    stretched_audio = segments_dir / f"stretched_{i:04d}.wav"
    stretch_audio(dubbed_file, stretched_audio, actual_adj_dur)

    # Step 4: 合并
    merge_video_audio(adjusted_video, stretched_audio, final_segment)

    # Step 3: 拉伸音频，精确匹配调整后的实际视频时长
    actual_adj_dur = get_audio_duration(adjusted_video)
    stretched_audio = segments_dir / f"stretched_{i:04d}.wav"
    stretch_audio(dubbed_file, stretched_audio, actual_adj_dur)

    # Step 4: 合并
    merge_video_audio(adjusted_video, stretched_audio, final_segment)

    # 清理中间文件
    segment_video.unlink(missing_ok=True)
    adjusted_video.unlink(missing_ok=True)
    stretched_audio.unlink(missing_ok=True)

    return (i, f"bi-{vid_method}", str(final_segment))


def _fast_pipeline(srt_path, video_path, dubbed_dir, output_path, audio_stretch=False, scene_snap=False, adaptive_speed=False):
    """
    单命令 ffmpeg 快速管道：所有非 RIFE 段在一个 filter_complex 中处理。
    速度：1 个 ffmpeg 进程 vs 630 个子进程。
    """
    print("[WORK] 单命令快速管道...")

    timestamps = parse_srt(srt_path)
    ddir = Path(dubbed_dir)
    dubbed_files = sorted(ddir.glob("*.wav"))
    if len(timestamps) != len(dubbed_files):
        min_count = min(len(timestamps), len(dubbed_files))
        timestamps = timestamps[:min_count]
        dubbed_files = dubbed_files[:min_count]

    n = len(timestamps)

    # 如果段数多（>60），分批处理避免命令行超限
    MAX_SEGS_PER_BATCH = 30
    batch_count = max(1, (len(dubbed_files) + MAX_SEGS_PER_BATCH - 1) // MAX_SEGS_PER_BATCH)
    if batch_count > 1:
        print(f"[INFO] {n} 段字幕分 {batch_count} 批处理 (每批 ≤{MAX_SEGS_PER_BATCH} 段)")

        batch_outputs = []
        batch_dir = Path(str(output_path) + ".batches")
        batch_dir.mkdir(exist_ok=True)

        for bi in range(batch_count):
            si = bi * MAX_SEGS_PER_BATCH
            ei = min(si + MAX_SEGS_PER_BATCH, n)
            batch_out = batch_dir / f"batch_{bi:04d}.mp4"

            batch_timestamps = timestamps[si:ei]
            batch_wavs = dubbed_files[si:ei]

            if not _run_single_batch(batch_timestamps, batch_wavs, video_path, str(batch_out)):
                return False
            batch_outputs.append(str(batch_out))

        # 用 concat demuxer 合并批次
        concat_list = batch_dir / "concat.txt"
        with open(concat_list, 'w') as f:
            for p in batch_outputs:
                f.write(f"file '{Path(p).resolve()}'\n")
        subprocess.run(['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', str(concat_list),
                        '-c', 'copy', str(output_path)], capture_output=True)
        shutil.rmtree(batch_dir, ignore_errors=True)

        output_info = get_video_info(output_path)
        print(f"[OK] 分批完成 ({batch_count} 批)")
        if output_info:
            print(f"   输出时长: {output_info['duration']:.2f}秒")
        return True

    # 单批直接处理
    return _run_single_batch(timestamps, dubbed_files, video_path, output_path, audio_stretch, scene_snap, adaptive_speed)


def _run_single_batch(timestamps, dubbed_files, video_path, output_path, audio_stretch=False, scene_snap=False, adaptive_speed=False, scene_times=None):
    """处理一批段（单 ffmpeg 命令），返回 True/False"""
    n = len(timestamps)
    print(f"[INFO] {n} 段字幕, 单命令处理")

    # 场景吸附（可选）
    _scene_times = scene_times if scene_times is not None else []
    if scene_snap and not _scene_times:
        print("[INFO] 场景吸附检测中...")
        _scene_times = detect_scene_changes(video_path)

    # ── 预计算所有段的参数 ──
    # 统一格式: (start, effective_end, v_ratio, a_tempo, audio_path, next_start)
    segment_params = []
    
    # 两级变速对齐（全局+局部，防止极端变速）
    if adaptive_speed:
        total_desired = sum(end - start for (start, end), _ in zip(timestamps, dubbed_files))
        total_actual = sum(get_audio_duration(w) for _, w in zip(timestamps, dubbed_files))
        base_factor = max(0.8, min(1.2, total_desired / max(total_actual, 0.001) * 0.99))
        print(f"[INFO] 两级变速: base={base_factor:.3f} (总目标={total_desired:.1f}s, 实际={total_actual:.1f}s)")
    
    for i, ((start, end), wav) in enumerate(zip(timestamps, dubbed_files)):
        if scene_snap and scene_times:
            start = snap_to_scene(start, scene_times)
            end = snap_to_scene(end, scene_times)

        next_start = timestamps[i + 1][0] if i + 1 < len(timestamps) else None
        if scene_snap and scene_times and next_start is not None:
            next_start = snap_to_scene(next_start, scene_times)

        vid_dur = end - start
        aud_dur = get_audio_duration(wav)
        
        if adaptive_speed:
            # 两级变速：base * local，双限制防止音质劣化
            local = max(0.9, min(1.1, vid_dur / max(aud_dur * base_factor, 0.001)))
            a_tempo = base_factor * local
            effective_end = end
            v_ratio = 1.0
            pad_dur = 0.0
            if next_start is not None and effective_end > next_start:
                effective_end = next_start
            segment_params.append((start, effective_end, v_ratio, a_tempo, wav, next_start, pad_dur, effective_end))
            continue

        if audio_stretch:
            ratio = aud_dur / vid_dur
            if abs(ratio - 1) <= 0.20:
                segment_params.append((start, end, 1.0, ratio, wav, next_start, 0.0, end))
            else:
                split = ratio ** 0.5
                segment_params.append((start, end, split, split, wav, next_start, 0.0, end))
        else:
            # 音频优先策略
            gap_after = (next_start - end) if next_start is not None else 0.0
            extra = aud_dur - vid_dur

            if extra <= 0:
                # 音频比视频短 → 加速视频匹配音频，无静音
                effective_end = end  # 视频 trim 用到原始 SRT 窗口
                v_ratio = aud_dur / vid_dur
                a_tempo = 1.0
                pad_dur = 0.0
                segment_end = start + aud_dur  # 加速后实际结束点
            elif extra <= gap_after:
                effective_end = end + extra
                v_ratio = 1.0
                a_tempo = 1.0
                pad_dur = 0.0
                segment_end = effective_end
            elif aud_dur / (vid_dur + gap_after) <= 1.15:
                effective_end = end + gap_after
                v_ratio = 1.0
                a_tempo = aud_dur / (vid_dur + gap_after)
                pad_dur = 0.0
                segment_end = effective_end
            else:
                a_tempo = 1.15
                compressed_audio = aud_dur / 1.15
                v_ratio = compressed_audio / vid_dur
                effective_end = end
                pad_dur = 0.0
                segment_end = effective_end

            if next_start is not None and effective_end > next_start:
                effective_end = next_start

            segment_params.append((start, effective_end, v_ratio, a_tempo, wav, next_start, pad_dur, segment_end))

    # ── 构建 filter_complex ──
    video_input = str(Path(video_path).resolve())
    audio_inputs = [video_input]  # [0] = video
    filters = []
    concat_labels = []
    seg_idx = 0

    for idx, (s, e, vr, ar, wav, next_start, pad_dur, seg_end) in enumerate(segment_params):
        audio_inputs.append(str(Path(wav).resolve()))
        in_audio = idx + 1

        vl = f'v{seg_idx}'
        al = f'a{seg_idx}'
        filters.append(f"[0:v]trim=start={s}:end={e},setpts=(PTS-STARTPTS)*{vr}[{vl}]")
        # 计算淡入淡出参数（50ms，极短片段自适应缩短）
        seg_dur = (e - s) * vr
        fade_dur = min(0.05, seg_dur / 3)
        fade_out_st = seg_dur - fade_dur
        if abs(ar - 1.0) < 0.001:
            # 音频不拉伸，但可能需要补静音（音频比视频短）
            if pad_dur > 0.001:
                filters.append(f"[{in_audio}:a]anull,afade=t=in:d={fade_dur},afade=t=out:st={fade_out_st}:d={fade_dur},apad=whole_dur={seg_dur}[{al}]")
            else:
                filters.append(f"[{in_audio}:a]afade=t=in:d={fade_dur},afade=t=out:st={fade_out_st}:d={fade_dur}[{al}]")
        else:
            filters.append(f"[{in_audio}:a]rubberband=tempo={ar},afade=t=in:d={fade_dur},afade=t=out:st={fade_out_st}:d={fade_dur}[{al}]")
        concat_labels.extend([f"[{vl}]", f"[{al}]"])
        seg_idx += 1

        # 间隙段（用 segment_end 而非 effective_end 计算实际间隙）
        if next_start is not None:
            gap = next_start - seg_end
            if gap > 0.01:
                vl = f'v{seg_idx}'
                al = f'a{seg_idx}'
                filters.append(f"[0:v]trim=start={e}:end={next_start},setpts=PTS-STARTPTS[{vl}]")
                filters.append(f"aevalsrc=0.0:d={gap}:sample_rate=24000[{al}]")
                concat_labels.extend([f"[{vl}]", f"[{al}]"])
                seg_idx += 1

    total_segs = seg_idx
    filters.append(f"{''.join(concat_labels)}concat=n={total_segs}:v=1:a=1[outv][outa]")
    filter_graph = ';'.join(filters)

    # ── 执行 ──
    # 小数据集直接用命令行；大数据集用 @file（Windows 兼容写法）
    if total_segs <= 60:
        cmd = ['ffmpeg', '-y']
        for inp in audio_inputs:
            cmd.extend(['-i', str(Path(inp).resolve())])
        cmd.extend([
            '-filter_complex', filter_graph,
            '-map', '[outv]', '-map', '[outa]',
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
            '-c:a', 'aac', '-b:a', '192k', '-ar', '24000',
            str(Path(output_path).resolve())
        ])
    else:
        arg_path = Path(str(output_path) + ".ffargs.txt")
        with open(arg_path, 'w', encoding='utf-8') as f:
            for inp in audio_inputs:
                p = str(Path(inp).resolve()).replace('\\', '/')
                f.write(f"-i\n{p}\n")
            f.write(f"-filter_complex\n{filter_graph}\n")
            f.write("-map\n[outv]\n-map\n[outa]\n")
            f.write("-c:v\nlibx264\n-preset\nfast\n-crf\n18\n")
            f.write("-c:a\naac\n-b:a\n192k\n-ar\n16000\n")
            f.write(str(Path(output_path).resolve()).replace('\\', '/'))
        cmd = ['ffmpeg', '-y', f'@{arg_path.resolve()}']

    print(f"[WORK] 开始 ffmpeg ({total_segs} 段, 1 个进程)...")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    t1 = time.time()
    if total_segs > 60:
        arg_path.unlink(missing_ok=True)

    if result.returncode == 0:
        output_info = get_video_info(output_path)
        print(f"[OK] 快速管道完成 ({t1-t0:.0f}s)")
        if output_info:
            print(f"   输出文件: {output_path}")
            print(f"   输出时长: {output_info['duration']:.2f}秒")
        return True
    else:
        print(f"[ERROR] ffmpeg 失败 ({t1-t0:.0f}s): {result.stderr[-600:]}")
        return False


def process_video_with_dubbing(srt_path, video_path, dubbed_dir, output_path, workers=4, resume=False, audio_stretch=False, scene_snap=False, adaptive_speed=False):
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

    # ── 快速路径：无 RIFE 时用单命令 ffmpeg 处理全部段 ──
    if not _rife_available:
        result = _fast_pipeline(srt_path, video_path, dubbed_dir, output_path, audio_stretch, scene_snap, adaptive_speed)
        if result is not None:
            return result

    # ── 慢速路径：逐段子进程处理（RIFE 段需要）──

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

    # 5. 检测字幕间隙并生成静默音频片段
    gap_silence_files = []
    extended_items = []  # list of (kind, orig_idx, start, end, dubbed_file)
    for i, ((start, end), dubbed_file) in enumerate(zip(timestamps, dubbed_files), 1):
        extended_items.append(("normal", i, start, end, dubbed_file))
        if i < len(timestamps):
            next_start = timestamps[i][0]
            gap = next_start - end
            if gap > 0.01:
                silence_path = segments_dir / f"silence_{len(extended_items)+1:04d}.wav"
                subprocess.run(
                    [
                        'ffmpeg', '-y', '-f', 'lavfi', '-i', 'anullsrc=r=16000:cl=mono',
                        '-t', str(gap), '-acodec', 'pcm_s16le', str(silence_path)
                    ],
                    capture_output=True
                )
                gap_silence_files.append(silence_path)
                extended_items.append(("gap", None, end, next_start, silence_path))

    total_segments = len(extended_items)
    print(f"[INFO] 共 {len(timestamps)} 段字幕, 检测到 {total_segments - len(timestamps)} 个间隙, 总片段数 {total_segments}")

    # 同时检查已存在的输出文件
    for i in range(1, total_segments + 1):
        if (segments_dir / f"final_{i:04d}.mp4").exists():
            completed.add(i)

    if len(completed) > 0 and not resume:
        print(f"[INFO] 发现 {len(completed)} 个已存在的片段，将跳过")

    # 构建任务列表并预分类（应用场景边界吸附 + 两阶段调度）
    rife_tasks = []
    non_rife_tasks = []
    gap_ids = set()

    for idx, (kind, orig_idx, start, end, dubbed_file) in enumerate(extended_items, 1):
        if idx in completed:
            continue

        if kind == "gap":
            gap_ids.add(idx)
            # Gap segments: audio_stretch=False, no speed adjustment needed
            task = (idx, video_path, start, end, dubbed_file, segments_dir, False)
            non_rife_tasks.append(task)
            continue

        # Normal segment processing
        orig_start, orig_end = start, end
        if scene_snap and scene_times:
            start = snap_to_scene(start, scene_times)
            end = snap_to_scene(end, scene_times)
            if start != orig_start or end != orig_end:
                print(f"[SCENE] seg {orig_idx}: start {orig_start:.2f}→{start:.2f}, end {orig_end:.2f}→{end:.2f}")

        # 预分类：判断是否需要 RIFE GPU 插帧
        video_duration = end - start
        audio_duration = get_audio_duration(dubbed_file)
        speed_ratio = audio_duration / video_duration
        deviation = abs(speed_ratio - 1.0)

        if not audio_stretch:
            is_rife = (speed_ratio > 1.15 and _rife_available)
        elif deviation <= AUDIO_ONLY_THRESHOLD:
            is_rife = False
        else:
            split_ratio = speed_ratio ** 0.5
            is_rife = (split_ratio > 1.15 and _rife_available)

        task = (idx, video_path, start, end, dubbed_file, segments_dir, audio_stretch)
        if is_rife:
            rife_tasks.append(task)
        else:
            non_rife_tasks.append(task)

    method_counts = {
        "copy": 0, "setpts": 0, "rife": 0, "setpts-fallback": 0, "framerate": 0,
        "astretch": 0, "gap": 0,
        "bi-copy": 0, "bi-setpts": 0, "bi-rife": 0, "bi-setpts-fallback": 0, "bi-framerate": 0,
    }

    lock = threading.Lock()
    start_time = time.time()

    def _on_done(i, method):
        with lock:
            if method != "existing":
                method_counts[method] = method_counts.get(method, 0) + 1
            completed.add(i)

            # 写入检查点
            checkpoint = {
                "output": str(output_path),
                "completed": sorted(completed),
                "total": total_segments
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
                ("gap", method_counts.get("gap", 0)),
                ("copy", method_counts.get("copy", 0) + method_counts.get("bi-copy", 0)),
                ("framerate", method_counts.get("framerate", 0) + method_counts.get("bi-framerate", 0))
            ])
            remaining = avg_time * (total_segments - done_count)
            print(f"\r  完成: {done_count}/{total_segments} | {elapsed:.0f}s [{avg_time:.1f}s/段] 剩余{remaining:.0f}s | {counts_str}", end="", flush=True)

    # Phase 1: 非 RIFE 段并行处理
    if non_rife_tasks:
        print(f"[WORK] 非RIFE段并行处理 ({len(non_rife_tasks)}段, workers={workers})...")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_process_one_segment, task): task[0] for task in non_rife_tasks}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    i, method, final_path = future.result()
                    _on_done(i, method)
                except Exception as e:
                    print(f"\n[ERROR] 处理片段 {idx} 失败: {e}")
                    return False
        print()  # 换行

    # Phase 2: RIFE 段串行 GPU 处理
    if rife_tasks:
        print(f"[WORK] RIFE段串行GPU处理 ({len(rife_tasks)}段)...")
        for task in rife_tasks:
            try:
                i, method, final_path = _process_one_segment(task)
                _on_done(i, method)
            except Exception as e:
                print(f"\n[ERROR] 处理 RIFE 片段 {task[0]} 失败: {e}")
                return False
        print()  # 换行

    # 收集所有片段（按索引排序）
    adjusted_segments = []
    for i in range(1, total_segments + 1):
        final_segment = segments_dir / f"final_{i:04d}.mp4"
        if not final_segment.exists():
            print(f"[ERROR] 片段 {i} 输出文件缺失")
            return False
        adjusted_segments.append(str(final_segment))

    # 6. 合并所有段（统一 h264，concat -c copy 无重编码）
    print(f"[WORK] 策略统计: {', '.join(f'{k}={v}' for k,v in sorted(method_counts.items()) if v>0)}")
    print("[WORK] 合并所有调整后的片段...")

    concat_path = segments_dir / "concat.txt"
    with open(concat_path, 'w', encoding='utf-8') as f:
        for segment in adjusted_segments:
            abs_path = str(Path(segment).resolve()).replace(chr(92), '/')
            f.write(f"file '{abs_path}'\n")

    concat_cmd = ['ffmpeg', '-y', '-fflags', '+genpts',
        '-f', 'concat', '-safe', '0',
        '-i', str(concat_path),
        '-c:v', 'copy',
        '-c:a', 'aac', '-b:a', '192k', '-ar', '24000',
        '-vsync', '2',
        str(output_path)
    ]
    result = subprocess.run(concat_cmd, capture_output=True, text=True)

    if result.returncode == 0:
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
        description="视频配音对齐 — 将配音音频逐段对齐到原始视频，自动调整速度匹配时长",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""速度策略:
  |ratio-1| <= 0.20   → 纯音频拉伸 (视频零损耗, rubberband)
  > 0.20              → 双向分担 (音频+视频各自 sqrt(ratio))
                         减速用 RIFE GPU 插帧, 加速用 framerate

调度:
  无RIFE时  → 单 ffmpeg 命令一次性处理 (最快)
  有RIFE时  → 多线程并行非RIFE段 + GPU串行RIFE段

示例:
  # 默认模式 (快速, 无 RIFE)
  align-video video_zh.srt video.mp4 dubbing/ output.mp4

  # 高质量模式 (需 RIFE GPU)
  align-video video_zh.srt video.mp4 dubbing/ output.mp4 --rife

  # 大批量 + 断点续跑
  align-video video_zh.srt video.mp4 dubbing/ output.mp4 --resume

目录结构:
  dubbing/            ← 配音片段目录
    0001.wav             (第1段配音)
    0002.wav             (第2段配音)
    ...
  video_zh.srt        ← 字幕文件 (提供时间轴)
  video.mp4           ← 原始视频 (无音轨)
"""
    )
    parser.add_argument("srt", help="字幕文件路径 (.srt)")
    parser.add_argument("video", help="原始视频路径 (无音轨 .mp4)")
    parser.add_argument("dubbed_dir", help="配音片段目录 (含 0001.wav 0002.wav ...)")
    parser.add_argument("output", help="输出视频路径 (.mp4)")
    parser.add_argument("--workers", type=int, default=4, help="并行线程数 (默认4)")
    parser.add_argument("--resume", action="store_true", help="从 checkpoint 断点续跑")
    parser.add_argument("--audio-stretch", dest="audio_stretch", action="store_true",
                        help="启用音频拉伸 (旧模式)")
    parser.add_argument("--adaptive-speed", dest="adaptive_speed", action="store_true", default=False,
                        help="启用两级变速对齐 (全局+局部双调)")
    parser.add_argument("--scene-snap", action="store_true", help="吸附切点到场景边界")
    parser.add_argument("--rife", action="store_true",
                        help="启用 RIFE GPU 运动插帧 (默认关闭)")

    args = parser.parse_args()

    _rife_available = args.rife

    process_video_with_dubbing(
        args.srt,
        args.video,
        args.dubbed_dir,
        args.output,
        workers=args.workers,
        resume=args.resume,
        audio_stretch=args.audio_stretch,
        scene_snap=args.scene_snap,
        adaptive_speed=args.adaptive_speed
    )

def main():
    """CLI 入口 — uv tool install 后可直接 align-video 调用"""
    import sys, argparse

    parser = argparse.ArgumentParser(
        description="视频配音对齐 — 将配音音频逐段对齐到原始视频",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例:
  align-video video_zh.srt video.mp4 dubbing/ output.mp4
  align-video video_zh.srt video.mp4 dubbing/ output.mp4 --rife
  align-video video_zh.srt video.mp4 dubbing/ output.mp4 --resume

更多: align-video -h"""
    )
    parser.add_argument("srt", help="字幕文件 (.srt)")
    parser.add_argument("video", help="原始视频 (无音轨 .mp4)")
    parser.add_argument("dubbed_dir", help="配音片段目录 (含 0001.wav 0002.wav ...)")
    parser.add_argument("output", help="输出视频 (.mp4)")
    parser.add_argument("--workers", type=int, default=4, help="并行线程数 (默认4)")
    parser.add_argument("--resume", action="store_true", help="断点续跑")
    parser.add_argument("--audio-stretch", dest="audio_stretch", action="store_true",
                        help="启用音频拉伸 (视频优先模式)")
    parser.add_argument("--adaptive-speed", dest="adaptive_speed", action="store_true", default=False,
                        help="启用两级变速对齐")
    parser.add_argument("--scene-snap", action="store_true", help="吸附切点到场景边界")
    parser.add_argument("--rife", action="store_true", help="启用 RIFE GPU 插帧 (默认关闭)")

    args = parser.parse_args(sys.argv[1:]) if len(sys.argv) > 1 else parser.parse_args(['-h'])

    globals()['_rife_available'] = args.rife

    process_video_with_dubbing(
        args.srt, args.video, args.dubbed_dir, args.output,
        workers=args.workers, resume=args.resume,
        audio_stretch=args.audio_stretch,
        scene_snap=args.scene_snap,
        adaptive_speed=args.adaptive_speed
    )
