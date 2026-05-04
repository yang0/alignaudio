import os
import re
import subprocess
import tempfile
from pathlib import Path
import shutil
import json

def parse_srt(srt_path):
    """解析SRT文件，返回时间戳列表 [(start_seconds, end_seconds), ...]"""
    timestamps = []
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    pattern = r'(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*--\u003e\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})'
    matches = re.findall(pattern, content)
    
    for match in matches:
        sh, sm, ss, sms, eh, em, es, ems = match
        start = int(sh) * 3600 + int(sm) * 60 + int(ss) + int(sms) / 1000
        end = int(eh) * 3600 + int(em) * 60 + int(es) + int(ems) / 1000
        timestamps.append((start, end))
    
    return timestamps

def get_audio_duration(audio_path):
    """获取音频时长"""
    cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', str(audio_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    data = json.loads(result.stdout)
    return float(data['format']['duration'])

def get_video_info(video_path):
    """获取视频信息"""
    cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', str(video_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    data = json.loads(result.stdout)
    
    for stream in data['streams']:
        if stream['codec_type'] == 'video':
            return {
                'width': int(stream['width']),
                'height': int(stream['height']),
                'fps': eval(stream['r_frame_rate']),
                'duration': float(stream.get('duration', 0))
            }
    return None

def process_video_dubbing_pipeline(srt_path, video_path, dubbed_dir, output_path, subtitle_path=None):
    """
    完整工作流：根据字幕时间戳逐段调整视频速度，匹配配音音频
    
    Args:
        srt_path: 英文字幕文件路径（提供原始时间戳）
        video_path: 原始视频路径（带音频或不带音频都可以）
        dubbed_dir: 配音片段目录
        output_path: 最终输出视频路径
        subtitle_path: 可选的中文字幕文件路径，用于嵌入
    """
    print("[START] 开始视频配音对齐处理...")
    
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
    
    print(f"[INFO] 视频: {video_info['width']}x{video_info['height']} @ {video_info['fps']}fps, 时长{video_info['duration']:.2f}s")
    
    # 4. 计算每个片段的速度比
    print("[INFO] 分析速度差异...")
    speed_ratios = []
    total_original_duration = 0
    total_dubbed_duration = 0
    
    for i, ((start, end), dubbed_file) in enumerate(zip(timestamps, dubbed_files), 1):
        video_duration = end - start
        audio_duration = get_audio_duration(dubbed_file)
        speed_ratio = audio_duration / video_duration
        speed_ratios.append(speed_ratio)
        total_original_duration += video_duration
        total_dubbed_duration += audio_duration
    
    avg_speed = total_dubbed_duration / total_original_duration
    print(f"[INFO] 原视频总时长: {total_original_duration:.2f}s")
    print(f"[INFO] 配音总时长: {total_dubbed_duration:.2f}s")
    print(f"[INFO] 平均速度比: {avg_speed:.3f}")
    print(f"[INFO] 最大加速: {min(speed_ratios):.3f}, 最大减速: {max(speed_ratios):.3f}")
    
    # 5. 创建临时目录
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        segments_dir = tmpdir / "segments"
        segments_dir.mkdir()
        
        # 6. 逐段处理
        print("[WORK] 逐段处理视频...")
        final_segments = []
        
        for i, ((start, end), dubbed_file, speed_ratio) in enumerate(zip(timestamps, dubbed_files, speed_ratios), 1):
            video_duration = end - start
            
            segment_video = segments_dir / f"segment_{i:04d}.mp4"
            adjusted_video = segments_dir / f"adjusted_{i:04d}.mp4"
            final_segment = segments_dir / f"final_{i:04d}.mp4"
            
            # 切出原视频片段 (去掉音频)
            cmd = [
                'ffmpeg', '-y', '-ss', str(start), '-t', str(video_duration),
                '-i', str(video_path),
                '-c:v', 'copy', '-an',
                str(segment_video)
            ]
            subprocess.run(cmd, capture_output=True)
            
            # 调整视频速度
            # speed_ratio = audio_duration / video_duration
            # speed_ratio < 1: audio shorter than video → need to speed UP video
            # speed_ratio > 1: audio longer than video → need to slow DOWN video
            # setpts=speed_ratio*PTS:  speed_ratio<1 → smaller PTS → faster playback → shorter duration
            cmd = [
                'ffmpeg', '-y', '-i', str(segment_video),
                '-vf', f'setpts={speed_ratio}*PTS',
                '-an',
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
                str(adjusted_video)
            ]
            subprocess.run(cmd, capture_output=True)
            
            # 合并视频片段和配音音频
            cmd = [
                'ffmpeg', '-y',
                '-i', str(adjusted_video),
                '-i', str(dubbed_file),
                '-c:v', 'copy',
                '-c:a', 'aac', '-b:a', '192k',
                '-shortest',
                str(final_segment)
            ]
            subprocess.run(cmd, capture_output=True)
            
            final_segments.append(str(final_segment))
            
            # 清理中间文件
            segment_video.unlink(missing_ok=True)
            adjusted_video.unlink(missing_ok=True)
            
            if i % 50 == 0 or i == len(timestamps):
                print(f"  进度: {i}/{len(timestamps)} - 当前片段速度比: {speed_ratio:.3f}")
        
        # 7. 合并所有段
        print("[WORK] 合并所有片段...")
        concat_file = tmpdir / "concat.txt"
        with open(concat_file, 'w') as f:
            for segment in final_segments:
                f.write(f"file '{segment}'\n")
        
        merged_video = tmpdir / "merged.mp4"
        cmd = [
            'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
            '-i', str(concat_file),
            '-c', 'copy',
            str(merged_video)
        ]
        subprocess.run(cmd, capture_output=True)
        
        # 8. 添加字幕（如果需要）
        if subtitle_path and Path(subtitle_path).exists():
            print("[WORK] Embedding subtitles...")
            subtitle_file = Path(subtitle_path)
            
            # 字幕路径中可能有特殊字符，用单引号包裹
            subtitle_escaped = str(subtitle_file).replace(':', '\\:').replace('\\', '/')
            
            cmd = [
                'ffmpeg', '-y',
                '-i', str(merged_video),
                '-vf', f"subtitles='{subtitle_escaped}'",
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
                '-c:a', 'copy',
                str(output_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"[WARN] Subtitle embedding failed, copying without subtitles")
                print(f"[DEBUG] Error: {result.stderr[:200]}")
                shutil.copy(merged_video, output_path)
        else:
            shutil.copy(merged_video, output_path)
        
        # 验证输出
        if Path(output_path).exists():
            output_info = get_video_info(output_path)
            if output_info:
                print(f"[OK] 处理完成!")
                print(f"   输出: {output_path}")
                print(f"   分辨率: {output_info['width']}x{output_info['height']}")
                print(f"   时长: {output_info['duration']:.2f}s")
                print(f"   期望配音时长: {total_dubbed_duration:.2f}s")
                return True
        
        print("[ERROR] 输出文件创建失败")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 5:
        print("视频配音对齐工具")
        print("根据字幕时间戳逐段调整视频速度，匹配配音音频时长")
        print()
        print("用法: python align_video_dubbing.py <字幕.srt> <视频.mp4> <配音目录> <输出.mp4> [字幕文件.srt]")
        print()
        print("参数:")
        print("  字幕.srt      - 英文字幕文件（提供原始时间戳）")
        print("  视频.mp4      - 原始视频文件")
        print("  配音目录      - 配音片段目录（0001.wav, 0002.wav...）")
        print("  输出.mp4      - 最终输出文件路径")
        print("  字幕文件.srt  - 可选，要嵌入的中文字幕")
        print()
        print("示例:")
        print("  python align_video_dubbing.py test/video_en.srt test/video.mp4 test/dubbed_output test/output.mp4")
        print("  python align_video_dubbing.py test/video_en.srt test/video.mp4 test/dubbed_output test/output.mp4 test/video_zh.srt")
        sys.exit(1)
    
    srt_path = sys.argv[1]
    video_path = sys.argv[2]
    dubbed_dir = sys.argv[3]
    output_path = sys.argv[4]
    subtitle_path = sys.argv[5] if len(sys.argv) > 5 else None
    
    process_video_dubbing_pipeline(srt_path, video_path, dubbed_dir, output_path, subtitle_path)
