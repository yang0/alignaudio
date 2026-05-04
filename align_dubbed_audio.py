import os
import re
import subprocess
import tempfile
from pathlib import Path

def parse_srt(srt_path):
    """解析SRT文件，返回时间戳列表 [(start_seconds, end_seconds), ...]"""
    timestamps = []
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 匹配时间轴行: 00:00:00,150 --> 00:00:09,630
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
    import json
    data = json.loads(result.stdout)
    return float(data['format']['duration'])

def stretch_audio(input_path, output_path, target_duration, original_duration):
    """使用ffmpeg调整音频速度，使其匹配目标时长"""
    # 计算速度比: 如果配音比字幕长，需要加速(speed > 1)，反之减速
    speed = original_duration / target_duration
    
    # 使用 atempo 滤镜调整速度
    # atempo 支持范围 0.5-2.0，超出范围需要链式调用
    filters = []
    remaining_speed = speed
    
    while remaining_speed > 2.0:
        filters.append("atempo=2.0")
        remaining_speed /= 2.0
    while remaining_speed < 0.5:
        filters.append("atempo=0.5")
        remaining_speed /= 0.5
    
    filters.append(f"atempo={remaining_speed:.4f}")
    filter_chain = ",".join(filters)
    
    cmd = [
        'ffmpeg', '-y', '-i', str(input_path),
        '-af', filter_chain,
        '-ar', '16000', '-ac', '1',
        str(output_path)
    ]
    subprocess.run(cmd, capture_output=True)
    
    return output_path

def align_dubbed_audio(srt_path, dubbed_dir, output_path):
    """
    根据字幕时间戳对齐配音音频
    
    Args:
        srt_path: 字幕文件路径
        dubbed_dir: 配音片段目录 (包含 0001.wav, 0002.wav, ...)
        output_path: 输出对齐后的完整音频路径
    """
    print(f"[START] 开始对齐配音音频到字幕时间轴...")
    
    # 1. 解析字幕时间戳
    timestamps = parse_srt(srt_path)
    print(f"[INFO] 解析到 {len(timestamps)} 条字幕时间戳")
    
    # 2. 查找配音片段
    dubbed_dir = Path(dubbed_dir)
    dubbed_files = sorted(dubbed_dir.glob("*.wav"))
    print(f"[INFO] 找到 {len(dubbed_files)} 个配音片段")
    
    if len(timestamps) != len(dubbed_files):
        print(f"[WARN] 警告: 字幕数量({len(timestamps)})与配音片段数量({len(dubbed_files)})不匹配")
        min_count = min(len(timestamps), len(dubbed_files))
        timestamps = timestamps[:min_count]
        dubbed_files = dubbed_files[:min_count]
    
    # 3. 创建临时目录存放调整后的片段
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        adjusted_files = []
        concat_list = tmpdir / "concat_list.txt"
        
        # 4. 逐个调整配音片段速度并定位
        for i, ((start, end), dubbed_file) in enumerate(zip(timestamps, dubbed_files), 1):
            target_duration = end - start
            original_duration = get_audio_duration(dubbed_file)
            
            # 计算需要的速度调整
            speed_ratio = original_duration / target_duration
            
            # 输出调整后的片段（先拉伸/压缩速度）
            adjusted_file = tmpdir / f"adjusted_{i:04d}.wav"
            
            if 0.8 <= speed_ratio <= 1.25:
                # 速度变化在20%以内，直接atempo调整
                stretch_audio(dubbed_file, adjusted_file, target_duration, original_duration)
            else:
                # 速度变化较大，使用rubberband进行高质量时间拉伸
                cmd = [
                    'ffmpeg', '-y', '-i', str(dubbed_file),
                    '-af', f'rubberband=tempo={1/speed_ratio:.4f}',
                    '-ar', '16000', '-ac', '1',
                    str(adjusted_file)
                ]
                result = subprocess.run(cmd, capture_output=True)
                if result.returncode != 0:
                    # rubberband可能不可用，fallback到atempo
                    stretch_audio(dubbed_file, adjusted_file, target_duration, original_duration)
            
            adjusted_files.append({
                'file': adjusted_file,
                'start': start,
                'duration': target_duration
            })
            
            if i % 50 == 0:
                print(f"  进度: {i}/{len(timestamps)}")
        
        # 5. 创建静音垫片填充空隙
        print("[WORK] 生成静音垫片并合并音频...")
        
        # 创建concat demuxer输入文件
        with open(concat_list, 'w', encoding='utf-8') as f:
            current_time = 0.0
            for i, item in enumerate(adjusted_files, 1):
                start = item['start']
                
                # 如果当前时间早于片段开始时间，插入静音
                if current_time < start - 0.001:  # 允许1ms误差
                    silence_duration = start - current_time
                    silence_file = tmpdir / f"silence_{i:04d}.wav"
                    
                    cmd = [
                        'ffmpeg', '-y', '-f', 'lavfi', '-i',
                        f'anullsrc=r=16000:cl=mono', '-t', str(silence_duration),
                        '-acodec', 'pcm_s16le', str(silence_file)
                    ]
                    subprocess.run(cmd, capture_output=True)
                    f.write(f"file '{silence_file}'\n")
                    current_time = start
                
                # 写入调整后的配音片段
                f.write(f"file '{item['file']}'\n")
                current_time += item['duration']
        
        # 6. 使用concat demuxer合并所有片段
        cmd = [
            'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
            '-i', str(concat_list),
            '-c', 'copy',
            str(output_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            final_duration = get_audio_duration(output_path)
            print(f"[OK] 对齐完成!")
            print(f"   输出文件: {output_path}")
            print(f"   音频时长: {final_duration:.2f} 秒")
            print(f"   字幕结束: {timestamps[-1][1]:.2f} 秒")
            return True
        else:
            print(f"[FAIL] 合并失败: {result.stderr}")
            return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 4:
        print("用法: python align_dubbed_audio.py <字幕.srt> <配音目录> <输出.wav>")
        print("示例: python align_dubbed_audio.py test/video_zh.srt test/dubbed_output test/aligned_dubbed.wav")
        sys.exit(1)
    
    srt_path = sys.argv[1]
    dubbed_dir = sys.argv[2]
    output_path = sys.argv[3]
    
    align_dubbed_audio(srt_path, dubbed_dir, output_path)

def main():
    import sys
    if len(sys.argv) < 4:
        print("用法: align-audio <字幕.srt> <配音目录> <输出.wav>")
        sys.exit(1)
    align_dubbed_audio(sys.argv[1], sys.argv[2], sys.argv[3])
