import re
from pathlib import Path

def parse_srt_content(srt_path):
    """解析SRT文件内容，返回 [(index, start, end, text), ...]"""
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    entries = []
    # 分割成各个条目
    blocks = re.split(r'\n\n+', content.strip())
    
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue
        
        # 第一行是序号
        index = lines[0].strip()
        
        # 第二行是时间轴
        time_match = re.match(r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})', lines[1])
        if not time_match:
            continue
        
        start_time = time_match.group(1)
        end_time = time_match.group(2)
        
        # 剩余行是文本
        text = '\n'.join(lines[2:])
        
        entries.append({
            'index': index,
            'start': start_time,
            'end': end_time,
            'text': text
        })
    
    return entries

def time_to_seconds(time_str):
    """将时间字符串转为秒数"""
    match = re.match(r'(\d{2}):(\d{2}):(\d{2}),(\d{3})', time_str)
    if match:
        h, m, s, ms = map(int, match.groups())
        return h * 3600 + m * 60 + s + ms / 1000
    return 0

def seconds_to_time(seconds):
    """将秒数转为时间字符串 HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def get_audio_duration(audio_path):
    """获取音频时长"""
    import subprocess, json
    cmd = [
        'ffprobe', '-v', 'quiet', '-print_format', 'json',
        '-show_format', str(audio_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    data = json.loads(result.stdout)
    return float(data['format']['duration'])

def adjust_subtitle_timestamps(srt_path, dubbed_dir, output_path):
    """
    根据配音片段时长重新计算字幕时间戳
    
    Args:
        srt_path: 原字幕文件路径
        dubbed_dir: 配音片段目录
        output_path: 输出调整后的字幕文件
    """
    print("[START] 调整字幕时间戳...")
    
    # 1. 读取原字幕
    entries = parse_srt_content(srt_path)
    print(f"[INFO] 读取 {len(entries)} 条字幕")
    
    # 2. 获取配音片段
    dubbed_dir = Path(dubbed_dir)
    dubbed_files = sorted(dubbed_dir.glob("*.wav"))
    
    if len(entries) != len(dubbed_files):
        print(f"[WARN] 字幕数量({len(entries)})与配音片段({len(dubbed_files)})不匹配，取最小值")
        min_count = min(len(entries), len(dubbed_files))
        entries = entries[:min_count]
        dubbed_files = dubbed_files[:min_count]
    
    # 3. 计算新的时间戳
    current_time = 0.0
    new_entries = []
    
    for i, (entry, dubbed_file) in enumerate(zip(entries, dubbed_files), 1):
        audio_duration = get_audio_duration(dubbed_file)
        
        new_start = current_time
        new_end = current_time + audio_duration
        
        new_entries.append({
            'index': str(i),
            'start': seconds_to_time(new_start),
            'end': seconds_to_time(new_end),
            'text': entry['text']
        })
        
        current_time = new_end
    
    # 4. 写入新的字幕文件
    with open(output_path, 'w', encoding='utf-8-sig') as f:  # utf-8-sig for BOM
        for entry in new_entries:
            f.write(f"{entry['index']}\n")
            f.write(f"{entry['start']} --> {entry['end']}\n")
            f.write(f"{entry['text']}\n\n")
    
    print(f"[OK] 字幕调整完成!")
    print(f"   输出: {output_path}")
    print(f"   总时长: {seconds_to_time(current_time)}")
    
    return output_path

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 4:
        print("用法: python adjust_subtitle_timestamps.py <字幕.srt> <配音目录> <输出.srt>")
        print("示例: python adjust_subtitle_timestamps.py test/video_zh.srt test/dubbed_output test/video_zh_adjusted.srt")
        sys.exit(1)
    
    srt_path = sys.argv[1]
    dubbed_dir = sys.argv[2]
    output_path = sys.argv[3]
    
    adjust_subtitle_timestamps(srt_path, dubbed_dir, output_path)

def main():
    import sys
    if len(sys.argv) < 4:
        print("用法: adjust-subs <字幕.srt> <配音目录> <输出.srt>")
        sys.exit(1)
    adjust_subtitle_timestamps(sys.argv[1], sys.argv[2], sys.argv[3])
