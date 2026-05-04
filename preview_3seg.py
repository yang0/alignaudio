import subprocess, json, tempfile, shutil
from pathlib import Path

def get_dur(p):
    r = subprocess.run(['ffprobe','-v','quiet','-print_format','json','-show_format',str(p)],
                      capture_output=True, text=True)
    return float(json.loads(r.stdout)['format']['duration'])

# ── define segments ──
# (start, end) from video_en.srt, dubbed file, zh text
segments = [
    {
        'start': 0.150, 'end': 9.630,
        'audio': 'test/dubbed_output/0001.wav',
        'text': '如果你在用 ChatGPT、Claude、Claude Code、Claude Cowork 或 Codex，但不知道工作中该用哪个、怎么用，那就来对地方了。'
    },
    {
        'start': 9.790, 'end': 13.350,
        'audio': 'test/dubbed_output/0002.wav',
        'text': '因为今天这期视频，我们要聊的是如何高效使用 AI 智能体。'
    },
    {
        'start': 13.930, 'end': 24.330,
        'audio': 'test/dubbed_output/0003.wav',
        'text': '具体来说，我们要讲的是 Codex，OpenAI 的 AI 智能体工具。它类似 Claude Code，但界面更友好，既能做知识工作，也能做编码工作。'
    },
]

tmp = Path(tempfile.mkdtemp())
print(f"[TMP] {tmp}")

finals = []
cum_time = 0.0
sub_entries = []

for i, seg in enumerate(segments, 1):
    ad = get_dur(seg['audio'])
    vd = seg['end'] - seg['start']
    ratio = ad / vd
    print(f"Seg {i}: video={vd:.2f}s  audio={ad:.2f}s  ratio={ratio:.3f}")

    # 1) cut video
    cut = tmp / f"cut_{i}.mp4"
    subprocess.run(
        ['ffmpeg','-y','-ss',str(seg['start']),'-t',str(vd),
         '-i','test/video_no_audio.mp4','-c:v','copy','-an',str(cut)],
        capture_output=True)

    # 2) adjust speed (setpts=ratio*PTS slows down when ratio>1)
    adj = tmp / f"adj_{i}.mp4"
    subprocess.run(
        ['ffmpeg','-y','-i',str(cut),
         '-vf',f'setpts={ratio}*PTS',
         '-an','-c:v','libx264','-preset','fast','-crf','18',
         str(adj)],
        capture_output=True)

    # 3) merge video+audio
    fin = tmp / f"fin_{i}.mp4"
    subprocess.run(
        ['ffmpeg','-y','-i',str(adj),'-i',seg['audio'],
         '-c:v','copy','-c:a','aac','-b:a','192k','-shortest',
         str(fin)],
        capture_output=True)
    finals.append(str(fin))

    # subtitle entry with adjusted timestamps
    sub_entries.append((cum_time, cum_time + ad, seg['text']))
    cum_time += ad
    print(f"  -> new sub: {cum_time-ad:.2f}s - {cum_time:.2f}s")

# 4) concat all 3 segments
concat_list = tmp / "concat.txt"
with open(concat_list, 'w') as f:
    for p in finals:
        f.write(f"file '{p}'\n")

merged = 'test/preview_3seg_merged.mp4'
subprocess.run(
    ['ffmpeg','-y','-f','concat','-safe','0','-i',str(concat_list),
     '-c','copy', merged],
    capture_output=True)
print(f"[OK] Merged video: {merged}")

# 5) write adjusted SRT (Chinese subs with new timestamps)
def sec2ts(s):
    h = int(s//3600)
    m = int((s%3600)//60)
    sec = int(s%60)
    ms = int((s%1)*1000)
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"

srt_path = 'test/preview_3seg.srt'
with open(srt_path, 'w', encoding='utf-8-sig') as f:
    for i, (st, en, txt) in enumerate(sub_entries, 1):
        f.write(f"{i}\n{sec2ts(st)} --> {sec2ts(en)}\n{txt}\n\n")
print(f"[OK] Subtitle: {srt_path}")

# 6) add soft subtitles to video
final_out = 'test/preview_3seg_final.mp4'
subprocess.run(
    ['ffmpeg','-y','-i',merged,'-i',srt_path,
     '-c:v','copy','-c:a','copy',
     '-c:s','mov_text',
     '-map','0:v','-map','0:a','-map','1:s',
     '-metadata:s:s:0','language=zho',
     final_out],
    capture_output=True)
print(f"[OK] Final: {final_out}")

# 7) verify
vd = get_dur(final_out)
print(f"\n=== DONE ===")
print(f"Output: {final_out}")
print(f"Duration: {vd:.2f}s")
print(f"Expected audio total: {cum_time:.2f}s")

# cleanup
shutil.rmtree(tmp)
