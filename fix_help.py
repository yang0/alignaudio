with open('align_video_to_dubbing.py', 'r', encoding='utf-8') as f:
    code = f.read()

# Replace help text for short-mode
import re
code = re.sub(
    r'help="trim:.*?[^\\])"\)',
    'help="trim(default) | apad | speedup")',
    code
)

with open('align_video_to_dubbing.py', 'w', encoding='utf-8') as f:
    f.write(code)
print('Fixed')
