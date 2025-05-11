import regex as re

FILENUM = "4170"
current_content = ""

with open(f"raw_files/{FILENUM}.cha", "r") as f:
    lines = f.read()
    lines = lines.replace("\t", " ")
    lines = lines.replace("*B:", "Speaker 2:")
    lines = lines.replace("*PAR0:", "Speaker 2:")
    lines = lines.replace("*A:", "Speaker 1:")
    lines = lines.replace("*PAR1:", "Speaker 1:")
    lines = re.sub(r'\x15[^)]*?\x15', '', lines)
    lines = re.sub(r'&=\w+', '', lines)
    lines = re.sub(r'&-\w+', '', lines)
    lines = re.sub(r'\[=! (\w+)*?\]', '', lines)
    lines = lines.replace(' .', '.')
    lines = lines.replace(' ?', '?')
    lines = lines.replace(' .', '.')
    lines = lines.replace(' ?', '?')
    lines = lines.replace('[/]', '-')
    lines = lines.replace('+', ' ')
    lines = lines.split('\n')
    current_speaker = None
    set_fake = False
    for idx, line in enumerate(lines):
        if line.startswith("Speaker"):
            lines = lines[idx:]
            break


    for line in lines:
        if line.startswith("Speaker"):
            set_fake = False
            content = line.split(":")[1].strip()
            if content in ['mhm.', 'um.', 'uh.', 'oh.', '.']:
                continue
            if current_speaker == line.split(":")[0]:
                current_content += ' ' + line.split(":")[1].strip()
            else: 
                current_content += '\n' + line.strip()
                
            current_speaker = line.split(":")[0]
        elif line.strip() in ['', '@End'] or '%' in line:
            set_fake = True
            continue
        else:
            if not set_fake:
                current_content += ' ' + line.strip()


with open(f"processed_files/{FILENUM}.txt", "w") as f:
    f.write(current_content)