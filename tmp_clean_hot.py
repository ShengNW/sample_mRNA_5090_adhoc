from pathlib import Path

def clean_hot(path):
    text = Path(path).read_text()
    # Replace literal '\n' with real newlines
    text = text.replace('\\n', '\n')
    lines_out = []
    for chunk in text.strip().split('\n>'):
        if not chunk:
            continue
        if chunk.startswith('>'):
            chunk = chunk[1:]
        parts = chunk.split('\n')
        header = parts[0].strip()
        seq = ''.join(parts[1:]).replace('\n', '').replace(' ', '')
        if not header:
            continue
        lines_out.append('>' + header)
        lines_out.append(seq)
    Path(path).write_text('\n'.join(lines_out) + '\n')

clean_hot('outputs/analysis/hot_windows.fa')
print('cleaned hot_windows')
