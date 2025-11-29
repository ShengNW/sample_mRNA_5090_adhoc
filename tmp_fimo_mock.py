from pathlib import Path
import re

meme_path = Path('data/external/rbp/Ray2013_rbp_Homo_sapiens.meme')
lines = meme_path.read_text().splitlines()
mat = []
w = None
for idx, line in enumerate(lines):
    if line.startswith('MOTIF '):
        start = idx
        break
else:
    raise SystemExit('no MOTIF found')
for j in range(start + 1, len(lines)):
    line = lines[j]
    if line.startswith('letter-probability matrix'):
        m = re.search(r"w=\s*(\d+)", line)
        w = int(m.group(1)) if m else None
        for k in range(j + 1, j + 1 + (w or 0)):
            parts = lines[k].strip().split()
            if len(parts) < 4:
                break
            probs = list(map(float, parts[:4]))
            mat.append(probs)
        break
if not mat:
    raise SystemExit('no matrix rows')
bases = ['A', 'C', 'G', 'T']
cons = ''.join(bases[max(range(4), key=lambda idx: row[idx])] for row in mat)
fa = Path('outputs/analysis/fimo_mock.fa')
fa.write_text(f">mock\n{cons}\n")
print('consensus', cons)
