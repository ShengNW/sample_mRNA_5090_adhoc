from pathlib import Path
import subprocess
fa = Path('figs/final/fig3G_liver_top1.fa')
out = subprocess.run(['/root/miniconda3/bin/RNAfold','--noPS'], input=fa.read_text().encode(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
print('ret', out.returncode)
print('stdout head', out.stdout.decode()[:200])
print('stderr', out.stderr.decode())
