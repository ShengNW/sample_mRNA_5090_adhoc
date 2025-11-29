import pandas as pd
import subprocess

seq = pd.read_csv('outputs/figure_inputs/liver_target.csv').sort_values('pred', ascending=False).iloc[0]['utr3']
seq = str(seq).replace('\\n','').replace('\n','').upper().replace('T','U')
seq_short = seq[:200]
cmd=['/root/miniconda3/bin/RNAfold','--noPS']
out=subprocess.run(cmd, input=f">x\n{seq_short}\n".encode(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
print('ret', out.returncode, 'outlen', len(out.stdout))
print(out.stdout.decode())
print('stderr', out.stderr.decode())
