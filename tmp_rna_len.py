import pandas as pd
import subprocess
lengths=[200,400,800,1200]
seq = pd.read_csv('outputs/figure_inputs/liver_target.csv').sort_values('pred', ascending=False).iloc[0]['utr3']
seq = str(seq).replace('\\n','').replace('\n','').upper().replace('T','U')
cmd=['/root/miniconda3/bin/RNAfold','--noPS']
for L in lengths:
    sub = seq[:L]
    out=subprocess.run(cmd, input=f">x\n{sub}\n".encode(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print('L', L, 'ret', out.returncode, 'outlen', len(out.stdout))
