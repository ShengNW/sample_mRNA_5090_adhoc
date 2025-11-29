import math
import subprocess
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path('.').resolve()
FIG_DIR = ROOT / 'figs' / 'final'
DATA_DIR = FIG_DIR / 'fig_data'
FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


def top_seq(target_path: Path, max_len: int = 1200):
    df = pd.read_csv(target_path)
    top = df.sort_values('pred', ascending=False).iloc[0]
    seq_id = str(top['seq_id']).replace('\n',' ').replace('\\n',' ').split()[0]
    seq = str(top['utr3']).replace('\n','').replace('\\n','').upper().replace('T','U')
    seq = ''.join(ch for ch in seq if ch in {'A','C','G','U'})
    if len(seq) > max_len:
        seq = seq[:max_len]
    return seq_id, seq


def run_fold(seq_id: str, seq: str):
    fasta = f'>{seq_id}\n{seq}\n'
    res = subprocess.run(['/root/miniconda3/bin/RNAfold','--noPS'], input=fasta, text=True, capture_output=True)
    if res.returncode != 0:
        raise RuntimeError(res.stderr)
    lines = [ln.strip() for ln in res.stdout.splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError('empty stdout')
    # pick last line containing parentheses
    struct_line = None
    for ln in reversed(lines):
        if '(' in ln and ')' in ln:
            struct_line = ln
            break
    if struct_line is None:
        raise RuntimeError(f'no structure line: {lines[-3:]}')
    struct, energy = struct_line.split(' (')
    energy = float(energy.strip('()'))
    return struct, energy


def pairs(db: str):
    st=[]; ps=[]
    for i,ch in enumerate(db):
        if ch=='(':
            st.append(i)
        elif ch==')' and st:
            j=st.pop(); ps.append((j,i))
    return ps


def plot(struct: str, mfe: float, seq_id: str, prefix: str):
    ps = pairs(struct)
    plt.figure(figsize=(8,4))
    for i,j in ps:
        mid=(i+j)/2
        r=(j-i)/2
        theta=[k*math.pi/49 for k in range(50)]
        x=[mid + r*math.cos(t) for t in theta]
        y=[r*math.sin(t) for t in theta]
        plt.plot(x,y,color='#4c72b0',linewidth=0.8)
    plt.axhline(0,color='black',linewidth=0.5)
    plt.xlim(0,len(struct))
    plt.ylim(0,None)
    plt.xlabel('Position')
    plt.ylabel('Base-pair arc')
    plt.title(f'{prefix} ({seq_id}) MFE={mfe:.2f}')
    plt.tight_layout()
    plt.savefig(FIG_DIR/f'{prefix}.png', dpi=300)
    plt.savefig(FIG_DIR/f'{prefix}.svg')
    plt.close()
    pd.DataFrame({'seq_id':[seq_id],'dot_bracket':[struct],'mfe':[mfe]}).to_csv(DATA_DIR/f'{prefix}.structure.csv',index=False)


def main():
    for organ,file in [('liver','outputs/figure_inputs/liver_target.csv'),('adipose','outputs/figure_inputs/adipose_target.csv')]:
        seq_id, seq = top_seq(Path(file), max_len=1200)
        struct, mfe = run_fold(seq_id, seq)
        prefix=f'fig3H_{organ}_structure'
        plot(struct, mfe, seq_id, prefix)
        (FIG_DIR/f'{prefix}.fa').write_text(f'>{seq_id}\n{seq}\n')
        print('done', organ, 'len', len(seq), 'mfe', mfe)

if __name__=='__main__':
    main()
