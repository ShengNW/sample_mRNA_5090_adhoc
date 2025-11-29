import pandas as pd
from pathlib import Path

def write_clean_fa(target_file, fasta_path):
    df = pd.read_csv(target_file)
    top = df.sort_values('pred', ascending=False).iloc[0]
    seq_id = str(top['seq_id']).replace('\\n',' ').replace('\n',' ').split()[0]
    seq = str(top['utr3']).replace('\\n','').replace('\n','').upper().replace('T','U')
    seq = ''.join(ch for ch in seq if ch in {'A','C','G','U'})
    Path(fasta_path).write_text(f">{seq_id}\n{seq}\n")

def main():
    write_clean_fa('outputs/figure_inputs/liver_target.csv', 'figs/final/fig3G_liver_top1.fa')
    write_clean_fa('outputs/figure_inputs/adipose_target.csv', 'figs/final/fig3G_adipose_top1.fa')
    print('rewrote fastas')

if __name__ == '__main__':
    main()
