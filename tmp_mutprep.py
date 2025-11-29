import pandas as pd
from pathlib import Path
root = Path('.')
dl = pd.read_csv(root/'outputs/figure_inputs/liver_target.csv')
top = dl.sort_values('pred', ascending=False).iloc[0]
out = pd.DataFrame([
    {'seq_id': f"{top['seq_id']}_liverctx", 'utr5': top['utr5'], 'utr3': top['utr3'], 'organ_id': 35, 'pred': top['pred']},
    {'seq_id': f"{top['seq_id']}_adiposectx", 'utr5': top['utr5'], 'utr3': top['utr3'], 'organ_id': 0, 'pred': top['pred']},
])
path = root/'outputs/figure_inputs/mutscan_topseq.csv'
out.to_csv(path, index=False)
print('wrote', path)
