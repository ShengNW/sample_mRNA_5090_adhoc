import pandas as pd
from pathlib import Path
root = Path('.')
df = pd.read_csv(root/'data/derived/multi_model_/real_base.csv')
outdir = root/'outputs/figure_inputs'
outdir.mkdir(parents=True, exist_ok=True)
for target_name, alias, organ_id in [('Liver','liver',35), ('Adipose - Subcutaneous','adipose',0)]:
    sub = df[df['organ_name'] == target_name][['seq_id','utr5','utr3']].copy()
    if len(sub) > 50000:
        sub = sub.sample(50000, random_state=0)
    sub['organ_id'] = organ_id
    out = outdir / f'natural_{alias}.csv'
    sub.to_csv(out, index=False)
    print(alias, len(sub))
