# M1 (Mutation/Recombination Search) Code Reference

## Prompt M1-1: Entry Point and Call Flow
- **Entry script:** `src/gen/mutate_search.py` provides the `main()` CLI that runs the search loop and exports results.【F:src/gen/mutate_search.py†L70-L139】
- **Load seed pool:** seeds are read from `cfg["io"]["seed_csv"]` if present; otherwise random sequences are sampled with `sample_seed_pairs` using manifest-defined lengths and the target organ ID.【F:src/gen/mutate_search.py†L86-L105】
- **Propose candidates:** each generation builds `children` via either recombination (`recombine`) or random mutation (`rand_mutate`) of parents selected from the current population (or random seeds if empty).【F:src/gen/mutate_search.py†L108-L118】
- **Constraint filtering:** `constraint_ok` enforces GC bounds, homopolymer run limits, and banned motifs; only passing children are kept, and the initial population is also filtered with this rule.【F:src/gen/mutate_search.py†L98-L105】【F:src/gen/mutate_search.py†L117-L118】
- **Predictor scoring:** candidates (parents + children) are written to `_cand.csv`, scored by calling `python -m src.side.predict` via `score_batch`, and optionally adjusted with an RNAfold-derived MFE penalty from `maybe_mfe_penalty`.【F:src/gen/mutate_search.py†L120-L125】
- **Top-k selection per step:** scores are combined as `expr_weight * pred + mfe_weight * mfe`, sorted, and truncated to `population_size` to form the next generation; the best-so-far record is also updated.【F:src/gen/mutate_search.py†L123-L131】
- **Final output:** after the loop, the last scored table is sorted again and the top-K (CLI override or `io.topk_export`) with `utr5/utr3/organ_id/score_pred` columns are written to `m1_topk.csv`.【F:src/gen/mutate_search.py†L131-L136】

## Prompt M1-2: Mutation/Recombination Operators and Hyperparameters
- **Random substitution mutation (`rand_mutate`)**: selects `max(1, int(L * mutation_rate))` unique positions uniformly without replacement and replaces each nucleotide with a different random base from `AUGC`. The rate is `cfg["mutation_rate"]` (default 0.08).【F:src/gen/mutate_search.py†L11-L18】【F:configs/m1_search.yaml†L8-L9】
- **Single-point crossover recombination (`recombine`)**: chooses a random crossover index `k` in `[1, len(seq)-1]` and swaps suffixes between two parents, applied independently to 5' and 3' sequences. Triggered with probability `recomb_rate` (default 0.25) when at least two parents exist.【F:src/gen/mutate_search.py†L19-L22】【F:src/gen/mutate_search.py†L108-L114】【F:configs/m1_search.yaml†L8-L10】
- **Parent selection:** parents are drawn uniformly at random from the current population; if the population is empty, fresh random seeds are generated for mutation.【F:src/gen/mutate_search.py†L108-L116】
- **Other hyperparameters:** population size (`population_size`), number of evolutionary steps (`steps`), RNG seed (`seed`), and output top-K size (`io.topk_export` or CLI override) are defined in `configs/m1_search.yaml`.【F:configs/m1_search.yaml†L2-L23】
- **Special handling:** no motif preservation beyond later constraint filtering; mutation/recombination operators apply to full sequences with no protected regions.

## Prompt M1-3: Constraints (Hard vs. Soft)
- **Hard filters inside `constraint_ok`:**
  - **GC content** must lie between `constraint.gc_min` and `constraint.gc_max` over the concatenated 5'/3' sequence.【F:src/gen/mutate_search.py†L98-L101】【F:configs/m1_search.yaml†L11-L15】
  - **Homopolymer limit:** maximum run length enforced by `ban_homopolymers` (`constraint.ban_homopolymers`).【F:src/gen/mutate_search.py†L28-L36】【F:src/gen/mutate_search.py†L101-L103】【F:configs/m1_search.yaml†L11-L15】
  - **Banned motifs:** any motif in `constraint.ban_motifs` causes rejection (`has_banned_motif`).【F:src/gen/mutate_search.py†L38-L39】【F:src/gen/mutate_search.py†L101-L103】【F:configs/m1_search.yaml†L11-L15】
- **Soft penalties:** an MFE bonus/penalty is added to the score when RNAfold is available. The predicted expression score is weighted by `score.expr_weight`; MFE is weighted by `score.mfe_weight`. Although `specificity_weight` exists in the config, it is unused in code, and no off-target or diversity penalty is applied.【F:src/gen/mutate_search.py†L120-L127】【F:configs/m1_search.yaml†L16-L19】

## Prompt M1-4: Top-K Definition and Ranking
- **Score used:** `total = expr_weight * pred + mfe_weight * mfe`; only the target-organ prediction (`pred` from `src.side.predict`) is used—no off-target penalty or specificity term despite the config field.【F:src/gen/mutate_search.py†L123-L127】【F:configs/m1_search.yaml†L16-L19】
- **Deduplication/diversity:** candidates are not deduplicated, and there is no diversity filter (no clustering/edit-distance checks); population is simply truncated by score.【F:src/gen/mutate_search.py†L120-L128】
- **Selection size and tie-handling:** each generation keeps the top `population_size` sorted by `total` (descending). Final export takes the top-K defined by CLI `--topk` or `io.topk_export` (default 2000), again sorting by `total` descending; Pandas’ `sort_values` stable ordering determines tie resolution (original order preserved).【F:src/gen/mutate_search.py†L123-L136】【F:configs/m1_search.yaml†L20-L23】
