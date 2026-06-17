# Local Experiment Record

This file replaces the git-based logging from `program.md` for this workspace.

## Baseline

- Status: complete
- Script: `train_gpt.py` with local screening config only
- Result vs baseline: baseline
- Note: the repo baseline remains the full-scale default shard glob unless `TRAIN_FILES` or `VAL_FILES` are explicitly overridden for local subset experiments.
- Local screening command:
  `VAL_LOSS_EVERY=0 TRAIN_LOG_EVERY=10 WARMUP_STEPS=0 TRAIN_BATCH_TOKENS=32768 TRAIN_SEQ_LEN=128 ITERATIONS=300 VAL_MAX_TOKENS=1048576 TRAIN_FILES='./data/datasets/fineweb10B_sp1024/fineweb_train_00000[0-1].bin' VAL_FILES='./data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin' python train_gpt.py`
- Local baseline result:
  `final_int8_zlib_roundtrip_exact val_loss:3.46669979 val_bpb:2.07706185`
- Local baseline peak memory:
  `1034 MiB allocated, 1710 MiB reserved`

## Experiments

| Experiment | Change | val_bpb | peak_vram_mb | Result vs baseline | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| config-only | Added `TRAIN_FILES` and `VAL_FILES` env overrides in `train_gpt.py` | n/a | n/a | n/a | keep | Input selection only; full-scale behavior is unchanged when overrides are unset. |
| config-only | Added `VAL_MAX_TOKENS` env override in `train_gpt.py` | n/a | n/a | n/a | keep | Local screening can use smaller validation slices; full-scale behavior is unchanged when unset. |
| 1 | Local screening baseline | 2.07706185 | 1034 | baseline | keep | 300 steps, seq 128, batch 32768, 2 train shards, 1M validation tokens. |
| 2 | Increase default `matrix_lr` from `0.04` to `0.045` | 2.07971266 | 1034 | above baseline | discard | Slightly worse on the same local screening setup. |
| 3 | Increase default `beta2` from `0.95` to `0.98` | 2.08832099 | 1034 | above baseline | discard | Stronger Adam smoothing hurt the short local run. |
| 4 | Decrease default `beta1` from `0.90` to `0.85` | 2.07977009 | 1034 | above baseline | discard | Slightly worse than baseline, despite a small speedup. |
| 5 | Decrease default `muon_momentum` from `0.95` to `0.90` | 2.06727499 | 1034 | below baseline | keep | First clear local win; also reduced reserved memory to 1664 MiB. |
| 6 | Decrease default `muon_momentum` from `0.90` to `0.88` | 2.06370682 | 1034 | below baseline | keep | Better than the kept `0.90` run on the same local screen. |
| 7 | Decrease default `muon_momentum` from `0.88` to `0.86` | 2.05936513 | 1034 | below baseline | keep | Best result so far on the local screening setup. |
| 8 | Decrease default `muon_momentum` from `0.86` to `0.84` | 2.05813821 | 1034 | below baseline | keep | Small additional gain over the kept `0.86` run. |
| 9 | Decrease default `muon_momentum` from `0.84` to `0.82` | 2.05332927 | 1034 | below baseline | keep | Another clear gain; sweep is still improving. |
| 10 | Decrease default `muon_momentum` from `0.82` to `0.80` | 2.05047629 | 1034 | below baseline | keep | Best result so far; momentum sweep still trending down. |
| 11 | Decrease default `muon_momentum` from `0.80` to `0.78` | 2.04724883 | 1034 | below baseline | keep | Best result so far; sweep still has not turned over. |
| 12 | Decrease default `muon_momentum` from `0.78` to `0.76` | 2.04640683 | 1034 | below baseline | keep | Small additional improvement over the kept `0.78` run. |
| 13 | Decrease default `muon_momentum` from `0.76` to `0.74` | 2.04543666 | 1034 | below baseline | keep | Another small improvement; sweep still trending down. |
| 14 | Decrease default `muon_momentum` from `0.74` to `0.72` | 2.04487159 | 1034 | below baseline | keep | Small additional gain over the kept `0.74` run. |
| 15 | Decrease default `muon_momentum` from `0.72` to `0.70` | 2.04047331 | 1034 | below baseline | keep | Larger gain than the last few steps; best result so far. |
| 16 | Decrease default `muon_momentum` from `0.70` to `0.68` | 2.03872144 | 1034 | below baseline | keep | Best result so far; sweep still improving. |
| 17 | Decrease default `muon_momentum` from `0.68` to `0.66` | 2.04001913 | 1034 | below baseline | discard | Slightly worse than the kept `0.68` point, so the sweep likely turned over. |
| 18 | Increase default `muon_momentum` from `0.68` to `0.69` | 2.04025212 | 1034 | below baseline | discard | Also worse than the kept `0.68` point, so the local optimum looks near `0.68`. |
| 19 | Decrease default `scalar_lr` from `0.04` to `0.035` with kept `muon_momentum=0.68` | 2.03610324 | 1034 | below baseline | keep | New best local result. |
| 20 | Decrease default `matrix_lr` from `0.04` to `0.035` on top of experiment 19 | 2.03769698 | 1034 | below baseline | discard | Slight regression from the kept experiment 19 setting. |
| 21 | Decrease default `tied_embed_lr` from `0.05` to `0.04` on top of experiment 19 | 2.02946620 | 1034 | below baseline | keep | Large gain; new best local result by a clear margin. |
| 22 | Decrease default `tied_embed_lr` from `0.04` to `0.03` on top of experiment 21 | 2.03265043 | 1034 | below baseline | discard | Better than baseline, but worse than the kept experiment 21 setting. |
| 23 | Decrease default `beta1` from `0.90` to `0.85` on top of experiment 21 | 2.05094033 | 1034 | below baseline | discard | Much worse than the kept experiment 21 setting. |
| 24 | Decrease default `beta2` from `0.95` to `0.90` on top of experiment 21 | 2.02370838 | 1034 | below baseline | keep | Large gain; new best local result. |
| 25 | Decrease default `beta2` from `0.90` to `0.88` on top of experiment 24 | 2.02171375 | 1034 | below baseline | keep | Another clear improvement; new best local result. |
| 26 | Decrease default `beta2` from `0.88` to `0.86` on top of experiment 25 | 2.02132694 | 1034 | below baseline | keep | Small additional improvement; new best local result. |
| 27 | Decrease default `beta2` from `0.86` to `0.84` on top of experiment 26 | 2.01731477 | 1034 | below baseline | keep | Clear improvement; new best local result. |
| 28 | Decrease default `beta2` from `0.84` to `0.82` on top of experiment 27 | 2.01788894 | 1034 | below baseline | discard | Slight regression from the kept experiment 27 setting. |
| 29 | Flash attention enabled | 1.54856050 | skip | below baseline | 3 shards used 
| 29 | Flash attention enabled | 1.55824411 | skip | below baseline | 7 shards used 
