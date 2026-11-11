# Two Pixels, One Grey Level — replication package

Code and per-episode records behind every number in

> **Two Pixels, One Grey Level: What a Vision-Language-Action Benchmark Can and
> Cannot Resolve** (submitted to *Autonomous Robots*)

This package contains the evaluation harness hooks, the pixel-level probes and the
statistics scripts. It does **not** contain model weights or the benchmark itself;
both are public and listed under *Dependencies* below.

## What is here

| Directory | Contents |
| --- | --- |
| `eval/` | the harness hooks that make a LIBERO-Goal episode reproducible: pinned initial states, fixed seed, fixed image quantisation, fixed numerics |
| `mechanism/` | probes that connect a single wrist-camera pixel to an action chunk |
| `analysis/` | the statistical tests and the per-run comparisons quoted in the tables |
| `figures/` | the script that regenerates every figure from `records.csv` and the dumps |
| `export_records.py` | turns a tree of `eval_info.json` files into `records.csv` |

## Paper claim to script

| Claim | Where | Script |
| --- | --- | --- |
| 300 pinned initial states, identical across launches | §3 | `eval/p0_patch.py` (`P0_PIN_INIT_STATES`) |
| two launches of one checkpoint differ on 16.7% of episodes | Table 7 | `analysis/repeats.py` |
| rounding the image to multiples of 8 lowers that to 7.0% | Table 7 | `eval/p0_patch.py` (`P0_QUANT=8`), `analysis/repeats.py` |
| agent view identical, wrist differs in 1–4 pixels | Table 8 | `mechanism/agentview_pairs.py`, `mechanism/dumps50_analysis.py` |
| exact McNemar, Wilson intervals, paired *t* | Tables 7, 12, 13 | `analysis/mcnemar.py`, `analysis/final_stats.py`, `analysis/ttest_all.py` |
| round-half-to-even predicts chunk identity 13/13 | §5 | `mechanism/quant_merge.py`, `mechanism/chunk_signs_many.py` |
| forcing one pixel to the minority value flips the verdict | §5 | `eval/closure.py` |
| OpenVLA is stable because it never sees the wrist frame | Table 14 | `eval/eval_openvla_libero.py`, `analysis/cmp_openvla_runs.py` |
| every figure | App. | `figures/make_figs.py` |

## Per-episode records

`export_records.py` walks a tree of evaluation outputs and writes

```csv
run,task,episode,success
aaq0r0_s0,0,0,1
aaq0r0_s0,0,1,1
...
```

One row per episode per run. Every proportion in the paper is a mean over these
rows, so each can be recomputed with a two-line script. The `run` column encodes
the arm, the repeat index and the seed offset exactly as the launch script named
them.

## Reproducing one number

```bash
# 1. export the per-episode records from an evaluation tree
python export_records.py /path/to/pi05/runs -o records.csv

# 2. recompute the unstable-episode count of Table 7's first row (expect 50/300)
python - <<'EOF'
import collections, csv
per = collections.defaultdict(list)
for r in csv.DictReader(open("records.csv")):
    if r["run"].startswith("aaq0r"):
        per[(r["run"], r["task"], r["episode"])].append(int(r["success"]))
unstable = sum(1 for v in per.values() if len(set(v)) > 1)
print("%d/%d = %.1f%%" % (unstable, len(per), 100.0 * unstable / len(per)))
EOF
```

The scripts in `analysis/` read the evaluation tree directly and are kept in the
form in which they were run. `records.csv` exists so that the same numbers can be
recomputed without the tree, in any language.

## Dependencies

Not redistributed here, because they are large and are maintained upstream:

- **LIBERO / LIBERO-Goal benchmark** — https://github.com/Lifelong-Robot-Learning/LIBERO
- **π0.5 checkpoint** — `lerobot/pi05_base` on the Hugging Face Hub
- **OpenVLA checkpoint** — `openvla/openvla-7b` on the Hugging Face Hub
- **lerobot** — the evaluation driver that `eval/p0_patch.py` wraps

`requirements.txt` lists the Python packages the scripts import directly.

## Environment

The results were produced on a Slurm cluster (CentOS/OpenHPC) with one GPU per
evaluation, and on a single workstation with one RTX 5090, both under bf16. The
CUDA and PyTorch versions on the two machines differ; that difference is itself
one of the observations in the paper (Table 8).

## Licence

MIT. See `LICENSE`.

## Citing

Please cite the paper, not this repository, for the claims.