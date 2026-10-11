# Nidus: a virtual bone marrow niche

Official code and model release for **Nidus**, a 69M-parameter masked discrete diffusion transformer trained on 18.9 million bone-marrow-niche cells from 431 public datasets. Nidus generates complete cells across 130 niche cell types and 19 disease/treatment conditions, executes zero-shot knockout of any sufficiently represented panel gene, screens target-annotated compounds by target knockout, and can be operated conversationally by an LLM agent.

## Repository layout

```
model/          model definition, inference loader, KO library, gene panel and reference packs
agent/          conversational agent (LLM tool-calling layer) and BoneBench benchmark
evaluation/     control-calibration and evaluation jobs with input gene lists
```

Model weights (`pmdm_v3_oponly_step2000.pt`, 276 MB) are distributed in [Releases](../../releases) because they exceed GitHub's 100 MB file limit. The file is split into 14 parts; run

```bash
bash tools/download_weights.sh model/
```

to download all parts, reassemble them into `model/pmdm_v3_oponly_step2000.pt` and verify the MD5 checksum (`1c6c09b56401936331ae91f674394fb3`). On Windows without Git Bash, download all 14 `*.partXX` files manually and merge them with `copy /b pmdm_v3_oponly_step2000.pt.part00+part01+...+part13 pmdm_v3_oponly_step2000.pt`.

## Installation

```bash
pip install -r requirements.txt
```

Download the weights from the latest release (see above) and place the reassembled file in `model/`:

```
model/pmdm_v3_oponly_step2000.pt
```

## Quickstart

### Load the model

```python
import sys
sys.path.insert(0, "model")
from nidus_light_lib import load_light

env = load_light(base="model", dev="cuda")   # dev="cpu" also works, slower
```

### Generate cells

Use the generation helpers in `model/nidus_kofix_lib.py`; 250 cells per context take roughly one minute on a single consumer GPU.

### Zero-shot knockout

```python
from nidus_kofix_lib import gen_ko
# knockout of SP7 in the osteoblast context, 10 seeds x 250 cells
result = gen_ko(env, gene="SP7", cell_type="osteoblast", seeds=10, n_cells=250)
```

Any gene in `model/bone_genes_v4.txt` (16,105 panel genes) can be knocked out in sufficiently represented cell types, including genes never seen in perturbation data.

### Compound screening by target knockout

Annotate a compound with its protein targets, action type and potency, and knock the targets out in the relevant niche cell types. The 4,277-compound annotation library used in the paper was assembled from ChEMBL and DrugBank; see the manuscript Methods.

### Control calibration

Every knockout claim in the paper is reported against a 250-gene negative-control calibration (`evaluation/job_ctrl250.py`, `evaluation/job_ctrl3ct.py`). We recommend running the same calibration before interpreting any new knockout: control-subtracted deltas (ΔΔ) and control-calibrated percentiles are defined in the manuscript Methods.

### Conversational agent

```bash
cd agent
cp config.example.json config.json   # then add your own LLM API key
python bench_runner_v10.py            # runs BoneBench against the local model worker
```

`agent/bonebench_v4.json` contains the benchmark questions with per-question literature or database sources. The agent talks to any OpenAI-compatible LLM endpoint; the model worker (`server_worker_v3.py`) executes generation, knockout and scoring tools on a GPU machine.

## Data availability

The training corpus was assembled from public sources (GEO, CELLxGENE Census and others); dataset-level identifiers and processing steps are described in the manuscript and its supplementary tables. Reference packs needed for inference (`gene_mean_by_ct.npz`, `z0_pack.npz`) are included in `model/`.

## Citation

If you use Nidus, please cite the manuscript (details to be added upon publication).

## License

MIT (code). Model weights are released for research use.
