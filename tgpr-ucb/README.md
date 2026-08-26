# TGPR-UCB

Tree-Guided Policy Refinement with UCB1-Tuned and Distributional PR Rewards.

## Setup

```bash
pip install -e .
```

## Data

Put training data in `data/train.jsonl` and validated corpus in `data/validated_corpus.jsonl`.

Format:
```json
{"prompt": "import pandas as pd\n# Load data\n", "completion": "df = pd.read_csv('data.csv')\n"}
```

## Training

```bash
bash scripts/train_small.sh
bash scripts/train_7b.sh
bash scripts/train_13b.sh
bash scripts/train_20b.sh
```
