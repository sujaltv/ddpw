# DDP

## Setup

```bash
# with conda
conda env create --file environment.yaml
# with pip
pip install -r requirements.txt
```

## Snapshot

```bash
conda env export --no-build > environment.yaml
pip list --format=freeze > requirements.txt
```