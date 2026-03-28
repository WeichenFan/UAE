## 1. T2I Low-Pass Retrieval

```bash
python analysis/retrival_exp.py \
  --data-path /path/to/imagenet/val \
  --text-template "the image of {}" \
  --topk 1 \
  --retrieval-filter-shape radial \
  --retrieval-radial-norm axis
```

Default output file:
- `analysis/output/retrival_exp/output.png`

For the Recall@5 variant, change `--topk 1` to `--topk 5`.

## 2. Energy Exp

```bash
python analysis/energy_exp.py \
  --data-path /path/to/imagenet/val \
  --output-file /path/to/output.png
```
