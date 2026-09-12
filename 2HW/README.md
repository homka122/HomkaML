# Homework 2

Run from the repository root:

```bash
uv sync --frozen
uv run jupyter lab 2HW/hw2_svhn_classification.ipynb
```

Run regression tests:

```bash
uv run python -m unittest discover -s 2HW -p 'test_*.py' -v
```

Execute and save all notebook outputs:

```bash
OPENBLAS_NUM_THREADS=1 uv run jupyter nbconvert --execute --to notebook \
  --ExecutePreprocessor.timeout=1800 --inplace 2HW/hw2_svhn_classification.ipynb
```

The notebook expects SVHN cropped files in `2HW/`:

```text
2HW/train_32x32.mat
2HW/test_32x32.mat
```

If needed, download them from:

```bash
curl -L -o 2HW/train_32x32.mat http://ufldl.stanford.edu/housenumbers/train_32x32.mat
curl -L -o 2HW/test_32x32.mat http://ufldl.stanford.edu/housenumbers/test_32x32.mat
```
