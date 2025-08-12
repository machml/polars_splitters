# polars-splitters

Polars-based splitter functionalities for polars LazyFrames and DataFrames similar to `sklearn.model_selection.train_test_split` and `sklearn.model_selection.StratifiedKFold`.

## features

- split_into_train_eval
- split_into_k_folds
- sample: stratified sampling

## installation

```bash
pip install polars-splitters
```

## usage

```python
import polars as pl
from polars_splitters import split_into_train_eval, split_into_k_folds

df = pl.DataFrame(
    {
        "feature_1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "treatment": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
        "outcome": [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    }
)

df_train, df_test = split_into_train_eval(
    df,
    eval_rel_size=0.4,
    stratify_by=["treatment", "outcome"],
    shuffle=False,
)

folds = split_into_k_folds(
    df,
    k=3,
    stratify_by=["treatment", "outcome"],
    shuffle=False,
    as_lazy=False
)

df_sample = sample(
    df,
    fraction=0.5,
    stratify_by=["treatment", "outcome"],
)
```

## current limitations

- only supports polars eager API (pl.DataFrame): no pl.LazyFrame

## future work

- test: use uv & github actions for testing on multiple python versions, bounding polars versions

- [test] add unit tests for sample()
- [test] add tests for handling ties
- [feat] implement handling of ties