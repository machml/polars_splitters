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
from polars_splitters import split_into_train_eval, split_into_k_folds, sample

df = pl.DataFrame(
    {
        "feature_1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "treatment": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
        "outcome": [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    }
)

# Split into train and eval
df_train, df_test = split_into_train_eval(
    df,
    eval_rel_size=0.4,
    stratify_by=["treatment", "outcome"],
    shuffle=False,
)
print(df_train, df_test)

# Split into k folds
folds = split_into_k_folds(
    df,
    k=3,
    stratify_by=["treatment", "outcome"],
    shuffle=False,
)

# e.g. get the pair df_train, df_eval for the first fold
df_train, df_val = folds[0]["train"], folds[0]["eval"]
print(df_train, df_val)

# Stratified sample
df_sample = sample(
    df,
    fraction=0.5,
    stratify_by=["treatment", "outcome"],
)

print(df_sample)
```

## current limitations

- only supports polars eager API (pl.DataFrame): no pl.LazyFrame

## future work

- [test] add unit tests for sample()

- [test] add tests for handling ties

- [feat] implement handling of ties
