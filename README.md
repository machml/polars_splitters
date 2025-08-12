# polars-splitters

<p align="center">
 <em>stratified splitting of polars dataframes</em>
</p>
<p align="center">
<a href="https://github.com/machml/polars-splitters/actions" target="_blank">
<img src="https://github.com/machml/polars_splitters/actions/workflows/test.yml/badge.svg" alt="Tests">
</a>
<img src="https://img.shields.io/pypi/v/polars-splitters?color=%2334D058&label=pypi%20package" alt="Package version">
<a href="https://pypi.org/project/polars-splitters" target="_blank">
<img src="https://img.shields.io/pypi/pyversions/polars-splitters.svg?color=%2334D058" alt="Supported Python versions">
</a>
</p>

Polars-based splitter functionalities for polars DataFrames similar to sklearn.model_selection.train_test_split and sklearn.model_selection.StratifiedKFold.

## features

- split_into_train_eval
- split_into_k_folds
- sample: stratified sampling

## installation

```shell
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

- improve resilience against ties in quantized float columns in stratify_by
