from collections import Counter

import pytest
from polars import DataFrame, LazyFrame, concat
from pytest_check import check

from polars_splitters.core.splitters import split_into_k_folds, split_into_train_eval

SEED = 173


class TestSplitIntoTrainEval:
    @pytest.mark.parametrize("from_lazy", [False, True])
    @pytest.mark.parametrize("eval_rel_size, expected_eval_size", [(0.3, 3), (0.4, 4)])
    @pytest.mark.parametrize("stratify_by", [None, "treatment", ["treatment", "outcome"]])
    @pytest.mark.parametrize("shuffle", [False, True])
    @pytest.mark.parametrize("as_lazy", [False, True])
    @pytest.mark.parametrize("as_dict", [False, True])
    @pytest.mark.parametrize("validate", [False, True])
    def test_output_types_e_sizes_inputting_basic_df(
        self,
        df_basic,
        from_lazy,
        eval_rel_size,
        expected_eval_size,
        stratify_by,
        shuffle,
        as_lazy,
        as_dict,
        validate,
    ):
        result = split_into_train_eval(
            df=df_basic(from_lazy),
            eval_rel_size=eval_rel_size,
            stratify_by=stratify_by,
            shuffle=shuffle,
            seed=SEED,
            as_lazy=as_lazy,
            as_dict=as_dict,
            validate=validate,
            rel_size_deviation_tolerance=0.1,
        )
        if as_dict:
            check.is_instance(result, dict)
            if as_lazy:
                check.is_instance(result["train"], LazyFrame)
                check.is_instance(result["eval"], LazyFrame)

                df_train, df_eval = result["train"].collect(), result["eval"].collect()

            elif not as_lazy:
                check.is_instance(result["train"], DataFrame)
                check.is_instance(result["eval"], DataFrame)

                df_train, df_eval = result["train"], result["eval"]

        elif not as_dict:
            check.is_instance(result, tuple)
            if as_lazy:
                check.is_instance(result[0], LazyFrame)
                check.is_instance(result[1], LazyFrame)

                df_train, df_eval = result[0].collect(), result[1].collect()

            elif not as_lazy:
                check.is_instance(result[0], DataFrame)
                check.is_instance(result[1], DataFrame)

                df_train, df_eval = result[0], result[1]

        check.is_true(df_train.shape == (10 - expected_eval_size, 3))
        check.is_true(df_eval.shape == (expected_eval_size, 3))

        # non-overlappingness of df_train, df_eval
        if not from_lazy and as_dict and not as_lazy and not shuffle:
            check.is_true(result["train"].shape[0] + result["eval"].shape[0] == df_basic(from_lazy).shape[0])

            df_concat = concat([result["train"], result["eval"]])
            n_duplicates = df_concat.is_duplicated().sum()
            check.is_true(n_duplicates == 0)

    @pytest.mark.parametrize("from_lazy", [False, True])
    @pytest.mark.parametrize("n_input", [400])
    @pytest.mark.parametrize("eval_rel_size", [0.3, 0.4])
    @pytest.mark.parametrize("stratify_by", [None, "treatment", ["treatment", "outcome"]])
    @pytest.mark.parametrize("shuffle", [False, True])
    def test_output_sizes_inputting_df_ubools(
        self,
        df_ubools,
        from_lazy,
        n_input,
        eval_rel_size,
        stratify_by,
        shuffle,
    ):
        result = split_into_train_eval(
            df=df_ubools(from_lazy, n=n_input),
            eval_rel_size=eval_rel_size,
            stratify_by=stratify_by,
            shuffle=shuffle,
            seed=SEED,
            as_lazy=False,
            as_dict=True,
            validate=True,
            rel_size_deviation_tolerance=0.1,
        )

        expected_eval_size = int(n_input * eval_rel_size)
        df_train, df_eval = result["train"], result["eval"]
        check.is_true(df_train.shape == (n_input - expected_eval_size, 5))
        check.is_true(df_eval.shape == (expected_eval_size, 5))

    @pytest.mark.parametrize("stratify_by", [None, "treatment", ["treatment", "outcome"]])
    def test_stratification_inputting_basic_df(self, df_basic, stratify_by):
        result = split_into_train_eval(
            df=df_basic(from_lazy=False),
            eval_rel_size=0.3,
            stratify_by=stratify_by,
            shuffle=False,
            seed=SEED,
            as_lazy=False,
            as_dict=True,
            validate=True,
            rel_size_deviation_tolerance=0.1,
        )

        if stratify_by is None:
            check.equal(Counter(result["train"]["treatment"]), Counter({1: 6, 0: 1}))
            check.equal(Counter(result["eval"]["treatment"]), Counter({0: 3}))
        elif stratify_by == "treatment":
            check.equal(Counter(result["train"]["treatment"]), Counter({1: 4, 0: 3}))
            check.equal(Counter(result["eval"]["treatment"]), Counter({1: 2, 0: 1}))
        elif stratify_by == ["treatment", "outcome"]:
            check.equal(
                Counter(zip(result["train"]["treatment"], result["train"]["outcome"])),
                Counter({(0, 0): 3, (1, 0): 2, (1, 1): 2}),
            )
            check.equal(
                Counter(zip(result["eval"]["treatment"], result["eval"]["outcome"])),
                Counter({(0, 0): 1, (1, 0): 1, (1, 1): 1}),
            )

    @pytest.mark.parametrize("stratify_by", [None, "treatment", ["treatment", "outcome"]])
    def test_stratification_inputting_df_ubools(self, df_ubools, stratify_by):
        n_input = 400
        eval_rel_size = 0.3
        result = split_into_train_eval(
            df=df_ubools(from_lazy=False, n=n_input, seed=SEED),
            eval_rel_size=eval_rel_size,
            stratify_by=stratify_by,
            shuffle=False,
            seed=SEED,
            as_lazy=False,
            as_dict=True,
            validate=True,
            rel_size_deviation_tolerance=0.1,
        )

        if stratify_by is None:
            # results should not (necessarily) be stratified
            check.equal(Counter(result["train"]["treatment"]), Counter({0: 132, 1: 148}))
            check.equal(Counter(result["eval"]["treatment"]), Counter({0: 68, 1: 52}))
        elif stratify_by == "treatment":
            # results should be well stratified according to treatment...
            check.equal(Counter(result["train"]["treatment"]), Counter({0: 140, 1: 140}))
            check.equal(Counter(result["eval"]["treatment"]), Counter({0: 60, 1: 60}))

            # ...but not necessarily according to treatment & outcome
            check.equal(
                Counter(zip(result["train"]["treatment"], result["train"]["outcome"])),
                Counter({(0, 0): 69, (0, 1): 71, (1, 0): 72, (1, 1): 68}),
            )
            check.equal(
                Counter(zip(result["eval"]["treatment"], result["eval"]["outcome"])),
                Counter({(0, 0): 31, (0, 1): 29, (1, 0): 28, (1, 1): 32}),
            )

        elif stratify_by == ["treatment", "outcome"]:
            # results should be well stratified according to treatment & outcome...
            check.equal(
                Counter(zip(result["train"]["treatment"], result["train"]["outcome"])),
                Counter({(0, 0): 70, (0, 1): 70, (1, 0): 70, (1, 1): 70}),
            )
            check.equal(
                Counter(zip(result["eval"]["treatment"], result["eval"]["outcome"])),
                Counter({(0, 0): 30, (0, 1): 30, (1, 0): 30, (1, 1): 30}),
            )

            # ...as well as according to treatment and outcome individually
            check.equal(Counter(result["train"]["treatment"]), Counter({0: 140, 1: 140}))
            check.equal(Counter(result["eval"]["treatment"]), Counter({0: 60, 1: 60}))


class TestSplitIntoKFolds:
    pass
    # TODO: test folds & contents types
    # TODO: test folds sizes
    # TODO: test non-overlappingness of df_train, df_eval in each fold
    # TODO: test non-overlappingness of df_evals for different folds
    # TODO: test validation: stratification by float column
    # TODO: test validation: stratification, rel_size_deviation_tolerance
