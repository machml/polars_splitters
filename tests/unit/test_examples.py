import polars as pl
from polars.testing import assert_frame_equal

from polars_splitters import sample, split_into_k_folds, split_into_train_eval


class TestQuickstartExamples:
    def setup_method(self):
        self.df = pl.DataFrame(
            {
                "feature_1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                "treatment": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                "outcome": [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            }
        )

    def test_split_into_train_eval(self):
        df_train, df_test = split_into_train_eval(
            self.df,
            eval_rel_size=0.3,
            stratify_by=["treatment", "outcome"],
            shuffle=True,
            rel_size_deviation_tolerance=0.1,
        )
        assert isinstance(df_train, pl.DataFrame)
        assert isinstance(df_test, pl.DataFrame)
        assert len(df_train) + len(df_test) == len(self.df)

    def test_split_into_k_folds(self):
        folds = split_into_k_folds(
            self.df,
            k=3,
            stratify_by=["treatment", "outcome"],
            shuffle=False,
        )
        assert isinstance(folds, list)
        assert all(isinstance(f, dict) for f in folds)
        assert all("train" in f and "eval" in f for f in folds)

        for fold in folds:
            assert isinstance(fold, dict)
            assert isinstance(fold["train"], pl.DataFrame)
            assert isinstance(fold["eval"], pl.DataFrame)
        #assert sum(fold["train"]) + len(fold["eval"]) for fold in folds) == len(self.df)

    def test_sample(self):
        df_sample = sample(
            self.df, fraction=0.5, stratify_by=["treatment", "outcome"]
        )
        assert isinstance(df_sample, pl.DataFrame)
        assert 0 < len(df_sample) < len(self.df)


class TestSplitIntoTrainEvalDocExample:
    def test_docstring_example(self):
        df = pl.DataFrame(
            {
                "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                "treatment": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                "outcome": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            }
        )
        df_train, df_eval = split_into_train_eval(
            df, eval_rel_size=0.4, stratify_by=["treatment", "outcome"], shuffle=False, seed=173
        )
        expected_train = pl.DataFrame(
            {
                "feature_1": [3.0, 4.0, 5.0, 7.0, 8.0, 10.0],
                "treatment": [0, 0, 0, 1, 1, 1],
                "outcome": [0, 0, 0, 0, 0, 1],
            }
        )
        expected_eval = pl.DataFrame(
            {
                "feature_1": [1.0, 2.0, 6.0, 9.0],
                "treatment": [0, 0, 1, 1],
                "outcome": [0, 0, 0, 1],
            }
        )
        assert_frame_equal(df_train, expected_train)
        assert_frame_equal(df_eval, expected_eval)
