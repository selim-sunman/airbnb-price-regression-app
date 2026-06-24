import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from src.preprocess import Preprocess


class TestPreprocess:

    def test_init_valid(self, sample_df):
        mock_logger = MagicMock()
        preprocess = Preprocess(mock_logger, sample_df)

        assert preprocess.logger is mock_logger
        assert preprocess.df.equals(sample_df)

    def test_init_invalid_logger(self, sample_df):
        with pytest.raises(TypeError):
            Preprocess(object(), sample_df)

    def test_init_invalid_df(self):
        mock_logger = MagicMock()

        with pytest.raises(TypeError):
            Preprocess(mock_logger, [1, 2, 3])

    def test_outlier_operations(self, sample_df):
        mock_logger = MagicMock()
        preprocess = Preprocess(mock_logger, sample_df)
        result = preprocess.outlier_operations()

        upper = sample_df[sample_df["price"] > 0]["price"].quantile(0.99)

        assert (result["price"] > 0).all()
        assert result["minimum_nights"].max() <= 365
        assert result["price"].max() <= upper

    def test_outlier_operations_error(self, sample_df):
        mock_logger = MagicMock()
        sample_df = sample_df.drop(["price", "minimum_nights"], axis=1)
        preprocess = Preprocess(mock_logger, sample_df)

        with pytest.raises(ValueError):
            preprocess.outlier_operations()
        
        mock_logger.exception.assert_called_once()

    def test_missing_value_operations(self, sample_df):
        mock_logger = MagicMock()
        preproceess = Preprocess(mock_logger, sample_df)
        result = preproceess.missing_value_operations()

        epoch = pd.Timestamp("1970-01-01")

        assert epoch in result["last_review"].values
        assert result["reviews_per_month"].notna().all()

    def test_missing_value_operations_error(self, sample_df):
        mock_logger = MagicMock()
        sample_df = sample_df.drop(["last_review", "reviews_per_month"], axis=1)
        preprocess = Preprocess(mock_logger, sample_df)

        with pytest.raises(ValueError):
            preprocess.missing_value_operations()

        mock_logger.exception.assert_called_once()

    def test_skew_operations(self, sample_df):
        mock_logger = MagicMock()
        preprocess = Preprocess(mock_logger, sample_df)

        result = preprocess.skew_operations()

        assert "log_reviews" in result.columns
        assert "log_rpm" in result.columns

        np.testing.assert_allclose(
            result["log_reviews"],
            np.log1p(sample_df["number_of_reviews"])
        )

        np.testing.assert_allclose(
            result["log_rpm"],
            np.log1p(sample_df["reviews_per_month"])
        )

    def test_skew_operations_type_error(self, sample_df):
        mock_logger = MagicMock()
        preprocess = Preprocess(mock_logger, sample_df)

        with pytest.raises(TypeError):
            preprocess.skew_operations(df=[1,2,3])

        mock_logger.exception.assert_called_once()

    def test_skew_operations_value_error(self, sample_df):
        mock_logger = MagicMock()
        sample_df = sample_df.drop(["number_of_reviews", "reviews_per_month"], axis=1)
        preprocess = Preprocess(mock_logger, sample_df)

        with pytest.raises(ValueError):
            preprocess.skew_operations()

        mock_logger.exception.assert_called_once()


    def test_run_pipeline(self, sample_df):
        mock_logger = MagicMock()
        result = Preprocess(mock_logger, sample_df).run_pipeline()

        assert "log_reviews" in result.columns
        assert "log_rpm" in result.columns
        assert result["reviews_per_month"].isna().sum() == 0