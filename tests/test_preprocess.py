import pytest
import numpy as np
import pandas as pd
from src.preprocess import Preprocess


class TestPreprocess:

    def test_init_valid(self, sample_df, mock_logger):
        preprocess = Preprocess(mock_logger, sample_df)

        assert preprocess.logger is mock_logger
        assert preprocess.df.equals(sample_df)

    def test_init_invalid_logger(self, sample_df):
        with pytest.raises(TypeError):
            Preprocess(object(), sample_df)

    def test_init_invalid_df(self, mock_logger):
        with pytest.raises(TypeError):
            Preprocess(mock_logger, [1, 2, 3])

    def test_validate_columns_type_error(self, sample_df, mock_logger):
        required = ["last_review", "reviews_per_month"]
        preprocess = Preprocess(mock_logger, sample_df)

        with pytest.raises(TypeError):
            preprocess.validate_columns([1, 2, 3], required)

    def test_validate_columns_missing_columns(self, sample_df, mock_logger):
        required = ["last_review", "reviews_per_month"]
        sample_df = sample_df.drop(required, axis=1)
        preprocess = Preprocess(mock_logger, sample_df)

        with pytest.raises(ValueError):
            preprocess.validate_columns(sample_df, required)

    def test_outlier_operations(self, sample_df, mock_logger):
        preprocess = Preprocess(mock_logger, sample_df)
        result = preprocess.outlier_operations(sample_df)

        upper = sample_df[sample_df["price"] > 0]["price"].quantile(0.99)

        assert (result["price"] > 0).all()
        assert result["minimum_nights"].max() <= 365
        assert result["price"].max() <= upper

    def test_outlier_operations_error(self, sample_df, mock_logger):
        sample_df = sample_df.drop(["price", "minimum_nights"], axis=1)
        preprocess = Preprocess(mock_logger, sample_df)

        assert "price" not in sample_df.columns

        with pytest.raises(ValueError):
            preprocess.outlier_operations(sample_df)
        
        mock_logger.exception.assert_called_once()

    def test_missing_value_operations(self, sample_df, mock_logger):
        preprocess = Preprocess(mock_logger, sample_df)
        result = preprocess.missing_value_operations(sample_df)

        epoch = pd.Timestamp("1970-01-01")

        assert epoch in result["last_review"].values
        assert result["reviews_per_month"].notna().all()

    def test_missing_value_operations_error(self, sample_df, mock_logger):
        sample_df = sample_df.drop(["last_review", "reviews_per_month"], axis=1)
        preprocess = Preprocess(mock_logger, sample_df)

        with pytest.raises(ValueError):
            preprocess.missing_value_operations(sample_df)

        mock_logger.exception.assert_called_once()

    def test_skew_operations(self, sample_df, mock_logger):
        preprocess = Preprocess(mock_logger, sample_df)

        result = preprocess.skew_operations(sample_df)

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

    def test_skew_operations_type_error(self, sample_df, mock_logger):
        preprocess = Preprocess(mock_logger, sample_df)

        with pytest.raises(TypeError):
            preprocess.skew_operations(df=[1,2,3])

        mock_logger.exception.assert_called_once()

    def test_skew_operations_value_error(self, sample_df, mock_logger):
        sample_df = sample_df.drop(["number_of_reviews", "reviews_per_month"], axis=1)
        preprocess = Preprocess(mock_logger, sample_df)

        with pytest.raises(ValueError):
            preprocess.skew_operations(sample_df)

        mock_logger.exception.assert_called_once()

    def test_run_pipeline(self, sample_df, mock_logger):
        result = Preprocess(mock_logger, sample_df).run_pipeline()

        assert (result["price"] > 0).all()
        assert result["minimum_nights"].max() <= 365
        assert result["reviews_per_month"].notna().all()
        assert "log_reviews" in result.columns
        assert "log_rpm" in result.columns
        assert result["reviews_per_month"].isna().sum() == 0