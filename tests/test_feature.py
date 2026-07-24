import pytest
import pandas as pd
from unittest.mock import MagicMock
from src.feature import Feature

class TestFeature:

    def test_init_valid(self, sample_df, sample_config, mock_logger):
        feature = Feature(sample_config, sample_df, mock_logger)

        assert sample_config is not None
        assert feature.logger is mock_logger
        assert feature.df.equals(sample_df)

    def test_init_invalid_logger(self, sample_df, sample_config):
        with pytest.raises(TypeError):
            Feature(sample_config, sample_df, object())

    def test_init_invalid_df(self, sample_config, mock_logger):
        with pytest.raises(TypeError):
            Feature(sample_config, [1, 2, 3], mock_logger)
        
    def test_init_invalid_config(self, sample_df, mock_logger):
        with pytest.raises(TypeError):
            Feature(object(), sample_df, mock_logger)

    def test_validate_columns_type_error(self, sample_df, sample_config, mock_logger):
        required = ["last_review", "reviews_per_month"]
        feature = Feature(sample_config, sample_df, mock_logger)

        with pytest.raises(TypeError):
            feature.validate_columns([1, 2, 3], required)

    def test_validate_columns_value_error(self, sample_df, sample_config, mock_logger):
        required = ["last_review", "reviews_per_month"]
        sample_df = sample_df.drop(required, axis=1)
        feature = Feature(sample_config, sample_df, mock_logger)

        with pytest.raises(ValueError):
            feature.validate_columns(sample_df, required)

    def test_extract_date_feature(self, sample_config,sample_df, mock_logger):
        feature = Feature(sample_config, sample_df, mock_logger)
        date_features = feature.extract_date_features()

        assert "last_review_year" in date_features.columns 
        assert "last_review_month" in date_features.columns 
        assert "last_review_day" in date_features.columns 

    def test_extract_date_feature_error(self, sample_config, sample_df, mock_logger):
        sample_df = sample_df.drop("last_review", axis=1)
        feature = Feature(sample_config, sample_df, mock_logger)
        
        with pytest.raises(ValueError):
            feature.extract_date_features()

    def test_create_feature(self, sample_config, sample_df, mock_logger):
        feature = Feature(sample_config, sample_df, mock_logger)
        create_feature = feature.create_features(sample_df)

        assert "neigh_ad_rate" in create_feature.columns
        assert "professional_host" in create_feature.columns
        assert "estimated_booked_days" in create_feature.columns

    def test_create_feature_error(self, sample_config, sample_df, mock_logger):
        required_cols = ["id", "host_id", "neighbourhood", "calculated_host_listings_count", "availability_365"]
        sample_df = sample_df.drop(required_cols, axis=1)
        feature = Feature(sample_config, sample_df, mock_logger)
        
        with pytest.raises(ValueError):
            feature.create_features(sample_df)

    def test_encode_neighbourhood_frequency(self, sample_config, sample_df, mock_logger):
        feature = Feature(sample_config, sample_df, mock_logger)
        transform_feature = feature.encode_neighbourhood_frequency(sample_df)

        assert "neighbourhood_freq" in transform_feature.columns

    def test_encode_neighbourhood_frequency_error(self, sample_config, sample_df, mock_logger):
        sample_df = sample_df.drop("neighbourhood", axis=1)
        feature = Feature(sample_config, sample_df, mock_logger)

        with pytest.raises(ValueError):
            feature.encode_neighbourhood_frequency(sample_df)

    def test_remove_columns(self, sample_df, sample_config, mock_logger):
        drop_cols = sample_config["preprocessing"]["drop_cols"]

        feature = Feature(sample_config, sample_df, mock_logger)
        remove = feature.remove_columns(sample_df)

        for col in drop_cols:
            assert col not in remove.columns

    def test_run_pipeline(self, sample_config, sample_df, mock_logger):
        result = Feature(sample_config, sample_df, mock_logger).run_pipeline()

        assert "last_review_year" in result.columns
        assert "last_review_month" in result.columns
        assert "last_review_day" in result.columns
        
        assert "neigh_ad_rate" in result.columns
        assert "professional_host" in result.columns
        assert "estimated_booked_days" in result.columns
        assert "neighbourhood_freq" in result.columns

        assert "name" not in result.columns
        assert "number_of_reviews" not in result.columns
        assert "availability_365" not in result.columns
        assert "neighbourhood" not in result.columns
        