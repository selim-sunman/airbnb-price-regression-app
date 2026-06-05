import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from src.data_loader import DataLoader


class  TestDataLoader:

    def test_config(self, temp_config):

        mock_logger = MagicMock()
        loader = DataLoader(temp_config, mock_logger)
        
        assert loader.config is not None
        assert loader.logger is mock_logger

    def test_config_error(self):
        mock_logger = MagicMock()

        with pytest.raises(Exception):
            DataLoader(config={}, logger=mock_logger)
        mock_logger.error.assert_called_once()


    def test_load_csv(self, temp_config, sample_df):
        mock_logger = MagicMock()

        with patch("src.data_loader.pd.read_csv", return_value=sample_df):
            loader = DataLoader(temp_config, mock_logger)
            result = loader.load_csv()

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 10
        assert "price" in result.columns


    def test_load_csv_file_not_found(self, temp_config):
        mock_logger = MagicMock()

        with patch("src.data_loader.pd.read_csv", side_effect=FileNotFoundError):
            loader = DataLoader(temp_config, mock_logger)
            with pytest.raises(FileNotFoundError):
                loader.load_csv()
        
        mock_logger.error.assert_called_once()

    
    def test_load_csv_error(self, temp_config):
        mock_logger = MagicMock()

        with patch("src.data_loader.pd.read_csv", side_effect=Exception("Parse error")):
            loader = DataLoader(temp_config, mock_logger)
            with pytest.raises(Exception):
                loader.load_csv()
        mock_logger.error.assert_called_once()

    
    def test_load_csv_real_file(self, temp_config, sample_df, tmp_path):
        csv_path = tmp_path / "raw.csv"
        sample_df.to_csv(csv_path, index=False)

        temp_config["paths"]["raw_path"] = str(csv_path)

        mock_logger = MagicMock()
        loader = DataLoader(temp_config, mock_logger)
        result = loader.load_csv()

        assert result.shape == sample_df.shape
        assert list(result.columns) == list(sample_df.columns)