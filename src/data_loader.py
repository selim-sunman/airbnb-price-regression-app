import pandas as pd
import loguru
from src.schemas import AppConfig


class DataLoader:
    """
    The class responsible for uploading CSV data files.

    Attributes:
        logger: Application-wide logger object.
        config (AppConfig): Validated application configuration object.
    """
    
    def __init__(self, config: dict, logger):
        """
        Initializes the DataLoader object and validates its configuration.

        Args:
            config: Dictionary containing application configuration
            logger: Logger instance used for logging messages.
                Expected to implement: info(), error(), warning().

        Raises:
            TypeError: If logger does not implement required logging methods
            ValidationError: Thrown if the configuration does not match the schema.
        """

        required_logger_methods = ("info", "error", "warning")

        if logger is None or not all(callable(getattr(logger, m, None)) for m in required_logger_methods):
            raise TypeError(f"Invalid logger: missing required methods {required_logger_methods}")
        
        self.logger = logger

        try:
            self.config = AppConfig(**config)
        except Exception as e:
            self.logger.error(f"Configuration validation error: {e}")
            raise

    def load_csv(self) -> pd.DataFrame:
        """
        Reads the CSV file from the path specified in the configuration.

        Returns:
            pd.DataFrame: DataFrame object created from the loaded data.

        Raises:
            FileNotFoundError: Thrown if the file is not found at the specified path.
            Exception: Thrown if an unexpected error occurs while reading the file.
        """

        self.logger.info(f"Loading raw data from: {self.config.paths.raw_path}")

        try:
            df = pd.read_csv(self.config.paths.raw_path)
        except FileNotFoundError as e:
            self.logger.error(f"Raw data file not found at: {e}")
            raise
        except Exception:
            self.logger.error(f"Failed to read data from {self.config.paths.raw_path}")
            raise

        if df.empty:
            self.logger.warning("Loaded CSV is empty.")

        self.logger.info("Data uploaded successfully.")

        return df