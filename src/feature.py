import pandas as pd
from src.schemas import AppConfig
class Feature:
    def __init__(self, config: dict, df: pd.DataFrame, logger):
        """
        Initializes a Feature instance.

        Args:
            df (pd.DataFrame): Input dataset for feature engineering.
            logger: Logger instance used for logging information, warnings, and errors.
            config: Dictionary containing application configuration

        Raises:
            TypeError: If logger does not implement the required logging methods or if df is not a pandas DataFrame.
        """

        required_logger_methods = ("info", "error", "warning", "exception")

        if logger is None or not all(callable(getattr(logger, m, None)) for m in required_logger_methods):
            raise TypeError(f"Invalid logger: missing required methods {required_logger_methods}")

        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")

        self.df = df
        self.logger = logger

        try:
            self.config = AppConfig(**config)
        except Exception as e:
            self.logger.error(f"Configuration validation error: {e}")
            raise

    def validate_columns(self, df: pd.DataFrame, required: list[str]) -> None:
        """
        Validates that all required columns exist in the dataset.

        Args:
            df (pd.DataFrame): Input dataset.
            required (list[str]): List of required column names.

        Raises:
            ValueError: If one or more required columns are missing.
        """

        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")

        missing = [c for c in required if c not in df.columns]

        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")

    def extract_date_features(self) -> pd.DataFrame:
        """
        Extracts date-based features from the 'last_review' column.

        Operations:
            - Extracts the review year.
            - Extracts the review month.
            - Extracts the review day.

        Returns:
            pd.DataFrame: Dataset with extracted date features.

        Raises:
            ValueError: If the 'last_review' column is missing.
            Exception: Propagates any unexpected exception encountered during date feature extraction after logging the error.
        """
        
        try:

            df = self.df.copy()

            self.logger.info("Starting date feature extraction.")

            self.validate_columns(df, ["last_review"])

            df["last_review"] = pd.to_datetime(df["last_review"], errors="coerce")

            df["last_review_year"] = df["last_review"].dt.year.astype("Int64")
            df["last_review_month"] = df["last_review"].dt.month.astype("Int64")
            df["last_review_day"] = df["last_review"].dt.day.astype("Int64")

            self.logger.info("Date feature extraction completed successfully.")

            return df

        except Exception:
            self.logger.exception("An unexpected error occurred while extracting date features.")
            raise
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates new engineered features from the dataset.

        Operations:
            - Computes the neighborhood advertisement rate.
            - Identifies professional hosts.
            - Estimates booked days from availability.

        Returns:
            pd.DataFrame: Dataset with engineered features.

        Raises:
            ValueError: If one or more required columns are missing.
            Exception: Propagates any unexpected exception encountered during feature creation after logging the error.
        """

        try:
            df = df.copy()

            self.logger.info("Starting feature engineering.")

            required_cols = ["id", "host_id", "neighbourhood", "calculated_host_listings_count", "availability_365"]
            
            self.validate_columns(df, required_cols)
            
            host_neigh_count = df.groupby(["host_id", "neighbourhood"])["id"].transform("count")
            neigh_count = df.groupby("neighbourhood")["id"].transform("count")

            df["neigh_ad_rate"] = host_neigh_count / neigh_count
            df["professional_host"] = (df["calculated_host_listings_count"] > 1).astype(int)
            df["estimated_booked_days"] = 365 - df["availability_365"]

            self.logger.info("Feature engineering completed successfully.")

            return df
        
        except Exception:
            self.logger.exception("An unexpected error occurred while creating features.")
            raise

    def encode_neighbourhood_frequency(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies frequency encoding to the neighbourhood feature.

        Operations:
            - The frequency of each neighbourhood is calculated based on its
            occurrence in the input DataFrame and stored in a new column named ``neighbourhood_freq``.

        Returns:
            pd.DataFrame: A dataset with a `neighbourhood_freq` column added to it.

        Raises:
            ValueError: If one or more of the required columns are missing.
            Exception: Propagates unexpected exceptions encountered during frequency encoding after the error is logged.
        """
        try:
            df = df.copy()

            self.logger.info("Applying frequency encoding to the 'neighbourhood' feature.")

            self.validate_columns(df, ["neighbourhood"])

            freq_map = df["neighbourhood"].value_counts(normalize=True)

            df["neighbourhood_freq"] = df["neighbourhood"].map(freq_map)

            self.logger.info("Successfully applied frequency encoding to the 'neighbourhood' feature.")

            return df
        
        except Exception:
            self.logger.exception("An unexpected error occurred during feature frequency encoding.")
            raise
            
    def remove_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes unnecessary columns from the dataset.

        Returns:
            pd.DataFrame: Dataset with unnecessary columns removed.

        Raises:
            Exception: Propagates any unexpected exception encountered during column removal after logging the error.
        """

        try:
            df = df.copy()

            self.logger.info("Starting column removal.")

            drop_cols = self.config.preprocessing.drop_cols
            df = df.drop(drop_cols, axis=1, errors="ignore")

            self.logger.info("Column removal completed successfully.")

            return df

        except Exception:
            self.logger.exception("An unexpected error occurred while removing unnecessary columns.")
            raise

    def run_pipeline(self) -> pd.DataFrame:
        """
        Executes the complete feature engineering pipeline.

        The following steps are performed sequentially:
            1. Date feature extraction
            2. Feature engineering
            3. Removal of unnecessary columns

        Returns:
            pd.DataFrame: Dataset with engineered features.

        Raises:
            ValueError: If required columns are missing during any step.
            TypeError: If invalid input data is encountered.
            Exception: Propagates any unexpected exception raised by the feature engineering pipeline.
        """

        try:
            self.logger.info("Starting feature engineering pipeline.")

            df = self.extract_date_features()
            df = self.create_features(df)
            df = self.encode_neighbourhood_frequency(df)
            df = self.remove_columns(df)

            self.logger.info(
                "Feature engineering pipeline completed successfully."
            )

            return df

        except Exception:
            self.logger.exception("Feature engineering pipeline failed.")
            raise