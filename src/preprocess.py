import numpy as np
import pandas as pd


class Preprocess:
    """
    Provides preprocessing utilities for tabular datasets.

    The class performs three main preprocessing tasks:

    - Handles outliers
    - Fills missing values
    - Applies transformations to reduce feature skewness

    These operations prepare the dataset for feature engineering and model training.

    Attributes:
        logger: Logger instance used for logging events and errors.
        df (pd.DataFrame): Dataset to preprocess.
    """

    def __init__(self, logger, df: pd.DataFrame, price_cap_percentile: float = 0.99, max_nights: int = 365):
        """
        Initializes the preprocessing pipeline.

        Args:
            logger: Logger instance implementing the required logging methods.
            df (pd.DataFrame): Input dataset.
            price_cap_percentile (float, optional): 
                Percentile used to cap extreme values in the ``price`` column.
                Defaults to 0.99.
            max_nights (int, optional):
                Maximum allowed value for ``minimum_nights``.
                Defaults to 365.

        Raises:
            TypeError:
                If ``logger`` does not implement the required logging methods or if ``df`` is not a pandas DataFrame.
        """

        required_logger_methods = ("info", "error", "warning", "exception")

        if logger is None or not all(callable(getattr(logger, m, None)) for m in required_logger_methods):
            raise TypeError(f"Invalid logger: missing required methods {required_logger_methods}")

        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        
        self.logger = logger
        self.df = df
        self.price_cap_percentile = price_cap_percentile
        self.max_nights = max_nights

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

    def outlier_operations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes or caps extreme values in the dataset.

        Processing steps:
            - Removes rows where ``price`` is less than or equal to zero.
            - Caps ``price`` values at the configured percentile.
            - Removes rows where ``minimum_nights`` exceeds the configured limit.

        Args:
            df (pd.DataFrame): Dataset to process.

        Returns:
            pd.DataFrame: Dataset after outlier handling.

        Raises:
            ValueError: If required columns are missing.
            Exception: Re-raises any unexpected exception after logging it.
        """

        try:
            df = df.copy()

            self.logger.info("Starting outlier processing.")

            self.validate_columns(df, ["price", "minimum_nights"])

            df = df.loc[df["price"] > 0].copy()
            upper = df["price"].quantile(self.price_cap_percentile)
            df["price"] = df["price"].clip(upper=upper)

            df = df.loc[df["minimum_nights"] <= self.max_nights].copy()

            self.logger.info("Outlier processing completed successfully.")

            return df

        except Exception:
            self.logger.exception("Unexpected error during outlier processing.")
            raise
    
    def missing_value_operations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handles missing values in selected columns.

        Processing steps:
            - Converts ``last_review`` to datetime.
            - Replaces missing ``last_review`` values with ``1970-01-01``.
            - Fills missing ``reviews_per_month`` values with ``0``.

        Args:
            df (pd.DataFrame): Dataset to process.

        Returns:
            pd.DataFrame: Dataset with missing values handled.

        Raises:
            ValueError: If required columns are missing.
            Exception: Re-raises any unexpected exception after logging it.
        """

        try:
                
            df = df.copy()

            self.logger.info("Starting missing value processing.")
            
            self.validate_columns(df, ["last_review", "reviews_per_month"])

            df["last_review"] = pd.to_datetime(df["last_review"], errors="coerce")
            df["last_review"] = df["last_review"].fillna(pd.Timestamp('1970-01-01'))

            df["reviews_per_month"] = df["reviews_per_month"].fillna(0)

            self.logger.info("Missing value processing completed successfully.")

            return df

        except Exception:
            self.logger.exception("An unexpected error occurred while filling in the missing values.")
            raise

    def skew_operations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reduces skewness in numerical features.

        Processing steps:
            - Applies a log1p transformation to ``number_of_reviews``.
            - Applies a log1p transformation to ``reviews_per_month``.
            - Stores the transformed values in ``log_reviews`` and ``log_rpm``.

        Args:
            df (pd.DataFrame): Dataset to process.

        Returns:
            pd.DataFrame: Dataset containing the transformed features.

        Raises:
            ValueError: If required columns are missing.
            Exception: Re-raises any unexpected exception after logging it.
        """
        
        try:

            df = df.copy()

            self.logger.info("Starting skewness transformation.")

            self.validate_columns(df, ["number_of_reviews", "reviews_per_month"])

            df["log_reviews"] = np.log1p(df["number_of_reviews"])
            df["log_rpm"] = np.log1p(df["reviews_per_month"])

            self.logger.info("Skewness transformation completed successfully.")

            return df

        except Exception:
            self.logger.exception("An unexpected error occurred while applying skewness transformation.")
            raise
    
    def run_pipeline(self) -> pd.DataFrame:
        """
        Executes the complete preprocessing pipeline.

        The pipeline performs the following steps:

            1. Outlier handling
            2. Missing value imputation
            3. Skewness reduction

        Returns:
            pd.DataFrame: Fully preprocessed dataset.

        Raises:
            Exception: Re-raises any exception encountered during preprocessing after logging the error.
        """

        try:
            self.logger.info("Starting preprocessing pipeline.")

            df = self.outlier_operations(self.df)
            df = self.missing_value_operations(df)
            df = self.skew_operations(df)

            self.logger.info("Preprocessing pipeline completed successfully.")

            return df

        except Exception:
            self.logger.exception("Preprocessing pipeline failed.")
            raise