import numpy as np
import pandas as pd


class Preprocess:
    """
    This class provides methods for handling missing values,
    processing outliers, and transforming skewed numerical features
    to improve data quality before feature engineering.

    Attributes:
        logger: Logger instance used for error reporting and process tracking.
        df (pd.DataFrame): Input dataset to be preprocessed.
    """

    def __init__(self, logger, df):
        """
        Initializes a Preprocess instance.

        Args:
            logger: Logger instance used for logging information, warnings, and errors.
                Expected to implement at least: info(), error(), warning() methods.
            df (pd.DataFrame): Input dataset to be preprocessed.

        Raises:
            TypeError: If logger does not implement required logging methods or if df is not a pandas DataFrame.
        """

        required_logger_methods = ("info", "error", "warning")

        if logger is None or not all(callable(getattr(logger, m, None)) for m in required_logger_methods):
            raise TypeError(f"Invalid logger: missing required methods {required_logger_methods}")

        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        
        self.logger = logger
        self.df = df

    def outlier_operations(self):
        """
        Processes outliers in the dataset.

        Operations:
            - Removes rows with non-positive values in the 'price' column.
            - Caps 'price' values at the 99th percentile.
            - Removes rows where 'minimum_nights' exceeds 365.

        Returns:
            pd.DataFrame: Dataset with outliers processed.

        Raises:
            ValueError: If the 'price' or 'minimum_nights' column is missing.
            Exception: Propagates any unexpected exception encountered during outlier processing after logging the error.
        """

        try:
            new_df = self.df.copy()

            if "price" in new_df.columns:
                new_df = new_df[new_df["price"] > 0]
                upper = new_df["price"].quantile(0.99)
                new_df["price"] = new_df["price"].clip(upper=upper)
            else:
                raise ValueError("'price' column is missing from the DataFrame.")
        
            if "minimum_nights" in new_df.columns:
                new_df = new_df[new_df["minimum_nights"] <= 365]
            else:
                raise ValueError("The 'minimum_nights' property was not found in 'new_df'.")
        
        except Exception:
            self.logger.exception("Unexpected error during outlier processing.")
            raise

        return new_df
    
    def missing_value_operations(self, df=None):
        """
        Handles missing values in the dataset.

        Operations:
            - Uses the provided DataFrame if available; otherwise uses self.df.
            - Converts the 'last_review' column to datetime format and fills
                missing values with '1970-01-01'.
            - Fills missing values in 'reviews_per_month' with 0, assuming
                that missing values indicate listings with no monthly reviews.

        Args:
            df (pd.DataFrame, optional): Input DataFrame. If None, self.df is used.

        Returns:
            pd.DataFrame: Dataset with missing values handled.

        Raises:
            TypeError: If df is provided and is not a pandas DataFrame.
            ValueError: If the required columns ('last_review' or 'reviews_per_month') are missing from the dataset.
            Exception: Propagates any unexpected exception encountered during missing value processing after logging the error.
        """

        try:

            if df is not None and not isinstance(df, pd.DataFrame):
                raise TypeError("Input must be a pandas DataFrame")
                
            source = (df if df is not None else self.df).copy()
                
            if "last_review" in source.columns:
                source["last_review"] = pd.to_datetime(source["last_review"])
                source["last_review"] = source["last_review"].fillna(pd.Timestamp('1970-01-01'))
            else:
                raise ValueError("The 'last_review' property was not found in 'source'.")
            
            if "reviews_per_month" in source.columns:
                source["reviews_per_month"] = source["reviews_per_month"].fillna(0)
            else:
                raise ValueError("The 'reviews_per_month' property was not found in 'source'.")

        except Exception:
            self.logger.exception("An unexpected error occurred while filling in the missing values.")
            raise

        return source

    def skew_operations(self, df=None):
        """
        Applies transformations to reduce skewness in numerical features.

        Operations:
            - Uses the provided DataFrame if available; otherwise uses self.df.
            - Applies a log1p transformation to 'number_of_reviews' and
          'reviews_per_month'.
            - Creates the derived features 'log_reviews' and 'log_rpm'.

        Args:
            df (pd.DataFrame, optional): Input DataFrame. If None, self.df is used.

        Returns:
            pd.DataFrame: Dataset with skewness-reduced features.

        Raises:
            TypeError: If df is provided and is not a pandas DataFrame.
            ValueError: If the required columns ('number_of_reviews' or 'reviews_per_month') are missing from the dataset.
            Exception: Propagates any unexpected exception encountered during skewness transformation after logging the error.
        """
        
        try:

            if df is not None and not isinstance(df, pd.DataFrame):
                raise TypeError("Input must be a pandas DataFrame")

            source = (df if df is not None else self.df).copy()

            required_cols = ["number_of_reviews", "reviews_per_month"]
            if all(col in source.columns for col in required_cols):
                source["log_reviews"] = np.log1p(source["number_of_reviews"])
                source["log_rpm"] = np.log1p(source["reviews_per_month"])
            else:
                raise ValueError("The properties in 'required_cols' could not be found in 'source'.")

        except Exception:
            self.logger.exception("An unexpected error occurred while applying the skewed value conversion.")
            raise

        return source
    
    def run_pipeline(self):
        """
        Executes the complete preprocessing pipeline.

        The following steps are performed sequentially:
            1. Outlier processing
            2. Missing value handling
            3. Skewness transformation

        Returns:
            pd.DataFrame: Fully preprocessed dataset.

        Raises:
            ValueError: If required columns are missing during any preprocessing step.
            TypeError: If invalid input data is encountered.
            Exception: Propagates any unexpected exception raised by the preprocessing steps.
        """

        outlier_df = self.outlier_operations()
        missing_df = self.missing_value_operations(df=outlier_df)
        final_df = self.skew_operations(df=missing_df)

        return final_df