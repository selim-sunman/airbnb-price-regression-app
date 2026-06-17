import numpy as np
import pandas as pd



class Preprocess:

    def __init__(self, logger, df):

        if not callable(getattr(logger, 'error', None)):
            raise TypeError("logger must have a callable 'error' method")

        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        
        self.logger = logger
        self.df = df

    def outlier_operations(self):

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
        
        except Exception as e:
            self.logger.error(f"Critical error that occurs during outlier operations: {e}")
            raise


        return new_df
    
    def missing_value_operations(self, df=None):

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

        except Exception as e:
            self.logger.error(f"Critical error that occurs during missing value operations: {e}")
            raise

        return source


    def skew_operations(self, df=None):
        
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

        except Exception as e:
            self.logger.error(f"Critical error that occurs during skew operations: {e}")
            raise

        return source
    

    def run_pipeline(self):

        outlier_df = self.outlier_operations()
        missing_df = self.missing_value_operations(df=outlier_df)
        final_df = self.skew_operations(df=missing_df)

        return final_df
    