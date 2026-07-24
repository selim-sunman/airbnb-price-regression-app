import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from schemas import AppConfig


class ModelTrainer:

    def __init__(self, config: dict, df: pd.DataFrame, logger):

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

    def split_data(self, target, random_state, test_size) -> tuple:
  
        X = self.df.drop(target, axis=1)
        y = self.df[target]

        return train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state
        )
    
    def build_pipeline(self, numeric_features, categorical_features) -> Pipeline:
        
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_features),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
            ]
        )

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", LinearRegression())
        ])

        return pipeline
    
    def hyperparameter_search(self, pipeline, param_dict, X_train, y_train) -> GridSearchCV:
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_dict,
            cv=5,
            scoring="r2",
            verbose=2,
            n_jobs=-1,
            error_score="raise"
        )

        search.fit(X_train, y_train)

        best_hyperparameters = search.best_params_
        best_cv_score = search.best_score_

        self.logger.info(f"Best hyperparameters: {best_hyperparameters} - Best CV score: {best_cv_score}")

        return search

    def run_training(self) -> tuple[Pipeline, pd.DataFrame, pd.Series]:

        target = self.config.train_settings.target_col
        random_state = self.config.train_settings.random_state
        test_size = self.config.train_settings.test_size
        numeric_features = self.config.preprocessing.numeric_features
        categorical_features = self.config.preprocessing.categorical_features
        param_dict = self.config.hyperparameters.params
        model_path = self.config.paths.model_path

        X_train, X_test, y_train, y_test = self.split_data(target, random_state, test_size)
        pipeline = self.build_pipeline(numeric_features, categorical_features)
        search = self.hyperparameter_search(pipeline, param_dict, X_train, y_train)

        best_model = search.best_estimator_

        try:
            self.logger.info("Saving the model...")
            joblib.dump(best_model, model_path)
            self.logger.info(f"Model saved to {model_path}")
        except Exception:
            self.logger.exception("Failed to save model.")
            raise

        return best_model, X_test, y_test