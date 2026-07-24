from pydantic import BaseModel

class DataConfig(BaseModel):
    raw_path: str
    processed_path: str
    model_path: str
    visualizer_path: str

class PreprocessingConfig(BaseModel):
    drop_cols: list
    numeric_features: list
    categorical_features: list

class TrainSettings(BaseModel):
    target_col: str
    random_state: int
    test_size: float

class HyperParameters(BaseModel):
    params: dict
    
class AppConfig(BaseModel):
    paths: DataConfig
    preprocessing: PreprocessingConfig
    train_settings: TrainSettings
    hyperparameters = HyperParameters