from pydantic import BaseModel


class DataConfig(BaseModel):
    raw_path: str
    processed_path: str
    model_path: str
    visualizer_path: str


class AppConfig(BaseModel):
    paths: DataConfig