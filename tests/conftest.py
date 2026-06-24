import pytest
import yaml
import copy
import pandas as pd
from pathlib import Path
from loguru import logger


PROJECT_ROOT = Path(__file__).parent.parent

@pytest.fixture(scope="session")
def sample_config():
    """
    Loads the real application configuration from config/config.yaml.

    Scoped to the session so the file is read only once across all tests.

    Returns:
        dict: Parsed YAML configuration dictionary.
    """

    config_path = PROJECT_ROOT / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


@pytest.fixture
def temp_config(sample_config, tmp_path):
    """
    Creates a temporary copy of the application config with all paths
    redirected to pytest's tmp_path directory.

    Ensures tests do not read from or write to real project files.

    Args:
        sample_config: The session-scoped base configuration fixture.
        tmp_path: Pytest built-in fixture providing a temporary directory.

    Returns:
        dict: A deep-copied config dictionary with overridden file paths.
    """

    cfg = copy.deepcopy(sample_config)
    cfg["paths"]["raw_path"] = str(tmp_path / "raw.csv")
    cfg["paths"]["processed_path"] = str(tmp_path / "processed.csv")
    cfg["paths"]["model_path"] = str(tmp_path / "model.joblib")
    cfg["paths"]["visualizer_path"] = str(tmp_path / "figures")
    return cfg



@pytest.fixture(autouse=True)
def disable_logging():
    """
    Suppresses all loguru log output during tests.

    Applied automatically to every test (autouse=True) to keep
    test output clean. Logging is re-enabled after each test via yield.
    """
    
    logger.disable("")
    yield
    logger.enable("")


@pytest.fixture
def sample_df():
    """
    Provides a realistic 10-row Airbnb listings DataFrame for use in tests.

    Covers a variety of neighbourhoods, room types, and price ranges
    to allow meaningful assertions without relying on external files.

    Returns:
        pd.DataFrame: Sample listings data with all expected schema columns.
    """

    return pd.DataFrame({
        "id" : [
            317905, 34205267, 12342297, 33527778, 13136376,
            4365276, 22968206, 32472023, 8609130, 24806106,
        ],
        "name": [
            "Two-bedroom in the heart of Chelsea",
            "Home Sweet Home :)",
            "Private Bedroom in the heart of Bushwick Him-1R-4",
            "3min to Grand Ave subway, newly built in 2015",
            "@Ferry,Large Private Rm,Renovated/Stylish,Views...",
            "Sonder | Stock Exchange | Timeless 3BR + Sofa Bed",
            "-CENTRAL PARK - 3 stops by subway(32'' TV room)",
            "Sonder | Stock Exchange | Smart 2BR + Sofa Bed",
            "Large House in Brooklyn near Express Subway",
            "My home."
        ],
        "host_id": [
            1631733, 913940, 19953913, 219517861, 16110448,
            3081990, 22748648, 209376540, 2520559, 132669029
        ],
        "host_name":  [
            "Lio & Kim And Yotam", "Katia", "Adam", "Lawrence",
            "Vida", "Qusuquzah", "Steve", "Umut Can", "Mark", "Ehssan"
        ],
        "neighbourhood_group": [
            "Brooklyn", "Queens", "Manhattan", "Manhattan", "Manhattan",
            "Brooklyn", "Brooklyn", "Queens", "Manhattan", "Bronx"
        ],
        "neighbourhood": [
            "Kensington", "Ridgewood", "Hell's Kitchen", "Financial District",
            "East Harlem", "Williamsburg", "Williamsburg", "College Point",
            "SoHo", "Throgs Neck"
        ],
        "latitude": [
            40.64354, 40.70666, 40.76116, 40.70763, 40.79658,
            40.70698, 40.71246, 40.76813, 40.72214, 40.81437,
        ],
        "longitude": [
            -73.97777, -73.90779, -73.99016, -74.01050, -73.93287,
            -73.95406, -73.96133, -73.84542, -73.99793, -73.82774
        ],
        "room_type": [
            "Entire home/apt", "Private room", "Private room",
            "Entire home/apt", "Entire home/apt", "Entire home/apt",
            "Private room", "Entire home/apt", "Entire home/apt",
            "Entire home/apt"
        ],
        "price": [89, 30, 120, 470, 199, 170, 90, 60, 150, 74],
        "minimum_nights": [3, 21, 2, 2, 2, 1, 5, 30, 365, 2],
        "number_of_reviews": [62, 0, 17, 5, 30, 141, 1, 0, 89, 37],
        "last_review": [
            "2019-06-11", "2017-06-30", "2019-05-28", "2015-10-18",
            "2018-08-03", None, "2019-04-23", "2019-05-23",
            "2016-04-04", "2019-03-25"
        ],
        "reviews_per_month": [
            2.08, 0.17, 0.76, None, 0.33,
            0.10, 1.42, 4.56, 0.03, 0.19,
        ],
        "calculated_host_listings_count": [1, 1, 1, 327, 1, 1, 1, 1, 1, 4],
        "availability_365": [189, 73, 0, 272, 30, 28, 0, 121, 55, 70]
    })



