import sys
from loguru import logger
from pathlib import Path



def setup_logger():

    log_dir = Path("logs")

    log_dir.mkdir(parents=True, exist_ok=True)

    log_file_path = log_dir / "app_{time:YYYY-MM-DD}.log"

    logger.remove()

    logger.add(
        sys.stdout,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )

    logger.add(
        log_file_path,
        rotation="10 MB",
        retention="7 days",
        compression="zip",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        encoding="utf-8"
    )

    return logger


if __name__ == "__main__":
    log = setup_logger()
    log.info("The application has been launched.")
    log.debug("This is a debug message (it only appears in the file).")
    log.warning("Attention! An alert has been generated.")
    log.error("Error: Database connection failed.")