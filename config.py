import os
from dotenv import load_dotenv

load_dotenv()


def get_env(name: str) -> str:
    value = os.getenv(name)
    if value is None:
        raise ValueError(f"Environment variable {name} is not set")
    return value


POSTGRES_URI = get_env("POSTGRES_URI")
INITIAL_DATA_PATH = get_env("INITIAL_DATA_PATH")
APPEND_DATA_PATH = get_env("APPEND_DATA_PATH")
TABLE_NAME = get_env("TABLE_NAME")
