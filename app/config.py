import functools
import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    twelve_labs_api_key: str
    assemblyai_api_key: str
    mongodb_uri: str


def _require(name: str) -> str:
    value = os.getenv(name, "")
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv()
    return Settings(
        twelve_labs_api_key=_require("TWELVE_LABS_API_KEY"),
        assemblyai_api_key=_require("ASSEMBLYAI_API_KEY"),
        mongodb_uri=_require("MONGODB_URI"),
    )
