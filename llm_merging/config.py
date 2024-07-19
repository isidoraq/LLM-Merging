from dotenv import find_dotenv
from pydantic_settings import BaseSettings

DOTENV_FILE_PATH = find_dotenv(".env") or None


class Settings(BaseSettings):

    data_dir: str = "/home/ubuntu/LLM-Merging/data"

    class Config:
        env_file = DOTENV_FILE_PATH
        env_file_encoding = "utf-8"
        extra = "allow"


settings = Settings()
