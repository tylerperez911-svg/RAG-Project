import os


class Settings:
    OLLAMA_URL: str = os.getenv(
        "OLLAMA_URL",
        "http://localhost:11434"
    )

    MODEL_NAME: str = os.getenv(
        "MODEL_NAME",
        "llama3.2:1b"
    )

    CHROMA_PATH: str = os.getenv(
        "CHROMA_PATH",
        "/app/chroma_data"
    )

    MAX_RESULTS: int = int(
        os.getenv("MAX_RESULTS", "3")
    )

    CONFIDENCE_THRESHOLD: float = float(
        os.getenv("CONFIDENCE_THRESHOLD", "0.0")
    )

    DEBUG: bool = os.getenv(
        "DEBUG",
        "false"
    ).lower() == "true"


settings = Settings()