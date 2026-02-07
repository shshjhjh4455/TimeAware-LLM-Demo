"""Configuration for TimeAware webapp backend."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Database paths
DATABASE_DIR = str(PROJECT_ROOT / "knowledge_enriched_rag" / "databases")

# Pipeline parameters
PIPELINE_CONFIG = {
    "database_dir": DATABASE_DIR,
    "use_vector_retrieval": True,
    "enable_detailed_logging": True,
    "device": "cuda",
    "top_k_chapters": 3,
    "knowledge_order": "chronological",
    "memory_top_k": 5,
    "memory_min_similarity": 0.45,
    "no_external_knowledge": False,
}

# Character configurations for Harry Potter
CHARACTER_CONFIG = {
    "Harry Potter": {
        "periods": [
            {"label": "1st-year / on the 1st of September", "value": "1st-year / on the 1st of september"},
            {"label": "1st-year / on Halloween", "value": "1st-year / on halloween"},
            {"label": "1st-year / on Christmas", "value": "1st-year / on christmas"},
            {"label": "1st-year / at the end of the scene", "value": "1st-year / at the end of the scene"},
            {"label": "2nd-year / on the 1st of September", "value": "2nd-year / on the 1st of september"},
            {"label": "2nd-year / on Halloween", "value": "2nd-year / on halloween"},
            {"label": "2nd-year / on Christmas", "value": "2nd-year / on christmas"},
            {"label": "2nd-year / at the end of the scene", "value": "2nd-year / at the end of the scene"},
            {"label": "3rd-year / on the 1st of September", "value": "3rd-year / on the 1st of september"},
            {"label": "3rd-year / on Halloween", "value": "3rd-year / on halloween"},
            {"label": "3rd-year / on Christmas", "value": "3rd-year / on christmas"},
            {"label": "3rd-year / at the end of the scene", "value": "3rd-year / at the end of the scene"},
            {"label": "4th-year / on the 1st of September", "value": "4th-year / on the 1st of september"},
            {"label": "4th-year / on Halloween", "value": "4th-year / on halloween"},
            {"label": "4th-year / on Christmas", "value": "4th-year / on christmas"},
            {"label": "4th-year / at the end of the scene", "value": "4th-year / at the end of the scene"},
            {"label": "5th-year / on the 1st of September", "value": "5th-year / on the 1st of september"},
            {"label": "5th-year / on Christmas", "value": "5th-year / on christmas"},
            {"label": "5th-year / at the end of the scene", "value": "5th-year / at the end of the scene"},
            {"label": "6th-year / on the 1st of September", "value": "6th-year / on the 1st of september"},
            {"label": "6th-year / on Christmas", "value": "6th-year / on christmas"},
            {"label": "6th-year / at the end of the scene", "value": "6th-year / at the end of the scene"},
            {"label": "7th-year / on the 1st of September", "value": "7th-year / on the 1st of september"},
            {"label": "7th-year / on Christmas", "value": "7th-year / on christmas"},
            {"label": "7th-year / at the end of the scene", "value": "7th-year / at the end of the scene"},
        ],
        "avatar": "HP",
    },
    "Hermione Granger": {
        "periods": [
            {"label": "1st-year / on the 1st of September", "value": "1st-year / on the 1st of september"},
            {"label": "1st-year / on Halloween", "value": "1st-year / on halloween"},
            {"label": "1st-year / on Christmas", "value": "1st-year / on christmas"},
            {"label": "1st-year / at the end of the scene", "value": "1st-year / at the end of the scene"},
            {"label": "2nd-year / on the 1st of September", "value": "2nd-year / on the 1st of september"},
            {"label": "2nd-year / on Halloween", "value": "2nd-year / on halloween"},
            {"label": "2nd-year / on Christmas", "value": "2nd-year / on christmas"},
            {"label": "2nd-year / at the end of the scene", "value": "2nd-year / at the end of the scene"},
            {"label": "3rd-year / on the 1st of September", "value": "3rd-year / on the 1st of september"},
            {"label": "3rd-year / on Halloween", "value": "3rd-year / on halloween"},
            {"label": "3rd-year / on Christmas", "value": "3rd-year / on christmas"},
            {"label": "3rd-year / at the end of the scene", "value": "3rd-year / at the end of the scene"},
            {"label": "4th-year / on the 1st of September", "value": "4th-year / on the 1st of september"},
            {"label": "4th-year / on Halloween", "value": "4th-year / on halloween"},
            {"label": "4th-year / on Christmas", "value": "4th-year / on christmas"},
            {"label": "4th-year / at the end of the scene", "value": "4th-year / at the end of the scene"},
            {"label": "5th-year / on the 1st of September", "value": "5th-year / on the 1st of september"},
            {"label": "5th-year / on Christmas", "value": "5th-year / on christmas"},
            {"label": "5th-year / at the end of the scene", "value": "5th-year / at the end of the scene"},
            {"label": "6th-year / on the 1st of September", "value": "6th-year / on the 1st of september"},
            {"label": "6th-year / on Christmas", "value": "6th-year / on christmas"},
            {"label": "6th-year / at the end of the scene", "value": "6th-year / at the end of the scene"},
            {"label": "7th-year / on the 1st of September", "value": "7th-year / on the 1st of september"},
            {"label": "7th-year / on Christmas", "value": "7th-year / on christmas"},
            {"label": "7th-year / at the end of the scene", "value": "7th-year / at the end of the scene"},
        ],
        "avatar": "HG",
    },
    "Ron Weasley": {
        "periods": [
            {"label": "1st-year / on the 1st of September", "value": "1st-year / on the 1st of september"},
            {"label": "1st-year / on Halloween", "value": "1st-year / on halloween"},
            {"label": "1st-year / on Christmas", "value": "1st-year / on christmas"},
            {"label": "1st-year / at the end of the scene", "value": "1st-year / at the end of the scene"},
            {"label": "2nd-year / on the 1st of September", "value": "2nd-year / on the 1st of september"},
            {"label": "2nd-year / on Halloween", "value": "2nd-year / on halloween"},
            {"label": "2nd-year / on Christmas", "value": "2nd-year / on christmas"},
            {"label": "2nd-year / at the end of the scene", "value": "2nd-year / at the end of the scene"},
            {"label": "3rd-year / on the 1st of September", "value": "3rd-year / on the 1st of september"},
            {"label": "3rd-year / on Halloween", "value": "3rd-year / on halloween"},
            {"label": "3rd-year / on Christmas", "value": "3rd-year / on christmas"},
            {"label": "3rd-year / at the end of the scene", "value": "3rd-year / at the end of the scene"},
            {"label": "4th-year / on the 1st of September", "value": "4th-year / on the 1st of september"},
            {"label": "4th-year / on Halloween", "value": "4th-year / on halloween"},
            {"label": "4th-year / on Christmas", "value": "4th-year / on christmas"},
            {"label": "4th-year / at the end of the scene", "value": "4th-year / at the end of the scene"},
            {"label": "5th-year / on the 1st of September", "value": "5th-year / on the 1st of september"},
            {"label": "5th-year / on Christmas", "value": "5th-year / on christmas"},
            {"label": "5th-year / at the end of the scene", "value": "5th-year / at the end of the scene"},
            {"label": "6th-year / on the 1st of September", "value": "6th-year / on the 1st of september"},
            {"label": "6th-year / on Christmas", "value": "6th-year / on christmas"},
            {"label": "6th-year / at the end of the scene", "value": "6th-year / at the end of the scene"},
            {"label": "7th-year / on the 1st of September", "value": "7th-year / on the 1st of september"},
            {"label": "7th-year / on Christmas", "value": "7th-year / on christmas"},
            {"label": "7th-year / at the end of the scene", "value": "7th-year / at the end of the scene"},
        ],
        "avatar": "RW",
    },
}

# Flask settings
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "false").lower() == "true"
