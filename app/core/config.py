"""
Core configuration for the medical imaging API
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # API Configuration
    API_TITLE = "Medical Imaging AI Endpoints"
    API_VERSION = "1.0.0"
    API_HOST = "0.0.0.0"
    API_PORT = 8000

    # Model Paths
    ATTCO_MODEL_PATH = "./AttCo/checkpoint/BraTS2020/JointFusionNet3D_v11/model.pt"
    SAM_MODEL_NAME = "Lorenzob/sam-brain-tumor-segmentation"
    BLIP2_MODEL_NAME = "Salesforce/blip2-opt-2.7b"

    # Supabase Configuration
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")

    # Other API Configuration
    OLLAMA_API_URL = os.getenv("API_URL", "http://localhost:11434/api/generate")
    OLLAMA_MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-ocr")

    # Default Storage
    DEFAULT_BUCKET_NAME = "hackathon-bucket"


settings = Settings()
