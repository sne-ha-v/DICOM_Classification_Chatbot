import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Application configuration"""
    APP_TITLE = "Lung Nodule AI Assistant"
    APP_ICON = "🫁"
    PAGE_LAYOUT = "wide"

    # Model settings
    MODEL_PATH = "models/nodule_classifier_resnet_attention.h5"
    CONFIDENCE_THRESHOLD = 0.5

    # API settings
    API_BASE_URL = "https://dicomclassificationchatbot-production.up.railway.app"

    # LLM settings
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")  # Optional - free tier works without
    HUGGINGFACE_MODEL = "microsoft/DialoGPT-medium"  # Open-source conversational model

    # Dataset info
    DATASET_NAME = "LUNA16"
    TOTAL_SCANS = 888
    TOTAL_NODULES = 1186
    NUM_RADIOLOGISTS = 4

    # File upload settings
    ALLOWED_EXTENSIONS = ["nii", "nii.gz", "dcm", "dicom", "tcia"]
    MAX_FILE_SIZE = 100  # MB