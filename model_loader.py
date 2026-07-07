"""
model_loader.py

Loads the LocateAnything-3B model and processor.
"""

import torch
from transformers import AutoModel, AutoProcessor

from config import MODEL_NAME


def load_model():
    """
    Load the LocateAnything processor and model.

    Returns:
        processor: Hugging Face processor
        model: LocateAnything model
    """

    print("Loading processor...")

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )

    print("Loading model...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )

    model.to(device)

    model.eval()

    print(f"Model loaded successfully on {device.upper()}!")

    return processor, model