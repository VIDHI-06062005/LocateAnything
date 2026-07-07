"""
inference.py

Runs inference using NVIDIA LocateAnything-3B.
"""

import gc

import torch
import streamlit as st

from model_loader import load_model
from config import GENERATION_CONFIG


# ---------------------------------------------------
# Load model only once
# ---------------------------------------------------

@st.cache_resource
def get_model():
    return load_model()


processor, model = get_model()


def run_inference(image, query):
    """
    Run inference using LocateAnything-3B.

    Args:
        image (PIL.Image): Input image
        query (str): User query

    Returns:
        str: Model response
    """

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image
                },
                {
                    "type": "text",
                    "text": query
                }
            ]
        }
    ]

    text = processor.py_apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    images, videos = processor.process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=images,
        videos=videos,
        return_tensors="pt"
    )

    inputs = {
        k: v.to(model.device) if hasattr(v, "to") else v
        for k, v in inputs.items()
    }

    try:

        with torch.inference_mode():

            response = model.generate(
                pixel_values=inputs["pixel_values"],
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image_grid_hws=inputs.get("image_grid_hws"),
                tokenizer=processor.tokenizer,
                **GENERATION_CONFIG
            )

    except Exception as e:

        raise RuntimeError(f"Inference failed: {e}")

    finally:

        del inputs

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    generated_text = response[0]

    if isinstance(generated_text, list):
        generated_text = generated_text[0]

    return generated_text