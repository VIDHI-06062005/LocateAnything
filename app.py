"""
app.py

Streamlit application for NVIDIA LocateAnything-3B

Features:
- Single image upload
- Multiple queries
- Automatic image resize
- Download annotated results
"""
import os 
import streamlit as st
from PIL import Image

from inference import run_inference
from visualization import (
    extract_prediction,
    draw_prediction
)

from utils import (
    save_image,
    validate_query
)

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="LocateAnything-3B",
    page_icon="🔍",
    layout="wide"
)

# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🔍 LocateAnything-3B")

st.write(
    "Upload an image and enter one or more queries "
    "(one query per line)."
)

# --------------------------------------------------
# Upload Image
# --------------------------------------------------

uploaded_files = st.file_uploader(
    "Upload Images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# --------------------------------------------------
# Queries
# --------------------------------------------------

queries = st.text_area(
    "Enter one query per line",
    height=150,
    placeholder="""Find the chair
Find the bottle
Find the laptop
Find the person"""
)

# --------------------------------------------------
# Run Button
# --------------------------------------------------

if st.button("🚀 Run Inference"):

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    if not uploaded_files:
        st.error("Please upload an image.")
        st.stop()

    if not validate_query(queries):
        st.error("Please enter at least one query.")
        st.stop()

    # --------------------------------------------------
    # Query List
    # --------------------------------------------------

    query_list = [
        q.strip()
        for q in queries.split("\n")
        if q.strip()
    ]

    MAX_SIZE = 1024

    total_steps = len(uploaded_files) * len(query_list)

    current_step = 0
 
    progress_bar = st.progress(0)

    # --------------------------------------------------
    # Loop through Images
    # --------------------------------------------------

    for uploaded_file in uploaded_files:

        image = Image.open(uploaded_file).convert("RGB")

        if max(image.size) > MAX_SIZE:

            image.thumbnail(
                (MAX_SIZE, MAX_SIZE),
                Image.Resampling.LANCZOS
            )

        st.header(f"📷 {uploaded_file.name}")

        st.image(
            image,
            width=500
        )

        st.divider()


    # --------------------------------------------------
    # Run Each Query
    # --------------------------------------------------

        for index, query in enumerate(query_list):

            current_step += 1

            progress_bar.progress(
                current_step / total_steps
            )

            st.subheader(
                f"🔎 Query {index + 1}: {query}"
            )

            with st.spinner("Running inference..."):

                generated_text = run_inference(
                    image,
                    query
                )

            label, box = extract_prediction(
                generated_text
            )

            if label is None or box is None:

                st.warning(
                    "No object detected."
                )

                st.divider()

                continue

            result_image = draw_prediction(
                image,
                label,
                box
            )
            base_name = os.path.splitext(uploaded_file.name)[0]

            filename = f"{base_name}_{label}_{index}.jpg"
            
            save_path = save_image(
                result_image,
                filename=filename
                
            )

            st.image(
                result_image,
                caption=f"Detected: {label}",
                width=500
            )

            st.success(
                f"Query: {query} → Detected: {label}"
            )

            with open(save_path, "rb") as file:
 
                st.download_button(
                    label="📥 Download Result",
                    data=file,
                    file_name=filename,
                    mime="image/jpeg",
                    key=f"{base_name}_{label}_{index}"
                )

            st.divider()

    progress_bar.empty()

# --------------------------------------------------
# Footer
# --------------------------------------------------

st.markdown("---")

st.caption(
    "Powered by NVIDIA LocateAnything-3B"
)