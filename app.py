import streamlit as st
from PIL import Image
import numpy as np
import tempfile
import os

st.set_page_config(page_title="Image Caption Generator", layout="centered")


@st.cache_resource
def load_generator():
    """
    Loads caption_generator.py (and therefore the trained model + ResNet50)
    only ONCE across the whole Streamlit session, instead of on every
    button click / script rerun.
    """
    import caption_generator as generate
    return generate


generate = load_generator()

st.title("Image Caption Generator Using Deep Learning")

uploaded_file = st.file_uploader(
    "Select an image file",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Generate Caption", type="primary"):
        # caption_generator's functions expect a file path, so save the
        # uploaded file to a temporary location on disk first
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        try:
            with st.spinner("Encoding the image..."):
                photo = generate.encode_image(tmp_path).reshape((1, 2048))

            with st.spinner("Generating caption..."):
                caption = generate.predict_caption(photo)

            st.subheader("Generated Caption")
            st.write(caption)
        finally:
            os.remove(tmp_path)  # clean up the temp file regardless of success/failure
else:
    st.info("Upload a JPG or PNG image to get started.")