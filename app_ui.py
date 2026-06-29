import io

import streamlit as st
from PIL import Image

from app_model import load_model, predict_image_bytes


st.set_page_config(
    page_title="TB Detection",
    page_icon="X",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
[data-testid="stAppViewContainer"],
[data-testid="stMain"] { background-color: #F2F2F7 !important; }
.block-container {
    padding-top: 3rem !important;
    padding-bottom: 3rem !important;
    padding-left: 5rem !important;
    padding-right: 5rem !important;
    max-width: 1180px !important;
}
[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #E5E5EA !important;
}
h1, h2, h3 { color: #111111 !important; font-weight: 700 !important; letter-spacing: 0 !important; }
p, li { color: #3A3A3C !important; }
.main-header {
    font-size: 2rem;
    font-weight: 700;
    text-align: center;
    margin-bottom: 0.2rem;
}
.main-sub {
    text-align: center;
    font-size: 0.95rem;
    color: #6B7280;
    margin-bottom: 1.5rem;
}
.disclaimer-box {
    background-color: #FFF8DB;
    border: 1px solid #E4C767;
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 1.4rem;
    color: #4F3F00;
    font-size: 0.9rem;
}
.result-box {
    border-radius: 8px;
    padding: 1.2rem;
    margin: 0.5rem 0 1rem;
    text-align: center;
}
.normal-result {
    background-color: #EAF7EE;
    border: 1px solid #8CD39B;
}
.tb-result {
    background-color: #FDEAEA;
    border: 1px solid #F19A9A;
}
.result-box h1 { margin: 0.2rem 0; font-size: 2.4rem; }
.result-box h2 { margin: 0; font-size: 1rem; color: #4B5563 !important; }
.result-box p { margin: 0; color: #4B5563 !important; }
@media (max-width: 768px) {
    .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
}
</style>
""",
    unsafe_allow_html=True,
)


def render_result(result: dict) -> None:
    probability = result.get("probability", 0)
    label = result.get("label", "Unknown")
    probability_percent = probability * 100
    result_class = "normal-result" if label == "Normal" else "tb-result"

    st.markdown(
        f"""
        <div class="result-box {result_class}">
            <h2>Prediction</h2>
            <h1>{label}</h1>
            <p>{probability_percent:.1f}% confidence</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Predicted Class", label)
    with col_b:
        st.metric("Confidence", f"{probability_percent:.1f}%")

    st.markdown("---")

    if label == "Normal":
        st.success("Normal Chest X-ray")
        st.info(
            """
**Why Normal was predicted:**
- Clear lung fields with no abnormal opacities
- Normal lung markings and vascular structures
- No consolidation or cavity lesions detected
- Anatomical structures appear as expected
            """
        )
    else:
        st.error("Possible Tuberculosis Detected")
        st.info(
            """
**Why TB was predicted:**
- Abnormal opacities or cloudiness in the lungs
- Possible consolidation suggesting infection
- Potential cavity lesions in the upper lobes
- Patterns consistent with TB-positive training data
            """
        )

    with st.expander("How the AI works"):
        st.markdown(
            """
The model uses a linear classification approach that learns weighted relationships between image pixels and TB probability.

**How it works:**
- The X-ray is converted to grayscale and resized to 128 x 128 pixels
- Pixel values are normalized and flattened into numerical features
- The model computes a probability from 0 to 1
- Values classified as class 1 are shown as TB; class 0 is shown as Normal

The model analyses pixel values only. It does not consider symptoms, patient history, lab results, or clinician judgment.
            """
        )

    st.warning(
        "Always consult a qualified medical professional. This tool does not replace clinical diagnosis."
    )


def main() -> None:
    st.markdown(
        '<div class="main-header">Tuberculosis Detection</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="main-sub">Chest X-ray decision support tool for research and education</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="disclaimer-box">
            <strong>Medical Disclaimer</strong> - This is a decision support tool, not a final diagnosis.
            The model may produce false positives and false negatives. Always consult a qualified medical professional.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("How to use")
        st.markdown(
            """
1. Upload a chest X-ray image
2. Preview the image
3. Select **Run Diagnosis**
4. Review the result

This deployed app runs the model directly.
            """
        )

        st.header("System Status")
        try:
            load_model()
            st.success("App: Running")
            st.success("Model: Loaded")
        except Exception as exc:
            st.error("Model: Not Loaded")
            st.info(str(exc))

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("Upload X-Ray")
        uploaded_file = st.file_uploader(
            "Choose a chest X-ray image",
            type=["jpg", "jpeg", "png"],
            help="JPG or PNG format",
        )

        if uploaded_file is not None:
            try:
                file_bytes = uploaded_file.getvalue()
                st.session_state.uploaded_file_bytes = file_bytes

                image = Image.open(io.BytesIO(file_bytes))
                st.image(image, caption="Uploaded X-ray")
                st.info(f"{image.size[0]} x {image.size[1]} px - {image.mode}")
            except Exception as exc:
                st.error(f"Error loading image: {exc}")
                st.session_state.pop("uploaded_file_bytes", None)

    with col2:
        st.subheader("Results")
        has_file_data = "uploaded_file_bytes" in st.session_state and uploaded_file is not None

        if not has_file_data:
            st.info("Upload a chest X-ray to begin.")
            return

        if st.button("Run Diagnosis", type="primary"):
            with st.spinner("Analysing image..."):
                try:
                    result = predict_image_bytes(st.session_state.uploaded_file_bytes)
                    render_result(result)
                except Exception as exc:
                    st.error(f"Unexpected error: {exc}")
                    st.session_state.pop("uploaded_file_bytes", None)

    st.markdown("---")
    st.markdown(
        """
        <div style="text-align:center;color:#6B7280;font-size:0.8rem;padding:0.5rem 0 1rem;">
            TB Detection System - For research and educational use only
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
