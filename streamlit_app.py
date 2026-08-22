import os
import tempfile
import streamlit as st

from agent.plant_agent import plant_agent


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="AI Plant Doctor",
    page_icon="🌿",
    layout="wide"
)


# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f0fdf4, #dcfce7);
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    /* HERO */

    .hero {
        background: linear-gradient(135deg, #14532d, #15803d);
        border-radius: 28px;
        padding: 45px 25px;
        text-align: center;
        margin-bottom: 40px;
        box-shadow: 0 15px 40px rgba(20, 83, 45, 0.20);
    }

    .hero-title {
        color: white;
        font-size: 48px;
        font-weight: 800;
        margin: 0;
    }

    .hero-subtitle {
        color: #dcfce7;
        font-size: 19px;
        margin-top: 12px;
    }

    /* SECTION */

    .section-title {
        color: #14532d;
        font-size: 30px;
        font-weight: 800;
        margin-top: 30px;
        margin-bottom: 20px;
    }

    /* INFO */

    .info-card {
        background: white;
        border: 1px solid #d1fae5;
        border-radius: 22px;
        padding: 25px;
        margin-bottom: 25px;
        box-shadow: 0 10px 30px rgba(20, 83, 45, 0.08);
    }

    .info-title {
        color: #166534;
        font-size: 23px;
        font-weight: 750;
        margin-bottom: 10px;
    }

    .info-text {
        color: #475569;
        font-size: 16px;
        line-height: 1.6;
    }

    /* RESULT CARDS */

    .result-card {
        background: white;
        border: 1px solid #d1fae5;
        border-radius: 22px;
        padding: 28px 20px;
        min-height: 150px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(20, 83, 45, 0.08);
    }

    .result-label {
        color: #64748b;
        font-size: 17px;
        font-weight: 650;
        margin-bottom: 18px;
    }

    .result-value {
        color: #166534;
        font-size: 26px;
        font-weight: 800;
        word-break: break-word;
    }

    /* ADVICE */

    .advice-card {
        background: white;
        border: 1px solid #d1fae5;
        border-radius: 22px;
        padding: 30px;
        margin-top: 10px;
        color: #1f2937;
        box-shadow: 0 10px 30px rgba(20, 83, 45, 0.08);
    }

    .advice-card h1,
    .advice-card h2,
    .advice-card h3,
    .advice-card h4 {
        color: #166534 !important;
    }

    .advice-card p,
    .advice-card li {
        color: #374151 !important;
        line-height: 1.7;
    }

    .footer {
        text-align: center;
        color: #64748b;
        margin-top: 50px;
    }
    /* Fix Streamlit text visibility on light background */

.stMarkdown,
.stMarkdown p,
.stMarkdown li,
.stMarkdown span {
    color: #374151 !important;
}

.stCaption,
[data-testid="stCaptionContainer"] {
    color: #64748b !important;
}

[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] p {
    color: #374151 !important;
}

[data-testid="stSpinner"] {
    color: #374151 !important;
}

[data-testid="stSpinner"] p {
    color: #374151 !important;
}

.stProgress + div,
.stProgress + div p {
    color: #374151 !important;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# HERO
# ============================================================

st.markdown(
    '<div class="hero">'
    '<div class="hero-title">🌿 AI Plant Doctor</div>'
    '<div class="hero-subtitle">'
    'Deep Learning + Generative AI for Plant Health'
    '</div>'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# UPLOAD
# ============================================================

st.markdown(
    '<div class="section-title">📷 Upload Your Plant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="info-card">'
    '<div class="info-title">Upload a clear plant leaf image</div>'
    '<div class="info-text">'
    'Our AI will analyze the image, identify the possible disease, '
    'estimate confidence and severity, and generate personalized '
    'treatment advice.'
    '</div>'
    '</div>',
    unsafe_allow_html=True
)


uploaded_file = st.file_uploader(
    "Choose a plant image",
    type=["jpg", "jpeg", "png", "webp"]
)


# ============================================================
# AFTER IMAGE UPLOAD
# ============================================================

if uploaded_file is not None:

    st.markdown(
        '<div class="section-title">🌱 Selected Plant</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(
        [1, 1],
        gap="large"
    )

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    with col1:

        st.image(
            uploaded_file,
            caption="Selected plant image",
            use_container_width=True
        )

    # --------------------------------------------------------
    # ANALYZE PANEL
    # --------------------------------------------------------

    with col2:

        st.markdown(
            '<div class="info-card">'
            '<div class="info-title">🔍 Ready to analyze?</div>'
            '<div class="info-text">'
            'Click the button below to run the complete AI pipeline. '
            'The system will detect the disease, calculate severity, '
            'and generate treatment advice using Generative AI.'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )

        analyze = st.button(
            "🔍 Analyze Plant",
            type="primary",
            use_container_width=True
        )


    # ========================================================
    # ANALYSIS
    # ========================================================

    if analyze:

        temp_path = None

        try:

            # ------------------------------------------------
            # SAVE TEMP IMAGE
            # ------------------------------------------------

            extension = os.path.splitext(
                uploaded_file.name
            )[1]

            if extension.lower() not in [
                ".jpg",
                ".jpeg",
                ".png",
                ".webp"
            ]:
                extension = ".jpg"

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=extension
            ) as temp:

                temp.write(
                    uploaded_file.getbuffer()
                )

                temp_path = temp.name


            # ------------------------------------------------
            # RUN YOUR EXISTING AI AGENT
            # ------------------------------------------------

            with st.spinner(
                "🌿 AI Plant Doctor is analyzing your plant..."
            ):

                result = plant_agent(temp_path)


            # ------------------------------------------------
            # CHECK RESULT
            # ------------------------------------------------

            if not isinstance(result, dict):

                st.error(
                    "The AI agent returned an invalid result."
                )

            elif "message" in result:

                st.warning(
                    f"⚠️ {result['message']}"
                )

            else:

                # ============================================
                # GET DATA
                # ============================================

                disease = result.get(
                    "disease",
                    "Unknown"
                )

                confidence = result.get(
                    "confidence",
                    0
                )

                severity = result.get(
                    "severity",
                    "Unknown"
                )

                treatment = result.get(
                    "treatment",
                    "No treatment advice available."
                )


                # ============================================
                # CLEAN DATA
                # ============================================

                disease = str(
                    disease
                ).replace(
                    "_",
                    " "
                )

                severity = str(
                    severity
                )


                try:

                    confidence = float(
                        confidence
                    )

                except:

                    confidence = 0


                if confidence > 1:

                    confidence /= 100


                confidence = max(
                    0,
                    min(
                        confidence,
                        1
                    )
                )


                confidence_percent = (
                    confidence * 100
                )


                # ============================================
                # REPORT
                # ============================================

                st.markdown(
                    '<div class="section-title">'
                    '🧠 Plant Health Report'
                    '</div>',
                    unsafe_allow_html=True
                )


                r1, r2, r3 = st.columns(
                    3,
                    gap="large"
                )


                # --------------------------------------------
                # DISEASE
                # --------------------------------------------

                with r1:

                    st.markdown(
                        '<div class="result-card">'
                        '<div class="result-label">'
                        '🌿 Disease Detected'
                        '</div>'
                        '<div class="result-value">'
                        + disease +
                        '</div>'
                        '</div>',
                        unsafe_allow_html=True
                    )


                # --------------------------------------------
                # CONFIDENCE
                # --------------------------------------------

                with r2:

                    st.markdown(
                        '<div class="result-card">'
                        '<div class="result-label">'
                        '🎯 Confidence'
                        '</div>'
                        '<div class="result-value">'
                        f'{confidence_percent:.1f}%'
                        '</div>'
                        '</div>',
                        unsafe_allow_html=True
                    )


                # --------------------------------------------
                # SEVERITY
                # --------------------------------------------

                with r3:

                    severity_lower = severity.lower()

                    if severity_lower == "high":

                        icon = "🔴"
                        color = "#dc2626"

                    elif severity_lower == "medium":

                        icon = "🟠"
                        color = "#d97706"

                    else:

                        icon = "🟢"
                        color = "#15803d"


                    st.markdown(
                        '<div class="result-card">'
                        '<div class="result-label">'
                        '⚠️ Severity'
                        '</div>'
                        f'<div style="'
                        f'color:{color};'
                        f'font-size:26px;'
                        f'font-weight:800;">'
                        f'{icon} {severity}'
                        '</div>'
                        '</div>',
                        unsafe_allow_html=True
                    )


                # ============================================
                # CONFIDENCE BAR
                # ============================================

                st.markdown(
                    '<div class="section-title">'
                    '🎯 Detection Confidence'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.progress(
                    confidence
                )

                st.markdown(
                    f"**{confidence_percent:.1f}% confidence**"
                )


                # ============================================
                # AI ADVICE
                # ============================================

                st.markdown(
                    '<div class="section-title">'
                    '🤖 AI Plant Doctor Advice'
                    '</div>',
                    unsafe_allow_html=True
                )


                # IMPORTANT:
                # Do NOT put treatment inside custom HTML.
                # Streamlit renders Gemini's Markdown directly.

                # st.markdown(
                #     '<div class="advice-card">',
                #     unsafe_allow_html=True
                # )

                st.markdown(
                    str(treatment)
                )

                # st.markdown(
                #     '</div>',
                #     unsafe_allow_html=True
                # )


        # =====================================================
        # ERROR
        # =====================================================

        except Exception as e:

            st.error(
                "❌ An error occurred during analysis."
            )

            st.exception(e)


        # =====================================================
        # DELETE TEMP FILE
        # =====================================================

        finally:

            if temp_path is not None:

                try:

                    if os.path.exists(temp_path):

                        os.remove(
                            temp_path
                        )

                except:

                    pass


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="footer">'
    '🌱 AI Plant Doctor · Deep Learning + Generative AI'
    '</div>',
    unsafe_allow_html=True
)