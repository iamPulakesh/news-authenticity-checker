import sys
import os
import html
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rag.vectorstore import get_embeddings
from app.multimodal.ocr import _get_easyocr_reader

from dotenv import load_dotenv
load_dotenv()
import streamlit as st

st.set_page_config(
    page_title="News Authenticity Checker",
    page_icon="",
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items={},
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global base size */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
        font-size: 16px !important;
    }
    .main { padding-top: 0.5rem; }
    
    /* Hide scrollbar */
    ::-webkit-scrollbar {
        width: 0px !important;
        display: none !important;
    }
    * {
        scrollbar-width: none !important; 
        -ms-overflow-style: none !important;
    }
    
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }

    /* Streamlit element overrides */
    .stMarkdown, .stMarkdown p, .stText, .stCaption,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] {
        font-size: 1rem !important;
    }
    .stTextInput input, .stTextArea textarea {
        font-size: 1rem !important;
    }
    label, .stRadio label, .stCheckbox label {
        font-size: 0.95rem !important;
    }

    [data-testid="stToolbar"] { display: none !important; }
    #MainMenu { display: none !important; }
    footer { display: none !important; }
    header { display: none !important; }
    [data-testid="stSidebar"] { display: none !important; }
    [data-testid="collapsedControl"] { display: none !important; }

    /* Hero header */
    .hero-wrap {
        text-align: center;
        padding: 1.2rem 0 0.4rem;
    }
    .hero-title {
        font-size: 2.2rem !important;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.15rem;
        letter-spacing: -0.5px;
    }
    .hero-sub {
        font-size: 1.05rem !important;
        color: #888;
        margin-bottom: 0.8rem;
    }

    /* Disclaimer */
    .disclaimer {
        background: linear-gradient(135deg, #1a1a2e 0%, #23274a 100%);
        border-radius: 10px;
        padding: 0.75rem 1.1rem;
        margin-bottom: 1rem;
        border-left: 3px solid #ffc107;
        color: #c8d0dc;
        font-size: 0.95rem !important;
        line-height: 1.55;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.3rem;
        justify-content: center;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1rem !important;
        padding: 0.5rem 1.2rem;
        border-radius: 8px;
    }

    /* Button */
    .stButton > button {
        font-size: 1rem !important;
        padding: 0.55rem 2.2rem !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        letter-spacing: 0.3px;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #E72107 0%, #E72107 100%) !important;
        border: none !important;
        color: #fff !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px);
    }

    /* Verdict cards */
    .verdict-card {
        border-radius: 14px;
        padding: 1.5rem 1.8rem;
        margin: 0.8rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,.1);
    }
    .verdict-real       { background: linear-gradient(135deg, #0f5132 0%, #198754 100%); color: #fff; }
    .verdict-fake       { background: linear-gradient(135deg, #842029 0%, #dc3545 100%); color: #fff; }
    .verdict-misleading { background: linear-gradient(135deg, #664d03 0%, #ffc107 100%); color: #000; }
    .verdict-unverified { background: linear-gradient(135deg, #41464b 0%, #6c757d 100%); color: #fff; }
    .verdict-label { font-size: 2rem !important; font-weight: 700; }
    .verdict-conf  { font-size: 1.1rem !important; opacity: 0.9; margin-top: 0.2rem; }

    /* Claim chips */
    .claim-chip {
        border-radius: 10px;
        padding: 0.85rem 1.1rem;
        margin-bottom: 0.5rem;
        box-shadow: 0 1px 6px rgba(0,0,0,.05);
        font-size: 1rem !important;
    }
    .claim-supported    { border-left: 3px solid #198754; background: #0d2818; color: #d4edda; }
    .claim-contradicted { border-left: 3px solid #dc3545; background: #2c0b0e; color: #f5c6cb; }
    .claim-unverifiable { border-left: 3px solid #6c757d; background: #1e1e2a; color: #ccc; }
    .claim-title { font-weight: 600; font-size: 1rem !important; }
    .claim-evidence { font-size: 0.92rem !important; color: #aaa; margin-top: 4px; }
    .claim-badge {
        display: inline-block;
        font-size: 0.78rem !important;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 5px;
        margin-left: 6px;
        vertical-align: middle;
    }
    .badge-high   { background: #198754; color: #fff; }
    .badge-medium { background: #ffc107; color: #000; }
    .badge-low    { background: #6c757d; color: #fff; }

    /* Source pills */
    .source-pill {
        display: inline-block;
        background: #2a2e3e;
        color: #ccc;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.9rem !important;
        margin: 3px 4px;
    }

    /* Hide anchor link icon on headings */
    .hero-title a,
    h1 a, h2 a, h3 a {
        display: none !important;
    }
    [data-testid="stHeaderActionElements"] {
        display: none !important;
    }

    /* Section headers */
    .section-head {
        font-size: 1.2rem !important;
        font-weight: 600;
        margin: 1.1rem 0 0.5rem;
        color: #ffc107;
    }

    /* Empty state */
    .empty-state {
        text-align: center;
        padding: 2.5rem 0;
        color: #999;
    }
    .empty-state .icon { font-size: 2.8rem !important; margin-bottom: 0.4rem; }
    .empty-state .msg  { font-size: 1.1rem !important; }
    .empty-state .sub  { font-size: 0.95rem !important; color: #aaa; }
</style>
""", unsafe_allow_html=True)


st.markdown(
    '<div class="hero-wrap">'
    '<h1 class="hero-title">News Authenticity Checker</h1>'
    '<p class="hero-sub">Built to fight misinformation on the internet</p>'
    '</div>',
    unsafe_allow_html=True,
)


st.markdown(
    '<div class="disclaimer">'
    '<strong>Disclaimer</strong> -- Results may not always be 100% accurate. Although this tool achieves strong accuracy on well-sourced news articles, it is not infallible. '
    'Always verify critical and controversial claims with professional fact-checkers and primary sources.'
    '</div>',
    unsafe_allow_html=True,
)

@st.cache_resource(show_spinner=False)
def preload_models():
    """Wakes up the large HuggingFace + OCR models on boot so the first user run is fast."""

    # Pre-load embedding models into memory
    get_embeddings()
    # Pre-load EasyOCR into memory
    _get_easyocr_reader()
    return True

with st.spinner("Waking up models... (this may take a few seconds)"):
    preload_models()

tab_url, tab_image, tab_text = st.tabs(["Paste URL", "Image Upload", "Paste Text"])

input_value = None
input_is_text = False

with tab_url:
    url_input = st.text_input(
        "Enter a news article URL",
        placeholder="https://www.bbc.com/news/...",
        key="url_input",
        label_visibility="collapsed",
    )
    if url_input:
        input_value = url_input.strip()

with tab_image:
    uploaded = st.file_uploader(
        "Upload a screenshot or image",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        key="image_upload",
        label_visibility="collapsed",
    )
    if uploaded:
        st.image(uploaded, caption="Uploaded Image", width="stretch")
        suffix = Path(uploaded.name).suffix
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(uploaded.read())
        tmp.flush()
        input_value = tmp.name

with tab_text:
    text_input = st.text_area(
        "Paste article paragraph or news headlines",
        placeholder="Paste the full article text, a headline, or a paragraph...",
        height=140,
        max_chars=5000,
        key="text_input",
        label_visibility="collapsed",
    )
    if text_input and len(text_input.strip()) > 10:
        input_value = text_input.strip()
        input_is_text = True


col1, col2, col3 = st.columns([2, 1, 2])
with col2:
    run_clicked = st.button(
        "Analyze",
        width="stretch",
        type="primary",
        disabled=not input_value,
    )


if run_clicked and input_value:
    with st.spinner(""):
        progress_bar = st.progress(0, text=" Starting process...")
        progress_bar.progress(10, text=" Scanning the Web...")

        try:
            from app.agent.runner import run_fact_check

            progress_bar.progress(25, text=" Verifying facts...")
            progress_bar.progress(45, text=" Extracting claims...")
            progress_bar.progress(65, text=" Finalizing verdict...")

            verdict = run_fact_check(input_value)
            progress_bar.progress(100, text=" Done!")

        except Exception as e:
            st.error(f" Pipeline failed: {e}")
            st.stop()


    st.markdown("---")

    # Verdict card
    verdict_value = verdict.verdict.value.lower()
    css_class = f"verdict-{verdict_value}"

    st.markdown(f"""
    <div class="verdict-card {css_class}">
        <div class="verdict-label">{verdict.verdict_emoji()}</div>
        <div class="verdict-conf">Confidence: {verdict.confidence_bar()}</div>
    </div>
    """, unsafe_allow_html=True)

    # Article info
    if verdict.article_title:
        st.markdown(f"**Topic:** {html.escape(verdict.article_title)}")

    # Reasoning summary
    if verdict.reasoning_summary:
        st.markdown('<p class="section-head"> Summary</p>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="summary-box">{html.escape(verdict.reasoning_summary)}</div>',
            unsafe_allow_html=True,
        )

    # Claims analysis
    if verdict.claims_analyzed:
        st.markdown(
            f'<p class="section-head"> Claims ({len(verdict.claims_analyzed)})</p>',
            unsafe_allow_html=True,
        )

        for i, claim in enumerate(verdict.claims_analyzed, 1):
            status_lower = claim.status.lower()
            if "support" in status_lower:
                chip_class, icon = "claim-supported", ""
            elif "contradict" in status_lower:
                chip_class, icon = "claim-contradicted", ""
            else:
                chip_class, icon = "claim-unverifiable", ""

            conf_lower = claim.confidence.lower()
            badge_class = (
                "badge-high" if conf_lower == "high"
                else "badge-medium" if conf_lower == "medium"
                else "badge-low"
            )

            safe_claim    = html.escape(claim.claim)
            safe_evidence = html.escape(claim.evidence)
            st.markdown(f"""
            <div class="claim-chip {chip_class}">
                <div class="claim-title">
                    {icon} {safe_claim}
                    <span class="claim-badge {badge_class}">{claim.confidence}</span>
                </div>
                <div class="claim-evidence"> {safe_evidence}</div>
            </div>
            """, unsafe_allow_html=True)

    # Sources
    if verdict.sources_consulted:
        st.markdown('<p class="section-head"> Sources</p>', unsafe_allow_html=True)
        links_html = ""
        for src in verdict.sources_consulted:
            src = src.strip()
            if not src:
                continue
            if src.startswith("http://") or src.startswith("https://"):
                # Truncate display text for long URLs
                display = src if len(src) <= 80 else src[:77] + "..."
                links_html += (
                    f'<a href="{src}" target="_blank" '
                    f'style="display:block; color:#7eb8f7; text-decoration:none; '
                    f'font-size:0.9rem; padding:4px 0; overflow:hidden; '
                    f'text-overflow:ellipsis; white-space:nowrap;">'
                    f'{display}</a>'
                )
            else:
                links_html += f'<span class="source-pill">{src}</span>'
        st.markdown(links_html, unsafe_allow_html=True)

    # Clean up temp image
    if uploaded and input_value and os.path.exists(input_value):
        try:
            os.unlink(input_value)
        except Exception:
            pass


elif not run_clicked:
    st.markdown(
        '<div class="empty-state">'
        '<p class="msg">Paste a URL, upload an image or type a headline</p>'
        '<p class="sub">The agent will extract claims, search for evidence and sources</p>'
        '</div>',
        unsafe_allow_html=True,
    
    )
