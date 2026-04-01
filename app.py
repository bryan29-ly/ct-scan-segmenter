"""
ScanAssist — AI-powered CT scan pre-screening tool.
Automatic organ segmentation + plain-language radiology summary.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from PIL import Image

from inference import load_model, segment_slice
from report import generate_report

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ScanAssist · AI Pre-Screening",
    page_icon="🩻",
    layout="wide",
)

# ── Styling ──────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&display=swap');

    html, body, .stApp, .block-container {
        font-family: 'DM Sans', sans-serif;
    }
    .block-container {
        max-width: 1100px;
        padding-top: 1.5rem;
    }

    /* Header banner */
    .header-banner {
        background: linear-gradient(135deg, #1B1F3B 0%, #2D3561 100%);
        border-radius: 12px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        color: white;
    }
    .header-banner h1 {
        margin: 0 0 0.3rem 0;
        font-size: 1.9rem;
        font-weight: 700;
        color: white;
    }
    .header-banner p {
        margin: 0;
        font-size: 1rem;
        opacity: 0.85;
        line-height: 1.5;
    }
    .header-badge {
        display: inline-block;
        background: rgba(255,255,255,0.15);
        border: 1px solid rgba(255,255,255,0.25);
        border-radius: 20px;
        padding: 0.25rem 0.8rem;
        font-size: 0.75rem;
        margin-bottom: 0.8rem;
        letter-spacing: 0.03em;
    }

    /* Pipeline steps */
    .pipeline {
        display: flex;
        gap: 0.5rem;
        margin: 1.2rem 0 0.5rem 0;
        flex-wrap: wrap;
    }
    .pipeline-step {
        background: rgba(255,255,255,0.12);
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-size: 0.82rem;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    .pipeline-arrow {
        font-size: 0.9rem;
        opacity: 0.5;
    }

    /* Metric cards */
    .metric-row {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        flex: 1;
        background: #f7f8fc;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        border: 1px solid #e8eaf0;
    }
    .metric-card .value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1B1F3B;
    }
    .metric-card .label {
        font-size: 0.8rem;
        color: #6b7280;
        margin-top: 0.15rem;
    }

    /* Report box */
    .report-box {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 10px;
        padding: 1.5rem;
        margin-top: 0.5rem;
        line-height: 1.7;
    }
    .report-box p { margin: 0.4rem 0; }

    /* Example report */
    .example-report {
        background: #f8fafc;
        border: 1px dashed #cbd5e1;
        border-radius: 10px;
        padding: 1.5rem;
        margin-top: 0.5rem;
        line-height: 1.7;
        color: #475569;
    }
    .example-label {
        display: inline-block;
        background: #e2e8f0;
        border-radius: 4px;
        padding: 0.15rem 0.5rem;
        font-size: 0.7rem;
        font-weight: 600;
        color: #64748b;
        margin-bottom: 0.6rem;
        letter-spacing: 0.04em;
    }

    /* Disclaimer */
    .disclaimer {
        background: #fffbeb;
        border: 1px solid #fde68a;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        font-size: 0.78rem;
        color: #92400e;
        margin-top: 1rem;
    }

    /* Challenge info box */
    .challenge-box {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        font-size: 0.82rem;
        color: #1e40af;
        margin-top: 0.5rem;
    }

    .stTabs [data-baseweb="tab-list"] { gap: 2rem; }
</style>
""", unsafe_allow_html=True)

# ── Constants ────────────────────────────────────────────────────────────────

EXAMPLES_DIR = Path(__file__).parent / "examples"
NUM_CLASSES = 54

EXAMPLE_REPORT = """**Overview:** 16 anatomical structures were segmented in this CT slice, \
covering 9,197 foreground pixels (3.6% of the image).

**Key findings:**
- Two structures dominate: Structure 36 (24.4%) and Structure 09 (22.6%), \
together accounting for nearly half the segmented area.
- Structure 53 (9.3%) and Structure 34 (8.1%) are moderately sized, \
while most others are small (≤8% each).
- Structure 42 (0.5%) and Structure 43 (0.7%) are notably tiny.

**Recommendation:** No anomalies flagged — routine review recommended.

*This is an automated pre-screening. All findings must be validated by a qualified radiologist.*"""

# ── Helpers ──────────────────────────────────────────────────────────────────


@st.cache_resource
def get_model():
    return load_model()


def build_colormap(num_classes: int):
    base = plt.cm.get_cmap("tab20", 20)
    colors = [base(i % 20) for i in range(num_classes)]
    all_colors = [(0.0, 0.0, 0.0, 0.0)] + colors
    return mcolors.ListedColormap(all_colors)


def compute_structure_stats(mask: np.ndarray) -> list[dict]:
    unique_labels = np.unique(mask)
    unique_labels = unique_labels[unique_labels > 0]
    total_fg = int(np.sum(mask > 0))
    stats = []
    for label in sorted(unique_labels):
        count = int(np.sum(mask == label))
        stats.append({
            "label": int(label),
            "pixels": count,
            "pct_of_foreground": round(100 * count / total_fg, 1) if total_fg > 0 else 0.0,
        })
    return stats


def render_overlay(image: np.ndarray, mask: np.ndarray, alpha: float):
    cmap = build_colormap(NUM_CLASSES)
    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
    ax.imshow(image, cmap="gray", vmin=0, vmax=255)
    masked = np.ma.masked_where(mask == 0, mask)
    ax.imshow(masked, cmap=cmap, vmin=0, vmax=NUM_CLASSES, alpha=alpha)
    ax.axis("off")
    fig.tight_layout(pad=0)
    return fig


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("#### ⚙️ Controls")
    alpha = st.slider("Overlay opacity", 0.0, 1.0, 0.55, 0.05)

    st.markdown("---")
    st.markdown("#### 🤖 Mistral AI Report")
    mistral_key = st.text_input(
        "API Key",
        type="password",
        help="Get a free key at console.mistral.ai",
    )
    st.caption(
        "Generates a plain-language summary from segmentation output. Without a key, an example report is shown.")

    st.markdown("---")
    st.markdown("#### About the model")
    st.markdown(
        "**PlainConv U-Net** · 7-stage encoder  \n"
        "**54** anatomical structures  \n"
        "**256×256** input resolution  \n"
        "Trained on partially-annotated data  \n"
        "with marginal Dice + Focal loss"
    )

    st.markdown("---")
    st.markdown("#### Credits")
    st.caption(
        "Model trained during the [ENS × Raidium Data Challenge](https://challengedata.ens.fr/)  \n"
        "CT images from the [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) dataset  \n"
        "AI reports powered by [Mistral AI](https://mistral.ai/)"
    )
    st.caption("Built for the Alan × Mistral AI Health Hack 2025")


# ── Header banner ────────────────────────────────────────────────────────────

st.markdown("""
<div class="header-banner">
    <div class="header-badge">🩻 AI PRE-SCREENING TOOL</div>
    <h1>ScanAssist</h1>
    <p>
        Upload an abdominal CT slice and get instant organ segmentation
        with an AI-generated summary — helping clinicians triage faster
        and patients understand their scans.
    </p>
    <div class="pipeline">
        <div class="pipeline-step">📤 CT Slice</div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-step">🧠 U-Net Segmentation</div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-step">📊 Structure Analysis</div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-step">💬 Mistral Report</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── The Challenge (collapsible) ──────────────────────────────────────────────

with st.expander("🧪 The Challenge — what makes this problem interesting"):
    st.markdown("""
**This model was built during the [ENS × Raidium Data Challenge](https://challengedata.ens.fr/)**, 
a national data science competition on medical image segmentation.

The task: segment anatomical structures from abdominal CT scans. But the dataset 
comes with three compounding constraints that make standard approaches fail:

**1. Partial annotations** — Each training image is only labeled for *some* organs. 
The liver might be annotated on scan A but unlabeled on scan B — even though it's 
clearly visible. A naive model interprets the missing label as "this is background," 
learning the wrong thing. We solved this with a **marginal loss** that masks out 
unannotated classes, so the model only learns from what's actually labeled.

**2. Severe class imbalance** — 54 structures ranging from the liver (thousands 
of pixels) to tiny vessels (a few dozen pixels). Rare structures get crushed by 
dominant ones during training. We used **inverse-square-root frequency sampling** 
to upweight images containing rare organs.

**3. Limited labeled data** — Only 800 annotated images (partially), plus 1200 
raw unlabeled scans. A standard supervised setup would drastically overfit.

CT images are sourced from the [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) 
dataset (Wasserthal et al., 2023).
    """)

# ── How it works (collapsible) ───────────────────────────────────────────────

with st.expander("ℹ️ How it works — from pixels to report"):
    st.markdown("""
**Step 1 — Segmentation:** A 7-stage PlainConv U-Net with deep supervision 
processes the CT slice and outputs per-pixel probabilities for 54 anatomical 
classes. Only structures above a confidence threshold (0.5) are retained.

**Step 2 — Analysis:** The segmentation mask is analyzed to extract structure 
count, pixel areas, relative proportions, and coverage statistics.

**Step 3 — AI Report (Mistral):** The structured statistics are sent to 
Mistral AI, which generates a plain-language pre-screening summary. The report 
highlights dominant structures, flags anomalies, and suggests next steps — 
bridging the gap between raw AI output and clinical understanding.

**Clinical use case:** Pre-screening tool for high-volume imaging centers. 
A radiologist receives a first-pass analysis before reviewing the scan, 
saving time on routine cases and focusing attention on flagged anomalies. 
Patients get a readable summary they can actually understand.
    """)

# ── Input: examples or upload ────────────────────────────────────────────────

tab_examples, tab_upload = st.tabs(["📂 Example slices", "⬆️ Upload your own"])

selected_image = None
source_name = None

with tab_examples:
    example_files = sorted(EXAMPLES_DIR.glob(
        "*.npy")) if EXAMPLES_DIR.exists() else []
    if not example_files:
        st.info(
            "No example files in `examples/`. Add .npy slices (256×256 uint8) to get started.")
    else:
        cols = st.columns(min(len(example_files), 5))
        for i, fpath in enumerate(example_files):
            img = np.load(fpath)
            with cols[i % len(cols)]:
                st.image(img, caption=fpath.stem, clamp=True, width=120)
                if st.button("Analyze", key=f"ex_{i}"):
                    selected_image = img
                    source_name = fpath.stem

with tab_upload:
    uploaded = st.file_uploader(
        "Upload a CT slice (.png or .npy, 256×256 grayscale)", type=["npy", "png"]
    )
    if uploaded is not None:
        if uploaded.name.endswith(".npy"):
            selected_image = np.load(uploaded)
        else:
            # PNG → grayscale → clip to training range
            pil_img = Image.open(uploaded).convert("L")
            selected_image = np.array(pil_img).astype(np.uint8)
        source_name = uploaded.name

# ── Segmentation ─────────────────────────────────────────────────────────────

if selected_image is not None:
    if selected_image.shape != (256, 256):
        st.error(f"Expected shape (256, 256), got {selected_image.shape}.")
        st.stop()

    with st.spinner("Running segmentation model…"):
        model = get_model()
        mask = segment_slice(model, selected_image)

    stats = compute_structure_stats(mask)
    n_structures = len(stats)
    total_fg = sum(s["pixels"] for s in stats)
    coverage = round(100 * total_fg / (256 * 256), 1)
    largest = max(stats, key=lambda s: s["pixels"]) if stats else None

    # ── Metrics row ──────────────────────────────────────────────────────────

    if stats:
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-card">
                <div class="value">{n_structures}</div>
                <div class="label">Structures detected</div>
            </div>
            <div class="metric-card">
                <div class="value">{coverage}%</div>
                <div class="label">Slice coverage</div>
            </div>
            <div class="metric-card">
                <div class="value">{total_fg:,} px</div>
                <div class="label">Segmented area</div>
            </div>
            <div class="metric-card">
                <div class="value">#{largest['label']:02d}</div>
                <div class="label">Largest structure ({largest['pct_of_foreground']}%)</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Visual comparison ────────────────────────────────────────────────────

    col_orig, col_seg = st.columns(2)

    with col_orig:
        st.markdown("**Original CT slice**")
        fig_orig, ax_orig = plt.subplots(figsize=(5, 5), dpi=100)
        ax_orig.imshow(selected_image, cmap="gray")
        ax_orig.axis("off")
        fig_orig.tight_layout(pad=0)
        st.pyplot(fig_orig)

    with col_seg:
        st.markdown("**Segmentation overlay**")
        fig_seg = render_overlay(selected_image, mask, alpha)
        st.pyplot(fig_seg)

    # ── Structure table ──────────────────────────────────────────────────────

    with st.expander(f"📋 All {n_structures} structures"):
        col_a, col_b = st.columns(2)
        half = len(stats) // 2 + 1
        with col_a:
            for s in stats[:half]:
                st.write(
                    f"**#{s['label']:02d}** — {s['pixels']:,} px ({s['pct_of_foreground']}%)")
        with col_b:
            for s in stats[half:]:
                st.write(
                    f"**#{s['label']:02d}** — {s['pixels']:,} px ({s['pct_of_foreground']}%)")

    # ── Mistral AI report ────────────────────────────────────────────────────

    st.markdown("---")
    st.markdown("### 💬 AI-Generated Radiology Summary")
    st.caption(
        "Powered by Mistral AI — translates segmentation data into a readable pre-screening report.")

    if mistral_key:
        with st.spinner("Mistral is analyzing the segmentation…"):
            report = generate_report(stats, mistral_key)
        st.markdown(
            f'<div class="report-box">{report}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="example-report">
            <div class="example-label">EXAMPLE OUTPUT — enter a Mistral API key for live generation</div>
            {EXAMPLE_REPORT.replace(chr(10), '<br>')}
        </div>
        """, unsafe_allow_html=True)

    # ── Disclaimer ───────────────────────────────────────────────────────────

    st.markdown("""
    <div class="disclaimer">
        ⚕️ <strong>Disclaimer:</strong> ScanAssist is an AI-powered pre-screening tool
        designed to assist — not replace — qualified medical professionals. Outputs
        are not diagnostic and should always be reviewed by a radiologist.
    </div>
    """, unsafe_allow_html=True)

    # ── What's next ──────────────────────────────────────────────────────────

    with st.expander("🚀 What's next — scaling ScanAssist"):
        st.markdown("""
**This demo is a proof of concept.** Here's how it could become a real clinical tool:

**Semantic labels.** The current model outputs structure IDs (1–54) without 
organ names. Training on a larger, fully-annotated dataset (e.g. the full 
TotalSegmentator with 104 labeled classes) would give the model semantic 
understanding — "this is the liver," not just "this is structure #9."

**Smarter reports via RAG.** Instead of generating reports from pixel counts 
alone, the Mistral integration could query a **medical knowledge base** 
(clinical guidelines, anatomy references, pathology databases) through 
retrieval-augmented generation. This would produce reports grounded in 
medical literature — e.g. flagging that a structure's size deviates from 
population norms, or suggesting differential diagnoses based on location 
and morphology.

**3D volume analysis.** Currently operating on single 2D slices. Extending 
to full 3D CT volumes would enable volumetric organ measurements, which are 
far more clinically relevant than 2D pixel counts.

**Integration with health platforms.** A tool like this could feed into 
platforms like Alan — connecting automated scan analysis with the patient's 
health record, prevention recommendations, and care coordination.
        """)

else:
    # ── Empty state ──────────────────────────────────────────────────────────

    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            "<div style='text-align:center; padding: 3rem 0; color: #9ca3af;'>"
            "<p style='font-size: 2.5rem; margin-bottom: 0.5rem;'>🩻</p>"
            "<p style='font-size: 1.1rem; font-weight: 500;'>Select an example or upload a CT slice to start</p>"
            "<p style='font-size: 0.85rem;'>The model analyzes abdominal CT images in under 1 second.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
