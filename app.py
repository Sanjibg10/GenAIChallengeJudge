"""
Prompt Charade
A local Python-based application that ranks team members' AI-generated images
against a "Target Image" using CLIP-based visual similarity and prompt analysis.
"""

import os
import re
import base64
from io import BytesIO
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity


def pil_to_base64_thumbnail(image: Image.Image, size: tuple[int, int] = (80, 80)) -> str:
    """Convert a PIL image to a base64-encoded thumbnail string for embedding in HTML."""
    thumb = image.copy()
    thumb.thumbnail(size, Image.Resampling.LANCZOS)
    buffer = BytesIO()
    thumb.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ──────────────────────────────────────────────
# Page Configuration & Custom CSS
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Prompt Charade",
    page_icon="🎭",
    layout="wide",
)

CUSTOM_CSS = """
<style>
    /* Main background - warm gradient */
    .stApp {
        background: linear-gradient(160deg, #FFF6F0 0%, #F0F0FF 40%, #E8FFF5 70%, #FFF8E8 100%);
    }

    /* Header styling */
    h1 {
        background: linear-gradient(135deg, #FF6B6B, #845EC2, #00C9A7) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        text-align: center;
        font-weight: 900 !important;
        font-size: 2.8rem !important;
        letter-spacing: -0.5px;
    }
    h2 {
        color: #845EC2 !important;
        font-weight: 700 !important;
    }
    h3 {
        color: #00C9A7 !important;
        font-weight: 600 !important;
    }

    /* Sidebar - vibrant gradient */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #FFEEF8 0%, #F0E6FF 50%, #E6FFF8 100%);
        border-right: 3px solid #845EC2;
    }
    [data-testid="stSidebar"] h2 {
        color: #D65DB1 !important;
    }
    [data-testid="stSidebar"] h3 {
        color: #845EC2 !important;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        border: 2px dashed #D65DB1;
        border-radius: 16px;
        padding: 14px;
        background: rgba(255, 255, 255, 0.7);
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #FFFFFF, #FFF6F0);
        border: 2px solid transparent;
        border-image: linear-gradient(135deg, #FF6B6B, #845EC2, #00C9A7) 1;
        border-radius: 16px;
        padding: 18px;
        box-shadow: 0 4px 15px rgba(132, 94, 194, 0.12);
    }
    [data-testid="stMetric"] label {
        color: #845EC2 !important;
        font-weight: 600 !important;
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #FF6B6B !important;
        font-weight: 800 !important;
    }

    /* Table styling */
    table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(132, 94, 194, 0.15);
    }
    thead tr {
        background: linear-gradient(135deg, #FF6B6B, #D65DB1, #845EC2, #00C9A7) !important;
    }
    thead th {
        color: #FFFFFF !important;
        padding: 16px 18px !important;
        font-weight: 700 !important;
        font-size: 15px !important;
        text-align: left !important;
        text-shadow: 0 1px 2px rgba(0,0,0,0.15);
    }
    tbody tr {
        background-color: #FFFFFF;
        transition: all 0.2s ease;
    }
    tbody tr:nth-child(even) {
        background: linear-gradient(90deg, #FFF6F0, #F9F0FF);
    }
    tbody tr:hover {
        background: linear-gradient(90deg, #FFE8E8, #F0E0FF, #E0FFF5);
        transform: scale(1.005);
    }
    tbody td {
        padding: 14px 18px !important;
        font-size: 14px !important;
        border-bottom: 1px solid #F0E6FF !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #FF6B6B, #D65DB1, #845EC2) !important;
        color: white !important;
        border: none;
        border-radius: 12px;
        font-weight: 700;
        padding: 0.6rem 2rem;
        font-size: 16px;
        box-shadow: 0 4px 15px rgba(214, 93, 177, 0.3);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #FF5252, #C44DB1, #7040B0) !important;
        color: white !important;
        box-shadow: 0 6px 20px rgba(214, 93, 177, 0.4);
        transform: translateY(-1px);
    }

    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #FFF0F5, #F0E6FF, #E6FFF5);
        border-radius: 16px;
        padding: 22px;
        margin: 14px 0;
        border-left: 5px solid;
        border-image: linear-gradient(180deg, #FF6B6B, #845EC2, #00C9A7) 1;
        box-shadow: 0 3px 12px rgba(132, 94, 194, 0.08);
    }

    /* Expander */
    .streamlit-expanderHeader {
        font-weight: 600 !important;
        font-size: 16px !important;
    }

    /* Divider */
    hr {
        border: none;
        height: 3px;
        background: linear-gradient(90deg, #FF6B6B, #D65DB1, #845EC2, #00C9A7, transparent);
        border-radius: 2px;
    }

    /* Fun sparkle animation on header */
    @keyframes shimmer {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    h1 {
        background-size: 200% auto !important;
        animation: shimmer 4s ease infinite !important;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Animated Loading / Splash Screen
# ──────────────────────────────────────────────
LOADING_SCREEN = """
<style>
    /* Loading overlay – covers the page until Streamlit renders content */
    #splash-overlay {
        position: fixed;
        top: 0; left: 0;
        width: 100vw; height: 100vh;
        z-index: 99999;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, #FFF6F0 0%, #F0E6FF 40%, #E6FFF8 70%, #FFF8E8 100%);
        animation: fadeOut 0.6s ease 2.5s forwards;
        pointer-events: none;
    }

    /* Bouncing emoji characters */
    .splash-emojis {
        font-size: 60px;
        display: flex;
        gap: 18px;
        margin-bottom: 30px;
    }
    .splash-emojis span {
        display: inline-block;
        animation: bounce 1.4s ease infinite;
    }
    .splash-emojis span:nth-child(1) { animation-delay: 0s; }
    .splash-emojis span:nth-child(2) { animation-delay: 0.15s; }
    .splash-emojis span:nth-child(3) { animation-delay: 0.3s; }
    .splash-emojis span:nth-child(4) { animation-delay: 0.45s; }
    .splash-emojis span:nth-child(5) { animation-delay: 0.6s; }

    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-25px); }
    }

    /* App title on splash */
    .splash-title {
        font-size: 42px;
        font-weight: 900;
        background: linear-gradient(135deg, #FF6B6B, #D65DB1, #845EC2, #00C9A7);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: shimmer 3s ease infinite;
        margin-bottom: 16px;
    }

    /* Fun rotating messages */
    .splash-msg {
        font-size: 20px;
        font-weight: 600;
        color: #845EC2;
        animation: msgCycle 6s ease infinite;
    }
    @keyframes msgCycle {
        0%, 30%   { content: ""; opacity: 1; }
        33%, 63%  { opacity: 0; }
        66%, 96%  { opacity: 1; }
        100%      { opacity: 1; }
    }

    /* Colorful progress dots */
    .splash-dots {
        display: flex;
        gap: 10px;
        margin-top: 24px;
    }
    .splash-dots span {
        width: 14px; height: 14px;
        border-radius: 50%;
        animation: dotPulse 1.4s ease infinite;
    }
    .splash-dots span:nth-child(1) { background: #FF6B6B; animation-delay: 0s; }
    .splash-dots span:nth-child(2) { background: #D65DB1; animation-delay: 0.2s; }
    .splash-dots span:nth-child(3) { background: #845EC2; animation-delay: 0.4s; }
    .splash-dots span:nth-child(4) { background: #00C9A7; animation-delay: 0.6s; }
    .splash-dots span:nth-child(5) { background: #FF6B6B; animation-delay: 0.8s; }

    @keyframes dotPulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.6); opacity: 1; }
    }

    /* Fade out the overlay */
    @keyframes fadeOut {
        to { opacity: 0; visibility: hidden; }
    }
</style>

<div id="splash-overlay">
    <div class="splash-emojis">
        <span>🎭</span><span>🎨</span><span>🖼️</span><span>🏆</span><span>🚀</span>
    </div>
    <div class="splash-title">Prompt Charade</div>
    <div class="splash-msg" id="splash-msg-text">... warming up the magic ...</div>
    <div class="splash-dots">
        <span></span><span></span><span></span><span></span><span></span>
    </div>
</div>

<script>
    // Cycle through fun messages
    const msgs = [
        "... warming up the magic ...",
        "... getting the pixels ready ...",
        "... summoning the AI judges ...",
        "... polishing the leaderboard ...",
        "... almost showtime ...",
    ];
    let idx = 0;
    const el = document.getElementById("splash-msg-text");
    if (el) {
        setInterval(() => {
            idx = (idx + 1) % msgs.length;
            el.style.opacity = 0;
            setTimeout(() => {
                el.textContent = msgs[idx];
                el.style.opacity = 1;
            }, 300);
        }, 2000);
    }
</script>
"""
st.markdown(LOADING_SCREEN, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# CLIP Model Loading (cached)
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading CLIP model (first run may take a minute)...")
def load_clip_model():
    """Load CLIP model and processor locally."""
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()
    return model, processor


def get_image_embedding(model, processor, image: Image.Image) -> np.ndarray:
    """Compute the CLIP image embedding for a PIL image."""
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        output = model.get_image_features(**inputs)
    # Handle both raw tensor and BaseModelOutputWithPooling
    if hasattr(output, "image_embeds"):
        embedding = output.image_embeds
    elif hasattr(output, "pooler_output"):
        embedding = output.pooler_output
    elif isinstance(output, torch.Tensor):
        embedding = output
    else:
        # Fallback: try indexing as if it's a tuple/list
        embedding = output[0] if not isinstance(output, torch.Tensor) else output
    embedding = embedding / embedding.norm(p=2, dim=-1, keepdim=True)
    return embedding.cpu().numpy()


def compute_similarity(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings."""
    return float(cosine_similarity(emb_a, emb_b)[0][0])


def get_text_embedding(model, processor, text: str) -> np.ndarray:
    """Compute the CLIP text embedding for a given text string. Returns shape (1, D)."""
    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        output = model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
    # Handle BaseModelOutputWithPooling vs raw tensor
    if hasattr(output, "pooler_output"):
        embedding = output.pooler_output  # shape (1, 512)
    elif isinstance(output, torch.Tensor):
        embedding = output
    else:
        embedding = output.pooler_output if hasattr(output, "pooler_output") else output[0]
    # Ensure 2D shape (1, D)
    if embedding.dim() == 1:
        embedding = embedding.unsqueeze(0)
    elif embedding.dim() == 3:
        # last_hidden_state shape (1, seq_len, D) – take CLS token
        embedding = embedding[:, 0, :]
    embedding = embedding / embedding.norm(p=2, dim=-1, keepdim=True)
    return embedding.cpu().numpy()


# Visual aspect categories for CLIP-based element comparison
VISUAL_ASPECTS = {
    "Color Tone": [
        "warm colors like red, orange, and yellow tones",
        "cool colors like blue, green, and purple tones",
        "neutral or muted earth tones",
        "vibrant and highly saturated colors",
    ],
    "Brightness": [
        "bright and well-lit scene",
        "dark and moody low-light scene",
        "soft diffused lighting",
    ],
    "Subject": [
        "a person or human portrait",
        "an animal or creature",
        "a landscape or natural scenery",
        "an everyday object or still life",
        "an abstract or geometric pattern",
        "architecture or buildings",
    ],
    "Setting": [
        "outdoor natural environment with trees or sky",
        "indoor room or interior space",
        "urban cityscape with buildings and streets",
        "fantasy or surreal dreamlike setting",
        "plain or simple background",
    ],
    "Style": [
        "photorealistic photograph",
        "cartoon or illustration",
        "oil painting or watercolor artwork",
        "digital art or 3D render",
        "minimalist or flat design",
    ],
    "Composition": [
        "close-up detailed view",
        "wide panoramic view",
        "centered symmetrical layout",
        "dynamic angled or action composition",
    ],
    "Mood": [
        "cheerful and upbeat atmosphere",
        "calm and peaceful serene atmosphere",
        "dramatic and intense atmosphere",
        "mysterious and dark atmosphere",
    ],
    "Detail Level": [
        "highly detailed with intricate textures",
        "smooth and clean with minimal details",
    ],
}


def compute_aspect_text_embeddings(model, processor) -> dict:
    """Pre-compute CLIP text embeddings for all visual aspect descriptions."""
    aspect_embeddings = {}
    for aspect_name, descriptions in VISUAL_ASPECTS.items():
        text_embs = []
        for desc in descriptions:
            t_emb = get_text_embedding(model, processor, f"a photo with {desc}")
            text_embs.append(t_emb)
        aspect_embeddings[aspect_name] = {
            "descriptions": descriptions,
            "embeddings": np.vstack(text_embs),
        }
    return aspect_embeddings


def analyze_visual_elements(
    target_emb: np.ndarray,
    sub_emb: np.ndarray,
    aspect_embeddings: dict,
) -> dict:
    """
    Compare target and submission images across visual aspects using CLIP.
    Returns a dict with 'matching' and 'non_matching' lists of findings.
    """
    matching = []
    non_matching = []

    for aspect_name, data in aspect_embeddings.items():
        descriptions = data["descriptions"]
        text_embs = data["embeddings"]

        target_sims = cosine_similarity(target_emb, text_embs)[0]
        sub_sims = cosine_similarity(sub_emb, text_embs)[0]

        target_best_idx = int(np.argmax(target_sims))
        sub_best_idx = int(np.argmax(sub_sims))

        target_desc = descriptions[target_best_idx]
        sub_desc = descriptions[sub_best_idx]

        if target_best_idx == sub_best_idx:
            matching.append(f"**{aspect_name}** \u2014 Both share _{target_desc}_")
        else:
            non_matching.append(
                f"**{aspect_name}** \u2014 Target has _{target_desc}_, "
                f"submission has _{sub_desc}_"
            )

    return {"matching": matching, "non_matching": non_matching}


# ──────────────────────────────────────────────
# Directory Crawler
# ──────────────────────────────────────────────
def crawl_submissions(folder_path: str) -> list[dict]:
    """
    Walk the directory tree and pair images with their prompt files.
    Expected structure:
        [Root Folder]/[Member Name]/[generated_image.png]
        [Root Folder]/[Member Name]/[prompt.txt]
    """
    submissions = []
    root = Path(folder_path)

    if not root.exists() or not root.is_dir():
        return submissions

    for member_dir in sorted(root.iterdir()):
        if not member_dir.is_dir():
            continue

        member_name = member_dir.name
        image_file = None
        prompt_file = None
        prompt_text = ""

        for f in member_dir.iterdir():
            lower_name = f.name.lower()
            if f.is_file():
                # Find image files
                if lower_name.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")):
                    image_file = f
                # Find prompt file
                if lower_name.endswith(".txt"):
                    prompt_file = f

        if prompt_file and prompt_file.exists():
            try:
                prompt_text = prompt_file.read_text(encoding="utf-8").strip()
            except Exception:
                prompt_text = ""

        if image_file:
            submissions.append({
                "member": member_name,
                "image_path": str(image_file),
                "prompt_text": prompt_text,
                "has_prompt": bool(prompt_text),
            })

    return submissions


# ──────────────────────────────────────────────
# Prompt Analysis Heuristic
# ──────────────────────────────────────────────
AI_KEYWORD_PATTERNS = [
    r"\b8k\b", r"\b4k\b", r"\bultra\s*hd\b",
    r"\bmasterpiece\b", r"\bhighly\s+detailed\b", r"\bultra\s+detailed\b",
    r"\bphotorealistic\b", r"\bhyper\s*realistic\b",
    r"\bunreal\s+engine\b", r"\boctane\s+render\b", r"\bray\s+tracing\b",
    r"\bartstation\b", r"\btrending\s+on\b",
    r"\bcinematic\s+lighting\b", r"\bvolumetric\s+lighting\b",
    r"\bbokeh\b", r"\bdepth\s+of\s+field\b",
    r"\bsharp\s+focus\b", r"\bsuperb\s+detail\b",
    r"\baward[\s-]*winning\b", r"\bstudio\s+quality\b",
    r"\bhdr\b", r"\bdetailed\s+texture\b",
    r"\b(greg|artgerm|wlop|beeple)\b",
]


def analyze_prompt(prompt_text: str) -> dict:
    """
    Analyze a prompt to determine if it is 'natural language' or 'AI-optimized'.
    Returns a dict with:
        - style: "natural" | "ai_optimized" | "mixed"
        - naturalness_score: float 0-1 (1 = very natural)
        - keyword_count: int
        - avg_sentence_length: float
        - details: str
    """
    if not prompt_text:
        return {
            "style": "unknown",
            "naturalness_score": 0.5,
            "keyword_count": 0,
            "avg_sentence_length": 0,
            "details": "No prompt provided",
        }

    text_lower = prompt_text.lower()

    # Count AI keywords
    keyword_count = 0
    found_keywords = []
    for pattern in AI_KEYWORD_PATTERNS:
        matches = re.findall(pattern, text_lower)
        if matches:
            keyword_count += len(matches)
            found_keywords.append(pattern.replace(r"\b", "").replace("\\s+", " ").replace("\\s*", ""))

    # Analyze sentence structure
    sentences = re.split(r'[.!?]+', prompt_text)
    sentences = [s.strip() for s in sentences if s.strip()]
    num_sentences = max(len(sentences), 1)

    words = prompt_text.split()
    word_count = len(words)
    avg_sentence_length = word_count / num_sentences

    # Count comma-separated tokens (AI prompts tend to be comma-heavy)
    comma_segments = [seg.strip() for seg in prompt_text.split(",") if seg.strip()]
    comma_ratio = len(comma_segments) / max(word_count, 1)

    # Heuristic scoring (0 = very AI-optimized, 1 = very natural)
    score = 1.0

    # Penalize for AI keywords
    keyword_penalty = min(keyword_count * 0.12, 0.5)
    score -= keyword_penalty

    # Penalize for high comma ratio (keyword-stuffed prompts)
    if comma_ratio > 0.25:
        score -= min((comma_ratio - 0.25) * 1.5, 0.3)

    # Reward longer, well-formed sentences
    if avg_sentence_length > 8:
        score += 0.1
    elif avg_sentence_length < 4:
        score -= 0.15

    # Penalize very short prompts that are just keywords
    if word_count < 5:
        score -= 0.15

    # Reward prompts with natural language connectors
    natural_connectors = ["with", "and", "the", "in", "of", "that", "which", "while", "showing"]
    connector_count = sum(1 for w in words if w.lower() in natural_connectors)
    connector_ratio = connector_count / max(word_count, 1)
    if connector_ratio > 0.1:
        score += 0.1

    score = max(0.0, min(1.0, score))

    # Classify style
    if score >= 0.7:
        style = "natural"
    elif score <= 0.35:
        style = "ai_optimized"
    else:
        style = "mixed"

    # Build details string
    if keyword_count > 0:
        kw_note = f"Found {keyword_count} AI-style keyword(s)"
    else:
        kw_note = "No AI-style keywords detected"

    details = f"{kw_note}. Avg sentence length: {avg_sentence_length:.1f} words. Style: {style}."

    return {
        "style": style,
        "naturalness_score": round(score, 3),
        "keyword_count": keyword_count,
        "avg_sentence_length": round(avg_sentence_length, 1),
        "details": details,
    }


# ──────────────────────────────────────────────
# Scoring & Ranking
# ──────────────────────────────────────────────
def compute_final_scores(submissions: list[dict]) -> list[dict]:
    """
    Compute final score for each submission.
    Visual similarity is the primary factor (90%).
    Prompt naturalness is the tie-breaker (10%).
    """
    for sub in submissions:
        vis = sub.get("visual_similarity", 0.0)
        nat = sub.get("prompt_analysis", {}).get("naturalness_score", 0.5)

        # Primary: visual similarity (90%), Tie-breaker: prompt naturalness (10%)
        final = vis * 0.90 + nat * 0.10
        sub["final_score"] = round(final, 4)

    # Sort by final score descending
    submissions.sort(key=lambda x: x["final_score"], reverse=True)

    # Assign ranks
    for i, sub in enumerate(submissions):
        sub["rank"] = i + 1

    return submissions


def generate_evaluation_note(sub: dict) -> str:
    """Generate a human-readable evaluation note for each submission."""
    vis = sub.get("visual_similarity", 0.0)
    prompt_info = sub.get("prompt_analysis", {})
    style = prompt_info.get("style", "unknown")
    nat_score = prompt_info.get("naturalness_score", 0.5)

    parts = []

    # Visual similarity commentary
    if vis >= 0.90:
        parts.append("Exceptional visual match")
    elif vis >= 0.80:
        parts.append("Strong visual similarity")
    elif vis >= 0.70:
        parts.append("Good visual resemblance")
    elif vis >= 0.55:
        parts.append("Moderate similarity - some elements match")
    else:
        parts.append("Low similarity - significant differences from target")

    # Prompt commentary
    if not sub.get("has_prompt"):
        parts.append("no prompt file provided")
    elif style == "natural":
        parts.append("well-crafted natural prompt")
    elif style == "ai_optimized":
        parts.append("heavily keyword-optimized prompt")
    elif style == "mixed":
        parts.append("prompt mixes natural language with AI keywords")

    # Tie-breaker note if applicable
    if vis >= 0.70 and style == "natural":
        parts.append("bonus for creative prompting")

    return "; ".join(parts) + "."


# ──────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────
def main():
    # Header
    st.markdown("# 🎭 Prompt Charade")
    st.markdown(
        '<div class="info-box">'
        "<b>Welcome to Prompt Charade!</b> 🎉 Upload your target image and browse to the submissions folder. "
        "We'll use a local CLIP model to evaluate visual similarity and analyze prompts as a tie-breaker. "
        "Let the games begin! 🚀"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # Sidebar inputs
    with st.sidebar:
        st.markdown("## 🎨 Inputs")

        st.markdown("### 🖼️ Target Image")
        target_file = st.file_uploader(
            "Upload the reference image",
            type=["png", "jpg", "jpeg", "webp", "bmp"],
            help="This is the image all submissions will be compared against.",
        )

        st.markdown("### 📂 Submissions Folder")

        # Initialize session state for folder selection
        if "selected_folder" not in st.session_state:
            st.session_state["selected_folder"] = ""

        # Browse Folder button – opens native OS folder picker
        if st.button("📁 Browse Folder", key="browse_folder", width="stretch"):
            try:
                import tkinter as tk
                from tkinter import filedialog

                root = tk.Tk()
                root.withdraw()
                # Bring the dialog to the front on Windows
                root.wm_attributes("-topmost", 1)
                folder = filedialog.askdirectory(
                    master=root,
                    title="Select Submissions Folder",
                )
                root.destroy()
                if folder:
                    st.session_state["selected_folder"] = folder
                    st.rerun()
            except Exception:
                st.warning(
                    "Native folder dialog not available on this system. "
                    "Please type the path below instead."
                )

        # Show currently selected folder
        if st.session_state["selected_folder"]:
            st.success(f"📁 {st.session_state['selected_folder']}")

        # Fallback: manual path entry
        folder_input = st.text_input(
            "Or type folder path manually",
            placeholder="/path/to/submissions",
            key="folder_manual_input",
        )
        if folder_input:
            st.session_state["selected_folder"] = folder_input

        submissions_path = st.session_state["selected_folder"]

        st.markdown("---")
        run_button = st.button("🚀 Run Evaluation", width="stretch")

    # Main area
    if target_file:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("### 🎯 Target Image")
            target_image = Image.open(target_file).convert("RGB")
            st.image(target_image, width="stretch")

        if submissions_path:
            with col2:
                submissions = crawl_submissions(submissions_path)
                if submissions:
                    st.markdown(f"### 📂 Found **{len(submissions)}** submission(s)")
                    member_names = [s["member"] for s in submissions]
                    st.markdown(", ".join(f"`{m}`" for m in member_names))
                else:
                    st.warning("No valid submissions found. Check the folder structure.")
    else:
        st.info("👈 Upload a **Target Image** and select a **Submissions Folder** in the sidebar to get started!")

    # Run evaluation
    if run_button:
        if not target_file:
            st.error("Please upload a target image first.")
            return
        if not submissions_path:
            st.error("Please enter a submissions folder path.")
            return

        submissions = crawl_submissions(submissions_path)
        if not submissions:
            st.error("No valid submissions found in the specified folder.")
            return

        target_image = Image.open(target_file).convert("RGB")

        st.markdown("---")
        st.markdown("## ⚙️ Evaluation in Progress...")

        # Load CLIP model
        model, processor = load_clip_model()

        # Compute target embedding
        progress_bar = st.progress(0, text="Computing target image embedding...")
        target_emb = get_image_embedding(model, processor, target_image)
        progress_bar.progress(5, text="Target embedding computed. Preparing visual analysis...")

        # Pre-compute text embeddings for visual element analysis
        aspect_embs = compute_aspect_text_embeddings(model, processor)
        progress_bar.progress(10, text="Visual analysis ready. Evaluating submissions...")

        # Process each submission
        total = len(submissions)
        for i, sub in enumerate(submissions):
            pct = int(10 + (i + 1) / total * 80)
            progress_bar.progress(pct, text=f"Evaluating {sub['member']} ({i+1}/{total})...")

            # Load submission image
            try:
                sub_image = Image.open(sub["image_path"]).convert("RGB")
                sub_emb = get_image_embedding(model, processor, sub_image)
                similarity = compute_similarity(target_emb, sub_emb)
                sub["visual_similarity"] = round(similarity, 4)
                sub["image"] = sub_image
                sub["embedding"] = sub_emb
            except Exception as e:
                sub["visual_similarity"] = 0.0
                sub["image"] = None
                sub["embedding"] = None
                sub["error"] = str(e)

            # Analyze prompt
            sub["prompt_analysis"] = analyze_prompt(sub["prompt_text"])

            # Analyze visual elements (key findings)
            if sub.get("embedding") is not None:
                sub["key_findings"] = analyze_visual_elements(
                    target_emb, sub["embedding"], aspect_embs
                )
            else:
                sub["key_findings"] = {"matching": [], "non_matching": []}

        # Compute final scores and rank
        progress_bar.progress(95, text="Computing final rankings...")
        submissions = compute_final_scores(submissions)

        # Generate evaluation notes
        for sub in submissions:
            sub["evaluation_note"] = generate_evaluation_note(sub)

        progress_bar.progress(100, text="Evaluation complete!")

        # ── Display Results ──
        st.markdown("---")
        st.markdown("## 🏆 Leaderboard")

        # Summary metrics
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Total Submissions", len(submissions))
        with col_b:
            avg_sim = np.mean([s["visual_similarity"] for s in submissions])
            st.metric("Avg Visual Similarity", f"{avg_sim:.1%}")
        with col_c:
            top_member = submissions[0]["member"] if submissions else "N/A"
            st.metric("Top Performer", top_member)

        st.markdown("")

        # Build leaderboard table with thumbnails
        rank_emojis = {1: "🥇", 2: "🥈", 3: "🥉"}

        html_rows = ""
        for sub in submissions:
            rank = sub["rank"]
            emoji = rank_emojis.get(rank, f"#{rank}")
            rank_display = f"{emoji}" if rank <= 3 else f"#{rank}"
            vis_pct = f"{sub['visual_similarity']:.1%}"
            nat_score = sub["prompt_analysis"]["naturalness_score"]
            prompt_score_pct = f"{nat_score:.0%}"
            style_label = sub["prompt_analysis"]["style"].replace("_", " ").title()
            if style_label == "Natural":
                badge_bg = "#00C9A7"
            elif style_label == "Ai Optimized":
                badge_bg = "#FF6B6B"
            else:
                badge_bg = "#FFB347"
            prompt_cell = (
                f'{prompt_score_pct}<br>'
                f'<span style="font-size:10px;background:{badge_bg};color:#fff;'
                f'padding:2px 8px;border-radius:10px;">{style_label}</span>'
            )
            final_pct = f"{sub['final_score']:.1%}"

            # Generate thumbnail
            if sub.get("image"):
                b64 = pil_to_base64_thumbnail(sub["image"], size=(80, 80))
                img_html = f'<img src="data:image/png;base64,{b64}" style="width:80px;height:80px;object-fit:cover;border-radius:8px;border:2px solid #D65DB1;box-shadow:0 2px 6px rgba(132,94,194,0.2);" />'
            else:
                img_html = '<span style="color:#999;">N/A</span>'

            html_rows += f"""
            <tr>
                <td style="text-align:center;font-size:20px;">{rank_display}</td>
                <td><strong>{sub['member']}</strong></td>
                <td style="text-align:center;">{img_html}</td>
                <td style="text-align:center;">{vis_pct}</td>
                <td style="text-align:center;">{prompt_cell}</td>
                <td style="text-align:center;font-weight:700;">{final_pct}</td>
                <td style="font-size:13px;">{sub['evaluation_note']}</td>
            </tr>"""

        # Calculate dynamic height based on number of rows (header ~50px + ~100px per row + padding)
        table_height = 60 + len(submissions) * 105

        leaderboard_html = f"""
        <html>
        <body style="margin:0;padding:0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:transparent;">
        <table style="width:100%;border-collapse:separate;border-spacing:0;border-radius:16px;overflow:hidden;box-shadow:0 4px 20px rgba(132,94,194,0.15);">
            <thead>
                <tr style="background:linear-gradient(135deg,#FF6B6B,#D65DB1,#845EC2,#00C9A7);">
                    <th style="color:#fff;padding:14px 12px;font-weight:700;text-align:center;">Rank</th>
                    <th style="color:#fff;padding:14px 12px;font-weight:700;">Team Member</th>
                    <th style="color:#fff;padding:14px 12px;font-weight:700;text-align:center;">Submission</th>
                    <th style="color:#fff;padding:14px 12px;font-weight:700;text-align:center;">Visual Similarity</th>
                    <th style="color:#fff;padding:14px 12px;font-weight:700;text-align:center;">Prompt Score</th>
                    <th style="color:#fff;padding:14px 12px;font-weight:700;text-align:center;">Final Score</th>
                    <th style="color:#fff;padding:14px 12px;font-weight:700;">Evaluation Note</th>
                </tr>
            </thead>
            <tbody>
                {html_rows}
            </tbody>
        </table>
        </body>
        </html>
        """
        components.html(leaderboard_html, height=table_height, scrolling=False)

        # ── Detailed Breakdown ──
        st.markdown("---")
        st.markdown("## 📋 Detailed Breakdown")

        for sub in submissions:
            rank = sub["rank"]
            emoji = rank_emojis.get(rank, "🔹")
            with st.expander(f"{emoji} {sub['member']} — Rank #{rank} ({sub['final_score']:.1%})", expanded=(rank <= 3)):
                det_col1, det_col2 = st.columns([1, 2])
                with det_col1:
                    if sub.get("image"):
                        st.image(sub["image"], caption=f"{sub['member']}'s submission", width="stretch")
                    else:
                        st.warning("Could not load image.")

                with det_col2:
                    st.markdown(f"**Visual Similarity:** `{sub['visual_similarity']:.1%}`")
                    st.markdown(f"**Final Score:** `{sub['final_score']:.1%}`")
                    st.markdown(f"**Evaluation:** {sub['evaluation_note']}")

                    if sub["has_prompt"]:
                        pa = sub["prompt_analysis"]
                        st.markdown(f"**Prompt Style:** {pa['style'].replace('_', ' ').title()}")
                        st.markdown(f"**Naturalness Score:** `{pa['naturalness_score']:.2f}`")
                        st.markdown(f"**AI Keywords Found:** {pa['keyword_count']}")
                        st.markdown("**Prompt Text:**")
                        st.code(sub["prompt_text"], language=None)
                    else:
                        st.info("No prompt.txt provided for this submission.")

                # Key Findings — Target vs. Submission
                if sub.get("key_findings"):
                    findings = sub["key_findings"]
                    st.markdown("---")
                    st.markdown("**🔍 Key Findings — Target vs. Submission:**")

                    kf_col1, kf_col2 = st.columns(2)
                    with kf_col1:
                        if findings["matching"]:
                            st.markdown("**✅ Matching Elements:**")
                            for finding in findings["matching"]:
                                st.markdown(f"- {finding}")
                        else:
                            st.info("No clearly matching visual elements detected.")
                    with kf_col2:
                        if findings["non_matching"]:
                            st.markdown("**❌ Non-Matching Elements:**")
                            for finding in findings["non_matching"]:
                                st.markdown(f"- {finding}")
                        else:
                            st.info("No clearly differing visual elements detected.")

        # Footer
        st.markdown("---")
        st.markdown(
            '<p style="text-align:center; color:#999; font-size:13px;">'
            "🎭 Prompt Charade | Powered by CLIP (local) | No paid APIs used | Let the fun continue! 🎉"
            "</p>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
