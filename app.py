"""
GenAI Image Challenge Judge
A local Python-based application that ranks team members' AI-generated images
against a "Target Image" using CLIP-based visual similarity and prompt analysis.
"""

import os
import re
from pathlib import Path

import streamlit as st
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity


# ──────────────────────────────────────────────
# Page Configuration & Custom CSS
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="GenAI Image Challenge Judge",
    page_icon="🎨",
    layout="wide",
)

CUSTOM_CSS = """
<style>
    /* Main background */
    .stApp {
        background-color: #F5F7FA;
    }

    /* Header styling */
    h1 {
        color: #6C3FC5 !important;
        text-align: center;
        font-weight: 800 !important;
    }
    h2, h3 {
        color: #2CB5A0 !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 3px solid #6C3FC5;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        border: 2px dashed #6C3FC5;
        border-radius: 12px;
        padding: 12px;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 2px 8px rgba(108, 63, 197, 0.08);
    }

    /* Table styling */
    table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    }
    thead tr {
        background: linear-gradient(135deg, #6C3FC5, #2CB5A0) !important;
    }
    thead th {
        color: #FFFFFF !important;
        padding: 14px 18px !important;
        font-weight: 700 !important;
        font-size: 15px !important;
        text-align: left !important;
    }
    tbody tr {
        background-color: #FFFFFF;
    }
    tbody tr:nth-child(even) {
        background-color: #F9F7FD;
    }
    tbody tr:hover {
        background-color: #EDE7F9;
    }
    tbody td {
        padding: 12px 18px !important;
        font-size: 14px !important;
        border-bottom: 1px solid #ECECEC !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6C3FC5, #2CB5A0);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 700;
        padding: 0.5rem 2rem;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #5A32A8, #23A08D);
        color: white;
    }

    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #EDE7F9, #D4F5EF);
        border-radius: 12px;
        padding: 20px;
        margin: 12px 0;
        border-left: 4px solid #6C3FC5;
    }

    /* Divider */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, #6C3FC5, #2CB5A0, transparent);
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


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
    st.markdown("# 🎨 GenAI Image Challenge Judge")
    st.markdown(
        '<div class="info-box">'
        "<b>Welcome!</b> Upload your target image and point to the submissions folder. "
        "This tool uses a local CLIP model to evaluate visual similarity and analyzes prompts as a tie-breaker."
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # Sidebar inputs
    with st.sidebar:
        st.markdown("## 🖼️ Inputs")

        st.markdown("### Target Image")
        target_file = st.file_uploader(
            "Upload the reference image",
            type=["png", "jpg", "jpeg", "webp", "bmp"],
            help="This is the image all submissions will be compared against.",
        )

        st.markdown("### Submissions Folder")
        submissions_path = st.text_input(
            "Enter the full path to the submissions directory",
            placeholder="/path/to/submissions",
            help="Each subfolder should be named after a team member and contain an image and optional prompt.txt.",
        )

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
        st.info("👈 Upload a **Target Image** in the sidebar to get started.")

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
        progress_bar.progress(10, text="Target embedding computed.")

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
            except Exception as e:
                sub["visual_similarity"] = 0.0
                sub["image"] = None
                sub["error"] = str(e)

            # Analyze prompt
            sub["prompt_analysis"] = analyze_prompt(sub["prompt_text"])

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

        # Build leaderboard table
        rank_emojis = {1: "🥇", 2: "🥈", 3: "🥉"}
        table_rows = []
        for sub in submissions:
            rank = sub["rank"]
            emoji = rank_emojis.get(rank, f"#{rank}")
            rank_display = f"{emoji}" if rank <= 3 else f"#{rank}"
            vis_pct = f"{sub['visual_similarity']:.1%}"
            prompt_style = sub["prompt_analysis"]["style"].replace("_", " ").title()
            final_pct = f"{sub['final_score']:.1%}"

            table_rows.append(
                f"| {rank_display} | **{sub['member']}** | {vis_pct} | {prompt_style} | {final_pct} | {sub['evaluation_note']} |"
            )

        table_md = (
            "| Rank | Team Member | Visual Similarity | Prompt Style | Final Score | Evaluation Note |\n"
            "|:----:|:------------|:-----------------:|:------------:|:-----------:|:----------------|\n"
            + "\n".join(table_rows)
        )
        st.markdown(table_md, unsafe_allow_html=True)

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

        # Footer
        st.markdown("---")
        st.markdown(
            '<p style="text-align:center; color:#999; font-size:13px;">'
            "GenAI Image Challenge Judge | Powered by CLIP (local) | No paid APIs used"
            "</p>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
