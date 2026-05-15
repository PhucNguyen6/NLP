"""Streamlit chat UI for sentiment comment analysis."""

from __future__ import annotations

import streamlit as st

from inference_pipeline import get_pipeline

st.set_page_config(page_title="Sentiment Chat Box", page_icon="💬", layout="centered")

ENCODER_LABELS = {
    "xlmroberta": "XLM-RoBERTa",
    "doc2vec": "Doc2Vec",
    "tfidf": "TF-IDF",
}


@st.cache_resource(show_spinner=False)
def load_pipeline(use_llm: bool, primary_encoder: str):
    """Load pipeline once per (LLM toggle, encoder) — encoder đổi sẽ tải instance mới."""
    return get_pipeline(use_llm=use_llm, auto_expand_dict=True, primary_encoder=primary_encoder)


def submit_comment(comment_text: str, pipeline, top_k: int) -> None:
    """Analyze one comment and append into chat history."""
    cleaned = (comment_text or "").strip()
    if not cleaned:
        st.warning("Vui lòng nhập bình luận trước khi gửi.")
        return

    st.session_state.messages.append({"role": "user", "content": cleaned})
    with st.spinner("Đang phân tích bình luận..."):
        result = pipeline.analyze_comment(
            cleaned,
            return_explanation=st.session_state.use_llm,
            top_k=top_k,
            language="auto",
        )
    st.session_state.messages.append({"role": "assistant", "result": result})


def get_label_color(label: str) -> str:
    """Map sentiment labels to UI colors."""
    normalized = (label or "").strip().lower()
    if normalized == "khen":
        return "green"
    if normalized in ("che", "chê"):
        return "red"
    if normalized in ("trung lap", "trung lập"):
        return "orange"
    return "gray"


def colorize_label(label: str) -> str:
    """Return HTML badge text for a label."""
    color = get_label_color(label)
    safe_label = label if label else "không xác định"
    return (
        f"<span style='padding:3px 10px; border-radius:12px; background:{color}; "
        f"color:white; font-weight:600;'>{safe_label}</span>"
    )


def render_result(result: dict) -> None:
    """Render model output in friendly format."""
    final_label = result.get("sentiment_module", {}).get("final_label", "không xác định")
    predictions = result.get("sentiment_module", {}).get("predictions", {})
    llm_explanation = result.get("llm_explanation")
    latency = result.get("metadata", {}).get("latency_ms", "-")
    primary = result.get("metadata", {}).get("primary_encoder", "—")

    st.markdown(
        f"**Nhận xét tổng quan:** {colorize_label(final_label)}",
        unsafe_allow_html=True,
    )
    st.caption(f"Mô hình chính: **{ENCODER_LABELS.get(primary, primary)}** | Độ trễ: {latency} ms")

    with st.expander("Chi tiết dự đoán theo từng mô hình", expanded=False):
        if not predictions:
            st.write("Không có dự đoán từ mô hình SVM.")
        for model_name, item in predictions.items():
            label = item.get("label", "không xác định")
            confidence = item.get("confidence")
            label_badge = colorize_label(label)
            if confidence is None:
                st.markdown(f"- {model_name}: {label_badge}", unsafe_allow_html=True)
            else:
                st.markdown(
                    f"- {model_name}: {label_badge} (độ tin cậy: {confidence:.2%})",
                    unsafe_allow_html=True,
                )

    if llm_explanation:
        st.markdown("**Giải thích bởi LLM:**")
        st.write(llm_explanation)


def main() -> None:
    st.title("💬 Chat Box Phân Tích Bình Luận")
    st.write("Nhập 1 bình luận, nhấn **Enter** hoặc bấm **Gửi** để xem kết quả.")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "use_llm" not in st.session_state:
        st.session_state.use_llm = True

    with st.sidebar:
        st.header("Tùy chọn")
        st.session_state.use_llm = st.toggle("Bật giải thích LLM", value=st.session_state.use_llm)
        primary_encoder = st.selectbox(
            "Mô hình chính (SVM + RAG)",
            options=["xlmroberta", "doc2vec", "tfidf"],
            format_func=lambda k: ENCODER_LABELS[k],
            key="primary_encoder_choice",
            help="Nhãn cuối và truy vấn RAG dùng đúng encoder này. Cần DB đã nạp embedding cùng loại (populate_embeddings).",
        )
        top_k = st.slider("Số tài liệu RAG (top_k)", min_value=1, max_value=10, value=5, step=1)
        if st.button("Xóa lịch sử chat", type="secondary"):
            st.session_state.messages = []
            st.rerun()

    pipeline = load_pipeline(st.session_state.use_llm, primary_encoder)

    for message in st.session_state.messages:
        if message.get("role") == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                render_result(message["result"])

    with st.form("chat_form", clear_on_submit=True):
        user_comment = st.text_input("Nhập bình luận", placeholder="VD: Bài viết này rất hữu ích...")
        submitted = st.form_submit_button("Gửi")

    if submitted:
        submit_comment(user_comment, pipeline, top_k)
        st.rerun()


if __name__ == "__main__":
    main()
