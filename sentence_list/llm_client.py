# -*- coding: utf-8 -*-
"""
LLM Client - LM Studio (OpenAI-compatible) for sentiment explanation
"""

import logging
import json
import os
import re
import unicodedata
from typing import Dict, Optional, List
import time

import requests

from config import LLM_CONFIG, LOGGING_CONFIG

# Setup logging
logging.basicConfig(level=getattr(logging, LOGGING_CONFIG['level']),
                   format=LOGGING_CONFIG['format'])
logger = logging.getLogger(__name__)


class LMStudioClient:
    """Client for LM Studio API (OpenAI-compatible endpoint)"""
    
    def __init__(self, url: str = None, model_name: str = None):
        """
        Initialize LM Studio client
        
        Args:
            url: LM Studio API endpoint URL
            model_name: Model name to use
        """
        self.url = url or LLM_CONFIG['lm_studio_url']
        self.model_name = model_name or LLM_CONFIG['model_name']
        self.temperature = LLM_CONFIG['temperature']
        self.max_tokens = LLM_CONFIG['max_tokens']
        self.timeout = LLM_CONFIG['timeout']
        
        logger.info(f"Initializing LM Studio client: {self.url}")
        self._verify_connection()
    
    def _verify_connection(self) -> bool:
        """
        Verify connection to LM Studio
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            response = requests.get(
                f"{self.url.replace('/v1', '')}/models",
                timeout=5
            )
            if response.status_code == 200:
                logger.info("Connected to LM Studio successfully")
                return True
        except Exception as e:
            logger.warning(f"Failed to verify LM Studio connection: {e}")
        
        return False
    
    def generate_sentiment_explanation(self, query: str, context: str = None,
                                      sentiment_label: str = None) -> str:
        """
        Generate sentiment explanation using LLM
        
        Args:
            query: Original query/comment
            context: Retrieved context from RAG
            sentiment_label: Predicted sentiment label
            
        Returns:
            LLM-generated explanation
        """
        
        prompt = self._build_sentiment_prompt(
            query, context, sentiment_label
        )
        
        try:
            response = self._call_llm(prompt)
            return self._postprocess_sentiment_explanation(
                response_text=response,
                fallback_label=sentiment_label,
                source_comment=query,
            )
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            return self._postprocess_sentiment_explanation(
                response_text="",
                fallback_label=sentiment_label,
                source_comment=query,
            )
    
    def _build_sentiment_prompt(self, query: str, context: str = None,
                               sentiment_label: str = None) -> str:
        """
        Build prompt for sentiment analysis
        
        Args:
            query: Original query
            context: Retrieved context
            sentiment_label: Predicted sentiment
            
        Returns:
            Formatted prompt
        """
        
        system_prompt = """Bạn là trợ lý phân tích cảm xúc tiếng Việt.
Hãy trả lời tự nhiên, rõ nghĩa, đúng ngữ cảnh đời thường.
Không dùng văn phong dịch máy, không pha tiếng Anh nếu không cần thiết.

Nhiệm vụ:
1. Xác định cảm xúc chính của bình luận.
2. Chỉ ra từ/cụm từ thể hiện cảm xúc nổi bật.
3. Giải thích ngắn gọn, dễ hiểu, bám sát nội dung bình luận.
4. Gợi ý phản hồi lịch sự và thực tế (nếu phù hợp)."""
        
        sentiment_instruction = ""
        if sentiment_label:
            sentiment_instruction = (
                f"\nNhãn cảm xúc dự đoán từ hệ thống: {sentiment_label}\n"
                "Hãy xem đây là gợi ý tham khảo và có thể điều chỉnh nếu thấy chưa hợp lý."
            )

        angle = self._sentiment_angle_instructions(sentiment_label)
        
        context_section = ""
        if context:
            context_section = f"\n\nNgữ cảnh tham khảo (RAG):\n{context}"
        
        prompt = f"""{system_prompt}{angle}

Bình luận cần phân tích:
{query}{sentiment_instruction}{context_section}

Yêu cầu định dạng trả lời (tiếng Việt có dấu):
1) Kết quả cảm xúc: <khen/che/trung lập hoặc nhãn phù hợp>
2) Từ khóa cảm xúc: <chỉ được liệt kê các cụm CHÉP NGUYÊN VĂN từ bình luận gốc; cách nhau bằng dấu phẩy — không được thêm từ có ý nghĩa tương tự nếu từ đó không xuất hiện trong bình luận>
3) Giải thích: <2-3 câu, rõ ràng, tránh sáo rỗng>
4) Gợi ý phản hồi: <1 câu ngắn, lịch sự, nếu cần>

Lưu ý:
- Không suy diễn quá mức ngoài nội dung bình luận.
- Mục “Từ khóa cảm xúc” chỉ là trích đoạn từ bình luận, không được dùng từ chỉ có trong ngữ cảnh RAG nếu từ đó không nằm trong bình luận gốc.
- Nếu bình luận trung tính, hãy nói rõ vì sao trung tính.
- Chỉ trả lời một lần theo đúng 4 mục 1)–4) ở trên; không thêm đoạn tóm tắt lặp lại (ví dụ "Câu trả lời cuối cùng").
- Không chèn thêm tiêu đề kiểu "Gợi ý phản hồi lịch sự" bên trong mục Giải thích — gợi ý chỉ nằm ở mục 4.

Trả lời:"""
        
        return prompt

    def _normalize_label(self, label: Optional[str]) -> str:
        """Normalize free-form labels to target Vietnamese labels."""
        if not label:
            return "trung lập"
        value = label.strip().lower()
        mapping = {
            "khen": "khen",
            "positive": "khen",
            "praise": "khen",
            "che": "chê",
            "chê": "chê",
            "negative": "chê",
            "criticism": "chê",
            "trung lap": "trung lập",
            "trung lập": "trung lập",
            "neutral": "trung lập",
        }
        return mapping.get(value, label.strip())

    def _sentiment_angle_instructions(self, sentiment_label: Optional[str]) -> str:
        """Hướng giải thích + gợi ý phản hồi bám theo nhãn cảm xúc (khen / chê / trung lập)."""
        v = (sentiment_label or "").strip().lower()
        if v in ("khen", "praise", "positive"):
            return (
                "\n**Góc trả lời (bắt buộc bám theo nhãn KHEN):** Coi bình luận là **tích cực / khen**. "
                "Giải thích thể hiện đồng cảm với điểm tốt người dùng nêu. "
                "Mục \"Gợi ý phản hồi\" hãy viết theo hướng **cảm ơn, khẳng định, khuyến khích** tiếp tục tương tác tích cực."
            )
        if v in ("che", "chê", "criticism", "negative"):
            return (
                "\n**Góc trả lời (bắt buộc bám theo nhãn CHÊ):** Coi bình luận là **phê / chê / tiêu cực**. "
                "Giải thích công tâm vì sao có sắc thái đó. "
                "Mục \"Gợi ý phản hồi\" hãy viết theo hướng **tiếp nhận phản hồi, lịch sự, hướng tới cải thiện hoặc làm rõ hiểu lầm** — không phủ nhận thô."
            )
        return (
            "\n**Góc trả lời (TRUNG LẬP):** Giữ giọng khách quan; "
            "gợi ý phản hồi mở, có thể hỏi thêm ngữ cảnh nếu cần."
        )

    def _normalize_for_contains(self, s: str) -> str:
        """Lowercase and strip combining marks for loose Vietnamese matching."""
        s = unicodedata.normalize("NFKD", s or "")
        return "".join(c for c in s.lower() if not unicodedata.combining(c))

    def _strip_keyword_wrappers(self, phrase: str) -> str:
        phrase = phrase.strip().strip("\"'”“")
        if phrase.startswith("<") and phrase.endswith(">"):
            phrase = phrase[1:-1].strip()
        return phrase.strip()

    def _ground_keywords_in_comment(self, comment: str, keywords_raw: str) -> str:
        """
        Keep only phrases that literally appear in the user's comment (no LLM synonyms).
        """
        if not (comment or "").strip():
            return ""
        blob = self._strip_keyword_wrappers(keywords_raw or "").strip("<> ")
        if not blob:
            return ""
        pieces = [p.strip() for p in re.split(r"[,;,，、]", blob) if p.strip()]
        seen = []
        comment_lower = comment.lower()
        cn = self._normalize_for_contains(comment)

        for piece in pieces:
            phrase = self._strip_keyword_wrappers(piece)
            if len(phrase) < 2:
                continue

            matched_span = None
            if phrase.lower() in comment_lower:
                try:
                    m = re.search(re.escape(phrase), comment, re.IGNORECASE)
                    matched_span = m.group(0) if m else phrase
                except re.error:
                    matched_span = phrase
            elif self._normalize_for_contains(phrase) in cn:
                matched_span = phrase

            if matched_span:
                normalized_key = matched_span.strip().lower()
                if normalized_key not in [x.strip().lower() for x in seen]:
                    seen.append(matched_span.strip())

        return ", ".join(seen)

    def _extract_section_value(
        self, text: str, section_names: List[str], extra_stop: str = ""
    ) -> str:
        """Extract value for section title variants from model output."""
        if not text:
            return ""
        stop_ahead = (
            r"(?=\n\d+\)|\n(?:Kết quả cảm xúc|Từ khóa cảm xúc|Giải thích|Gợi ý phản hồi)\s*:"
            r"|\n\s*\*{0,2}\s*Câu trả lời cuối"
            r"|\n\s*Câu trả lời cuối"
            r"|\n\s*Final answer\s*:"
            r"|\Z)"
        ) + (extra_stop or "")
        for name in section_names:
            pattern = rf"{name}\s*:\s*(.+?){stop_ahead}"
            matched = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
            if matched:
                return re.sub(r"\s+", " ", matched.group(1)).strip(" -\n\t")
        return ""

    def _strip_embedded_polite_suggestion(self, s: str) -> str:
        """Bo khoi 'Goi y phan hoi lich su' model nhet vao muc giai thich (sau khi thanh 1 dong)."""
        if not s:
            return s
        patterns = [
            r"\s+\*{0,2}\s*Gợi ý phản hồi lịch sự\s*[:：]",
            r"\s+Gợi ý phản hồi lịch sự\s*[:：]",
            r"\s+\*{0,2}\s*Gợi ý phản hồi\s*lịch\s*sự\s*[:：]",
            r"\s+Gợi ý phản hồi\s*[:：]\s*(?=Cảm ơn|Chúng ta|Bạn có thể|Hãy |Mình )",
        ]
        cut = len(s)
        for pat in patterns:
            m = re.search(pat, s, flags=re.IGNORECASE)
            if m:
                cut = min(cut, m.start())
        return s[:cut].strip()

    def _strip_duplicate_summary_tail(self, text: str) -> str:
        """Cat bo phan model tom tat lai (Cau tra loi cuoi cung / lap lai muc 1)."""
        if not text:
            return text
        cut = len(text)
        for pat in (
            r"\n\s*\*{0,2}\s*Câu trả lời cuối cùng\s*[:：]",
            r"\n\s*Câu trả lời cuối cùng\s*[:：]",
            r"\n\s*Final answer\s*[:：]",
            r"\n\s*Kết luận\s*[:：]",
        ):
            m = re.search(pat, text, flags=re.IGNORECASE | re.MULTILINE)
            if m:
                cut = min(cut, m.start())
        trimmed = text[:cut].strip()

        m1 = re.search(r"1\)\s*Kết quả cảm xúc", trimmed, flags=re.IGNORECASE)
        if not m1:
            return trimmed
        rest = trimmed[m1.end() :]
        m2 = re.search(r"1\)\s*Kết quả cảm xúc", rest, flags=re.IGNORECASE)
        if m2:
            return trimmed[: m1.end() + m2.start()].strip()
        return trimmed

    def _postprocess_sentiment_explanation(
        self,
        response_text: str,
        fallback_label: Optional[str],
        source_comment: str = "",
    ) -> str:
        """Force a stable, readable 4-section Vietnamese explanation."""
        mode = (os.getenv("LLM_POSTPROCESS") or "full").strip().lower()
        if mode == "minimal":
            t = (response_text or "").strip()
            if t and t != "Unable to generate response":
                return self._strip_duplicate_summary_tail(t)

        text = (response_text or "").strip()
        compact_text = re.sub(r"\r\n?", "\n", text)
        compact_text = self._strip_duplicate_summary_tail(compact_text)

        sentiment = self._extract_section_value(
            compact_text,
            ["Kết quả cảm xúc", "Ket qua cam xuc", "Cảm xúc", "Cam xuc"],
        )
        keywords = self._extract_section_value(
            compact_text,
            ["Từ khóa cảm xúc", "Tu khoa cam xuc", "Từ khóa", "Tu khoa"],
        )
        _explain_extra = (
            r"|\n\s*\*{0,2}\s*Gợi ý phản hồi lịch"
            r"|\n\s*Gợi ý phản hồi lịch sự"
            r"|\n\s*#{1,3}\s*Gợi ý phản hồi"
            r"|\n\s*4\)\s*Gợi ý"
            r"|\n\s*\*{0,2}\s*Gợi ý phản hồi\s*[:：]"
        )
        explanation = self._extract_section_value(
            compact_text,
            ["Giải thích", "Giai thich", "Phân tích", "Phan tich"],
            extra_stop=_explain_extra,
        )
        suggestion = self._extract_section_value(
            compact_text,
            ["Gợi ý phản hồi", "Goi y phan hoi", "Gợi ý", "Goi y"],
        )

        if not explanation:
            tail = self._strip_duplicate_summary_tail(compact_text)
            tail = re.sub(r"^\d+\)\s*[^\n]+\n?", "", tail).strip()
            explanation = re.sub(r"\s+", " ", tail[:1200]).strip(" -\n\t") if tail else ""

        explanation = self._strip_embedded_polite_suggestion(explanation)

        normalized_label = self._normalize_label(sentiment or fallback_label)

        grounded = self._ground_keywords_in_comment(source_comment or "", keywords)
        if grounded:
            keywords = grounded
        elif keywords:
            keywords = (
                "Không có cụm nào trong phần từ khóa của model trùng với bình luận gốc "
                "(thường do model đưa ra từ đồng nghĩa). Xem giải thích hoặc bình luận gốc."
            )
        else:
            keywords = "Chưa trích xuất rõ từ khóa trùng trong bình luận."
        if not explanation:
            explanation = "Chưa có đủ dữ liệu để giải thích rõ ràng."
        if not suggestion:
            suggestion = "Bạn có thể phản hồi ngắn gọn, lịch sự và hỏi thêm ngữ cảnh để hiểu đúng ý."

        return (
            f"1) Kết quả cảm xúc: {normalized_label}\n"
            f"2) Từ khóa cảm xúc: {keywords}\n"
            f"3) Giải thích: {explanation}\n"
            f"4) Gợi ý phản hồi: {suggestion}"
        )
    
    def _call_llm(self, prompt: str, temperature: float = None,
                 max_tokens: int = None, top_p: float = None,
                 top_k: int = None, retries: int = 3) -> str:
        """
        Call LM Studio API
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Max tokens in response
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            retries: Number of retry attempts
            
        Returns:
            Generated text response
        """
        
        temperature = temperature or self.temperature
        max_tokens = max_tokens or self.max_tokens
        top_p = top_p or LLM_CONFIG['top_p']
        top_k = top_k or LLM_CONFIG['top_k']
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "top_k": top_k,
            "stream": False
        }
        
        for attempt in range(retries):
            try:
                response = requests.post(
                    f"{self.url}/completions",
                    json=payload,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if 'choices' in data and len(data['choices']) > 0:
                        text = data['choices'][0].get('text', '').strip()
                        logger.debug(f"LLM response: {text[:100]}...")
                        return text
                else:
                    logger.error(f"LLM API error: {response.status_code}")
            
            except requests.Timeout:
                logger.warning(f"LLM request timeout (attempt {attempt + 1}/{retries})")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
            except Exception as e:
                logger.error(f"LLM API error (attempt {attempt + 1}/{retries}): {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
        
        logger.error("Failed to get response from LLM after retries")
        return "Unable to generate response"
    
    def generate_emotion_keywords(self, text: str) -> List[str]:
        """
        Extract emotion keywords from text
        
        Args:
            text: Input text
            
        Returns:
            List of emotion keywords
        """
        
        prompt = f"""Hay doc binh luan tieng Viet sau va trich xuat cac tu/cum tu the hien cam xuc.

Binh luan: {text}

Chi tra ve danh sach tu/cum tu cam xuc, moi dong mot muc, khong giai thich them."""
        
        try:
            response = self._call_llm(prompt, temperature=0.3, max_tokens=100)
            keywords = [line.strip() for line in response.split('\n') if line.strip()]
            return keywords
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []
    
    def generate_context_summary(self, documents: List[str]) -> str:
        """
        Generate summary of context documents
        
        Args:
            documents: List of context documents
            
        Returns:
            Summary of context
        """
        
        doc_text = "\n".join([f"- {doc}" for doc in documents[:5]])  # Use top 5
        
        prompt = f"""Tom tat cac vi du cam xuc sau de rut ra chu de chung.

Du lieu:
{doc_text}

Yeu cau:
- Tom tat ngan gon bang tieng Viet
- Neu ro mau cam xuc noi bat

Tom tat:"""
        
        try:
            response = self._call_llm(prompt, temperature=0.5, max_tokens=200)
            return response
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return "Unable to generate summary"
    
    def compare_sentiments(self, text1: str, text2: str) -> Dict:
        """
        Compare sentiment between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Comparison analysis
        """
        
        prompt = f"""So sanh cam xuc giua 2 binh luan tieng Viet:

Binh luan 1: {text1}

Binh luan 2: {text2}

Tra loi bang tieng Viet theo dang:
1) Cam xuc binh luan 1
2) Cam xuc binh luan 2
3) Diem khac nhau chinh
4) Diem giong nhau

Ket qua so sanh:"""
        
        try:
            response = self._call_llm(prompt, max_tokens=300)
            return {'comparison': response}
        except Exception as e:
            logger.error(f"Error comparing sentiments: {e}")
            return {'error': str(e)}
    
    def get_model_info(self) -> Dict:
        """Get information about loaded model"""
        try:
            response = requests.get(
                f"{self.url}/models",
                timeout=5
            )
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
        
        return {'model': self.model_name, 'status': 'unknown'}


# Global LLM client instance
_llm_client = None


def get_llm_client(url: str = None, model_name: str = None) -> LMStudioClient:
    """
    Get or create LLM client singleton
    
    Args:
        url: LM Studio API endpoint
        model_name: Model name
        
    Returns:
        LMStudioClient instance
    """
    global _llm_client
    if _llm_client is None:
        _llm_client = LMStudioClient(url, model_name)
    return _llm_client
