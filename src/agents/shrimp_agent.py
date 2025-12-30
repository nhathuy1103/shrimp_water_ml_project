from typing import Dict, Any, Optional, List
import json
import re
import random

from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from src.llms.openai_llm import OpenAILM

def _norm_text(s: str) -> str:
    return (s or "").strip().lower()

def _has_any_water_data(w: Dict[str, Any]) -> bool:
    keys = ["NHIET_DO", "PH", "DO", "NO2", "NH4", "DO_MAN", "DO_TRONG", "PO43", "NO3", "COD", "DO_KIEM"]
    return any((k in w) and (w.get(k) not in [None, "", "None"]) for k in keys)

def _is_greeting(text: str) -> bool:
    t = _norm_text(text)
    prefixes = ["hi", "hii", "hello", "helo", "hey", "chào", "chao", "xin chào", "xin chao", "alo", "a lô"]
    return any(t == p or t.startswith(p + " ") for p in prefixes)

def _is_analysis_only(text: str) -> bool:
    t = _norm_text(text)
    keywords = [
        "phân tích", "phan tich", "đánh giá", "danh gia", "nhận xét", "nhan xet",
        "hiện trạng", "hien trang", "tình trạng", "tinh trang",
        "kết quả", "ket qua", "đang như thế nào", "dang nhu the nao", "phân loại", "phan loai", "review",
        "coi giúp", "coi dum", "coi sao", "coi thử", "coi thu"
    ]
    return any(k in t for k in keywords)

def _is_advice(text: str) -> bool:
    t = _norm_text(text)
    keywords = [
        "tư vấn", "tu van", "xử lý", "xu ly", "giải pháp", "giai phap", "khuyến nghị", "khuyen nghi",
        "nên làm", "nen lam", "cải thiện", "cai thien", "hướng dẫn", "huong dan",
        "kế hoạch", "ke hoach", "làm sao", "lam sao", "cách", "cach", "phải làm gì", "phai lam gi",
        "giúp", "giup", "cứu", "cuu"
    ]
    return any(k in t for k in keywords)

def _is_symptom_question(text: str) -> bool:
    t = _norm_text(text)
    keywords = [
        "dấu hiệu", "dau hieu", "triệu chứng", "trieu chung",
        "bơi", "boi", "bỏ ăn", "bo an", "đen mang", "den mang", "đứt râu", "dut rau",
        "mềm vỏ", "mem vo", "đỏ thân", "do than", "lờ đờ", "lo do", "nổi đầu", "noi dau",
        "chết", "chet", "đốm trắng", "dom trang", "gan tụy", "gan tuy", "phân trắng", "phan trang",
        "rụng râu", "rung rau", "rụng đuôi", "rung duoi", "đóng rong", "dong rong", "đóng nhớt", "dong nhot"
    ]
    return ("tôm" in t or "tom" in t) and any(k in t for k in keywords)

def _is_smalltalk_or_meta(text: str) -> bool:
    t = _norm_text(text)
    keys = ["bạn là ai", "ban la ai", "giới thiệu", "gioi thieu", "help", "giúp tôi", "giup toi", "hướng dẫn dùng", "huong dan dung"]
    return any(k in t for k in keys)

def _rule_intent(text: str) -> str:
    t = _norm_text(text)
    if not t:
        return "unknown"
    if _is_greeting(t):
        return "greet"
    if _is_symptom_question(t):
        return "symptom"
    if _is_smalltalk_or_meta(t):
        return "meta"
    a = _is_analysis_only(t)
    b = _is_advice(t)
    if a and not b:
        return "analysis"
    if b and not a:
        return "advice"
    if a and b:
        return "ambiguous"
    return "unknown"

def _risk_to_priority(pred: Dict[str, Any]) -> str:
    t1 = _norm_text((pred or {}).get("task1_text", ""))
    t3 = _norm_text((pred or {}).get("task3_text", ""))
    if "nguy" in t1 or "không đạt" in t3:
        return "P1"
    return "P2"

def _render_compact(water_data: Dict[str, Any], prediction: Dict[str, Any]) -> str:
    vib_text = prediction.get("task1_text", "Không có")
    vib_est = prediction.get("task2_vibrio_est", "Không có")
    env_text = prediction.get("task3_text", "Không có")
    algae_text = prediction.get("task4_text", "Không có")
    priority = _risk_to_priority(prediction)
    lines = [
        "DỮ LIỆU AO (tóm tắt)",
        f"- Điểm: {water_data.get('DIEM_QUAN_TRAC')} | Xã/Huyện: {water_data.get('XA')}/{water_data.get('HUYEN')}",
        f"- Nhiệt độ: {water_data.get('NHIET_DO')} | pH: {water_data.get('PH')} | DO: {water_data.get('DO')}",
        f"- Độ mặn: {water_data.get('DO_MAN')} | Độ trong: {water_data.get('DO_TRONG')} | Kiềm: {water_data.get('DO_KIEM')}",
        f"- NO2: {water_data.get('NO2')} | NO3: {water_data.get('NO3')} | NH4: {water_data.get('NH4')} | PO43: {water_data.get('PO43')} | COD: {water_data.get('COD')}",
        "",
        "KẾT QUẢ MÔ HÌNH (4 TASK)",
        f"- Vibrio: {vib_text}",
        f"- Vibrio ước lượng: ~{vib_est} CFU/ml",
        f"- Môi trường: {env_text}",
        f"- Tảo thức ăn: {algae_text}",
        f"- Mức ưu tiên: {priority}",
    ]
    return "\n".join([x for x in lines if x is not None]).strip()

def _localize_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("Bạn", "Mình").replace("bạn", "mình")
    text = text.replace("Tôi", "Em").replace("tôi", "em")
    text = re.sub(r"\bkhông\b", "hông", text, flags=re.IGNORECASE)
    text = re.sub(r"\bnhanh\b", "lẹ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bnghiêm trọng\b", "căng", text, flags=re.IGNORECASE)
    return text.strip()

def _pick(*arr: str) -> str:
    return random.choice([a for a in arr if a])

class ShrimpAgent:
    def __init__(self, vectordb=None, llm: Optional[ChatOpenAI] = None):
        self.llm = llm or OpenAILM(model_name="gpt-4.1-mini", temperature=0.15).get_llm()
        self.qa_chain = None
        if vectordb is not None:
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                retriever=vectordb.as_retriever(search_kwargs={"k": 4}),
                chain_type="stuff",
            )

    def _llm_intent(self, question: str, has_data: bool, has_pred: bool) -> str:
        sys = """
Bạn là bộ phân loại ý định hội thoại cho chatbot nuôi tôm (miền Tây).
Chỉ trả về JSON 1 dòng, không thêm chữ khác.
Các nhãn:
- greet
- analysis (phân tích hiện trạng nước, không giải pháp)
- advice (tư vấn xử lý theo nước/ao)
- symptom (dấu hiệu tôm bất thường/bệnh)
- knowledge (kiến thức chung: tảo, pH, DO, Vibrio... không cần dữ liệu)
- meta (hỏi cách dùng/giới thiệu bot)
""".strip()

        usr = json.dumps({
            "question": question,
            "has_water_data": has_data,
            "has_prediction": has_pred
        }, ensure_ascii=False)

        out = self.llm.invoke([{"role": "system", "content": sys}, {"role": "user", "content": usr}]).content or ""
        out = out.strip()
        try:
            j = json.loads(out)
            it = _norm_text(j.get("intent", ""))
            allowed = {"greet", "analysis", "advice", "symptom", "knowledge", "meta"}
            return it if it in allowed else "knowledge"
        except Exception:
            return "knowledge"

    def answer(self, question: str, water_data: Optional[Dict[str, Any]] = None, prediction: Optional[Dict[str, Any]] = None) -> str:
        water_data = water_data or {}
        prediction = prediction or {}
        has_data = _has_any_water_data(water_data)
        has_pred = bool(prediction)

        it = _rule_intent(question)
        if it in {"unknown", "ambiguous"}:
            it = self._llm_intent(question, has_data, has_pred)

        if it == "greet":
            return _pick(
                "Dạ chào mình 👋 Mình muốn em **phân tích nước**, **tư vấn xử lý**, hay **hỏi dấu hiệu tôm** nè?",
                "Chào bà con 👋 Mình hỏi em kiểu nào: **phân tích**, **tư vấn**, hay **dấu hiệu tôm** nghen?"
            )

        if it == "meta":
            return _pick(
                "Dạ em hỗ trợ 3 kiểu: **phân tích nước** (không giải pháp), **tư vấn xử lý**, và **hỏi dấu hiệu tôm bất thường**. Mình cứ hỏi tự nhiên nghen.",
                "Mình cứ nhập số nước rồi bấm **Dự đoán** để em coi theo mô hình. Còn hỏi dấu hiệu tôm/bệnh thì hỏi trực tiếp cũng được."
            )

        if it in {"analysis", "advice"} and (not has_data or not has_pred):
            if it == "analysis":
                return "Dạ muốn **phân tích nước** thì mình nhập số liệu rồi bấm **Dự đoán** trước nghen, để em coi đúng theo ao mình."
            return "Dạ muốn **tư vấn xử lý theo ao** thì mình nhập số liệu rồi bấm **Dự đoán** trước nghen. Còn nếu hỏi **kiến thức chung** thì mình hỏi luôn cũng được."

        if it == "symptom":
            system = """
Bạn là trợ lý cho người nuôi tôm quảng canh ở Cà Mau, nói kiểu miền Tây, dễ hiểu.
Bắt buộc:
- Tập trung đúng dấu hiệu tôm/bệnh, không tự chuyển sang phân tích nước nếu người dùng không đưa số.
- Trả lời theo khung 4 mục.
- Không hướng dẫn dùng kháng sinh/hóa chất theo liều.
- Nếu khẩn (tôm chết nhanh/nổi đầu nhiều) phải cảnh báo và khuyên liên hệ cán bộ địa phương.
""".strip()
            user = f"""
Câu hỏi: {question}

Trả lời đúng format:

**1) Mình đang thấy gì**
- ...

**2) Khả năng đang gặp (2–4)**
- A: ...
- B: ...
- C: ...

**3) Em hỏi thêm 1–2 câu để xác định**
- Câu 1: ...
- Câu 2: ...

**4) Mình coi/kiểm tra an toàn tại ao**
- ...
""".strip()
            resp = self.llm.invoke([{"role": "system", "content": system}, {"role": "user", "content": user}])
            return _localize_text((resp.content or "").strip())

        if it == "analysis":
            compact = _render_compact(water_data, prediction)
            system = """
Bạn là trợ lý PHÂN TÍCH môi trường nuôi tôm quảng canh ở Cà Mau, nói kiểu miền Tây.
Bắt buộc:
- CHỈ phân tích hiện trạng dựa trên dữ liệu và kết quả mô hình.
- Tuyệt đối KHÔNG đưa giải pháp/khuyến nghị/kế hoạch/hướng dẫn.
- Tránh các từ/cụm: xử lý, nên làm, khuyến nghị, đề xuất, hướng dẫn, kế hoạch, liều, dùng, bổ sung, tăng, giảm.
- Nếu thiếu dữ liệu quan trọng, hỏi tối đa 2 câu ngắn.
""".strip()
            user = f"""
{compact}

Câu hỏi: {question}

Trả lời đúng format:

**1) Đánh giá**
- ...

**2) Các điểm đạt (tối đa 5)**
- ...

**3) Các điểm lệch ngưỡng**
- từng chỉ số: (hiện tại | chuẩn | hệ quả/rủi ro)

**4) Rủi ro tổng hợp theo 4 task**
- ...
""".strip()
            resp = self.llm.invoke([{"role": "system", "content": system}, {"role": "user", "content": user}])
            return _localize_text((resp.content or "").strip())

        if it == "advice":
            compact = _render_compact(water_data, prediction)
            rag_snippet = ""
            if self.qa_chain is not None:
                rag_query = "Tóm tắt tối đa 6 gạch đầu dòng về xử lý Vibrio, DO thấp, pH lệch, NO2/NH4 cao, quản lý tảo trong nuôi tôm quảng canh."
                try:
                    rag_resp = self.qa_chain.invoke({"query": rag_query})
                    rag_snippet = (rag_resp.get("result") or "").strip()
                except Exception:
                    rag_snippet = ""

            system = """
Bạn là trợ lý TƯ VẤN nuôi tôm quảng canh ở Cà Mau, nói kiểu miền Tây, dễ hiểu.
Bắt buộc:
- Bám sát dữ liệu ao + 4 task.
- Nói rõ chỉ số lệch ngưỡng và vì sao nguy.
- Gạch đầu dòng, ngắn, dễ làm theo.
- Có ưu tiên P1/P2/P3 và mốc thời gian.
- Không khuyến khích lạm dụng kháng sinh/hóa chất.
""".strip()
            user = f"""
{compact}

Tham khảo tài liệu (nếu có):
{rag_snippet if rag_snippet else "(không có)"}

Câu hỏi: {question}

Trả lời đúng format:

**1) Đánh giá nhanh**
- ...

**2) Vấn đề chính**
- từng chỉ số: (hiện tại | chuẩn | vì sao nguy)

**3) Kế hoạch theo ưu tiên**
- P1 (24h): ...
- P2 (3 ngày): ...
- P3 (1–2 tuần): ...

**4) Lưu ý an toàn sinh học**
- ...
""".strip()
            resp = self.llm.invoke([{"role": "system", "content": system}, {"role": "user", "content": user}])
            return _localize_text((resp.content or "").strip())

        system = """
Bạn là trợ lý kỹ thuật nuôi tôm quảng canh ở Cà Mau, nói kiểu miền Tây, dễ hiểu.
Bắt buộc:
- Trả lời kiến thức chung theo câu hỏi (tảo, pH, DO, Vibrio, môi trường...).
- Không bắt người dùng phải “phân tích/tư vấn” nếu họ hỏi chung.
- Nếu cần thông tin để sát thực tế, hỏi tối đa 2 câu.
- Không khuyến khích lạm dụng kháng sinh/hóa chất.
""".strip()
        user = f"""
Câu hỏi: {question}

Trả lời gọn, dễ hiểu, đúng giọng miền Tây. Nếu câu hỏi đang thiếu thông tin để kết luận chắc, hỏi lại tối đa 2 câu.
""".strip()
        resp = self.llm.invoke([{"role": "system", "content": system}, {"role": "user", "content": user}])
        return _localize_text((resp.content or "").strip())
