from __future__ import annotations

import os
from functools import wraps
from datetime import date, datetime, timedelta

from flask import Flask, request, render_template, session, redirect, url_for, jsonify
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash, check_password_hash

import markdown as md
import bleach
from markupsafe import Markup
from sqlalchemy import func, and_
from lunardate import LunarDate

from src.pipeline.predict_pipeline import CustomData, PredictionPipeline
from src.agents.rag_loader import load_rag_db
from src.agents.shrimp_agent import ShrimpAgent


from src.models.db_models import (
    db, nguoi_dung, mau_nuoc, tin_nhan_chat, thu_hoach,
    thong_bao, da_doc_thong_bao
)

load_dotenv()

ALLOWED_TAGS = [
    "p", "br", "strong", "em", "ul", "ol", "li", "code", "pre", "blockquote", "hr",
    "h1", "h2", "h3", "h4", "h5", "h6", "a"
]
ALLOWED_ATTRS = {"a": ["href", "title", "target", "rel"]}


# =========================
# Utils
# =========================
def render_md_safe(text: str) -> Markup:
    html = md.markdown(text or "", extensions=["extra", "sane_lists"])
    clean = bleach.clean(html, tags=ALLOWED_TAGS, attributes=ALLOWED_ATTRS, strip=True)
    clean = bleach.linkify(clean)
    return Markup(clean)


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "nguoi_dung_id" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


def get_lunar_notice(today: date | None = None):
    today = today or date.today()
    lunar = LunarDate.fromSolarDate(today.year, today.month, today.day)
    d = lunar.day

    if d in (4, 5, 6):
        return (
            f"🦐 Hôm nay ngày {d} âm lịch, bà con còn xổ vuông không ạ? "
            "Nếu nhà mình đã đóng cống rồi thì mình nhớ kiểm tra kỹ lưỡng nước trước khi giữ lại trong vuông nghen.",
            "warn",
        )

    if d in (18, 19, 20):
        return (
            f"🦐 Hôm nay ngày {d} âm lịch, nước lớn đó ạ. "
            "Bà con nhớ coi lại nước ngoài sông kỹ rồi hẵng bơm vô vuông nghen.",
            "warn",
        )

    return (None, None)


def get_officer_codes():
    raw = (os.getenv("OFFICER_CODES", "") or "").strip()
    return {x.strip().upper() for x in raw.split(",") if x.strip()}


def staff_code_valid(code: str) -> bool:
    codes = get_officer_codes()
    if not codes:
        return True
    return (code or "").strip().upper() in codes


def officer_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if session.get("vai_tro") != "officer":
            return redirect(url_for("profile"))
        if not session.get("ma_can_bo_hop_le"):
            return redirect(url_for("profile"))
        return f(*args, **kwargs)
    return decorated


pipeline = PredictionPipeline()
try:
    vectordb = load_rag_db()
except Exception:
    vectordb = None
agent = ShrimpAgent(vectordb=vectordb)


def get_session_id() -> str:
    if "sid" not in session:
        session["sid"] = os.urandom(8).hex()
    return session["sid"]


def _to_float(form, key: str) -> float:
    return float((form.get(key) or "").strip())


def _to_int(form, key: str) -> int:
    return int((form.get(key) or "").strip())


def _to_kg(form, key: str) -> float:
    v = (form.get(key) or "0").strip().replace(",", "")
    try:
        return float(v)
    except Exception:
        return 0.0


def _to_int_optional(form, key: str):
    v = (form.get(key) or "").strip().replace(",", "")
    if not v:
        return None
    try:
        return int(float(v))
    except Exception:
        return None


def _group_key(d: date, group: str) -> str:
    if group == "month":
        return d.strftime("%Y-%m")
    if group == "year":
        return d.strftime("%Y")
    return d.isoformat()


def _parse_days(value: str | None) -> int:
    try:
        d = int((value or "30").strip())
    except Exception:
        d = 30
    return d if d in (7, 30, 90, 365) else 30


def _parse_group(value: str | None, allowed=("day", "month", "year")) -> str:
    g = (value or allowed[0]).strip().lower()
    return g if g in allowed else allowed[0]


def _parse_area(value: str | None, allowed=("huyen", "xa")) -> str:
    a = (value or allowed[0]).strip().lower()
    return a if a in allowed else allowed[0]


def _refresh_flags_from_user(u: nguoi_dung):
    session["tinh"] = u.tinh or ""
    session["vai_tro"] = (u.vai_tro or "farmer")
    session["biet_danh"] = u.biet_danh or ""
    session["ma_can_bo"] = u.ma_can_bo or ""

    if session["vai_tro"] == "officer":
        session["ma_can_bo_hop_le"] = bool(staff_code_valid(session["ma_can_bo"]))
    else:
        session["ma_can_bo_hop_le"] = False

    session.modified = True


def _norm_scope(s: str | None) -> str:
    return (s or "").strip().lower()


def _calc_notice_level(result: dict) -> str:
    t1 = (result.get("task1_text") or "").lower()
    t3 = (result.get("task3_text") or "").lower()

    if ("cao" in t1) or ("không phù hợp" in t3) or ("nguy hiểm" in t3):
        return "danger"
    if ("trung" in t1) or ("cảnh báo" in t3):
        return "warn"
    return "info"


def _build_notice_text(input_dict: dict, result: dict) -> str:
    return (
        f"Điểm: {input_dict.get('diem_quan_trac','')}, "
        f"Nhiệt độ: {input_dict.get('nhiet_do','')}°C, pH: {input_dict.get('ph','')}, DO: {input_dict.get('do','')}.\n"
        f"Vibrio: {result.get('task1_text','')} (≈ {result.get('task2_vibrio_est','')} CFU/ml).\n"
        f"Môi trường: {result.get('task3_text','')}. Tảo thức ăn: {result.get('task4_text','')}."
    )


def get_commune_notices_for_user(user_id: int, xa: str | None, limit: int = 5):
    scope = _norm_scope(xa)
    if not scope:
        return []

    q = (
        db.session.query(thong_bao)
        .outerjoin(
            da_doc_thong_bao,
            and_(
                da_doc_thong_bao.thong_bao_id == thong_bao.id,
                da_doc_thong_bao.nguoi_dung_id == user_id,
            )
        )
        .filter(thong_bao.scope_type == "commune")
        .filter(thong_bao.scope_value == scope)
        .filter(da_doc_thong_bao.id.is_(None))
        .order_by(thong_bao.tao_luc.desc(), thong_bao.id.desc())
        .limit(limit)
    )
    return q.all()


# =========================
# App factory
# =========================
def create_app() -> Flask:
    app = Flask(__name__)
    app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")

    _configure_db(app)
    db.init_app(app)

    _register_hooks(app)
    _register_routes(app)

    return app


def _configure_db(app: Flask) -> None:
    mysql_user = os.getenv("MYSQL_USER", "root")
    mysql_password = os.getenv("MYSQL_PASSWORD", "root123")
    mysql_host = os.getenv("MYSQL_HOST", "localhost")
    mysql_port = os.getenv("MYSQL_PORT", "3306")
    mysql_db = os.getenv("MYSQL_DB", "shrimp_db")

    app.config["SQLALCHEMY_DATABASE_URI"] = (
        f"mysql+pymysql://{mysql_user}:{mysql_password}@{mysql_host}:{mysql_port}/{mysql_db}?charset=utf8mb4"
    )
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {"pool_pre_ping": True, "pool_recycle": 1800}


def _register_hooks(app: Flask) -> None:
    @app.before_request
    def sync_user_session():
        uid = session.get("nguoi_dung_id")
        if not uid:
            return
        u = nguoi_dung.query.get(uid)
        if not u:
            session.clear()
            return
        _refresh_flags_from_user(u)


# =========================
# Routes
# =========================
def _register_routes(app: Flask) -> None:
    @app.route("/", methods=["GET"])
    def index():
        if "nguoi_dung_id" not in session:
            return redirect(url_for("login"))
        return redirect(url_for("predict_datapoint"))

    # ---------- Auth ----------
    @app.route("/register", methods=["GET", "POST"])
    def register():
        if request.method == "GET":
            return render_template("register.html")

        ten_dang_nhap = (request.form.get("username") or "").strip()
        mat_khau = (request.form.get("password") or "").strip()

        ho_ten = (request.form.get("full_name") or "").strip()
        so_dien_thoai = (request.form.get("phone") or "").strip()
        xac_nhan = (request.form.get("confirm_password") or "").strip()
        tinh = (request.form.get("province") or "").strip()
        huyen = (request.form.get("district") or "").strip()
        xa = (request.form.get("commune") or "").strip()

        if not ten_dang_nhap or not mat_khau:
            return render_template("register.html", error="Thiếu username hoặc password")
        if mat_khau != xac_nhan:
            return render_template("register.html", error="Mật khẩu xác nhận không khớp.")
        if nguoi_dung.query.filter_by(ten_dang_nhap=ten_dang_nhap).first():
            return render_template("register.html", error="Username đã tồn tại")

        u = nguoi_dung(
            ten_dang_nhap=ten_dang_nhap,
            mat_khau_hash=generate_password_hash(mat_khau),
            ho_ten=ho_ten or None,
            so_dien_thoai=so_dien_thoai or None,
            tinh=tinh or None,
            huyen=huyen or None,
            xa=xa or None,
            vai_tro="farmer",
        )
        db.session.add(u)
        db.session.commit()
        return redirect(url_for("login"))

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "GET":
            return render_template("login.html")

        ten_dang_nhap = (request.form.get("username") or "").strip()
        mat_khau = (request.form.get("password") or "").strip()

        u = nguoi_dung.query.filter_by(ten_dang_nhap=ten_dang_nhap).first()
        if not u or not check_password_hash(u.mat_khau_hash, mat_khau):
            return render_template("login.html", error="Sai tài khoản hoặc mật khẩu")

        session.clear()
        session["nguoi_dung_id"] = u.id
        session["username"] = u.ten_dang_nhap
        _refresh_flags_from_user(u)

        session["chat_history"] = []
        session["scroll_to_bottom"] = False
        return redirect(url_for("predict_datapoint"))

    @app.route("/logout", methods=["GET"])
    def logout():
        session.clear()
        return redirect(url_for("login"))

    # ---------- Predict ----------
    @app.route("/predictdata", methods=["GET", "POST"])
    @login_required
    def predict_datapoint():
        session.permanent = True

        if request.method == "GET":
            uid = session.get("nguoi_dung_id")
            u = nguoi_dung.query.get(uid)
            lunar_text, lunar_level = get_lunar_notice()
            notices = get_commune_notices_for_user(uid, u.xa if u else None, limit=5)

            return render_template(
                "home.html",
                results=session.get("last_results"),
                chat_history=session.get("chat_history", []),
                scroll_to_bottom=session.pop("scroll_to_bottom", False),
                notice_text=lunar_text,
                notice_level=lunar_level,
                notices=notices,
            )

        lunar_text, lunar_level = get_lunar_notice()

        diem_quan_trac = (request.form.get("DIEM_QUAN_TRAC") or "").strip()
        xa = (request.form.get("XA") or "").strip()
        huyen = (request.form.get("HUYEN") or "").strip()

        custom_data = CustomData(
            diem_quan_trac=diem_quan_trac,
            xa=xa,
            huyen=huyen,
            nhiet_do=_to_float(request.form, "NHIET_DO"),
            ph=_to_float(request.form, "PH"),
            do=_to_float(request.form, "DO"),
            do_man=_to_float(request.form, "DO_MAN"),
            do_trong=_to_float(request.form, "DO_TRONG"),
            do_kiem=_to_float(request.form, "DO_KIEM"),
            no2=_to_float(request.form, "NO2"),
            no3=_to_float(request.form, "NO3"),
            nh4=_to_float(request.form, "NH4"),
            po43=_to_float(request.form, "PO43"),
            cod=_to_float(request.form, "COD"),
            nam=_to_int(request.form, "NAM"),
            thang=_to_int(request.form, "THANG"),
            ngay=_to_int(request.form, "NGAY"),
        )

        input_df = custom_data.get_data_as_dataframe()
        result = pipeline.predict(input_df)

        session["last_results"] = result
        session["last_input"] = input_df.to_dict(orient="records")[0]
        session.modified = True

        row = mau_nuoc(
            nguoi_dung_id=session.get("nguoi_dung_id"),
            diem_quan_trac=diem_quan_trac,
            xa=xa,
            huyen=huyen,
            nhiet_do=float(custom_data.nhiet_do),
            ph=float(custom_data.ph),
            do=float(custom_data.do),
            do_man=float(custom_data.do_man),
            do_trong=float(custom_data.do_trong),
            do_kiem=float(custom_data.do_kiem),
            no2=float(custom_data.no2),
            no3=float(custom_data.no3),
            nh4=float(custom_data.nh4),
            po43=float(custom_data.po43),
            cod=float(custom_data.cod),
            nam=int(custom_data.nam),
            thang=int(custom_data.thang),
            ngay=int(custom_data.ngay),
            nguy_co_vibrio_text=result.get("task1_text"),
            vibrio_uoc_luong=float(result.get("task2_vibrio_est")) if result.get("task2_vibrio_est") is not None else None,
            phu_hop_moi_truong_text=result.get("task3_text"),
            tao_thuc_an_text=result.get("task4_text"),
        )

        try:
            db.session.add(row)
            db.session.commit()
        except Exception:
            db.session.rollback()

        # officer tạo thông báo cho xã
        try:
            uid = session.get("nguoi_dung_id")
            u = nguoi_dung.query.get(uid)

            is_officer = (session.get("vai_tro") == "officer") and bool(session.get("ma_can_bo_hop_le"))
            if is_officer and u:
                scope_xa = _norm_scope(u.xa or xa)
                if scope_xa:
                    last_input = session.get("last_input") or {}
                    tb = thong_bao(
                        nguoi_tao_id=uid,
                        scope_type="commune",
                        scope_value=scope_xa,
                        tieu_de=f"Đánh giá môi trường nước – Xã {u.xa or xa}",
                        noi_dung=_build_notice_text(last_input, result),
                        muc_do=_calc_notice_level(result),
                        mau_nuoc_id=row.id,
                    )
                    db.session.add(tb)
                    db.session.commit()
        except Exception:
            db.session.rollback()

        uid = session.get("nguoi_dung_id")
        u = nguoi_dung.query.get(uid)
        notices = get_commune_notices_for_user(uid, u.xa if u else None, limit=5)

        return render_template(
            "home.html",
            results=result,
            chat_history=session.get("chat_history", []),
            scroll_to_bottom=False,
            notice_text=lunar_text,
            notice_level=lunar_level,
            notices=notices,
        )

    # ---------- Notices ----------
    @app.route("/api/notices/read", methods=["POST"])
    @login_required
    def mark_notice_read():
        uid = session.get("nguoi_dung_id")
        notice_id = (request.form.get("notice_id") or "").strip()
        if not notice_id.isdigit():
            return jsonify({"ok": False, "error": "notice_id invalid"}), 400

        tb = thong_bao.query.get(int(notice_id))
        if not tb:
            return jsonify({"ok": False, "error": "not found"}), 404

        u = nguoi_dung.query.get(uid)
        user_scope = _norm_scope(u.xa if u else None)

        allow = False
        if tb.target_user_id and tb.target_user_id == uid:
            allow = True
        elif tb.scope_type == "commune" and tb.scope_value == user_scope:
            allow = True

        if not allow:
            return jsonify({"ok": False, "error": "forbidden"}), 403

        existed = da_doc_thong_bao.query.filter_by(thong_bao_id=tb.id, nguoi_dung_id=uid).first()
        if existed:
            return jsonify({"ok": True, "already": True})

        try:
            db.session.add(da_doc_thong_bao(thong_bao_id=tb.id, nguoi_dung_id=uid))
            db.session.commit()
            return jsonify({"ok": True})
        except Exception:
            db.session.rollback()
            return jsonify({"ok": False, "error": "db error"}), 500

    # ---------- Chat ----------
    @app.route("/chat", methods=["POST"])
    @login_required
    def chat():
        session.permanent = True
        cau_hoi = (request.form.get("chat_input") or "").strip()
        if not cau_hoi:
            return redirect(url_for("predict_datapoint"))

        sid = get_session_id()
        uid = session.get("nguoi_dung_id")

        last_results = session.get("last_results")
        last_input = session.get("last_input")

        try:
            db.session.add(tin_nhan_chat(nguoi_dung_id=uid, session_id=sid, vai="user", noi_dung=cau_hoi))
            db.session.commit()
        except Exception:
            db.session.rollback()

        if last_results is None or last_input is None:
            tra_loi = "Dạ mình nhập thông số nước rồi bấm **Dự đoán** trước nghen, rồi hỏi em tiếp."
        else:
            tra_loi = agent.answer(question=cau_hoi, water_data=last_input, prediction=last_results)

        try:
            db.session.add(tin_nhan_chat(nguoi_dung_id=uid, session_id=sid, vai="assistant", noi_dung=tra_loi))
            db.session.commit()
        except Exception:
            db.session.rollback()

        chat_history = session.get("chat_history", [])
        chat_history.append({"role": "user", "content": cau_hoi, "content_html": render_md_safe(cau_hoi)})
        chat_history.append({"role": "assistant", "content": tra_loi, "content_html": render_md_safe(tra_loi)})

        session["chat_history"] = chat_history
        session["scroll_to_bottom"] = True
        session.modified = True

        return redirect(url_for("predict_datapoint"))

    @app.route("/chat/clear", methods=["POST"])
    @login_required
    def clear_chat():
        session.permanent = True
        session.pop("chat_history", None)
        session["scroll_to_bottom"] = False
        session.modified = True
        return redirect(url_for("predict_datapoint"))

    # ---------- Harvest (farmer) ----------
    @app.route("/harvest", methods=["GET", "POST"])
    @login_required
    def harvest():
        uid = session.get("nguoi_dung_id")

        if request.method == "POST":
            harvest_date_str = (request.form.get("harvest_date") or "").strip()
            ngay_thu_hoach = datetime.strptime(harvest_date_str, "%Y-%m-%d").date() if harvest_date_str else date.today()

            rec = thu_hoach(
                nguoi_dung_id=uid,
                ngay_thu_hoach=ngay_thu_hoach,
                kg_tom_su=_to_kg(request.form, "kg_tom_su"),
                kg_tom_the=_to_kg(request.form, "kg_tom_the"),
                kg_tom_bac=_to_kg(request.form, "kg_tom_bac"),
                co_tom_su_con_kg=_to_int_optional(request.form, "tom_su_size"),
                ghi_chu=(request.form.get("note") or "").strip() or None,
            )
            if (rec.kg_tom_su or 0) <= 0:
                rec.co_tom_su_con_kg = None

            try:
                db.session.add(rec)
                db.session.commit()
            except Exception:
                db.session.rollback()

            return redirect(url_for("harvest"))

        rows = (
            thu_hoach.query
            .filter_by(nguoi_dung_id=uid)
            .order_by(thu_hoach.ngay_thu_hoach.desc(), thu_hoach.id.desc())
            .limit(60)
            .all()
        )
        return render_template("harvest.html", rows=rows, today=date.today().isoformat())

    # ✅ NEW: Harvest summary API (farmer) — để dashboard harvest.html gọi được
    @app.route("/api/harvest/summary", methods=["GET"])
    @login_required
    def harvest_summary():
        uid = session.get("nguoi_dung_id")
        days = _parse_days(request.args.get("days"))
        group = _parse_group(request.args.get("group"), allowed=("day", "month", "year"))

        start_date = date.today() - timedelta(days=days - 1)

        records = (
            thu_hoach.query
            .filter(thu_hoach.nguoi_dung_id == uid)
            .filter(thu_hoach.ngay_thu_hoach >= start_date)
            .order_by(thu_hoach.ngay_thu_hoach.asc(), thu_hoach.id.asc())
            .all()
        )

        # group theo day/month/year
        time_map: dict[str, dict] = {}
        total_size_weighted_sum = 0.0
        total_size_weight = 0.0

        for r in records:
            k = _group_key(r.ngay_thu_hoach, group)
            bucket = time_map.setdefault(k, {
                "su": 0.0, "the": 0.0, "bac": 0.0,
                "size_weighted_sum": 0.0, "size_weight": 0.0
            })

            su_kg = float(r.kg_tom_su or 0)
            the_kg = float(r.kg_tom_the or 0)
            bac_kg = float(r.kg_tom_bac or 0)

            bucket["su"] += su_kg
            bucket["the"] += the_kg
            bucket["bac"] += bac_kg

            # trung bình cỡ tôm sú: ưu tiên weighted theo kg_tom_su, nếu kg=0 thì weight=1
            if r.co_tom_su_con_kg is not None:
                size_val = float(r.co_tom_su_con_kg)
                weight = su_kg if su_kg > 0 else 1.0
                bucket["size_weighted_sum"] += size_val * weight
                bucket["size_weight"] += weight

                total_size_weighted_sum += size_val * weight
                total_size_weight += weight

        labels = sorted(time_map.keys())
        series_su = [time_map[k]["su"] for k in labels]
        series_the = [time_map[k]["the"] for k in labels]
        series_bac = [time_map[k]["bac"] for k in labels]
        series_size = [
            (time_map[k]["size_weighted_sum"] / time_map[k]["size_weight"]) if time_map[k]["size_weight"] > 0 else None
            for k in labels
        ]

        total_su = float(sum(series_su))
        total_the = float(sum(series_the))
        total_bac = float(sum(series_bac))
        avg_size = (total_size_weighted_sum / total_size_weight) if total_size_weight > 0 else None

        return jsonify({
            "ok": True,
            "days": days,
            "group": group,
            "labels": labels,
            "series": {
                "tom_su": series_su,
                "tom_the": series_the,
                "tom_bac": series_bac,
                "tom_su_size": series_size,
            },
            "totals": {
                "tom_su": total_su,
                "tom_the": total_the,
                "tom_bac": total_bac,
                "all": total_su + total_the + total_bac,
            },
            "stats": {"avg_tom_su_size": avg_size},
        })

    # ---------- Officer Harvest Dashboard ----------
    @app.route("/officer/harvest", methods=["GET"])
    @login_required
    @officer_required
    def officer_harvest():
        return render_template("officer_harvest.html")

    @app.route("/api/officer/harvest/summary", methods=["GET"])
    @login_required
    @officer_required
    def officer_harvest_summary():
        days = _parse_days(request.args.get("days"))
        group = _parse_group(request.args.get("group"), allowed=("day", "month", "year"))
        area = _parse_area(request.args.get("area"), allowed=("huyen", "xa"))

        districts_raw = (request.args.get("districts") or "").strip()
        districts = [x.strip() for x in districts_raw.split(",") if x.strip()]

        commune = (request.args.get("commune") or "").strip()

        start_date = date.today() - timedelta(days=days - 1)

        if group == "month":
            bucket_col = func.date_format(thu_hoach.ngay_thu_hoach, "%Y-%m")
        elif group == "year":
            bucket_col = func.date_format(thu_hoach.ngay_thu_hoach, "%Y")
        else:
            bucket_col = func.date_format(thu_hoach.ngay_thu_hoach, "%Y-%m-%d")

        area_col = nguoi_dung.huyen if area == "huyen" else nguoi_dung.xa

        q = (
            db.session.query(
                bucket_col.label("bucket"),
                area_col.label("area_key"),
                func.coalesce(func.sum(thu_hoach.kg_tom_su), 0).label("sum_su"),
                func.coalesce(func.sum(thu_hoach.kg_tom_the), 0).label("sum_the"),
                func.coalesce(func.sum(thu_hoach.kg_tom_bac), 0).label("sum_bac"),
                func.coalesce(func.sum(thu_hoach.kg_tom_su + thu_hoach.kg_tom_the + thu_hoach.kg_tom_bac), 0).label("sum_all"),
            )
            .join(nguoi_dung, nguoi_dung.id == thu_hoach.nguoi_dung_id)
            .filter(thu_hoach.ngay_thu_hoach >= start_date)
        )

        tinh = (session.get("tinh") or "").strip()
        if tinh:
            q = q.filter(nguoi_dung.tinh == tinh)

        if districts:
            q = q.filter(nguoi_dung.huyen.in_(districts))

        if commune:
            q = q.filter(nguoi_dung.xa == commune)

        q = q.group_by(bucket_col, area_col).order_by(bucket_col.asc())

        rows = q.all()

        buckets = sorted({r.bucket for r in rows if r.bucket})
        keys = sorted({(r.area_key or "Chưa khai báo") for r in rows})

        time_map = {b: {"su": 0.0, "the": 0.0, "bac": 0.0, "all": 0.0} for b in buckets}
        by_area = {k: {"su": 0.0, "the": 0.0, "bac": 0.0, "all": 0.0} for k in keys}

        for r in rows:
            b = r.bucket
            k = r.area_key or "Chưa khai báo"

            su = float(r.sum_su or 0)
            the = float(r.sum_the or 0)
            bac = float(r.sum_bac or 0)
            allv = float(r.sum_all or 0)

            time_map[b]["su"] += su
            time_map[b]["the"] += the
            time_map[b]["bac"] += bac
            time_map[b]["all"] += allv

            by_area[k]["su"] += su
            by_area[k]["the"] += the
            by_area[k]["bac"] += bac
            by_area[k]["all"] += allv

        labels = buckets
        series_su = [time_map[b]["su"] for b in labels]
        series_the = [time_map[b]["the"] for b in labels]
        series_bac = [time_map[b]["bac"] for b in labels]

        total_su = sum(series_su)
        total_the = sum(series_the)
        total_bac = sum(series_bac)

        area_table = []
        for k in sorted(by_area.keys(), key=lambda x: by_area[x]["all"], reverse=True):
            area_table.append({
                "key": k,
                "tom_su": by_area[k]["su"],
                "tom_the": by_area[k]["the"],
                "tom_bac": by_area[k]["bac"],
                "all": by_area[k]["all"],
            })

        return jsonify({
            "ok": True,
            "days": days,
            "group": group,
            "area": area,
            "districts": districts,
            "commune": commune,
            "labels": labels,
            "series": {"tom_su": series_su, "tom_the": series_the, "tom_bac": series_bac},
            "totals": {"tom_su": total_su, "tom_the": total_the, "tom_bac": total_bac, "all": total_su + total_the + total_bac},
            "by_area": area_table,
        })

    # ---------- Officer: list districts/communes ----------
    @app.route("/api/officer/areas/districts", methods=["GET"])
    @login_required
    @officer_required
    def officer_districts():
        tinh = (session.get("tinh") or "").strip()

        q = (
            db.session.query(nguoi_dung.huyen)
            .filter(nguoi_dung.huyen.isnot(None))
            .filter(nguoi_dung.huyen != "")
        )
        if tinh:
            q = q.filter(nguoi_dung.tinh == tinh)

        rows = q.distinct().order_by(nguoi_dung.huyen.asc()).all()
        districts = [r[0] for r in rows if r[0]]
        return jsonify({"ok": True, "districts": districts})

    @app.route("/api/officer/areas/communes", methods=["GET"])
    @login_required
    @officer_required
    def officer_communes():
        tinh = (session.get("tinh") or "").strip()
        district = (request.args.get("district") or "").strip()

        q = (
            db.session.query(nguoi_dung.xa)
            .filter(nguoi_dung.xa.isnot(None))
            .filter(nguoi_dung.xa != "")
        )

        if tinh:
            q = q.filter(nguoi_dung.tinh == tinh)
        if district:
            q = q.filter(nguoi_dung.huyen == district)

        rows = q.distinct().order_by(nguoi_dung.xa.asc()).all()
        communes = [r[0] for r in rows if r[0]]
        return jsonify({"ok": True, "district": district, "communes": communes})

    # ---------- Profile ----------
    @app.route("/profile", methods=["GET", "POST"])
    @login_required
    def profile():
        u = nguoi_dung.query.get(session.get("nguoi_dung_id"))
        if not u:
            session.clear()
            return redirect(url_for("login"))

        if request.method == "GET":
            _refresh_flags_from_user(u)
            return render_template("profile.html", user=u)

        u.ho_ten = (request.form.get("full_name") or "").strip() or None
        u.so_dien_thoai = (request.form.get("phone") or "").strip() or None
        u.biet_danh = (request.form.get("nickname") or "").strip() or None
        u.tinh = (request.form.get("province") or "").strip() or None
        u.huyen = (request.form.get("district") or "").strip() or None
        u.xa = (request.form.get("commune") or "").strip() or None
        u.ap = (request.form.get("hamlet") or "").strip() or None

        vai_tro = (request.form.get("role") or "farmer").strip().lower()
        vai_tro = vai_tro if vai_tro in ("farmer", "officer") else "farmer"

        ma_can_bo = (request.form.get("staff_code") or "").strip()
        if vai_tro == "officer":
            if not ma_can_bo:
                return render_template("profile.html", user=u, error="Bạn chọn vai trò Cán bộ thì cần nhập Mã nhân viên.")
            if not staff_code_valid(ma_can_bo):
                return render_template("profile.html", user=u, error="Mã nhân viên không hợp lệ.")
            u.ma_can_bo = ma_can_bo.upper()
        else:
            u.ma_can_bo = None

        u.vai_tro = vai_tro

        try:
            db.session.commit()
        except Exception:
            db.session.rollback()
            return render_template("profile.html", user=u, error="Lỗi cập nhật. Vui lòng thử lại.")

        _refresh_flags_from_user(u)
        return redirect(url_for("predict_datapoint"))

    # ---------- Bio obs (officer) ----------
    @app.route("/bio-obs", methods=["GET"])
    @login_required
    @officer_required
    def bio_obs():
        return render_template("bio_obs.html")

    @app.route("/api/bio-obs/summary", methods=["GET"])
    @login_required
    @officer_required
    def bio_obs_summary():
        group = _parse_group(request.args.get("group"), allowed=("xa", "huyen"))
        metric = (request.args.get("metric") or "ph").strip().lower()
        days = _parse_days(request.args.get("days"))

        metric_map = {
            "nhiet_do": mau_nuoc.nhiet_do,
            "ph": mau_nuoc.ph,
            "do": mau_nuoc.do,
            "do_man": mau_nuoc.do_man,
            "do_trong": mau_nuoc.do_trong,
            "do_kiem": mau_nuoc.do_kiem,
            "no2": mau_nuoc.no2,
            "no3": mau_nuoc.no3,
            "nh4": mau_nuoc.nh4,
            "po43": mau_nuoc.po43,
            "cod": mau_nuoc.cod,
            "vibrio": mau_nuoc.vibrio_uoc_luong,
        }
        metric_col = metric_map.get(metric, mau_nuoc.ph)

        start_dt = datetime.now() - timedelta(days=days)
        key_col = mau_nuoc.xa if group == "xa" else mau_nuoc.huyen

        q = (
            db.session.query(
                key_col.label("key"),
                func.count(mau_nuoc.id).label("n"),
                func.avg(metric_col).label("avg_metric"),
                func.avg(mau_nuoc.nhiet_do).label("avg_nhiet_do"),
                func.avg(mau_nuoc.ph).label("avg_ph"),
                func.avg(mau_nuoc.do).label("avg_do"),
                func.avg(mau_nuoc.do_man).label("avg_do_man"),
                func.avg(mau_nuoc.do_trong).label("avg_do_trong"),
                func.avg(mau_nuoc.do_kiem).label("avg_do_kiem"),
                func.avg(mau_nuoc.no2).label("avg_no2"),
                func.avg(mau_nuoc.no3).label("avg_no3"),
                func.avg(mau_nuoc.nh4).label("avg_nh4"),
                func.avg(mau_nuoc.po43).label("avg_po43"),
                func.avg(mau_nuoc.cod).label("avg_cod"),
                func.avg(mau_nuoc.vibrio_uoc_luong).label("avg_vibrio"),
            )
            .filter(mau_nuoc.cap_nhat_luc >= start_dt)
            .group_by(key_col)
            .order_by(func.avg(metric_col).desc())
        )

        rows = q.all()
        labels, values, table = [], [], []

        for r in rows:
            k = (r.key or "Chưa khai báo")
            labels.append(k)
            values.append(float(r.avg_metric) if r.avg_metric is not None else None)
            table.append({
                "key": k,
                "n": int(r.n or 0),
                "avg_nhiet_do": float(r.avg_nhiet_do) if r.avg_nhiet_do is not None else None,
                "avg_ph": float(r.avg_ph) if r.avg_ph is not None else None,
                "avg_do": float(r.avg_do) if r.avg_do is not None else None,
                "avg_do_man": float(r.avg_do_man) if r.avg_do_man is not None else None,
                "avg_do_trong": float(r.avg_do_trong) if r.avg_do_trong is not None else None,
                "avg_do_kiem": float(r.avg_do_kiem) if r.avg_do_kiem is not None else None,
                "avg_no2": float(r.avg_no2) if r.avg_no2 is not None else None,
                "avg_no3": float(r.avg_no3) if r.avg_no3 is not None else None,
                "avg_nh4": float(r.avg_nh4) if r.avg_nh4 is not None else None,
                "avg_po43": float(r.avg_po43) if r.avg_po43 is not None else None,
                "avg_cod": float(r.avg_cod) if r.avg_cod is not None else None,
                "avg_vibrio": float(r.avg_vibrio) if r.avg_vibrio is not None else None,
            })

        return jsonify({"ok": True, "group": group, "metric": metric, "days": days, "labels": labels, "values": values, "table": table})

    # ---------- Latest ----------
    @app.route("/db/latest", methods=["GET"])
    @login_required
    def db_latest():
        latest = (
            mau_nuoc.query
            .filter_by(nguoi_dung_id=session.get("nguoi_dung_id"))
            .order_by(mau_nuoc.id.desc())
            .first()
        )
        if not latest:
            return {"ok": True, "latest": None}

        return {
            "ok": True,
            "latest": {
                "id": latest.id,
                "diem_quan_trac": latest.diem_quan_trac,
                "xa": latest.xa,
                "huyen": latest.huyen,
                "nhiet_do": latest.nhiet_do,
                "ph": latest.ph,
                "do": latest.do,
                "do_man": latest.do_man,
                "do_trong": latest.do_trong,
                "do_kiem": latest.do_kiem,
                "no2": latest.no2,
                "no3": latest.no3,
                "nh4": latest.nh4,
                "po43": latest.po43,
                "cod": latest.cod,
                "nam": latest.nam,
                "thang": latest.thang,
                "ngay": latest.ngay,
                "nguy_co_vibrio_text": latest.nguy_co_vibrio_text,
                "vibrio_uoc_luong": latest.vibrio_uoc_luong,
                "phu_hop_moi_truong_text": latest.phu_hop_moi_truong_text,
                "tao_thuc_an_text": latest.tao_thuc_an_text,
                "cap_nhat_luc": latest.cap_nhat_luc.isoformat() if latest.cap_nhat_luc else None,
            },
        }


app = create_app()

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(host="0.0.0.0", port=5000, debug=True)
