# src/models/db_models.py
from __future__ import annotations

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class nguoi_dung(db.Model):
    __tablename__ = "nguoi_dung"

    id = db.Column(db.Integer, primary_key=True)

    ten_dang_nhap = db.Column(db.String(60), unique=True, nullable=False)
    mat_khau_hash = db.Column(db.String(255), nullable=False)

    ho_ten = db.Column(db.String(120), nullable=True)
    so_dien_thoai = db.Column(db.String(30), nullable=True)

    tinh = db.Column(db.String(60), nullable=True)
    huyen = db.Column(db.String(60), nullable=True)
    xa = db.Column(db.String(60), nullable=True)

    biet_danh = db.Column(db.String(80), nullable=True)
    ap = db.Column(db.String(80), nullable=True)

    # farmer | officer
    vai_tro = db.Column(db.String(20), nullable=False, server_default="farmer")
    ma_can_bo = db.Column(db.String(50), nullable=True)

    tao_luc = db.Column(db.DateTime, server_default=db.func.now())
    cap_nhat_luc = db.Column(db.DateTime, server_default=db.func.now(), onupdate=db.func.now())


class mau_nuoc(db.Model):
    __tablename__ = "mau_nuoc"

    id = db.Column(db.Integer, primary_key=True)
    nguoi_dung_id = db.Column(db.Integer, db.ForeignKey("nguoi_dung.id"), index=True, nullable=False)

    diem_quan_trac = db.Column(db.String(100))
    xa = db.Column(db.String(100))
    huyen = db.Column(db.String(100))

    nhiet_do = db.Column(db.Float)
    ph = db.Column(db.Float)
    do = db.Column(db.Float)

    do_man = db.Column(db.Float)
    do_trong = db.Column(db.Float)
    do_kiem = db.Column(db.Float)

    no2 = db.Column(db.Float)
    no3 = db.Column(db.Float)
    nh4 = db.Column(db.Float)
    po43 = db.Column(db.Float)
    cod = db.Column(db.Float)

    nam = db.Column(db.Integer)
    thang = db.Column(db.Integer)
    ngay = db.Column(db.Integer)

    # kết quả dự đoán
    nguy_co_vibrio_text = db.Column(db.String(50))
    vibrio_uoc_luong = db.Column(db.Float, nullable=True)
    phu_hop_moi_truong_text = db.Column(db.String(50))
    tao_thuc_an_text = db.Column(db.String(50))

    cap_nhat_luc = db.Column(db.DateTime, server_default=db.func.now(), onupdate=db.func.now())


class tin_nhan_chat(db.Model):
    __tablename__ = "tin_nhan_chat"

    id = db.Column(db.Integer, primary_key=True)
    nguoi_dung_id = db.Column(db.Integer, db.ForeignKey("nguoi_dung.id"), index=True, nullable=True)

    session_id = db.Column(db.String(100), index=True)
    vai = db.Column(db.String(20))  # user | assistant
    noi_dung = db.Column(db.Text)

    cap_nhat_luc = db.Column(db.DateTime, server_default=db.func.now(), onupdate=db.func.now())


class thu_hoach(db.Model):
    __tablename__ = "thu_hoach"

    id = db.Column(db.Integer, primary_key=True)
    nguoi_dung_id = db.Column(db.Integer, db.ForeignKey("nguoi_dung.id"), index=True, nullable=False)

    ngay_thu_hoach = db.Column(db.Date, nullable=False, index=True)

    kg_tom_su = db.Column(db.Float, default=0)
    kg_tom_the = db.Column(db.Float, default=0)
    kg_tom_bac = db.Column(db.Float, default=0)

    co_tom_su_con_kg = db.Column(db.Integer, nullable=True)  # con/kg
    ghi_chu = db.Column(db.String(255), nullable=True)

    tao_luc = db.Column(db.DateTime, server_default=db.func.now())


class thong_bao(db.Model):
    __tablename__ = "thong_bao"

    id = db.Column(db.Integer, primary_key=True)

    nguoi_tao_id = db.Column(db.Integer, db.ForeignKey("nguoi_dung.id"), index=True, nullable=False)

    scope_type = db.Column(db.String(20), nullable=False, server_default="commune")
    scope_value = db.Column(db.String(100), nullable=False, index=True)

    target_user_id = db.Column(db.Integer, db.ForeignKey("nguoi_dung.id"), index=True, nullable=True)

    tieu_de = db.Column(db.String(150), nullable=False)
    noi_dung = db.Column(db.Text, nullable=False)

    muc_do = db.Column(db.String(20), nullable=False, server_default="info")

    mau_nuoc_id = db.Column(db.Integer, db.ForeignKey("mau_nuoc.id"), index=True, nullable=True)

    tao_luc = db.Column(db.DateTime, server_default=db.func.now(), index=True)


class da_doc_thong_bao(db.Model):
    __tablename__ = "da_doc_thong_bao"

    id = db.Column(db.Integer, primary_key=True)

    thong_bao_id = db.Column(db.Integer, db.ForeignKey("thong_bao.id"), index=True, nullable=False)
    nguoi_dung_id = db.Column(db.Integer, db.ForeignKey("nguoi_dung.id"), index=True, nullable=False)

    doc_luc = db.Column(db.DateTime, server_default=db.func.now())

    __table_args__ = (
        db.UniqueConstraint("thong_bao_id", "nguoi_dung_id", name="uq_read_notice"),
    )

