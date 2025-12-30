from flask import Blueprint, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash

from src.db.mysql import get_conn

auth_bp = Blueprint("auth", __name__)

@auth_bp.get("/register")
def register_page():
    return render_template("register.html")

@auth_bp.post("/register")
def register_submit():
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "").strip()
    full_name = request.form.get("full_name", "").strip()
    phone = request.form.get("phone", "").strip()
    province = request.form.get("province", "Cà Mau").strip()
    district = request.form.get("district", "").strip()
    commune = request.form.get("commune", "").strip()
    address_detail = request.form.get("address_detail", "").strip()

    if not username or not password:
        flash("Vui lòng nhập username và password.", "error")
        return redirect(url_for("auth.register_page"))

    pw_hash = generate_password_hash(password)

    conn = get_conn()
    cur = conn.cursor(dictionary=True)
    try:
        cur.execute(
            """
            INSERT INTO users (username, password_hash, full_name, phone, province, district, commune, address_detail)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (username, pw_hash, full_name, phone, province, district, commune, address_detail),
        )
        flash("Đăng ký thành công! Hãy đăng nhập.", "success")
        return redirect(url_for("auth.login_page"))
    except mysql.connector.errors.IntegrityError:
        flash("Username đã tồn tại. Hãy chọn username khác.", "error")
        return redirect(url_for("auth.register_page"))
    finally:
        cur.close()
        conn.close()

@auth_bp.get("/login")
def login_page():
    return render_template("login.html")

@auth_bp.post("/login")
def login_submit():
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "").strip()

    conn = get_conn()
    cur = conn.cursor(dictionary=True)
    try:
        cur.execute("SELECT * FROM users WHERE username=%s", (username,))
        user = cur.fetchone()
    finally:
        cur.close()
        conn.close()

    if not user or not check_password_hash(user["password_hash"], password):
        flash("Sai username hoặc password.", "error")
        return redirect(url_for("auth.login_page"))

    session.clear()
    session["user_id"] = user["id"]
    session["role"] = user["role"]  # farmer/admin
    flash("Đăng nhập thành công.", "success")
    return redirect(url_for("home"))

@auth_bp.get("/logout")
def logout():
    session.clear()
    flash("Bạn đã đăng xuất.", "success")
    return redirect(url_for("auth.login_page"))
