from functools import wraps
from flask import session, redirect, url_for

def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not session.get("user_id"):
            return redirect(url_for("auth.login_page"))
        return fn(*args, **kwargs)
    return wrapper
