from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictionPipeline

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        # Hiện form nhập dữ liệu
        return render_template("home.html")

    else:
        # Lấy dữ liệu từ form POST
        diem_quan_trac = request.form.get("DIEM_QUAN_TRAC")
        xa = request.form.get("XA")
        huyen = request.form.get("HUYEN")

        nhiet_do = float(request.form.get("NHIET_DO"))
        ph = float(request.form.get("PH"))
        do = float(request.form.get("DO"))
        do_man = float(request.form.get("DO_MAN"))
        do_trong = float(request.form.get("DO_TRONG"))
        do_kiem = float(request.form.get("DO_KIEM"))
        no2 = float(request.form.get("NO2"))
        no3 = float(request.form.get("NO3"))
        nh4 = float(request.form.get("NH4"))
        po43 = float(request.form.get("PO43"))
        cod = float(request.form.get("COD"))

        nam = int(request.form.get("NAM"))
        thang = int(request.form.get("THANG"))
        ngay = int(request.form.get("NGAY"))

        # Tạo CustomData
        custom_data = CustomData(
            diem_quan_trac=diem_quan_trac,
            xa=xa,
            huyen=huyen,
            nhiet_do=nhiet_do,
            ph=ph,
            do=do,
            do_man=do_man,
            do_trong=do_trong,
            do_kiem=do_kiem,
            no2=no2,
            no3=no3,
            nh4=nh4,
            po43=po43,
            cod=cod,
            nam=nam,
            thang=thang,
            ngay=ngay,
        )

        input_df = custom_data.get_data_as_dataframe()

        # Gọi pipeline dự đoán
        pipeline = PredictionPipeline()
        result = pipeline.predict(input_df)

        # Trả kết quả ra lại trang home.html
        return render_template("home.html", results=result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
