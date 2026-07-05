import os
import time

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    send_from_directory
)

from werkzeug.utils import secure_filename

from predict import FruitPredictor

# ==========================================
# Flask Config
# ==========================================

app = Flask(__name__)
app.secret_key = "fruitvision_ai"

UPLOAD_FOLDER = "uploads"
MODEL_PATH = "model/best_xception.keras"
CLASS_JSON = "class_names.json"

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ==========================================
# Load Model
# ==========================================

predictor = FruitPredictor(
    model_path=MODEL_PATH,
    class_json=CLASS_JSON
)

# ==========================================
# Helper
# ==========================================


def allowed_file(filename):
    return (
        "." in filename and
        filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )

# ==========================================
# Home
# ==========================================


@app.route("/")
def home():
    return render_template("index.html")

# ==========================================
# About
# ==========================================


@app.route("/about")
def about():
    return render_template("about.html")

# ==========================================
# Predict
# ==========================================


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        flash("Silakan pilih gambar.")
        return redirect(url_for("home"))

    file = request.files["image"]

    if file.filename == "":
        flash("Silakan pilih gambar.")
        return redirect(url_for("home"))

    if not allowed_file(file.filename):
        flash("Format file tidak didukung.")
        return redirect(url_for("home"))

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    file.save(filepath)

    start = time.perf_counter()

    result = predictor.predict(filepath)

    inference_time = round(
        time.perf_counter() - start,
        4
    )

    return render_template(
        "result.html",
        image=filename,
        prediction=result["prediction"],
        confidence=round(result["confidence"] * 100, 2),
        top3=result["top3"],
        inference_time=inference_time
    )

# ==========================================
# Upload Folder
# ==========================================


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename
    )

# ==========================================
# Run
# ==========================================


if __name__ == "__main__":
    app.run(
        debug=True
    )
