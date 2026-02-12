from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load trained CNN model
model = load_model("model.h5")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""
    confidence = ""
    img_path = ""

    if request.method == "POST":
        file = request.files["file"]
        if file:
            img_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(img_path)

            img = image.load_img(img_path, target_size=(64, 64))
            img_array = image.img_to_array(img)
            img_array = img_array.reshape(1, 64, 64, 3) / 255.0

            pred = model.predict(img_array)[0][0]

            if pred > 0.5:
                prediction = "Dog"
                confidence = f"{pred * 100:.2f}%"
            else:
                prediction = "Cat"
                confidence = f"{(1 - pred) * 100:.2f}%"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        img_path=img_path
    )

if __name__ == "__main__":
    app.run(debug=True)
