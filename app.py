from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

# Load the saved model
model = load_model("model.h5")

@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    if request.method == "POST":
        f = request.files["file"]
        filepath = os.path.join("uploads", f.filename)
        f.save(filepath)

        img = image.load_img(filepath, target_size=(100, 100))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0) / 255.0

        pred = model.predict(x)
        if pred[0][0] > 0.5:
            result = "Cat"
        else:
            result = "Dog"
        
    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
