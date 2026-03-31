import markdown
import os
from flask import Flask, render_template, request
from agent.plant_agent import plant_agent   
app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    filename = None
    message = None

    if request.method == "POST":
        file = request.files["image"]

        if file:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            result = plant_agent(filepath)
            filename = file.filename

            if "message" in result:
                message = result["message"]
                result = None
            else:
                raw_text = result["treatment"]
                result["treatment"] = markdown.markdown(raw_text)

    return render_template(
        "index.html",
        result=result,
        filename=filename,
        message=message
    )


if __name__ == "__main__":
    app.run(debug=True)
    