from flask import*
import os

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = "static"

# @app.route("/", methods=["GET", "POST"])


@app.route("/")
def F():
   # if request.method == "POST":
    i = os.path.join(app.config["UPLOAD_FOLDER"], "setosa.jpg")
    return render_template("img.html", result=i)
   # return render_template("img.html")


if __name__ == "__main__":
    app.run(debug=True)
