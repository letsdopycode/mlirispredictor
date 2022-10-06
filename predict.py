from flask import*
import pickle
import os

#initialize app
app = Flask(__name__)

#load prediction model
model = pickle.load(open("flowerpredictor.pkl", "rb"))


# image directory
app.config["UPLOAD_FOLDER"] = "static"


@app.route("/", methods=["POST", "GET"])
def pred():
  if request.method == "POST":
    sepal_length = request.form.get("sl")
    sepal_width = request.form.get("sw")
    petal_length = request.form.get("pl")
    petal_width = request.form.get("pw")
  # p = model.predict([[4.9, 3.0, 1.5, 0.5]])

    print(sepal_length, sepal_width, petal_length, petal_width)
    l = [sepal_length, sepal_width, petal_length, petal_width]
    data = [float(i) for i in l]
    p = model.predict([data])
    # 0-setosa
    # 1-versicolor
    # 2-virginia
    flowers = ["setosa", "virgicolor", "virginia"]

    # return "iris"+" "+flowers[p[0]]
    pr = flowers[p[0]]
    l = []
    l.append(pr)
    if pr == "setosa":
        i = os.path.join(app.config["UPLOAD_FOLDER"], "setosa.jpg")
        l.append(i)
    if pr == "virgicolor":
        i = os.path.join(app.config["UPLOAD_FOLDER"], "virgicolor.jpg")
        l.append(i)
    else:
        i = os.path.join(app.config["UPLOAD_FOLDER"], "virginia.jpg")
        l.append(i)
    return render_template("output.html", result=l)
  return render_template("inputdata.html")


if __name__ == "__main__":
    app.run(debug=True)
