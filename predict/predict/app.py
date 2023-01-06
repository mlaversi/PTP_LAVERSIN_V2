from flask import Flask, request, render_template

app = Flask(__name__)
from run import TextPredictionModel




# Doesn't work but it's explained in the ReadMe

@app.route("/", methods=['POST', 'GET'])
def index():

    if request.method == "POST":
        question = request.form["question"]
        print(question)
        path = "C:/Users/mathi/OneDrive/Bureau/poc-to-prod-capstone/train/data/artefacts/2023-01-03-12-42-59" #Advice put your absolute path :)
        prediction_object = TextPredictionModel.from_artefacts(path)
        predic = prediction_object.predict([question], top_k=1)
        print("The prediction without the top_k", predic)
        return (str(predic))
    return render_template("render.html")

if __name__ == '__main__':
    app.run()
