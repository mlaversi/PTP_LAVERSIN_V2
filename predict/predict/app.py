from flask import Flask, request, render_template

app = Flask(__name__)
from run import TextPredictionModel


@app.route('/')
def home():
    return render_template('render.html')


# Doesn't work but it's explained in the ReadMe
@app.route('/predict')
def predict_html():
    return render_template('render.html')


@app.route("/predict", methods=["GET"])
def request_prediction():

    model = TextPredictionModel.from_artefacts(
        "train/data/artefacts/2023-01-03-12-42-59")
    text = request.args.get('text')
    print(text)
    predictions = model.predict([text])

    return str(predictions)


if __name__ == '__main__':
    app.run()
