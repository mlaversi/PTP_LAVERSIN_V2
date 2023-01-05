from flask import Flask, request, render_template
app = Flask(__name__)
from run import TextPredictionModel
@app.route('/')
def home():
    return render_template('render.html')


@app.route("/predict", methods=['POST'])
def get_prediction():
    model = TextPredictionModel.from_artefacts("train/data/artefacts/train/2023-01-03-22-59-05")
    text = request.form['text']
    predictions = model.predict([text], top_k=3)

    return str(predictions)

if __name__ == '__main__':
    app.run()

