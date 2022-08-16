from flask import Flask, render_template, request
from model_inference import inference

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    abstract = request.form['abstract']
    CPC_code = inference(abstract)

    return render_template('cpc_pred.html', CPC_code=CPC_code)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
