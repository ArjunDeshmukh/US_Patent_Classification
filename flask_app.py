from flask import Flask, render_template, request
from trained_model_inference import inference

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home_abstract_cpc_subclass.html')


@app.route('/predict', methods=['POST'])
def predict():
    abstract = request.form['abstract']
    CPC_code = inference(abstract)
    #title = request.form['title']
    #CPC_code = inference(title)

    # return render_template('cpc_subclass_pred.html', CPC_code=CPC_code)
    return render_template('cpc_section_pred.html', CPC_code=CPC_code)


if __name__ == '__main__':
    #app.run(host='0.0.0.0', port=8080)
    app.run(debug=True)
