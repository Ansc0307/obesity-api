from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# Cargar modelo y escalador
modelo = joblib.load('modelo_obesidad.pkl')
escalador = joblib.load('escalador.pkl')

@app.route('/', methods=['GET'])
def index():
    return jsonify({"mensaje": "API de predicción de obesidad"})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        valores = np.array([list(map(float, data.values()))])
        valores_escalados = escalador.transform(valores)
        pred = modelo.predict(valores_escalados)
        return jsonify({'prediccion': str(pred[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
