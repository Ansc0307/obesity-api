from flask import Flask, request, jsonify
from flask_cors import CORS  # 👈 Importar
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)  # 👈 Habilitar CORS para todos los orígenes

# Cargar modelo y escalador
modelo = joblib.load('modelo_obesidad.pkl')
escalador = joblib.load('escalador.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    valores = np.array([list(data.values())])
    valores_escalados = escalador.transform(valores)
    pred = modelo.predict(valores_escalados)
    return jsonify({'prediccion': pred[0]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
