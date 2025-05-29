from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Cargar modelo y escalador
modelo = joblib.load('modelo_obesidad.pkl')
escalador = joblib.load('escalador.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # dict con los valores
    valores = np.array([list(data.values())])
    valores_escalados = escalador.transform(valores)
    pred = modelo.predict(valores_escalados)
    return jsonify({'prediccion': pred[0]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
