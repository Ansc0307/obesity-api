from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# Cargar modelo y escalador
modelo = joblib.load('modelo_obesidad.pkl')
escalador = joblib.load('escalador.pkl')

# Definir el orden esperado de las características
expected_features = ['Age', 'Gender', 'Weight', 'CALC', 'FAVC', 'FCVC', 'NCP', 'SCC', 'SMOKE', 'CH2O', 'FAF', 'TUE', 'MTRANS']

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # Validar que el JSON no esté vacío y que contenga todas las claves necesarias
    if not data:
        return jsonify({'error': 'No se recibieron datos JSON'}), 400

    missing = [feat for feat in expected_features if feat not in data]
    if missing:
        return jsonify({'error': f'Faltan campos en el JSON: {missing}'}), 400

    try:
        # Extraer valores en el orden correcto
        valores = np.array([[data[feat] for feat in expected_features]])

        # Aplicar escalado y predecir
        valores_escalados = escalador.transform(valores)
        pred = modelo.predict(valores_escalados)

        return jsonify({'prediccion': pred[0]})

    except Exception as e:
        # Retornar error con mensaje para facilitar depuración
        return jsonify({'error': f'Error al procesar la predicción: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
