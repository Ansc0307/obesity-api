# API Predictiva para Obesidad (Modelo en Flask)

### ¿Qué hace?
Esta API recibe datos desde una app Flutter y devuelve la predicción del modelo entrenado (`modelo_obesidad.pkl`) sobre el nivel de obesidad.

### Archivos incluidos:
- `api.py`: Código de la API.
- `requirements.txt`: Librerías necesarias.
- `modelo_obesidad.pkl`: Tu modelo entrenado.
- `escalador.pkl`: Escalador usado en el preprocesamiento.

### ¿Cómo desplegarlo en Render?
1. Sube todos los archivos a un repositorio de GitHub.
2. Entra a [https://render.com](https://render.com).
3. Crea un nuevo servicio "Web Service" desde tu repo.
4. Selecciona:
   - Start command: `python api.py`
   - Environment: Python 3.10 o superior
   - Build command: `pip install -r requirements.txt`

### URL de la API
Una vez desplegado, la API estará accesible en algo como:
```
https://mi-api-obesidad.onrender.com/predict
```

Envía un JSON con las entradas del usuario y te devuelve la predicción.
