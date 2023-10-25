from flask import Flask, render_template, request
import pandas as pd
import joblib  # Para cargar el modelo de Random Forest
import numpy as np

app = Flask(__name__)
# Cargar el modelo de Random Forest previamente entrenado
rf_model = joblib.load('modelo_random_forest.pkl')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        estrato = float(request.form['estrato'])
        edad = float(request.form['edad'])
        genero = float(request.form['genero'])
        total_materias_perdidas = float(
            request.form['total_materias_perdidas'])
        ult_promedio_acad = float(request.form['ult_promedio_acad'])

        # Crear un arreglo NumPy con los valores de entrada
        input_data = np.array(
            [[estrato, edad, genero, total_materias_perdidas, ult_promedio_acad]])

        # Realizar la predicción con el modelo Random Forest
        prediction = rf_model.predict(input_data)

        # Interpretar la predicción
        if prediction[0] == 0:
            result = 'No Desierta'
        else:
            result = 'Desierta'

        return render_template('index.html', prediction_result=result)


if __name__ == '__main__':
    app.run(debug=True)
