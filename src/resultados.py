# Guardar resultados num dataframe
import pandas as pd


resultados = []

# Classificação
resultados.append({
    "Tipo": "Classificação",
    "Modelo": "Logistic Regression",
    "Accuracy": 0.74, 
    "Precision_good": 0.79,
    "Recall_good": 0.85,
    "F1_good": 0.82,
    "Lucro": -11300
})

resultados.append({
    "Tipo": "Classificação",
    "Modelo": "Random Forest",
    "Accuracy": 0.75,
    "Precision_good": 0.78,
    "Recall_good": 0.90,
    "F1_good": 0.83,
    "Lucro": -11300
})

# Regressão
resultados.append({
    "Tipo": "Regressão",
    "Modelo": "Linear Regression",
    "MAE": 1201.63,
    "RMSE": 2901801.76,
    "Lucro": 12100
})

resultados.append({
    "Tipo": "Regressão",
    "Modelo": "Random Forest",
    "MAE": 1129.02,
    "RMSE": 2735992.09,
    "Lucro": 13400
})

resultados.append({
    "Tipo": "Regressão",
    "Modelo": "Gradient Boosting",
    "MAE": 1081.48,
    "RMSE": 2641612.56,
    "Lucro": 15800
})

resultados.append({
    "Tipo": "Regressão",
    "Modelo": "XGBoost",
    "MAE": 1220.60,
    "RMSE": 3054055.43,
    "Lucro": 13000
})

# Converter em DataFrame e guardar
df_resultados = pd.DataFrame(resultados)
df_resultados.to_csv("data/ResultadosFinais.csv", index=False)

print("\nResultados guardados")