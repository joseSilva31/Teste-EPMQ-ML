import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

# Carregar o dataset
df = pd.read_csv("data/DatasetCredit-g.csv")

# Separar as features e os targets
X = df.drop(columns=['class', 'credit_amount'])  # features
y_class = df['class']                            # target de classificação
y_reg = df['credit_amount']                      # target de regressão

# Identificar as colunas numéricas e categóricas
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include=['object']).columns

print("Numéricas:", num_cols)
print("Categóricas:", cat_cols)

# Pré-processamento
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),      # Escalar numéricas
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)  # Codificar categóricas
    ]
)

# Dividir os dados de treino e de teste
X_train, X_test, y_train_class, y_test_class = train_test_split(
    X, y_class, test_size=0.3, random_state=42, stratify=y_class
)

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X, y_reg, test_size=0.3, random_state=42
)

print("\n-----------------------------------------------")
print("\nModelos de Classificação - Logistic Regression")
print("\n-----------------------------------------------")

# Criar pipeline
clf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Treinar e fazer previsões com o Logistic Regression
clf_pipeline.fit(X_train, y_train_class)
y_pred_log = clf_pipeline.predict(X_test)

# Avaliar com matriz de confusão
cm_log = confusion_matrix(y_test_class, y_pred_log, labels=["good", "bad"])
print("Matriz de confusão:\n", cm_log)

sns.heatmap(cm_log, annot=True, fmt="d", cmap="Blues", xticklabels=["good", "bad"], yticklabels=["good", "bad"])
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.title("Logistic Regression - Matriz de Confusão")
plt.savefig("resultados/matriz_confusao_logReg.png", dpi=300, bbox_inches="tight")
plt.close()

# Relatório de métricas
print("\n Relatório de classificação - Logistic Regression:")
print(classification_report(y_test_class, y_pred_log))

print("\n-----------------------------------------------")
print("\nModelos de Classificação - Random Forest")
print("\n-----------------------------------------------")

# Criar pipeline
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Treinar e fazer previsões com o Random Forest
rf_pipeline.fit(X_train, y_train_class)
y_pred_rf = rf_pipeline.predict(X_test)

# Avaliar com matriz de confusão
cm_rf = confusion_matrix(y_test_class, y_pred_rf, labels=["good", "bad"])
print("Matriz de confusão:\n", cm_rf)

sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Greens", xticklabels=["good","bad"], yticklabels=["good","bad"])
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.title("Random Forest - Matriz de Confusão")
plt.savefig("resultados/matriz_confusao_rf.png", dpi=300, bbox_inches="tight")
plt.close()

# Relatório de métricas
print("\n Relatório de classificação - Random Forest:")
print(classification_report(y_test_class, y_pred_rf))

# Previsões no conjunto de teste
df_resultados = pd.DataFrame({
    "Real": y_test_class,
    "Previsto_Logistic": y_pred_log,
    "Previsto_RF": y_pred_rf,
})

# Exportar para CSV
df_resultados.to_csv("data/ResultadosClassificacao.csv", index=False)

print("\n----------------")
print("\nCalcular lucro")
print("\n----------------")

def calcular_lucro(cm):
    # cm = Matriz do Enunciado {[1,1] = 0; [1,2] = -200; [2,1] = -200; [2,2] = 100}
    lucro = (cm[0,0] * 0) + (cm[0,1] * -200) + (cm[1,0] * -200) + (cm[1,1] * 100)
    return lucro

print("\n Lucro Logistic Regression:", calcular_lucro(cm_log))
print(" Lucro Random Forest:", calcular_lucro(cm_rf))



print("\n-----------------------------------------------")
print("\nModelos de Regressão - Previsão do Credit Amount")
print("\n-----------------------------------------------")

# -----------------------------------------------)
# Regressão Linear
# -----------------------------------------------)

reg_lin_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

reg_lin_pipeline.fit(X_train_reg, y_train_reg)
y_pred_lin = reg_lin_pipeline.predict(X_test_reg)

print("\nMétricas Regressão Linear:")
print("MAE:", mean_absolute_error(y_test_reg, y_pred_lin))
print("RMSE:", mean_squared_error(y_test_reg, y_pred_lin))

# -----------------------------------------------
# Random Forest Regressor
# -----------------------------------------------

reg_rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

reg_rf_pipeline.fit(X_train_reg, y_train_reg)
y_pred_rf_reg = reg_rf_pipeline.predict(X_test_reg)

print("\nMétricas Random Forest Regressor:")
print("MAE:", mean_absolute_error(y_test_reg, y_pred_rf_reg))
print("RMSE:", mean_squared_error(y_test_reg, y_pred_rf_reg))

# -----------------------------------------------
# Gradient Boosting Regressor
# -----------------------------------------------

reg_gb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(random_state=42))
])

reg_gb_pipeline.fit(X_train_reg, y_train_reg)
y_pred_gb_reg = reg_gb_pipeline.predict(X_test_reg)

print("\nMétricas Gradient Boosting Regressor:")
print("MAE:", mean_absolute_error(y_test_reg, y_pred_gb_reg))
print("RMSE:", mean_squared_error(y_test_reg, y_pred_gb_reg))

# -----------------------------------------------
# XGBoost 
# -----------------------------------------------

xgb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(random_state=42, objective='reg:squarederror'))
])

xgb_pipeline.fit(X_train_reg, y_train_reg)
y_pred_xgb = xgb_pipeline.predict(X_test_reg)

print("\nMétricas XGBoost:")
print("MAE:", mean_absolute_error(y_test_reg, y_pred_xgb))
print("RMSE:", mean_squared_error(y_test_reg, y_pred_xgb))


# -----------------------------------------------
# Calculo do Lucro conforme o enunciado
# -----------------------------------------------

# lucro = 100 * (|Y - P| / Y < 0.3)
def calcular_lucro_reg(y_true, y_pred):
    acertos = abs(y_true - y_pred) / y_true < 0.3
    return acertos.sum() * 100

print("\n-----------------------------------------------")
print("\nCalcular lucro Regressão")
print("\n-----------------------------------------------")
print(" Lucro Regressão Linear:", calcular_lucro_reg(y_test_reg, y_pred_lin))
print(" Lucro Random Forest Regressor:", calcular_lucro_reg(y_test_reg, y_pred_rf_reg))
print(" Lucro Gradient Boosting Regressor:", calcular_lucro_reg(y_test_reg, y_pred_gb_reg))
print(" Lucro XGBoost:", calcular_lucro_reg(y_test_reg, y_pred_xgb))

# --------------------------------------------------------------------
# Importância de cada atributo para os modelos com melhores resultados
# --------------------------------------------------------------------

print("\n-----------------------------------------------------------")
print("Importância de cada atributo através do feature importances")
print("-------------------------------------------------------------")

# Para o Random Forest Classifier
rf_model = rf_pipeline.named_steps['classifier']
rf_preproc = rf_pipeline.named_steps['preprocessor']

# Obter nomes das features após o OneHotEncoding
ohe = rf_preproc.named_transformers_['cat']
encoded_cat_cols = ohe.get_feature_names_out(cat_cols)
all_feature_names = list(num_cols) + list(encoded_cat_cols)

# Importância das features no Random Forest
fi_rf = rf_model.feature_importances_
fi_rf_df = pd.DataFrame({'feature': all_feature_names, 'importance': fi_rf})
fi_rf_df = fi_rf_df.sort_values('importance', ascending=False).head(20)

plt.figure(figsize=(8,6))
sns.barplot(x='importance', y='feature', data=fi_rf_df)
plt.title('Feature Importances - RandomForestClassifier')
plt.tight_layout()
plt.savefig("resultados/feature_importances_rf.png", dpi=300, bbox_inches="tight")
plt.close()

# Para o Gradient Boosting Regressor
gb_model = reg_gb_pipeline.named_steps['regressor']
gb_preproc = reg_gb_pipeline.named_steps['preprocessor']

ohe_reg = gb_preproc.named_transformers_['cat']
encoded_cat_cols_reg = ohe_reg.get_feature_names_out(cat_cols)
all_feature_names_reg = list(num_cols) + list(encoded_cat_cols_reg)

fi_gb = gb_model.feature_importances_
fi_gb_df = pd.DataFrame({'feature': all_feature_names_reg, 'importance': fi_gb})
fi_gb_df = fi_gb_df.sort_values('importance', ascending=False).head(20)

plt.figure(figsize=(8,6))
sns.barplot(x='importance', y='feature', data=fi_gb_df)
plt.title('Feature Importances - GradientBoostingRegressor')
plt.tight_layout()
plt.savefig("resultados/feature_importances_gb.png", dpi=300, bbox_inches="tight")
plt.close()




