import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report

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

# Criar pipeline através do Logistic Regression
from sklearn.linear_model import LogisticRegression

clf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

# ----------------------------------------------------
# Treinamento e Avaliação do Modelo de Classificação
# ----------------------------------------------------

# Treinar o modelo de classificação
clf_pipeline.fit(X_train, y_train_class)

# Fazer previsões
y_pred = clf_pipeline.predict(X_test)

# Avaliar com matriz de confusão
cm = confusion_matrix(y_test_class, y_pred, labels=["good", "bad"])
print("Matriz de confusão:\n", cm)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["good", "bad"], yticklabels=["good", "bad"])
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.title("Matriz de Confusão - Classificação de Risco")
plt.show()

# Relatório de métricas
print("\n Relatório de classificação:")
print(classification_report(y_test_class, y_pred))

