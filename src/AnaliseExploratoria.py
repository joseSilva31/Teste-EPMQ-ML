import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar dataset
df = pd.read_csv("data/DatasetCredit-g.csv")

# -------------------------------
# Análise Exploratória de Dados
# -------------------------------

# Estrutura do dataset
print("\n Dimensão do dataset:", df.shape)
print("\n Colunas do dataset:", df.columns.tolist())

# Tipos de variáveis
print("\n Tipos de dados:\n", df.dtypes)

# Valores Nulos
print("\n Valores Nulos:\n", df.isnull().sum())

# Estatísticas descritivas
print("\n Estatísticas numéricas:\n", df.describe())
print("\n Estatísticas categóricas:\n", df.describe(include=["object"]))

# Distribuição da variável target
print("\n Distribuição da variável 'class':\n", df['class'].value_counts())

sns.countplot(x='class', data=df)
plt.title("Distribuição da variável alvo (Risco)")
plt.show()

# Distribuição de variáveis numéricas
num_cols = ['duration', 'credit_amount', 'age']

df[num_cols].hist(bins=20, figsize=(10, 5))
plt.suptitle("Distribuição das variáveis numéricas")
plt.show()

# Correlação entre variáveis numéricas
corr = df[num_cols].corr()
print("\n Correlação entre variáveis numéricas:\n", corr)

sns.heatmap(corr, annot=True, cmap="Blues")
plt.title("Matriz de correlação (numéricas)")
plt.show()
