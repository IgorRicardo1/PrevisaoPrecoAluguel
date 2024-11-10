import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score  # Corrigido com importação do cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  

sns.set(style="whitegrid")
plt.rc("figure", figsize=(10, 6))

# Carregamento dos dados
data = pd.read_csv('datasettreino.csv', header=None, names=['preco', 'tipo', 'area', 'quartos', 'bairro'], skiprows=1)
print(data.head())
X = data[['tipo', 'area', 'quartos', 'bairro']]
y = data['preco']

# Divisão dos dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Criação do pipeline
pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()), 
    ('regressor', RandomForestRegressor())
])

# Parâmetros para GridSearchCV
param_grid = {
    'regressor__n_estimators': [200, 300, 500],
    'regressor__max_depth': [None, 10, 20, 30],
}

# Busca por hiperparâmetros
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
print(f"Melhor combinação de parâmetros: {grid_search.best_params_}")

# Melhor modelo
best_model = grid_search.best_estimator_

# Validação cruzada para avaliação do modelo com o melhor modelo
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
print(f"MSE Médio com Validação Cruzada: {-np.mean(cv_scores)}")

# Treinamento e predição com o melhor modelo
y_pred = best_model.predict(X_test)

# Cálculo das métricas
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Erro Quadrático Médio: {rmse}")
print(f"Erro Absoluto Médio: {mae}")
print(f"Coeficiente de Determinação: {r2}")

# Salvar o modelo treinado
joblib.dump(best_model, 'modelo_treinado.pkl')
print("Modelo treinado salvo como 'modelo_treinado.pkl'")

# Gráfico de comparação entre preço real e preço previsto
sns.regplot(x=y_test, y=y_pred, scatter_kws={'alpha':0.6}, line_kws={'color': 'red'})
plt.xlabel("Preço Real")
plt.ylabel("Preço Predito")
plt.title("Comparação entre Preço Real e Predito Com Os Dados Testes")
plt.show()
