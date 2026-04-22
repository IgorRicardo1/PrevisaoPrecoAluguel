import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  

sns.set(style="whitegrid")
plt.rc("figure", figsize=(10, 6))

data = pd.read_csv('datasettreino.csv')

X_train = data[['tipo', 'area', 'quartos', 'bairro']]
y_train = data['preco']

categorical_features = ['tipo', 'bairro']
numerical_features = ['area', 'quartos']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', 'passthrough', numerical_features)
    ])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

param_grid = {
    'regressor__n_estimators': [200, 300],
    'regressor__max_depth': [None, 10, 20],
}

print("Iniciando treinamento avançado com busca de hiperparâmetros...")
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

print(f"Melhor combinação de parâmetros: {grid_search.best_params_}")

best_model = grid_search.best_estimator_

y_pred_cv = cross_val_predict(best_model, X_train, y_train, cv=5)

rmse = np.sqrt(mean_squared_error(y_train, y_pred_cv))
mae = mean_absolute_error(y_train, y_pred_cv)
r2 = r2_score(y_train, y_pred_cv)

print("\nMétricas de Validação Cruzada (Treino):")
print(f"Erro Quadrático Médio (RMSE): {rmse:.2f}")
print(f"Erro Absoluto Médio (MAE): {mae:.2f}")
print(f"Coeficiente de Determinação (R2): {r2:.2f}")

joblib.dump(best_model, 'modelo_treinado.pkl')
print("\nModelo treinado salvo como 'modelo_treinado.pkl'")

sns.regplot(x=y_train, y=y_pred_cv, scatter_kws={'alpha':0.6}, line_kws={'color': 'red'})
plt.xlabel("Preço Real")
plt.ylabel("Preço Predito")
plt.title("Comparação entre Preço Real e Predito (Treino Profissional)")
plt.savefig('grafico_treino.png')
print("Gráfico salvo como 'grafico_treino.png'")
