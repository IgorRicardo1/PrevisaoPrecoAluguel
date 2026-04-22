import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('datasetoriginal.csv')

df.insert(0, 'id_imovel', df.index)

df_treino, df_cliente_real = train_test_split(df, test_size=0.2, random_state=42)

df_treino.to_csv('datasettreino.csv', index=False)

df_cliente_real.to_csv('datasetcliente_real.csv', index=False)

df_cliente_features = df_cliente_real.drop(columns=['preco'])
df_cliente_features.to_csv('datasetcliente.csv', index=False)

print("Dados separados com sucesso com a coluna 'id_imovel':")
print(f"- Treino: {len(df_treino)} registros salvos em 'datasettreino.csv'")
print(f"- Cliente (Para Previsão): {len(df_cliente_features)} registros salvos em 'datasetcliente.csv'")
print(f"- Cliente (Gabarito Real): {len(df_cliente_real)} registros salvos em 'datasetcliente_real.csv'")
