import pandas as pd

df = pd.read_csv('datasetoriginal.csv')

df1 = df.iloc[::2] 
df2 = df.iloc[1::2] 


df1.to_csv('datasettreino.csv', index=False)
df2.to_csv('datasetcliente.csv', index=False)
