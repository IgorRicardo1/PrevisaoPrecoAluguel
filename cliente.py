import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import joblib
import os

model_path = 'modelo_treinado.pkl'
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    messagebox.showerror("Erro", f"O modelo '{model_path}' não foi encontrado. Por favor, execute o treino.py primeiro.")
    exit()

def escolher_arquivo():
    filepath = filedialog.askopenfilename(
        title="Selecione um arquivo CSV",
        filetypes=[("Arquivos CSV", "*.csv")]
    )
    
    if filepath:
        try:
            data = pd.read_csv(filepath)
            
            colunas_necessarias = ['tipo', 'area', 'quartos', 'bairro']
            if not all(col in data.columns for col in colunas_necessarias):
                messagebox.showerror("Erro", "O arquivo CSV deve conter as colunas obrigatórias: 'tipo', 'area', 'quartos' e 'bairro'.")
                return

            X_cliente = data[colunas_necessarias]
            previsoes = model.predict(X_cliente)
            
            data['preco_predito'] = previsoes

            resultado_texto = "Sucesso! Primeiras 5 previsões:\n\n" + data.head(5).to_string(index=False)
            caixa_texto.config(state='normal')
            caixa_texto.delete('1.0', tk.END)
            caixa_texto.insert('1.0', resultado_texto)
            caixa_texto.config(state='disabled')

            base, ext = os.path.splitext(filepath)
            save_path = base + '_com_previsoes.csv'
            data.to_csv(save_path, index=False)
            
            messagebox.showinfo("Concluído", f"As previsões foram salvas com sucesso em:\n{save_path}")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao processar o arquivo: {str(e)}")

root = tk.Tk()
root.title("Previsor de Preços de Aluguel")
root.geometry("600x350")

label = tk.Label(root, text="Escolha um arquivo CSV do cliente para prever os preços:")
label.pack(pady=10)

botao_escolher = tk.Button(root, text="Escolher arquivo e prever", command=escolher_arquivo)
botao_escolher.pack(pady=10)

caixa_texto = tk.Text(root, wrap='word', height=10, width=70)
caixa_texto.config(state='disabled')
caixa_texto.pack(pady=10)

root.mainloop()
