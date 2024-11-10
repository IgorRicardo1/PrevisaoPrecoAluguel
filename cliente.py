import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import joblib

# Carrega o modelo treinado
model_path = 'modelo_treinado.pkl'
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    messagebox.showerror("Erro", f"O modelo '{model_path}' não foi encontrado.")
    exit()

# Função para escolher arquivo CSV e fazer previsões
def escolher_arquivo():
    filepath = filedialog.askopenfilename(
        title="Selecione um arquivo CSV",
        filetypes=[("Arquivos CSV", "*.csv")]
    )
    
    if filepath:
        try:
            # Carregar o arquivo CSV
            data = pd.read_csv(filepath)
            if not all(col in data.columns for col in ['tipo', 'area', 'quartos', 'bairro']):
                messagebox.showerror("Erro", "O arquivo CSV deve conter as colunas: 'tipo', 'area', 'quartos' e 'bairro'.")
                return

            # Fazer previsões
            previsoes = model.predict(data[['tipo', 'area', 'quartos', 'bairro']])
            data['preco_predito'] = previsoes

            # Exibir as primeiras previsões em uma nova janela
            resultado_janela = tk.Toplevel(root)
            resultado_janela.title("Previsões de Preços")
            
            texto = tk.Text(resultado_janela, wrap='word')
            texto.insert('1.0', data.head(10).to_string(index=False))
            texto.config(state='disabled')
            texto.pack()

            # Salvar resultados em um novo arquivo CSV
            save_path = filepath.replace('.csv', '_com_previsoes.csv')
            data.to_csv(save_path, index=False)
            messagebox.showinfo("Concluído", f"As previsões foram salvas em '{save_path}'.")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao processar o arquivo: {str(e)}")

# Configuração da janela principal
root = tk.Tk()
root.title("Previsor de Preços de Aluguel")
root.geometry("400x200")

label = tk.Label(root, text="Escolha um arquivo CSV para prever os preços de aluguel:")
label.pack(pady=20)

botao_escolher = tk.Button(root, text="Escolher arquivo", command=escolher_arquivo)
botao_escolher.pack(pady=10)

root.mainloop()
