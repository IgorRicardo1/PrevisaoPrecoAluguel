import pandas as pd
from tkinter import filedialog, messagebox
import tkinter as tk
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def comparar_arquivos():
    filepath_real = filedialog.askopenfilename(
        title="Selecione o arquivo CSV com os preços reais (datasetcliente_real.csv)",
        filetypes=[("Arquivos CSV", "*.csv")]
    )
    if not filepath_real:
        return

    filepath_predito = filedialog.askopenfilename(
        title="Selecione o arquivo CSV com os preços preditos",
        filetypes=[("Arquivos CSV", "*.csv")]
    )
    if not filepath_predito:
        return

    try:
        data_real = pd.read_csv(filepath_real)
        data_predito = pd.read_csv(filepath_predito)

        if 'id_imovel' not in data_real.columns or 'id_imovel' not in data_predito.columns:
            messagebox.showerror("Erro Crítico", "Os arquivos não possuem a coluna 'id_imovel'. É impossível garantir um cruzamento seguro sem ela.")
            return

        if 'preco' not in data_real.columns:
            messagebox.showerror("Erro", "O arquivo de gabarito real não possui a coluna 'preco'.")
            return
        if 'preco_predito' not in data_predito.columns:
            messagebox.showerror("Erro", "O arquivo de predições não possui a coluna 'preco_predito'.")
            return

        data_merged = pd.merge(data_real[['id_imovel', 'preco']], 
                               data_predito[['id_imovel', 'preco_predito']], 
                               on='id_imovel', 
                               how='inner')
                               
        if len(data_merged) == 0:
            messagebox.showerror("Erro", "Nenhum id_imovel corresponde entre os dois arquivos!")
            return

        preco_real = data_merged['preco']
        preco_predito = data_merged['preco_predito']

        mae = mean_absolute_error(preco_real, preco_predito)
        rmse = np.sqrt(mean_squared_error(preco_real, preco_predito))
        r2 = r2_score(preco_real, preco_predito)
        
        resultado_texto = (
            f"Métricas de Validação Perfeitas (Baseadas em ID):\n\n"
            f"Imóveis Avaliados: {len(data_merged)}\n"
            f"Erro Absoluto Médio (MAE): {mae:.2f}\n"
            f"Erro Quadrático Médio (RMSE): {rmse:.2f}\n"
            f"Coeficiente de Determinação (R2): {r2:.2f}"
        )
        
        caixa_texto.config(state='normal')
        caixa_texto.delete('1.0', tk.END)
        caixa_texto.insert('1.0', resultado_texto)
        caixa_texto.config(state='disabled')
        
    except Exception as e:
        messagebox.showerror("Erro", f"Erro ao processar os arquivos: {str(e)}")

root = tk.Tk()
root.title("Comparador de CSV de Preços de Aluguel (À Prova de Falhas)")
root.geometry("600x350")

label = tk.Label(root, text="Selecione o arquivo de Gabarito Real e o arquivo de Previsões:")
label.pack(pady=10)

botao_comparar = tk.Button(root, text="Comparar arquivos via ID", command=comparar_arquivos)
botao_comparar.pack(pady=10)

caixa_texto = tk.Text(root, wrap='word', height=10, width=60)
caixa_texto.config(state='disabled')
caixa_texto.pack(pady=10)

root.mainloop()
