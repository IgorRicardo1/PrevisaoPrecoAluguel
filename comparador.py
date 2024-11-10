import pandas as pd
from tkinter import filedialog, messagebox
import tkinter as tk
import numpy as np
from sklearn.metrics import mean_absolute_error

# Função para selecionar e carregar dois arquivos CSV e comparar os preços
def comparar_arquivos():
    # Selecionar o arquivo com preços reais
    filepath_real = filedialog.askopenfilename(
        title="Selecione o arquivo CSV com os preços reais",
        filetypes=[("Arquivos CSV", "*.csv")]
    )
    if not filepath_real:
        return

    # Selecionar o arquivo com preços preditos
    filepath_predito = filedialog.askopenfilename(
        title="Selecione o arquivo CSV com os preços preditos",
        filetypes=[("Arquivos CSV", "*.csv")]
    )
    if not filepath_predito:
        return

    try:
        # Carregar os dados dos arquivos
        data_real = pd.read_csv(filepath_real)
        data_predito = pd.read_csv(filepath_predito)

        # Renomear colunas para garantir consistência
        data_real.rename(columns={'preco': 'preco_real'}, inplace=True)
        data_predito.rename(columns={'preco_predito': 'preco_predito'}, inplace=True)

        # Mesclar os dois dataframes com base nas colunas comuns ('tipo', 'area', 'quartos', 'bairro')
        data_merged = pd.merge(
            data_real, data_predito,
            on=['tipo', 'area', 'quartos', 'bairro'],
            suffixes=('_real', '_predito')
        )

        # Calcular a média dos desvios
        mae = mean_absolute_error(data_merged['preco_real'], data_merged['preco_predito'])
        media_desvio = np.mean(np.abs(data_merged['preco_real'] - data_merged['preco_predito']))
        
        # Exibir a média dos desvios
        resultado_janela = tk.Toplevel(root)
        resultado_janela.title("Resultados da Comparação")
        resultado_texto = f"Média do Desvio Absoluto (MAE): {mae:.2f}\nDesvio Médio dos Preços: {media_desvio:.2f}"
        
        texto = tk.Text(resultado_janela, wrap='word')
        texto.insert('1.0', resultado_texto)
        texto.config(state='disabled')
        texto.pack()
    except Exception as e:
        messagebox.showerror("Erro", f"Erro ao processar os arquivos: {str(e)}")

# Configuração da janela principal
root = tk.Tk()
root.title("Comparador de CSV de Preços de Aluguel")
root.geometry("400x200")

label = tk.Label(root, text="Selecione dois arquivos CSV para comparar os preços de aluguel:")
label.pack(pady=20)

botao_comparar = tk.Button(root, text="Comparar arquivos", command=comparar_arquivos)
botao_comparar.pack(pady=10)

root.mainloop()
