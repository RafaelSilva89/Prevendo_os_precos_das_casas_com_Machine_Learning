"""Avaliação dos modelos de regressão: métricas (padrão ml_error), relatório de validação, comparação e ficha."""
import json
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from src.graficos import COR_DESTAQUE, COR_PRINCIPAL, reais, salvar, titulo


def ml_error(nome, y, yhat):
    """Métricas de regressão de um modelo, com os erros em reais (R$).

    MAE: erro absoluto médio · MAPE: erro percentual médio (em relação ao aluguel real) ·
    RMSE: raiz do erro quadrático médio (pune mais os erros grandes) · R²: parte das diferenças explicada ·
    MPE: erro percentual médio COM sinal (positivo = o modelo superestima).
    """
    y, yhat = np.asarray(y, dtype=float), np.asarray(yhat, dtype=float)
    return pd.DataFrame({
        'MAE (R$)': mean_absolute_error(y, yhat),
        'MAPE': np.mean(np.abs((y - yhat) / y)),
        'RMSE (R$)': root_mean_squared_error(y, yhat),
        'R²': r2_score(y, yhat),
        'MPE': np.mean((yhat - y) / y),
    }, index=[nome])


def relatorio_validacao(nome, modelo, X_treino, y_treino, X_validacao, y_validacao, nome_arquivo=None):
    """Avalia um modelo já treinado na validação, no formato padrão do projeto.

    Mostra: amostra gabarito x previsão, métricas (ml_error), MAE no treino x validação (sobreajuste) e
    os gráficos de previsto x real e de distribuição do erro. Devolve (métricas, previsões da validação).
    """
    previsto = modelo.predict(X_validacao)
    real = np.asarray(y_validacao, dtype=float)
    print('AMOSTRA: GABARITO x PREVISÃO DA MÁQUINA (aluguel em R$)')
    display(pd.DataFrame({'Gabarito': real, 'Previsão da máquina': previsto,
                          'Erro (R$)': previsto - real}).head(6).round(0))

    metricas = ml_error(nome, y_validacao, previsto)
    mae_treino = mean_absolute_error(y_treino, modelo.predict(X_treino))
    metricas['MAE treino (R$)'] = mae_treino
    print('=' * 50)
    print(f'MÉTRICAS DE VALIDAÇÃO: {nome}')
    print('=' * 50)
    print(metricas.drop(columns='MAE treino (R$)').round(3).T.to_string(header=False))
    print(f'\nMAE no treino: {reais(mae_treino)} | na validação: {reais(metricas["MAE (R$)"].iloc[0])} '
          f'| diferença: {metricas["MAE (R$)"].iloc[0] - mae_treino:+,.0f} R$'.replace(',', '.'))

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
    axs[0].scatter(previsto, real, s=8, alpha=0.35, color=COR_PRINCIPAL)
    limites = [0, max(previsto.max(), real.max())]
    axs[0].plot(limites, limites, color=COR_DESTAQUE, ls='--', label='Previsão perfeita')
    axs[0].set(xlabel='Aluguel previsto (R$)', ylabel='Aluguel real (R$)', title='Previsto x real')
    axs[0].legend(frameon=False)
    sns.histplot(previsto - real, bins=50, color=COR_PRINCIPAL, ax=axs[1])
    axs[1].axvline(0, color=COR_DESTAQUE, ls='--')
    axs[1].set(xlabel='Erro (previsto - real, R$)', ylabel='Imóveis', title='Distribuição do erro')
    sns.despine()
    plt.suptitle(nome, fontsize=14, x=0.01, ha='left', y=1.02)
    if nome_arquivo:
        salvar(nome_arquivo)
    plt.show()
    return metricas, previsto


def plot_comparacao_modelos(comparacao, nome_arquivo, referencia='Média (referência)'):
    """Painel com uma barra por modelo em cada métrica e a linha do modelo de referência (que só chuta a média)."""
    modelos = comparacao.index.tolist()
    cores = [COR_DESTAQUE if m == referencia else COR_PRINCIPAL for m in modelos]
    fig, axes = plt.subplots(1, 4, figsize=(24, 5.5))
    plt.subplots_adjust(wspace=0.75)
    for ax, metrica in zip(axes, ['MAE (R$)', 'MAPE', 'RMSE (R$)', 'R²']):
        valores = comparacao[metrica]
        ax.barh(modelos, valores, color=cores, alpha=0.85)
        ax.set_xlim(min(0, valores.min()), valores.max() * 1.35)
        for i, v in enumerate(valores):
            ax.text(v, i, f' {v:.3f}' if metrica in ('R²', 'MAPE') else f' {reais(v)}', va='center', fontsize=9)
        if referencia in comparacao.index:
            ax.axvline(comparacao.loc[referencia, metrica], color='gray', ls='--', lw=1, label='modelo que chuta a média')
        ax.set_title(metrica + (' (maior é melhor)' if metrica == 'R²' else ' (menor é melhor)'), fontweight='bold')
        ax.invert_yaxis()
        sns.despine(ax=ax)
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, loc='lower center', ncol=1, fontsize=10,
               bbox_to_anchor=(0.5, -0.06))
    titulo('Comparação dos modelos na validação')
    salvar(nome_arquivo)
    plt.show()


def faixa_relativa(y, previsto, cobertura=0.80):
    """Faixa de erro RELATIVA: entre quais múltiplos da previsão o valor real costuma ficar.

    Ex.: (0,78; 1,25) = em 80% dos casos, o aluguel real fica entre 78% e 125% do previsto. Uma faixa em
    proporção acompanha o tamanho do aluguel (um erro de R$ 500 é grande para R$ 1.000 e pequeno para R$ 15.000).
    """
    razao = np.asarray(y, dtype=float) / np.asarray(previsto, dtype=float)
    cauda = (1 - cobertura) / 2
    return tuple(float(v) for v in np.quantile(razao, [cauda, 1 - cauda]))


def cobertura_faixa(y, previsto, faixa):
    """Parcela dos imóveis cujo valor real cai dentro da faixa (previsto x limite inferior, previsto x superior)."""
    y, previsto = np.asarray(y, dtype=float), np.asarray(previsto, dtype=float)
    return float(((y >= previsto * faixa[0]) & (y <= previsto * faixa[1])).mean())


def erro_por_grupo(y, previsto, grupos):
    """Tradução do erro para o negócio: erro médio por grupo (ex.: cidade) e cenários de melhor e pior caso."""
    tabela = pd.DataFrame({'real': np.asarray(y, dtype=float), 'previsto': np.asarray(previsto, dtype=float),
                           'grupo': pd.Series(grupos).reset_index(drop=True)})  # Series: mantém a ordem das categorias
    tabela['erro_abs'] = (tabela['previsto'] - tabela['real']).abs()
    resumo = tabela.groupby('grupo', observed=True).agg(imoveis=('real', 'size'), aluguel_real=('real', 'mean'),
                                                        aluguel_previsto=('previsto', 'mean'), MAE=('erro_abs', 'mean'))
    resumo['MAPE'] = (tabela.assign(ape=tabela['erro_abs'] / tabela['real'])
                      .groupby('grupo', observed=True)['ape'].mean())
    resumo['pior cenário'] = resumo['aluguel_previsto'] - resumo['MAE']
    resumo['melhor cenário'] = resumo['aluguel_previsto'] + resumo['MAE']
    return resumo.sort_values('MAE', ascending=False)


def salvar_ficha_modelo(caminho, **campos):
    """Grava a ficha do modelo (JSON): tudo o que é preciso saber para usá-lo corretamente."""
    ficha = {'data_treino': datetime.now().strftime('%Y-%m-%d %H:%M'), **campos}
    with open(caminho, 'w', encoding='utf-8') as arquivo:
        json.dump(ficha, arquivo, ensure_ascii=False, indent=2, default=float)
    return ficha
