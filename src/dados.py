"""Leitura, tradução e limpeza da base de imóveis para alugar."""
import numpy as np
import pandas as pd

from src.config import (ANDAR_MAXIMO, AREA_MAXIMA, COLUNAS, CONDOMINIO_MAX_M2, FEATURES, IPTU_MAX_M2)


def carregar_base(caminho):
    """Lê o CSV do Kaggle, traduz as colunas e converte os campos de texto em números.

    - `aceita_animal` e `mobiliado` viram 1/0;
    - `andar` "-" vira 0 e marca `eh_casa` = 1 (85% dos "-" não pagam condomínio: são casas, não térreo);
    - `iptu_zero` = 1 quando o IPTU informado é zero (isento, incluso no condomínio ou não informado).
    """
    df = pd.read_csv(caminho).rename(columns=COLUNAS)
    df['aceita_animal'] = (df['aceita_animal'] == 'acept').astype(int)
    df['mobiliado'] = (df['mobiliado'] == 'furnished').astype(int)
    df['eh_casa'] = (df['andar'] == '-').astype(int)
    df['andar'] = pd.to_numeric(df['andar'].replace('-', '0')).astype(int)
    df['iptu_zero'] = (df['iptu'] == 0).astype(int)
    return df


def regras_erro(df):
    """Marca os erros de digitação (uma coluna True/False por regra), sem olhar o aluguel.

    Assim, as mesmas regras servem para limpar a base e para recusar um imóvel impossível no app.
    """
    return pd.DataFrame({
        f'área > {AREA_MAXIMA:,} m²'.replace(',', '.'): df['area'] > AREA_MAXIMA,
        f'condomínio > R$ {CONDOMINIO_MAX_M2}/m²': df['condominio'] / df['area'] > CONDOMINIO_MAX_M2,
        f'IPTU > R$ {IPTU_MAX_M2}/m²': df['iptu'] / df['area'] > IPTU_MAX_M2,
        f'andar > {ANDAR_MAXIMO}': df['andar'] > ANDAR_MAXIMO,
    }, index=df.index)


def limpar_base(df):
    """Remove as linhas duplicadas e os erros de digitação. Devolve (base limpa, resumo do que saiu)."""
    sem_duplicatas = df.drop_duplicates()
    erros = regras_erro(sem_duplicatas)
    limpa = sem_duplicatas[~erros.any(axis=1)].reset_index(drop=True)
    resumo = pd.Series({'linhas originais': len(df), 'duplicadas': len(df) - len(sem_duplicatas),
                        **{f'erro: {regra}': int(n) for regra, n in erros.sum().items()},
                        'linhas finais': len(limpa)})
    return limpa, resumo


def imovel_para_modelo(imovel):
    """Converte o dicionário de um imóvel (como o app recebe) para a tabela de entrada do modelo."""
    linha = dict(imovel)
    linha['eh_casa'] = int(linha.get('eh_casa', 0))
    if linha['eh_casa']:
        linha['andar'] = 0
    linha['iptu_zero'] = int(linha['iptu'] == 0)
    return pd.DataFrame([linha])[FEATURES]


def faixa_area(area):
    """Faixas de área usadas para traduzir o erro do modelo (m²)."""
    return pd.cut(area, bins=[0, 50, 80, 120, 200, 400, np.inf],
                  labels=['até 50 m²', '51 a 80 m²', '81 a 120 m²', '121 a 200 m²', '201 a 400 m²', 'mais de 400 m²'])
