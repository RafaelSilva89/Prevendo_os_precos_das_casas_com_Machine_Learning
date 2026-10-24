"""Testes das funções de src/: leitura e limpeza da base, métricas, faixa de erro e testes estatísticos."""
import numpy as np
import pandas as pd
import pytest

from src.avaliacao import cobertura_faixa, erro_por_grupo, faixa_relativa, ml_error
from src.config import ARQUIVO_BASE, FEATURES
from src.dados import carregar_base, faixa_area, imovel_para_modelo, limpar_base, regras_erro
from src.estatistica import regra_iqr, teste_kruskal, teste_mann_whitney, teste_spearman

IMOVEL = dict(cidade='São Paulo', area=70, quartos=2, banheiros=1, vagas=1, andar=5, aceita_animal=1, mobiliado=0,
              condominio=600, iptu=100)


@pytest.fixture(scope='module')
def base():
    return carregar_base(ARQUIVO_BASE)


def test_carregar_base_traduz_e_converte(base):
    assert {'cidade', 'aluguel', 'condominio', 'iptu', 'eh_casa', 'iptu_zero'} <= set(base.columns)
    assert set(base['mobiliado'].unique()) == {0, 1}
    assert set(base['aceita_animal'].unique()) == {0, 1}
    assert base['andar'].dtype.kind == 'i'
    # andar "-" vira 0 e marca casa
    assert (base.loc[base['eh_casa'] == 1, 'andar'] == 0).all()


def test_regras_erro_nao_usam_o_aluguel():
    imoveis = pd.DataFrame([dict(IMOVEL), dict(IMOVEL, condominio=1_117_000), dict(IMOVEL, area=46_335),
                            dict(IMOVEL, iptu=313_700), dict(IMOVEL, andar=301)])
    erros = regras_erro(imoveis)
    assert erros.any(axis=1).tolist() == [False, True, True, True, True]
    assert erros.sum().tolist() == [1, 1, 1, 1]


def test_limpar_base_remove_duplicatas_e_erros(base):
    limpa, resumo = limpar_base(base)
    assert not limpa.duplicated().any()
    assert not regras_erro(limpa).any(axis=None)
    assert resumo['linhas finais'] == len(limpa)
    assert resumo['linhas originais'] - resumo['duplicadas'] - resumo.filter(like='erro').sum() <= len(limpa)


def test_imovel_para_modelo_cria_as_colunas_do_modelo():
    entrada = imovel_para_modelo(dict(IMOVEL, iptu=0))
    assert entrada.columns.tolist() == FEATURES
    assert entrada.loc[0, 'iptu_zero'] == 1 and entrada.loc[0, 'eh_casa'] == 0
    casa = imovel_para_modelo(dict(IMOVEL, eh_casa=1, andar=7))
    assert casa.loc[0, 'andar'] == 0


def test_faixa_area():
    faixas = faixa_area(pd.Series([30, 50, 51, 500]))
    assert faixas.astype(str).tolist() == ['até 50 m²', 'até 50 m²', '51 a 80 m²', 'mais de 400 m²']


def test_ml_error_valores_conhecidos():
    metricas = ml_error('teste', [100, 200], [110, 180]).iloc[0]
    assert metricas['MAE (R$)'] == pytest.approx(15)
    assert metricas['MAPE'] == pytest.approx((0.10 + 0.10) / 2)
    assert metricas['MPE'] == pytest.approx((0.10 - 0.10) / 2)
    assert ml_error('perfeito', [1, 2, 3], [1, 2, 3]).iloc[0]['R²'] == pytest.approx(1)


def test_faixa_relativa_e_cobertura():
    gerador = np.random.default_rng(0)
    previsto = gerador.uniform(1_000, 10_000, 5_000)
    real = previsto * gerador.lognormal(0, 0.2, 5_000)
    faixa = faixa_relativa(real, previsto, cobertura=0.80)
    assert faixa[0] < 1 < faixa[1]
    assert cobertura_faixa(real, previsto, faixa) == pytest.approx(0.80, abs=0.01)


def test_erro_por_grupo():
    resumo = erro_por_grupo([100, 100, 200], [110, 90, 260], ['A', 'A', 'B'])
    assert resumo.loc['A', 'MAE'] == pytest.approx(10)
    assert resumo.loc['B', 'MAPE'] == pytest.approx(0.30)
    assert resumo.index[0] == 'B'  # ordenado do maior erro para o menor
    assert resumo.loc['A', 'pior cenário'] == pytest.approx(100 - 10)


def test_testes_estatisticos():
    gerador = np.random.default_rng(1)
    a, b = gerador.normal(10, 1, 300), gerador.normal(12, 1, 300)
    assert teste_mann_whitney(b, a)['tamanho do efeito'] > 0.5
    valores = np.concatenate([a, b])
    assert teste_kruskal(valores, ['a'] * 300 + ['b'] * 300)['tamanho do efeito'] > 0.3
    assert teste_spearman(valores, valores * 2)['estatística'] == pytest.approx(1)


def test_regra_iqr():
    resultado = regra_iqr(pd.Series([1, 2, 3, 4, 5, 100]))
    assert resultado['outliers'] == 1
