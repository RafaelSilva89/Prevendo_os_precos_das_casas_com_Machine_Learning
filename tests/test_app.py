"""Teste do app Streamlit: abre sem erro, sugere um aluguel e reage às características do imóvel."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / 'app.py')


def abrir():
    return AppTest.from_file(APP, default_timeout=60).run()


def valor_sugerido(app):
    return int(app.metric[0].value.replace('R$ ', '').replace('.', ''))


def test_app_abre_e_sugere_um_aluguel():
    app = abrir()
    assert not app.exception
    assert app.metric[0].label == 'Aluguel sugerido'
    assert valor_sugerido(app) > 0


def test_imovel_maior_tem_aluguel_maior():
    app = abrir()
    app.sidebar.number_input[0].set_value(40).run()
    pequeno = valor_sugerido(app)
    app.sidebar.number_input[0].set_value(250).run()
    grande = valor_sugerido(app)
    assert not app.exception
    assert grande > pequeno


def test_imovel_impossivel_e_recusado():
    app = abrir()
    app.sidebar.number_input[1].set_value(20000).run()   # condomínio de R$ 20 mil em 70 m²
    assert not app.exception
    assert len(app.error) == 1
