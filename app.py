"""App Streamlit: aluguel sugerido para um imóvel (com faixa de erro) e panorama dos aluguéis em 5 cidades."""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

RAIZ = Path(__file__).resolve().parent
sys.path.append(str(RAIZ))

from src.config import CIDADES  # noqa: E402
from src.dados import imovel_para_modelo, regras_erro  # noqa: E402

st.set_page_config(page_title='Aluguel de imóveis', page_icon='🏠', layout='wide')


@st.cache_resource
def carregar_modelo():
    pacote = joblib.load(RAIZ / 'Model' / 'modelo_aluguel.pkl')
    ficha = json.loads((RAIZ / 'Model' / 'modelo_aluguel.json').read_text(encoding='utf-8'))
    return pacote, ficha


@st.cache_data
def carregar_base():
    return pd.read_csv(RAIZ / 'Datasets' / 'imoveis_limpos.csv')


def reais(valor):
    return 'R$ ' + f'{valor:,.0f}'.replace(',', '.')


pacote, ficha = carregar_modelo()
base = carregar_base()

st.title('🏠 Quanto cobrar de aluguel?')
st.caption('Modelo treinado com 10 mil anúncios de 5 cidades (base pública do Kaggle). '
           'A previsão é uma estimativa, sempre acompanhada da faixa em que o aluguel real costuma ficar.')

aba_simulador, aba_base, aba_modelo = st.tabs(['💰 Simulador', '📊 Aluguéis por cidade', 'ℹ️ Sobre o modelo'])

with aba_simulador:
    with st.sidebar:
        st.header('Dados do imóvel')
        cidade = st.selectbox('Cidade', CIDADES)
        tipo = st.radio('Tipo', ['Apartamento', 'Casa'], horizontal=True)
        area = st.number_input('Área (m²)', min_value=10, max_value=2000, value=70, step=5)
        quartos = st.slider('Quartos', 1, 10, 2)
        banheiros = st.slider('Banheiros', 1, 10, 2)
        vagas = st.slider('Vagas de garagem', 0, 10, 1)
        andar = st.slider('Andar', 0, 51, 5, disabled=tipo == 'Casa')
        mobiliado = st.toggle('Mobiliado')
        aceita_animal = st.toggle('Aceita animal', value=True)
        condominio = st.number_input('Condomínio (R$/mês)', min_value=0, max_value=20000, value=600, step=50)
        iptu = st.number_input('IPTU (R$/mês)', min_value=0, max_value=10000, value=100, step=10)

    imovel = dict(cidade=cidade, area=area, quartos=quartos, banheiros=banheiros, vagas=vagas,
                  andar=0 if tipo == 'Casa' else andar, aceita_animal=int(aceita_animal), mobiliado=int(mobiliado),
                  condominio=condominio, iptu=iptu, eh_casa=int(tipo == 'Casa'))
    entrada = imovel_para_modelo(imovel)
    erros = regras_erro(entrada).iloc[0]

    if erros.any():
        st.error('Valores fora do padrão da base: ' + ', '.join(erros[erros].index) +
                 '. O modelo não foi treinado para esse tipo de imóvel.')
    else:
        previsto = float(pacote['modelo'].predict(entrada[pacote['features']])[0])
        inferior, superior = previsto * pacote['faixa'][0], previsto * pacote['faixa'][1]
        mesma_cidade = base[base['cidade'] == cidade]
        mediana_m2 = (mesma_cidade['aluguel'] / mesma_cidade['area']).median()

        col1, col2, col3 = st.columns(3)
        col1.metric('Aluguel sugerido', reais(previsto))
        col2.metric('Faixa provável (80%)', f'{reais(inferior)} a {reais(superior)}')
        col3.metric('Aluguel por m²', reais(previsto / area),
                    delta=f'{previsto / area / mediana_m2 - 1:+.0%} vs mediana da cidade', delta_color='off')
        st.caption(f'Em 80% dos imóveis do teste, o aluguel real ficou entre {pacote["faixa"][0]:.0%} e '
                   f'{pacote["faixa"][1]:.0%} do valor previsto.')
        if previsto > 12_000:
            st.warning('Imóvel de alto padrão: na base, quase todos os aluguéis acima de R$ 15 mil foram registrados '
                       'como R$ 15.000 (teto da coleta). Para esses imóveis, o modelo tende a subestimar o preço.')

        parecidos = mesma_cidade[(mesma_cidade['quartos'] == quartos) & (mesma_cidade['area'].between(area * 0.8, area * 1.2))]
        if len(parecidos) >= 5:
            st.subheader(f'{len(parecidos)} anúncios parecidos em {cidade}')
            st.caption(f'Mesmo número de quartos e área entre {area * 0.8:.0f} e {area * 1.2:.0f} m²')
            figura = px.histogram(parecidos, x='aluguel', nbins=30, labels={'aluguel': 'Aluguel (R$)'})
            figura.add_vline(x=previsto, line_dash='dash', line_color='#C0392B', annotation_text='sugerido')
            figura.update_layout(yaxis_title='Anúncios', height=320, margin=dict(t=20, b=20))
            st.plotly_chart(figura, use_container_width=True)

with aba_base:
    resumo = (base.assign(aluguel_m2=base['aluguel'] / base['area'])
              .groupby('cidade').agg(anuncios=('aluguel', 'size'), aluguel_mediano=('aluguel', 'median'),
                                     aluguel_m2_mediano=('aluguel_m2', 'median'), area_mediana=('area', 'median'))
              .sort_values('aluguel_m2_mediano', ascending=False))
    st.subheader('Aluguel mediano por cidade')
    st.dataframe(resumo.style.format({'aluguel_mediano': reais, 'aluguel_m2_mediano': 'R$ {:.0f}',
                                      'area_mediana': '{:.0f} m²'}), use_container_width=True)
    cidade_escolhida = st.selectbox('Ver uma cidade', CIDADES, key='cidade_base')
    dados_cidade = base[base['cidade'] == cidade_escolhida]
    figura = px.box(dados_cidade, x='quartos', y='aluguel', points=False,
                    labels={'quartos': 'Quartos', 'aluguel': 'Aluguel (R$)'},
                    title=f'Aluguel por número de quartos em {cidade_escolhida}')
    st.plotly_chart(figura, use_container_width=True)

with aba_modelo:
    metricas = ficha['metricas_no_teste']
    col1, col2, col3 = st.columns(3)
    col1.metric('Erro médio (MAE)', reais(metricas['MAE (R$)']))
    col2.metric('Erro percentual (MAPE)', f"{metricas['MAPE']:.0%}")
    col3.metric('R²', f"{metricas['R²']:.2f}")
    st.markdown(f"**Modelo:** {ficha['modelo']}  \n**Treinado com:** {ficha['treinado_com']}")
    st.markdown('**Avisos:**\n' + '\n'.join(f'- {aviso}' for aviso in ficha['avisos']))
