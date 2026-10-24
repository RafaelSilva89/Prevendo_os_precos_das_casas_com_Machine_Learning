# 🏠 Prevendo o Aluguel de Imóveis com Machine Learning

**Quanto cobrar de aluguel por um imóvel? Um modelo treinado com 10 mil anúncios de 5 cidades responde, com a margem de erro dita com todas as letras.**

<div align="center">
<img src="Imagens/Capa.png" width="350px">
</div>

[![Testes](https://github.com/RafaelSilva89/Prevendo_os_precos_das_casas_com_Machine_Learning/actions/workflows/testes.yml/badge.svg)](https://github.com/RafaelSilva89/Prevendo_os_precos_das_casas_com_Machine_Learning/actions/workflows/testes.yml)

**Python · Pandas · Scikit-learn · HistGradientBoosting · XGBoost · LightGBM · SciPy · Pytest · GitHub Actions · Streamlit · CRISP-DM**

| 📊 Destaques | |
|---|---|
| Base | **10.692 anúncios** de aluguel em **São Paulo, Rio de Janeiro, Belo Horizonte, Porto Alegre e Campinas** (Kaggle); **10.322** depois de remover duplicatas e erros de digitação |
| Modelo | Sugere o aluguel com erro médio de **R$ 1.249** (**37%** do aluguel) e explica **65%** das diferenças de preço, contra R² zero de quem chuta a média |
| Margem de erro | Em **8 de cada 10** imóveis, o aluguel real fica entre **56% e 157%** do valor sugerido (**81,3%** de acerto no teste) |
| O que encarece | **Área**, **IPTU e condomínio** (padrão do imóvel), **banheiros**, **mobília** e **cidade**. Mobiliar soma **+44%** ao mesmo imóvel; em Porto Alegre, ele custa **28% menos** que em São Paulo |
| Qualidade dos dados | A v2 encontrou o que a v1 não viu: um **"modelo da média" que usava as respostas**, **358 duplicatas**, **12 erros de digitação** (condomínio de R$ 1,1 milhão) e um **teto escondido** de R$ 15 mil no aluguel |
| Entrega | Notebook CRISP-DM, **13 testes automatizados** (GitHub Actions) e **app Streamlit** que sugere o aluguel com a faixa de erro |

---

## 📑 Sumário

1. [Contexto do problema](#1-contexto-do-problema)
2. [Objetivo](#2-objetivo)
3. [Base de dados](#3-base-de-dados)
4. [Estrutura do projeto](#4-estrutura-do-projeto)
5. [Como executar](#5-como-executar)
6. [Etapas do projeto (CRISP-DM)](#6-etapas-do-projeto-crisp-dm)
7. [Principais descobertas da análise exploratória](#7-principais-descobertas-da-análise-exploratória)
8. [Construção do modelo](#8-construção-do-modelo)
9. [Resultado do modelo](#9-resultado-do-modelo)
10. [App interativo (Streamlit)](#10-app-interativo-streamlit)
11. [O que mudou da v1 para a v2](#11-o-que-mudou-da-v1-para-a-v2)
12. [Insights](#12-insights)
13. [Limites](#13-limites)
14. [Próximos passos](#14-próximos-passos)

---

## 1. Contexto do problema

<img src="Imagens/Capa2.png" width="120px" align="right">

Definir o aluguel de um imóvel é uma decisão difícil para **proprietários**, **imobiliárias** e **plataformas de anúncio**. Preço **alto demais** deixa o imóvel vazio por meses; preço **baixo demais** deixa dinheiro na mesa durante todo o contrato. Hoje, esse preço costuma sair da experiência do corretor ou de uma comparação manual com anúncios parecidos.

Um modelo treinado com milhares de anúncios dá uma **referência objetiva**, rápida e igual para todos, desde que diga também **o quanto pode errar**.

## 2. Objetivo

> **Perguntas de negócio:**
> 1. **Quanto cobrar** de aluguel por um imóvel, dadas as suas características? Com que **margem de erro**?
> 2. **O que mais encarece** um aluguel: cidade, tamanho, banheiros, vagas, mobília, andar?
> 3. Em quais **cidades** e **tamanhos de imóvel** a sugestão é mais confiável?
> 4. Quais anúncios estão **muito acima ou muito abaixo** do preço esperado?

## 3. Base de dados

A base é a [*Brazilian houses to rent*](https://www.kaggle.com/datasets/rubenssjr/brasilian-houses-to-rent) (Kaggle), com anúncios de 2020, salva em `Datasets/house_data.csv`.

| Coluna | Descrição |
|---|---|
| `cidade` | São Paulo, Rio de Janeiro, Belo Horizonte, Porto Alegre ou Campinas |
| `area`, `quartos`, `banheiros`, `vagas`, `andar` | Características do imóvel (andar "-" = **casa**) |
| `aceita_animal`, `mobiliado` | Sim ou não |
| `condominio`, `iptu` | Valores mensais (R$) |
| `aluguel` | 🎯 **Alvo**: aluguel mensal (R$) |
| `seguro_incendio`, `total` | ⚠️ Calculados a partir do aluguel: **ficam fora do modelo** (vazamento) |

## 4. Estrutura do projeto

```
Prevendo_os_precos_das_casas_com_Machine_Learning/
├── Datasets/
│   ├── house_data.csv                  # base original (Kaggle)
│   └── imoveis_limpos.csv              # base tratada (gerada no notebook, usada pelo app)
├── Imagens/                            # gráficos gerados pelos notebooks
├── Model/
│   ├── modelo_aluguel.pkl              # pipeline final + faixa de erro (usado pelo app)
│   ├── modelo_aluguel.json             # ficha do modelo (colunas, métricas, hiperparâmetros, avisos)
│   └── Modelo_Random_Forest_Regressor_v100.pkl   # modelo da versão original
├── docs/
│   └── decisoes_analiticas.md          # registro das decisões do projeto (DA-01 a DA-12)
├── reports/
│   ├── comparacao_modelos.csv          # métricas dos 7 algoritmos na validação
│   ├── erro_por_cidade.csv             # tradução do erro por cidade
│   └── erro_por_faixa_de_area.csv      # tradução do erro por tamanho do imóvel
├── Notebooks/
│   ├── Prevendo os preços das casas com Machine Learning.ipynb        # versão original
│   └── Prevendo os preços das casas com Machine Learning - v2.ipynb   # versão reorganizada (CRISP-DM)
├── src/
│   ├── config.py                       # caminhos, semente, colunas e limites das regras de qualidade
│   ├── dados.py                        # leitura, tradução, limpeza e preparo de um imóvel novo
│   ├── estatistica.py                  # testes das hipóteses (com tamanho de efeito) e regra do IQR
│   ├── selecao.py                      # seleção de variáveis (Random Forest + Informação Mútua)
│   ├── avaliacao.py                    # ml_error, relatório de validação, faixa de erro, tradução do erro e ficha
│   └── graficos.py                     # gráficos e formatação em português
├── tests/
│   ├── test_funcoes.py                 # 10 testes das funções de src/
│   └── test_app.py                     # 3 testes do app (AppTest do Streamlit)
├── Webapp/                             # API Flask da versão original (mantida para mostrar a evolução)
├── .github/workflows/testes.yml        # roda os testes a cada push (GitHub Actions)
├── app.py                              # app Streamlit
├── pytest.ini
├── requirements.txt
└── README.md
```

## 5. Como executar

```bash
# 1. Clonar o repositório
git clone https://github.com/RafaelSilva89/Prevendo_os_precos_das_casas_com_Machine_Learning.git
cd Prevendo_os_precos_das_casas_com_Machine_Learning

# 2. Instalar as dependências
pip install -r requirements.txt

# 3. Rodar os testes
pytest

# 4. Abrir o notebook e executar todas as células (gera a base tratada e o modelo usados pelo app)
jupyter notebook "Notebooks/Prevendo os preços das casas com Machine Learning - v2.ipynb"

# 5. Abrir o app
streamlit run app.py
```

## 6. Etapas do projeto (CRISP-DM)

| Fase | O que foi feito |
|---|---|
| **01. Entendimento do negócio** | Definição das 4 perguntas e do público |
| **02. Preparação dos dados** | Tradução das colunas, remoção de duplicatas e de erros de digitação (regras que não olham o aluguel), outliers pela regra do IQR, retirada das variáveis com vazamento, EDA com 10 hipóteses (cada uma com teste estatístico e tamanho do efeito) |
| **03. Modelagem** | Divisão 60/20/20 estratificada por cidade, seleção de variáveis, 7 algoritmos (uma seção cada) com validação cruzada, ajuste fino, faixa de erro, calibração e teste anti-vazamento |
| **04. Avaliação** | Erro no teste, comparação com a v1 em São Paulo, anúncios fora do preço esperado, **tradução do erro por cidade e por tamanho** e variáveis mais importantes |
| **05. Implantação** | Modelo salvo com ficha, simulação de imóveis novos e app Streamlit |

## 7. Principais descobertas da análise exploratória

### 🔎 Antes de tudo: entender o que a base diz

- **358 anúncios duplicados** (3,3%) e **12 erros de digitação**: condomínio de **R$ 1.117.000** num imóvel de 155 m², IPTU de **R$ 313.700** em 42 m², área de **46.335 m²** com 4 quartos e um apartamento no **301º andar**. As regras que encontram esses erros usam só as características do imóvel, nunca o aluguel, e por isso também protegem o app.
- O andar **"-"** é **casa**: 84,4% desses imóveis não pagam condomínio, contra 3,6% dos que têm andar.
- O **seguro-incêndio** é sempre ~1,3% do aluguel, e o **total** soma o aluguel: os dois "entregam" a resposta.
- **Um teto escondido:** R$ 15.000 é o 3º valor de aluguel mais comum (**231 anúncios**), e só **12** passam disso. A coleta limitou o preço, e acima disso o modelo tende a subestimar.

### As hipóteses

| # | Hipótese | Resultado | Evidência |
|---|---|---|---|
| **H1** | O aluguel varia por cidade, SP é a mais cara | ✅ Verdadeira | Mediana de **R$ 3.500** (SP) a **R$ 1.500** (Campinas) |
| **H2** | Imóveis maiores, aluguel maior | ✅ Verdadeira | Relação forte (rho **0,73**) |
| **H3** | Mobiliados têm aluguel maior | ✅ Verdadeira | **R$ 3.600** contra **R$ 2.460**, com a mesma área |
| **H4** | Aceitar animal não muda o aluguel | ⚠️ Quase | **R$ 2.800** contra **R$ 2.500**: efeito pequeno |
| **H5** | Casas têm aluguel maior | ✅ Pelo tamanho | **R$ 3.900** contra **R$ 2.500**, com mais que o dobro da área |
| **H6** | O preço por m² varia por cidade | ✅ Verdadeira | **SP e Rio ~R$ 30/m²**; BH, Porto Alegre e Campinas entre R$ 18 e R$ 22 |
| **H7** | Casas são mais baratas por m² | ✅ Verdadeira | **R$ 21,2** contra **R$ 28,9/m²** |
| **H8** | Banheiros dizem mais que quartos | ✅ Verdadeira | **0,72** contra **0,60** de correlação com o aluguel |
| **H9** | Andares altos, aluguel maior | ⚠️ Parcial | **R$ 1.900** → **R$ 4.500**, mas a área também cresce |
| **H10** | Condomínio e IPTU acompanham o aluguel | ✅ Verdadeira | **0,73** e **0,75** |

<div align="center">
<img src="Imagens/H6 Aluguel por m2 por cidade.png" width="650px">
</div>

> 🧪 **Testes estatísticos:** cada hipótese tem um teste e uma medida de **tamanho do efeito**: Kruskal-Wallis + épsilon² (cidades), Mann-Whitney + correlação de postos (mobiliado, animal, casa) e Spearman (área, banheiros, andar, condomínio, IPTU). Com 10 mil anúncios, quase tudo dá p-valor minúsculo: o que importa é o tamanho do efeito, não o p.

## 8. Construção do modelo

| Etapa | Como foi feito |
|---|---|
| **Variáveis** | 12 características conhecidas **antes** de definir o preço (cidade, área, quartos, banheiros, vagas, andar, casa, mobília, animal, condomínio, IPTU e IPTU zero). Seguro e total ficam de fora |
| **Divisão** | 60% treino, 20% validação e 20% teste, **estratificada por cidade**; o teste é aberto uma única vez |
| **Seleção** | Random Forest **de regressão** + Informação Mútua: área e IPTU no topo; ficar só com as 5 melhores aumenta o erro, então as 12 ficam |
| **Algoritmos** | Média (referência), Regressão Linear, Lasso, Random Forest, XGBoost, LightGBM e HistGradientBoosting, todos em `Pipeline`, com **validação cruzada** (média ± desvio) |
| **Escolha** | **HistGradientBoosting**: 1º lugar em todas as métricas e na validação cruzada (R$ 1.308 ± 34) |
| **Ajuste fino** | `RandomizedSearchCV` com 30 combinações; o erro ficou igual, mas o **sobreajuste caiu** (diferença treino × validação de R$ 293 para R$ 117) |
| **Faixa de erro** | **Relativa** (56% a 157% do previsto), porque o erro em reais cresce com o aluguel. Uma faixa fixa em reais cobriria 99% dos imóveis baratos e só 50% dos caros |
| **Anti-vazamento** | Com o seguro-incêndio, o erro "cairia" para **R$ 108** (R² 0,99): o modelo só desfaria a conta. Fica de fora |

<div align="center">
<img src="Imagens/Comparacao dos modelos.png" width="850px">
</div>

## 9. Resultado do modelo

| Métrica (teste, 2.065 imóveis) | Modelo final | Chutar a média |
|---|---|---|
| Erro médio (MAE) | **R$ 1.249** | R$ 2.544 |
| Erro percentual (MAPE) | **36,7%** | 112% |
| R² | **0,65** | 0 |

**Tradução do erro para o negócio:**

| Cidade | Aluguel médio | Erro médio | Erro % |
|---|---|---|---|
| São Paulo | R$ 4.757 | R$ 1.478 | 36,4% |
| Belo Horizonte | R$ 3.636 | R$ 1.155 | 34,3% |
| Rio de Janeiro | R$ 3.213 | R$ 1.095 | 38,8% |
| Porto Alegre | R$ 2.224 | R$ 775 | 36,5% |
| Campinas | R$ 2.323 | R$ 724 | 39,0% |

- A sugestão é mais confiável para **apartamentos de 51 a 80 m²** (erro de **30,7%**) e menos para imóveis de **201 a 400 m²** (**43,6%**).
- O modelo é **bem calibrado**: em 10 faixas de preço, a média prevista fica a no máximo 5% da média real.
- **18,8%** dos anúncios ficam fora da faixa de 80%. Os mais extremos parecem **erros no próprio aluguel** (ex.: R$ 12.000 por um apartamento de 75 m² em Campinas, que o modelo estimou em R$ 991). Uma plataforma poderia usar o modelo como **filtro de qualidade** de anúncios.

<div align="center">
<img src="Imagens/Importancia por permutacao.png" width="650px">
</div>

## 10. App interativo (Streamlit)

O app tem 3 abas:
1. **💰 Simulador:** informe as características do imóvel e receba o **aluguel sugerido**, a **faixa de 80%**, o aluguel por m² comparado com a mediana da cidade e a distribuição dos **anúncios parecidos**. Imóveis fora das regras de qualidade são recusados com o motivo, e imóveis de alto padrão recebem o aviso do teto de R$ 15 mil.
2. **📊 Aluguéis por cidade:** mediana do aluguel, do aluguel por m² e da área em cada cidade, e o aluguel por número de quartos.
3. **ℹ️ Sobre o modelo:** métricas no teste e avisos da ficha do modelo.

```bash
streamlit run app.py
```

> ℹ️ O app usa `Datasets/imoveis_limpos.csv` e `Model/modelo_aluguel.pkl`, gerados ao executar o notebook até a Fase 05.

## 11. O que mudou da v1 para a v2

A versão original foi mantida em `Notebooks/` (e a API Flask em `Webapp/`) para mostrar a evolução.

| # | Versão original (v1) | Problema | Versão v2 |
|---|---|---|---|
| 1 | "Modelo da média" com R² de 0,68, o melhor de todos | Calculava a média do aluguel por área **nos próprios imóveis da validação**: usava o gabarito | Referência honesta (média do treino, R² zero), com a conta da v1 reproduzida ao lado |
| 2 | Outliers detectados (Grubbs, KNN), mas mantidos | Condomínio de R$ 1,1 milhão e IPTU de R$ 313 mil no treino | **12 erros de digitação** removidos por regras que não olham o aluguel; outliers reais mantidos |
| 3 | Duplicatas mantidas | **358** anúncios repetidos, que podiam cair no treino e no teste | Duplicatas removidas antes da separação |
| 4 | Andar "-" = 0 (térreo) | 84% dos "-" não pagam condomínio: são **casas** | Variável `eh_casa` |
| 5 | Seleção com `RandomForestClassifier` | Tratava cada valor de aluguel como uma classe | `RandomForestRegressor` + `mutual_info_regression` com semente |
| 6 | R² de 0,94 nos gráficos de erro | Medido **no treino** | Toda métrica vem de dados que o modelo não viu; R² real no teste: **0,65** |
| 7 | "Erro de 4%" para 717 m² e "erro menor que 0,5%" | Era MAPE de **404%** e de **50%**, num agrupamento por m² exato (1 imóvel por grupo) | Erro traduzido **por cidade e por faixa de área**, com cenários de pior e melhor caso |
| 8 | Só São Paulo | Descartava 45% da base | **5 cidades**, com a cidade como variável; em SP, MAE de **R$ 1.478** contra R$ 1.553 da v1 |
| 9 | Sem faixa de erro | O usuário recebia só um número | **Faixa relativa de 80%**, conferida no teste (81,3%) |
| 10 | `max_features='auto'`, `skopt`, sem `random_state` | Não roda nas versões atuais e não é reproduzível | `Pipeline`, `RandomizedSearchCV`, sementes fixas e validação cruzada |
| 11 | API Flask com SQL montado por texto e um `.pkl` com nome inexistente | A API não rodava e aceitava injeção de SQL | App Streamlit com validação das entradas e **13 testes** no GitHub Actions |
| 12 | Hipóteses só com gráficos | Não diferencia efeito real de acaso | **10 hipóteses** com teste estatístico e tamanho do efeito |
| 13 | Sem registro das decisões | — | `docs/decisoes_analiticas.md`, ficha do modelo em JSON e pasta `reports/` |

## 12. Insights

1. **A régua de comparação também precisa ser testada:** o "modelo da média" da v1 colava a resposta e parecia o melhor modelo.
2. **Validar os campos na hora do cadastro:** 12 erros de digitação e 358 duplicatas estavam no treino da v1, e há aluguéis com um zero a mais.
3. **São Paulo e Rio têm o mesmo preço por m² (~R$ 30)**; BH, Porto Alegre e Campinas ficam entre R$ 18 e R$ 22.
4. **Mobiliar vale a pena:** com a mesma área, o mobiliado custa bem mais, e o modelo estima **+44%** para o mesmo imóvel típico.
5. **Banheiros dizem mais que quartos** sobre o aluguel: eles medem o padrão do imóvel.
6. **Por m², a casa é mais barata que o apartamento** (R$ 21 contra R$ 29).
7. **Cuidado com colunas derivadas do alvo:** com o seguro-incêndio, o modelo teria R² de 0,99 e seria inútil.
8. **Olhar os valores mais frequentes, não só a média e o máximo:** foi assim que apareceu o teto de R$ 15 mil.
9. **O modelo explica 65% do preço.** O resto depende do bairro, do acabamento e da conservação, que a base não tem: é uma **referência**, não uma avaliação.

## 13. Limites

- **Sem bairro nem endereço:** dentro de uma cidade, a localização é o fator mais importante do preço. É a principal razão do erro de 37%.
- **Anúncios de 2020:** os valores não incluem a inflação dos aluguéis desde então.
- **Preço anunciado, não fechado:** o aluguel pode ser negociado para baixo.
- **Só 5 cidades** e imóveis dentro das regras de qualidade.
- **Teto de R$ 15 mil:** para imóveis de alto padrão, o modelo tende a subestimar.
- **Erros no próprio aluguel** continuam na base, porque não dá para separá-los dos imóveis realmente caros sem olhar o aluguel.

## 14. Próximos passos

1. **Acrescentar a localização** (bairro ou CEP), o que mais deve reduzir o erro.
2. **Tratar o teto de R$ 15 mil** como valor censurado ("15 mil ou mais").
3. **Treinar com uma perda robusta** ou com o **log do aluguel**, para reduzir o peso dos erros de digitação e o viés nos imóveis baratos.
4. **Intervalos conformes** (*conformal prediction*) para a faixa de erro, por cidade e tamanho.
5. **Atualizar os preços** com anúncios recentes ou pela inflação do aluguel (IVAR/FGV).
6. **Publicar o app** no Streamlit Community Cloud e gravar um GIF para o README.
