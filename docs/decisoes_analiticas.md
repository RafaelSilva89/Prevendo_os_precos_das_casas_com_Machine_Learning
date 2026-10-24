# Registro de decisões analíticas

Este documento registra as principais decisões do projeto, no formato
**Contexto → Decisão → Alternativas consideradas → Consequência → Onde está**.
Cada decisão tem um código (DA-XX) para ser citada em commits e pull requests.

Os números vêm das saídas do notebook [`Notebooks/Prevendo os preços das casas com Machine Learning - v2.ipynb`](../Notebooks/).

| Código | Decisão |
|---|---|
| [DA-01](#da-01--as-5-cidades-em-vez-de-só-são-paulo) | As 5 cidades em vez de só São Paulo |
| [DA-02](#da-02--remoção-das-duplicatas-antes-da-separação) | Remoção das duplicatas antes da separação |
| [DA-03](#da-03--regras-de-erro-de-digitação-que-não-olham-o-aluguel) | Regras de erro de digitação que não olham o aluguel |
| [DA-04](#da-04--andar---como-casa) | Andar "-" como casa |
| [DA-05](#da-05--retirada-do-seguro-incêndio-e-do-total) | Retirada do seguro-incêndio e do total |
| [DA-06](#da-06--alvo-em-reais) | Alvo em reais |
| [DA-07](#da-07--referência-honesta-no-lugar-da-média-por-área-da-v1) | Referência honesta no lugar da "média por área" da v1 |
| [DA-08](#da-08--separação-602020-estratificada-por-cidade) | Separação 60/20/20 estratificada por cidade |
| [DA-09](#da-09--todas-as-12-variáveis) | Todas as 12 variáveis |
| [DA-10](#da-10--histgradientboosting-ajustado) | HistGradientBoosting ajustado |
| [DA-11](#da-11--faixa-de-erro-relativa) | Faixa de erro relativa |
| [DA-12](#da-12--teto-de-r-15-mil-mantido-e-avisado) | Teto de R$ 15 mil: mantido e avisado |

---

## DA-01 — As 5 cidades em vez de só São Paulo

- **Contexto:** a versão original filtrava só São Paulo (5.887 anúncios) como "produto mínimo viável" e descartava 45% da base.
- **Decisão:** usar as 5 cidades (10.322 imóveis depois da limpeza), com a cidade como variável (one-hot dentro do `Pipeline`). Decisão aprovada pelo usuário no plano da v2.
- **Alternativas consideradas:** manter só São Paulo (comparação direta com a v1, mas o modelo valeria só para SP); um modelo por cidade (pouco dado em Campinas, 853 anúncios).
- **Consequência:** um modelo e um app úteis para as 5 cidades, e o erro traduzido cidade por cidade. Nos imóveis de São Paulo do teste, a v2 tem MAE de R$ 1.478 e R² de 0,61, contra R$ 1.553 e 0,54 da v1.
- **Onde está:** `CIDADES` e `FEATURES` em `src/config.py`; Fase 04, itens 1 e 2.

## DA-02 — Remoção das duplicatas antes da separação

- **Contexto:** 358 linhas (3,3%) são anúncios repetidos, idênticos em todas as colunas. Com elas, o mesmo imóvel pode cair no treino e no teste.
- **Decisão:** remover as duplicatas exatas antes de separar treino, validação e teste.
- **Alternativas consideradas:** manter (versão original); separar mantendo as cópias juntas (por grupo).
- **Consequência:** o teste não tem imóveis "já vistos" pelo modelo.
- **Onde está:** `limpar_base` em `src/dados.py`; itens 2.4 e 2.6 da Fase 02.

## DA-03 — Regras de erro de digitação que não olham o aluguel

- **Contexto:** a base tem valores impossíveis: condomínio de R$ 1.117.000 num imóvel de 155 m², IPTU de R$ 313.700 em 42 m², área de 46.335 m² com 4 quartos e andar 301. A versão original os detectou (Grubbs e KNN, que marcou 1.070 linhas), mas não tratou nenhum. A regra do IQR marca de 6% a 10% de cada coluna, quase tudo imóveis caros de verdade.
- **Decisão:** remover só o que é impossível, por 4 regras que usam as características do imóvel e **nunca o aluguel**: área acima de 2.000 m², condomínio acima de R$ 100/m², IPTU acima de R$ 100/m² e andar acima de 60. Elas marcam 12 imóveis.
- **Alternativas consideradas:** remover pelo IQR (apagaria o mercado de alto padrão); regras com o aluguel, como "condomínio maior que o aluguel" (não poderiam ser aplicadas a um imóvel novo, que ainda não tem aluguel).
- **Consequência:** as mesmas regras limpam a base e recusam, no app, um imóvel fora do padrão. Erros no próprio aluguel (um zero a mais) continuam na base: a Fase 04 mostra os casos mais extremos.
- **Onde está:** limites em `src/config.py`; `regras_erro` em `src/dados.py`; item 2.6 da Fase 02; `app.py`.

## DA-04 — Andar "-" como casa

- **Contexto:** 23,0% dos anúncios têm andar "-". Desses, 84,4% não pagam condomínio, contra 3,6% dos imóveis com andar. A versão original transformava o "-" em andar 0 ("térreo").
- **Decisão:** criar `eh_casa` (1 quando o andar é "-") e manter `andar` = 0 para as casas. Criar também `iptu_zero`.
- **Alternativas consideradas:** andar 0 sem a marcação (versão original); andar vazio (exigiria preenchimento).
- **Consequência:** nas árvores, as duas variáveis não mudam o erro (andar 0 só aparece nas casas), mas deixam o modelo mais fácil de explicar e permitem a análise casa × apartamento (H5 e H7: a casa é mais barata por m², R$ 21,2 contra R$ 28,9).
- **Onde está:** `carregar_base` em `src/dados.py`; item 2.5.2 e item 4 da Fase 02; confirmação no item 4.3 da Fase 03.

## DA-05 — Retirada do seguro-incêndio e do total

- **Contexto:** o seguro-incêndio é ~1,3% do aluguel (98% dos anúncios entre 0,7% e 1,8%) e o total é a soma do aluguel com as taxas. Os dois são calculados a partir do aluguel.
- **Decisão:** retirá-los do modelo (a versão original já fazia isso) e mostrar o efeito no teste anti-vazamento.
- **Alternativas consideradas:** nenhuma razoável: eles não existem antes de definir o aluguel.
- **Consequência:** com o seguro, o MAE "cairia" de R$ 1.312 para R$ 108 (R² 0,991); com o total, para R$ 102. O erro real do modelo é de ~R$ 1.250.
- **Onde está:** `PROIBIDAS` em `src/config.py`; seção 🛡️ da Fase 02 e item 9 da Fase 03.

## DA-06 — Alvo em reais

- **Contexto:** o aluguel tem cauda longa (assimetria 1,80, média de R$ 3.964 contra mediana de R$ 2.750). Treinar com o log do aluguel costuma reduzir o erro relativo.
- **Decisão:** prever o aluguel em reais, como na versão original (escolha do usuário no plano da v2).
- **Alternativas consideradas:** treinar com o log do aluguel e voltar para reais na previsão.
- **Consequência:** os aluguéis altos pesam mais no treino; o modelo superestima os baratos em termos relativos (MPE de +18%). Por isso a avaliação também usa MAPE e erro por faixa de área, e o log aparece como próximo passo.
- **Onde está:** `ALVO` em `src/config.py`; item 2.7 da Fase 02.

## DA-07 — Referência honesta no lugar da "média por área" da v1

- **Contexto:** o "modelo da média" da versão original calculava a média do aluguel por área **nos próprios imóveis da validação**, ou seja, usava o gabarito. Reproduzindo a conta: R² de 0,625 e MAE de R$ 1.410, melhor que a Regressão Linear (R$ 1.432). Na v1, ele parecia o melhor modelo de todos.
- **Decisão:** usar como referência a média do **treino** (`DummyRegressor`) e mostrar, ao lado, o cálculo da v1.
- **Alternativas consideradas:** mediana do treino (mesma ideia, MAE um pouco menor).
- **Consequência:** a referência tem R² zero e MAE de R$ 2.566; o modelo final reduz o erro em 51% no teste.
- **Onde está:** item 5 (Média) da Fase 03.

## DA-08 — Separação 60/20/20 estratificada por cidade

- **Contexto:** São Paulo tem 55% dos anúncios; uma divisão aleatória poderia deixar Campinas com poucos imóveis em alguma parte.
- **Decisão:** treino, validação e teste com 60/20/20, estratificados por cidade; o teste é aberto uma única vez, no modelo final.
- **Alternativas consideradas:** a divisão da versão original (64/16/20, sem estratificação, com os modelos comparados só numa validação de 942 imóveis de SP, sem validação cruzada).
- **Consequência:** 6.192 / 2.065 / 2.065 imóveis, com São Paulo em 55% de cada parte e aluguel médio entre R$ 3.936 e R$ 3.977.
- **Onde está:** item 4.1 da Fase 03.

## DA-09 — Todas as 12 variáveis

- **Contexto:** Random Forest e Informação Mútua concordam no topo (área e IPTU); só `aceita_animal` tem informação mútua zero.
- **Decisão:** manter as 12 variáveis.
- **Alternativas consideradas:** só as 5 melhores da Informação Mútua (MAE de R$ 1.436 na validação cruzada, contra R$ 1.337 com todas).
- **Consequência:** cidade, mobília e andar acrescentam informação que as 5 maiores não têm.
- **Onde está:** item 4.3 da Fase 03.

## DA-10 — HistGradientBoosting ajustado

- **Contexto:** na validação cruzada, os 4 modelos de árvore ficam entre R$ 1.308 e R$ 1.346, diferença do tamanho do desvio (~R$ 35). O HistGradientBoosting é o 1º em todas as métricas. O ajuste fino não reduziu o erro na validação (R$ 1.312 contra R$ 1.309), mas reduziu o sobreajuste (diferença treino × validação de R$ 293 para R$ 117).
- **Decisão:** HistGradientBoosting ajustado (taxa de aprendizado 0,02, 400 árvores, 15 folhas, 40 imóveis por folha, l2 = 10), retreinado com treino + validação (`clone`).
- **Alternativas consideradas:** o modelo padrão (mesmo erro, mais sobreajuste); LightGBM (empatado, 2º lugar); Random Forest (escolha da versão original; na v2, ele tem forte sobreajuste: R$ 490 no treino contra R$ 1.343 na validação).
- **Consequência:** no teste, MAE de R$ 1.249, MAPE de 36,7% e R² de 0,65.
- **Onde está:** itens 5 a 7 da Fase 03; `Model/modelo_aluguel.json`.

## DA-11 — Faixa de erro relativa

- **Contexto:** o erro em reais cresce com o aluguel. Uma faixa fixa em reais (−R$ 1.933 a +R$ 2.294) cobre 81,6% do teste no total, mas 99,4% dos imóveis mais baratos e só 49,8% dos mais caros.
- **Decisão:** faixa **relativa**: o aluguel real fica entre 56% e 157% do previsto em 80% dos imóveis da validação.
- **Alternativas consideradas:** faixa fixa em reais; intervalos conformes por grupo (próximo passo).
- **Consequência:** cobertura de 81,3% no teste e entre 77,9% e 86,7% em todos os quartis de preço. É a faixa que o app mostra.
- **Onde está:** `faixa_relativa` e `cobertura_faixa` em `src/avaliacao.py`; item 8 da Fase 03; `app.py`.

## DA-12 — Teto de R$ 15 mil: mantido e avisado

- **Contexto:** R$ 15.000 é o 3º valor de aluguel mais comum da base (231 anúncios, 2,2%), e só 12 anúncios passam disso. A pilha num valor redondo, sem a queda gradual esperada nos aluguéis caros, indica que a coleta limitou o preço: boa parte desses anúncios custa "R$ 15 mil ou mais".
- **Decisão:** manter os 231 anúncios, documentar o teto e mostrar um aviso no app quando a sugestão passa de R$ 12 mil.
- **Alternativas consideradas:** remover os anúncios de R$ 15.000 (apagaria o alto padrão e criaria outro viés); tratar como valor censurado, com modelo de sobrevivência (próximo passo).
- **Consequência:** para imóveis de alto padrão, o modelo tende a subestimar o preço; no gráfico previsto × real, esses anúncios formam a linha horizontal do topo.
- **Onde está:** item 2.7 da Fase 02; limites no fim do notebook; `app.py`.
