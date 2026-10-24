"""Constantes do projeto: caminhos, semente, colunas do modelo e regras de qualidade dos dados."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
ARQUIVO_BASE = RAIZ / 'Datasets' / 'house_data.csv'
ARQUIVO_LIMPO = RAIZ / 'Datasets' / 'imoveis_limpos.csv'
PASTA_MODELO = RAIZ / 'Model'
PASTA_REPORTS = RAIZ / 'reports'

SEMENTE = 42
ALVO = 'aluguel'

# Tradução das colunas do Kaggle para português
COLUNAS = {
    'city': 'cidade', 'area': 'area', 'rooms': 'quartos', 'bathroom': 'banheiros',
    'parking spaces': 'vagas', 'floor': 'andar', 'animal': 'aceita_animal', 'furniture': 'mobiliado',
    'hoa (R$)': 'condominio', 'rent amount (R$)': 'aluguel', 'property tax (R$)': 'iptu',
    'fire insurance (R$)': 'seguro_incendio', 'total (R$)': 'total',
}

CIDADES = ['São Paulo', 'Rio de Janeiro', 'Belo Horizonte', 'Porto Alegre', 'Campinas']

# Variáveis do modelo: só o que se sabe do imóvel ANTES de definir o aluguel
FEATURES_ORIGINAIS = ['cidade', 'area', 'quartos', 'banheiros', 'vagas', 'andar', 'aceita_animal', 'mobiliado',
                      'condominio', 'iptu']
FEATURES_CRIADAS = ['eh_casa', 'iptu_zero']
FEATURES = FEATURES_ORIGINAIS + FEATURES_CRIADAS

# Calculadas a partir do próprio aluguel: proibidas no modelo (vazamento)
PROIBIDAS = ['seguro_incendio', 'total']

# Regras de erro de digitação: usam só as características do imóvel (nunca o aluguel),
# por isso valem também para validar um imóvel novo no app
AREA_MAXIMA = 2_000          # m²: acima disso não é um imóvel residencial comum (o maior da base válida tem 2.000)
CONDOMINIO_MAX_M2 = 100      # R$/m²: o 99,9º percentil da base é ~67 R$/m²
IPTU_MAX_M2 = 100            # R$/m² por mês: o 99,9º percentil é ~30 R$/m²
ANDAR_MAXIMO = 60            # o prédio residencial mais alto da base tem 51 andares

N_JOBS = 2   # núcleos em paralelo: limitado para não esgotar a memória da máquina
