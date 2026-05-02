## Solução implementada:

### Escopo espacial e temporal

- **Local:** Belo Horizonte — estação pluviométrica CEMADEN **Pampulha** (`codEstacao` `310620007A`), alinhada ao arquivo INMET **A521 BELO HORIZONTE (PAMPULHA)** e às coordenadas da NASA POWER **−19,866° / −43,956°** (mesmo ponto da CEMADEN).
- **Ano:** 2024 (CEMADEN mensal + INMET anual + NASA horário agregado ao dia civil).

### Horizonte de previsão (≥ 24 h)

- **Target:** soma diária de `valorMedida` (mm) no **dia seguinte** ao dia das covariáveis — ou seja, com informações disponíveis até o fim do dia *D*, prevê-se a chuva acumulada no dia *D+1* em grade civil (`America/Sao_Paulo`). Isso implica horizonte **mínimo de 24 h** em relação ao início do dia alvo e é adequado a alertas operacionais em escala diária.

### Integração dos três datasets

| Fonte | Uso |
|--------|-----|
| **CEMADEN** | Soma diária de `valorMedida` por estação; o **target** é essa soma deslocada para o dia seguinte. |
| **INMET** | Médias diárias das variáveis horárias (UTC convertido para Minas) e **soma diária** da precipitação horária (primeira coluna de medições). |
| **NASA POWER** | API *hourly point* (`community=AG`), parâmetros exigidos no desafio. A API devolve precipitação como **PRECTOTCORR**; tratamos como **`PRECTOT`** no dataset (nome canônico do desafio). Agregação diária: médias horárias por variável; média móvel de 7 dias em `nasa_PRECTOT` como proxy de umidade/persistência. Cache em `data/nasa_hourly_2024_pampulha.csv`. |

### Processamento

- CEMADEN: leitura em **chunks** (`pandas.read_csv(..., chunksize=...)`) com `decimal=','`, apenas colunas necessárias — boas práticas para CSV grandes em memória limitada.
- NASA: download **mensal** para reduzir falhas de rede/timeout; junção dos meses após validação.

### Modelo e artefatos

- Modelo: **Gradient Boosting** (`sklearn.ensemble.GradientBoostingRegressor`), validação **cronológica** (80% primeiros dias / 20% últimos dias).
- Saídas numéricas e gráficas em `reports/` (nulos, tipos, distribuição da target, correlações, métricas, real *vs* previsto, importância de features, MSE por estágio — treino e teste).
- Dataset tabular final: `output/dataset_final.csv`.

### Como reproduzir

```powershell
cd Desafio_IA_Junior
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/build_dataset.py
python src/train.py
```

Ou em um único passo: `python src/run_all.py`.

A primeira execução de `build_dataset` chama a API NASA e grava cache em `data/` (pode levar alguns minutos).

### Interpretação das métricas

Precipitação diária é **altamente assimétrica** (muitos dias secos e poucos extremos); **R²** pode ficar baixo ou negativo mesmo com pipeline correto. As métricas em `reports/metricas_modelo.json` devem ser lidas junto com os gráficos e com uma linha de base ingênua (média ou persistência), conforme o relatório do avaliador pedir.
