# Desafio Técnico — Engenheiro de ML/IA Júnior

Seu desafio é, a partir de **3 datasets**, construir um **único dataset final** para o treinamento de um modelo capaz de **prever precipitação de chuva** para um local definido a partir dos dados fornecidos.

## Datasets disponíveis

### CEMADEN
Dataset de pluviometria contendo medições de pluviômetros em diversas cidades do estado de Minas Gerais.  
Cada arquivo dentro da pasta `CEMADEN` representa um **mês do ano de 2024**.

### INMET
Dataset meteorológico contendo dados climáticos de diferentes cidades do estado de Minas Gerais.  
Cada arquivo representa uma **cidade no ano de 2024**.

### NASA POWER
Dataset de medições satelitais obtido via API.

A documentação para requisição dos dados da NASA está disponível em:  
<https://power.larc.nasa.gov/docs/services/api/temporal/hourly/>

Os parâmetros obrigatórios a serem extraídos da NASA são:

- `PRECTOT`
- `PS`
- `ALLSKY_SFC_SW_DWN`
- `T2M`
- `T2MDEW`
- `RH2M`
- `WS10M`
- `WD10M`
- `U10M`
- `V10M`
- `QV2M`
- `T2MWET`

## Requisitos obrigatórios

- É obrigatório utilizar informações provenientes dos **três datasets** na construção do dataset final.
- É obrigatório utilizar a coluna `valorMedida` do CEMADEN como base da variável **target**.
- É obrigatório construir um **único dataset final**, pronto para modelagem.
- É obrigatório treinar **ao menos um modelo de predição**.
- É obrigatório fornecer um arquivo `requirements.txt` consistente com a solução entregue, permitindo a reprodução do ambiente utilizado.

## Análises e saídas obrigatórias

Além da construção do dataset e do treinamento do modelo, sua entrega deve conter, no mínimo:

- análise de valores nulos do dataset final;
- análise dos tipos das colunas do dataset final;
- distribuição da variável target;
- matriz de correlação entre as variáveis numéricas do dataset final;
- correlação das variáveis com a target;
- tabela com métricas do modelo, contendo no mínimo:
  - `MAE`
  - `MSE`
  - `RMSE`
  - `R²`
- gráfico mostrando a evolução de alguma métrica de erro ou curva de treinamento, quando aplicável ao modelo escolhido;
- gráfico comparando valores reais e previstos;
- análise de importância das features, quando aplicável ao modelo escolhido;
- curvas de treino e validação, quando aplicável ao modelo escolhido.

## Entrega

A entrega deve ser realizada por meio de um repositório no GitHub. A organização dos arquivos deve, obrigatoriamente, seguir a estrutura abaixo:

```text
orig/
data/
src/
output/
reports/
README.md
requirements.txt
```

## Observações

Fica a critério do candidato definir:

- a estratégia de tratamento e integração dos dados;
- o horizonte de previsão, desde que seja de no mínimo **24 horas** e justificado tecnicamente;
- as colunas utilizadas;
- as transformações aplicadas;
- o modelo adotado.

A documentação da entrega deve permitir compreender com clareza os critérios adotados ao longo da construção do dataset e do modelo.

**A avaliação considerará não apenas o desempenho final do modelo, mas também a coerência metodológica, a qualidade da construção do dataset, a capacidade de sustentar tecnicamente as decisões tomadas, o uso adequado de Git, a organização do repositório, a clareza do código e a reprodutibilidade da solução.**

---

## Solução implementada (neste repositório)

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
