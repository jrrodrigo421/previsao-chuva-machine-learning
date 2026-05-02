## Solução implementada

### Escopo espacial e temporal

- **Local:** Belo Horizonte — estação pluviométrica CEMADEN **Pampulha** (`codEstacao` `310620007A`), alinhada ao arquivo INMET **A521 BELO HORIZONTE (PAMPULHA)** e às coordenadas da NASA POWER **−19,866° / −43,956°** (mesmo ponto da CEMADEN).
- **Ano:** 2024 (CEMADEN mensal + INMET anual + NASA POWER horário → agregação ao **dia civil** em **`America/Sao_Paulo`**).

### Horizonte de previsão (≥ 24 h)

- **Target:** soma diária de `valorMedida` (mm) no **dia seguinte** ao das covariáveis: com dados até o fim do dia *D*, prevê-se a chuva acumulada em *D+1* (mesma grade civil MG). Satisfaz horizonte mínimo de 24 h e é adequado a uso operacional diário.

### Integração dos três datasets

| Fonte | Uso |
|--------|-----|
| **CEMADEN** | Soma diária de `valorMedida`; **target** = soma do dia seguinte. **Features autorregressivas** (lags e médias móveis até *D*, sem vazamento) derivadas da própria série diária. |
| **INMET** | UTC (`Hora UTC`) → **`America/Sao_Paulo`**; precipitação horária somada ao dia; demais medições médias ao dia; colunas estáveis com nomes semânticos (`inmet_precip`, `inmet_temp_ar`, …). **Direção do vento** do INMET é omitida (graus são circulares; vento aparece vetorialmente na NASA como `U10M`/`V10M`). |
| **NASA POWER** | API *hourly* *point*, `community=AG`, **`time-standard=UTC`** para o horário ficar auditável ([doc da Hourly API](https://power.larc.nasa.gov/docs/services/api/temporal/hourly/) — antes do `time-standard=` o padrão da API era LST, que não coincide com fusos políticos). Resposta pode trazer **`PRECTOTCORR`**; renomeamos para **`PRECTOT`** nos nomes de colunas do dataset. Agregação diária em MG; média móvel de 7 dias em **`nasa_PRECTOT_roll7`**. Cache: `data/nasa_hourly_2024_pampulha_utc.csv`. |

### Processamento

- **CEMADEN:** chunks + `decimal=','`; **`datahora` interpretada como UTC** e convertida para MG antes do agrupamento por dia.
- **NASA:** download mensal com retries/backoff (evita timeouts em um ano inteiro numa única requisição).

### Modelo e artefatos

- **Modelo:** `GradientBoostingRegressor` (usa `staged_predict` para o gráfico de erro por estágio exigido no desafio; ver [User Guide sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html) vs. `HistGradientBoostingRegressor`, mais indicado quando *n_samples* é bem maior).
- **Holdout cronológico:** 80% / 20% (primeiros vs. últimos dias) para métricas principais (`MAE`, `MSE`, `RMSE`, `R²`) e para gráficos *real × previsto* e curvas por estágio.
- **Linhas de base** no mesmo holdout: média histórica do treino e **persistência** (chuva do dia *D* como palpite para *D+1*, coluna `cemaden_lag1`).
- **Validação cronológica adicional:** `TimeSeriesSplit` — média e desvio por fold ficam consolidados em `metricas_modelo.json` e detalhe por fold em `metricas_cv_fold.csv`.
- **Saídas** em `reports/`: nulos, tipos, distribuição da target, correlações, métricas, *real × previsto*, importâncias, MSE por estágio treino/teste.
- **Dataset:** `output/dataset_final.csv`.
- **Modelo gravado:** após cada `train.py` bem-sucedido (sem `--no-save`), ficam **`output/modelo_chuva.joblib`** e **`modelo_chuva.metadata.json`**; versões antigas ficam como `*.backup`. Inferência: `python src/predict.py --input ...`. Opções `--no-save`, `--eval-only`, `--compare-prev` — ver **`plano_modelo_inteligencia_persistente.md`**.
### Como reproduzir

```powershell
cd desafio_ia
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/build_dataset.py
python src/train.py
```

Ou: `python src/run_all.py`.

A primeira vez que `build_dataset` roda, a NASA é baixada e o cache em `data/` pode levar vários minutos. O nome do cache (**`…_pampulha_utc.csv`**) é **diferente** do arquivo antigo `…_pampulha.csv`, de modo que a primeira rodada depois dessa mudança refaz o download já com **`time-standard=UTC`** sem depender de apagar caches manualmente.

### Persistência / inferência

Resumo já implementado:

- **`python src/train.py`** — guarda modelo + metadata (substitui ciclo anterior com backup).
- **`python src/predict.py --input seu.csv`** (opcional **`--output output/previsoes.csv`**) — só carrega modelo; entrada precisa ter todas as colunas `feature_columns` do JSON.

### Interpretação das métricas

Chuva diária é assimétrica; **R²** pode ficar baixo. Compare sempre com **`holdout_baseline_mean_*`**, **`holdout_baseline_persist_*`** e com os agregados **`cv_*_mean`** / **`cv_*_std`** no `metricas_modelo.json`, além dos gráficos — não apenas um número isolado.
