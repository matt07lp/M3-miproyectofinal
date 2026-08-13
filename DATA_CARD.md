# Data Card — SEC Filings Corpus

## Descripción
Corpus de filings financieros 10-Q de la SEC, procesados para tareas de
NLP financiero: búsqueda semántica (RAG) y generación de instrucciones.

## Fuente
- **Origen**: SEC EDGAR (https://www.sec.gov/edgar)
- **Documentos**: 10-Q (informes trimestrales)
- **Tickers**: AAPL, AMZN, GOOGL, META, MSFT, NVDA (y otros según descarga)
- **Periodo**: ~2020-2024

## Pipeline de Procesamiento
1. Descarga vía `sec-edgar-downloader` → `full-submission.txt`
2. Extracción del bloque 10-Q desde archivos SGML multi-documento
3. Limpieza v6: HTML/SGML, binary, XBRL, fix fragmentación Inline XBRL
4. Chunking: ~4,000 chars, 800 chars de solapamiento
5. Dataset de instrucciones: 8 plantillas, matching por keywords

## Estructura de Archivos
sec_data/
├── sec-edgar-filings/   # ~150 full-submission.txt (crudos)
├── cleaned/             # textos limpios por ticker
├── chunks/
│   └── all_chunks.jsonl # chunks con metadatos
└── instructions/
    ├── train.jsonl       # 85% instrucciones
    └── val.jsonl         # 15% instrucciones

## Versionado
- Gestionado con **DVC** (Data Version Control)
- Remote: almacenamiento local (`~/dvc_storage`)
- Reproducible con `dvc pull` tras clonar el repositorio

## Licencias
- **Datos originales**: Documentos públicos de la SEC (dominio público,
  sin restricciones de copyright bajo ley federal de EE.UU.)
- **Dataset derivado**: CC BY 4.0

## PII
No se aplica redacción de PII. Los filings de la SEC son documentos
públicos y el corpus se enfoca exclusivamente en desempeño financiero
corporativo, no en datos personales.

## Sesgos Conocidos
1. **Sesgo de selección**: Solo empresas tecnológicas de gran
   capitalización. No representa al mercado completo.
2. **Sesgo temporal**: Periodo 2020-2024. Eventos como COVID-19 e
   inflación 2022 influyen en el lenguaje de los filings.
3. **Sesgo de formato**: Solo 10-Q (trimestrales), no incluye 10-K
   (anuales), 8-K ni proxy statements.
4. **Sesgo idiomático**: Todo el contenido está en inglés.

## Uso Previsto
- Investigación académica y educativa
- Desarrollo de sistemas RAG sobre información financiera
- Fine-tuning de modelos de lenguaje para dominio financiero

## Limitaciones
- No constituye asesoría financiera
- El campo `output` del dataset de instrucciones requiere generación
  posterior (manual o con LLM)
