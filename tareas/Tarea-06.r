# ==============================================================================
# ECON 520 - Tarea 06: Edgar API (SEC) & Text Mining Financiero
# Alumno: Simón Guercovich
# Corpus: Reportes 20-F de emisores argentinos en NYSE (YPF, GGAL, PAM)
# ==============================================================================

library(tidyverse)
library(tidytext)
library(jsonlite)
library(httr)
library(scales)

# ------------------------------------------------------------------------------
# 1. Definicion de empresas y User-Agent (SEC EDGAR)
# ------------------------------------------------------------------------------

sec_headers <- user_agent("Simon Guercovich sguercovich@economicas.uba.ar")

# CIK oficiales para YPF, Grupo Financiero Galicia y Pampa Energia
empresas <- tibble(
  ticker = c("YPF", "GGAL", "PAM"),
  nombre = c("YPF S.A.", "Grupo Financiero Galicia", "Pampa Energía"),
  cik    = c("0000904851", "0001118590", "0001479893")
)

empresas

# ------------------------------------------------------------------------------
# 2. Descarga de los ultimos reportes anuales (Formulario 20-F)
# ------------------------------------------------------------------------------

descargar_ultimo_20f <- function(cik_num, ticker_name) {
  message(paste("Buscando último 20-F para:", ticker_name))
  
  url_submissions <- paste0("https://data.sec.gov/submissions/CIK", cik_num, ".json")
  res <- GET(url_submissions, sec_headers)
  
  if (status_code(res) != 200) {
    warning(paste("No se pudo acceder a la metadata de:", ticker_name))
    return(NULL)
  }
  
  data_sec <- fromJSON(content(res, as = "text", encoding = "UTF-8"))
  recent <- as_tibble(data_sec$filings$recent)
  
  # Filtramos el formulario 20-F mas reciente
  ultimo_20f <- recent %>%
    filter(form == "20-F") %>%
    slice(1)
  
  if (nrow(ultimo_20f) == 0) return(NULL)
  
  accession_num <- gsub("-", "", ultimo_20f$accessionNumber)
  doc_name <- ultimo_20f$primaryDocument
  cik_sin_ceros <- as.numeric(cik_num)
  
  doc_url <- paste0("https://www.sec.gov/Archives/edgar/data/", cik_sin_ceros, "/", accession_num, "/", doc_name)
  
  res_doc <- GET(doc_url, sec_headers)
  if (status_code(res_doc) != 200) return(NULL)
  
  raw_html <- content(res_doc, as = "text", encoding = "UTF-8")
  
  # Limpieza de tags HTML/XML para obtener texto plano
  texto_limpio <- raw_html %>%
    str_replace_all("<[^>]+>", " ") %>%
    str_replace_all("&[a-zA-Z]+;", " ") %>%
    str_replace_all("\\s+", " ")
  
  tibble(
    ticker = ticker_name,
    filing_date = ultimo_20f$filingDate,
    report_date = ultimo_20f$reportDate,
    texto = texto_limpio
  )
}

# Ejecutamos la descarga secuencial
reportes_list <- list()
for (i in seq_len(nrow(empresas))) {
  reportes_list[[i]] <- descargar_ultimo_20f(empresas$cik[i], empresas$ticker[i])
  Sys.sleep(0.4)
}

# Consolidamos la tabla final
reportes_df <- bind_rows(reportes_list) %>%
  left_join(empresas %>% select(ticker, nombre), by = "ticker")

# Verificamos volumen de palabras procesadas
reportes_df %>%
  mutate(total_palabras = str_count(texto, "\\S+")) %>%
  select(ticker, nombre, filing_date, total_palabras)


# ------------------------------------------------------------------------------
# 3. Analisis de Texto: Incidencia de palabras macroeconomicas
# ------------------------------------------------------------------------------

macro_terms <- c(
  "inflation", "devaluation", "exchange controls", "default", 
  "restructuring", "imf", "capital controls", "price controls", 
  "country risk", "embi"
)

incidencia_macro <- reportes_df %>%
  crossing(term = macro_terms) %>%
  mutate(frecuencia = str_count(str_to_lower(texto), fixed(term))) %>%
  select(ticker, nombre, term, frecuencia)

# Tabla resumen
incidencia_macro %>%
  pivot_wider(names_from = ticker, values_from = frecuencia) %>%
  arrange(desc(YPF))

# Grafico 1: Variables macroeconomicas
ggplot(incidencia_macro, aes(x = reorder(term, frecuencia, sum), y = frecuencia, fill = ticker)) +
  geom_col(position = "dodge") +
  coord_flip() +
  scale_fill_brewer(palette = "Set1") +
  labs(
    title = "Menciones de Variables Macroeconómicas en Reportes 20-F",
    subtitle = "YPF, Grupo Financiero Galicia y Pampa Energía (SEC EDGAR)",
    x = "Término macroeconómico",
    y = "Cantidad de menciones",
    fill = "Empresa"
  ) +
  theme_minimal(base_size = 11)


# ------------------------------------------------------------------------------
# 4. Palabras distintivas por empresa (TF-IDF) [Corregido]
# ------------------------------------------------------------------------------

data("stop_words")

# Tokenizacion limpia a nivel palabra
tidy_reportes <- reportes_df %>%
  select(ticker, nombre, texto) %>%
  unnest_tokens(word, texto) %>%
  anti_join(stop_words, by = "word") %>%
  filter(!str_detect(word, "^[0-9]+$"), nchar(word) > 2)

# Calculo de TF-IDF
reportes_tfidf <- tidy_reportes %>%
  count(ticker, word, sort = TRUE) %>%
  bind_tf_idf(word, ticker, n)

# Filtramos las 10 palabras con mayor tf_idf por cada firma
top_tfidf <- reportes_tfidf %>%
  group_by(ticker) %>%
  slice_max(tf_idf, n = 10, with_ties = FALSE) %>%
  ungroup()

# Grafico 2: TF-IDF corregido
top_tfidf %>%
  mutate(word = reorder_within(word, tf_idf, ticker)) %>%
  ggplot(aes(x = tf_idf, y = word, fill = ticker)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ ticker, scales = "free_y", ncol = 3) +
  scale_y_reordered() +
  labs(
    title = "Palabras Distintivas por Empresa (TF-IDF)",
    subtitle = "Términos específicos del modelo de negocio en reportes 20-F",
    x = "TF-IDF",
    y = NULL
  ) +
  theme_minimal(base_size = 11) +
  theme(
    strip.text = element_text(face = "bold", size = 11),
    axis.text.y = element_text(size = 9)
  )


# ------------------------------------------------------------------------------
# 5. Analisis de Sentimiento Financiero (Loughran-McDonald)
# ------------------------------------------------------------------------------

loughran <- get_sentiments("loughran")

sentimiento_financiero <- tidy_reportes %>%
  inner_join(loughran, by = "word") %>%
  count(ticker, sentiment, sort = TRUE)

# Grafico 3: Frecuencia absoluta de sentimientos
ggplot(sentimiento_financiero, aes(x = reorder(sentiment, n, sum), y = n, fill = ticker)) +
  geom_col(position = "dodge") +
  coord_flip() +
  scale_fill_brewer(palette = "Dark2") +
  labs(
    title = "Sentimiento Financiero por Empresa (Loughran-McDonald)",
    subtitle = "Frecuencia de términos según categoría de riesgo y contexto corporativo",
    x = "Categoría",
    y = "Frecuencia de palabras",
    fill = "Empresa"
  ) +
  theme_minimal(base_size = 11)

# Grafico 4: Composicion relativa porcentual
sentimiento_financiero %>%
  group_by(ticker) %>%
  mutate(proporcion = n / sum(n)) %>%
  ggplot(aes(x = ticker, y = proporcion, fill = sentiment)) +
  geom_col(position = "fill") +
  scale_y_continuous(labels = scales::percent_format()) +
  scale_fill_brewer(palette = "Set2") +
  labs(
    title = "Composición Relativa de Sentimiento Financiero",
    subtitle = "Distribución porcentual en los reportes anuales",
    x = "Empresa",
    y = "Porcentaje",
    fill = "Sentimiento"
  ) +
  theme_minimal(base_size = 11)