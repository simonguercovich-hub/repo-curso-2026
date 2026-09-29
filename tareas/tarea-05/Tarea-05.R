# Tarea 05 - Text Mining

library(tidyverse)
library(tidytext)
library(topicmodels)
library(igraph)
library(ggraph)

# Carga de datos
df <- read_csv("DATA-T9-mckinsey-mind-the-gap-articles-20251020.csv")

glimpse(df)


# 1. Exploracion del corpus y comparabilidad -------------------------------

# Renombramos la primera columna para tener un id comodo
df <- df %>% 
  rename(id = ...1)

# Cantidad de articulos y rango temporal
nrow(df)
range(df$date)

# Vemos algunos titulos para entender la tematica
df %>% 
  select(id, title, date) %>% 
  head(10)

# Longitud en palabras por articulo para ver si son comparables
df <- df %>%
  mutate(n_words = str_count(article_text, "\\S+"))

summary(df$n_words)

# El corpus tiene 146 articulos del newsletter "Mind the Gap" de McKinsey (marzo 2022 a octubre 2025).
# Tratan sobre habitos, consumo, trabajo y tecnologia en la Generacion Z.
# En cuanto a comparabilidad, la longitud es bastante homogenea: la mediana es de 611 palabras
# y el 50% central (IQR) se ubica entre 540 y 690 palabras. Son perfectamente comparables entre si.


# 2. Tokenizacion y palabras destacadas ------------------------------------

# Pasamos a formato tidy: una palabra por fila
tidy_words <- df %>%
  select(id, title, date, article_text) %>%
  unnest_tokens(word, article_text)

# Sacamos stopwords en ingles y numeros sueltos
data(stop_words)

tidy_clean <- tidy_words %>%
  anti_join(stop_words, by = "word") %>%
  filter(!str_detect(word, "^[0-9]+$"))

# Palabras mas frecuentes de todo el corpus
tidy_clean %>%
  count(word, sort = TRUE) %>%
  head(15)

# Grafico de palabras mas frecuentes
tidy_clean %>%
  count(word, sort = TRUE) %>%
  slice_max(n, n = 15) %>%
  ggplot(aes(x = reorder(word, n), y = n)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(title = "Palabras mas frecuentes del corpus", x = NULL, y = "Frecuencia") +
  theme_minimal()

# Vemos que dominan palabras como gen, z, zers, workers, consumers, time.
# Tienen sentido con el eje del newsletter. Aparecen tambien terminos corporativos (mckinsey, partner).


# 3. Analisis de sentimiento -----------------------------------------------

# A. Sentimiento binario con Bing (positivo vs negativo)
bing <- get_sentiments("bing")

sentiment_bing <- tidy_clean %>%
  inner_join(bing, by = "word") %>%
  count(id, title, date, sentiment) %>%
  pivot_wider(names_from = sentiment, values_from = n, values_fill = 0) %>%
  mutate(score_bing = positive - negative)

# Los 5 articulos mas positivos
sentiment_bing %>% arrange(desc(score_bing)) %>% select(title, score_bing) %>% head(5)

# Los 5 articulos mas negativos
sentiment_bing %>% arrange(score_bing) %>% select(title, score_bing) %>% head(5)

# B. Sentimiento con graduacion con AFINN (-5 a +5)
afinn <- get_sentiments("afinn")

sentiment_afinn <- tidy_clean %>%
  inner_join(afinn, by = "word") %>%
  group_by(id, title, date) %>%
  summarise(score_afinn = mean(value), .groups = "drop")

summary(sentiment_afinn$score_afinn)

# Vemos cuantos articulos quedan positivos vs negativos segun AFINN
sentiment_afinn %>%
  mutate(tipo = if_else(score_afinn > 0, "Positivo", "Negativo")) %>%
  count(tipo)

# En ambos diccionarios el tono es marcadamente positivo (en AFINN la media es ~0.78 y hay casi todos positivos).
# Es coherente con notas de consultoria dirigidas a management. 
# Las pocas notas con puntaje negativo tocan temas de estres laboral, burnout o incertidumbre economica.


# 4. Topic Modeling con LDA (k = 10 y k = 15) ------------------------------

# Para que el LDA no agrupe todo alrededor de palabras obvias que estan en casi todas las notas,
# armamos un filtro con las palabras corporativas y las variantes de Gen Z
custom_stops <- tibble(word = c("gen", "z", "zers", "mckinsey", "partner", 
                                "percent", "global", "it's", "it’s", "edition"))

tidy_para_lda <- tidy_clean %>%
  anti_join(custom_stops, by = "word")

# Armamos la matriz documento-termino (DTM)
dtm <- tidy_para_lda %>%
  count(id, word) %>%
  cast_dtm(id, word, n)

# Corremos los dos modelos
set.seed(1234)
lda_10 <- LDA(dtm, k = 10, control = list(seed = 1234))
lda_15 <- LDA(dtm, k = 15, control = list(seed = 1234))

# Grafico de top terminos para k = 10
tidy(lda_10, matrix = "beta") %>%
  group_by(topic) %>%
  slice_max(beta, n = 8) %>%
  ungroup() %>%
  mutate(term = reorder_within(term, beta, topic)) %>%
  ggplot(aes(beta, term, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free", ncol = 5) +
  scale_y_reordered() +
  labs(title = "Topicos con k = 10", x = "Beta", y = NULL) +
  theme_minimal()

# Grafico de top terminos para k = 15
tidy(lda_15, matrix = "beta") %>%
  group_by(topic) %>%
  slice_max(beta, n = 8) %>%
  ungroup() %>%
  mutate(term = reorder_within(term, beta, topic)) %>%
  ggplot(aes(beta, term, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free", ncol = 5) +
  scale_y_reordered() +
  labs(title = "Topicos con k = 15", x = "Beta", y = NULL) +
  theme_minimal()

# Comparacion k=10 vs k=15:
# Con k=10 los temas son generales pero claros: trabajo/oficina, consumo navideño, salud/bienestar y autos/movilidad.
# Con k=15 se desagregan mejor algunos subtemas (separa moda de tecnologia o salud mental de retail),
# aunque algunos topicos empiezan a compartir palabras genericas como people o time.


# 5. AVANZADO: Bigramas y Negaciones ---------------------------------------

# Armamos bigramas (pares de palabras)
bigramas <- df %>%
  unnest_tokens(bigram, article_text, token = "ngrams", n = 2) %>%
  separate(bigram, c("word1", "word2"), sep = " ")

# Filtramos stopwords en ambas posiciones
bigramas_clean <- bigramas %>%
  filter(!word1 %in% stop_words$word,
         !word2 %in% stop_words$word,
         !str_detect(word1, "^[0-9]+$"),
         !str_detect(word2, "^[0-9]+$"))

# Conteo de bigramas mas frecuentes
bigramas_top <- bigramas_clean %>%
  count(word1, word2, sort = TRUE)

head(bigramas_top, 15)

# Red de bigramas mas frecuentes (n >= 8)
red_bigramas <- bigramas_top %>%
  filter(n >= 8) %>%
  graph_from_data_frame()

set.seed(1234)
ggraph(red_bigramas, layout = "fr") +
  geom_edge_link(aes(edge_alpha = n), edge_color = "steelblue", show.legend = FALSE) +
  geom_node_point(color = "midnightblue", size = 3) +
  geom_node_text(aes(label = name), vjust = 1.4, hjust = 1, size = 3.5, repel = TRUE) +
  labs(title = "Red de Bigramas en Mind the Gap",
       subtitle = "Pares de palabras que aparecen juntos al menos 8 veces") +
  theme_void()

# Analisis de Negaciones ("NOT word")
# Buscamos cuando una palabra con carga afectiva viene precedida por una negacion
palabras_negacion <- c("not", "no", "never", "without")

negaciones <- bigramas %>%
  filter(word1 %in% palabras_negacion) %>%
  inner_join(afinn, by = c("word2" = "word")) %>%
  count(word1, word2, value, sort = TRUE) %>%
  mutate(score_invertido = -value * n)

head(negaciones, 10)

# Grafico de palabras cuyo sentimiento cambia por la negacion
negaciones %>%
  slice_max(abs(score_invertido), n = 12) %>%
  mutate(par = reorder(paste(word1, word2), score_invertido)) %>%
  ggplot(aes(x = par, y = score_invertido, fill = score_invertido > 0)) +
  geom_col(show.legend = FALSE) +
  coord_flip() +
  scale_fill_manual(values = c("firebrick", "forestgreen")) +
  labs(
    title = "Sentimiento invertido por negaciones",
    subtitle = "Palabras con puntaje AFINN precedidas por not, no, never o without",
    x = NULL,
    y = "Impacto del sentimiento corregido"
  ) +
  theme_minimal()