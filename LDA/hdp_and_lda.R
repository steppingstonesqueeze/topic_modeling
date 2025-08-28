
# ----- Install (first time) -----
# install.packages(c("gutenbergr", "dplyr", "stringr", "tidyr", "tidytext",
#                    "tm", "topicmodels", "purrr", "tibble"))
# # For HDP (optional, GitHub):
# install.packages("remotes"); remotes::install_github("nicolaroberts/hdp")

library(gutenbergr)
library(dplyr)
library(stringr)
library(tidyr)
library(tidytext)
library(tm)
library(topicmodels)
library(purrr)
library(tibble)

get_author_corpus <- function(author_name, n_books = 10, language = "en") {
  # 1) Find candidate works by author (filter language, dedupe by title)
  works <- gutenberg_works() %>%
    filter(str_detect(author, regex(author_name, ignore_case = TRUE))) %>%
    filter(is.na(language) | language == language) %>%
    filter(!is.na(title)) %>%
    distinct(title, .keep_all = TRUE) %>%        # drop alt editions
    slice_head(n = n_books) %>%
    select(gutenberg_id, title, author, language)
  
  if (nrow(works) == 0) stop("No works found for that author/language.")
  
  # 2) Download text & strip boilerplate
  texts <- gutenberg_download(works$gutenberg_id, meta_fields = "title",
                              strip = TRUE) %>%
    rename(book_title = title)
  
  # 3) Tokenize and clean (tidytext)
  tidy_tokens <- texts %>%
    unnest_tokens(word, text) %>%
    filter(!str_detect(word, "^\\d+$")) %>%
    anti_join(stop_words, by = "word") %>%
    filter(str_detect(word, "^[a-z']+$"))
  
  # 4) Build a DTM at the book level (one doc per book)
  counts <- tidy_tokens %>%
    count(book_title, word, sort = FALSE)
  
  dtm <- counts %>%
    cast_dtm(document = book_title, term = word, value = n)
  
  # Drop empty docs if any
  nonempty <- slam::row_sums(dtm) > 0
  dtm <- dtm[nonempty, ]
  
  list(works = works, dtm = dtm, tokens = tidy_tokens)
}

run_lda <- function(dtm, k = 10, seed = 123) {
  set.seed(seed)
  topicmodels::LDA(dtm, k = k, control = list(seed = seed))
}

top_terms <- function(model, dtm, topn = 10) {
  beta <- terms(model, topn)  # topic -> top terms (character matrix)
  as_tibble(beta, .name_repair = "minimal") %>%
    mutate(topic = paste0("Topic_", seq_len(n()))) %>%
    relocate(topic)
}

# ----- Optional: HDP in pure R (Gibbs) -----
run_hdp <- function(dtm, burnin = 2000, n = 100, space = 50,
                    cos.merge = 0.9, min.cluster.size = 5, seed = 123) {
  # Requires: remotes::install_github("nicolaroberts/hdp")
  library(hdp)
  
  X <- as.matrix(dtm)
  storage.mode(X) <- "integer"
  set.seed(seed)
  
  st  <- hdp_quick_init(X)
  ch  <- hdp_posterior(st, burnin = burnin, n = n, space = space)
  mdl <- hdp_extract_components(ch, cos.merge = cos.merge,
                                min.cluster.size = min.cluster.size)
  
  list(
    model = mdl,
    topic_word = comp_categ_distn(mdl),  # K x V
    doc_topic  = comp_dp_distn(mdl)      # D x K
  )
}

# ---------------- Example usage ----------------
# Change author_name to whoever you want (e.g., "Mark Twain", "H. G. Wells", "Rabindranath Tagore")
author_name <- "Wodehouse"

data <- get_author_corpus(author_name, n_books = 10, language = "en")

# LDA with k=8 topics
lda_fit <- run_lda(data$dtm, k = 8)
lda_top <- top_terms(lda_fit, data$dtm, topn = 12)
print(lda_top)

# If you want HDP instead of LDA:
# hdp_res <- run_hdp(data$dtm, burnin=2000, n=80, space=50, cos.merge=0.9)
# str(hdp_res$doc_topic)  # document-topic
# str(hdp_res$topic_word) # topic-word

# Document-topic proportions for LDA:
doc_topics <- as_tibble(posterior(lda_fit)$topics, .name_repair = "minimal") %>%
  mutate(document = rownames(data$dtm)) %>%
  relocate(document)
print(doc_topics)
