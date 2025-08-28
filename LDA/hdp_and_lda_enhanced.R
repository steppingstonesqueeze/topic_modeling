# Enhanced Topic Modeling with Comprehensive Visualizations and File Saving
# ----- Install packages (first time) -----
# install.packages(c("gutenbergr", "dplyr", "stringr", "tidyr", "tidytext",
#                    "tm", "topicmodels", "purrr", "tibble", "ggplot2", 
#                    "plotly", "wordcloud", "RColorBrewer", "viridis",
#                    "corrplot", "pheatmap", "DT", "htmlwidgets"))
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
library(ggplot2)
library(plotly)
library(wordcloud)
library(RColorBrewer)
library(viridis)
library(corrplot)
library(pheatmap)
library(DT)
library(htmlwidgets)

# Create timestamped output directory
create_output_dir <- function() {
  timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
  dir_name <- paste0("topic_modeling_results_", timestamp)
  dir.create(dir_name, showWarnings = FALSE)
  return(dir_name)
}

get_author_corpus <- function(author_name, n_books = 10, language = "en", min_words = 1000) {
  cat("Searching for works by", author_name, "...\n")
  
  # 1) Find candidate works by author (filter language, dedupe by title)
  works <- gutenberg_works() %>%
    filter(str_detect(author, regex(author_name, ignore_case = TRUE))) %>%
    filter(is.na(language) | language == language) %>%
    filter(!is.na(title)) %>%
    distinct(title, .keep_all = TRUE) %>%        # drop alt editions
    slice_head(n = n_books) %>%
    select(gutenberg_id, title, author, language)
  
  if (nrow(works) == 0) stop("No works found for that author/language.")
  
  cat("Found", nrow(works), "works. Downloading texts...\n")
  
  # 2) Download text & strip boilerplate
  texts <- gutenberg_download(works$gutenberg_id, meta_fields = "title",
                              strip = TRUE) %>%
    rename(book_title = title)
  
  # 3) Tokenize and clean (tidytext)
  tidy_tokens <- texts %>%
    unnest_tokens(word, text) %>%
    filter(!str_detect(word, "^\\d+$")) %>%
    anti_join(stop_words, by = "word") %>%
    filter(str_detect(word, "^[a-z']+$")) %>%
    filter(nchar(word) > 2)  # Remove very short words
  
  # 4) Filter books with sufficient content
  book_word_counts <- tidy_tokens %>%
    count(book_title) %>%
    filter(n >= min_words)
  
  if (nrow(book_word_counts) == 0) {
    stop("No books have enough words (min:", min_words, ")")
  }
  
  tidy_tokens <- tidy_tokens %>%
    filter(book_title %in% book_word_counts$book_title)
  
  # 5) Build a DTM at the book level (one doc per book)
  counts <- tidy_tokens %>%
    count(book_title, word, sort = FALSE)
  
  dtm <- counts %>%
    cast_dtm(document = book_title, term = word, value = n)
  
  # Drop empty docs if any
  nonempty <- slam::row_sums(dtm) > 0
  dtm <- dtm[nonempty, ]
  
  cat("Final corpus:", nrow(dtm), "documents,", ncol(dtm), "unique terms\n")
  
  list(works = works, dtm = dtm, tokens = tidy_tokens, word_counts = book_word_counts)
}

run_lda <- function(dtm, k = 10, seed = 123, iterations = 2000) {
  cat("Running LDA with k =", k, "topics...\n")
  set.seed(seed)
  topicmodels::LDA(dtm, k = k, control = list(seed = seed))
}

extract_lda_results <- function(model, dtm, topn = 15) {
  # Topic-word distributions (beta)
  topic_words <- tidy(model, matrix = "beta") %>%
    arrange(topic, desc(beta))
  
  # Document-topic distributions (gamma)  
  doc_topics <- tidy(model, matrix = "gamma") %>%
    arrange(document, desc(gamma))
  
  # Top terms per topic
  top_terms <- topic_words %>%
    group_by(topic) %>%
    slice_max(beta, n = topn) %>%
    ungroup() %>%
    arrange(topic, desc(beta))
  
  # Topic prevalence
  topic_prevalence <- doc_topics %>%
    group_by(topic) %>%
    summarise(
      mean_gamma = mean(gamma),
      median_gamma = median(gamma),
      max_gamma = max(gamma),
      n_dominant_docs = sum(gamma > 0.5),
      .groups = "drop"
    ) %>%
    arrange(desc(mean_gamma))
  
  list(
    topic_words = topic_words,
    doc_topics = doc_topics,
    top_terms = top_terms,
    topic_prevalence = topic_prevalence
  )
}

# Visualization functions
plot_top_terms <- function(top_terms, n_topics = NULL, terms_per_topic = 10) {
  if (is.null(n_topics)) n_topics <- max(top_terms$topic)
  
  plot_data <- top_terms %>%
    filter(topic <= n_topics) %>%
    group_by(topic) %>%
    slice_max(beta, n = terms_per_topic) %>%
    ungroup() %>%
    mutate(
      term = reorder_within(term, beta, topic),
      topic = paste("Topic", topic)
    )
  
  p <- ggplot(plot_data, aes(beta, term, fill = factor(topic))) +
    geom_col(show.legend = FALSE) +
    facet_wrap(~ topic, scales = "free", ncol = 3) +
    scale_y_reordered() +
    scale_fill_viridis_d() +
    labs(
      title = "Top Terms per Topic (LDA)",
      subtitle = paste("Showing top", terms_per_topic, "terms for each topic"),
      x = "Beta (topic-word probability)",
      y = "Terms"
    ) +
    theme_minimal() +
    theme(
      strip.text = element_text(size = 12, face = "bold"),
      plot.title = element_text(size = 16, face = "bold"),
      axis.text.y = element_text(size = 9)
    )
  
  return(p)
}

plot_topic_prevalence <- function(topic_prevalence) {
  p <- ggplot(topic_prevalence, aes(x = reorder(paste("Topic", topic), mean_gamma), 
                                   y = mean_gamma)) +
    geom_col(fill = "steelblue", alpha = 0.8) +
    geom_text(aes(label = paste0(round(mean_gamma * 100, 1), "%")), 
              hjust = -0.1, size = 3) +
    coord_flip() +
    labs(
      title = "Topic Prevalence Across Corpus",
      subtitle = "Average gamma (document-topic probability) per topic",
      x = "Topic",
      y = "Mean Gamma"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 14, face = "bold"),
      axis.text.x = element_text(angle = 45, hjust = 1)
    )
  
  return(p)
}

plot_document_topics <- function(doc_topics, top_n_docs = 15) {
  # Get documents with most skewed topic distributions
  doc_entropy <- doc_topics %>%
    group_by(document) %>%
    summarise(
      entropy = -sum(gamma * log(gamma + 1e-10)),
      max_gamma = max(gamma),
      dominant_topic = topic[which.max(gamma)],
      .groups = "drop"
    ) %>%
    arrange(entropy)  # Lower entropy = more focused
  
  selected_docs <- head(doc_entropy, top_n_docs)$document
  
  plot_data <- doc_topics %>%
    filter(document %in% selected_docs) %>%
    mutate(
      document = str_trunc(document, 30),
      topic = paste("Topic", topic)
    )
  
  p <- ggplot(plot_data, aes(x = gamma, y = reorder(document, gamma), fill = topic)) +
    geom_col() +
    scale_fill_viridis_d() +
    labs(
      title = "Document-Topic Distributions",
      subtitle = paste("Showing", top_n_docs, "most focused documents"),
      x = "Gamma (document-topic probability)",
      y = "Documents",
      fill = "Topic"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 14, face = "bold"),
      legend.position = "bottom"
    )
  
  return(p)
}

create_topic_wordclouds <- function(top_terms, output_dir, n_topics = 6) {
  # Create individual wordclouds for each topic
  for (i in 1:min(n_topics, max(top_terms$topic))) {
    topic_data <- top_terms %>%
      filter(topic == i) %>%
      head(50)  # Top 50 terms
    
    if (nrow(topic_data) > 0) {
      png(file.path(output_dir, paste0("wordcloud_topic_", i, ".png")), 
          width = 800, height = 600, res = 150)
      
      par(mar = c(0, 0, 2, 0))
      wordcloud(words = topic_data$term, 
                freq = topic_data$beta * 1000,  # Scale up for better visualization
                min.freq = 1,
                max.words = 50,
                random.order = FALSE,
                rot.per = 0.35,
                colors = brewer.pal(8, "Dark2"),
                main = paste("Topic", i))
      title(main = paste("Topic", i, "Word Cloud"), line = 1)
      
      dev.off()
      cat("Saved wordcloud for Topic", i, "\n")
    }
  }
}

plot_topic_correlation <- function(doc_topics) {
  # Create topic correlation matrix
  topic_matrix <- doc_topics %>%
    select(document, topic, gamma) %>%
    pivot_wider(names_from = topic, values_from = gamma, 
                names_prefix = "Topic_", values_fill = 0) %>%
    column_to_rownames("document") %>%
    as.matrix()
  
  topic_cor <- cor(topic_matrix)
  
  # Create correlation plot
  p <- corrplot(topic_cor, 
                method = "color",
                type = "upper",
                order = "hclust",
                tl.cex = 0.8,
                tl.col = "black",
                tl.srt = 45,
                title = "Topic Correlation Matrix",
                mar = c(0, 0, 1, 0))
  
  return(topic_cor)
}

create_interactive_topic_plot <- function(top_terms, doc_topics, output_dir) {
  # Create interactive plot with plotly
  plot_data <- top_terms %>%
    group_by(topic) %>%
    slice_max(beta, n = 10) %>%
    ungroup() %>%
    mutate(topic = paste("Topic", topic))
  
  p <- ggplot(plot_data, aes(x = beta, y = reorder_within(term, beta, topic), 
                             text = paste("Term:", term, "<br>Beta:", round(beta, 4)))) +
    geom_point(size = 2, color = "steelblue") +
    facet_wrap(~ topic, scales = "free_y", ncol = 3) +
    scale_y_reordered() +
    labs(
      title = "Interactive Topic Terms",
      x = "Beta (topic-word probability)",
      y = "Terms"
    ) +
    theme_minimal()
  
  interactive_plot <- ggplotly(p, tooltip = "text")
  
  # Save as HTML
  htmlwidgets::saveWidget(interactive_plot, 
                          file.path(output_dir, "interactive_topics.html"))
  
  return(interactive_plot)
}

save_results <- function(results, model, dtm, output_dir, author_name) {
  cat("Saving results to", output_dir, "...\n")
  
  # Save topic-word distributions
  write.csv(results$topic_words, 
            file.path(output_dir, "topic_word_distributions.csv"), 
            row.names = FALSE)
  
  # Save document-topic distributions  
  write.csv(results$doc_topics,
            file.path(output_dir, "document_topic_distributions.csv"),
            row.names = FALSE)
  
  # Save top terms
  write.csv(results$top_terms,
            file.path(output_dir, "top_terms_per_topic.csv"),
            row.names = FALSE)
  
  # Save topic prevalence
  write.csv(results$topic_prevalence,
            file.path(output_dir, "topic_prevalence.csv"),
            row.names = FALSE)
  
  # Save model object
  save(model, dtm, file = file.path(output_dir, "lda_model.RData"))
  
  # Create summary report
  sink(file.path(output_dir, "analysis_summary.txt"))
  cat("LDA Topic Modeling Analysis Summary\n")
  cat("==================================\n\n")
  cat("Author:", author_name, "\n")
  cat("Analysis Date:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
  cat("Number of Documents:", nrow(dtm), "\n")
  cat("Number of Terms:", ncol(dtm), "\n")
  cat("Number of Topics:", model@k, "\n\n")
  
  cat("Topic Prevalence (Top 5):\n")
  print(head(results$topic_prevalence, 5))
  cat("\n")
  
  cat("Most Representative Terms per Topic:\n")
  for (i in 1:min(5, model@k)) {
    topic_terms <- results$top_terms %>%
      filter(topic == i) %>%
      head(5) %>%
      pull(term)
    cat("Topic", i, ":", paste(topic_terms, collapse = ", "), "\n")
  }
  sink()
  
  cat("Results saved successfully!\n")
}

# Optional: HDP implementation (requires hdp package)
run_hdp <- function(dtm, burnin = 2000, n = 100, space = 50,
                    cos.merge = 0.9, min.cluster.size = 5, seed = 123) {
  if (!requireNamespace("hdp", quietly = TRUE)) {
    cat("HDP package not available. Install with:\n")
    cat("remotes::install_github('nicolaroberts/hdp')\n")
    return(NULL)
  }
  
  library(hdp)
  
  X <- as.matrix(dtm)
  storage.mode(X) <- "integer"
  set.seed(seed)
  
  cat("Running HDP (this may take a while)...\n")
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

# ---------------- Main Analysis Function ----------------
run_complete_analysis <- function(author_name = "Wodehouse", n_books = 15, k = 8, 
                                 save_files = TRUE, create_wordclouds = TRUE) {
  
  cat("\n=== Starting Complete Topic Modeling Analysis ===\n")
  cat("Author:", author_name, "\n")
  cat("Books to analyze:", n_books, "\n")
  cat("Number of topics (k):", k, "\n\n")
  
  # Create output directory
  output_dir <- NULL
  if (save_files) {
    output_dir <- create_output_dir()
    cat("Output directory:", output_dir, "\n\n")
  }
  
  # Get corpus
  data <- get_author_corpus(author_name, n_books = n_books, language = "en")
  
  # Run LDA
  lda_fit <- run_lda(data$dtm, k = k)
  
  # Extract results
  results <- extract_lda_results(lda_fit, data$dtm, topn = 20)
  
  # Create visualizations
  cat("Creating visualizations...\n")
  
  # Top terms plot
  p1 <- plot_top_terms(results$top_terms, terms_per_topic = 8)
  print(p1)
  if (save_files) {
    ggsave(file.path(output_dir, "top_terms_plot.png"), p1, 
           width = 12, height = 8, dpi = 300)
  }
  
  # Topic prevalence
  p2 <- plot_topic_prevalence(results$topic_prevalence)
  print(p2)
  if (save_files) {
    ggsave(file.path(output_dir, "topic_prevalence.png"), p2,
           width = 10, height = 6, dpi = 300)
  }
  
  # Document topics
  p3 <- plot_document_topics(results$doc_topics)
  print(p3)
  if (save_files) {
    ggsave(file.path(output_dir, "document_topics.png"), p3,
           width = 10, height = 8, dpi = 300)
  }
  
  # Topic correlations
  if (save_files) {
    png(file.path(output_dir, "topic_correlations.png"), 
        width = 800, height = 600, res = 150)
    topic_cor <- plot_topic_correlation(results$doc_topics)
    dev.off()
  }
  
  # Word clouds
  if (create_wordclouds && save_files) {
    create_topic_wordclouds(results$top_terms, output_dir, n_topics = k)
  }
  
  # Interactive plot
  if (save_files) {
    create_interactive_topic_plot(results$top_terms, results$doc_topics, output_dir)
  }
  
  # Save all results
  if (save_files) {
    save_results(results, lda_fit, data$dtm, output_dir, author_name)
  }
  
  cat("\n=== Analysis Complete ===\n")
  if (save_files) {
    cat("All files saved to:", output_dir, "\n")
  }
  
  return(list(
    model = lda_fit,
    results = results,
    data = data,
    output_dir = output_dir
  ))
}

# ---------------- Example Usage ----------------
# Run complete analysis for Wodehouse
analysis <- run_complete_analysis(
  author_name = "Wodehouse",
  n_books = 12, 
  k = 6,
  save_files = TRUE,
  create_wordclouds = TRUE
)

# Print summary
cat("\nTopic Prevalence:\n")
print(analysis$results$topic_prevalence)

cat("\nTop 5 terms for each topic:\n")
analysis$results$top_terms %>%
  group_by(topic) %>%
  slice_max(beta, n = 5) %>%
  summarise(top_terms = paste(term, collapse = ", "), .groups = "drop") %>%
  mutate(topic = paste("Topic", topic)) %>%
  print()

# For HDP analysis (optional):
# hdp_results <- run_hdp(analysis$data$dtm, burnin = 1000, n = 50)
# if (!is.null(hdp_results)) {
#   cat("HDP found", ncol(hdp_results$topic_word), "topics\n")
# }