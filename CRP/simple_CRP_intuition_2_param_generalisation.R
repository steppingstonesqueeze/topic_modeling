# An intuition into Chinese Restaurant Process #

# Clearly can see the rich get richer phenomenon when the
#
# 2 parameter model now with theta (strength) and theta (discount)

# Probability customer sits at empty table is : 
# (theta + NUM_OCC_TABLES*theta) / (n + theta) (for us n <- n-1 just notationally)

# probability customer chooses one of available :
# if table with b customerts, (b - theta) / (n + theta) (for us, n-1 replaces n)

library(ggplot2)
library(dplyr)

set.seed(12) # repro

sample_with_chosen_probs <- function(numbers, prob_tables) {
  len_numbers <- length(numbers)
  ones <- rep(1, len_numbers)
  tables <- cumsum(ones)
  
  return (sample(tables, 1, prob = prob_tables)) # which table does the customer go to
}

update_number_counts <- function(numbers, last_chosen_table) {
  numbers[last_chosen_table] <- numbers[last_chosen_table] + 1 # update this table by 1
  return (numbers)
}

new_table_probability_two_param <- function(denom, theta = 1, alpha = 0.5, num_occ_tables = 1) { # denom = tot_cust + 1 1 for new cust
  return ((theta + num_occ_tables * alpha) / (denom + theta - 1)) # always 1 / (n+1)
}

update_old_table_probability_two_param<- function(numbers, denom, theta = 1, alpha = 0.5) {
  return((numbers - alpha) / (denom + theta - 1))
}
# invariant : numbers + 1 = denom always (a PDF or rather a PMF)

N_cust <- 4000
theta_value <- 1.0
alpha_value <- 0.5

n <- 1
numbers <- NULL
numbers <- c(numbers, 1) # first customer

for (n in 2:N_cust) {
  
  num_occ_tables <- length(numbers)
  ntp <- new_table_probability_two_param(n, theta = theta_value, alpha = alpha_value, num_occ_tables)
  otp <- update_old_table_probability_two_param(numbers, n, theta = theta_value, alpha = alpha_value)
  
  pt <- c(otp, ntp) # new probability vector
  
  # orig number of tables before update
  orig_tables <- length(numbers)
  
  numbers <- c(numbers, 0) # for new table ; unideal but we do not expect many clusters, so fine
  
  chosen_table <- sample_with_chosen_probs(numbers, pt)
  
  if (chosen_table <= orig_tables) {
    # new customer sat at old table
    numbers <- numbers[-length(numbers)]
  }
  
  # update
  numbers <- update_number_counts(numbers, chosen_table) 
  
  i
}


# Final counts across tables #

numbers

# Fractions of customers across tables

numbers / sum(numbers)

# customer df
customer_df <- data.frame(
  table_number = seq_len(length(numbers)),
  occupancy = numbers
)

# barplot

ggplot(
  data = customer_df,aes(x = table_number, y = occupancy, fill = as.factor(table_number))
) + geom_bar(stat = "identity", show.legend = FALSE)

Sys.sleep(5)
# Look at distributions
# AKA How many tables have an occupancy between 2^k and 2^(k+1) ?

# Assuming your dataframe is called 'crp_df' with columns 'table_number', 'occupancy'

# Create power-of-2 bins
crp_binned <- customer_df %>%
  mutate(
    bin_label = case_when(
      occupancy == 1 ~ "1",
      occupancy == 2 ~ "2", 
      occupancy %in% 3:4 ~ "3-4",
      occupancy %in% 5:8 ~ "5-8",
      occupancy %in% 9:16 ~ "9-16",
      occupancy %in% 17:32 ~ "17-32",
      occupancy %in% 33:64 ~ "33-64",
      occupancy %in% 65:128 ~ "65-128",
      occupancy %in% 129:256 ~ "129-256",
      occupancy %in% 257:512 ~ "257-512",
      occupancy %in% 513:1024 ~ "513-1024",
      TRUE ~ "Other"
    ),
    bin_order = case_when(
      occupancy == 1 ~ 1,
      occupancy == 2 ~ 2,
      occupancy %in% 3:4 ~ 3,
      occupancy %in% 5:8 ~ 4,
      occupancy %in% 9:16 ~ 5,
      occupancy %in% 17:32 ~ 6,
      occupancy %in% 33:64 ~ 7,
      occupancy %in% 65:128 ~ 8,
      occupancy %in% 129:256 ~ 9,
      occupancy %in% 257:512 ~ 10,
      occupancy %in% 513:1024 ~ 11,
      TRUE ~ 12
    )
  ) %>%
  count(bin_label, bin_order, name = "table_count") %>%
  arrange(bin_order)

# Plot
ggplot(crp_binned, aes(x = reorder(bin_label, bin_order), y = table_count)) +
  geom_col(fill = "steelblue", alpha = 0.8) +
  geom_text(aes(label = table_count), vjust = -0.3) +
  labs(
    title = "CRP Table Distribution for two param (Power-of-2 Bins)",
    x = "Number of Customers per Table",
    y = "Number of Tables"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

