# Chinese Restaurant Process (CRP) — One-Param & Two-Param Simulations (R)

This README documents two small R scripts that simulate the Chinese Restaurant Process (CRP) and visualize the resulting cluster (table) sizes. The first script uses the **one-parameter CRP** with concentration/strength `θ`. The second uses the **two-parameter CRP** with **discount `α`** and **concentration `θ`**.

---

## Contents

- [What these scripts do](#what-these-scripts-do)  
- [Quick start](#quick-start)  
- [Dependencies](#dependencies)  
- [Files](#files)  
- [How to run](#how-to-run)  
- [Parameters you can change](#parameters-you-can-change)  
- [Outputs and visuals](#outputs-and-visuals)  
- [How it works (intuitions)](#how-it-works-intuitions)  
- [Troubleshooting](#troubleshooting)  
- [Performance tips](#performance-tips)  
- [Extending to LDA / IBP](#extending-to-lda--ibp)  
- [License](#license)

---

## What these scripts do

- **Simulate arrivals of `N_cust` customers** to an infinite set of tables under CRP dynamics.
- **One-parameter CRP (Code 1)**: probability of joining an **existing** table is proportional to its occupancy; probability of opening a **new** table is proportional to `θ`.  
- **Two-parameter CRP (Code 2)**: probability of joining an existing table with `b` customers is proportional to `b − α`; probability of opening a new table is proportional to `θ + α·(K)`, where `K` is the number of currently occupied tables.
- Produce:
  - A **bar plot** of table occupancies.
  - A **binned histogram** of table counts using **power-of-two occupancy bins** (1, 2, 3–4, 5–8, …).

---

## Quick start

```bash
# From a terminal
Rscript simple_CRP_intuition.R
Rscript simple_CRP_intuition_2_param_generalisation.R
```

Or, in RStudio: open each script and click **Run**.

---

## Dependencies

Both scripts use:

- `ggplot2`
- `dplyr`

Install once if needed:

```r
install.packages(c("ggplot2", "dplyr"))
```

---

## Files

- **`simple_CRP_intuition.R`** — One-parameter CRP simulation and plots.  
  - Key hyperparameter: `theta_value` (a.k.a. concentration or strength).
- **`simple_CRP_intuition_2_param_generalisation.R`** — Two-parameter CRP simulation and plots.  
  - Key hyperparameters: `theta_value` (concentration) and `alpha_value` (discount).

---

## How to run

### CLI (recommended)

```bash
Rscript simple_CRP_intuition.R
Rscript simple_CRP_intuition_2_param_generalisation.R
```

### R / RStudio

```r
source("simple_CRP_intuition.R")
source("simple_CRP_intuition_2_param_generalisation.R")
```

---

## Parameters you can change

In each script, near the top:

- **Number of customers**:
  ```r
  N_cust <- 4000
  ```
- **CRP hyperparameters**:
  - **One-param script**:
    ```r
    theta_value <- 20.0    # Larger => more clusters (tables)
    ```
  - **Two-param script**:
    ```r
    theta_value <- 1.0     # Concentration
    alpha_value <- 0.7     # Discount in [0,1)
    ```

General guidance:
- Increasing **`theta_value`** → more new tables overall (more clusters, smaller average occupancy).
- Increasing **`alpha_value`** (two-param CRP) → heavier tails in the cluster-size distribution; more small tables relative to the one-param case. Must satisfy `0 ≤ α < 1`.

---

## Outputs and visuals

1. **Printed vectors**:
   - `numbers` — final occupancy per table (length = number of tables).
   - `numbers / sum(numbers)` — occupancy fractions (cluster weights).

2. **Plots** (pop up in your device/Plots pane):
   - **Bar plot of occupancies** per table (`ggplot`).
   - **Binned counts** of tables by **power-of-two occupancy bins** (1, 2, 3–4, 5–8, …).

3. **Saving plots** (optional):  
   Add after a plot to save to disk:
   ```r
   ggsave("crp_barplot.png", width = 8, height = 5, dpi = 150)
   ggsave("crp_bins.png", width = 8, height = 5, dpi = 150)
   ```

---

## How it works (intuitions)

### One-parameter CRP
- At step `n` (the nth customer), with current table sizes `numbers` and total `n−1` existing customers:
  - **New table** probability:  
    \[
      \Pr(\text{new}) = \frac{\theta}{(n - 1) + \theta}
    \]
  - **Existing table `k`** probability (with size \( n_k \)):  
    \[
      \Pr(k) = \frac{n_k}{(n - 1) + \theta}
    \]
- This creates **“rich get richer”** dynamics: bigger tables tend to attract more customers.

### Two-parameter CRP (Pitman–Yor / discount process)
- With discount `α` and concentration `θ`:
  - **New table** probability with `K` current tables:  
    \[
      \Pr(\text{new}) = \frac{\theta + \alpha K}{(n - 1) + \theta}
    \]
  - **Existing table `k`** probability:  
    \[
      \Pr(k) = \frac{n_k - \alpha}{(n - 1) + \theta}
    \]
- The discount `α` reduces the pull of large tables slightly and increases the chance of new tables in proportion to `K`. This yields **heavier-tailed** cluster size distributions than the one-parameter CRP.

---

## Troubleshooting

- **Dangling `i` line in the loop**  
  Both scripts (as pasted) contain a stray line `i` at the end of the main loop:
  ```r
  # update
  numbers <- update_number_counts(numbers, chosen_table) 
  i
  ```
  This will raise `Error: object 'i' not found` in R.  
  **Fix:** delete that lone `i` line (it serves no purpose).

- **Negative probabilities (two-param CRP)**  
  You must ensure `0 ≤ α < 1` and that **all** occupancies `n_k ≥ 1`. The script maintains `n_k ≥ 1` by construction. If you modify logic, keep `n_k - α ≥ 0`.

- **Missing packages**  
  Install `ggplot2` and `dplyr` as shown above.

- **No plot window**  
  If running headless (e.g., CI server), `ggsave` to files instead of displaying.

---

## Performance tips

- The simulation is **O(N_cust)** with small constant factors.  
- If you scale beyond ~1e6 customers, consider:
  - Preallocating and storing only summary stats instead of full per-table history.
  - Avoiding repeated vector concatenation for `numbers` (use lists or preallocate capacity if you know approximate number of tables).
  - Turning off plotting during the run.

---

## Extending to LDA / IBP

- **LDA**: You can use CRP as a prior over **topic allocations** (e.g., via the Dirichlet Process mixture analogue) to get **nonparametric LDA** (HDP-LDA).  
- **IBP**: For latent **feature** allocation (instead of clusters), switch to the **Indian Buffet Process**, which yields sparse binary feature matrices.  
- We can add **clean, minimal R and Python** implementations next:
  - **HDP-style topic models** (CRP/CRF for documents & tables/topics).
  - **IBP Gibbs sampler** with visualizations of feature usage.

---

## License

Use, modify, and share freely. If you publish or share results, a small attribution back to this repo/script set is appreciated.
