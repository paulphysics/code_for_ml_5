# =============================================================================
# glmnet_model.R — Elastic-Net Logistic Regression (binary)
# Depends on helpers_macro.R:
#   ensure_targets, to_matrix, glmnet_fit_cv, predict_glmnet_prob, expanding_oos,
#   metrics_at, youden_thr, plot_roc_test, plot_pr_test, plot_calibration,
#   plot_timeline, stamp_outdir, save_plot, save_tbl
# =============================================================================
suppressPackageStartupMessages({
  library(readr); library(dplyr); library(tidyr); library(tibble)
  library(ggplot2); library(pROC); library(PRROC); library(caret)
  library(glmnet)
})

source("helpers_macro.R")
set.seed(42)

REGION      <- "US"
DATA_FILE   <- "data/macro_quarterly_FRED_teaching_1999_2024.csv"
TARGET_ID   <- "policy_cut_next"  # or: unemp_up_next, infl_up_next, policy_cut_next, flatten_next, recession_next
FEATURES    <- c("gdp_growth_qoq","UNEMP","inflation_yoy","POL","term_spread")

TEST_SPLIT  <- 0.70
OOS_START   <- NULL            # e.g. "2010-01-01" or NULL → first test

ALPHAS      <- c(0, 0.25, 0.5, 0.75, 1.0)  # ridge→lasso grid
N_FOLDS     <- 5


MODEL_TAG <- "ENET"
OUT_DIR   <- stamp_outdir(MODEL_TAG, REGION, TARGET_ID)

# --------------------- KNOBS ---------------------------------------------------
USE_L1SE    <- TRUE
# --------------------- LOAD + TARGET ------------------------------------------
message("Loading data…")
all <- readr::read_csv(DATA_FILE, show_col_types = FALSE) %>%
  dplyr::filter(region == REGION) %>%
  dplyr::arrange(date_q) %>%
  ensure_targets() %>%
  tidyr::drop_na(dplyr::all_of(c(FEATURES, TARGET_ID)))

stopifnot(nrow(all) > 50)
y_all <- as.integer(all[[TARGET_ID]])
cat("\n=== DATA SUMMARY (", REGION, ") ===\n", sep = "")
cat("Range:", as.character(min(all$date_q)), "→", as.character(max(all$date_q)),
    "| n =", nrow(all), "\n")
cat("Class prevalence (", TARGET_ID, "==1):", round(mean(y_all), 3), "\n\n")

# --------------------- CHRONO SPLIT -------------------------------------------
n    <- nrow(all)
cut  <- floor(n * TEST_SPLIT)
train <- all[1:cut, ]
test  <- all[(cut+1):n, ]

Xtr <- to_matrix(train, FEATURES); ytr <- as.integer(train[[TARGET_ID]])
Xte <- to_matrix(test,  FEATURES); yte <- as.integer(test[[TARGET_ID]])

# --------------------- CV across alpha ----------------------------------------
cv_list <- lapply(ALPHAS, function(a) glmnet_fit_cv(
  Xtr, ytr, alpha = a, nfolds = N_FOLDS, use_l1se = USE_L1SE
))

# choose alpha by lowest CV deviance at chosen lambda
cv_score <- vapply(cv_list, function(z) {
  lam <- z$lambda
  idx <- which.min(abs(z$cv$lambda - lam))
  z$cv$cvm[idx]
}, numeric(1))

best_idx    <- which.min(cv_score)
alpha_star  <- ALPHAS[best_idx]
fit_star    <- cv_list[[best_idx]]$fit
lambda_star <- cv_list[[best_idx]]$lambda
nz_star     <- cv_list[[best_idx]]$nz

cat(sprintf("Selected alpha = %.2f | lambda = %.4g (%s)\n",
            alpha_star, lambda_star, if (USE_L1SE) "1se" else "min"))
cat("Non-zero coefficients at chosen lambda:", nz_star, "\n")

# --------------------- TEST PREDICTION + METRICS -------------------------------
prob_test  <- predict_glmnet_prob(fit_star, newx = Xte, s = lambda_star)
thr_youden <- youden_thr(prob_test, yte)

cat("\n=== 70/30 TEST METRICS — Elastic-Net ===\n")
mt_7030 <- dplyr::bind_rows(
  metrics_at("ENet (test)", prob_test, yte, thr = 0.50),
  metrics_at("ENet (test)", prob_test, yte, thr = thr_youden)
)
print(mt_7030)

# --------------------- Plots (test) — unified styling -------------------------
p_roc <- plot_roc_test(yte, prob_test,
                       sprintf("ROC — %s (70/30 test)", REGION))
p_pr  <- plot_pr_test (yte, prob_test,
                       sprintf("Precision–Recall — %s (70/30 test)", REGION))
p_cal <- plot_calibration(yte, prob_test,
                          sprintf("Calibration (deciles) — %s (70/30 test)", REGION))

print(p_roc); print(p_pr); print(p_cal)
save_plot(p_roc, file.path(OUT_DIR, "test_roc.png"))
save_plot(p_pr,  file.path(OUT_DIR, "test_pr.png"))
save_plot(p_cal, file.path(OUT_DIR, "test_calibration.png"))

# --------------------- EXPANDING-WINDOW OOS (shared helper) --------------------
start_date <- if (!is.null(OOS_START)) as.Date(OOS_START) else all$date_q[cut + 1L]

fit_fn <- function(Xtr_, ytr_) glmnet_fit_cv(
  Xtr_, ytr_, alpha = alpha_star, nfolds = N_FOLDS, use_l1se = USE_L1SE
)
predict_fn <- function(model_obj, Xte_) predict_glmnet_prob(
  model_obj$fit, newx = Xte_, s = model_obj$lambda
)

oos <- expanding_oos(
  df = all, feature_names = FEATURES, target_col = TARGET_ID,
  start_date = start_date, fit_fn = fit_fn, predict_fn = predict_fn
)

# --- OOS metrics + plots + saving ---------------------------------------------
# --- OOS metrics + plots + saving ---------------------------------------------
oos <- oos[is.finite(oos$prob), , drop = FALSE]
prob_oos <- oos$prob
y_oos    <- as.integer(oos$y)
stopifnot(length(prob_oos) > 0L, min(prob_oos) >= 0, max(prob_oos) <= 1)

# --- remove these two lines (leaky) ---
# thr_oos <- youden_thr(prob_oos, y_oos)
# metrics_at(..., thr = thr_oos)

mt_oos <- dplyr::bind_rows(
  metrics_at(paste0(MODEL_TAG," (OOS)"), prob_oos, y_oos, thr = 0.50)
)
print(mt_oos)



p_roc_oos <- plot_roc_test(y_oos, prob_oos,
                           sprintf("ROC — %s (Expanding OOS)", REGION))
p_pr_oos  <- plot_pr_test (y_oos, prob_oos,
                           sprintf("Precision–Recall — %s (Expanding OOS)", REGION))
p_time    <- plot_timeline(
  oos$date_q, prob_oos, y_oos,
  sprintf("%s predicted P(%s=1) — %s (Expanding OOS)", MODEL_TAG, TARGET_ID, REGION),
  "Points mark quarters with next-quarter event = 1"
)

print(p_roc_oos); print(p_pr_oos); print(p_time)

# Save tables and plots to the same stamped folder
save_tbl(mt_7030, file.path(OUT_DIR, "metrics_test.csv"))
save_tbl(mt_oos,  file.path(OUT_DIR, "metrics_oos.csv"))
save_plot(p_roc_oos, file.path(OUT_DIR, "oos_roc.png"))
save_plot(p_pr_oos,  file.path(OUT_DIR, "oos_pr.png"))
save_plot(p_time,    file.path(OUT_DIR, "oos_timeline.png"))

cat(sprintf("\nSaved outputs to: %s\n", OUT_DIR))
