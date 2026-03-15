# =============================================================================
# xgb_model.R — Binary XGBoost (logistic) template
# Depends on helpers_macro.R:
#   load_data, build_features, split_chrono, derive_target, select_features,
#   expanding_oos, metrics_at, youden_thr,
#   plot_roc_test, plot_pr_test, plot_calibration, plot_timeline,
#   stamp_outdir, save_plot, save_tbl
# =============================================================================
suppressPackageStartupMessages({
  library(readr); library(dplyr); library(tidyr); library(tibble)
  library(ggplot2); library(pROC); library(PRROC); library(caret)
  library(xgboost)
})

# =================== CONFIG — STUDENTS EDIT ONLY ===================
REGION       <- "US"    # "US","UK","EA19"
DATA_FILE    <- "data/macro_quarterly_FRED_teaching_1999_2024.csv"
TARGET_ID    <- "UNEMP_UP_NEXT"   # see helpers: derive_target()
FEATURE_SET  <- "BASE"            # "BASE" or "BASE_PLUS"
TEST_SPLIT   <- 0.70
TARGET_OPTS  <- list(quantile_q = 0.75)  # only used for *_QTL targets
OOS_START    <- NULL                     # e.g. "2010-01-01" or NULL → first test
SEED         <- 42


# Model knobs  (suggested safe ranges in comments)
XGB_KNOBS <- list(
  nrounds         = 400L,  # total number of trees; try e.g. 100–800 (not 1000s)
  eta             = 0.05,  # learning rate; typical range 0.01–0.3 (must be >0 and <1)
  max_depth       = 3L,    # tree depth; usually 2–6 for this kind of problem
  subsample       = 0.8,   # fraction of rows per tree; keep in 0.5–1.0
  colsample_bytree= 0.8,   # fraction of features per tree; keep in 0.5–1.0
  min_child_weight= 1      # min data weight in a leaf; reasonable range 1–10
)

# ===================================================================


set.seed(SEED)
source("helpers_macro.R")

MODEL_TAG <- "XGB"

# --- Load, features, split anchor -----------------------------------
all0 <- load_data(DATA_FILE, REGION)
all1 <- build_features(all0)

# Anchor split using features-only frame
spl <- split_chrono(all1, TEST_SPLIT)

# Derive target with leakage-safe thresholds based on train period only
all2 <- derive_target(all1, TARGET_ID, TARGET_OPTS, train_until_date = spl$cut_date)

# Select features and drop NA rows for chosen FEATURES + TARGET
FEATURES <- select_features(all2, FEATURE_SET)$names
all3 <- all2 %>%
  tidyr::drop_na(dplyr::all_of(c(FEATURES, TARGET_ID)))

# Rebuild train/test via dates to respect anchor
train_idx <- which(all3$date_q <= spl$cut_date)
test_idx  <- which(all3$date_q >= spl$first_test_date)

stopifnot(length(train_idx) > 5L, length(test_idx) > 5L)

# --- Build X/y matrices ------------------------------------------------
Xtr <- as.matrix(all3[train_idx, FEATURES, drop = FALSE])
ytr <- as.integer(all3[[TARGET_ID]][train_idx])

Xte <- as.matrix(all3[test_idx,  FEATURES, drop = FALSE])
yte <- as.integer(all3[[TARGET_ID]][test_idx])

dtr <- xgb.DMatrix(data = Xtr, label = ytr)
dte <- xgb.DMatrix(data = Xte, label = yte)

# --- Fit XGBoost ------------------------------------------------------
params <- list(
  objective        = "binary:logistic",
  eval_metric      = "logloss",
  eta              = XGB_KNOBS$eta,
  max_depth        = XGB_KNOBS$max_depth,
  subsample        = XGB_KNOBS$subsample,
  colsample_bytree = XGB_KNOBS$colsample_bytree,
  min_child_weight = XGB_KNOBS$min_child_weight,
  nthread          = 0
)

xgb_fit <- xgb.train(
  params  = params,
  data    = dtr,
  nrounds = XGB_KNOBS$nrounds,
  verbose = 0
)

# --- Predict and evaluate (70/30 test) --------------------------------
xgb_prob_test <- as.numeric(predict(xgb_fit, newdata = dte))
thr_youden    <- youden_thr(xgb_prob_test, yte)

cat(sprintf("\n=== DATA SUMMARY (%s) ===\n", REGION))
cat("Range:", as.character(min(all3$date_q)), "→", as.character(max(all3$date_q)),
    "| n =", nrow(all3), "\n")
cat("Class prevalence (", TARGET_ID, "==1):", round(mean(all3[[TARGET_ID]]), 3), "\n\n")

cat("\n=== 70/30 TEST METRICS — XGBoost ===\n")
mt_test <- dplyr::bind_rows(
  metrics_at("XGB (test)", xgb_prob_test, yte, thr = 0.50),
  metrics_at("XGB (test)", xgb_prob_test, yte, thr = thr_youden)
)
print(mt_test)

cat("\nConfusion @ 0.50:\n")
print(caret::confusionMatrix(
  factor(ifelse(xgb_prob_test >= 0.5, 1, 0), levels = c(0,1)),
  factor(yte, levels = c(0,1)), positive = "1")$table)

cat("\nConfusion @ Youden (", round(thr_youden, 3), "):\n", sep = "")
print(caret::confusionMatrix(
  factor(ifelse(xgb_prob_test >= thr_youden, 1, 0), levels = c(0,1)),
  factor(yte, levels = c(0,1)), positive = "1")$table)

# --- Plots (test) -----------------------------------------------------
p_roc <- plot_roc_test(yte, xgb_prob_test, sprintf("ROC — %s (70/30 test)", REGION))
p_pr  <- plot_pr_test (yte, xgb_prob_test, sprintf("Precision–Recall — %s (70/30 test)", REGION))
p_cal <- plot_calibration(yte, xgb_prob_test, sprintf("Calibration (deciles) — %s (test)", REGION))

print(p_roc); print(p_pr); print(p_cal)

# --- Expanding-window OOS ---------------------------------------------
fit_fn <- function(Xtr_, ytr_) {
  xgb.train(
    params  = params,
    data    = xgb.DMatrix(data = Xtr_, label = ytr_),
    nrounds = XGB_KNOBS$nrounds,
    verbose = 0
  )
}
predict_fn <- function(mod, Xte_) {
  predict(mod, newdata = xgb.DMatrix(data = Xte_))
}

start_date <- if (!is.null(OOS_START)) as.Date(OOS_START) else spl$first_test_date
oos <- expanding_oos(all3, FEATURES, TARGET_ID, start_date, fit_fn, predict_fn)

# OOS metrics
oos <- oos[is.finite(oos$prob), , drop = FALSE]
y_oos    <- as.integer(oos$y)
prob_oos <- oos$prob
stopifnot(length(prob_oos) > 0L, min(prob_oos) >= 0, max(prob_oos) <= 1)

thr_oos <- youden_thr(prob_oos, y_oos)

cat("\n=== EXPANDING-WINDOW OOS METRICS — XGBoost ===\n")
mt_oos <- dplyr::bind_rows(
  metrics_at("XGB (OOS)", prob_oos, y_oos, thr = 0.50),
  metrics_at("XGB (OOS)", prob_oos, y_oos, thr = thr_oos)
)
print(mt_oos); cat(sprintf("\nYouden threshold (OOS): %.3f\n", thr_oos))

# OOS plots
p_roc_oos <- plot_roc_test(y_oos, prob_oos, sprintf("ROC — %s (Expanding OOS)", REGION))
p_pr_oos  <- plot_pr_test (y_oos, prob_oos, sprintf("Precision–Recall — %s (Expanding OOS)", REGION))
p_tline   <- plot_timeline(oos$date_q, prob_oos, y_oos,
                           sprintf("Predicted P(%s=1) — %s (Expanding OOS)", TARGET_ID, REGION),
                           "Points mark quarters with next-quarter event = 1")

print(p_roc_oos); print(p_pr_oos); print(p_tline)

# --- Save outputs (stamped run folder) --------------------------------
outdir <- stamp_outdir(MODEL_TAG, REGION, TARGET_ID)
save_tbl(mt_test, file.path(outdir, "metrics_test.csv"))
save_tbl(mt_oos,  file.path(outdir, "metrics_oos.csv"))

save_plot(p_roc,     file.path(outdir, "test_roc.png"))
save_plot(p_pr,      file.path(outdir, "test_pr.png"))
save_plot(p_cal,     file.path(outdir, "test_calibration.png"))
save_plot(p_roc_oos, file.path(outdir, "oos_roc.png"))
save_plot(p_pr_oos,  file.path(outdir, "oos_pr.png"))
save_plot(p_tline,   file.path(outdir, "oos_timeline.png"))

cat(sprintf("\nSaved outputs to: %s\n", outdir))

