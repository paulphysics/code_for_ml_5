# =============================================================================
# rf_model.R — Random Forest (ranger) for binary classification
# Depends on helpers_macro.R:
#   ensure_targets, expanding_oos, metrics_at, youden_thr,
#   plot_roc_test, plot_pr_test, plot_calibration, plot_timeline,
#   stamp_outdir, save_plot, save_tbl
# =============================================================================
suppressPackageStartupMessages({
  library(readr); library(dplyr); library(tidyr); library(tibble)
  library(ggplot2); library(pROC); library(PRROC); library(caret)
  library(ranger)
})

source("helpers_macro.R")
set.seed(42)

# --------------- Small local helper: safe positive-class extraction ------------
positive_prob <- function(pred_matrix, pos = "1") {
  stopifnot(is.matrix(pred_matrix))
  j <- if (pos %in% colnames(pred_matrix)) which(colnames(pred_matrix) == pos) else ncol(pred_matrix)
  as.numeric(pred_matrix[, j])
}


REGION      <- "US"   # "US","UK","EA19"
DATA_FILE   <- "data/macro_quarterly_FRED_teaching_1999_2024.csv"
TARGET_ID   <- "unemp_up_next"   # or "recession_next"
FEATURES    <- c("gdp_growth_qoq","UNEMP","inflation_yoy","POL","term_spread")

TEST_SPLIT  <- 0.70
OOS_START   <- NULL    # e.g. "2010-01-01" or NULL → first test point
# --------------------- KNOBS ---------------------------------------------------
# RF hyperparameters students can tweak  (suggested safe ranges in comments)
N_TREES       <- 800L   # number of trees; try e.g. 200–1500 (must be positive integer)
MTRY          <- NA     # number p of features per split; NA = sqrt(p), otherwise use 1–p (do NOT exceed p)
MIN_NODE      <- 5L     # minimum observations in a terminal node; typical range 1–20 (>=1)
CLASS_WEIGHTS <- NULL   # NULL = no reweighting; or e.g. list(`0` = 1, `1` = 1.2–3)
# --------------------- LOAD + TARGET ------------------------------------------

MODEL_TAG <- "RF"
OUT_DIR   <- stamp_outdir(MODEL_TAG, REGION, TARGET_ID)

message("Loading data…")
all <- read_csv(DATA_FILE, show_col_types = FALSE) %>%
  filter(region == REGION) %>%
  arrange(date_q) %>%
  ensure_targets() %>%
  tidyr::drop_na(dplyr::all_of(c(FEATURES, TARGET_ID)))

stopifnot(nrow(all) > 50)
y_all <- as.integer(all[[TARGET_ID]])
cat("\n=== DATA SUMMARY (", REGION, ") ===\n", sep = "")
cat("Range:", as.character(min(all$date_q)), "→", as.character(max(all$date_q)),
    "| n =", nrow(all), "\n")
cat("Class prevalence (", TARGET_ID, "==1):", round(mean(y_all), 3), "\n\n")

# --------------------- CHRONO SPLIT -------------------------------------------
n   <- nrow(all)
cut <- floor(n * TEST_SPLIT)
train <- all[1:cut, ]
test  <- all[(cut+1):n, ]

# --------------------- FIT RF (70/30 test) ------------------------------------
train_xy <- train[, FEATURES, drop = FALSE]
train_xy[[TARGET_ID]] <- factor(train[[TARGET_ID]], levels = c(0, 1))

rf_fit <- ranger::ranger(
  formula       = as.formula(paste(TARGET_ID, "~ .")),
  data          = train_xy,
  num.trees     = N_TREES,
  mtry          = if (is.na(MTRY)) floor(sqrt(length(FEATURES))) else MTRY,
  min.node.size = MIN_NODE,
  probability   = TRUE,
  importance    = "impurity",
  class.weights = CLASS_WEIGHTS
)

# Test predictions
pmat <- predict(rf_fit, data = test[, FEATURES, drop = FALSE])$predictions
rf_prob_test <- positive_prob(pmat, pos = "1")

# Metrics (test)
thr_youden <- youden_thr(rf_prob_test, as.integer(test[[TARGET_ID]]))
cat("\n=== 70/30 TEST METRICS — Random Forest ===\n")
mt_7030 <- dplyr::bind_rows(
  metrics_at("RF (test)", rf_prob_test, as.integer(test[[TARGET_ID]]), thr = 0.50),
  metrics_at("RF (test)", rf_prob_test, as.integer(test[[TARGET_ID]]), thr = thr_youden)
)
print(mt_7030)

# --------------------- Plots (test) — unified styling -------------------------
p_roc <- plot_roc_test(as.integer(test[[TARGET_ID]]), rf_prob_test,
                       sprintf("ROC — %s (70/30 test)", REGION))
p_pr  <- plot_pr_test (as.integer(test[[TARGET_ID]]), rf_prob_test,
                       sprintf("Precision–Recall — %s (70/30 test)", REGION))
p_cal <- plot_calibration(as.integer(test[[TARGET_ID]]), rf_prob_test,
                          sprintf("Calibration (deciles) — %s (70/30 test)", REGION))

print(p_roc); print(p_pr); print(p_cal)
save_plot(p_roc, file.path(OUT_DIR, "test_roc.png"))
save_plot(p_pr,  file.path(OUT_DIR, "test_pr.png"))
save_plot(p_cal, file.path(OUT_DIR, "test_calibration.png"))

# --------------------- EXPANDING-WINDOW OOS (shared helper) --------------------
start_date <- if (!is.null(OOS_START)) as.Date(OOS_START) else all$date_q[cut + 1L]

fit_fn <- function(Xtr_, ytr_) {
  df_tr <- as.data.frame(Xtr_)
  df_tr[[TARGET_ID]] <- factor(ytr_, levels = c(0, 1))
  ranger::ranger(
    formula       = as.formula(paste(TARGET_ID, "~ .")),
    data          = df_tr,
    num.trees     = N_TREES,
    mtry          = if (is.na(MTRY)) floor(sqrt(ncol(Xtr_))) else MTRY,
    min.node.size = MIN_NODE,
    probability   = TRUE,
    class.weights = CLASS_WEIGHTS,
    importance    = "impurity"
  )
}

predict_fn <- function(model_obj, Xte_) {
  preds <- predict(model_obj, data = as.data.frame(Xte_))$predictions
  positive_prob(preds, pos = "1")
}

oos <- expanding_oos(
  df = all, feature_names = FEATURES, target_col = TARGET_ID,
  start_date = start_date, fit_fn = fit_fn, predict_fn = predict_fn
)

# --- OOS metrics + plots + saving ---------------------------------------------
oos <- oos[is.finite(oos$prob), , drop = FALSE]
prob_oos <- oos$prob
y_oos    <- as.integer(oos$y)
stopifnot(length(prob_oos) > 0L, min(prob_oos) >= 0, max(prob_oos) <= 1)

thr_oos <- youden_thr(prob_oos, y_oos)

cat("\n=== EXPANDING-WINDOW OOS METRICS — Random Forest ===\n")
mt_oos <- dplyr::bind_rows(
  metrics_at("RF (OOS)", prob_oos, y_oos, thr = 0.50),
  metrics_at("RF (OOS)", prob_oos, y_oos, thr = thr_oos)
)
print(mt_oos); cat(sprintf("\nYouden threshold (OOS): %.3f\n", thr_oos))

p_roc_oos <- plot_roc_test(y_oos, prob_oos, sprintf("ROC — %s (Expanding OOS)", REGION))
p_pr_oos  <- plot_pr_test (y_oos, prob_oos, sprintf("Precision–Recall — %s (Expanding OOS)", REGION))
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
