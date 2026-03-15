# =============================================================================
# gam_model.R — Generalized Additive Model (mgcv) for binary classification
# Depends on helpers_macro.R:
#   load_data, build_features, split_chrono, derive_target, select_features,
#   expanding_oos, metrics_at, youden_thr,
#   plot_roc_test, plot_pr_test, plot_calibration, plot_timeline,
#   stamp_outdir, save_plot, save_tbl
# =============================================================================

# =================== CONFIG — STUDENTS EDIT ONLY ===================
REGION       <- "US"    # "US","UK","EA19"
DATA_FILE    <- "data/macro_quarterly_FRED_teaching_1999_2024.csv"
TARGET_ID    <- "UNEMP_UP_NEXT"      # see helpers: derive_target()
FEATURE_SET  <- "BASE"               # "BASE" or "BASE_PLUS"
TEST_SPLIT   <- 0.70
TARGET_OPTS  <- list(quantile_q = 0.75)  # used only for *_QTL targets
OOS_START    <- NULL                      # e.g. "2010-01-01" or NULL → first test
SEED <- 42
# =================== CONFIG — STUDENTS EDIT ONLY ===================

GAM_K     <- 4L       # basis dimension per smooth term; try 3–6 (must be >= 3)
LINK_TYPE <- "logit"  # link for binomial GAM:
# allowed values: "logit", "probit", "cloglog"

# (rest: REGION, DATA_FILE, TARGET_ID, FEATURE_SET, TEST_SPLIT, etc.)


# ===================================================================

suppressPackageStartupMessages({
  library(readr); library(dplyr); library(tidyr); library(tibble)
  library(ggplot2); library(pROC); library(PRROC); library(caret)
  library(mgcv)
})

set.seed(SEED)
source("helpers_macro.R")

MODEL_TAG <- "GAM"

# --- Load, features, split anchor -----------------------------------
all0 <- load_data(DATA_FILE, REGION)
all1 <- build_features(all0)

# Anchor split on features-only frame
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

# --- Build design frames ------------------------------------------------
dtr <- all3[train_idx, c("date_q", FEATURES, TARGET_ID), drop = FALSE]
dte <- all3[test_idx,  c("date_q", FEATURES, TARGET_ID), drop = FALSE]
ytr <- as.integer(dtr[[TARGET_ID]])
yte <- as.integer(dte[[TARGET_ID]])

# --- Build a GAM formula: smooths on all features ---------------------
smooth_terms <- paste0("s(", FEATURES, ", k=4)")
gam_form <- as.formula(paste("y ~", paste(smooth_terms, collapse = " + ")))

# --- Fit GAM -----------------------------------------------------------
df_tr <- dtr[, FEATURES, drop = FALSE]
df_tr$y <- ytr

gam_fit <- mgcv::gam(gam_form, data = df_tr, family = binomial(link = LINK_TYPE), method = "REML")

# --- Predict and evaluate (70/30 test) --------------------------------
df_te <- dte[, FEATURES, drop = FALSE]
gam_prob_test <- as.numeric(predict(gam_fit, newdata = df_te, type = "response"))
thr_youden    <- youden_thr(gam_prob_test, yte)

cat(sprintf("\n=== DATA SUMMARY (%s) ===\n", REGION))
cat("Range:", as.character(min(all3$date_q)), "→", as.character(max(all3$date_q)),
    "| n =", nrow(all3), "\n")
cat("Class prevalence (", TARGET_ID, "==1):", round(mean(all3[[TARGET_ID]]), 3), "\n\n")

cat("\n=== 70/30 TEST METRICS — GAM ===\n")
mt_test <- dplyr::bind_rows(
  metrics_at("GAM (test)", gam_prob_test, yte, thr = 0.50),
  metrics_at("GAM (test)", gam_prob_test, yte, thr = thr_youden)
)
print(mt_test)

# --- Plots (test) -----------------------------------------------------
p_roc <- plot_roc_test(yte, gam_prob_test, sprintf("ROC — %s (70/30 test)", REGION))
p_pr  <- plot_pr_test (yte, gam_prob_test, sprintf("Precision–Recall — %s (70/30 test)", REGION))
p_cal <- plot_calibration(yte, gam_prob_test, sprintf("Calibration (deciles) — %s (70/30 test)", REGION))

print(p_roc); print(p_pr); print(p_cal)

# --- Expanding-window OOS ---------------------------------------------
fit_fn <- function(Xtr_, ytr_) {
  df_tr_ <- as.data.frame(Xtr_)
  df_tr_$y <- as.integer(ytr_)
  # Rebuild a formula using the same column names seen by expanding_oos()
  st <- paste0("s(", colnames(Xtr_), ", k=4)")
  fm <- as.formula(paste("y ~", paste(st, collapse = " + ")))
  mgcv::gam(fm, data = df_tr_, family = binomial(link = LINK_TYPE), method = "REML")
}
predict_fn <- function(mod, Xte_) {
  as.numeric(predict(mod, newdata = as.data.frame(Xte_), type = "response"))
}

start_date <- if (!is.null(OOS_START)) as.Date(OOS_START) else spl$first_test_date
oos <- expanding_oos(all3, FEATURES, TARGET_ID, start_date, fit_fn, predict_fn)

# OOS metrics
oos <- oos[is.finite(oos$prob), , drop = FALSE]
y_oos    <- as.integer(oos$y)
prob_oos <- oos$prob
stopifnot(length(prob_oos) > 0L, min(prob_oos) >= 0, max(prob_oos) <= 1)

thr_oos <- youden_thr(prob_oos, y_oos)

cat("\n=== EXPANDING-WINDOW OOS METRICS — GAM ===\n")
# KEEP only fixed or train-derived threshold:
mt_oos <- dplyr::bind_rows(
  metrics_at("GAM (OOS)", prob_oos, y_oos, thr = 0.50)
)
print(mt_oos); cat(sprintf("\nYouden threshold (OOS): %.3f\n", thr_oos))

# OOS plots
p_roc_oos <- plot_roc_test(y_oos, prob_oos, sprintf("ROC — %s (Expanding OOS)", REGION))
p_pr_oos  <- plot_pr_test (y_oos, prob_oos, sprintf("Precision–Recall — %s (Expanding OOS)", REGION))
p_tline   <- plot_timeline(oos$date_q, prob_oos, y_oos,
                           sprintf("%s predicted P(%s=1) — %s (Expanding OOS)", MODEL_TAG, TARGET_ID, REGION),
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
