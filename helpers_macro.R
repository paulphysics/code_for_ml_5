# helpers_macro.R
# Shared utilities for macro coursework templates

# --------- Data I/O ------------------------------------------------------------
load_data <- function(path, region) {
  df <- readr::read_csv(path, show_col_types = FALSE)
  df <- df[df$region == region, , drop = FALSE]
  df <- dplyr::arrange(df, date_q)
  df
}

# Add simple deltas for optional BASE_PLUS features
build_features <- function(df) {
  df %>%
    dplyr::mutate(
      dUNEMP  = UNEMP - dplyr::lag(UNEMP, 1),
      dINFL   = inflation_yoy - dplyr::lag(inflation_yoy, 1),
      dSPREAD = term_spread - dplyr::lag(term_spread, 1),
      dGDP    = gdp_growth_qoq - dplyr::lag(gdp_growth_qoq, 1)
    )
}

# Choose feature set
select_features <- function(df, feature_set = c("BASE","BASE_PLUS")) {
  feature_set <- match.arg(feature_set)
  feats <- switch(
    feature_set,
    BASE      = c("gdp_growth_qoq","UNEMP","inflation_yoy","POL","term_spread"),
    BASE_PLUS = c("gdp_growth_qoq","UNEMP","inflation_yoy","POL","term_spread",
                  "dUNEMP","dINFL","dSPREAD","dGDP")
  )
  list(X = df[, feats, drop = FALSE], names = feats)
}

# --------- Chronological split --------------------------------------------------
split_chrono <- function(df, test_split = 0.70) {
  n   <- nrow(df)
  stopifnot(n >= 10)
  cut <- floor(n * test_split)
  cut <- max(1L, min(cut, n - 1L))
  list(
    cut_index       = cut,
    cut_date        = df$date_q[cut],
    first_test_date = df$date_q[cut + 1L]
  )
}

# --------- Scaling --------------------------------------------------------------
scale_fit  <- function(dfX) {
  num <- dplyr::mutate_all(dfX, as.numeric)
  mu  <- sapply(num, mean, na.rm = TRUE)
  sdv <- sapply(num, stats::sd, na.rm = TRUE); sdv[sdv == 0] <- 1
  list(mu = mu, sd = sdv)
}
scale_apply <- function(dfX, sf) {
  as.data.frame(scale(dfX, center = sf$mu[colnames(dfX)], scale = sf$sd[colnames(dfX)]))
}

# --------- Targets (leakage-safe) ----------------------------------------------
# target_id: one of
#   "UNEMP_UP_NEXT", "INFL_UP_NEXT", "FLATTEN_NEXT",
#   "POL_UP_QTL", "GDP_SLOW_QTL", "STAG_NEXT", "RECESSION_NEXT"
# target_opts:
#   list(quantile_q = 0.75) for *_QTL targets
derive_target <- function(df, target_id, target_opts, train_until_date) {
  q <- df %>%
    dplyr::mutate(
      UNEMP_lead  = dplyr::lead(UNEMP, 1),
      INFL_lead   = dplyr::lead(inflation_yoy, 1),
      SPREAD_lead = dplyr::lead(term_spread, 1),
      GDP_lead    = dplyr::lead(gdp_growth_qoq, 1),
      POL_lead    = dplyr::lead(POL, 1)
    )
  train_mask <- q$date_q <= train_until_date
  lab <- switch(
    toupper(target_id),
    "UNEMP_UP_NEXT"     = as.integer(q$UNEMP_lead  > q$UNEMP),
    "INFL_UP_NEXT"      = as.integer(q$INFL_lead   > q$inflation_yoy),
    "FLATTEN_NEXT"      = as.integer(q$SPREAD_lead < q$term_spread),
    "POLICY_CUT_NEXT"   = as.integer(q$POL_lead    < q$POL),              # ← add
    "POL_UP_QTL" = {
      stopifnot(!is.null(target_opts$quantile_q))
      dpol_next <- q$POL_lead - q$POL
      thr <- stats::quantile(dpol_next[train_mask], probs = target_opts$quantile_q, na.rm = TRUE, type = 7)
      as.integer(dpol_next >= thr)
    },
    "GDP_SLOW_QTL" = {
      stopifnot(!is.null(target_opts$quantile_q))
      dgdp_next <- q$GDP_lead - q$gdp_growth_qoq
      thr <- stats::quantile(dgdp_next[train_mask], probs = 1 - target_opts$quantile_q, na.rm = TRUE, type = 7)
      as.integer(dgdp_next <= thr)
    },
    "STAG_NEXT"         = as.integer((q$UNEMP_lead > q$UNEMP) & (q$INFL_lead > q$inflation_yoy)),
    "RECESSION_NEXT"    = as.integer(q$recession_next),
    stop("Unknown target_id: ", target_id)
  )
  q[[target_id]] <- lab
  q
}


# --------- Metrics --------------------------------------------------------------
pr_auc <- function(probs, y) {
  pr <- PRROC::pr.curve(scores.class0 = probs[y == 1], scores.class1 = probs[y == 0], curve = FALSE)
  unname(pr$auc.integral)
}

youden_thr <- function(probs, y) {
  rocobj <- pROC::roc(response = as.integer(y), predictor = as.numeric(probs), quiet = TRUE)
  thr    <- pROC::coords(rocobj, x = "best", best.method = "youden", ret = "threshold")
  if (is.data.frame(thr) || is.matrix(thr)) thr <- thr[1, 1]
  as.numeric(thr)
}




bestF1_thr <- function(prob, y) {
  thr <- seq(0.05, 0.95, by = 0.01)
  f1  <- sapply(thr, function(t) {
    p <- ifelse(prob >= t, 1L, 0L)
    TP <- sum(p == 1 & y == 1); FP <- sum(p == 1 & y == 0); FN <- sum(p == 0 & y == 1)
    prec <- ifelse(TP + FP == 0, 0, TP / (TP + FP))
    rec  <- ifelse(TP + FN == 0, 0, TP / (TP + FN))
    ifelse(prec + rec == 0, 0, 2 * prec * rec / (prec + rec))
  })
  thr[which.max(f1)]
}
metrics_at <- function(name, probs, y, thr) {
  pred <- ifelse(probs >= thr, 1L, 0L)
  cm   <- caret::confusionMatrix(
    data = factor(pred, levels = c(0, 1)),
    reference = factor(y,   levels = c(0, 1)),
    positive = "1"
  )
  tibble::tibble(
    model        = name,
    threshold    = round(thr, 3),
    ROC_AUC      = round(as.numeric(pROC::auc(pROC::roc(y, probs, quiet = TRUE))), 3),
    PR_AUC       = round(pr_auc(probs, y), 3),
    Accuracy     = round(unname(cm$overall["Accuracy"]), 3),
    Kappa        = round(unname(cm$overall["Kappa"]), 3),
    Recall       = round(unname(cm$byClass["Sensitivity"]), 3),
    Precision    = round(unname(cm$byClass["Precision"]), 3),
    Specificity  = round(unname(cm$byClass["Specificity"]), 3),
    Bal_Accuracy = round(unname(cm$byClass["Balanced Accuracy"]), 3),
    Brier        = round(mean((probs - y)^2), 4)
  )
}

# --------- Plotters -------------------------------------------------------------
plot_roc_test <- function(y, probs, title) {
  rocobj <- pROC::roc(y, probs, quiet = TRUE)
  aucv   <- as.numeric(pROC::auc(rocobj))
  pROC::ggroc(list(Model = rocobj), legacy.axes = TRUE) +
    ggplot2::labs(title = title, x = "False Positive Rate (1 − Specificity)", y = "True Positive Rate") +
    ggplot2::annotate("text", x = 0.6, y = 0.2, label = sprintf("AUC = %.3f", aucv)) +
    ggplot2::theme_minimal()
}
plot_pr_test <- function(y, probs, title) {
  pr <- PRROC::pr.curve(scores.class0 = probs[y == 1], scores.class1 = probs[y == 0], curve = TRUE)
  base_rate <- mean(y)
  df <- tibble::tibble(recall = pr$curve[, 1], precision = pr$curve[, 2])
  ggplot2::ggplot(df, ggplot2::aes(recall, precision)) +
    ggplot2::geom_line() +
    ggplot2::geom_hline(yintercept = base_rate, linetype = "dashed") +
    ggplot2::labs(title = title, subtitle = sprintf("P(1) baseline = %.3f | PR-AUC = %.3f", base_rate, unname(pr$auc.integral)),
                  x = "Recall", y = "Precision") +
    ggplot2::coord_equal() + ggplot2::theme_minimal()
}
plot_calibration <- function(y, probs, title) {
  df <- tibble::tibble(prob = probs, y = as.integer(y)) %>%
    dplyr::mutate(bin = dplyr::ntile(prob, 10)) %>%
    dplyr::group_by(bin) %>%
    dplyr::summarise(mean_pred = mean(prob), obs_rate = mean(y), n = dplyr::n(), .groups = "drop")
  ggplot2::ggplot(df, ggplot2::aes(mean_pred, obs_rate)) +
    ggplot2::geom_point() + ggplot2::geom_line() +
    ggplot2::geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
    ggplot2::labs(title = title, x = "Mean predicted probability", y = "Observed event rate") +
    ggplot2::theme_minimal()
}
plot_timeline <- function(dates, probs, y, title, subtitle) {
  df <- tibble::tibble(date_q = dates, prob = probs, y = y)
  ggplot2::ggplot(df, ggplot2::aes(date_q, prob)) +
    ggplot2::geom_line() +
    ggplot2::geom_point(data = df[df$y == 1, , drop = FALSE],
                        ggplot2::aes(date_q, prob),
                        inherit.aes = FALSE, size = 2, alpha = 0.85) +
    ggplot2::labs(title = title, subtitle = subtitle, x = NULL, y = "Predicted probability") +
    ggplot2::theme_minimal()
}

# --------- Expanding-window OOS -------------------------------------------------
# fit_fn: function(Xtr, ytr) -> model
# predict_fn: function(model, Xte) -> numeric probs
# Add train-thresholding inside the OOS loop
expanding_oos <- function(df, feature_names, target_col, start_date, fit_fn, predict_fn, thr_fun = NULL) {
  n <- nrow(df); out <- vector("list", 0L)
  for (i in seq_len(n)) {
    if (df$date_q[i] < start_date) next
    tr <- df[df$date_q < df$date_q[i], , drop = FALSE]
    te <- df[i, , drop = FALSE]
    if (!nrow(tr)) next
    Xtr <- as.matrix(tr[, feature_names, drop = FALSE])
    ytr <- as.integer(tr[[target_col]])
    Xte <- as.matrix(te[, feature_names, drop = FALSE])

    mod <- fit_fn(Xtr, ytr)
    pr_te <- as.numeric(predict_fn(mod, Xte))

    thr <- NA_real_; yhat <- NA_integer_
    if (!is.null(thr_fun)) {
      pr_tr <- as.numeric(predict_fn(mod, Xtr))
      thr   <- thr_fun(pr_tr, ytr)          # train-only threshold
      yhat  <- as.integer(pr_te >= thr)     # one-step classification
    }
    out[[length(out) + 1L]] <-
      tibble::tibble(date_q = te$date_q, y = as.integer(te[[target_col]]),
                     prob = pr_te, thr = thr, yhat = yhat)
  }
  if (length(out) == 0L)
    return(tibble::tibble(date_q = as.Date(character()), y = integer(), prob = numeric(),
                          thr = numeric(), yhat = integer()))
  dplyr::bind_rows(out)
}


# --------- Output helpers -------------------------------------------------------
stamp_outdir <- function(model_tag, region, target_id) {
  ts <- format(Sys.time(), "%Y%m%d_%H%M%S")
  dir <- file.path("outputs", sprintf("%s_%s_%s_%s", model_tag, region, target_id, ts))
  dir.create(dir, recursive = TRUE, showWarnings = FALSE)
  dir
}
save_tbl <- function(df, path) readr::write_csv(df, path)
save_plot <- function(plot_obj, path, width = 7, height = 5) ggplot2::ggsave(filename = path, plot = plot_obj, width = width, height = height, dpi = 120)



ensure_targets <- function(df) {
  # Requires at least date_q, UNEMP, gdp_growth_qoq
  req <- c("date_q", "UNEMP", "gdp_growth_qoq")
  miss <- setdiff(req, names(df))
  if (length(miss)) stop("ensure_targets(): missing columns: ", paste(miss, collapse=", "))

  df <- df %>% arrange(date_q)

  # 1) Next-quarter unemployment uptick
  if (!"unemp_up_next" %in% names(df)) {
    df <- df %>% mutate(unemp_up_next = as.integer(dplyr::lead(UNEMP, 1) > UNEMP))
  } else {
    df <- df %>% mutate(unemp_up_next = as.integer(unemp_up_next))
  }

  # 1b) Next-quarter inflation uptick
  if ("inflation_yoy" %in% names(df)) {
    if (!"infl_up_next" %in% names(df)) {
      df <- df %>% mutate(infl_up_next = as.integer(dplyr::lead(inflation_yoy, 1) > inflation_yoy))
    } else {
      df <- df %>% mutate(infl_up_next = as.integer(infl_up_next))
    }
  }

  # 1c) Next-quarter policy cut (lead(POL) < POL)
  if ("POL" %in% names(df)) {
    if (!"policy_cut_next" %in% names(df)) {
      df <- df %>% mutate(policy_cut_next = as.integer(dplyr::lead(POL, 1) < POL))
    } else {
      df <- df %>% mutate(policy_cut_next = as.integer(policy_cut_next))
    }
  }

  # 1d) Next-quarter term-structure flattening (lead(spread) < spread)
  if ("term_spread" %in% names(df)) {
    if (!"flatten_next" %in% names(df)) {
      df <- df %>% mutate(flatten_next = as.integer(dplyr::lead(term_spread, 1) < term_spread))
    } else {
      df <- df %>% mutate(flatten_next = as.integer(flatten_next))
    }
  }

  # 2) Recession tag (proxy if absent): two consecutive negative QoQ GDP growth
  if (!"recession" %in% names(df)) {
    df <- df %>% mutate(recession = as.numeric(
      gdp_growth_qoq < 0 & dplyr::lag(gdp_growth_qoq, 1) < 0
    ))
  } else {
    df <- df %>% mutate(recession = as.numeric(recession))
  }

  # 3) Next-quarter recession flag
  if (!"recession_next" %in% names(df)) {
    df <- df %>% mutate(recession_next = dplyr::lead(recession, 1))
  } else {
    df <- df %>% mutate(recession_next = as.integer(recession_next))
  }

  df
}

predict_glmnet_prob <- function(fit, newx, s) {
  p <- drop(glmnet::predict.glmnet(fit, newx = newx, type = "response", s = s))
  if (any(!is.finite(p)) || any(p < 0) || any(p > 1)) {
    eta <- drop(glmnet::predict.glmnet(fit, newx = newx, type = "link", s = s))
    p <- plogis(eta)
  }
  pmin(pmax(p, 1e-9), 1 - 1e-9)
}



# ---- Small shared helpers for glmnet ------------------------------------------
to_matrix <- function(df, cols) {
  m <- as.matrix(df[, cols, drop = FALSE]); storage.mode(m) <- "double"; m
}

glmnet_fit_cv <- function(Xtr, ytr, alpha, nfolds = 5, use_l1se = TRUE, standardize = TRUE) {
  cv <- glmnet::cv.glmnet(
    x = Xtr, y = ytr, family = "binomial", alpha = alpha,
    nfolds = nfolds, type.measure = "deviance", standardize = standardize
  )

  lam1se <- cv$lambda.1se
  lammin <- cv$lambda.min

  # fit at lambda.1se
  fit1 <- glmnet::glmnet(x = Xtr, y = ytr, family = "binomial",
                         alpha = alpha, lambda = lam1se, standardize = standardize)
  beta1 <- as.matrix(glmnet::coef.glmnet(fit1, s = lam1se))
  nz1   <- sum(beta1[-1, , drop = FALSE] != 0)

  # if all non-intercept coefs are zero, refit at lambda.min
  if (use_l1se && nz1 == 0) {
    fit2 <- glmnet::glmnet(x = Xtr, y = ytr, family = "binomial",
                           alpha = alpha, lambda = lammin, standardize = standardize)
    beta2 <- as.matrix(glmnet::coef.glmnet(fit2, s = lammin))
    nz2   <- sum(beta2[-1, , drop = FALSE] != 0)
    return(list(fit = fit2, lambda = lammin, cv = cv, nz = nz2))
  } else {
    return(list(fit = fit1, lambda = lam1se, cv = cv, nz = nz1))
  }
}


