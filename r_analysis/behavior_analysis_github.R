## ============================================================================
## Behavior Analysis: BL6 vs. CD1 social interaction sessions
## 9 animals, 2 cohorts (Coh1, Coh2) -- deepOF bout-level output
## PNG version (interactive HTML/plotly version is a separate later script)
## ============================================================================

## ---- 0. Packages -----------------------------------------------------------
required_pkgs <- c("tidyverse","patchwork", "coin", "boot", "ggbeeswarm", "ggrepel")

new_pkgs <- required_pkgs[!(required_pkgs %in% installed.packages()[, "Package"])]
if (length(new_pkgs) > 0) install.packages(new_pkgs)

library(tidyverse)
library(patchwork)
library(coin)
library(boot)
library(ggbeeswarm)
library(ggrepel)


# ---- BCa Bootstrap CI (replaces percentile bootstrap for small n) ----
bca_ci <- function(x, R = 2000, conf = 0.95) {
  x <- x[is.finite(x)]
  if (length(x) < 3 || all(x == x[1], na.rm = TRUE)) {
    return(c(NA_real_, NA_real_))
  }
  boot_mean <- function(data, indices) mean(data[indices], na.rm = TRUE)
  boot_obj <- tryCatch(
    boot::boot(x, boot_mean, R = R),
    error = function(e) NULL
  )
  if (is.null(boot_obj)) return(c(NA_real_, NA_real_))
  ci <- tryCatch(
    boot::boot.ci(boot_obj, type = "bca", conf = conf)$bca[4:5],
    error = function(e) c(NA_real_, NA_real_)
  )
  ci
}




## ---- 1. CONFIG: paths ------------------------------------------------------
path_bl6_dir <- "/Users/Path/to/Folder"
path_cd1_dir <- "/Users/Path/to/Folder"

## cell-count/overlap output from the histology script (for the later
## behavior-vs-cell-count correlation block, once both datasets are ready)
path_cellcount_dir <- "/Users/Path/to/Folder"

output_dir <- "/Users/Path/to/Folder"
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

# ---- Start console logging ----
log_file <- file.path(output_dir, "console_output.log")
sink(log_file, split = TRUE)
message("Console output logged to: ", log_file)

## ---- 1b. CONFIG: behaviors -------------------------------------------------
## Edit these two vectors to add/remove behaviors from the analysis.
behaviors_directional    <- c("nose2tail", "nose2body", "following")
behaviors_nondirectional <- c("nose2nose", "sniffing", "moving", "sidebyside", "immobility")
all_behaviors <- c(behaviors_directional, behaviors_nondirectional)

## ---- 1c. CONFIG: minimum bout duration (bouts shorter than this are dropped) ----
## Single fixed threshold, applied to all behaviors.
min_duration_default <- 0.5   # seconds

get_min_duration <- function(behavior) {
  min_duration_default
}

## ---- 1d. CONFIG: cohorts ----------------------------------------------------
## Same grouping as the histology analysis, but count_0 IS included here
## (it has no cell-count data, but behavior data exists for it).
cohort1_animals <- c("count_0", "count_3", "count_8", "count_10", "count_2")
cohort2_animals <- c("count_4", "count_5", "count_6", "count_9")
all_animals <- c(cohort1_animals, cohort2_animals)

## ---- 1e. CONFIG: color palettes (from ipacl_analysis.R) --------------------
pastel_pal <- c("#FFCBE1", "#D6E5BD", "#F9E1A8", "#BCD8EC", "#DCCCEC", "#FFDAB4", "#C4E0D9", "#F5D0C5")
stroke_pal <- c("#D97CA0", "#8FA86E", "#C9A24B", "#5A93BB", "#A98FC7", "#D99A5C", "#89BEB5", "#D99A8C")
session_pal        <- c(BL6 = pastel_pal[4], CD1 = pastel_pal[1])
session_stroke_pal <- c(BL6 = stroke_pal[4], CD1 = stroke_pal[1])
session_shape_pal  <- c(BL6 = 16, CD1 = 17)

cohort_pal <- c(Coh1 = pastel_pal[4], Coh2 = pastel_pal[1]) 
cohort_stroke_pal <- c(Coh1 = stroke_pal[4], Coh2 = stroke_pal[1])

## ---- Fixed color mapping for behaviors (consistent across all ethograms) ----
behavior_colors <- setNames(
  pastel_pal[1:8],
  c("nose2nose", "nose2tail", "nose2body", "sniffing", 
    "moving", "sidebyside", "immobility", "following")
)

## ---- 1f. THEME --------------------------------------------------------------
theme_thesis <- theme_minimal(base_size = 15) +
  theme(
    panel.grid.minor = element_blank(),
    plot.title = element_text(face = "bold", size = 15, hjust = 0.5),
    plot.subtitle = element_text(size = 11, color = "grey30", hjust = 0.5),
    plot.caption = element_text(size = 11, color = "grey45", hjust = 0.5, face = "italic",
                                lineheight = 1.3, margin = margin(t = 8)),
    plot.margin = margin(t = 12, r = 50, b = 10, l = 80),
    strip.text = element_text(size = 12, face = "bold"),
    axis.title = element_text(size = 13),
    axis.text = element_text(size = 12),
    legend.position = "bottom",
    legend.text = element_text(size = 11),
    legend.title = element_blank()
  )

## Nicer display names for plot titles (edit here if wording should change)
behavior_display_names <- c(
  nose2nose  = "Nose-to-nose",
  nose2tail  = "Nose-to-tail",
  nose2body  = "Nose-to-body",
  sniffing   = "Sniffing",
  moving     = "Moving",
  sidebyside = "Side-by-side",
  immobility = "Immobility"
)

sig_stars <- function(p) {
  ifelse(is.na(p), "",
  ifelse(p < 0.001, "***",
  ifelse(p < 0.01,  "**",
  ifelse(p < 0.05,  "*", "ns"))))
}

## ============================================================================
## 2. HELPER FUNCTIONS
## ============================================================================

## Strip the session-type suffix from the Animal column (CD1 files have it,
## e.g. "count_0_CD1"; BL6 files don't), so the animal ID is consistent
## across both session types.
strip_animal_suffix <- function(x) {
  x <- sub("_CD1$", "", x)
  x <- sub("_BL6$", "", x)
  x
}

## Parse a Behaviour string into actor / target / behavior.
## BL6-file labels already use exp/stim. CD1-file labels use strain names
## (BL6/CD1) instead -- BL6 = exp animal, CD1 = stim animal -- so those get
## mapped onto the same exp/stim scheme here, unifying both file types.
parse_behaviour <- function(behaviour_str, session_type) {
  parts <- strsplit(behaviour_str, "_")[[1]]
  n <- length(parts)
  behavior <- parts[n]
  actor_tokens <- parts[-n]

  map_token <- function(tok) {
    if (session_type == "CD1") {
      if (tok == "BL6") return("exp")
      if (tok == "CD1") return("stim")
    }
    if (tok %in% c("exp", "stim")) return(tok)
    NA_character_
  }

  if (length(actor_tokens) == 0) {
    actor <- NA_character_; target <- NA_character_
  } else if (length(actor_tokens) == 1) {
    actor <- map_token(actor_tokens[1]); target <- NA_character_
  } else if (length(actor_tokens) == 2) {
    actor  <- map_token(actor_tokens[1])
    target <- map_token(actor_tokens[2])
  } else {
    stop(paste("Unexpected Behaviour format:", behaviour_str))
  }
  list(actor = actor, target = target, behavior = behavior)
}

## hh:mm:ss.ffffff -> seconds
hms_to_sec <- function(x) {
  parts <- strsplit(x, ":")
  sapply(parts, function(p) {
    h <- as.numeric(p[1]); m <- as.numeric(p[2]); s <- as.numeric(p[3])
    h * 3600 + m * 60 + s
  })
}

## Read every CSV in one session-type folder into one long data frame.
read_session_folder <- function(dir_path, session_type) {
  files <- list.files(dir_path, pattern = "\\.csv$", full.names = TRUE)
  if (length(files) == 0) {
    warning(paste("No CSV files found in:", dir_path))
    return(NULL)
  }
  all_rows <- list()
  for (f in files) {
    df <- read.csv(f, stringsAsFactors = FALSE)
    colnames_lower <- tolower(names(df))
    
    # find columns (case-insensitive)
    animal_col <- grep("^animal$", colnames_lower, value = TRUE)
    start_col  <- grep("^start$", colnames_lower, value = TRUE)
    end_col    <- grep("^end$", colnames_lower, value = TRUE)
    behav_col  <- grep("^behavior$|^behaviour$", colnames_lower, value = TRUE)
    dur_col    <- grep("duration", colnames_lower, value = TRUE)
    
    # check that all required columns exist
    missing <- c()
    if (length(animal_col) == 0) missing <- c(missing, "animal")
    if (length(start_col) == 0)  missing <- c(missing, "start")
    if (length(end_col) == 0)    missing <- c(missing, "end")
    if (length(behav_col) == 0)  missing <- c(missing, "behavior")
    
    if (length(missing) > 0) {
      stop(paste("In file", basename(f), "missing columns:", paste(missing, collapse = ", "),
                 "\nAvailable columns:", paste(names(df), collapse = ", ")))
    }
    
    # use the columns found
    df_clean <- data.frame(
      animal_id    = strip_animal_suffix(df[[animal_col[1]]]),
      session_type = session_type,
      start_sec    = hms_to_sec(df[[start_col[1]]]),
      end_sec      = hms_to_sec(df[[end_col[1]]]),
      behavior     = df[[behav_col[1]]],
      stringsAsFactors = FALSE
    )
    
    # duration column
    if (length(dur_col) == 1) {
      df_clean$duration_sec <- as.numeric(df[[dur_col[1]]])
    } else {
      stop(paste("No duration column found in", basename(f), "– columns:", paste(names(df), collapse = ", ")))
    }
    
    # Parse behavior
    parsed <- lapply(df_clean$behavior, parse_behaviour, session_type = session_type)
    df_clean$actor  <- sapply(parsed, function(p) p$actor)
    df_clean$target <- sapply(parsed, function(p) p$target)
    df_clean$behavior <- sapply(parsed, function(p) p$behavior)
    
    all_rows[[f]] <- df_clean[, c("animal_id", "session_type", "behavior",
                                  "actor", "target", "start_sec", "end_sec", "duration_sec")]
  }
  do.call(rbind, all_rows)
}

## ============================================================================
## 3. READ + FILTER
## ============================================================================
bl6_data <- read_session_folder(path_bl6_dir, "BL6")
cd1_data <- read_session_folder(path_cd1_dir, "CD1")
all_data <- rbind(bl6_data, cd1_data)

threshold_vec <- sapply(all_data$behavior, get_min_duration)
n_before <- nrow(all_data)
all_data <- all_data[all_data$duration_sec >= threshold_vec, ]
message("Min-duration filter: kept ", nrow(all_data), " of ", n_before, " bouts.")

all_data$cohort <- ifelse(all_data$animal_id %in% cohort1_animals, "Coh1",
                    ifelse(all_data$animal_id %in% cohort2_animals, "Coh2", NA))

## ============================================================================
## 4. AGGREGATION: three views per behavior -- total (both animals), exp-only,
## stim-only -- plus the existing direction-split (initiated_by), with
## explicit zeros so an animal with 0 bouts shows up as 0, not missing.
## ============================================================================
full_grid <- expand.grid(animal_id = all_animals, session_type = c("BL6", "CD1"),
                          behavior = all_behaviors, stringsAsFactors = FALSE)

## Generic aggregator: actor_filter = NULL for "total" (all rows), or
## "exp"/"stim" to keep only rows where that animal was the actor.
aggregate_view <- function(actor_filter = NULL) {
  rows <- if (is.null(actor_filter)) all_data else all_data[all_data$actor == actor_filter & !is.na(all_data$actor), ]
  
  # Fallback for empty rows: dur_sum/freq_sum remain as 0-row table with the correct columns
  if (nrow(rows) == 0) {
    dur_sum <- data.frame(animal_id = character(), session_type = character(), 
                          behavior = character(), total_duration_sec = numeric())
    freq_sum <- data.frame(animal_id = character(), session_type = character(), 
                           behavior = character(), frequency = numeric())
  } else {
    dur_sum  <- aggregate(duration_sec ~ animal_id + session_type + behavior, rows, sum)
    freq_sum <- aggregate(duration_sec ~ animal_id + session_type + behavior, rows, length)
    names(dur_sum)[4]  <- "total_duration_sec"
    names(freq_sum)[4] <- "frequency"
  }
  
  agg <- merge(full_grid, dur_sum, all.x = TRUE)
  agg <- merge(agg, freq_sum, all.x = TRUE)
  agg$total_duration_sec[is.na(agg$total_duration_sec)] <- 0
  agg$frequency[is.na(agg$frequency)] <- 0
  agg$cohort <- ifelse(agg$animal_id %in% cohort1_animals, "Coh1", "Coh2")
  agg
}

agg_total <- aggregate_view(NULL)
agg_exp   <- aggregate_view("exp")
agg_stim  <- aggregate_view("stim")

# ---- Safety check: every animal has both sessions for every behaviour----
# (confirms that animal-level aggregation is correct)
balance_check <- agg_total %>%
  group_by(behavior, animal_id) %>%
  summarise(n_sessions = n_distinct(session_type), .groups = "drop") %>%
  filter(n_sessions != 2)

if (nrow(balance_check) > 0) {
  warning("Not all animals have both sessions for every behavior – check the data!")
  print(balance_check)
} else {
  message("All animals have both sessions for every behavior – aggregation correct.")
}

## Behaviors where an actor is ever recorded at all (excludes purely
## non-directional, no-actor behaviors like immobility -- an exp/stim split
##, so they only get a "total" view/plot).
behaviors_with_actor <- Filter(function(b) any(!is.na(all_data$actor[all_data$behavior == b])), all_behaviors)

## Existing direction-split (initiated_by exp/stim), only for the truly
## directional behaviors (nose2tail, nose2body) -- zero-filled like the
## other aggregations, so an animal with 0 exp-initiated (or stim-initiated)
## bouts shows up as 0, not missing.
full_grid_direction <- expand.grid(animal_id = all_animals, session_type = c("BL6", "CD1"),
                                    behavior = behaviors_directional, initiated_by = c("exp", "stim"),
                                    stringsAsFactors = FALSE)

directional_rows <- all_data[all_data$behavior %in% behaviors_directional & !is.na(all_data$actor), ]
dur_dir  <- aggregate(duration_sec ~ animal_id + session_type + behavior + actor, directional_rows, sum)
freq_dir <- aggregate(duration_sec ~ animal_id + session_type + behavior + actor, directional_rows, length)
names(dur_dir)[5]  <- "total_duration_sec"
names(freq_dir)[5] <- "frequency"
names(dur_dir)[names(dur_dir) == "actor"]   <- "initiated_by"
names(freq_dir)[names(freq_dir) == "actor"] <- "initiated_by"

agg_direction <- merge(full_grid_direction, dur_dir, all.x = TRUE)
agg_direction <- merge(agg_direction, freq_dir, all.x = TRUE)
agg_direction$total_duration_sec[is.na(agg_direction$total_duration_sec)] <- 0
agg_direction$frequency[is.na(agg_direction$frequency)] <- 0
agg_direction$cohort <- ifelse(agg_direction$animal_id %in% cohort1_animals, "Coh1", "Coh2")

# ============================================================================
# BL6 vs. CD1 – Main analysis (only duration, 8 behaviours, Holm‑corrected)
# ============================================================================

# --- 1. Behaviours ---
behaviors_directional <- c("nose2tail", "nose2body", "following")
behaviors_nondirectional <- c("nose2nose", "sniffing", "moving", "sidebyside", "immobility")
all_behaviors <- c(behaviors_directional, behaviors_nondirectional)

# Aggregated data for "total" (non-directional) and "exp" (directional) / (agg_total and agg_exp already exist from Section 4)

#Choose the appropriate view for each behavior / Build a table mapping behavior -> view
view_lookup <- tibble(
  behavior = all_behaviors,
  view = ifelse(behavior %in% behaviors_directional, "exp", "total")
)

# --- 4. function: paired Wilcoxon + effect size + BCa (coin::wilcoxsign_test) ---
run_paired_wilcoxon <- function(agg_df, behavior, metric = "total_duration_sec") {
  sub <- agg_df[agg_df$behavior == behavior, ]
  
  # data in long format for coin::wilcoxsign_test
  long_df <- sub %>%
    select(animal_id, session_type, all_of(metric)) %>%
    pivot_longer(cols = all_of(metric), names_to = "metric_type", values_to = "value") %>%
    mutate(
      animal_id = factor(animal_id),                     
      session_type = factor(session_type, levels = c("BL6", "CD1"))  
    ) %>%
    arrange(animal_id, session_type)
  
  # wide format for x/y (for effect size and BCa)
  wide <- long_df %>%
    select(animal_id, session_type, value) %>%
    pivot_wider(names_from = session_type, values_from = value)
  
  x <- wide$BL6
  y <- wide$CD1
  n_pairs <- sum(complete.cases(x, y))
  
  if (n_pairs < 2 || all(x == y, na.rm = TRUE)) {
    return(tibble(
      behavior = behavior,
      n = n_pairs,
      p_value = NA_real_,
      effect_r = NA_real_,
      ci_low = NA_real_,
      ci_high = NA_real_,
      note = ifelse(n_pairs < 2, "too few pairs", "all diffs zero")
    ))
  }
  
  # ---- coin::wilcoxsign_test  ----
  wt <- coin::wilcoxsign_test(value ~ session_type | animal_id, 
                              data = long_df, 
                              distribution = "exact")
  p_value <- coin::pvalue(wt)
  
  # rank-biserial r (via coin)
  r <- tryCatch({
    as.numeric(coin::statistic(wt, "standardized")) / sqrt(n_pairs)
  }, error = function(e) NA_real_)
  
  # BCa Bootstrap 95% CI (indicative due to small n)
  diffs <- x - y
  set.seed(123)
  ci <- bca_ci(diffs, R = 2000)
  
  tibble(
    behavior = behavior,
    n = n_pairs,
    p_value = p_value,
    effect_r = r,
    ci_low = unname(ci[1]),
    ci_high = unname(ci[2]),
    note = ""
  )
}

# --- 5. run tests for all 8 behaviours ---
results_dur <- map_dfr(all_behaviors, function(b) {
  view <- view_lookup$view[view_lookup$behavior == b]
  agg_df <- if (view == "total") agg_total else agg_exp
  run_paired_wilcoxon(agg_df, b, metric = "total_duration_sec")
})

# --- 6. Holm-correction (one family, 8 Tests) ---
results_dur <- results_dur %>%
  mutate(p_holm = p.adjust(p_value, method = "holm"))

# save
write_csv(results_dur, file.path(output_dir, "bl6_vs_cd1_duration_8behaviors_holm.csv"))

main_test_medians <- map_dfr(all_behaviors, function(b) {
  view <- view_lookup$view[view_lookup$behavior == b]
  agg_df <- if (view == "total") agg_total else agg_exp
  sub <- agg_df[agg_df$behavior == b, ]
  tibble(
    behavior = b,
    median_BL6 = median(sub$total_duration_sec[sub$session_type == "BL6"], na.rm = TRUE),
    median_CD1 = median(sub$total_duration_sec[sub$session_type == "CD1"], na.rm = TRUE)
  )
})
write_csv(main_test_medians, file.path(output_dir, "bl6_vs_cd1_medians.csv"))
# --- 7. plot-function (with cohort-shapes) ---
make_paired_duration_plot <- function(behavior, view) {
  agg_df <- if (view == "total") agg_total else agg_exp
  plot_df <- agg_df[agg_df$behavior == behavior, ]
  
  # n for x-axis
  n_animals <- length(unique(plot_df$animal_id))
  x_labels <- setNames(
    paste0(c("BL6", "CD1"), "\n(n=", n_animals, ")"),
    c("BL6", "CD1")
  )
  
  stats_row <- results_dur[results_dur$behavior == behavior, ]
  
  subtitle_txt <- if (!is.na(stats_row$p_holm)) {
    paste0(
      "Wilcoxon (Holm): p = ", signif(stats_row$p_holm, 3),
      ifelse(stats_row$p_holm < 0.05, " *", ""),
      ", r = ", round(stats_row$effect_r, 2),
      "\n95% CI [", round(stats_row$ci_low, 1), ", ", round(stats_row$ci_high, 1), "]",
      " (BCa, indicative at n=", n_animals, ")"
    )
  } else {
    "no test (too few pairs or all equal)"
  }
  
  display_name <- if (behavior %in% names(behavior_display_names)) {
    behavior_display_names[[behavior]]
  } else behavior
  view_suffix <- if (view == "total") " (total)" else " (exp-only)"
  title_txt <- paste0(display_name, view_suffix, " – BL6 vs. CD1 (duration)")
  
  # ---- Median per group ----
  medians <- plot_df %>%
    group_by(session_type) %>%
    summarise(median_val = median(total_duration_sec, na.rm = TRUE), .groups = "drop")
  
  # ---- Plot ----
  p <- ggplot(plot_df, aes(x = session_type, y = total_duration_sec,
                           fill = session_type, color = session_type,
                           shape = cohort)) +
    geom_line(aes(group = animal_id), color = "grey50", alpha = 0.5, linewidth = 0.8) +
    geom_point(aes(fill = session_type), size = 3, stroke = 1, alpha = 0.9,
               position = position_nudge(x = 0)) +
    # Median as thick horizontal line
    geom_segment(data = medians,
                 aes(x = as.numeric(factor(session_type)) - 0.25,
                     xend = as.numeric(factor(session_type)) + 0.25,
                     y = median_val, yend = median_val,
                     color = session_type),
                 inherit.aes = FALSE, linewidth = 1.2) +
    scale_shape_manual(values = c(16, 17)) +
    scale_fill_manual(values = session_pal) +
    scale_colour_manual(values = session_stroke_pal) +
    scale_x_discrete(labels = x_labels) +
    scale_y_continuous(limits = c(0, NA), expand = expansion(mult = c(0, 0.1))) +
    labs(
      title = title_txt,
      subtitle = subtitle_txt,
      x = "Session type",
      y = "Total duration (s)",
      caption = paste0(
        "Minimum bout duration: ", get_min_duration(behavior), "s. ",
        if (view == "exp") "Only bouts initiated by the experimental animal." else "Both animals combined (total)."
      )
    ) +
    theme_thesis +
    theme(legend.position = "none") 
}


# --- 9. additionally: Mega-Grid with all 8 plots (2x4) ---
plots_list <- map(all_behaviors, function(b) {
  view <- view_lookup$view[view_lookup$behavior == b]
  p <- make_paired_duration_plot(b, view)
  p <- p + labs(title = sub(" \\(duration\\)$", "", p$labels$title))
  p +
    theme(
      plot.title = element_text(size = 11, face = "bold", hjust = 0.5),
      plot.subtitle = element_text(size = 10, color = "grey30", hjust = 0.5),
      axis.title = element_text(size = 11),
      axis.text = element_text(size = 10),
      plot.caption = element_blank()
    )
})

mega_plot <- wrap_plots(plots_list, ncol = 4, guides = "collect") +
  plot_annotation(
    title = "BL6 vs. CD1 – Duration per behavior",
    subtitle = paste0(
      "Paired Wilcoxon, Holm-corrected over 8 tests. n = ", 
      length(unique(agg_total$animal_id))
    ),
    caption = paste0(
      "Minimum bout duration: 0.5s. Directional behaviors (nose2tail, nose2body, following): exp-only view.\n",
      "Non-directional behaviors (nose2nose, sniffing, moving, sidebyside, immobility): total view. ",
      "BCa bootstrap CI – indicative."
    ),
    theme = theme(
      plot.title = element_text(hjust = 0.5, face = "bold", size = 20, color = "black"),
      plot.subtitle = element_text(hjust = 0.5, size = 13, color = "grey30"),
      plot.caption = element_text(hjust = 0.5, size = 12, color = "grey45", face = "italic", margin = margin(t = 10, b = 10))
    )
  ) &
  theme(legend.position = "bottom")

ggsave(file.path(output_dir, "bl6_vs_cd1_duration_mega.png"),
       mega_plot, width = 14, height = 8, dpi = 300)

message("Neue BL6-vs-CD1-Dauer-Analyse abgeschlossen. Ergebnisse in ", output_dir)


## ============================================================================
## 8. DIRECTION-SPLIT (exploratory): exp vs. stim initiator, only for duration
## nose2tail, nose2body, following -- separate per session (BL6, CD1)
## EXPLORATORY: no formal p-value correction; report effect size + CI.
## ============================================================================

#' Paired comparison: exp-initiated vs. stim-initiated bouts (same animal, same session)
#' Returns raw p-value (for information only), effect size r, and bootstrap CI.
run_paired_direction_comparison <- function(behavior, session_filter, metric_col = "total_duration_sec") {
  sub <- agg_direction[agg_direction$behavior == behavior & agg_direction$session_type == session_filter, ]
  
  # data in long format for coin::wilcoxsign_test
  long_df <- sub %>%
    select(animal_id, initiated_by, all_of(metric_col)) %>%
    pivot_longer(cols = all_of(metric_col), names_to = "metric_type", values_to = "value") %>%
    mutate(
      animal_id = factor(animal_id),                     # <-- NEU
      initiated_by = factor(initiated_by, levels = c("exp", "stim"))  # <-- NEU
    ) %>%
    arrange(animal_id, initiated_by)
  
  # wide format for x/y (for effect size and BCa)
  wide <- long_df %>%
    select(animal_id, initiated_by, value) %>%
    pivot_wider(names_from = initiated_by, values_from = value)
  
  exp_vals <- wide$exp
  stim_vals <- wide$stim
  
  if (all(exp_vals == stim_vals, na.rm = TRUE) || length(exp_vals) < 2) {
    return(data.frame(behavior = behavior, session_type = session_filter,
                      n = length(exp_vals), p_value = NA, effect_r = NA,
                      ci_low = NA, ci_high = NA, note = "all diffs zero or too few"))
  }
  
  # ---- coin::wilcoxsign_test ----
  wt <- coin::wilcoxsign_test(value ~ initiated_by | animal_id, 
                              data = long_df, 
                              distribution = "exact")
  p_value <- coin::pvalue(wt)
  
  # rank-biserial r
  r <- tryCatch({
    as.numeric(coin::statistic(wt, "standardized")) / sqrt(length(exp_vals))
  }, error = function(e) NA_real_)
  
  # BCa Bootstrap 95% CI
  diffs <- exp_vals - stim_vals
  set.seed(123)
  ci <- bca_ci(diffs, R = 2000)
  
  data.frame(
    behavior = behavior,
    session_type = session_filter,
    n = length(exp_vals),
    p_value = p_value,
    effect_r = r,
    ci_low = unname(ci[1]),
    ci_high = unname(ci[2]),
    note = ""
  )
}


# --- Run tests (duration, only for the 3 directional behaviors) ---
direction_results <- list()
for (session_filter in c("BL6", "CD1")) {
  dur_r <- do.call(rbind, lapply(behaviors_directional, run_paired_direction_comparison,
                                 session_filter = session_filter, metric_col = "total_duration_sec"))
  # NO p.adjust here — explicitly exploratory, correction would be decoration
  direction_results[[session_filter]] <- list(dur = dur_r)
  write_csv(dur_r, file.path(output_dir, paste0("direction_split_duration_", session_filter, "_exploratory.csv")))
}


# --- Run tests (frequency, only for the 3 directional behaviors) ---
direction_results_freq <- list()
for (session_filter in c("BL6", "CD1")) {
  freq_r <- do.call(rbind, lapply(behaviors_directional, run_paired_direction_comparison,
                                  session_filter = session_filter, metric_col = "frequency"))
  direction_results_freq[[session_filter]] <- list(freq = freq_r)
  write_csv(freq_r, file.path(output_dir, paste0("direction_split_frequency_", session_filter, "_exploratory.csv")))
}

# --- Plot function for direction split (exploratory) ---
make_direction_pointplot <- function(behavior, metric_col, y_label, session_filter, stats_row) {
  plot_df <- agg_direction[agg_direction$behavior == behavior & agg_direction$session_type == session_filter, ]
  
  n_animals <- length(unique(plot_df$animal_id))
  x_labels <- setNames(
    paste0(c("exp", "stim"), "\n(n=", n_animals, ")"),
    c("exp", "stim")
  )
  
  # Median per group
  medians <- plot_df %>%
    group_by(initiated_by) %>%
    summarise(median_val = median(.data[[metric_col]], na.rm = TRUE), .groups = "drop")
  
  # Exploratory subtitle: effect size + CI, raw p in parentheses
  subtitle_txt <- if (nrow(stats_row) == 1 && !is.na(stats_row$effect_r)) {
    paste0(
      "Exploratory: r = ", round(stats_row$effect_r, 2),
      ", 95% CI [", round(stats_row$ci_low, 1), ", ", round(stats_row$ci_high, 1), "]",
      if (!is.na(stats_row$p_value)) paste0(" (raw p = ", signif(stats_row$p_value, 3), ")") else ""
    )
  } else {
    "no interpretable difference between directions in this sample"
  }
  
  display_name <- if (behavior %in% names(behavior_display_names)) {
    behavior_display_names[[behavior]]
  } else behavior
  title_txt <- paste0(display_name, " (duration), exp- vs. stim-initiated, ", session_filter, " sessions")
  
  p <- ggplot(plot_df, aes(x = initiated_by, y = .data[[metric_col]],
                           fill = initiated_by, colour = initiated_by)) +

    # Paired lines
    geom_line(aes(group = animal_id), color = "grey50", alpha = 0.6, linewidth = 0.8) +
    # Individual points with cohort shapes
    geom_point(aes(shape = cohort, fill = initiated_by),
               size = 3, stroke = 1, alpha = 0.9, position = position_nudge(x = 0)) +
    # Median as thick horizontal line
    scale_shape_manual(values = c(16, 17)) +
    geom_segment(data = medians,
                 aes(x = as.numeric(factor(initiated_by)) - 0.25,
                     xend = as.numeric(factor(initiated_by)) + 0.25,
                     y = median_val, yend = median_val,
                     color = initiated_by),
                 inherit.aes = FALSE, linewidth = 1.2) +
    scale_shape_manual(values = c(16, 17)) +
    scale_fill_manual(values = c(exp = pastel_pal[2], stim = pastel_pal[5])) +
    scale_colour_manual(values = c(exp = stroke_pal[2], stim = stroke_pal[5])) +
    scale_x_discrete(labels = x_labels) +
    scale_y_continuous(limits = c(0, NA), expand = expansion(mult = c(0, 0.1))) +
    labs(title = title_txt, subtitle = subtitle_txt,
         caption = paste0(
           "Exploratory: exp-initiated vs. stim-initiated bouts within the same session. ",
           "Minimum bout duration: ", get_min_duration(behavior), "s. ",
           "No multiple-testing correction applied (raw p shown for completeness)."
         ),
         x = "Initiated by", y = y_label) +
    theme_thesis +
    theme(legend.position = "none")
  
  return(p)
}

# --- Generate and save plots (duration + frequency) ---
for (session_filter in c("BL6", "CD1")) {
  for (b in behaviors_directional) {
    stats_dur_row <- direction_results[[session_filter]]$dur[
      direction_results[[session_filter]]$dur$behavior == b, ]
    p_dur <- make_direction_pointplot(b, "total_duration_sec", "Total duration (s)", 
                                      session_filter, stats_dur_row)
    ggsave(file.path(output_dir, paste0("direction_", b, "_", session_filter, "_exploratory.png")),
           p_dur, width = 5, height = 5, dpi = 300)
  }
}

for (session_filter in c("BL6", "CD1")) {
  for (b in behaviors_directional) {
    stats_freq_row <- direction_results_freq[[session_filter]]$freq[
      direction_results_freq[[session_filter]]$freq$behavior == b, ]
    p_freq <- make_direction_pointplot(b, "frequency", "Frequency (bout count)",
                                       session_filter, stats_freq_row)
    ggsave(file.path(output_dir, paste0("direction_", b, "_", session_filter, "_frequency_exploratory.png")),
           p_freq, width = 5, height = 5, dpi = 300)
  }
}

# --- Direction Split Mega-Grid (3 behaviors × 2 sessions = 6 plots) ---
direction_plots <- list()
for (session_filter in c("BL6", "CD1")) {
  for (b in behaviors_directional) {
    stats_dur_row <- direction_results[[session_filter]]$dur[
      direction_results[[session_filter]]$dur$behavior == b, ]
    p <- make_direction_pointplot(b, "total_duration_sec", "Total duration (s)", 
                                  session_filter, stats_dur_row) +
      theme(
        plot.title = element_text(size = 11, face = "bold", hjust = 0.5),
        plot.subtitle = element_text(size = 10, color = "grey30", hjust = 0.5),
        axis.title = element_text(size = 11),
        axis.text = element_text(size = 10),
        plot.caption = element_blank()
      )
    direction_plots <- c(direction_plots, list(p))
  }
}

mega_direction <- wrap_plots(direction_plots, ncol = 3, guides = "collect") +
  plot_annotation(
    title = paste0("Direction Split – exp vs. stim (exploratory) n = ", length(unique(agg_direction$animal_id))),
    subtitle = "Duration of directed behaviors (nose2tail, nose2body, following) by initiator.",
    caption = paste(
      "Exploratory: no multiple-testing correction (raw p shown).",
      "Minimum bout duration: 0.5s. BCa bootstrap CI – indicative."
    ),
    theme = theme(
      plot.title = element_text(hjust = 0.5, face = "bold", size = 20, color = "black"),
      plot.subtitle = element_text(hjust = 0.5, size = 13, color = "grey30"),
      plot.caption = element_text(hjust = 0.5, size = 12, color = "grey45", face = "italic", margin = margin(t = 10, b = 10))
    )
  ) &
  theme(legend.position = "bottom")

ggsave(file.path(output_dir, "direction_split_mega.png"),
       mega_direction, width = 15, height = 8, dpi = 300)

direction_plots_freq <- list()
for (session_filter in c("BL6", "CD1")) {
  for (b in behaviors_directional) {
    stats_freq_row <- direction_results_freq[[session_filter]]$freq[
      direction_results_freq[[session_filter]]$freq$behavior == b, ]
    p <- make_direction_pointplot(b, "frequency", "Frequency (bout count)",
                                  session_filter, stats_freq_row) +
      theme(
        plot.title = element_text(size = 11, face = "bold", hjust = 0.5),
        plot.subtitle = element_text(size = 10, color = "grey30", hjust = 0.5),
        axis.title = element_text(size = 11),
        axis.text = element_text(size = 10),
        plot.caption = element_blank()
      )
    direction_plots_freq <- c(direction_plots_freq, list(p))
  }
}

mega_direction_freq <- wrap_plots(direction_plots_freq, ncol = 3, guides = "collect") +
  plot_annotation(
    title = paste0("Direction Split – exp vs. stim (exploratory) n = ", length(unique(agg_direction$animal_id))),
    subtitle = "Frequency of directed behaviors (nose2tail, nose2body, following) by initiator.",
    caption = paste(
      "Exploratory: no multiple-testing correction (raw p shown).",
      "Minimum bout duration: 0.5s. BCa bootstrap CI – indicative."
    ),
    theme = theme(
      plot.title = element_text(hjust = 0.5, face = "bold", size = 20, color = "black"),
      plot.subtitle = element_text(hjust = 0.5, size = 13, color = "grey30"),
      plot.caption = element_text(hjust = 0.5, size = 12, color = "grey45", face = "italic", margin = margin(t = 10, b = 10))
    )
  ) &
  theme(legend.position = "bottom")

ggsave(file.path(output_dir, "direction_split_mega_frequency.png"),
       mega_direction_freq, width = 15, height = 8, dpi = 300)

message("Direction split mega-grid saved to: ", output_dir)

message("Direction-split (exploratory) completed. Results saved to: ", output_dir)

# =====================================================================
# 9. ETHOGRAM / TIME-COURSE (30s bins, aggregated + per-animal raster)
# =====================================================================
bin_width <- 30

session_lengths <- all_data %>%
  group_by(session_type) %>%
  summarise(len = max(end_sec), .groups = "drop")

make_bins <- function(session_type) {
  len <- session_lengths$len[session_lengths$session_type == session_type]
  seq(0, ceiling(len / bin_width) * bin_width, by = bin_width)
}

bin_events <- function(df) {
  df %>%
    rowwise() %>%
    group_map(~ {
      row <- .x
      bin_idx_start <- floor(row$start_sec / bin_width)
      bin_idx_end <- floor((row$end_sec - 1e-9) / bin_width)
      idxs <- bin_idx_start:bin_idx_end
      tibble(
        animal_id = row$animal_id, session_type = row$session_type,
        behavior = row$behavior,
        bin_start = idxs * bin_width,
        dur_in_bin = map_dbl(idxs, function(i) {
          bs <- i * bin_width; be <- bs + bin_width
          max(0, min(row$end_sec, be) - max(row$start_sec, bs))
        })
      )
    }) %>%
    bind_rows()
}

ethogram_binned <- all_data %>%
  group_split(session_type) %>%
  map_dfr(bin_events)

plot_ethogram_raster <- function(st) {
  d <- all_data %>% filter(session_type == st, behavior %in% all_behaviors)
  
  # cohort symbols 
  animal_labels <- d %>%
    distinct(animal_id) %>%
    mutate(cohort = ifelse(animal_id %in% cohort1_animals, "Coh1", "Coh2")) %>%
    mutate(label = case_when(
      cohort == "Coh1" ~ paste0("● ", animal_id),
      cohort == "Coh2" ~ paste0("▲ ", animal_id)
    )) %>%
    arrange(animal_id)
  
  # y-axis
  animal_order <- animal_labels$animal_id
  
  ggplot(d, aes(xmin = start_sec, xmax = end_sec,
                ymin = as.numeric(factor(animal_id, levels = animal_order)) - 0.4,
                ymax = as.numeric(factor(animal_id, levels = animal_order)) + 0.4,
                fill = behavior)) +
    geom_rect(color = NA) +
    scale_fill_manual(values = behavior_colors) +
    scale_y_continuous(
      breaks = seq_along(animal_order),
      labels = animal_labels$label
    ) +
    labs(title = paste("Per-animal ethogram —", st),
         x = "Time (s)", y = "Animal", fill = "Behavior") +
    theme_thesis
}


# =====================================================================
# ETHOGRAM WITHOUT "MOVING" (to see social behaviors better)
# =====================================================================
# Since "moving" dominates the ethogram, this version excludes it
# to make social interactions (nose2nose, nose2tail, etc.) visible.

# ---- Define behaviors excluding moving ----
behaviors_no_moving <- setdiff(all_behaviors, "moving")

# ---- Raster plot without moving ----
plot_ethogram_raster_no_moving <- function(st) {
  d <- all_data %>% 
    filter(session_type == st, behavior %in% behaviors_no_moving)
  
  animal_labels <- d %>%
    distinct(animal_id) %>%
    mutate(cohort = ifelse(animal_id %in% cohort1_animals, "Coh1", "Coh2")) %>%
    mutate(label = case_when(
      cohort == "Coh1" ~ paste0("● ", animal_id),
      cohort == "Coh2" ~ paste0("▲ ", animal_id)
    )) %>%
    arrange(animal_id)
  
  animal_order <- animal_labels$animal_id
  
  ggplot(d, aes(xmin = start_sec, xmax = end_sec,
                ymin = as.numeric(factor(animal_id, levels = animal_order)) - 0.4,
                ymax = as.numeric(factor(animal_id, levels = animal_order)) + 0.4,
                fill = behavior)) +
    geom_rect(color = NA) +
    scale_fill_manual(values = behavior_colors) +
    scale_y_continuous(
      breaks = seq_along(animal_order),
      labels = animal_labels$label
    ) +
    labs(title = paste("Per-animal ethogram —", st, "(without moving)"),
         x = "Time (s)", y = "Animal", fill = "Behavior") +
    theme_thesis
}

# =====================================================================
# TIME BUDGET PER ANIMAL (stacked bar) 
# =====================================================================
plot_time_budget_per_animal <- function(st, behaviors_to_use) {
  session_len <- session_lengths$len[session_lengths$session_type == st]
  
  d <- agg_total %>% filter(session_type == st, behavior %in% behaviors_to_use) %>%
    mutate(pct_time = 100 * total_duration_sec / session_len)
  
  animal_labels <- d %>%
    distinct(animal_id) %>%
    mutate(cohort = ifelse(animal_id %in% cohort1_animals, "Coh1", "Coh2")) %>%
    mutate(label = case_when(
      cohort == "Coh1" ~ paste0("● ", animal_id),
      cohort == "Coh2" ~ paste0("▲ ", animal_id)
    )) %>%
    arrange(animal_id)
  animal_order <- animal_labels$animal_id
  
  ggplot(d, aes(x = factor(animal_id, levels = animal_order), y = pct_time, fill = behavior)) +
    geom_col(width = 0.7, color = NA) +
    scale_fill_manual(values = behavior_colors) +
    scale_x_discrete(labels = setNames(animal_labels$label, animal_labels$animal_id)) +
    labs(x = "Animal", y = "Percentage of session time (%)", fill = "Behavior") +
    theme_thesis
}

# ---- Combine per-animal panel + raster panel, per session, per moving-status ----
make_combined_ethogram_fig <- function(st, behaviors_to_use, raster_fn) {
  n_anim <- length(unique(all_data$animal_id[all_data$session_type == st]))
  top <- plot_time_budget_per_animal(st, behaviors_to_use) +
    labs(title = paste("Time budget per animal —", st))
  bottom <- raster_fn(st)
  top / bottom +
    plot_layout(heights = c(1, 1.5))
}

# ---- Build the four session x moving-status combinations ----
fig_combined_bl6_moving    <- make_combined_ethogram_fig("BL6", all_behaviors, plot_ethogram_raster)
fig_combined_cd1_moving    <- make_combined_ethogram_fig("CD1", all_behaviors, plot_ethogram_raster)
fig_combined_bl6_no_moving <- make_combined_ethogram_fig("BL6", behaviors_no_moving, plot_ethogram_raster_no_moving)
fig_combined_cd1_no_moving <- make_combined_ethogram_fig("CD1", behaviors_no_moving, plot_ethogram_raster_no_moving)

# ---- Mega, WITH moving: BL6 over CD1 -> Appendix ----
mega_ethogram_with_moving <- wrap_plots(fig_combined_bl6_moving, fig_combined_cd1_moving, ncol = 1, guides = "collect") +
  plot_annotation(
    title = "Ethogram — time budget per animal and per-animal raster (including moving)",
    caption = paste0(
      "Top panel per session: total duration per behaviour, one bar per animal.\n",
      "Bottom panel per session: each animal's own raw bouts over time. ",
      "Min bout duration: 0.5s. Cohorts: ● = Coh1, ▲ = Coh2."
    ),
    theme = theme(
      plot.title = element_text(hjust = 0.5, face = "bold", size = 20),
      plot.caption = element_text(hjust = 0.5, size = 12, color = "grey45", face = "italic", margin = margin(t = 10, b = 10))
    )
  ) & theme(legend.position = "bottom", plot.margin = margin(t = 10, r = 25, b = 25, l = 25))

ggsave(file.path(output_dir, "ethogram_mega_with_moving.png"),
       mega_ethogram_with_moving, width = 12, height = 17, dpi = 300)

# ---- Mega, WITHOUT moving: BL6 over CD1 -> Results ----
mega_ethogram_no_moving <- wrap_plots(fig_combined_bl6_no_moving, fig_combined_cd1_no_moving, ncol = 1, guides = "collect") +
  plot_annotation(
    title = "Ethogram — time budget per animal and per-animal raster (excluding moving)",
    caption = paste0(
      "Top panel per session: total duration per behaviour, one bar per animal (moving excluded).\n",
      "Bottom panel per session: each animal's own raw bouts over time (moving excluded). ",
      "Min bout duration: 0.5s. Cohorts:● = Coh1, ▲ = Coh2."
    ),
    theme = theme(
      plot.title = element_text(hjust = 0.5, face = "bold", size = 20),
      plot.caption = element_text(hjust = 0.5, size = 12, color = "grey45", face = "italic", margin = margin(t = 10, b = 10))
    )
  ) & theme(legend.position = "bottom", plot.margin = margin(t = 10, r = 25, b = 25, l = 25))

ggsave(file.path(output_dir, "ethogram_mega_no_moving.png"),
       mega_ethogram_no_moving, width = 12, height = 17, dpi = 300)

message("Ethogram mega-grids (per-animal + raster, with/without moving) saved to: ", output_dir)

# =====================================================================
# 12. BOUT-DURATION DISTRIBUTION (bout-level, descriptive, PER ANIMAL)
# =====================================================================

plot_duration_dist_per_animal <- function(beh) {
  d <- all_data %>% filter(behavior == beh)
  
  d <- d %>%
    mutate(cohort = ifelse(animal_id %in% cohort1_animals, "Coh1", "Coh2"))
  
  n_bouts_total <- nrow(d)
  n_animals <- length(unique(d$animal_id))
  
  # Medians per animal and session
  medians <- d %>%
    group_by(animal_id, session_type) %>%
    summarise(median_dur = median(duration_sec, na.rm = TRUE), .groups = "drop")
  
  p <- ggplot(d, aes(x = session_type, y = duration_sec,
                     fill = session_type, color = session_type)) +
    
    geom_beeswarm(aes(shape = cohort), size = 1.5, alpha = 0.7, cex = 2) +
    # Median as thick horizontal line per animal
    geom_segment(data = medians,
                 aes(x = as.numeric(factor(session_type)) - 0.15,
                     xend = as.numeric(factor(session_type)) + 0.15,
                     y = median_dur, yend = median_dur,
                     color = session_type), 
                 inherit.aes = FALSE, linewidth = 0.8) +
    scale_shape_manual(values = c(Coh1 = 16, Coh2 = 17)) +
    scale_fill_manual(values = session_pal) +
    scale_colour_manual(values = session_stroke_pal) +
    facet_wrap(~ animal_id, ncol = 3, scales = "free_y") +
    scale_y_continuous(limits = c(0, NA), expand = expansion(mult = c(0, 0.05))) +
    labs(
      title = paste0("Bout duration distribution — ", beh),
      subtitle = paste("Per animal (descriptive only). N =", n_animals, "animals,", n_bouts_total, "total bouts."),
      x = "Session type", 
      y = "Bout duration (s)",
      caption = paste(
        "Each point = one bout. Descriptive only, no formal test.",
        "Minimum bout duration:", min_duration_default, "s."
      )
    ) +
    theme_thesis +
    theme(
      legend.position = "bottom",
      strip.text = element_text(face = "bold", size = 8)
    )
  
  return(p)
}

# --- Generate and save plots (one per behavior, with facets per animal) ---
for (beh in all_behaviors) {
  p <- plot_duration_dist_per_animal(beh)
  ggsave(file.path(output_dir, sprintf("duration_dist_%s.png", beh)),
         p, width = 8, height = 7, dpi = 300)
}

# --- Mega-Grids: split into two groups of 4 behaviors, each keeping its own animal-facets ---
behavior_groups <- split(all_behaviors, ceiling(seq_along(all_behaviors) / 4))

for (i in seq_along(behavior_groups)) {
  group_behaviors <- behavior_groups[[i]]
  group_plots <- lapply(group_behaviors, function(beh) {
    plot_duration_dist_per_animal(beh) +
      theme(
        plot.title = element_text(size = 11, face = "bold"),
        plot.subtitle = element_text(size = 8),
        strip.text = element_text(size = 7)
      )
  })
  
  mega_dur <- wrap_plots(group_plots, ncol = 2, guides = "collect") +
    plot_annotation(
      title = paste0("Bout Duration Distribution — Group ", i, " of ", length(behavior_groups)),
      caption = "Each point = one bout. Descriptive only, no formal test. Minimum bout duration: 0.5s.",
      theme = theme(
        plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
        plot.caption = element_text(hjust = 0.5, size = 9, color = "grey45", face = "italic")
      )
    ) &
    theme(legend.position = "bottom")
  
  ggsave(file.path(output_dir, sprintf("duration_dist_mega_group%d.png", i)),
         mega_dur, width = 20, height = 16, dpi = 300)
}

# ---- Behaviors included in this summary only ----
bout_summary_behaviors <- c("nose2tail", "nose2body", "following", 
                            "nose2nose", "sniffing", "sidebyside", 
                            "moving", "immobility")

# ---- Compact summary CSV: bout duration stats per animal x behavior x session ----
bout_duration_summary <- all_data %>%
  filter(behavior %in% bout_summary_behaviors) %>%
  group_by(animal_id, behavior, session_type) %>%
  summarise(
    n_bouts = n(),
    mean_dur = mean(duration_sec, na.rm = TRUE),
    median_dur = median(duration_sec, na.rm = TRUE),
    sd_dur = sd(duration_sec, na.rm = TRUE),
    min_dur = min(duration_sec, na.rm = TRUE),
    max_dur = max(duration_sec, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(cohort = ifelse(animal_id %in% cohort1_animals, "Coh1", "Coh2")) %>%
  arrange(behavior, session_type, animal_id)

write_csv(bout_duration_summary, file.path(output_dir, "bout_duration_summary.csv"))

message("Duration distribution mega-grids (2 groups of 4) saved to: ", output_dir)

message("Duration distribution (per animal) completed.")

# =====================================================================
# 13. FIRST-BOUT LATENCY (composite event: first social contact of any type)
# =====================================================================
# Definition: latency to the first bout of ANY social behavior
# (nose2nose, nose2tail, nose2body, sniffing, sidebyside, following).
# Moving and immobility are excluded.
# Censoring: animals that never show a social behavior in a session
# are assigned the session duration (conservative, stated in Methods).

# ---- 1. Define social behaviors ----
social_behaviors <- c("nose2nose", "nose2tail", "nose2body", "sidebyside", "following")

# ---- 2. Session duration (max end time per session) ----
session_max <- all_data %>%
  group_by(session_type) %>%
  summarise(session_duration_sec = max(end_sec, na.rm = TRUE), .groups = "drop")

# ---- 3. Compute composite latency per animal and session ----
latency_composite <- all_data %>%
  filter(behavior %in% social_behaviors) %>%
  group_by(animal_id, session_type) %>%
  slice_min(start_sec, n = 1, with_ties = FALSE) %>%
  ungroup() %>%
  select(animal_id, session_type, latency_sec = start_sec, first_behavior = behavior, first_actor = actor) %>%
  # Fill in animals that NEVER showed a social behavior (censored)
  complete(animal_id = all_animals, session_type = c("BL6", "CD1")) %>%
  left_join(session_max, by = "session_type") %>%
  mutate(
    latency_sec = ifelse(is.na(latency_sec), session_duration_sec, latency_sec),
    # Add cohort for plotting
    cohort = ifelse(animal_id %in% cohort1_animals, "Coh1", "Coh2")
  ) %>%
  arrange(animal_id, session_type)

# ---- 4. Single paired Wilcoxon test (no correction needed) ----
latency_wide <- latency_composite %>%
  select(animal_id, session_type, latency_sec) %>%
  pivot_wider(names_from = session_type, values_from = latency_sec)

# Remove any NA (shouldn't happen due to censoring)
latency_wide <- latency_wide[complete.cases(latency_wide), ]

# long format for coin::wilcoxsign_test
latency_long <- latency_composite %>%
  select(animal_id, session_type, latency_sec) %>%
  mutate(
    animal_id = factor(animal_id),                     # <-- NEU
    session_type = factor(session_type, levels = c("BL6", "CD1"))  # <-- NEU
  ) %>%
  arrange(animal_id, session_type)

wt <- coin::wilcoxsign_test(latency_sec ~ session_type | animal_id, 
                            data = latency_long, 
                            distribution = "exact")
p_value <- coin::pvalue(wt)
# rank-biserial r
r <- tryCatch({
  st <- coin::wilcoxsign_test(latency_wide$BL6 ~ latency_wide$CD1, distribution = "exact")
  as.numeric(coin::statistic(st, "standardized")) / sqrt(nrow(latency_wide))
}, error = function(e) NA_real_)

# BCa CI on the mean paired difference (BL6 - CD1)
diffs <- latency_wide$BL6 - latency_wide$CD1
set.seed(123)
ci <- bca_ci(diffs, R = 2000)

# Combine results
latency_summary <- tibble(
  n_pairs = nrow(latency_wide),
  p_value = p_value,
  effect_r = r,
  ci_low = unname(ci[1]),
  ci_high = unname(ci[2])
)
write_csv(latency_summary, file.path(output_dir, "latency_first_social_composite_stats.csv"))

# ---- 5. Plot: paired points (BL6 vs CD1) with cohort shapes ----
# ---- Median per group ----
medians_lat <- latency_composite %>%
  group_by(session_type) %>%
  summarise(median_val = median(latency_sec, na.rm = TRUE), .groups = "drop")

write_csv(latency_composite, file.path(output_dir, "latency_per_animal_session.csv"))
write_csv(medians_lat, file.path(output_dir, "latency_medians_by_session.csv"))

# ---- Plot ----
p_latency_composite <- ggplot(latency_composite, aes(x = session_type, y = latency_sec,
                                                     fill = session_type, color = session_type,
                                                     shape = cohort)) +
  geom_line(aes(group = animal_id), color = "grey50", alpha = 0.6, linewidth = 0.8) +
  geom_point(aes(fill = session_type), size = 3, stroke = 1, alpha = 0.9,
             position = position_nudge(x = 0)) +
  # Median as thick horizontal line
  geom_segment(data = medians_lat,
               aes(x = as.numeric(factor(session_type)) - 0.25,
                   xend = as.numeric(factor(session_type)) + 0.25,
                   y = median_val, yend = median_val,
                   color = session_type),
               inherit.aes = FALSE, linewidth = 1.2) +
  scale_shape_manual(values = c(16, 17)) +
  scale_fill_manual(values = session_pal) +
  scale_colour_manual(values = session_stroke_pal) +
  scale_x_discrete(labels = setNames(
    paste0(c("BL6", "CD1"), "\n(n=", length(unique(latency_composite$animal_id)), ")"),
    c("BL6", "CD1")
  )) +
  scale_y_continuous(limits = c(0, NA), expand = expansion(mult = c(0, 0.1))) +
  labs(
    title = "Latency to first social contact (composite)",
    subtitle = paste0(
      "Paired Wilcoxon: p = ", signif(p_value, 3),
      ", r = ", round(r, 2),
      "\n95% CI [", round(ci[1], 1), ", ", round(ci[2], 1), "]",
      " (BCa, n=", length(unique(latency_composite$animal_id)), ", indicative)"
    ),
    x = "Session type",
    y = "Latency (s)",
    caption = paste0(
      "Composite event: first bout of any social behavior.\n",
      "Animals with no social contact are censored at session duration (", 
      round(session_max$session_duration_sec[1], 0), "s).",
      " Cohorts: ● = Coh1, ▲ = Coh2."
    )
  )+
  theme_thesis +
  theme(legend.position = "none")

ggsave(file.path(output_dir, "latency_first_social_composite.png"),
       p_latency_composite, width = 8, height = 5, dpi = 300)

message("Composite latency analysis completed. Results saved to: ", output_dir)


# =====================================================================
# 14. ANIMAL x BEHAVIOR HEATMAP (frequency + duration, per session type,
#    z-normalized per behavior column; raw value shown in each cell)
# =====================================================================
plot_animal_heatmap <- function(st, metric) {
  d <- agg_total %>% filter(session_type == st) %>% select(animal_id, behavior, value = all_of(metric))
  d <- d %>% group_by(behavior) %>%
    mutate(z = if (sd(value) == 0) 0 else as.numeric(scale(value))) %>%
    ungroup()
  label_fmt <- if (metric == "frequency") "%d" else "%.1f"
  
  metric_label <- if (metric == "frequency") "Frequency" else "Duration"
  
  heatmap_data <- bind_rows(
    agg_total %>% filter(session_type == "BL6") %>% select(animal_id, behavior, frequency, total_duration_sec) %>% mutate(session_type = "BL6"),
    agg_total %>% filter(session_type == "CD1") %>% select(animal_id, behavior, frequency, total_duration_sec) %>% mutate(session_type = "CD1")
  ) %>%
    group_by(session_type, behavior) %>%
    mutate(
      z_frequency = if (sd(frequency) == 0) 0 else as.numeric(scale(frequency)),
      z_duration  = if (sd(total_duration_sec) == 0) 0 else as.numeric(scale(total_duration_sec))
    ) %>%
    ungroup()
  write_csv(heatmap_data, file.path(output_dir, "animal_behavior_heatmap_data.csv"))
  
  # cohort-symbols for y-axis-labels
  animal_labels <- d %>%
    distinct(animal_id) %>%
    mutate(cohort = ifelse(animal_id %in% cohort1_animals, "Coh1", "Coh2")) %>%
    mutate(label = case_when(
      cohort == "Coh1" ~ paste0("● ", animal_id),
      cohort == "Coh2" ~ paste0("▲ ", animal_id)
    )) %>%
    arrange(animal_id)
  
  ggplot(d, aes(x = behavior, y = factor(animal_id, levels = animal_labels$animal_id), fill = z)) +
    geom_tile(color = "white") +
    geom_text(aes(label = sprintf(label_fmt, value)), size = 3) +
    scale_fill_gradient2(
      low = "#5A93BB",       
      mid = "#FBF8FD",       
      high = "#D97CA0",      
      midpoint = 0,
      limits = c(-max(abs(d$z)), max(abs(d$z))),
      guide = guide_colorbar(title = "z-score")) +
    scale_y_discrete(labels = setNames(animal_labels$label, animal_labels$animal_id)) +
    labs(
      title = sprintf("%s by animal — %s", metric_label, st),
      subtitle = paste(
        "Color = z-score across animals (per behavior); numbers = raw values.",
        "Note: z-scoring of n=", length(unique(d$animal_id)), 
        " animals makes the colour scale sensitive to individual animals."
      ),
      x = "Behavior", 
      y = "Animal", 
      fill = "z-score",
      caption = "Every animal x behavior cell shown individually."
    ) +
    theme_thesis +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}

ggsave(file.path(output_dir, "heatmap_freq_BL6.png"), plot_animal_heatmap("BL6", "frequency"), width = 7, height = 5, dpi = 300)
ggsave(file.path(output_dir, "heatmap_freq_CD1.png"), plot_animal_heatmap("CD1", "frequency"), width = 7, height = 5, dpi = 300)
ggsave(file.path(output_dir, "heatmap_dur_BL6.png"),  plot_animal_heatmap("BL6", "total_duration_sec"),  width = 7, height = 5, dpi = 300)
ggsave(file.path(output_dir, "heatmap_dur_CD1.png"),  plot_animal_heatmap("CD1", "total_duration_sec"),  width = 7, height = 5, dpi = 300)


message("Done with behavioural analysis. Plots and tables saved to: ", output_dir)




# =====================================================================
# MEGA PLOTS GENERATION 
# =====================================================================

# =====================================================================
# ANIMAL × BEHAVIOR HEATMAP MEGA-GRID (single caption at bottom)
# =====================================================================

heatmap_dur_bl6  <- plot_animal_heatmap("BL6", "total_duration_sec") + 
  labs(title = "Duration – BL6", caption = NULL, subtitle = NULL)
heatmap_dur_cd1  <- plot_animal_heatmap("CD1", "total_duration_sec") + 
  labs(title = "Duration – CD1", caption = NULL, subtitle = NULL)
heatmap_freq_bl6 <- plot_animal_heatmap("BL6", "frequency") + 
  labs(title = "Frequency – BL6", caption = NULL, subtitle = NULL)
heatmap_freq_cd1 <- plot_animal_heatmap("CD1", "frequency") + 
  labs(title = "Frequency – CD1", caption = NULL, subtitle = NULL)

row_dur <- wrap_plots(heatmap_dur_bl6, heatmap_dur_cd1, ncol = 2) +
  plot_annotation(theme = theme(legend.position = "bottom"))
row_freq <- wrap_plots(heatmap_freq_bl6, heatmap_freq_cd1, ncol = 2) +
  plot_annotation(theme = theme(legend.position = "bottom"))

mega_heatmaps <- (row_dur / row_freq) +
  plot_annotation(
    title = "Animal × Behavior heatmaps – z‑scored frequency / duration",
    subtitle = paste(
      "Color = z-score across animals (per behavior); numbers = raw values.",
      "Note: z-scoring of n=", length(unique(agg_total$animal_id)), 
      " animals makes the colour scale sensitive to individual animals."
    ),
    caption = paste(
      "Every animal × behavior cell shown individually. BL6 and CD1 sessions.",
      "Values: raw numbers on tiles, color = z-score.",
      "Cohorts: ● = Coh1, ▲ = Coh2."
    ),
    theme = theme(
      plot.title = element_text(hjust = 0.5, face = "bold", size = 20),
      plot.subtitle = element_text(hjust = 0.5, size = 13, color = "grey30"),
      plot.caption = element_text(hjust = 0.5, size = 12, color = "grey45", face = "italic", margin = margin(t = 10, b = 10))
    )
  ) &
  theme(legend.position = "bottom")

ggsave(file.path(output_dir, "heatmap_mega.png"),
       mega_heatmaps, width = 14, height = 10, dpi = 300)

# =====================================================================
# CORRELATION WITH HISTOLOGY (descriptive scatterplots, no p-values)
# =====================================================================

# ---- 1. Load histology data ----
histology_data <- read_csv(file.path(path_cellcount_dir, "animal_level_data.csv"))

histology_summary <- histology_data %>%
  mutate(animal_id = str_replace(animal_id, "^count(\\d+)$", "count_\\1")) %>%
  select(animal_id, 
         gfp_density = manual_GFP_density,
         cfos_density = manual_cFos_density,
         colocalized_pct = manual_coloc_pct,
         overlap_index)

# Only animals that have histology data (count_0 and count_2 excluded)
animals_with_histology <- histology_summary$animal_id

# ---- 2. Define behaviors to correlate (same as main test) ----
corr_behaviors <- c("nose2tail", "nose2nose", "nose2body", "sidebyside", "following")

hist_session_map <- tibble(
  histology_measure = c("gfp_density", "cfos_density", "colocalized_pct", "overlap_index"),
  session_type = list("BL6", "CD1", c("BL6", "CD1"), c("BL6", "CD1"))
)

# ---- 3. Build correlation data for BOTH sessions, using view_lookup (exp for directed, total for non-directed) ----
view_lookup <- tibble(
  behavior = corr_behaviors,
  view = ifelse(behavior %in% behaviors_directional, "exp", "total")
)

corr_data_list <- list()
for (b in corr_behaviors) {
  view <- view_lookup$view[view_lookup$behavior == b]
  agg_df <- if (view == "total") agg_total else agg_exp
  
  df <- agg_df %>%
    filter(behavior == b, session_type %in% c("BL6", "CD1")) %>%
    select(animal_id, session_type, total_duration_sec) %>%
    mutate(behavior = b) %>%
    filter(animal_id %in% animals_with_histology)
  
  corr_data_list[[b]] <- df
}

corr_data_long <- bind_rows(corr_data_list)

# Wide format: one column per behavior x session, e.g. nose2tail_BL6, nose2tail_CD1
corr_data <- corr_data_long %>%
  mutate(col_name = paste0(behavior, "_", session_type)) %>%
  select(animal_id, col_name, total_duration_sec) %>%
  pivot_wider(names_from = col_name, values_from = total_duration_sec) %>%
  left_join(histology_summary, by = "animal_id")

# ---- Which session each histology measure is correlated against ----
hist_session_map <- tibble(
  histology_measure = c("gfp_density", "cfos_density", "colocalized_pct", "overlap_index"),
  session_type = list("BL6", "CD1", c("BL6", "CD1"), c("BL6", "CD1"))
)

# ---- Export correlation data as CSV ----
write_csv(corr_data, file.path(output_dir, "correlation_data_for_scatterplots.csv"))

# ---- 4. Function to make a single scatterplot ----
make_corr_scatter <- function(data, x_var, y_var, x_label, y_label, title) {
  plot_data <- data %>%
    select(animal_id, x = all_of(x_var), y = all_of(y_var)) %>%
    filter(!is.na(x), !is.na(y))
  
  if (nrow(plot_data) < 3) {
    return(NULL)
  }
  
  rho <- cor(plot_data$x, plot_data$y, method = "spearman", use = "complete.obs")
  
  plot_data <- plot_data %>%
    mutate(cohort = ifelse(animal_id %in% cohort1_animals, "Coh1", "Coh2"))
  
  p <- ggplot(plot_data, aes(x = x, y = y)) +
    geom_smooth(aes(group = 1), method = "lm", se = FALSE, color = "grey50", linewidth = 0.6, linetype = "dashed") +
    geom_point(aes(fill = cohort, color = cohort, shape = cohort),
               size = 3.5, alpha = 0.9, stroke = 1.2) +
    geom_text_repel(aes(label = animal_id),
                    size = 3, color = "grey30", 
                    box.padding = 0.5, point.padding = 0.3,
                    show.legend = FALSE) +
    scale_fill_manual(values = cohort_pal) +
    scale_color_manual(values = cohort_stroke_pal) +
    scale_shape_manual(values = c(Coh1 = 16, Coh2 = 17)) +
    labs(
      title = title,
      subtitle = paste0("Spearman rho = ", round(rho, 2), " (descriptive, n = ", nrow(plot_data), ")"),
      x = x_label,
      y = y_label,
      caption = "Descriptive only – no p-values. Inspect individual animals."
    ) +
    theme_thesis +
    theme(legend.position = "bottom")
  
  return(p)
}
# ---- Export correlation data as CSV ----
write_csv(corr_data, file.path(output_dir, "correlation_data_for_scatterplots.csv"))

# ---- 5. Generate all scatterplots, using the correct session per histology measure ----
hist_measures <- c("gfp_density", "cfos_density", "colocalized_pct", "overlap_index")
hist_labels <- c(
  gfp_density = "GFP density (cells/mm²)",
  cfos_density = "cFos density (cells/mm²)",
  colocalized_pct = "Colocalization (%)",
  overlap_index = "Overlap Index"
)

behavior_labels <- c(
  nose2tail = "Nose-to-tail (duration, s)",
  nose2nose = "Nose-to-nose (duration, s)",
  nose2body = "Nose-to-body (duration, s)",
  sidebyside = "Side-by-side (duration, s)",
  following = "Following (duration, s)",
  sniffing = "Sniffing (duration, s)"
)

# Build the full list of (behavior, session, histology_measure) combinations to plot
corr_plot_specs <- list()
for (hm in hist_measures) {
  sessions_for_hm <- hist_session_map$session_type[hist_session_map$histology_measure == hm][[1]]
  for (s in sessions_for_hm) {
    for (beh in corr_behaviors) {
      corr_plot_specs[[length(corr_plot_specs) + 1]] <- list(
        behavior = beh, session = s, hist_measure = hm,
        x_var = paste0(beh, "_", s)
      )
    }
  }
}

for (spec in corr_plot_specs) {
  p <- make_corr_scatter(
    data = corr_data,
    x_var = spec$x_var,
    y_var = spec$hist_measure,
    x_label = paste0(behavior_labels[spec$behavior], " (", spec$session, ")"),
    y_label = hist_labels[spec$hist_measure],
    title = paste0(behavior_labels[spec$behavior], " (", spec$session, ") vs. ", hist_labels[spec$hist_measure])
  )
  if (!is.null(p)) {
    fname <- paste0("corr_scatter_", spec$behavior, "_", spec$session, "_vs_", spec$hist_measure, ".png")
    ggsave(file.path(output_dir, fname), p, width = 5, height = 5, dpi = 300)
  }
}

# ---- 6. Mega-Grids for all correlation scatterplots, split by session ----
build_corr_plots <- function(specs) {
  plots <- list()
  i <- 1
  for (spec in specs) {
    p <- make_corr_scatter(
      data = corr_data,
      x_var = spec$x_var,
      y_var = spec$hist_measure,
      x_label = paste0(behavior_labels[spec$behavior], " (", spec$session, ")"),
      y_label = hist_labels[spec$hist_measure],
      title = paste0(str_remove(behavior_labels[spec$behavior], " \\(duration, s\\)"), " vs. ", hist_labels[spec$hist_measure])
    )
    if (!is.null(p)) {
      p <- p + labs(caption = NULL) +
        theme(
          plot.title = element_text(size = 11, face = "bold", hjust = 0.5),
          plot.subtitle = element_text(size = 10, color = "grey30", hjust = 0.5),
          axis.title = element_text(size = 11),
          axis.text = element_text(size = 10),
          legend.position = "none"
        )
      plots[[i]] <- p
      i <- i + 1
    }
  }
  plots
}

specs_bl6 <- Filter(function(s) s$session == "BL6", corr_plot_specs)
specs_cd1 <- Filter(function(s) s$session == "CD1", corr_plot_specs)

corr_plots_bl6 <- build_corr_plots(specs_bl6)
corr_plots_cd1 <- build_corr_plots(specs_cd1)

grDevices::pdf(file.path(output_dir, "tmp_null_device.pdf"), width = 14, height = 20)

mega_corr_bl6 <- wrap_plots(corr_plots_bl6, ncol = 3, guides = "collect") +
  plot_annotation(
    title = "Behaviour vs. Histology, BL6 session – descriptive scatterplots. n = 7 animals.",
    subtitle = "Spearman rho shown (descriptive). Each point = one animal.",
    caption = paste0(
      "Exp-only for directional behaviors (nose2tail, nose2body, following); total for non-directional (nose2nose, sidebyside, sniffing).\n",
      "GFP density is BL6-specific (tagged during BL6 interaction); Colocalization % and Overlap Index shown here for BL6."
    ),
    theme = theme(
      plot.title = element_text(hjust = 0.5, face = "bold", size = 20, color = "black"),
      plot.subtitle = element_text(hjust = 0.5, size = 13, color = "grey30"),
      plot.caption = element_text(hjust = 0.5, size = 12, color = "grey45", face = "italic", margin = margin(t = 10, b = 10))
    )
  ) &
  theme(legend.position = "bottom")

mega_corr_cd1 <- wrap_plots(corr_plots_cd1, ncol = 3, guides = "collect") +
  plot_annotation(
    title = "Behaviour vs. Histology, CD1 session – descriptive scatterplots. n = 7 animals.",
    subtitle = "Spearman rho shown (descriptive). Each point = one animal.",
    caption = paste0(
      "Exp-only for directional behaviors (nose2tail, nose2body, following); total for non-directional (nose2nose, sidebyside, sniffing).\n",
      "cFos density is CD1-specific (tagged during CD1 interaction); Colocalization % and Overlap Index shown here for CD1."
    ),
    theme = theme(
      plot.title = element_text(hjust = 0.5, face = "bold", size = 20, color = "black"),
      plot.subtitle = element_text(hjust = 0.5, size = 13, color = "grey30"),
      plot.caption = element_text(hjust = 0.5, size = 12, color = "grey45", face = "italic", margin = margin(t = 10, b = 10))
    )
  ) &
  theme(legend.position = "bottom")

dev.off()
file.remove(file.path(output_dir, "tmp_null_device.pdf"))

ggsave(file.path(output_dir, "correlation_scatter_mega_BL6.png"),
       mega_corr_bl6, width = 14, height = 3.2 * ceiling(length(corr_plots_bl6) / 3), dpi = 300)
ggsave(file.path(output_dir, "correlation_scatter_mega_CD1.png"),
       mega_corr_cd1, width = 14, height = 3.2 * ceiling(length(corr_plots_cd1) / 3), dpi = 300)

message("Correlation mega-grids (BL6, CD1) saved to: ", output_dir)

# =====================================================================
# CSV EXPORTS FOR ALL ANALYSES
# =====================================================================

# ---- 1. BL6 vs. CD1 duration stats (already saved) ----
# results_dur is already saved as bl6_vs_cd1_duration_8behaviors_holm.csv

# ---- 2. Direction split stats (already saved) ----
# direction_split_duration_BL6_exploratory.csv and CD1_exploratory.csv

# ---- 3. Composite latency stats (already saved) ----
# latency_first_social_composite_stats.csv

# ---- 4. Correlation rho values ----
correlation_rho_table <- tibble()
for (spec in corr_plot_specs) {
  plot_data <- corr_data %>%
    select(animal_id, x = all_of(spec$x_var), y = all_of(spec$hist_measure)) %>%
    filter(!is.na(x), !is.na(y))
  
  if (nrow(plot_data) >= 3) {
    rho <- cor(plot_data$x, plot_data$y, method = "spearman", use = "complete.obs")
    n_animals <- nrow(plot_data)
    correlation_rho_table <- bind_rows(
      correlation_rho_table,
      tibble(
        behavior = spec$behavior,
        session_type = spec$session,
        histology_measure = spec$hist_measure,
        spearman_rho = rho,
        n_animals = n_animals
      )
    )
  }
}

write_csv(correlation_rho_table, file.path(output_dir, "correlation_spearman_rho_table.csv"))

# ---- 5. Ethogram data (NEW) ----
# Export the binned ethogram data as CSV
write_csv(ethogram_binned, file.path(output_dir, "ethogram_binned_data.csv"))

# ---- 6. Aggregated behavior data per animal (NEW) ----
write_csv(agg_total, file.path(output_dir, "behavior_total_per_animal.csv"))
write_csv(agg_exp, file.path(output_dir, "behavior_exp_only_per_animal.csv"))
write_csv(agg_stim, file.path(output_dir, "behavior_stim_only_per_animal.csv"))
write_csv(agg_direction, file.path(output_dir, "behavior_direction_split_per_animal.csv"))

# ---- 7. Direction split stats summary (NEW) ----
direction_summary <- bind_rows(
  direction_results$BL6$dur %>% mutate(session = "BL6", metric = "duration"),
  direction_results$CD1$dur %>% mutate(session = "CD1", metric = "duration"),
  direction_results_freq$BL6$freq %>% mutate(session = "BL6", metric = "frequency"),
  direction_results_freq$CD1$freq %>% mutate(session = "CD1", metric = "frequency")
)
write_csv(direction_summary, file.path(output_dir, "direction_split_summary.csv"))

message("All CSVs exported to: ", output_dir)



# =====================================================================
# DONE
# =====================================================================
message("All plots successfully saved to: ", output_dir)

# ---- Stop console logging ----
sink()
message("Console output saved to: ", log_file)