## ============================================================================
## IPACL Engram Analysis
## BL6 vGATE (GFP) x cFos study — 7 animals with count data (count0 and count2
## excluded: no viable count data), 2 cohorts (Coh1, Coh2)
## ============================================================================

## ---- 0. Packages -----------------------------------------------------------
required_pkgs <- c("tidyverse", "xml2", "jsonlite","patchwork", "rstatix", "coin", "boot", "ggrepel")

new_pkgs <- required_pkgs[!(required_pkgs %in% installed.packages()[, "Package"])]
if (length(new_pkgs) > 0) install.packages(new_pkgs)

library(tidyverse)
library(xml2)
library(jsonlite)
library(patchwork)
library(rstatix)
library(coin)
library(boot)
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



## ---- Check if running interactively or being sourced for HTML -----------
## If this script is being sourced from ipacl_interactive_html.R, skip
## saving PNGs and CSVs to avoid overwriting existing files.
SKIP_SAVING <- exists("INTERACTIVE_MODE") && INTERACTIVE_MODE

## ---- 1. CONFIG: fill in your paths here -----------------------------------
## Manual XML files: one folder per cohort, with subfolders per animal inside
path_manual_coh1 <- "/Users/Path/to/Folder"
path_manual_coh2 <- "/Users/Path/to/Folder"

## GeoJSON ROI files: one folder per cohort (flat, no animal subfolders)
path_geojson_coh1 <- "/Users/Path/to/Folder"
path_geojson_coh2 <- "/Users/Path/to/Folder"

## Automatic (QuPath) colocalisation CSVs, one per cohort
path_auto_coh1 <- "/Users/Path/to/CSV"
path_auto_coh2 <- "/Users/Path/to/CSV"

## Where plots get saved
output_dir <- "/Users/Path/to/Folder"
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

## Pastel palette
pastel_pal <- c("#FFCBE1", "#D6E5BD", "#F9E1A8", "#BCD8EC", "#DCCCEC", "#FFDAB4")
## Matching darker stroke/outline for each pastel fill (same hue family, not black/grey) -
## keeps clouds/boxplots readable in print/grayscale and coordinated rather than clashing
stroke_pal <- c("#D97CA0", "#8FA86E", "#C9A24B", "#5A93BB", "#A98FC7", "#D99A5C")
cohort_pal <- c(Coh1 = pastel_pal[4], Coh2 = pastel_pal[1])   # blue vs pink
cohort_stroke_pal <- c(Coh1 = stroke_pal[4], Coh2 = stroke_pal[1])
channel_pal <- c(GFP = pastel_pal[2], cFos = pastel_pal[1], Colocalized = pastel_pal[4])
channel_stroke_pal <- c(GFP = stroke_pal[2], cFos = stroke_pal[1], Colocalized = stroke_pal[4])

## Known animal -> cohort assignment
coh1_animals <- c("count0", "count2", "count3", "count8", "count10")
coh2_animals <- c("count4", "count5", "count6", "count9")

## Images used to train the automatic Ilastik/QuPath models - separate lists
## per model, since GFP and cFos were trained independently. Used later to
## check whether manual-automatic agreement is inflated on images the
## pipeline was actually trained on (overfitting check).
training_images_gfp  <- c("count3_1", "count8_2", "count8_4", "count10_4")
training_images_cfos <- c("count3_1", "count3_4", "count8_2", "count10_2")

## ============================================================================
## 2. HELPER FUNCTIONS
## ============================================================================
#' Lin's Concordance Correlation Coefficient with BCa bootstrap CI
compute_ccc <- function(x, y, n_boot = 2000, conf_level = 0.95) {
  keep <- complete.cases(x, y)
  x <- x[keep]; y <- y[keep]
  n <- length(x)
  
  if (n < 3) return(list(ccc = NA_real_, ci_low = NA_real_, ci_high = NA_real_, n = n))
  
  # Manual CCC
  mean_x <- mean(x); mean_y <- mean(y)
  var_x <- var(x); var_y <- var(y)
  cov_xy <- cov(x, y)
  ccc <- (2 * cov_xy) / (var_x + var_y + (mean_x - mean_y)^2)
  
  # BCa Bootstrap CI
  boot_ccc_fn <- function(data, indices) {
    d <- data[indices, ]
    mx <- mean(d[,1]); my <- mean(d[,2])
    vx <- var(d[,1]); vy <- var(d[,2])
    cxy <- cov(d[,1], d[,2])
    (2 * cxy) / (vx + vy + (mx - my)^2)
  }
  data_mat <- cbind(x, y)
  boot_obj <- tryCatch(
    boot::boot(data_mat, boot_ccc_fn, R = n_boot),
    error = function(e) NULL
  )
  if (is.null(boot_obj)) {
    ci <- c(NA_real_, NA_real_)
  } else {
    ci <- tryCatch(
      boot::boot.ci(boot_obj, type = "bca", conf = conf_level)$bca[4:5],
      error = function(e) c(NA_real_, NA_real_)
    )
  }
  
  list(ccc = ccc, ci_low = unname(ci[1]), ci_high = unname(ci[2]), n = n)
}

#' Confidence intervals for Bland-Altman limits of agreement
compute_loa_ci <- function(manual, auto, conf_level = 0.95) {
  diff <- auto - manual
  mean_diff <- mean(diff, na.rm = TRUE)
  sd_diff <- sd(diff, na.rm = TRUE)
  n <- sum(!is.na(diff))
  
  loa_lower <- mean_diff - 1.96 * sd_diff
  loa_upper <- mean_diff + 1.96 * sd_diff
  
  z <- qnorm(1 - (1 - conf_level)/2)
  se_loa <- sqrt(3 * sd_diff^2 / n)
  
  ci_loa_lower <- c(loa_lower - z * se_loa, loa_lower + z * se_loa)
  ci_loa_upper <- c(loa_upper - z * se_loa, loa_upper + z * se_loa)
  
  list(
    bias = mean_diff,
    bias_ci = c(mean_diff - z * sd_diff/sqrt(n), mean_diff + z * sd_diff/sqrt(n)),
    loa_lower = loa_lower,
    loa_upper = loa_upper,
    ci_loa_lower = ci_loa_lower,
    ci_loa_upper = ci_loa_upper,
    n = n
  )
}

#' Extract the "countX_Y" image ID from any filename - case-insensitive,
#' always normalized to lowercase (some files are "Count4_1", others "count9_2";
#' without normalizing, those would be treated as different/unmatched IDs and
#' silently become NA, which is what was causing count4/5/6 to disappear)
get_image_id <- function(filename) {
  str_to_lower(str_extract(filename, regex("count\\d+_\\d+", ignore_case = TRUE)))
}

#' Get animal ID ("countX") from an image ID ("countX_Y") - image_id is already
#' lowercase by this point (from get_image_id), so no case issue here
get_animal_id <- function(image_id) {
  str_extract(image_id, "^count\\d+")
}

#' Get section number (Y) from an image ID ("countX_Y")
get_section_num <- function(image_id) {
  as.integer(str_extract(image_id, "(?<=_)\\d+$"))
}

#' Parse one Fiji Cell Counter XML file
#' Returns a 1-row tibble: image_id, calibration_um_per_px, GFP, cFOS, both
parse_fiji_xml <- function(xml_path) {
  image_id <- get_image_id(basename(xml_path))
  doc <- tryCatch(read_xml(xml_path), error = function(e) NULL)
  if (is.null(doc)) {
    warning(paste("Could not read XML:", xml_path))
    return(NULL)
  }
  
  calib <- as.numeric(xml_text(xml_find_first(doc, ".//X_Calibration")))
  
  marker_types <- xml_find_all(doc, ".//Marker_Type")
  if (length(marker_types) == 0) {
    warning(paste("No Marker_Type nodes found in:", xml_path))
    return(NULL)
  }
  
  # ---- Koordinaten extrahieren ----
  all_markers <- map_dfr(marker_types, function(mt) {
    name <- xml_text(xml_find_first(mt, "./Name"))
    markers <- xml_find_all(mt, "./Marker")
    coords <- map_dfr(markers, function(m) {
      tibble(
        x = as.numeric(xml_text(xml_find_first(m, "./MarkerX"))),
        y = as.numeric(xml_text(xml_find_first(m, "./MarkerY")))
      )
    })
    coords$marker_type <- name
    coords
  })
  

  
  # Standardize marker types (same as before)
  all_markers <- all_markers %>%
    mutate(marker_type_clean = case_when(
      str_to_lower(marker_type) %in% c("gfp", "gfp+", "type 1") ~ "GFP",
      str_to_lower(marker_type) %in% c("cfos", "cfos+", "type 2") ~ "cFOS",
      str_to_lower(marker_type) == "both" | str_to_lower(marker_type) == "type 3" ~ "both",
      TRUE ~ NA_character_
    )) %>%
    filter(!is.na(marker_type_clean))
  
  # Counts (for backward compatibility)
  counts <- all_markers %>%
    group_by(marker_type_clean) %>%
    summarise(n_cells = n(), .groups = "drop") %>%
    pivot_wider(names_from = marker_type_clean, values_from = n_cells, values_fill = 0)
  
  # Ensure all three columns exist
  for (needed in c("GFP", "cFOS", "both")) {
    if (!(needed %in% names(counts))) counts[[needed]] <- 0
  }
  
  # Return: counts + coordinates as nested list
  tibble(
    image_id = image_id,
    calibration_um_per_px = calib,
    manual_GFP_only = counts$GFP[1],
    manual_cFOS_only = counts$cFOS[1],
    manual_both = counts$both[1],
    coords = list(all_markers %>% select(x, y, marker_type_clean))
  )
}

#' Shoelace formula for polygon area (in px^2), given a list of [x,y] coordinate pairs
shoelace_area <- function(coords) {
  x <- coords[, 1]
  y <- coords[, 2]
  n <- length(x)
  # standard shoelace formula; works for closed or unclosed rings
  area <- abs(sum(x * c(y[-1], y[1]) - c(x[-1], x[1]) * y)) / 2
  area
}

#' Parse one GeoJSON ROI file -> area in um^2
#' px_size_um: um per pixel (read per-image from the matching XML for safety)
parse_geojson_area <- function(geojson_path, px_size_um) {
  image_id <- get_image_id(basename(geojson_path))
  gj <- tryCatch(fromJSON(geojson_path, simplifyVector = FALSE),
                 error = function(e) NULL)
  if (is.null(gj)) {
    warning(paste("Could not read GeoJSON:", geojson_path))
    return(NULL)
  }

  features <- gj$features
  total_area_px2 <- 0

  for (feat in features) {
    geom <- feat$geometry
    if (is.null(geom)) next
    if (geom$type == "Polygon") {
      # exterior ring only = first ring
      ring <- geom$coordinates[[1]]
      coords <- do.call(rbind, lapply(ring, function(pt) c(pt[[1]], pt[[2]])))
      total_area_px2 <- total_area_px2 + shoelace_area(coords)
    } else if (geom$type == "MultiPolygon") {
      for (poly in geom$coordinates) {
        ring <- poly[[1]]
        coords <- do.call(rbind, lapply(ring, function(pt) c(pt[[1]], pt[[2]])))
        total_area_px2 <- total_area_px2 + shoelace_area(coords)
      }
    }
  }

  tibble(
    image_id = image_id,
    roi_area_um2 = total_area_px2 * (px_size_um^2)
  )
}

#' Read all Fiji XML files recursively under a cohort folder
read_all_manual <- function(folder) {
  empty <- tibble(image_id = character(), calibration_um_per_px = double(),
                   manual_GFP_only = integer(), manual_cFOS_only = integer(), manual_both = integer())
  if (!dir.exists(folder)) {
    warning(paste("Folder does not exist (check path/typo):", folder))
    return(empty)
  }
  files <- list.files(folder, pattern = "\\.xml$", recursive = TRUE, full.names = TRUE)
  if (length(files) == 0) {
    warning(paste("No XML files found under:", folder))
    return(empty)
  }
  out <- map_dfr(files, function(f) {
    res <- parse_fiji_xml(f)
    if (is.null(res)) return(NULL)
    res
  })
  if (nrow(out) == 0) empty else out
}

#' Rotate coordinates around centroid
rotate_coords <- function(x, y, angle_deg) {
  angle_rad <- angle_deg * pi / 180
  cx <- mean(x); cy <- mean(y)
  x_rot <- cx + (x - cx) * cos(angle_rad) - (y - cy) * sin(angle_rad)
  y_rot <- cy + (x - cx) * sin(angle_rad) + (y - cy) * cos(angle_rad)
  list(x = x_rot, y = y_rot)
}

#' Count overlapping cells within a radius (robust to NAs)
count_overlap <- function(x1, y1, x2, y2, radius_um) {
  # Remove any NA coordinates
  valid1 <- complete.cases(x1, y1)
  valid2 <- complete.cases(x2, y2)
  x1 <- x1[valid1]; y1 <- y1[valid1]
  x2 <- x2[valid2]; y2 <- y2[valid2]
  
  if (length(x1) == 0 || length(x2) == 0) return(0)
  
  n_overlap <- 0
  for (i in seq_along(x1)) {
    dists <- sqrt((x1[i] - x2)^2 + (y1[i] - y2)^2)
    if (any(dists <= radius_um, na.rm = TRUE)) n_overlap <- n_overlap + 1
  }
  n_overlap
}

#' Rotation null test for one animal (robust to missing coordinates)
rotation_null_test <- function(animal_id, coords_df, n_rotations = 1000, radius_um = 5) {
  # Extract coordinates
  gfp_x <- unlist(coords_df[coords_df$animal_id == animal_id, "x_GFP"])
  gfp_y <- unlist(coords_df[coords_df$animal_id == animal_id, "y_GFP"])
  cfos_x <- unlist(coords_df[coords_df$animal_id == animal_id, "x_cFOS"])
  cfos_y <- unlist(coords_df[coords_df$animal_id == animal_id, "y_cFOS"])
  
  # Remove NAs and check if enough data
  gfp_x <- gfp_x[!is.na(gfp_x)]
  gfp_y <- gfp_y[!is.na(gfp_y)]
  cfos_x <- cfos_x[!is.na(cfos_x)]
  cfos_y <- cfos_y[!is.na(cfos_y)]
  
  if (length(gfp_x) < 2 || length(cfos_x) < 2) return(NA_real_)
  
  real_overlap <- count_overlap(gfp_x, gfp_y, cfos_x, cfos_y, radius_um)
  
  set.seed(123 + which(unique(coords_df$animal_id) == animal_id))
  rot_overlaps <- replicate(n_rotations, {
    angle <- runif(1, 0, 360)
    rot <- rotate_coords(gfp_x, gfp_y, angle)
    count_overlap(rot$x, rot$y, cfos_x, cfos_y, radius_um)
  })
  
  p_ge <- (sum(rot_overlaps >= real_overlap, na.rm = TRUE) + 1) / (n_rotations + 1)
  p_le <- (sum(rot_overlaps <= real_overlap, na.rm = TRUE) + 1) / (n_rotations + 1)
  p_val <- min(2 * min(p_ge, p_le), 1)
  return(p_val)
}
#' Read all GeoJSON files under a cohort folder, using per-file calibration
#' looked up from the manual_data (matched by image_id)
read_all_geojson <- function(folder, manual_data) {
  empty <- tibble(image_id = character(), roi_area_um2 = double())
  if (!dir.exists(folder)) {
    warning(paste("Folder does not exist (check path/typo):", folder))
    return(empty)
  }
  files <- list.files(folder, pattern = "\\.geojson$", recursive = TRUE, full.names = TRUE)
  if (length(files) == 0) {
    warning(paste("No GeoJSON files found under:", folder))
    return(empty)
  }
  out <- map_dfr(files, function(f) {
    image_id <- get_image_id(basename(f))
    px_size <- manual_data$calibration_um_per_px[manual_data$image_id == image_id]
    if (length(px_size) == 0) {
      warning(paste("No matching calibration for", image_id, "- skipping area calc, using NA"))
      px_size <- NA_real_
    } else {
      px_size <- px_size[1]
    }
    res <- parse_geojson_area(f, px_size)
    if (is.null(res)) return(NULL)
    res
  })
  if (nrow(out) == 0) empty else out
}

#' Read one cohort's automatic QuPath CSV and standardize column names
read_auto_csv <- function(path, cohort_label) {
  if (!file.exists(path)) {
    stop(paste0(cohort_label, ": automatic CSV not found at path (check for typos): ", path))
  }
  df <- read_csv(path, show_col_types = FALSE)
  df %>%
    mutate(image_id = get_image_id(Image)) %>%  # same case-insensitive normalization as manual/geojson
    mutate(
      cohort = cohort_label,
      auto_GFP_only = GFP_only,
      auto_cFOS_only = cFOS_only,
      auto_both = Double_positive
    ) %>%
    select(image_id, cohort, auto_GFP_only, auto_cFOS_only, auto_both)
}

## ============================================================================
## 3. BUILD THE FULL IMAGE-LEVEL DATASET
## ============================================================================

build_cohort_data <- function(manual_dir, geojson_dir, auto_csv, cohort_label) {
  manual <- read_all_manual(manual_dir)
  geo <- read_all_geojson(geojson_dir, manual)
  auto <- read_auto_csv(auto_csv, cohort_label)

  # CRITICAL: drop any row whose image_id could not be parsed (NA) BEFORE
  # joining. Leaving NAs in causes full_join to match every NA row against
  # every other NA row (a many-to-many cartesian blowup) - this was silently
  # inflating counts by duplicating rows many times over. A file that produces
  # NA here has a name that doesn't match "countX_Y" (typo, system file, etc.)
  # and gets reported below instead of silently corrupting every sum.
  n_bad_manual <- sum(is.na(manual$image_id))
  n_bad_geo <- sum(is.na(geo$image_id))
  n_bad_auto <- sum(is.na(auto$image_id))
  if (n_bad_manual > 0) warning(paste0(cohort_label, ": ", n_bad_manual, " manual file(s) had an unparseable name (image_id=NA) and were dropped - check for typos or stray files."))
  if (n_bad_geo > 0) warning(paste0(cohort_label, ": ", n_bad_geo, " GeoJSON file(s) had an unparseable name (image_id=NA) and were dropped."))
  if (n_bad_auto > 0) warning(paste0(cohort_label, ": ", n_bad_auto, " automatic CSV row(s) had an unparseable Image value (NA) and were dropped."))
  manual <- manual %>% filter(!is.na(image_id))
  geo <- geo %>% filter(!is.na(image_id))
  auto <- auto %>% filter(!is.na(image_id))

  # Also guard against genuine duplicate image_ids within one source (e.g. the
  # same file appearing twice because of a nested backup/copy subfolder) -
  # same many-to-many risk, so warn and keep only the first occurrence.
  dedupe_warn <- function(df, label) {
    dup_ids <- df$image_id[duplicated(df$image_id)]
    if (length(dup_ids) > 0) {
      warning(paste0(cohort_label, " ", label, ": duplicate image_id(s) found (kept first occurrence only): ",
                      paste(unique(dup_ids), collapse = ", ")))
      df <- df %>% distinct(image_id, .keep_all = TRUE)
    }
    df
  }
  manual <- dedupe_warn(manual, "manual")
  geo <- dedupe_warn(geo, "geojson")
  auto <- dedupe_warn(auto, "automatic")

  # Report any mismatches so nothing silently gets dropped
  missing_geo <- setdiff(manual$image_id, geo$image_id)
  missing_manual <- setdiff(geo$image_id, manual$image_id)
  missing_auto <- setdiff(manual$image_id, auto$image_id)
  if (length(missing_geo) > 0)
    message(cohort_label, ": images with manual data but no matching GeoJSON: ",
            paste(missing_geo, collapse = ", "))
  if (length(missing_manual) > 0)
    message(cohort_label, ": GeoJSON files with no matching manual XML: ",
            paste(missing_manual, collapse = ", "))
  if (length(missing_auto) > 0)
    message(cohort_label, ": images with manual data but no automatic count: ",
            paste(missing_auto, collapse = ", "))

  manual %>%
    full_join(geo, by = "image_id") %>%
    full_join(auto, by = "image_id") %>%
    mutate(
      cohort = cohort_label,
      animal_id = get_animal_id(image_id),
      section_num = get_section_num(image_id)
    )
}

data_coh1 <- build_cohort_data(path_manual_coh1, path_geojson_coh1, path_auto_coh1, "Coh1")
data_coh2 <- build_cohort_data(path_manual_coh2, path_geojson_coh2, path_auto_coh2, "Coh2")

image_data <- bind_rows(data_coh1, data_coh2) %>%
  mutate(cohort = factor(cohort, levels = c("Coh1", "Coh2")))


## ---- Sanity check: does every animal fall in the cohort we expect? --------
animal_check <- image_data %>%
  distinct(animal_id, cohort) %>%
  mutate(expected_cohort = case_when(
    animal_id %in% coh1_animals ~ "Coh1",
    animal_id %in% coh2_animals ~ "Coh2",
    TRUE ~ "UNKNOWN"
  ))
mismatches <- animal_check %>% filter(as.character(cohort) != expected_cohort)
if (nrow(mismatches) > 0) {
  message("WARNING - animal/cohort mismatches found:")
  print(mismatches)
}

## ---- Check for suspiciously duplicated manual counts ----------------------
## Different image_ids (different files) that happen to have EXACTLY the same
## GFP-only / cFos-only / both counts are worth a manual look - could be a
## genuine coincidence, but could also be a copy-paste mistake when the XML
## files were created (e.g. one file's marker data accidentally reused for
## another image). This does NOT catch filename duplicates (that's the
## separate dedupe_warn check above) - only duplicate CONTENT under different names.
suspicious_duplicate_counts <- image_data %>%
  filter(!is.na(manual_GFP_only)) %>%
  group_by(manual_GFP_only, manual_cFOS_only, manual_both) %>%
  filter(n() > 1) %>%
  ungroup() %>%
  arrange(manual_GFP_only, manual_cFOS_only, manual_both) %>%
  select(image_id, cohort, manual_GFP_only, manual_cFOS_only, manual_both)

if (nrow(suspicious_duplicate_counts) > 0) {
  message(nrow(suspicious_duplicate_counts), " image(s) share IDENTICAL manual GFP/cFos/both counts with at least one other image:")
  print(suspicious_duplicate_counts)
  message("-> Not necessarily an error, but worth a manual double-check of these specific XML files.")
} else {
  message("No suspiciously duplicated manual counts found across images.")
}
write_csv(suspicious_duplicate_counts, file.path(output_dir, "suspicious_duplicate_counts.csv"))

## ============================================================================
## 4. DERIVED VARIABLES (image level)
## ============================================================================

image_data <- image_data %>%
  mutate(
    # Manual totals (only + both, matching how QuPath reports totals)
    manual_GFP_total  = manual_GFP_only + manual_both,
    manual_cFos_total = manual_cFOS_only + manual_both,
    manual_N          = manual_GFP_only + manual_cFOS_only + manual_both,

    # Automatic totals
    auto_GFP_total  = auto_GFP_only + auto_both,
    auto_cFos_total = auto_cFOS_only + auto_both,
    auto_N          = auto_GFP_only + auto_cFOS_only + auto_both,

    # Chance overlap (sum-based denominator, since true total neuron count is unavailable)
    manual_chance_overlap = (manual_GFP_total / manual_N) * (manual_cFos_total / manual_N) * manual_N,
    auto_chance_overlap   = (auto_GFP_total / auto_N) * (auto_cFos_total / auto_N) * auto_N,

    # Overlap as % of the GFP or cFos pool (no area normalization needed - it's a ratio)
    manual_overlap_pct_GFP  = 100 * manual_both / manual_GFP_total,
    manual_overlap_pct_cFos = 100 * manual_both / manual_cFos_total,
    auto_overlap_pct_GFP    = 100 * auto_both / auto_GFP_total,
    auto_overlap_pct_cFos   = 100 * auto_both / auto_cFos_total,

    # Colocalization % (both as a share of ALL labeled cells, N) - the "both" analog
    # of overlap_pct_GFP/cFos; this is also how the automatic QuPath output defines it
    manual_coloc_pct = 100 * manual_both / manual_N,
    auto_coloc_pct   = 100 * auto_both   / auto_N,

    # Densities (cells per mm^2) - for checking injection-size / batch effects.
    # Computed for all three counted populations (GFP, cFos, both), not just GFP/cFos.
    manual_GFP_density  = manual_GFP_total  / (roi_area_um2 / 1e6),
    manual_cFos_density = manual_cFos_total / (roi_area_um2 / 1e6),
    manual_both_density = manual_both       / (roi_area_um2 / 1e6),
    auto_GFP_density    = auto_GFP_total    / (roi_area_um2 / 1e6),
    auto_cFos_density   = auto_cFos_total   / (roi_area_um2 / 1e6),
    auto_both_density   = auto_both         / (roi_area_um2 / 1e6),

    # Training-image flags + manual-automatic delta, for the overfitting check
    is_training_gfp  = image_id %in% training_images_gfp,
    is_training_cfos = image_id %in% training_images_cfos,
    is_training_either = is_training_gfp | is_training_cfos,
    delta_GFP  = auto_GFP_total  - manual_GFP_total,
    delta_cFos = auto_cFos_total - manual_cFos_total
  )

## ---- Relative rostrocaudal position within each animal --------------------
## Uses the section's RANK within its animal's own sequence (1st, 2nd, 3rd...
## image from anterior to posterior). Split into
## thirds (anterior/middle/posterior) per animal so animals with different
## numbers of sections are still comparable.
image_data <- image_data %>%
  arrange(animal_id, section_num) %>%
  group_by(animal_id) %>%
  mutate(rc_bin = if (n() >= 3 && !anyNA(section_num)) ntile(section_num, 3) else NA_integer_) %>%
  ungroup() %>%
  mutate(rc_bin = factor(rc_bin, levels = 1:3, labels = c("Anterior", "Middle", "Posterior")))

## ============================================================================
## 5. AGGREGATE PER ANIMAL (sum counts first, then compute ratios -
##    avoids "ratio of ratios" bias vs. averaging per-image percentages)
## ============================================================================
animal_data <- image_data %>%
  group_by(animal_id, cohort) %>%
  summarise(
    n_images = n(),
    across(c(manual_GFP_only, manual_cFOS_only, manual_both,
             auto_GFP_only, auto_cFOS_only, auto_both,
             roi_area_um2),
           ~ sum(.x, na.rm = TRUE))
  ) %>%
  mutate(
    manual_GFP_total  = manual_GFP_only + manual_both,
    manual_cFos_total = manual_cFOS_only + manual_both,
    manual_N          = manual_GFP_only + manual_cFOS_only + manual_both,
    auto_GFP_total  = auto_GFP_only + auto_both,
    auto_cFos_total = auto_cFOS_only + auto_both,
    auto_N          = auto_GFP_only + auto_cFOS_only + auto_both,
    manual_chance_overlap = (manual_GFP_total / manual_N) * (manual_cFos_total / manual_N) * manual_N,
    auto_chance_overlap   = (auto_GFP_total / auto_N) * (auto_cFos_total / auto_N) * auto_N,
    manual_overlap_pct_GFP  = 100 * manual_both / manual_GFP_total,
    manual_overlap_pct_cFos = 100 * manual_both / manual_cFos_total,
    auto_overlap_pct_GFP    = 100 * auto_both / auto_GFP_total,
    auto_overlap_pct_cFos   = 100 * auto_both / auto_cFos_total,
    manual_coloc_pct = 100 * manual_both / manual_N,
    auto_coloc_pct   = 100 * auto_both   / auto_N,
    manual_GFP_density  = manual_GFP_total  / (roi_area_um2 / 1e6),
    manual_cFos_density = manual_cFos_total / (roi_area_um2 / 1e6),
    manual_both_density = manual_both       / (roi_area_um2 / 1e6),
    manual_GFP_density  = manual_GFP_total  / (roi_area_um2 / 1e6),
    manual_cFos_density = manual_cFos_total / (roi_area_um2 / 1e6),
    manual_both_density = manual_both       / (roi_area_um2 / 1e6),
    auto_GFP_density    = auto_GFP_total    / (roi_area_um2 / 1e6),
    auto_cFos_density   = auto_cFos_total   / (roi_area_um2 / 1e6),
    auto_both_density   = auto_both         / (roi_area_um2 / 1e6)
     ) %>%
  # ---- Order animals within each cohort by numeric ID ----
mutate(animal_num = as.integer(str_extract(animal_id, "\\d+"))) %>%
  arrange(cohort, animal_num) %>%
  group_by(cohort) %>%
  mutate(cohort_order = row_number()) %>%
  ungroup()

# ---- Mirror correction for rotation null test (based on base_side) ----
mirror_info <- read_csv(file.path(output_dir, "image_mirror_info.csv"))

image_data <- image_data %>%
  left_join(mirror_info, by = "image_id") %>%
  mutate(coords = map2(coords, base_side, function(df, side) {
    if (is.na(side) || side != 1) return(df)
    ref_x <- mean(df$x, na.rm = TRUE)
    df$x <- 2 * ref_x - df$x
    df
  }))

# ---- Aggregate coordinates per animal (separate pipeline) ----
coords_per_animal <- image_data %>%
  select(animal_id, coords) %>%
  unnest(coords) %>%
  group_by(animal_id, marker_type_clean) %>%
  summarise(
    x = list(x),
    y = list(y),
    .groups = "drop"
  ) %>%
  pivot_wider(names_from = marker_type_clean, values_from = c(x, y), names_sep = "_")

test_coords <- coords_per_animal %>%
  mutate(
    n_gfp = map_int(x_GFP, length),
    n_cfos = map_int(x_cFOS, length),
    mean_x_gfp = map_dbl(x_GFP, mean),
    mean_x_cfos = map_dbl(x_cFOS, mean)
  ) %>%
  select(animal_id, n_gfp, n_cfos, mean_x_gfp, mean_x_cfos)

print(test_coords)

message("Animal-level dataset: ", nrow(animal_data), " animals (expect 7 = 9 total minus count0 and count2, which have no count data)")

## ============================================================================
## 6. STATISTICAL TESTS
## ----------------------------------------------------------------------------

## Shared theme 
theme_thesis <- theme_minimal(base_size = 15) +
  theme(
    panel.grid.minor = element_blank(),
    plot.title = element_text(face = "bold", size = 15, hjust = 0.5),
    plot.subtitle = element_text(size = 9.5, color = "grey30", hjust = 0.5),
    plot.caption = element_text(size = 11, color = "grey45", hjust = 0.5, face = "italic",
                                 lineheight = 1.3, margin = margin(t = 8)),
    plot.margin = margin(t = 25, r = 40, b = 10, l = 40),
    strip.text = element_text(size = 9, face = "bold"),
    legend.position = "bottom",
    legend.text = element_text(color = "grey30", size = 11),
    legend.title = element_blank()
  )

#' Robust y-limits: zoom the view to Q1-1.5*IQR .. Q3+1.5*IQR (standard Tukey
#' fence) so one extreme value can't squash the rest of the data into a
#' sliver at one end. Returns NULL (no zoom) if nothing actually falls outside.
compute_robust_ylim <- function(x) {
  x <- x[is.finite(x)]
  q <- quantile(x, c(0.25, 0.75), na.rm = TRUE)
  iqr <- q[2] - q[1]
  if (iqr == 0) return(NULL)
  fence <- c(q[1] - 1.5 * iqr, q[2] + 1.5 * iqr)  # standard Tukey fence
  full_range <- range(x, na.rm = TRUE)
  lower <- max(full_range[1], fence[1])
  upper <- min(full_range[2], fence[2])
  if (lower <= full_range[1] && upper >= full_range[2]) return(NULL)  # nothing clipped
  c(lower, upper)
}

#' Warn (don't silently fail) if any Inf/NaN values show up in ratio columns -
#' these arise from division by zero (e.g. an image/animal with 0 GFP cells)
check_finite <- function(df, cols, label) {
  bad <- purrr::map_lgl(cols, ~ any(!is.finite(df[[.x]]), na.rm = TRUE))
  if (any(bad)) {
    warning(paste0(label, ": non-finite values (Inf/NaN, likely division by zero) found in: ",
                   paste(cols[bad], collapse = ", "),
                   " - check images/animals with zero GFP or cFos counts."))
  }
}

#' Conventional significance stars alongside the exact p-value (never instead
#' of it - the raw p is always shown too, per the "show it even if not
#' significant" principle already agreed on for this thesis)
sig_stars <- function(p) {
  dplyr::case_when(
    is.na(p) ~ "",
    p < 0.001 ~ "***",
    p < 0.01  ~ "**",
    p < 0.05  ~ "*",
    p < 0.1   ~ ".",
    TRUE ~ ""
  )
}

## ---- 6a. Normality check (QQ-plots, not a blind Shapiro-Wilk gate - n=8/9 has low power) --
check_finite(image_data, c("manual_overlap_pct_GFP", "manual_overlap_pct_cFos", "manual_coloc_pct",
                            "auto_overlap_pct_GFP", "auto_overlap_pct_cFos", "auto_coloc_pct"),
             "image_data")

qq_gfp_cfos_diff <- ggplot(animal_data, aes(sample = manual_GFP_total - manual_cFos_total)) +
  stat_qq(size = 2) + stat_qq_line() +
  labs(title = "QQ-plot: GFP - cFos", subtitle = "difference, manual, per animal") +
  theme_thesis

qq_real_chance_diff <- ggplot(animal_data, aes(sample = manual_both - manual_chance_overlap)) +
  stat_qq(size = 2) + stat_qq_line() +
  labs(title = "QQ-plot: Real - Chance overlap", subtitle = "difference, manual, per animal") +
  theme_thesis


## ---- 6b. Real vs. chance overlap per animal (paired) ----------------------
# Define animal IDs if not already done
animal_ids <- unique(animal_data$animal_id)

test_real_vs_chance_t  <- t.test(animal_data$manual_both, animal_data$manual_chance_overlap, paired = TRUE)
# long-format data for coin::wilcoxsign_test
real_chance_long <- animal_data %>%
  select(animal_id, manual_both, manual_chance_overlap) %>%
  pivot_longer(cols = c(manual_both, manual_chance_overlap),
               names_to = "type", values_to = "n_cells") %>%
  complete(animal_id, type, fill = list(n_cells = 0)) %>%
  mutate(
    type = factor(type, levels = c("manual_both", "manual_chance_overlap")),
    animal_id = factor(animal_id) 
  ) %>%
  arrange(animal_id, type)

test_real_vs_chance_wc <- coin::wilcoxsign_test(n_cells ~ type | animal_id,
                                                data = real_chance_long,
                                                distribution = "exact")
test_real_vs_chance_wc_p <- coin::pvalue(test_real_vs_chance_wc)
# extract Z-value
z_real <- as.numeric(coin::statistic(test_real_vs_chance_wc, type = "standardized"))
# total number of observations (2 measurements x n animals)
n_pairs_real <- length(unique(real_chance_long$animal_id))
# effect size r
r_real <- abs(z_real) / sqrt(n_pairs_real)

# for the statistics table
test_real_vs_chance_wc_stat <- z_real
eff_real_vs_chance <- rstatix::wilcox_effsize(real_chance_long, n_cells ~ type, paired = TRUE)

real_chance_long %>% count(animal_id) %>% pull(n) %>% unique()
# should only return `2`, then every animal is complete

# ---- QC: GFP vs cFos (paired Wilcoxon) ----
# long-format data for coin::wilcoxsign_test
gfp_vs_cfos_long <- animal_data %>%
  select(animal_id, manual_GFP_total, manual_cFos_total) %>%
  pivot_longer(cols = c(manual_GFP_total, manual_cFos_total),
               names_to = "channel", values_to = "n_cells") %>%
  mutate(channel = factor(channel, levels = c("manual_GFP_total", "manual_cFos_total"),
                          labels = c("GFP", "cFos"))) %>%
  complete(animal_id, channel, fill = list(n_cells = 0)) %>%
  mutate(
    animal_id = factor(animal_id)
  ) %>%
  arrange(animal_id, channel)

gfp_vs_cfos_test <- coin::wilcoxsign_test(n_cells ~ channel | animal_id,
                                          data = gfp_vs_cfos_long,
                                          distribution = "exact")
gfp_vs_cfos_p <- coin::pvalue(gfp_vs_cfos_test)
z_gfp <- as.numeric(coin::statistic(gfp_vs_cfos_test, type = "standardized"))
n_pairs_gfp <- length(unique(gfp_vs_cfos_long$animal_id))
r_gfp <- abs(z_gfp) / sqrt(n_pairs_gfp)

gfp_vs_cfos_stat <- z_gfp

real_chance_long %>% count(animal_id) %>% pull(n) %>% unique()
# should only return '2', then every animal is complete

# ---- Chance overlap descriptive stats ----
chance_mean <- mean(animal_data$manual_chance_overlap, na.rm = TRUE)
chance_median <- median(animal_data$manual_chance_overlap, na.rm = TRUE)
chance_iqr <- IQR(animal_data$manual_chance_overlap, na.rm = TRUE)

# ---- Colocalized descriptive stats ----
coloc_mean <- mean(animal_data$manual_both, na.rm = TRUE)
coloc_median <- median(animal_data$manual_both, na.rm = TRUE)
coloc_iqr <- IQR(animal_data$manual_both, na.rm = TRUE)

gfp_mean <- mean(animal_data$manual_GFP_total, na.rm = TRUE)
gfp_median <- median(animal_data$manual_GFP_total, na.rm = TRUE)
gfp_iqr <- IQR(animal_data$manual_GFP_total, na.rm = TRUE)
cfos_mean <- mean(animal_data$manual_cFos_total, na.rm = TRUE)
cfos_median <- median(animal_data$manual_cFos_total, na.rm = TRUE)
cfos_iqr <- IQR(animal_data$manual_cFos_total, na.rm = TRUE)
# ---- Colocalized density descriptive stats (for Fig2 subtitle) ----
coloc_density_mean <- mean(animal_data$manual_both_density, na.rm = TRUE)
coloc_density_median <- median(animal_data$manual_both_density, na.rm = TRUE)
coloc_density_iqr <- IQR(animal_data$manual_both_density, na.rm = TRUE)

## Overlap Index (0-1): how close the real overlap is to its theoretical maximum
animal_data <- animal_data %>%
  mutate(
    overlap_index = (manual_both - manual_chance_overlap) /
      (pmin(manual_GFP_total, manual_cFos_total) - manual_chance_overlap)
  )

check_finite(animal_data, c("manual_overlap_pct_GFP", "manual_overlap_pct_cFos",
                             "manual_coloc_pct", "overlap_index"), "animal_data")


## Bootstrap CI on the real-vs-chance difference
boot_diff <- animal_data$manual_both - animal_data$manual_chance_overlap
boot_diff <- boot_diff[is.finite(boot_diff)]
if (length(boot_diff) >= 3) {
  set.seed(1)
  boot_ci <- bca_ci(boot_diff, R = 2000)
  names(boot_ci) <- c("2.5%", "97.5%")
} else {
  warning("Not enough finite animals to bootstrap a confidence interval – boot_ci set to NA.")
  boot_ci <- c(`2.5%` = NA_real_, `97.5%` = NA_real_)
}

# ---- Rotation null: primary (5 µm) and sensitivity (10 µm) ----

# Primary: conservative radius (5 µm, matches nuclear size)
rotation_p_5 <- map_dbl(animal_ids, function(id) {
  rotation_null_test(id, coords_per_animal, radius_um = 5)
})
names(rotation_p_5) <- animal_ids

# Sensitivity: permissive radius (10 µm, matches QuPath pipeline)
rotation_p_10 <- map_dbl(animal_ids, function(id) {
  rotation_null_test(id, coords_per_animal, radius_um = 10)
})
names(rotation_p_10) <- animal_ids

# Add to animal_data
animal_data <- animal_data %>%
  left_join(tibble(animal_id = names(rotation_p_5), rotation_null_p_5 = rotation_p_5), by = "animal_id") %>%
  left_join(tibble(animal_id = names(rotation_p_10), rotation_null_p_10 = rotation_p_10), by = "animal_id")

# Summarize results for subtitle
n_total <- sum(!is.na(animal_data$rotation_null_p_5))
n_sig_5 <- sum(animal_data$rotation_null_p_5 < 0.05, na.rm = TRUE)
n_sig_10 <- sum(animal_data$rotation_null_p_10 < 0.05, na.rm = TRUE)

rotation_summary <- paste0(
  "Rotation null: ", n_sig_5, "/", n_total, " animals p<0.05 (5µm)",
  if (n_total > 0) paste0(" | ", n_sig_10, "/", n_total, " p<0.05 (10µm)") else ""
)

# ---- Rotation null test: Visualisation for each animal ----
# For each animal, plot the distribution of overlap from 1000 rotations
# with the real overlap marked as a purple line.

# This function assumes coords_per_animal is already defined
plot_rotation_null <- function(animal_id, coords_df, radius_um = 5, n_rotations = 1000) {
  # Extract coordinates for this animal
  gfp_x <- unlist(coords_df[coords_df$animal_id == animal_id, "x_GFP"])
  gfp_y <- unlist(coords_df[coords_df$animal_id == animal_id, "y_GFP"])
  cfos_x <- unlist(coords_df[coords_df$animal_id == animal_id, "x_cFOS"])
  cfos_y <- unlist(coords_df[coords_df$animal_id == animal_id, "y_cFOS"])
  
  valid_gfp <- complete.cases(gfp_x, gfp_y)
  valid_cfos <- complete.cases(cfos_x, cfos_y)
  gfp_x <- gfp_x[valid_gfp]; gfp_y <- gfp_y[valid_gfp]
  cfos_x <- cfos_x[valid_cfos]; cfos_y <- cfos_y[valid_cfos]
  
  if (length(gfp_x) < 2 || length(cfos_x) < 2) return(NULL)
  
  real_overlap <- count_overlap(gfp_x, gfp_y, cfos_x, cfos_y, radius_um)
  
  set.seed(123 + which(unique(coords_df$animal_id) == animal_id))
  rot_overlaps <- replicate(n_rotations, {
    angle <- runif(1, 0, 360)
    rot <- rotate_coords(gfp_x, gfp_y, angle)
    count_overlap(rot$x, rot$y, cfos_x, cfos_y, radius_um)
  })
  
  p_ge <- (sum(rot_overlaps >= real_overlap, na.rm = TRUE) + 1) / (n_rotations + 1)
  p_le <- (sum(rot_overlaps <= real_overlap, na.rm = TRUE) + 1) / (n_rotations + 1)
  p_val <- min(2 * min(p_ge, p_le), 1)
  df <- data.frame(overlap = rot_overlaps)
  cohort <- ifelse(animal_id %in% coh1_animals, "Coh1", "Coh2")
  
  dens <- density(df$overlap, na.rm = TRUE)
  max_density <- max(dens$y, na.rm = TRUE)
  
  p <- ggplot(df, aes(x = overlap)) +
    geom_density(fill = pastel_pal[2], color = stroke_pal[2], alpha = 0.6, bw = "nrd0") +
    geom_rug(alpha = 0.3, sides = "b", color = stroke_pal[2]) +
    geom_vline(xintercept = real_overlap, color = stroke_pal[5], linewidth = 1.2) +
    labs(
      title = paste0(
        "Rotation null test – ",
        ifelse(cohort == "Coh1", "● ", "▲ "),
        animal_id,
        " (", radius_um, "µm)"
      ),
      subtitle = paste0(
        "Cohort: ", cohort, " | Real overlap = ", real_overlap, " | p = ", signif(p_val, 3)
      ),
      x = "Overlap count",
      y = "Density",
      caption = NULL
    ) +
    theme_thesis +
    coord_cartesian(clip = "off") +
    theme(
      legend.position = "none",
      plot.title = element_text(hjust = 0.5, face = "bold", size = 11),
      plot.subtitle = element_text(hjust = 0.5, size = 12, color = "grey30"),
      plot.margin = margin(t = 12, r = 25, b = 10, l = 10)
    )
  return(p)
}

# ---- Generate rotation null plots for all animals (5µm and 10µm) ----
rotation_plots_5 <- list()
rotation_plots_10 <- list()

for (id in animal_ids) {
  p5 <- plot_rotation_null(id, coords_per_animal, radius_um = 5)
  if (!is.null(p5)) rotation_plots_5[[id]] <- p5
  
  p10 <- plot_rotation_null(id, coords_per_animal, radius_um = 10)
  if (!is.null(p10)) rotation_plots_10[[id]] <- p10
}

# ---- Mega-Grid: rotation null plots (5µm) ----
if (length(rotation_plots_5) > 0) {
  mega_rotation_5 <- wrap_plots(rotation_plots_5, ncol = 3, guides = "collect") +
    plot_annotation(
      title = "Rotation null test (5µm) – each animal individually",
      subtitle = "Purple line = real overlap | Green density = null distribution (1000 rotations)",
      caption = paste(
        "p < 0.05 indicates real overlap differs from the  rotation-based null.", "Cohorts: ● = Coh1, ▲ = Coh2."
      ),
      theme = theme(
        plot.title = element_text(hjust = 0.5, face = "bold", size = 20),
        plot.subtitle = element_text(hjust = 0.5, size = 13, color = "grey30"),
        plot.caption = element_text(hjust = 0.5, size = 12, color = "grey45", face = "italic", margin = margin(t = 10, b = 10))
      )
    ) &
    theme(legend.position = "bottom")
  
  ggsave(file.path(output_dir, "rotation_null_mega_5um.png"),
         mega_rotation_5, width = 12, height = 8, dpi = 300)
}

# ---- Mega-Grid: rotation null plots (10µm) ----
if (length(rotation_plots_10) > 0) {
  mega_rotation_10 <- wrap_plots(rotation_plots_10, ncol = 3, guides = "collect") +
    plot_annotation(
      title = "Rotation null test (10µm) – each animal individually",
      subtitle = "Purple line = real overlap | Green density = null distribution (1000 rotations)",
      caption = paste(
        "p < 0.05 indicates real overlap differs from the  rotation-based null. Cohorts: ● = Coh1, ▲ = Coh2."
      ),
      theme = theme(
        plot.title = element_text(hjust = 0.5, face = "bold", size = 20),
        plot.subtitle = element_text(hjust = 0.5, size = 13, color = "grey30"),
        plot.caption = element_text(hjust = 0.5, size = 12, color = "grey45", face = "italic", margin = margin(t = 10, b = 10))
      )
    ) &
    theme(legend.position = "bottom")
  
  ggsave(file.path(output_dir, "rotation_null_mega_10um.png"),
         mega_rotation_10, width = 12, height = 8, dpi = 300)
}

message("Rotation null visualisation saved to: ", output_dir)

## ---- 6c. (continued) Overlap percentages: %GFP vs. %cFos (paired) ----

real_chance_long %>% count(animal_id) %>% pull(n) %>% unique()
# should return `2`, then every animal is complete

# ---- 6d. Manual vs. automatic correlation + agreement (image level) ----

# Spearman correlation
cor_gfp_sp  <- cor.test(image_data$manual_GFP_total, image_data$auto_GFP_total, method = "spearman")
cor_cfos_sp <- cor.test(image_data$manual_cFos_total, image_data$auto_cFos_total, method = "spearman")
cor_both_sp <- cor.test(image_data$manual_both, image_data$auto_both, method = "spearman")

# Lin's Concordance Correlation Coefficient (with bootstrap CI)
ccc_gfp <- compute_ccc(image_data$manual_GFP_total, image_data$auto_GFP_total)
ccc_cfos <- compute_ccc(image_data$manual_cFos_total, image_data$auto_cFos_total)
ccc_both <- compute_ccc(image_data$manual_both, image_data$auto_both)

# Optional: Bland-Altman stats (already in Section 6e, but we add CIs for LoA)
# These are used later for Fig 7 subtitles
loa_gfp <- compute_loa_ci(image_data$manual_GFP_total, image_data$auto_GFP_total)
loa_cfos <- compute_loa_ci(image_data$manual_cFos_total, image_data$auto_cFos_total)
loa_both <- compute_loa_ci(image_data$manual_both, image_data$auto_both)


## ---- 6e. Bland-Altman stats (manual vs automatic, image level) -----------
bland_altman_stats <- function(manual, auto) {
  diff <- auto - manual
  avg  <- (manual + auto) / 2
  mean_diff <- mean(diff, na.rm = TRUE)
  sd_diff   <- sd(diff, na.rm = TRUE)
  tibble(
    mean_diff = mean_diff,
    sd_diff = sd_diff,
    loa_lower = mean_diff - 1.96 * sd_diff,
    loa_upper = mean_diff + 1.96 * sd_diff,
    avg = list(avg),
    diff = list(diff)
  )
}

ba_gfp  <- bland_altman_stats(image_data$manual_GFP_total, image_data$auto_GFP_total)
ba_cfos <- bland_altman_stats(image_data$manual_cFos_total, image_data$auto_cFos_total)
ba_both <- bland_altman_stats(image_data$manual_both, image_data$auto_both)


## ---- 6f. Training vs. non-training images (overfitting check) ------------
##  delta_GFP/delta_cFos = automatic - manual.
training_test_gfp <- tryCatch(
  coin::wilcox_test(delta_GFP ~ factor(is_training_gfp), data = image_data %>% filter(!is.na(delta_GFP)), distribution = "exact"),
  error = function(e) NULL
)
training_p_gfp <- if (!is.null(training_test_gfp)) coin::pvalue(training_test_gfp) else NA_real_

training_test_cfos <- tryCatch(
  coin::wilcox_test(delta_cFos ~ factor(is_training_cfos), data = image_data %>% filter(!is.na(delta_cFos)), distribution = "exact"),
  error = function(e) NULL
)
training_p_cfos <- if (!is.null(training_test_cfos)) coin::pvalue(training_test_cfos) else NA_real_

# ---- Mean absolute error by training status (descriptive) ----
mean_abs_error_gfp_train <- mean(abs(image_data$delta_GFP[image_data$is_training_gfp]), na.rm = TRUE)
mean_abs_error_gfp_heldout <- mean(abs(image_data$delta_GFP[!image_data$is_training_gfp]), na.rm = TRUE)
mean_abs_error_cfos_train <- mean(abs(image_data$delta_cFos[image_data$is_training_cfos]), na.rm = TRUE)
mean_abs_error_cfos_heldout <- mean(abs(image_data$delta_cFos[!image_data$is_training_cfos]), na.rm = TRUE)

## ============================================================================
## 7. PLOTS
## ============================================================================

# ----  points + lines + median (no raincloud, no boxplot) ----
make_paired_pointplot <- function(long_df, value_col, group_col, id_col, title, ylab,
                                  robust_ylim = FALSE, custom_subtitle = NULL) {
  # Add cohort info if not present
  if (!"cohort" %in% names(long_df)) {
    long_df <- long_df %>%
      left_join(animal_data %>% select(animal_id, cohort), by = "animal_id")
  }
  
  n_per_group <- long_df %>%
    group_by(.data[[group_col]]) %>%
    summarise(n = n_distinct(.data[[id_col]]), .groups = "drop")
  group_levels <- levels(factor(long_df[[group_col]]))
  x_labels <- setNames(
    paste0(group_levels, "\n(n=", n_per_group$n[match(group_levels, n_per_group[[group_col]])], ")"),
    group_levels
  )
  
  # Medians per group
  medians <- long_df %>%
    group_by(.data[[group_col]]) %>%
    summarise(median_val = median(.data[[value_col]], na.rm = TRUE), .groups = "drop")
  
  if (is.null(custom_subtitle)) {
    wide_for_test <- long_df %>%
      select(all_of(c(id_col, group_col, value_col))) %>%
      pivot_wider(names_from = all_of(group_col), values_from = all_of(value_col))
    test_p <- tryCatch(
      wilcox.test(wide_for_test[[group_levels[1]]], wide_for_test[[group_levels[2]]], paired = TRUE)$p.value,
      error = function(e) NA_real_
    )
    p_label <- if (is.na(test_p)) "Wilcoxon (paired): p = NA" else paste0("Wilcoxon (paired): p = ", signif(test_p, 3), " ", sig_stars(test_p))
  } else {
    p_label <- custom_subtitle
  }
  
  p <- ggplot(long_df, aes(x = .data[[group_col]], y = .data[[value_col]],
                           fill = .data[[group_col]], colour = .data[[group_col]])) +
    geom_line(aes(group = .data[[id_col]]), color = "grey50", alpha = 0.6, linewidth = 0.8,
              position = position_nudge(x = 0)) +
    geom_point(aes(group = .data[[id_col]], shape = cohort, fill = .data[[group_col]]),
               size = 3, alpha = 0.9, stroke = 1.2,
               position = position_nudge(x = 0)) +
    geom_segment(data = medians,
                 aes(x = as.numeric(factor(.data[[group_col]])) - 0.15,
                     xend = as.numeric(factor(.data[[group_col]])) + 0.15,
                     y = median_val, yend = median_val,
                     color = .data[[group_col]]),
                 inherit.aes = FALSE, linewidth = 1.2) +
    scale_shape_manual(values = c(16, 17)) +
    scale_fill_manual(values = stroke_pal) +
    scale_colour_manual(values = stroke_pal) +
    scale_x_discrete(labels = x_labels) +
    scale_y_continuous(limits = c(0, NA), expand = expansion(mult = c(0, 0.15))) +
    labs(title = title, subtitle = p_label, y = ylab, x = NULL) +
    theme_thesis +
    guides(
      fill = guide_legend(override.aes = list(
        shape = 21,
        colour = stroke_pal[1:nlevels(factor(long_df[[group_col]]))]
      )),
      colour = "none",
      shape = guide_legend(override.aes = list(shape = c(16, 17)))
    ) 
  
  if (robust_ylim) {
    ylim <- compute_robust_ylim(long_df[[value_col]])
    if (!is.null(ylim)) {
      p <- p + coord_cartesian(ylim = ylim, clip = "off")
    }
  }
  p
}

fig2_subtitle <- paste0(
  "GFP vs. cFos (QC): p = ", signif(gfp_vs_cfos_p, 3), " ", sig_stars(gfp_vs_cfos_p),
  ", r = ", round(r_gfp, 3),
  "  |  Colocalized: mean = ", round(coloc_density_mean, 1), ", median = ", round(coloc_density_median, 1),
  " (IQR = ", round(coloc_density_iqr, 1), ")  [descriptive only]"
)

fig2_long <- animal_data %>%
  select(animal_id, manual_GFP_density, manual_cFos_density, manual_both_density) %>%
  pivot_longer(cols = c(manual_GFP_density, manual_cFos_density, manual_both_density),
               names_to = "channel", values_to = "n_cells") %>%
  mutate(channel = factor(channel, levels = c("manual_GFP_density", "manual_cFos_density", "manual_both_density"),
                          labels = c("GFP", "cFos", "Colocalized")))

fig2_n_per_group <- fig2_long %>%
  group_by(channel) %>%
  summarise(n = n_distinct(animal_id), .groups = "drop")
fig2_x_labels <- setNames(
  paste0(levels(fig2_long$channel), "\n(n=", fig2_n_per_group$n[match(levels(fig2_long$channel), fig2_n_per_group$channel)], ")"),
  levels(fig2_long$channel)
)

fig2 <- make_paired_pointplot(fig2_long, "n_cells", "channel", "animal_id",
                              "GFP+ vs. cFos+ density (Colocalized shown descriptively)",
                              "Density (cells/mm²)",
                              custom_subtitle = fig2_subtitle)
fig2 <- fig2 +
  scale_fill_manual(values = c(GFP = stroke_pal[2], cFos = stroke_pal[1], Colocalized = stroke_pal[3])) +
  scale_colour_manual(values = c(GFP = stroke_pal[2], cFos = stroke_pal[1], Colocalized = stroke_pal[3])) +
  guides(fill = guide_legend(override.aes = list(shape = 21, colour = c(stroke_pal[2], stroke_pal[1], stroke_pal[3]))))

# ---- Fig 3: Overlap percentages (descriptive only) ----
fig3_long <- animal_data %>%
  select(animal_id, cohort, manual_overlap_pct_GFP, manual_overlap_pct_cFos) %>%
  pivot_longer(cols = c(manual_overlap_pct_GFP, manual_overlap_pct_cFos),
               names_to = "ratio_type", values_to = "pct") %>%
  mutate(ratio_type = recode(ratio_type,
                             manual_overlap_pct_GFP = "% of GFP+ reactivated",
                             manual_overlap_pct_cFos = "% of cFos+ pre-tagged"))

# ---- Descriptive stats only ----
fig3_median_gfp <- median(animal_data$manual_overlap_pct_GFP, na.rm = TRUE)
fig3_iqr_gfp <- IQR(animal_data$manual_overlap_pct_GFP, na.rm = TRUE)
fig3_median_cfos <- median(animal_data$manual_overlap_pct_cFos, na.rm = TRUE)
fig3_iqr_cfos <- IQR(animal_data$manual_overlap_pct_cFos, na.rm = TRUE)

fig3_subtitle <- paste0(
  "Descriptive only (no test):",
  "  %GFP+ median = ", round(fig3_median_gfp, 1), "% (IQR = ", round(fig3_iqr_gfp, 1), "%)",
  "  |  %cFos+ median = ", round(fig3_median_cfos, 1), "% (IQR = ", round(fig3_iqr_cfos, 1), "%)"
)

fig3 <- make_paired_pointplot(fig3_long, "pct", "ratio_type", "animal_id",
                              "Overlap as % of each population", "% double-positive",
                              custom_subtitle = fig3_subtitle)

# ---- Fig 3: Add formula as caption inside the plot ----
fig3 <- fig3 + labs(
  caption = "Formula: % of GFP+ = (Colocalized / GFP_total) × 100; % of cFos+ = (Colocalized / cFos_total) × 100"
)

## ---- Fig 4: Real vs chance overlap -----------------------------------------
fig4_long <- animal_data %>%
  select(animal_id, cohort, manual_both, manual_chance_overlap) %>%
  pivot_longer(cols = c(manual_both, manual_chance_overlap),
               names_to = "overlap_type", values_to = "n_cells") %>%
  mutate(overlap_type = recode(overlap_type,
                                manual_both = "Real overlap",
                                manual_chance_overlap = "Chance overlap"))

# ---- Fig 4 subtitle with effect size, Overlap Index, Bootstrap CI ----
fig4_subtitle <- paste0(
  "Wilcoxon (primary): p = ", signif(test_real_vs_chance_wc_p, 3), " ", sig_stars(test_real_vs_chance_wc_p),
  ", r = ", round(eff_real_vs_chance$effsize, 3),
  "\nOverlap Index (mean) = ", round(mean(animal_data$overlap_index, na.rm = TRUE), 3),
  " | Bootstrap 95% CI [", round(boot_ci[1], 1), ", ", round(boot_ci[2], 1), "]",
  "\nSensitivity: t-test p = ", signif(test_real_vs_chance_t$p.value, 3),
  " | ", rotation_summary,
  "\n(BCa bootstrap CI, n=7 – indicative, not nominal)"
)

fig4 <- make_paired_pointplot(fig4_long, "n_cells", "overlap_type", "animal_id",
                              "Real vs. chance double-positive overlap", "Cell count",
                              robust_ylim = TRUE, custom_subtitle = fig4_subtitle)

## ---- Fig 4b: per-animal difference Real - Chance, shows the consistency directly ----

diff_data <- animal_data %>%
  mutate(real_minus_chance = manual_both - manual_chance_overlap,
         direction = ifelse(real_minus_chance > 0, "Real > Chance", "Chance > Real"))

fig4b <- ggplot(diff_data, aes(x = reorder(animal_id, real_minus_chance), y = real_minus_chance, color = direction, shape = cohort)) +
  geom_segment(aes(xend = animal_id, y = 0, yend = real_minus_chance), linewidth = 1) +
  geom_point(aes(shape = cohort), size = 3) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey40") +
  scale_color_manual(values = c("Real > Chance" = stroke_pal[2], "Chance > Real" = stroke_pal[1])) +
  scale_shape_manual(values = c(16, 17)) +
  coord_flip() +
  labs(title = "Per-animal difference: Real \u2212 Chance overlap",
       subtitle = paste0(sum(diff_data$real_minus_chance > 0), " of ", nrow(diff_data), " animals: Real > Chance"),
       x = NULL, y = "Real \u2212 Chance overlap (cell count)") +
  theme_thesis +
  theme(legend.position = "bottom")


## ---- Fig 6: Manual vs automatic scatterplots (image level) ---------------
make_scatter <- function(df, xcol, ycol, title, ccc_obj, rho_obj, acceptance = 0.90) {
  max_val <- max(df[[xcol]], df[[ycol]], na.rm = TRUE)
  n_img <- sum(!is.na(df[[xcol]]) & !is.na(df[[ycol]]))
  
  subtitle <- paste0(
    "Lin's CCC = ", round(ccc_obj$ccc, 2), "\n",
    "(95% BCa CI [", round(ccc_obj$ci_low, 2), ", ", round(ccc_obj$ci_high, 2), "], indicative)\n",
    "Spearman rho = ", round(rho_obj$estimate, 2)
  )
  
  ggplot(df, aes(x = .data[[xcol]], y = .data[[ycol]], color = cohort, shape = cohort)) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey40") +
    geom_point(aes(fill = cohort, color = cohort, shape = cohort), size = 2.8, alpha = 0.85, stroke = 1.1) +
    scale_color_manual(values = cohort_stroke_pal) +
    scale_fill_manual(values = cohort_stroke_pal) +
    scale_shape_manual(values = c(16, 17)) +
    coord_equal(xlim = c(0, max_val), ylim = c(0, max_val)) +
    labs(title = title, x = "Manual count", y = "Automatic count", subtitle = subtitle) +
    theme_thesis +
    theme(plot.subtitle = element_text(size = 10, color = "grey30", hjust = 0.5))
}
# ---- Fig 6 ----
fig6a <- make_scatter(image_data, "manual_GFP_total", "auto_GFP_total",
                      "GFP total:", ccc_gfp, cor_gfp_sp)
fig6b <- make_scatter(image_data, "manual_cFos_total", "auto_cFos_total",
                      "cFos total:", ccc_cfos, cor_cfos_sp)
fig6c <- make_scatter(image_data, "manual_both", "auto_both",
                      "Double-positive:", ccc_both, cor_both_sp)

fig6 <- fig6a + fig6b + fig6c + 
  plot_layout(guides = "collect") &
  theme(legend.position = "bottom", plot.margin = margin(t = 10, r = 20, b = 10, l = 10),  panel.spacing = unit(2, "lines"))

# caption
fig6 <- fig6 + plot_annotation(
  caption = paste(
    "n = 39 images per channel. Acceptance criterion: CCC > 0.90.",
    "Note: images from the same animal are not independent (pseudoreplication).",
    "Training images are from Cohort 1 only."
  ),
  theme = theme(
    plot.caption = element_text(hjust = 0.5, size = 12, color = "grey45", face = "italic", margin = margin(t = 10, b = 10))
  )
)

# =====================================================================
# Bland-Altman Plots
# =====================================================================
make_bland_altman <- function(manual, auto, cohort_vec, title, loa_obj, acceptance = 20) {
  df <- tibble(
    avg = (manual + auto) / 2,
    diff = auto - manual,
    cohort = cohort_vec
  )
  n_img <- loa_obj$n
  
  subtitle <- paste0(
    "Bias = ", round(loa_obj$bias, 1),
    " (95% CI [", round(loa_obj$bias_ci[1], 1), ", ", round(loa_obj$bias_ci[2], 1), "])\n",
    "95% LoA [", round(loa_obj$loa_lower, 1), ", ", round(loa_obj$loa_upper, 1), "]",
    " (CI for LoA: [", round(loa_obj$ci_loa_lower[1], 1), ", ", round(loa_obj$ci_loa_upper[2], 1), "])"
  )
  
  ggplot(df, aes(x = avg, y = diff, color = cohort, shape = cohort)) +
    geom_hline(yintercept = loa_obj$bias, color = "grey30", linewidth = 0.8) +
    geom_hline(yintercept = c(loa_obj$loa_lower, loa_obj$loa_upper), linetype = "dashed", color = "grey50") +
    geom_point(aes(fill = cohort, color = cohort), size = 2.8, alpha = 0.85, stroke = 1.1) +
    scale_color_manual(values = cohort_stroke_pal) +
    scale_fill_manual(values = cohort_stroke_pal) +
    scale_shape_manual(values = c(16, 17)) +
    labs(title = title, x = "Mean of manual & automatic", y = "Automatic - Manual",
         subtitle = subtitle) +
    theme_thesis +
    theme(plot.subtitle = element_text(size = 13, color = "grey30", hjust = 0.5))
}

# ---- Fig 7 ----
fig7a <- make_bland_altman(image_data$manual_GFP_total, image_data$auto_GFP_total,
                           image_data$cohort, "Bland-Altman: GFP total", loa_gfp)
fig7b <- make_bland_altman(image_data$manual_cFos_total, image_data$auto_cFos_total,
                           image_data$cohort, "Bland-Altman: cFos total", loa_cfos)
fig7c <- make_bland_altman(image_data$manual_both, image_data$auto_both,
                           image_data$cohort, "Bland-Altman: double-positive", loa_both)

fig7 <- fig7a + fig7b + fig7c +
  plot_layout(guides = "collect") +
  plot_annotation(
    caption = paste(
      "n = 39 images per channel. Acceptance criterion: bias within ±20%.",
      "Training images are from Cohort 1 only."
    ),
    theme = theme(
      plot.caption = element_text(hjust = 0.5, size = 12, color = "grey45", face = "italic", margin = margin(t = 10, b = 10))
    )
  ) &
  theme(legend.position = "bottom", plot.margin = margin(t = 10, r = 20, b = 10, l = 10))

# =====================================================================
# Bland-Altman Fan Shape Check (optimized for n=39)
# =====================================================================

check_bland_altman_fanshape <- function(df, manual_col, auto_col, title) {
  
  # 1. Prepare data
  d <- df %>%
    filter(!is.na(.data[[manual_col]]), !is.na(.data[[auto_col]])) %>%
    mutate(
      avg = (.data[[manual_col]] + .data[[auto_col]]) / 2,
      diff = .data[[auto_col]] - .data[[manual_col]],
      abs_diff = abs(diff)
    )
  
  # ---- Linear regression of absolute differences (quantifies fan shape) ----
  abs_model <- lm(abs_diff ~ avg, data = d)
  slope <- coef(abs_model)[["avg"]]
  intercept <- coef(abs_model)[["(Intercept)"]]
  r_squared <- summary(abs_model)$r.squared
  slope_pval <- summary(abs_model)$coefficients["avg", "Pr(>|t|)"]
  
  # ---- Spearman Rho (robust, primary significance test) ----
  spearman_test <- cor.test(d$avg, d$abs_diff, method = "spearman", exact = FALSE)
  spearman_rho <- spearman_test$estimate
  spearman_pval <- spearman_test$p.value
  
  # ---- Decision: Fan shape present if Spearman p < 0.05 ----
  has_fanshape <- spearman_pval < 0.05
  status_text <- ifelse(has_fanshape, " * (fan shape detected)", " (no significant fan shape)")
  
  # ---- Plot: Bland-Altman with loess smoother ----
  # Define gray for loess line
  loess_gray <- "#898989"
  
  p <- ggplot(d, aes(x = avg, y = diff, color = cohort, shape = cohort)) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "grey50", linewidth = 0.6) +
    geom_point(size = 2.5, alpha = 0.8, stroke = 1) +
    geom_smooth(aes(x = avg, y = diff, group = 1), 
                method = "loess", se = TRUE, color = loess_gray,
                linewidth = 1, alpha = 0.3, inherit.aes = FALSE) +
    scale_color_manual(values = cohort_stroke_pal) +
    scale_shape_manual(values = c(16, 17)) +
    labs(
      title = paste("Bland-Altman Fan Shape Check –", title),
      subtitle = paste0(
        "Spearman rho = ", round(spearman_rho, 2), " (p = ", signif(spearman_pval, 3), ")", status_text,
        "\nSlope = ", round(slope, 3), " (R² = ", round(r_squared, 3), ")"
      ),
      x = "Mean of manual & automatic",
      y = "Difference (Automatic - Manual)",
      caption = NULL
    ) +
    theme_thesis +
    theme(legend.position = "bottom")
  
  return(list(
    plot = p,
    slope = slope,
    slope_p_value = slope_pval,
    r_squared = r_squared,
    spearman_rho = unname(spearman_rho),
    spearman_p_value = spearman_pval,
    n = nrow(d)
  ))
}

# ---- Run fan shape check for GFP, cFos, Both ----
fan_gfp <- check_bland_altman_fanshape(image_data, "manual_GFP_total", "auto_GFP_total", "GFP total")
fan_cfos <- check_bland_altman_fanshape(image_data, "manual_cFos_total", "auto_cFos_total", "cFos total")
fan_both <- check_bland_altman_fanshape(image_data, "manual_both", "auto_both", "Double-positive")

# ---- Combine into a mega-grid ----
mega_fanshape <- wrap_plots(
  fan_gfp$plot + labs(title = "GFP total", caption = NULL),
  fan_cfos$plot + labs(title = "cFos total", caption = NULL),
  fan_both$plot + labs(title = "Double-positive", caption = NULL),
  ncol = 3,
  guides = "collect"
) +
  plot_annotation(
    title = "Bland-Altman Fan Shape Check",
    subtitle = "Spearman correlation (primary) + linear regression (slope)",
    caption = paste(
      "Spearman p < 0.05 indicates significant fan shape (proportional bias).",
      "Slope quantifies error increase with mean.",
      "Loess smoother (gray) shows trend."
    ),
    theme = theme(
      plot.title = element_text(hjust = 0.5, face = "bold", size = 20),
      plot.subtitle = element_text(hjust = 0.5, size = 13, color = "grey30"),
      plot.caption = element_text(hjust = 0.5, size = 12, color = "grey45", face = "italic", margin = margin(t = 10, b = 10))
    )
  ) &
  theme(legend.position = "bottom")

ggsave(file.path(output_dir, "bland_altman_fanshape_check.png"),
       mega_fanshape, width = 15, height = 5.5, dpi = 300)

# ---- Save results as CSV ----
fan_results <- tibble(
  channel = c("GFP", "cFos", "Both"),
  n = c(fan_gfp$n, fan_cfos$n, fan_both$n),
  spearman_rho = c(fan_gfp$spearman_rho, fan_cfos$spearman_rho, fan_both$spearman_rho),
  spearman_p_value = c(fan_gfp$spearman_p_value, fan_cfos$spearman_p_value, fan_both$spearman_p_value),
  slope = c(fan_gfp$slope, fan_cfos$slope, fan_both$slope),
  slope_p_value = c(fan_gfp$slope_p_value, fan_cfos$slope_p_value, fan_both$slope_p_value),
  r_squared = c(fan_gfp$r_squared, fan_cfos$r_squared, fan_both$r_squared),
  fan_shape_detected = c(
    fan_gfp$spearman_p_value < 0.05,
    fan_cfos$spearman_p_value < 0.05,
    fan_both$spearman_p_value < 0.05
  )
)
write_csv(fan_results, file.path(output_dir, "bland_altman_fanshape_stats.csv"))

message("Bland-Altman fan shape check (optimized) saved to: ", output_dir)

# =====================================================================
# Fig 8: Overfitting check (error on training vs. held-out)
# =====================================================================

make_overfitting_plot <- function(df, manual_col, auto_col, training_col, title) {
  df <- df %>%
    filter(!is.na(.data[[manual_col]]), !is.na(.data[[auto_col]])) %>%
    mutate(
      delta = .data[[auto_col]] - .data[[manual_col]],   # auto - manual
      training_status = factor(ifelse(.data[[training_col]], "Training", "Held-out"),
                               levels = c("Held-out", "Training")),
      cohort = ifelse(animal_id %in% coh1_animals, "Coh1", "Coh2")
    )
  
  test_res <- coin::wilcox_test(delta ~ training_status, data = df, distribution = "exact")
  p_val <- coin::pvalue(test_res)
  
  n_train <- sum(df$training_status == "Training")
  n_held <- sum(df$training_status == "Held-out")
  
  medians <- df %>%
    group_by(training_status) %>%
    summarise(median_delta = median(delta, na.rm = TRUE), .groups = "drop")
  
  subtitle <- paste0(
    "Wilcoxon: p = ", signif(p_val, 3), " ", sig_stars(p_val),
    "  |  n(Training) = ", n_train, ", n(Held-out) = ", n_held
  )
  
  p <- ggplot(df, aes(x = training_status, y = delta, shape = cohort)) +
    geom_jitter(aes(fill = training_status, color = training_status, shape = cohort),
                width = 0.15, height = 0, size = 2.4,
                alpha = 0.9, stroke = 1.2) +
    geom_segment(data = medians,
                 aes(x = as.numeric(training_status) - 0.15,
                     xend = as.numeric(training_status) + 0.15,
                     y = median_delta, yend = median_delta,
                     color = training_status),
                 inherit.aes = FALSE, linewidth = 1.2) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "grey40") +
    scale_shape_manual(values = c(16, 17)) +
    scale_fill_manual(values = c("Training" = stroke_pal[6], "Held-out" = stroke_pal[4])) +
    scale_color_manual(values = c("Training" = stroke_pal[6], "Held-out" = stroke_pal[4])) +
    labs(title = title,
         subtitle = subtitle,
         y = "Automatic − Manual (error)", x = NULL) +
    theme_thesis +
    theme(legend.position = "bottom", plot.subtitle = element_text(size = 13, color = "grey30", hjust = 0.5))
  
  return(list(plot = p, p_value = p_val,
              median_train = medians$median_delta[medians$training_status == "Training"],
              median_heldout = medians$median_delta[medians$training_status == "Held-out"]))
}

overfit_gfp <- make_overfitting_plot(image_data, "manual_GFP_total", "auto_GFP_total",
                                     "is_training_gfp", "Overfitting check: GFP")
fig8_gfp <- overfit_gfp$plot
training_p_gfp <- overfit_gfp$p_value

overfit_cfos <- make_overfitting_plot(image_data, "manual_cFos_total", "auto_cFos_total",
                                      "is_training_cfos", "Overfitting check: cFos")
fig8_cfos <- overfit_cfos$plot
training_p_cfos <- overfit_cfos$p_value


# =====================================================================
# Fig 8b: Batch effect check (error stratified by cohort)
# =====================================================================

make_batch_plot <- function(df, manual_col, auto_col, title) {
  df <- df %>%
    filter(!is.na(.data[[manual_col]]), !is.na(.data[[auto_col]])) %>%
    mutate(
      delta = .data[[auto_col]] - .data[[manual_col]],  
      cohort = factor(cohort, levels = c("Coh1", "Coh2"))
    )
  
  test_res <- coin::wilcox_test(delta ~ cohort, data = df, distribution = "exact")
  p_val <- coin::pvalue(test_res)
  
  n_coh1 <- sum(df$cohort == "Coh1")
  n_coh2 <- sum(df$cohort == "Coh2")
  
  medians <- df %>%
    group_by(cohort) %>%
    summarise(median_delta = median(delta, na.rm = TRUE), .groups = "drop")
  
  subtitle <- paste0(
    "Wilcoxon (Coh1 vs. Coh2): p = ", signif(p_val, 3), " ", sig_stars(p_val),
    "  |  n(Coh1) = ", n_coh1, ", n(Coh2) = ", n_coh2
  )
  
  p <- ggplot(df, aes(x = cohort, y = delta, shape = cohort)) +
    geom_jitter(aes(fill = cohort, color = cohort, shape = cohort),
                width = 0.15, height = 0, size = 2.4,
                alpha = 0.9, stroke = 1.2) +
    geom_segment(data = medians,
                 aes(x = as.numeric(cohort) - 0.15,
                     xend = as.numeric(cohort) + 0.15,
                     y = median_delta, yend = median_delta,
                     color = cohort),
                 inherit.aes = FALSE, linewidth = 1.2) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "grey40") +
    scale_shape_manual(values = c(16, 17)) +
    scale_fill_manual(values = cohort_stroke_pal) +
    scale_color_manual(values = cohort_stroke_pal) +
    labs(title = title,
         subtitle = subtitle,
         y = "Automatic − Manual (error)", x = NULL) +
    theme_thesis +
    theme(legend.position = "bottom")
  
  return(list(plot = p, p_value = p_val, n_coh1 = n_coh1, n_coh2 = n_coh2,
              median_coh1 = medians$median_delta[medians$cohort == "Coh1"],
              median_coh2 = medians$median_delta[medians$cohort == "Coh2"]))
}

batch_gfp <- make_batch_plot(image_data, "manual_GFP_total", "auto_GFP_total",
                             "Batch effect check: GFP (Coh1 vs. Coh2)")
fig8_batch_gfp <- batch_gfp$plot
batch_test_gfp <- batch_gfp$p_value

batch_cfos <- make_batch_plot(image_data, "manual_cFos_total", "auto_cFos_total",
                              "Batch effect check: cFos (Coh1 vs. Coh2)")
fig8_batch_cfos <- batch_cfos$plot
batch_test_cfos <- batch_cfos$p_value

message("Batch GFP medians: Coh1 = ", round(batch_gfp$median_coh1, 1), ", Coh2 = ", round(batch_gfp$median_coh2, 1))
message("Batch cFos medians: Coh1 = ", round(batch_cfos$median_coh1, 1), ", Coh2 = ", round(batch_cfos$median_coh2, 1))

# =====================================================================
# FDR CORRECTION FOR PIPELINE VALIDATION (4 tests)
# =====================================================================

# Collect p-values from all pipeline validation tests
pipeline_p_values <- c(
  training_p_gfp,    # Overfitting GFP (extracted earlier)
  training_p_cfos,   # Overfitting cFos (extracted earlier)
  batch_test_gfp,    # Batch GFP (extracted from make_batch_plot)
  batch_test_cfos    # Batch cFos (extracted from make_batch_plot)
)

# Remove NAs and non-finite values
pipeline_p_values <- pipeline_p_values[!is.na(pipeline_p_values) & is.finite(pipeline_p_values)]

# Apply FDR correction (Benjamini-Hochberg)
if (length(pipeline_p_values) > 0) {
  pipeline_p_fdr <- p.adjust(pipeline_p_values, method = "BH")
} else {
  pipeline_p_fdr <- numeric(0)
}

# Create a named vector for easy lookup
names(pipeline_p_fdr) <- c("Overfitting_GFP", "Overfitting_cFos", "Batch_GFP", "Batch_cFos")[1:length(pipeline_p_fdr)]

fig8_gfp <- fig8_gfp + labs(subtitle = paste0(
  "Wilcoxon: p = ", signif(training_p_gfp, 3), " ", sig_stars(training_p_gfp),
  ", p_FDR = ", signif(pipeline_p_fdr["Overfitting_GFP"], 3),
  "  |  n(Training) = 4, n(Held-out) = 35"
))
fig8_cfos <- fig8_cfos + labs(subtitle = paste0(
  "Wilcoxon: p = ", signif(training_p_cfos, 3), " ", sig_stars(training_p_cfos),
  ", p_FDR = ", signif(pipeline_p_fdr["Overfitting_cFos"], 3),
  "  |  n(Training) = 4, n(Held-out) = 35"
))
fig8_batch_gfp <- fig8_batch_gfp + labs(subtitle = paste0(
  "Wilcoxon (Coh1 vs. Coh2): p = ", signif(batch_test_gfp, 3), " ", sig_stars(batch_test_gfp),
  ", p_FDR = ", signif(pipeline_p_fdr["Batch_GFP"], 3),
  "  |  n(Coh1) = ", batch_gfp$n_coh1, ", n(Coh2) = ", batch_gfp$n_coh2
))
fig8_batch_cfos <- fig8_batch_cfos + labs(subtitle = paste0(
  "Wilcoxon (Coh1 vs. Coh2): p = ", signif(batch_test_cfos, 3), " ", sig_stars(batch_test_cfos),
  ", p_FDR = ", signif(pipeline_p_fdr["Batch_cFos"], 3),
  "  |  n(Coh1) = ", batch_cfos$n_coh1, ", n(Coh2) = ", batch_cfos$n_coh2
))


# Print to console for debugging
message("Pipeline validation p-values (raw): ", paste(signif(pipeline_p_values, 3), collapse = ", "))
message("Pipeline validation p-values (FDR): ", paste(signif(pipeline_p_fdr, 3), collapse = ", "))

fig8 <- fig8_gfp + fig8_cfos +
  plot_layout(guides = "collect") +
  plot_annotation(
    caption = paste(
      "Note: images from the same animal are not independent (pseudoreplication).",
      "p-values are exploratory.",
      "Training images are from Cohort 1 only."
    ),
    theme = theme(
      plot.caption = element_text(hjust = 0.5, size = 12, color = "grey45", face = "italic", margin = margin(t = 10, b = 10))
    )
  ) &
  theme(legend.position = "bottom")


fig8_batch <- fig8_batch_gfp + fig8_batch_cfos +
  plot_layout(guides = "collect") +
  plot_annotation(
    caption = paste(
      "Note: images from the same animal are not independent (pseudoreplication).",
      "p-values are exploratory.",
      "Training images are from Cohort 1 only."
    ),
    theme = theme(
      plot.caption = element_text(hjust = 0.5, size = 12, color = "grey45", face = "italic", margin = margin(t = 10, b = 10))
    )
  ) &
  theme(legend.position = "bottom")

ggsave(file.path(output_dir, "fig8_overfitting_check.png"), fig8_batch, width = 12, height = 5.5, dpi = 300)
ggsave(file.path(output_dir, "fig8_batch_effect_check.png"), fig8_batch, width = 12, height = 5.5, dpi = 300)

## ---- Fig 8c: per-image manual vs. automatic, training images marked ------

make_scatter_training <- function(df, xcol, ycol, training_col, title, ccc_obj, rho_obj, acceptance = 0.90) {
  df <- df %>%
    filter(!is.na(.data[[xcol]]), !is.na(.data[[ycol]])) %>%
    mutate(
      group = factor(ifelse(.data[[training_col]], "Training image", "Non-training image"),
                     levels = c("Non-training image", "Training image")),
      cohort = ifelse(animal_id %in% coh1_animals, "Coh1", "Coh2")
    )
  
  max_val <- max(df[[xcol]], df[[ycol]], na.rm = TRUE)
  n_train <- sum(df$group == "Training image")
  n_nontrain <- sum(df$group == "Non-training image")
  
  subtitle <- paste0(
    "Lin's CCC = ", round(ccc_obj$ccc, 2), "\n",
    "(95% BCa CI [", round(ccc_obj$ci_low, 2), ", ", round(ccc_obj$ci_high, 2), "], indicative)\n",
    "Spearman rho = ", round(rho_obj$estimate, 2)
  )
  
  ggplot(df, aes(x = .data[[xcol]], y = .data[[ycol]], fill = group, color = group, shape = cohort)) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey40") +
    geom_point(aes(fill = group, color = group, shape = cohort),
               size = 2.8, alpha = 0.85, stroke = 1.1) +
    scale_fill_manual(values = c("Non-training image" = stroke_pal[4], "Training image" = stroke_pal[6])) +
    scale_color_manual(values = c("Non-training image" = stroke_pal[4], "Training image" = stroke_pal[6])) +
    scale_shape_manual(values = c(16, 17)) +
    coord_equal(xlim = c(0, max_val), ylim = c(0, max_val)) +
    labs(title = title,
         x = "Manual count",
         y = "Automatic count",
         subtitle = subtitle) +
    theme_thesis +
    theme(plot.subtitle = element_text(size = 10, color = "grey30", hjust = 0.5))
}

# ---- Fig 8c ----
fig8c_gfp <- make_scatter_training(image_data, "manual_GFP_total", "auto_GFP_total",
                                   "is_training_gfp", "GFP",
                                   ccc_gfp, cor_gfp_sp)
fig8c_cfos <- make_scatter_training(image_data, "manual_cFos_total", "auto_cFos_total",
                                    "is_training_cfos", "cFos",
                                    ccc_cfos, cor_cfos_sp)
fig8c_coloc <- make_scatter_training(image_data, "manual_both", "auto_both",
                                     "is_training_either", "Colocalized",
                                     ccc_both, cor_both_sp)

fig8c <- fig8c_gfp + fig8c_cfos + fig8c_coloc + 
  plot_layout(guides = "collect") &
  theme(legend.position = "bottom", plot.margin = margin(t = 10, r = 20, b = 10, l = 10), panel.spacing = unit(2, "lines"))

fig8c <- fig8c + plot_annotation(
  caption = paste(
    "Note: images from the same animal are not independent (pseudoreplication).",
    "Training images are from Cohort 1 only.",
    "n = 4 training / 35 held-out images for GFP and cFos."
  ),
  theme = theme(
    plot.caption = element_text(hjust = 0.5, size = 12, color = "grey45", face = "italic")
  )
)

if (exists("rostro_animal") && nrow(rostro_animal) > 0) {
  
  # =====================================================================
  # Per-Animal Agreement: manual vs. automatic (summed across images)
  # =====================================================================
  # This is the "per-animal aggregate"
  # Each point = one animal (n=7), showing the total manual count
  # vs. total automatic count across all images of that animal.
  
  # Prepare animal-level agreement data
  animal_agreement <- animal_data %>%
    select(animal_id, cohort,
           manual_GFP_density, auto_GFP_density,
           manual_cFos_density, auto_cFos_density,
           manual_both_density, auto_both_density) %>%
    pivot_longer(
      cols = c(manual_GFP_density, auto_GFP_density, manual_cFos_density, auto_cFos_density, manual_both_density, auto_both_density),
      names_to = c("source", "channel"),
      names_pattern = "(manual|auto)_(GFP_density|cFos_density|both_density)",
      values_to = "count"
    ) %>%
    mutate(
      channel = recode(channel,
                       GFP_density = "GFP",
                       cFos_density = "cFos",
                       both_density = "Colocalized"),
      source = recode(source, manual = "Manual", auto = "Automatic")
    ) %>%
    pivot_wider(names_from = source, values_from = count)
  
  # ---- Per-animal CCC/rho (for key_stats_summary export) ----
  ccc_animal_gfp <- compute_ccc(
    animal_agreement$Manual[animal_agreement$channel == "GFP"],
    animal_agreement$Automatic[animal_agreement$channel == "GFP"]
  )
  rho_animal_gfp <- cor(
    animal_agreement$Manual[animal_agreement$channel == "GFP"],
    animal_agreement$Automatic[animal_agreement$channel == "GFP"],
    method = "spearman"
  )
  
  ccc_animal_cfos <- compute_ccc(
    animal_agreement$Manual[animal_agreement$channel == "cFos"],
    animal_agreement$Automatic[animal_agreement$channel == "cFos"]
  )
  rho_animal_cfos <- cor(
    animal_agreement$Manual[animal_agreement$channel == "cFos"],
    animal_agreement$Automatic[animal_agreement$channel == "cFos"],
    method = "spearman"
  )
  
  ccc_animal_both <- compute_ccc(
    animal_agreement$Manual[animal_agreement$channel == "Colocalized"],
    animal_agreement$Automatic[animal_agreement$channel == "Colocalized"]
  )
  rho_animal_both <- cor(
    animal_agreement$Manual[animal_agreement$channel == "Colocalized"],
    animal_agreement$Automatic[animal_agreement$channel == "Colocalized"],
    method = "spearman"
  )
  
  # Create scatterplot with animal IDs
  make_animal_agreement_plot <- function(data, channel_name, max_val = NULL) {
    d <- data %>% filter(channel == channel_name)
    
    if (is.null(max_val)) {
      max_val <- max(d$Manual, d$Automatic, na.rm = TRUE)
    }
    
    # ---- Lin's CCC (with BCa CI) ----
    ccc_obj <- compute_ccc(d$Manual, d$Automatic, n_boot = 2000, conf_level = 0.95)
    
    # Spearman rho (deskriptiv, als Zusatz)
    rho <- cor(d$Manual, d$Automatic, method = "spearman", use = "complete.obs")
    
    # Cohort assignment
    d <- d %>%
      mutate(cohort = ifelse(animal_id %in% coh1_animals, "Coh1", "Coh2"))
    
    ggplot(d, aes(x = Manual, y = Automatic, color = cohort, shape = cohort)) +
      geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey50") +
      geom_point(size = 4, alpha = 0.9, stroke = 1.2) +
      geom_text_repel(aes(label = animal_id),
                      size = 3, color = "grey30",
                      box.padding = 0.5, point.padding = 0.3,
                      show.legend = FALSE) +
      scale_color_manual(values = cohort_stroke_pal) +
      scale_shape_manual(values = c(Coh1 = 16, Coh2 = 17)) +
      coord_equal(xlim = c(0, max_val), ylim = c(0, max_val)) +
      labs(
        title = paste(channel_name, "– manual vs. automatic (per animal)"),
        subtitle = paste0(
          "Lin's CCC = ", round(ccc_obj$ccc, 2), "\n",
          "(95% BCa CI [", round(ccc_obj$ci_low, 2), ", ", round(ccc_obj$ci_high, 2), "], indicative)\n",
          "Spearman rho = ", round(rho, 2), "\n",
          "Acceptance: CCC > 0.90"
        ),
        x = "Manual density (cells/mm²)",
        y = "Automatic density (cells/mm²)"
      ) +
      theme_thesis +
      theme(
        legend.position = "bottom",
        plot.margin = margin(t = 35, r = 20, b = 10, l = 20),
        axis.title = element_text(size = 10),
        axis.text = element_text(size = 9)
      )
  }
  
  # Determine max value across all channels for consistent scaling
  max_val_animal <- max(animal_agreement$Manual, animal_agreement$Automatic, na.rm = TRUE)
  
  # Generate plots for each channel
  plot_animal_GFP <- make_animal_agreement_plot(animal_agreement, "GFP", max_val_animal)
  plot_animal_cFos <- make_animal_agreement_plot(animal_agreement, "cFos", max_val_animal)
  plot_animal_Coloc <- make_animal_agreement_plot(animal_agreement, "Colocalized", max_val_animal)
  
  # Combine into a mega-grid
  mega_animal_agreement <- wrap_plots(
    plot_animal_GFP + labs(title = "GFP (per animal)"),
    plot_animal_cFos + labs(title = "cFos (per animal)"),
    plot_animal_Coloc + labs(title = "Colocalized (per animal)"),
    ncol = 3,
    guides = "collect"
  ) +
    plot_annotation(
      title = "Manual vs. Automatic agreement – per animal (n=7)",
      subtitle = "Values are summed across all images per animal. Descriptive only – no p-values.",
      caption = paste(
        "Each point = one animal. Values are summed across all images per animal (per-animal aggregate)."
      ),
      theme = theme(
        plot.title = element_text(hjust = 0.5, face = "bold", size = 20),
        plot.subtitle = element_text(hjust = 0.5, size = 13, color = "grey30"),
        plot.caption = element_text(hjust = 0.5, size = 12, color = "grey45", face = "italic", margin = margin(t = 10, b = 10))
      )
    ) &
    theme(legend.position = "bottom")
  
  ggsave(file.path(output_dir, "fig_animal_agreement.png"),
         mega_animal_agreement, width = 10, height = 5.5, dpi = 300)
  
  # ---- Fig 9: Rostrocaudal gradient (descriptive, no statistical test) ----
  rostro_long <- rostro_animal %>%
    select(animal_id, cohort, rc_bin, manual_GFP_density, manual_cFos_density, manual_both_density, manual_coloc_pct) %>%
    pivot_longer(cols = c(manual_GFP_density, manual_cFos_density, manual_both_density, manual_coloc_pct),
                 names_to = "metric", values_to = "value") %>%
    mutate(metric = recode(metric,
                           manual_GFP_density = "GFP density (cells/mm²)",
                           manual_cFos_density = "cFos density (cells/mm²)",
                           manual_both_density = "Colocalized density (cells/mm²)",
                           manual_coloc_pct = "Colocalization %"))
  
  rostro_labels <- tibble(
    metric = c("GFP density (cells/mm²)", "cFos density (cells/mm²)", 
               "Colocalized density (cells/mm²)", "Colocalization %"),
    metric_display = metric
  )
  
  rostro_long <- rostro_long %>%
    left_join(rostro_labels, by = "metric") %>%
    mutate(metric_with_p = metric_display) %>%
    mutate(metric_with_p = factor(metric_with_p, levels = unique(metric_display)))
  
  fig9 <- ggplot(rostro_long, aes(x = rc_bin, y = value, group = animal_id, color = cohort)) +
    geom_line(alpha = 0.5) +
    geom_point(aes(shape = cohort), size = 2.6, alpha = 0.85) +
    scale_shape_manual(values = c(16, 17)) +
    scale_color_manual(values = cohort_stroke_pal) +
    scale_y_continuous(expand = expansion(mult = c(0.05, 0.12))) +
    facet_wrap(~metric_with_p, scales = "free_y", axes = "all") +
    labs(title = "Rostrocaudal gradient (relative anterior-to-posterior position per animal)",
         subtitle = "Descriptive only",
         x = NULL, y = NULL) +
    theme_thesis +
    theme(panel.spacing = unit(1.3, "lines"), strip.text = element_text(size = 8, face = "bold"))
  
  # ---- Fig 9b: Rostrocaudal heatmap (descriptive) ----
  rostro_heat <- rostro_long %>%
    mutate(animal_num = as.integer(str_extract(as.character(animal_id), "\\d+"))) %>%
    arrange(animal_num) %>%
    mutate(animal_id = factor(animal_id, levels = unique(animal_id))) %>%
    group_by(metric) %>%
    mutate(
      value_clean = ifelse(is.finite(value), value, NA_real_),
      value_norm = if (sum(is.finite(value)) > 0 && diff(range(value_clean, na.rm = TRUE)) > 0)
        (value_clean - min(value_clean, na.rm = TRUE)) / (max(value_clean, na.rm = TRUE) - min(value_clean, na.rm = TRUE))
      else 0.5,
      tile_label = ifelse(is.finite(value), as.character(round(value, 1)), "\u2013")
    ) %>%
    ungroup()
  
  # Cohort symbols for y-axis labels
  animal_labels_heat <- rostro_heat %>%
    distinct(animal_id) %>%
    mutate(cohort = ifelse(animal_id %in% coh1_animals, "Coh1", "Coh2")) %>%
    mutate(label = case_when(
      cohort == "Coh1" ~ paste0("● ", animal_id),
      cohort == "Coh2" ~ paste0("▲ ", animal_id)
    )) %>%
    arrange(animal_id)
  
  fig9b <- ggplot(rostro_heat, aes(x = rc_bin, y = factor(animal_id, levels = animal_labels_heat$animal_id), fill = value_norm)) +
    geom_tile(color = "white", linewidth = 1.2) +
    geom_text(aes(label = tile_label), size = 3.5, color = "grey15") +
    scale_y_discrete(labels = setNames(animal_labels_heat$label, animal_labels_heat$animal_id)) +
    scale_fill_gradient(low = "#FBF8FD", high = stroke_pal[5], na.value = "grey92", guide = "none") +
    facet_wrap(~metric, scales = "free_x", nrow = 2, strip.position = "top") +
    labs(
      title = "Rostrocaudal pattern per animal (heatmap view)",
      subtitle = "Light = low, dark = high (scaled separately per metric); numbers shown are always the real values.",
      caption = "Cohorts: ● = Coh1, ▲ = Coh2.",
      x = NULL,
      y = "Animal"
    ) +
    theme_thesis +
    theme(
      panel.grid = element_blank(),
      axis.text.y = element_text(size = 8),
      axis.text.x = element_text(size = 10, color = "black", angle = 0, hjust = 0.5),
      axis.title.x = element_blank(),
      strip.background = element_rect(fill = "#F1EAF7", color = NA),
      strip.text = element_text(color = "#5C3D82", face = "bold", size = 9),
      panel.background = element_rect(fill = "white", color = NA),
      plot.background = element_rect(fill = "white", color = NA),
      panel.border = element_blank(),
      panel.spacing = unit(1, "cm"),
      legend.position = "none"
    )
  
  # ---- Save fig9b ----
  ggsave(file.path(output_dir, "fig9b_rostrocaudal_heatmap.png"),
         fig9b, width = 12, height = 8, dpi = 300)
  
}
  
## ---- Fig cohort: per-animal comparison within each cohort ---------------
## Three panels: raw counts, densities, colocalization % - each showing
## GFP, cFos and Both
## x-axis uses cohort_order (animal's position in the cohort sequence) so the
## within-cohort trend is directly visible; a linear trend line is added per
## channel as a visual guide (descriptive only, no formal test).

cohort_animals_counts_long <- animal_data %>%
  select(animal_id, cohort, cohort_order, manual_GFP_total, manual_cFos_total, manual_both) %>%
  pivot_longer(cols = c(manual_GFP_total, manual_cFos_total, manual_both),
               names_to = "channel", values_to = "value") %>%
  mutate(channel = recode(channel, manual_GFP_total = "GFP", manual_cFos_total = "cFos", manual_both = "Colocalized"))

cohort_animals_density_long <- animal_data %>%
  select(animal_id, cohort, cohort_order, manual_GFP_density, manual_cFos_density, manual_both_density) %>%
  pivot_longer(cols = c(manual_GFP_density, manual_cFos_density, manual_both_density),
               names_to = "channel", values_to = "value") %>%
  mutate(channel = recode(channel, manual_GFP_density = "GFP", manual_cFos_density = "cFos", manual_both_density = "Colocalized"))

cohort_animals_pct <- animal_data %>%
  select(animal_id, cohort, cohort_order, manual_coloc_pct)

## Facet strip labels get an explicit "(n = X)" per cohort - animal counts, not image counts
cohort_n <- animal_data %>% count(cohort) %>% tibble::deframe()
cohort_facet_labeller <- as_labeller(function(x) paste0(x, " (n=", cohort_n[x], ")"))

## ---- Counts panel ----
fig_cohort_counts <- ggplot(cohort_animals_counts_long, aes(x = cohort_order, y = value, color = channel, shape = cohort)) +
  geom_point(size = 3) +
  geom_smooth(method = "lm", se = FALSE, linewidth = 0.6, linetype = "dashed") +
  facet_grid(channel ~ cohort, scales = "free", labeller = labeller(cohort = cohort_facet_labeller)) +
  scale_color_manual(values = channel_stroke_pal) +
  scale_shape_manual(values = c(Coh1 = 16, Coh2 = 17)) +
  scale_x_continuous(breaks = scales::breaks_width(1)) +
  labs(title = "Raw cell counts: each animal's position within its cohort",
       y = "Cell count", x = "Animal (ordered within cohort)") +
  theme_thesis

## ---- Density panel ----
fig_cohort_density <- ggplot(cohort_animals_density_long, aes(x = cohort_order, y = value, color = channel, shape = cohort)) +
  geom_point(size = 3) +
  geom_smooth(method = "lm", se = FALSE, linewidth = 0.6, linetype = "dashed") +
  facet_grid(channel ~ cohort, scales = "free", labeller = labeller(cohort = cohort_facet_labeller)) +
  scale_color_manual(values = channel_stroke_pal) +
  scale_shape_manual(values = c(Coh1 = 16, Coh2 = 17)) +
  scale_x_continuous(breaks = scales::breaks_width(1)) +
  labs(title = "Density: each animal's position within its cohort",
       y = "Cells / mm²", x = "Animal (ordered within cohort)") +
  theme_thesis

## ---- Colocalization % panel ----
fig_cohort_coloc_pct <- ggplot(cohort_animals_pct, aes(x = cohort_order, y = manual_coloc_pct, color = cohort, shape = cohort)) +
  geom_point(size = 3) +
  geom_smooth(method = "lm", se = FALSE, linewidth = 0.6, linetype = "dashed") +
  facet_wrap(~cohort, scales = "free_x", labeller = cohort_facet_labeller) +
  scale_color_manual(values = cohort_stroke_pal) +
  scale_shape_manual(values = c(Coh1 = 16, Coh2 = 17)) +
  scale_x_continuous(breaks = scales::breaks_width(1)) +
  labs(title = "Colocalization %: each animal's position within its cohort",
       y = "Colocalization %", x = "Animal (ordered within cohort)") +
  theme_thesis

## ---- Combine into a mega-grid ----
fig_cohort_animals <- (fig_cohort_counts / fig_cohort_density / fig_cohort_coloc_pct) +
  plot_annotation(
    title = "Per-animal trend within cohort (processing order check)",
    subtitle = "Descriptive QC plot: no formal test was run due to small n per cohort (Coh1=3, Coh2=4). Underpowered by design.",
    theme = theme(
      plot.title = element_text(face = "bold", hjust = 0.5, size = 20),
      plot.subtitle = element_text(hjust = 0.5, size = 13, color = "grey30"),
      plot.caption = element_text(hjust = 0.5, size = 12, color = "grey45", face = "italic", margin = margin(t = 10, b = 10))
    )
  ) &
  theme(plot.margin = margin(t = 10, r = 20, b = 10, l = 10), panel.spacing = unit(1.1, "lines"))

## ---- Save ----
ggsave(file.path(output_dir, "fig_cohort_per_animal.png"),
       fig_cohort_animals, width = 10, height = 18, dpi = 300)


## ============================================================================
## 8. SAVE EVERYTHING
## ============================================================================

if (!SKIP_SAVING) {
ggsave(file.path(output_dir, "fig2_gfp_cfos_both.png"), fig2, width = 10, height = 6, dpi = 300)  
ggsave(file.path(output_dir, "fig3_overlap_ratios.png"), fig3, width = 7, height = 6, dpi = 300)
ggsave(file.path(output_dir, "fig8_overfitting_check.png"), fig8, width = 12, height = 5.5, dpi = 300)
ggsave(file.path(output_dir, "fig4_real_vs_chance.png"), fig4, width = 6, height = 6, dpi = 300)
ggsave(file.path(output_dir, "fig4b_diff_per_animal.png"), fig4b, width = 6, height = 5, dpi = 300)
ggsave(file.path(output_dir, "fig6_manual_vs_auto_scatter.png"), fig6, width = 14, height = 5.2, dpi = 300)
ggsave(file.path(output_dir, "fig7_bland_altman.png"), fig7, width = 14, height = 5.2, dpi = 300)
ggsave(file.path(output_dir, "fig8_overfitting_check.png"), fig8, width = 14, height = 5.5, dpi = 300)
ggsave(file.path(output_dir, "fig8c_per_image_training_marked.png"), fig8c, width = 14, height = 5.2, dpi = 300)
ggsave(file.path(output_dir, "fig9_rostrocaudal.png"), fig9, width = 10, height = 8, dpi = 300)
ggsave(file.path(output_dir, "qq_gfp_cfos_diff.png"), qq_gfp_cfos_diff, width = 5, height = 5, dpi = 300)
ggsave(file.path(output_dir, "qq_real_chance_diff.png"), qq_real_chance_diff, width = 5, height = 5, dpi = 300)

write_csv(image_data, file.path(output_dir, "image_level_data.csv"))
write_csv(animal_data, file.path(output_dir, "animal_level_data.csv"))
write_csv(key_stats_summary, file.path(output_dir, "key_stats_summary.csv"))
write_csv(rostro_animal, file.path(output_dir, "rostrocaudal_per_animal_bin.csv"))

# ---- Export %GFP and %cFos overlap (Fig 3) ----
# Descriptive only – no p-values or significance stars
# Raw data per animal
animal_data %>%
  mutate(
    pct_of_GFP = manual_overlap_pct_GFP,
    pct_of_cFos = manual_overlap_pct_cFos
  ) %>%
  select(animal_id, cohort, pct_of_GFP, pct_of_cFos) %>%
  write_csv(file.path(output_dir, "colocalization_percentages_descriptive.csv"))

# Optional: Summary statistics (Median + IQR)
summary_stats <- animal_data %>%
  summarise(
    GFP_median = median(manual_overlap_pct_GFP, na.rm = TRUE),
    GFP_IQR = IQR(manual_overlap_pct_GFP, na.rm = TRUE),
    cFos_median = median(manual_overlap_pct_cFos, na.rm = TRUE),
    cFos_IQR = IQR(manual_overlap_pct_cFos, na.rm = TRUE)
  ) %>%
  write_csv(file.path(output_dir, "colocalization_percentages_summary.csv"))

}

# ---- Consolidate every test result into one tidy CSV-able table ----
key_stats_summary <- tribble(
  ~block, ~test,                                  ~statistic,                                   ~p_value,                          ~p_fdr,                            ~effect_size,                ~note,
  # QC: GFP vs cFos (paired Wilcoxon) 
  "A",    "GFP vs cFos (Wilcoxon, QC)",              unname(gfp_vs_cfos_stat),         as.numeric(gfp_vs_cfos_p),           NA_real_,                            r_gfp,             "paired, n=7",
  "A",    "GFP+ counts (descriptive)",   NA_real_,  NA_real_,  NA_real_,  NA_real_,  paste0("mean=", round(gfp_mean,1), ", median=", round(gfp_median,1), " (IQR=", round(gfp_iqr,1), ")"),
  "A",    "cFOS+ counts (descriptive)",  NA_real_,  NA_real_,  NA_real_,  NA_real_,  paste0("mean=", round(cfos_mean,1), ", median=", round(cfos_median,1), " (IQR=", round(cfos_iqr,1), ")"),
  # Colocalized descriptive (no test)
  "A",    "Colocalized (descriptive)",               NA_real_,                                      NA_real_,                            NA_real_,                            NA_real_,                    paste0("mean=", round(coloc_mean,1), ", median=", round(coloc_median,1), " (IQR=", round(coloc_iqr,1), ")"),
  # Confirmatory: Real vs Chance – Block A (no correction, only 1 test)
  "A",    "Real vs Chance overlap (paired t-test)", unname(test_real_vs_chance_t$statistic),       test_real_vs_chance_t$p.value,       NA_real_,                            NA_real_,                    "",
  "A",    "Real vs Chance overlap (Wilcoxon)",      unname(test_real_vs_chance_wc_stat),      as.numeric(test_real_vs_chance_wc_p),            NA_real_,                            r_real,  "rank-biserial r",
  "A",    "Overlap Index (mean across animals)",   mean(animal_data$overlap_index, na.rm = TRUE), NA_real_,                            NA_real_,                            NA_real_,                    "0=chance level, 1=theoretical max",
  "A",    "Chance overlap (descriptive)",               NA_real_,  NA_real_,  NA_real_,  NA_real_,  paste0("mean=", round(chance_mean,1), ", median=", round(chance_median,1), " (IQR=", round(chance_iqr,1), ")"),
  "A",    "Bootstrap 95% CI, real-chance diff",     NA_real_,                                      NA_real_,                            NA_real_,                            NA_real_,                    paste0("[", round(boot_ci[1], 2), ", ", round(boot_ci[2], 2), "]"),
  # Pipeline validation (agreement, no p-values)
  "B",    "Agreement GFP (Lin's CCC + CI)",          ccc_gfp$ccc,                                   NA_real_,                            NA_real_,                            NA_real_,                    paste0("95% CI [", round(ccc_gfp$ci_low, 2), ", ", round(ccc_gfp$ci_high, 2), "]"),
  "B",    "Agreement cFos (Lin's CCC + CI)",         ccc_cfos$ccc,                                  NA_real_,                            NA_real_,                            NA_real_,                    paste0("95% CI [", round(ccc_cfos$ci_low, 2), ", ", round(ccc_cfos$ci_high, 2), "]"),
  "B",    "Agreement Both (Lin's CCC + CI)",         ccc_both$ccc,                                  NA_real_,                            NA_real_,                            NA_real_,                    paste0("95% CI [", round(ccc_both$ci_low, 2), ", ", round(ccc_both$ci_high, 2), "]"),
  "B",    "Agreement GFP, per-animal (Lin's CCC + CI)",   ccc_animal_gfp$ccc,    NA_real_,  NA_real_,  NA_real_,  paste0("95% CI [", round(ccc_animal_gfp$ci_low, 2), ", ", round(ccc_animal_gfp$ci_high, 2), "], rho=", round(rho_animal_gfp, 2)),
  "B",    "Agreement cFos, per-animal (Lin's CCC + CI)",  ccc_animal_cfos$ccc,   NA_real_,  NA_real_,  NA_real_,  paste0("95% CI [", round(ccc_animal_cfos$ci_low, 2), ", ", round(ccc_animal_cfos$ci_high, 2), "], rho=", round(rho_animal_cfos, 2)),
  "B",    "Agreement Both, per-animal (Lin's CCC + CI)",  ccc_animal_both$ccc,   NA_real_,  NA_real_,  NA_real_,  paste0("95% CI [", round(ccc_animal_both$ci_low, 2), ", ", round(ccc_animal_both$ci_high, 2), "], rho=", round(rho_animal_both, 2)),
  "B",    "Bland-Altman bias GFP",                   ba_gfp$mean_diff,                             NA_real_,                            NA_real_,                            NA_real_,                    paste0("SD=", round(ba_gfp$sd_diff,2), ", 95% LoA [", round(ba_gfp$loa_lower,2), ", ", round(ba_gfp$loa_upper,2), "]"),
  "B",    "Bland-Altman bias cFos",                  ba_cfos$mean_diff,                            NA_real_,                            NA_real_,                            NA_real_,                    paste0("SD=", round(ba_cfos$sd_diff,2), ", 95% LoA [", round(ba_cfos$loa_lower,2), ", ", round(ba_cfos$loa_upper,2), "]"),
  "B",    "Bland-Altman bias Both",                  ba_both$mean_diff,                            NA_real_,                            NA_real_,                            NA_real_,                    paste0("SD=", round(ba_both$sd_diff,2), ", 95% LoA [", round(ba_both$loa_lower,2), ", ", round(ba_both$loa_upper,2), "]"),
   # Pipeline validation (FDR-corrected p-values) – convert all coun p-values with as.numeric()
  "B",    "Overfitting GFP",                         NA_real_,   as.numeric(training_p_gfp),   pipeline_p_fdr[1],   NA_real_,   paste0("exploratory, FDR; median Training=", round(overfit_gfp$median_train,1), ", Held-out=", round(overfit_gfp$median_heldout,1)),
  "B",    "Overfitting cFos",                        NA_real_,   as.numeric(training_p_cfos),  pipeline_p_fdr[2],   NA_real_,   paste0("exploratory, FDR; median Training=", round(overfit_cfos$median_train,1), ", Held-out=", round(overfit_cfos$median_heldout,1)),
  "B",    "Batch GFP",                               NA_real_,                                      as.numeric(batch_test_gfp),          pipeline_p_fdr[3],                   NA_real_,                    paste0("exploratory, FDR; median Coh1=", round(batch_gfp$median_coh1,1), ", Coh2=", round(batch_gfp$median_coh2,1)),
  "B",    "Batch cFos",                              NA_real_,                                      as.numeric(batch_test_cfos),         pipeline_p_fdr[4],                   NA_real_,                    paste0("exploratory, FDR; median Coh1=", round(batch_cfos$median_coh1,1), ", Coh2=", round(batch_cfos$median_coh2,1)),
  "B",    "Overfitting GFP, mean |error|",  NA_real_,  NA_real_,  NA_real_,  NA_real_,  paste0("Training=", round(mean_abs_error_gfp_train,2), ", Held-out=", round(mean_abs_error_gfp_heldout,2)),
  "B",    "Overfitting cFos, mean |error|", NA_real_,  NA_real_,  NA_real_,  NA_real_,  paste0("Training=", round(mean_abs_error_cfos_train,2), ", Held-out=", round(mean_abs_error_cfos_heldout,2)),
  )


# ---- Print summary to console and log file ----
stats_log_con <- file(file.path(output_dir, "statistical_tests_console_log.txt"), open = "wt")
sink(stats_log_con, split = TRUE)

cat("\n==== KEY STATISTICAL RESULTS ====\n")

cat("\n--- GFP vs cFos (labelling, QC) ---\n")
print(gfp_vs_cfos_test)
cat("Effect size (rank-biserial r):", round(r_gfp, 3), "\n")
cat("\n--- Colocalized counts (descriptive) ---\n")
cat("mean =", round(coloc_mean, 1), ", median =", round(coloc_median, 1), ", IQR =", round(coloc_iqr, 1), "\n")

cat("\n--- Real vs Chance overlap (paired, per animal) ---\n")
print(test_real_vs_chance_t)
print(test_real_vs_chance_wc)
cat("Effect size (rank-biserial r):", round(eff_real_vs_chance$effsize, 3), "\n")
cat("Overlap Index (mean across animals):", round(mean(animal_data$overlap_index, na.rm = TRUE), 3), "\n")
cat("Bootstrap 95% CI on real-chance difference: [", round(boot_ci[1], 1), ",", round(boot_ci[2], 1), "]\n")

cat("\n--- Manual vs automatic agreement (image level, acceptance: CCC > 0.90) ---\n")
cat("GFP total:  Lin's CCC =", round(ccc_gfp$ccc, 3), " (95% CI [", round(ccc_gfp$ci_low, 3), ", ", round(ccc_gfp$ci_high, 3), "])",
    " | Spearman rho =", round(cor_gfp_sp$estimate, 3), "\n")
cat("cFos total: Lin's CCC =", round(ccc_cfos$ccc, 3), " (95% CI [", round(ccc_cfos$ci_low, 3), ", ", round(ccc_cfos$ci_high, 3), "])",
    " | Spearman rho =", round(cor_cfos_sp$estimate, 3), "\n")
cat("Both:       Lin's CCC =", round(ccc_both$ccc, 3), " (95% CI [", round(ccc_both$ci_low, 3), ", ", round(ccc_both$ci_high, 3), "])",
    " | Spearman rho =", round(cor_both_sp$estimate, 3), "\n")

cat("\n--- Overfitting check (Training vs. Held-out) ---\n")
cat("GFP delta (Auto - Manual) on Training vs. Held-out:\n")
print(training_test_gfp)
cat("p-value:", training_p_gfp, "\n")

cat("cFos delta (Auto - Manual) on Training vs. Held-out:\n")
print(training_test_cfos)
cat("p-value:", training_p_cfos, "\n")

cat("\n--- Pipeline validation (FDR-corrected) ---\n")
cat("Overfitting GFP: raw p =", signif(training_p_gfp, 3), ", FDR =", signif(pipeline_p_fdr[1], 3), "\n")
cat("Overfitting cFos: raw p =", signif(training_p_cfos, 3), ", FDR =", signif(pipeline_p_fdr[2], 3), "\n")
cat("Batch GFP: raw p =", signif(batch_test_gfp, 3), ", FDR =", signif(pipeline_p_fdr[3], 3), "\n")
cat("Batch cFos: raw p =", signif(batch_test_cfos, 3), ", FDR =", signif(pipeline_p_fdr[4], 3), "\n")
cat("Batch GFP medians: Coh1 =", batch_gfp$median_coh1, ", Coh2 =", batch_gfp$median_coh2, "\n")
cat("Batch cFos medians: Coh1 =", batch_cfos$median_coh1, ", Coh2 =", batch_cfos$median_coh2, "\n")

sink()
close(stats_log_con)

message("Done. All plots and processed data tables saved to: ", output_dir)
