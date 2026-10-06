# Daru (2024) step 2b: CoordinateCleaner::clean_coordinates with default tests plus duplicates,
# institution radius 100 m. Input: CSV with columns species, decimalLongitude, decimalLatitude and optionally
# group. Output: the flags CSV (column .summary = record kept).
# With a group column, each group (one species' record set) is cleaned by its own clean_coordinates call inside
# this one R session, which gives exactly the result of one session per group: the outlier test, and its switch
# to a raster approximation when any record set has >= 10,000 records, see only that group's records. Starting R
# and loading the packages and reference layers once instead of per group saves ~10 s per group.
suppressPackageStartupMessages(library(CoordinateCleaner))
args <- commandArgs(trailingOnly = TRUE)
x <- read.csv(args[1], stringsAsFactors = FALSE)
tests <- c("capitals", "centroids", "equal", "gbif", "institutions", "outliers", "seas", "zeros", "duplicates")
# Sea test reference: CoordinateCleaner's default is Natural Earth 50 m land, downloaded at run time through
# rnaturalearth (fails with newer rnaturalearth: "unused argument"). CC_SEAS_REF points at a local copy of the same
# layer (raw/geo/ne_50m_land); without it the package default is used.
ref <- Sys.getenv("CC_SEAS_REF")
seas_ref <- if (nzchar(ref) && file.exists(ref)) terra::vect(ref) else NULL
clean <- function(d) clean_coordinates(d, lon = "decimalLongitude", lat = "decimalLatitude", species = "species",
                                       tests = tests, inst_rad = 100, seas_ref = seas_ref, value = "spatialvalid",
                                       verbose = FALSE)
if ("group" %in% names(x)) {
  kept <- logical(nrow(x))
  for (g in unique(x$group)) {
    i <- which(x$group == g)
    d <- x[i, c("species", "decimalLongitude", "decimalLatitude")]
    rownames(d) <- NULL
    kept[i] <- clean(d)$.summary
  }
  flags <- data.frame(.summary = kept, check.names = FALSE)
} else {
  flags <- clean(x)
}
write.csv(flags, args[2], row.names = FALSE)
cat("records", nrow(x), "kept", sum(flags$.summary), "\n")
