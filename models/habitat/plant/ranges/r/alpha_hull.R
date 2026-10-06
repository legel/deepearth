# Daru (2024) step 3: rangeBuilder alpha hull, initialAlpha = 2, alpha increased until >= 99% of records are
# enclosed; all other parameters at their defaults. Writes the hull as GeoJSON (EPSG:4326).
suppressPackageStartupMessages({library(rangeBuilder); library(sf)})
args <- commandArgs(trailingOnly = TRUE)
# optional third argument "planar": run sf on GEOS instead of s2 (used only after s2 rejects a hull loop)
if (length(args) >= 3 && args[3] == "planar") sf_use_s2(FALSE)
x <- read.csv(args[1])
h <- getDynamicAlphaHull(x[, c("decimalLongitude", "decimalLatitude")], fraction = 0.99, initialAlpha = 2,
                         coordHeaders = c("decimalLongitude", "decimalLatitude"), verbose = FALSE)
st_write(st_as_sf(h[[1]]), args[2], driver = "GeoJSON", delete_dsn = TRUE, quiet = TRUE)
cat("alpha", h$alpha, "\n")
