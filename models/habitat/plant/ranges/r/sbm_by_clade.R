# Daru (2024) step 4 at clade level: castor::fit_sbm_const fitted per family (>= min_tips located tips) and
# once for the whole tree; each diffusivity D (km^2/Myr) is converted to the expected great-circle distance
# travelled in one year (castor 1.8.4 expected_distances_sbm; Daru names its predecessor expected_SBM_distance) and over 1,000 years for scale.
suppressPackageStartupMessages({library(castor); library(ape)})
args <- commandArgs(trailingOnly = TRUE)       # tree.nwk tips.csv out.csv [min_tips]
tree <- read.tree(args[1]); tips <- read.csv(args[2], stringsAsFactors = FALSE)
min_tips <- if (length(args) >= 4) as.integer(args[4]) else 10L
min_edge <- if (length(args) >= 5) as.numeric(args[5]) else 0.1        # Myr
tips <- tips[!is.na(tips$latitude) & tips$tip %in% tree$tip.label, ]
tips <- tips[!duplicated(tips$tip), ]
fit_one <- function(labels, clade) {
  sub <- keep.tip(tree, labels)
  sub$edge.length[sub$edge.length < min_edge] <- min_edge   # dating resolution floor (see provenance)
  tt <- tips[match(sub$tip.label, tips$tip), ]
  f <- tryCatch(fit_sbm_const(sub, tip_latitudes = tt$latitude, tip_longitudes = tt$longitude, radius = 6371),
                error = function(e) NULL)
  if (is.null(f) || !isTRUE(f$success)) return(NULL)
  D <- f$diffusivity
  data.frame(clade = clade, n_tips = length(labels), diffusivity_km2_per_Myr = D, loglik = f$loglikelihood,
             km_per_year = expected_distances_sbm(diffusivity = D, radius = 6371, deltas = 1e-6),
             km_per_kyr = expected_distances_sbm(diffusivity = D, radius = 6371, deltas = 1e-3),
             km_per_Myr = expected_distances_sbm(diffusivity = D, radius = 6371, deltas = 1))
}
fams <- table(tips$family); fams <- names(sort(fams[fams >= min_tips]))
first <- TRUE
emit <- function(r) {                      # append each fit as it completes
  if (is.null(r)) return(invisible())
  write.table(r, args[3], sep = ",", row.names = FALSE, col.names = first, append = !first)
  first <<- FALSE
}
for (fm in fams) emit(fit_one(tips$tip[tips$family == fm], fm))
emit(fit_one(tips$tip, "ALL"))            # whole tree last: slowest fit
cat("done\n")
