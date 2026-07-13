#!/usr/bin/env Rscript
# Dated patristic distances for the pollinator clades that have published chronograms (bees, ants, butterflies,
# hawkmoths). For each tree: normalize tips to binomial, prune to one tip per vocab-covered binomial, cophenetic()
# on the DATED branch lengths (Myr). Output <clade>_cophen.csv (binomial-labelled). Python then rescales each block
# and overwrites the within-clade entries of pollinator_distance.npy (replacing the topological approximation).
suppressMessages(library(ape))
T <- "/home/photon/4tb/deepcal_data/trees/"
OUT <- "/home/photon/4tb/deepcal_data/trees/dated_cophen/"
dir.create(OUT, showWarnings = FALSE)
vocab <- read.csv("deepearth/data/deepcal/pollinator/pollinator_vocab.csv", header = FALSE, stringsAsFactors = FALSE)
vset <- unique(tolower(gsub(" ", "_", vocab$V2[grepl("^[0-9]+$", vocab$V1)])))   # "genus_species"

binom <- function(labs) sapply(strsplit(labs, "_"), function(x) paste(tolower(x[1]), tolower(x[2]), sep = "_"))

clades <- list(
  bees        = "BEE_mat7_fulltree_tplo35_sf20lp.nwk",
  ants        = "ants_Nelsen2018.tre",
  butterflies = "butterflies_Kawahara2023.tre",
  moths       = "sphingidae_Couch2026.tre")

for (nm in names(clades)) {
  f <- paste0(T, clades[[nm]])
  tr <- tryCatch(read.tree(f), error = function(e) read.nexus(f))
  bn <- binom(tr$tip.label)
  keep <- bn %in% vset & !duplicated(bn)                       # one tip per vocab-covered binomial
  if (sum(keep) < 3) { cat(nm, ": <3 covered, skip\n"); next }
  sub <- keep.tip(tr, tr$tip.label[keep])
  sub$tip.label <- binom(sub$tip.label)                       # relabel tips to binomial
  cph <- cophenetic(sub)                                       # DATED patristic (Myr)
  write.csv(cph, paste0(OUT, nm, "_cophen.csv"))
  offdiag <- cph[upper.tri(cph)]
  cat(sprintf("%-12s covered=%d  mean_dated=%.1f Myr  min_congener=%.2f  max=%.1f\n",
              nm, nrow(cph), mean(offdiag), min(offdiag), max(offdiag)))
}
cat("DATED_PATRISTIC_DONE\n")
