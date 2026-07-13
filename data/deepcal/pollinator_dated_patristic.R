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
gen   <- function(x) sub("_.*", "", x)                        # genus part of a "genus_species" string

clades <- list(
  bees        = "BEE_mat7_fulltree_tplo35_sf20lp.nwk",
  ants        = "ants_Nelsen2018.tre",
  butterflies = "butterflies_Kawahara2023.tre",
  moths       = "sphingidae_Couch2026.tre",
  birds       = "hummingbirds_datelife_McGuire2014.nwk")   # datelife median chronogram (OpenTree store, McGuire2014); the raw hummingbirds.nwk had NO branch lengths

vgen <- gen(vset)                                             # genus of each vocab species

for (nm in names(clades)) {
  f <- paste0(T, clades[[nm]])
  tr <- tryCatch(read.tree(f), error = function(e) read.nexus(f))
  tbin <- binom(tr$tip.label); tgen <- gen(tbin)             # per-tip binomial + genus
  # HYBRID placement (matches the V.PhyloMaker plant tree): species-level distance where an exact tip exists,
  # else the species is placed at its GENUS crown (genus-level dated position) — lifts coverage ~10x for
  # groups whose chronogram samples ~1 species/genus (e.g. Kawahara2023 butterflies).
  sp <- vset[vgen %in% tgen]                                  # every vocab species whose genus is in the tree
  if (length(sp) < 3) { cat(nm, ": <3 covered, skip\n"); next }
  gi <- gen(sp)
  gcov <- unique(gi)                                          # one representative tip per covered genus, preferring an exact vocab tip
  rep_tip <- sapply(gcov, function(g) { idx <- which(tgen == g); ex <- idx[tbin[idx] %in% vset]
                                        tr$tip.label[if (length(ex)) ex[1] else idx[1]] })
  gt <- keep.tip(tr, rep_tip); gt$tip.label <- gcov[match(gt$tip.label, rep_tip)]
  Cg <- cophenetic(gt)                                        # genus-level dated patristic (Myr); Cg[g,g]=0
  D  <- Cg[gi, gi]; dimnames(D) <- list(sp, sp)              # expand to species (congeners -> 0 unless refined below)
  keep <- tbin %in% vset & !duplicated(tbin)                  # species-level exact tips
  nex <- sum(keep)
  if (nex >= 2) {                                             # overwrite exact-match block with true species-level distances
    st <- keep.tip(tr, tr$tip.label[keep]); st$tip.label <- binom(st$tip.label)
    Csp <- cophenetic(st); ex <- intersect(sp, rownames(Csp)); D[ex, ex] <- Csp[ex, ex]
  }
  write.csv(D, paste0(OUT, nm, "_cophen.csv"))
  offdiag <- D[upper.tri(D)]
  cat(sprintf("%-12s covered=%d (exact=%d, genus-placed=%d)  mean_dated=%.1f Myr  max=%.1f\n",
              nm, nrow(D), nex, nrow(D) - nex, mean(offdiag), max(offdiag)))
}
cat("DATED_PATRISTIC_DONE\n")
