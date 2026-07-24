#!/usr/bin/env Rscript
# Real DATED patristic distances for the PLANT species graph (rules 7-12): cophenetic (Myr) on ca_subtree.dated.nwk
# over the model species that ARE tree tips (~65%; the rest are inductively placed, rules 25/26, and keep the
# BioCLIP-embedding shadow at model-build time in fusion.py). Reads derived/model_species.csv (model_idx,tip_label)
# written by plant_dated_distance.py; writes derived/plant_cophen.csv (tip_label-labelled). Mirrors the pollinator
# dated pipeline. Replaces the crude distance_from_embedding shadow the ou-attention graph used for ALL species.
suppressMessages(library(ape))
cache <- "deepearth/data/deepcal"
tr  <- read.tree(file.path(cache, "ca_subtree.dated.nwk"))
ms  <- read.csv(file.path(cache, "derived/model_species.csv"), stringsAsFactors = FALSE)   # model_idx, tip_label
covered <- intersect(ms$tip_label, tr$tip.label)
cat("model species:", nrow(ms), " tree tips:", length(tr$tip.label), " covered:", length(covered), "\n")
sub <- keep.tip(tr, covered)
cph <- cophenetic(sub)                                          # DATED patristic (Myr) over covered model species
write.csv(cph, file.path(cache, "derived/plant_cophen.csv"))
od <- cph[upper.tri(cph)]
cat(sprintf("plant dated cophenetic: %d taxa | mean %.1f Myr | min %.3f | max %.1f\n",
            nrow(cph), mean(od), min(od[od > 0]), max(od)))
cat("PLANT_DATED_PATRISTIC_DONE\n")
