# Marginal distribution sampling (Reichstein et al. 2005) through REddyProc (Wutzler et al. 2018), for FluxFill-Bench.
# Usage: Rscript mds.R <in.csv> <out.csv> <lat> <lon> <utc_offset_h> <var> [<var> ...]
# in.csv: DateTime (local standard time, end of half-hour, "%Y-%m-%d %H:%M"), Rg, Tair, VPD (drivers, already
# filled with ERA5 where the tower value is hidden or missing), and one column per variable to fill with NA on
# every hidden value. out.csv: for each variable, <var>_f (filled) and <var>_fsd (standard deviation of the
# values MDS averaged), the MDS uncertainty REddyProc reports.
suppressMessages(library(REddyProc))
a <- commandArgs(trailingOnly = TRUE)
d <- read.csv(a[1], stringsAsFactors = FALSE)
d$DateTime <- as.POSIXct(d$DateTime, format = "%Y-%m-%d %H:%M", tz = "GMT")
vars <- a[6:length(a)]
ep <- sEddyProc$new("SITE", d, c(vars, "Rg", "Tair", "VPD"))
ep$sSetLocationInfo(LatDeg = as.numeric(a[3]), LongDeg = as.numeric(a[4]), TimeZoneHour = as.numeric(a[5]))
out <- data.frame(DateTime = format(d$DateTime, "%Y-%m-%d %H:%M"))
for (v in vars) {
  ep$sMDSGapFill(v, FillAll = TRUE, isVerbose = FALSE)
  r <- ep$sExportResults()
  out[[paste0(v, "_f")]] <- r[[paste0(v, "_f")]]
  out[[paste0(v, "_fsd")]] <- r[[paste0(v, "_fsd")]]
}
write.csv(out, a[2], row.names = FALSE)
