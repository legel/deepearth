PRE-REGISTRATION, written before the runs below exist.

N = 3 SEEDS (1337, 1338, 1339) for every arm and every control. I judge ONCE at N=3 and stop.
If a candidate regresses at N=3 it FAILS. I will not add a fourth seed to widen a floor, because
adding seeds until a floor covers a regression is tuning to pass, which the contract forbids.

WHY N=3 and not more: three is what the n_latents=24 control already has, and the third seed was run
to test whether the two-seed floor was under-estimated -- which it demonstrably was, clay 44x. The
count is therefore fixed by an instrument check, not by any candidate's result. More would be better
and I am saying so rather than doing it, precisely so the count cannot be read as chosen to fit.

RUNS NEEDED: screen_l24_calibrated seed 1339; screen_l24_combined seed 1339; screen seed 1339;
screen_calibrated seed 1339. Controls and arms then all sit at three seeds.

JUDGED ON: macro against the three-seed macro floor of its own control family, and every one of the
21 per-variable deltas against three-seed per-variable floors. Both families judged the same way:
n_latents 16 (the incumbent champion) and n_latents 24.

PREDICTION, recorded now: the incumbent (continuous_calibration at n_latents 16) stays clean, since
its two-seed floors were generous rather than collapsed. calibration-alone at 24 stays clean. The
combined arm at 24 is the uncertain one -- at two seeds it was clay 1.17x and water 1.2x, both just
over, and three-seed floors may or may not cover them. I am not predicting which.
