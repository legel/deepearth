# Plant

The probability that each 1 m cell of a site is habitat for each species of its regional flora. One absolute
probability, the same for the same inputs on any block of any site: no site mean, palette rank or block enters it.

## Equations

| quantity | equation | code |
|---|---|---|
| national intensity | $\ln \lambda_s(x) = \ln\ln 2 + g_s(x) - \mathrm{P5}_s$ at every 240 m cell under the site and 3 cells around it, $g_s$ the native habitat model's served score ([`../habitat/plant`](../habitat/plant)), $\mathrm{P5}_s$ its range threshold, so $1 - e^{-\lambda} = 0.5$ at the range edge; cells without national climate (open water) filled from their neighbours, bilinear to 1 m | [`prob_pass.py` `intensity`](prob_pass.py#L237), [`fill_gaps`](prob_pass.py#L241), [`bilinear`](prob_pass.py#L266) |
| probability | $\mathrm{cloglog}\, P_s(q) = \ln\lambda_s(q) + \left(G_s(q) - \log\overline{e^{G_s}}\right) + L + W + S + \mathrm{salt} + \mathrm{flood} + \log A + \log C$, $P = 1 - \exp(-e^{x})$; $G_s$ the E3DT site model's habitat term taken relative to its own 240 m cell; every other term a log at most 0 | [`prob_pass.py` `run`](prob_pass.py#L407) |
| trait windows | $\log\sigma\!\left(\dfrac{v - lo}{w}\right) + \log\sigma\!\left(\dfrac{hi - v}{w}\right)$: light share, $w$ 0.08; water $2.5 \times$ the cell's soil moisture (0 to 2) on the trait's 0 to 5, $w$ 0.5, or the drainage index where there is no hydrology, $w$ 0.15; pH $w$ 0.3; texture class $w$ 0.3; 0 where the value or the trait is unknown | [`window`](prob_pass.py#L69), [`run`](prob_pass.py#L384) |
| salinity | $\log \sum_k P(\mathrm{salt} = k)\, e^{-\max(0,\, \log_2 \mathrm{EC} - t_k)/0.5}$, tops $t_k = \log_2$ 2, 4, 8 dS/m for tolerance none, low, moderate (USDA PLANTS), none for high; intertidal ground without a survey EC reads 8 dS/m | [`hinge_mix`](prob_pass.py#L106), [`salinity_ec`](prob_pass.py#L118) |
| flooding | months saturated $m$ = the survey's months with the water table within 30 cm, at least 1 under frequent flooding or ponding, 12 under very frequent flooding; $\log \sum_k P(\mathrm{flood} = k)\, e^{-\max(0,\, m - t_k)/w}$, tops 0.5, 1 and 3 months (after Whitlow and Harris 1979), $w$ 0.5, 0.375, 0.25 months for shallow, moderate, deep roots | [`months_saturated`](prob_pass.py#L129), [`condition_terms`](prob_pass.py#L173) |
| seasonal water | $W = \log\sigma\!\left(\dfrac{2.5\, d - lo}{0.5}\right) + \log\sigma\!\left(\dfrac{hi - 2.5\, r_{wet}}{0.5}\right)$, $d = P_{sh}\, r_{dry} + P_{mod}\, \tfrac12(r + r_{dry}) + P_{deep}\, r$: the driest season against the species' low end, the wettest against its high end | [`condition_terms`](prob_pass.py#L203) |
| obligate aquatics | $P = 0$ off open water (moisture code 253, NWI subtidal) and off ground saturated 10 months a year or more | [`aquatic_allowed`](prob_pass.py#L143), [`aquatic_zero`](prob_pass.py#L166) |
| which species are aquatic | every alternative of the species' WCVP life form a hydrophyte form (Raunkiaer); a species WCVP leaves blank or calls a helophyte takes its genus's GIFT Life_form_2 vote where at least 5 species are coded and at least 80 % are hydrophytes; a sourced terrestrial habit, or a climber not obligately wetland, takes a species out | [`aquatic_wcvp.py` `aquatic_form`](aquatic_wcvp.py#L71), [`by_genus`](aquatic_wcvp.py#L65), [`exclusion`](aquatic_wcvp.py#L102) |
| habitable | $P \ge 0.5$ on plantable ground; an obligate aquatic on its allowed cells | [`aquatic_gate`](prob_pass.py#L155), [`HAB_P`](prob_pass.py#L45) |
| ranking | a species' site value: the 90th percentile of $100P$ over the block's plantable ground; at a cell, the species ordered by $P$ there (the top 25 kept for every cell with a LiDAR return) | [`run`](prob_pass.py#L440), [`run`](prob_pass.py#L428) |

The trait windows' bounds and the categorical traits (salt and flood tolerance, root depth) are the national trait
posteriors' (22,032 species); salt tolerance is observed for 1,916 of them and predicted for the rest, flood tolerance
for 4,653, root depth for 1,086.

## The aquatic list

[`aquatic_wcvp.json`](aquatic_wcvp.json): 1,119 accepted species and 2,376 synonyms, from WCVP (Kew, World Checklist of
Vascular Plants) with GIFT's genus vote (Weigelt et al. 2020) adding 133 species in Limnophila, Limosella, Marsilea,
Myriophyllum, Nymphaea and Nymphoides; 10 WCVP codings excluded, each with its reason in the file.

## Known errors

- The class tops are published class definitions; the decay widths are set by hand, with no fit to occurrence data.
- Where the survey has no value (Urban land), the salt and flood terms are 0.
- The seasonal swing is the top 5 cm's (SMAP-HydroBlocks), wider than the root zone's; the root-depth blend narrows it.
- Emergent wetland plants (helophytes) are not gated.

## Layout

`prob_pass.py` is the producer's step that makes the probability. It imports the producer's site reader, palette and
job modules and the E3DT site model, which are not part of this export: the file is published for its equations.
