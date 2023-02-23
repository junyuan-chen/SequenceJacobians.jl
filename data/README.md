# Example Data

Data files are provided here for the ease of testing and illustrations.
The data may have been processed for the ease of usage
and are stored in compressed (`.gz`) files.
See [`data/src/make.jl`](src/make.jl) for the source code
that generates these files from original data.

## Sources

| Name | Source | File | Note |
| :---: | :----: | :-------: | :--- |
| `bayes.csv.gz` | [Auclert et al. (2021)](https://doi.org/10.3982/ECTA17434) | `import_export/data/data_bayes.csv` | |
| `sw.csv.gz` | [Smets and Wouters (2007)](https://doi.org/10.3886/E116269V1) | `usmodel_data.xls` | Formatted file from [Auclert et al. (2021)](https://doi.org/10.3982/ECTA17434) `import_export/data/data_sw.csv` |
| `vlw.json.gz` | [vom Lehn and Winberry (2021)](https://doi.org/10.7910/DVN/CALDHX) | `Model Analysis/modelparm_37sec.mat`; `Model Analysis/inddat_TFP_37sec.mat` | |

## References

**Auclert, Adrien, Bence Bardóczy, Matthew Rognlie, and Ludwig Straub.** 2021.
"Supplement to 'Using the Sequence-Space Jacobian to Solve and Estimate Heterogeneous-Agent Models'."
*Econometrica Supplemental Material*, 89, https://doi.org/10.3982/ECTA17434.

**Smets, Frank, and Rafael Wouters.** 2007.
"Replication Data for: 'Shocks and Frictions in US Business Cycles: A Bayesian DSGE Approach'."
*American Economic Association* [publisher],
Inter-university Consortium for Political and Social Research [distributor],
https://doi.org/10.3886/E116269V1.

**vom Lehn, Christian, and Thomas Winberry.** 2021.
"Replication Data for: 'The Investment Network, Sectoral Comovement, and the Changing U.S. Business Cycle'."
*The Quarterly Journal of Economics Dataverse*, V1, https://doi.org/10.7910/DVN/CALDHX.
