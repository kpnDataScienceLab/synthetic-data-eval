# Synthetic data evaluation

This repository contains the script for the evaluation of the viewership synthetic data. 

## Details
The script takes as an input the original and the synthetic datasets. There are in total 5 metrics used in order to evaluate whether the synthetic dataset preserves the patterns and the characteristics of the original one.
The used metrics are:
- The Correlation (Euclidean) distance
- The two-sample Kolmogorov-Smirnov test
- The Jensen-Shannon divergence
- The Kullback-Leibler (KL) divergence
- The pairwise correlation difference (PCD)

### The Correlation (Euclidean) distance
Having calculated the correlation matrices within the attributes of the original dataset and the
attributes of the generated dataset, a suitable way to measure the similarity of these is to
calculate the sum of their pairwise euclidean distances -i.e. the sum of the euclidean distances
of every X ij and Y ij of the correlations matrices X and Y. These results are a suitable way to
measure the preservation of the intrinsic patterns occurring between the attributes of the original
dataset in the new synthetic dataset. The lower this metric is, the better the data generation tool
preserves the patterns.

### The two-sample Kolmogorov-Smirnov test
The two-sample Kolmogorov-Smirnov test is used to test whether two samples come from the same distribution.
The level of significance a is set as a = 0.05. If the generated p-value from the test is lower than a then it is probable that the two distributions are different. The threshold limit for this function is a list containing less than 10 elements.

### The Jensen-Shannon divergence
The Jensen-Shannon divergence, or JS divergence for short, is another way to quantify the difference (or similarity) between two probability distributions.
It uses the KL divergence to calculate a normalized score that is symmetrical. It is more useful as a measure as it provides a smoothed and normalized version of KL divergence, with scores between 0 (identical) and 1 (maximally different), when using the base-2 logarithm.

### The Kullback-Leibler (KL) divergence
The KL divergence, also called relative entropy, is computed over a pair of real and synthetic
marginal probability mass functions (PMF) for a given variable, and it measures the similarity of
the two PMFs. When both distributions are identical, the KL divergence is zero, while larger
values of the KL divergence indicate a larger discrepancy between the two PMFs. Note that the
KL divergence is computed for each variable independently; therefore, it does not measure
dependencies among the variables. Note that the KL divergence is defined at the variable level,
not over the entire dataset.

### The pairwise correlation difference (PCD)
PCD is intended to measure how much correlation among the variables the different methods
were able to capture. PCD measures the
difference in terms of Frobennius norm of the Pearson correlation matrices computed from real
and synthetic datasets. The smaller the PCD, the closer the synthetic data is to the real data in
terms of linear correlations across the variables. PCD is defined at the dataset level.