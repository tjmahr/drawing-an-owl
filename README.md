Latent class mixed effects models
================

<!-- README.md is generated from README.Rmd. Please edit that file -->

## Background

Proust-Lima, Philipps, and Liquet (2017) describe the statistical
machinery for their latent class mixed models. We note the following
equations from their paper.

Below is the likelihood contribution for one individual in the basic
linear mixed model that we know and love:

$$
\displaylines{
L_i = \phi_i(Y_i; \theta_1) \\
\phi: \textrm{MVN density} \\
i: \textrm{individual index}
}
$$

When there are $G$ latent classes, the likelihood becomes a weighted sum
of class-specific likelihoods:

$$
\displaylines{
L_i(\theta_G) = \sum_{g=1}^G \pi_{ig}\phi_{ig}(Y_i|c_i = g; \theta_G) \\
\pi_{ig}: \textrm{probability of group membership for an individual} \\
g: \textrm{group index}
}
$$

So, an individual makes $G$ contributions to the likelihood and each one
is weighted by their group membership probability. And when there is one
group, this equation reduces to the first likelihood equation.

Finally, group probabilities are defined as a multinomial logistic
model:

\$\$ \\ c_i : \\ X\_{ci} : \\ *{0g} +X^{}*{ci}\_{1g} :

\$\$

## A random intercept model

Repeated measuresments within each individual. Each group has their own
mean. There is one population for the varying effects.

``` r
individuals <- 1:40
n_obs <- 200
groups <- 3
overall_sigma <- 1

individual_to_group <- sample(seq_len(groups), length(individuals), replace = TRUE)
observation_to_individual <- sample(individuals, n_obs, replace = TRUE)
observation_to_group <- individual_to_group[observation_to_individual]

group_population_means <- seq_len(groups) * 2
group_population_sigmas <- rep(.5, groups)

individual_means <- rnorm(
  length(individuals),
  group_population_means[individual_to_group],
  group_population_sigmas[individual_to_group]
)

observations <- rnorm(
  n_obs,
  individual_means[observation_to_individual], 
  overall_sigma
)

library(tidyverse)
d <- tibble(
  y = observations,
  i = observation_to_individual,
  ci = observation_to_group
)

ggplot(d) + 
  aes(x = i, y = y) + 
  geom_point(aes(color = factor(ci))) +
  stat_smooth(
    aes(color = factor(ci)),
    data = function(x) { 
      x |> 
        group_by(ci, i) |> 
        summarise(y = mean(y), .groups = "drop")
    },
    method = "lm",
    formula = y ~ 1
  ) +
  guides(color = "none")
```

<img src="man/figures/README-unnamed-chunk-2-1.png" width="50%" />

``` r

d |> 
  group_by(ci) |> 
  summarise(mean(y), sd(y))
#> # A tibble: 3 × 3
#>      ci `mean(y)` `sd(y)`
#>   <int>     <dbl>   <dbl>
#> 1     1      2.04   0.941
#> 2     2      4.05   1.13 
#> 3     3      5.86   1.01
```

## References

<div id="refs" class="references csl-bib-body hanging-indent">

<div id="ref-JSSv078i02" class="csl-entry">

Proust-Lima, Cécile, Viviane Philipps, and Benoit Liquet. 2017.
“Estimation of Extended Mixed Models Using Latent Classes and Latent
Processes: The r Package Lcmm.” *Journal of Statistical Software* 78
(2): 1–56. <https://doi.org/10.18637/jss.v078.i02>.

</div>

</div>
