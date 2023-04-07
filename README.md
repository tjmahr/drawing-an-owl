Latent class mixed effects models
================

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

$$
\displaylines{
\pi_{ig} = 
  P(c_i = g | X_{ci}) = 
  \frac{
    e^{\xi_{0g} +X^{\top}_{ci}\xi_{1g}}
  }{
    \sum^{G}_{l=1}e^{\xi_{0l} +X^{\top}_{ci}\xi_{1l}}
  } \\
c_i : \textrm{the latent class for an individual} \\
X_{ci} : \textrm{time-indpendent covariates} \\
\xi_{0g} +X^{\top}_{ci}\xi_{1g} : \textrm{linear model for group membership}
}
$$

## sketch

- stan code for a random-intercept with no latent groups model
- update said model to have latent groups

Simulate some repeated measures data. There are no latent group effects
yet. Happy with the idea of storing stuff in dataframe and joining at
the end, not happy with the verbosity.

``` r
library(tidyverse)
rep_along <- function (along, x) rep_len(x, length(along))

simulate_data <- function(
  n_individuals = 20,
  n_obs = 100,
  n_groups = 3,
  sigma_y = .2,
  group_mean = NULL,
  group_sigma = NULL
) {
  if (is.null(group_mean)) {
    group_mean <- rep_len(0, n_groups)
    group_sigma <- rep_len(1, n_groups)
  }
  if (is.null(group_sigma)) {
    group_sigma <- rep_len(1, n_groups)
  }
  
  d_groups <- data.frame(
    group = seq_len(n_groups),
    group_mean = group_mean,
    group_sigma = group_sigma
  )
  
  d_observations <- data.frame(
    y = rep_len(NA, n_obs),
    individual = NA_integer_
  )
  
  d_individuals <- data.frame(
    individual = seq_len(n_individuals),
    group = NA_integer_,
    individual_mean = NA_real_
  )

  d_individuals$group <- sample(
    seq_len(n_groups), 
    n_individuals, 
    replace = TRUE
  )

  d_observations$individual <- sample(
    d_individuals$individual, 
    n_obs, 
    replace = TRUE
  )

  d_individuals$individual_mean <- rnorm(
    n_individuals,
    d_groups$group_mean[d_individuals$group],
    d_groups$group_sigma[d_individuals$group]
  )

  d_observations$y <- rnorm(
    n_obs,
    d_individuals$individual_mean[d_observations$individual], 
    sigma_y
  )

  d_observations |> 
    left_join(d_individuals, by = join_by(individual)) |> 
    left_join(d_groups, by = join_by(group)) |> 
    mutate(sigma_y = sigma_y)
}

d <- simulate_data()
```

Run a simple Stan model and check that it recovers the parameters.

``` r
m <- cmdstanr::cmdstan_model("1.stan")
data <- list(
  n_obs = nrow(d),
  n_ind = length(unique(d$individual)),
  individual = d$individual,
  n_groups = length(unique(d$group)),
  y = d$y
)
e <- m$sample(data, refresh = 0)
## Running MCMC with 4 sequential chains...
## Chain 1 Informational Message: The current Metropolis proposal is about to be rejected because of the following issue:
## Chain 1 Exception: normal_lpdf: Scale parameter is 0, but must be positive! (in 'C:/Users/Tristan/AppData/Local/Temp/RtmpANG1E7/model-43e449af1364.stan', line 21, column 2 to column 25)
## Chain 1 If this warning occurs sporadically, such as for highly constrained variable types like covariance matrices, then the sampler is fine,
## Chain 1 but if this warning occurs often then your model may be either severely ill-conditioned or misspecified.
## Chain 1
## Chain 1 finished in 0.4 seconds.
## Chain 2 finished in 0.5 seconds.
## Chain 3 Informational Message: The current Metropolis proposal is about to be rejected because of the following issue:
## Chain 3 Exception: normal_lpdf: Scale parameter is 0, but must be positive! (in 'C:/Users/Tristan/AppData/Local/Temp/RtmpANG1E7/model-43e449af1364.stan', line 21, column 2 to column 25)
## Chain 3 If this warning occurs sporadically, such as for highly constrained variable types like covariance matrices, then the sampler is fine,
## Chain 3 but if this warning occurs often then your model may be either severely ill-conditioned or misspecified.
## Chain 3
## Chain 3 finished in 0.3 seconds.
## Chain 4 Informational Message: The current Metropolis proposal is about to be rejected because of the following issue:
## Chain 4 Exception: normal_lpdf: Scale parameter is 0, but must be positive! (in 'C:/Users/Tristan/AppData/Local/Temp/RtmpANG1E7/model-43e449af1364.stan', line 23, column 2 to column 45)
## Chain 4 If this warning occurs sporadically, such as for highly constrained variable types like covariance matrices, then the sampler is fine,
## Chain 4 but if this warning occurs often then your model may be either severely ill-conditioned or misspecified.
## Chain 4
## Chain 4 finished in 0.3 seconds.
## 
## All 4 chains finished successfully.
## Mean chain execution time: 0.4 seconds.
## Total execution time: 1.8 seconds.
e_sum <- e$summary()

d_post_individuals <- e_sum |> 
  filter(variable |> startsWith("coef_b[")) |> 
  mutate(
    individual = readr::parse_number(variable),
    type = "alpha + ranef"
  ) |> 
  left_join(
    d |> distinct(individual, individual_mean), 
    by = join_by(individual)
  )

ggplot(d_post_individuals) + 
  aes(x = individual) + 
  geom_linerange(
    aes(ymin = q5, ymax = q95, color = "posterior 90%\n(inferred latent mean)"), 
  ) +
  geom_point(
    aes(y = individual_mean, color = "latent mean"),
    size = 3,
    alpha = .8
  ) + 
  geom_point(
    aes(y = y, color = "observed mean"),
    data = d,
    alpha = .8,
    stat = "summary",
    size = 3,
  ) +
  guides(
    color = guide_legend(
      override.aes = list(shape = c(19, 19, NA), linewidth = c(NA, NA, .5))
    )
  ) + 
  facet_wrap("type") +
  scale_color_manual(
    "quantities",
    values = palette.colors(3)[c(2, 3, 1)] |> unname()
  )
## No summary function supplied, defaulting to `mean_se()`
```

<img src="README_files/figure-gfm/simple-ri-model-1.png" width="80%" />

``` r
e_sum |> 
  filter(variable %in% c("alpha", "sigma_b", "sigma_y")) |> 
  left_join(
    d |> 
      distinct(group_mean, group_sigma, sigma_y) |> 
      rename(sigma_b = group_sigma, alpha = group_mean) |> 
      tidyr::pivot_longer(
        everything(), 
        names_to = "variable", 
        values_to = "value"
      )
  ) |> 
  ggplot() + 
    aes(x = variable) + 
    geom_linerange(aes(ymin = q5, ymax = q95)) +
    geom_point(
      aes(y = value),
      color = "blue",
      size = 3,
      alpha = .3
    )
## Joining with `by = join_by(variable)`
```

<img src="README_files/figure-gfm/simple-ri-model-2-1.png" width="40%" />

## Assignment to groups

Let’s assume the individuals belong to latent groups.

``` r
d <- simulate_data(
  n_groups = 3, 
  group_mean = c(-3, 1, 3.5)
)

ggplot(d) + 
  aes(x = individual, y = y) + 
  geom_point(aes(color = factor(group))) +
  geom_point(aes(y = individual_mean))
```

<img src="README_files/figure-gfm/latent-groups-1.png" width="80%" />

The random intercept model does okay.

``` r
m <- cmdstanr::cmdstan_model("1.stan")

data <- list(
  n_obs = nrow(d),
  n_ind = length(unique(d$individual)),
  individual = d$individual,
  n_groups = length(unique(d$group)),
  y = d$y
)
e <- m$sample(data, refresh = 0)
## Running MCMC with 4 sequential chains...
## 
## Chain 1 finished in 1.2 seconds.
## Chain 2 finished in 0.9 seconds.
## Chain 3 Informational Message: The current Metropolis proposal is about to be rejected because of the following issue:
## Chain 3 Exception: normal_lpdf: Scale parameter is 0, but must be positive! (in 'C:/Users/Tristan/AppData/Local/Temp/RtmpANG1E7/model-43e449af1364.stan', line 21, column 2 to column 25)
## Chain 3 If this warning occurs sporadically, such as for highly constrained variable types like covariance matrices, then the sampler is fine,
## Chain 3 but if this warning occurs often then your model may be either severely ill-conditioned or misspecified.
## Chain 3
## Chain 3 finished in 0.5 seconds.
## Chain 4 finished in 1.3 seconds.
## 
## All 4 chains finished successfully.
## Mean chain execution time: 1.0 seconds.
## Total execution time: 4.2 seconds.
e_sum <- e$summary()

d_post_individuals <- e_sum |> 
  filter(variable |> startsWith("coef_b[")) |> 
  mutate(
    individual = readr::parse_number(variable),
    type = "alpha + ranef"
  ) |> 
  left_join(
    d |> distinct(individual, individual_mean), 
    by = join_by(individual)
  )

ggplot(d_post_individuals) + 
  aes(x = individual) + 
  geom_linerange(
    aes(ymin = q5, ymax = q95, color = "posterior 90%\n(inferred latent mean)"), 
  ) +
  geom_point(
    aes(y = individual_mean, color = "latent mean"),
    size = 3,
    alpha = .8
  ) + 
  geom_point(
    aes(y = y, color = "observed mean"),
    data = d,
    alpha = .8,
    stat = "summary",
    size = 3,
  ) +
  guides(
    color = guide_legend(
      override.aes = list(shape = c(19, 19, NA), linewidth = c(NA, NA, .5))
    )
  ) + 
  facet_wrap("type") +
  scale_color_manual(
    "quantities",
    values = palette.colors(3)[c(2, 3, 1)] |> unname()
  )
## No summary function supplied, defaulting to `mean_se()`
```

<img src="README_files/figure-gfm/simple-ri-model-3-1.png" width="80%" />

<div id="refs" class="references csl-bib-body hanging-indent">

<div id="ref-JSSv078i02" class="csl-entry">

Proust-Lima, Cécile, Viviane Philipps, and Benoit Liquet. 2017.
“Estimation of Extended Mixed Models Using Latent Classes and Latent
Processes: The r Package Lcmm.” *Journal of Statistical Software* 78
(2): 1–56. <https://doi.org/10.18637/jss.v078.i02>.

</div>

</div>
