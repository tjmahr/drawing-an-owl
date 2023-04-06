data{
  int<lower=1> n_obs;  // observations
  int<lower=1> n_ind;  // individuals
  array[n_obs] int<lower=1, upper=n_ind> jj;

  int<lower=1> n_groups;  // latent groups

  array[n_obs] real y;

}

parameters {
  real alpha;               // fixed intercept term
  vector[n_ind] b;          // random intercepts
  real<lower=0> sigma_b;    // standard deviation of random intercepts
  real<lower=0> sigma_y;    // residual standard deviation
}

model {
  // Priors
  alpha ~ normal(0, 10);
  sigma_b ~ cauchy(0, 2.5);
  sigma_y ~ cauchy(0, 2.5);

  // Random intercept model
  for (j in 1:n_ind) {
    b[j] ~ normal(0, sigma_b);
  }
  for (n in 1:n_obs) {
    y[n] ~ normal(alpha + b[jj[n]], sigma_y);
  }
}
