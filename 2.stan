data{
  int<lower=1> n_obs;  // observations
  int<lower=1> n_ind;  // individuals
  array[n_obs] int<lower=1, upper=n_ind> individual;

  int<lower=1> n_groups;  // latent groups

  array[n_obs] real y;

}
parameters {
  real alpha;               // fixed intercept term
  vector[n_ind] b;          // random intercepts
  real<lower=0> sigma_b;    // standard deviation of random intercepts
  real<lower=0> sigma_y;    // residual standard deviation

  simplex[n_groups] probs;
  vector[n_groups] mean_group;
}

model {


  alpha ~ normal(0, 10);
  sigma_b ~ exponential(2);
  sigma_y ~ exponential(2);
  mean_group ~ normal(0, 10);
  b ~ normal(0, sigma_b);
  probs ~ dirichlet(rep_vector(1.0, n_groups));

  b ~ normal(0, sigma_b);

  y ~ normal(alpha + b[individual], sigma_y);
}
  // array[n_groups] real lmix;
  // for (group in 1:n_groups) {
  //   lmix[group] = log(probs[group]) + normal_lpdf( y | mean_group[group] + b[individual], sigma_y);
  //   target += log_sum_exp(lmix);
  // }


generated quantities{
  // following the idea in lme4 that coef() is fixef() + ranef()
  //vector[n_ind] coef_b = alpha + b;
}
