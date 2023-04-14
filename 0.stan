data {
  int<lower=1> n_obs;  // observations
  int<lower=1> n_groups;  // latent groups
  array[n_obs] real y;
}

transformed data {}

parameters {
  ordered[n_groups] mean_group;
  vector<lower=0>[n_groups] sigma_group;
  simplex[n_groups] probs;
}

transformed parameters {}

model {
  array[n_groups] real group_likelihoods;

  sigma_group ~ exponential(2);
  mean_group ~ normal(0, 10);
  probs ~ dirichlet(rep_vector(1.0, n_groups));

  for (i in 1:n_obs) {
    for (j in 1:n_groups) {
      group_likelihoods[j] = log(probs[j]) +
        normal_lpdf(y[i] | mean_group[j], sigma_group[j]);
    }
    target += log_sum_exp(group_likelihoods);
  }
}

generated quantities {
  matrix[n_obs, n_groups] g_probs;
  for (i in 1:n_obs) {
    vector[n_groups] terms;
    for (j in 1:n_groups) {
      terms[j] = log(probs[j]) +
        normal_lpdf(y[i] | mean_group[j], sigma_group[j]);
    }
    g_probs[i, ] = to_row_vector(softmax(terms));
  }
}
