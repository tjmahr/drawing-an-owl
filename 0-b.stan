data {
  int<lower=1> n_obs;
  int<lower=1> n_groups;
  int<lower=1> n_individuals;
  array[n_obs] int<lower=1, upper=n_individuals> individual;
  array[n_obs] real y;
}
transformed data {}
parameters {
  ordered[n_groups] mean_group;
  vector<lower=0>[n_groups] sigma_residuals;
  simplex[n_groups] probs;

  // the random intercepts start as z-scores
  array[n_groups] vector[n_individuals] standard_random_intercepts;
  // which are scaled for each group
  vector<lower=0>[n_groups] sigma_intercepts;
}
transformed parameters {
  // apply the actual scaling
  array[n_groups] vector[n_individuals] random_intercepts;
  for (i in 1:n_groups) {
    random_intercepts[i] = standard_random_intercepts[i] * sigma_intercepts[i];
  }
}
model {
  array[n_groups] real group_likelihoods;

  sigma_residuals ~ exponential(2);
  sigma_intercepts ~ exponential(2);
  mean_group ~ normal(0, 10);
  probs ~ dirichlet(rep_vector(1.0, n_groups));
  array[n_groups] vector[n_individuals] linear_terms;
  for (i in 1:n_groups) {
    standard_random_intercepts[i] ~ std_normal();
  }

  for (i in 1:n_individuals) {
    for (j in 1:n_groups) {
      linear_terms[j, i] = mean_group[j] + random_intercepts[j, i];
    }
  }

  for (i in 1:n_obs) {
    for (j in 1:n_groups) {
      group_likelihoods[j] = log(probs[j]) +
        normal_lpdf(y[i] | linear_terms[j, individual[i]], sigma_residuals[j]);
    }
    target += log_sum_exp(group_likelihoods);
  }
}
generated quantities {
  // matrix[n_obs, n_groups] g_probs;
  // for (i in 1:n_obs) {
  //   vector[n_groups] terms;
  //   for (j in 1:n_groups) {
  //     terms[j] = log(probs[j]) +
  //       normal_lpdf(y[i] | mean_group[j], sigma_group[j]);
  //   }
  //   g_probs[i, ] = to_row_vector(softmax(terms));
  // }
}
