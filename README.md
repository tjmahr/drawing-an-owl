Untitled
================

Proust-Lima, Philipps, and Liquet (2017) describes the statistical
machinery for their latent class mixed models. We note the following
equations in their paper.

Below is the likelihood contribution for one individual in the basic
linear mixed model that we know and love:

``` math
\displaylines{
L_i = \phi_i(Y_i; \theta_1) \\
\phi: \textrm{MVN density} \\
i: \textrm{individuals}
}
```

When there are $G$ latent classes, the likelihood becomes a weighted sum
of class-specific likelihoods:

$$\displaylines{
L_i(\theta_G) = \sum_{g=1}^G \pi_{ig}\phi_{ig}(Y_i|c_i = g; \theta_G) \\
\pi_{ig}: \textrm{probability of group membership for an individual} \\
g: \textrm{groups}
}$$

So, an individual makes $G$ contributions to the likelihood and each one
is weighted by their group membership probability. And when there is one
group, this equation reduces to the first likelihood equation.

Finally, group probabilities are defined as a multinomial logistic
model:

$$
\pi_{ig} = P(c_i = g | X_{ci}) = \frac{{}}{}
$$

<div id="refs" class="references csl-bib-body hanging-indent">

<div id="ref-JSSv078i02" class="csl-entry">

Proust-Lima, Cécile, Viviane Philipps, and Benoit Liquet. 2017.
“Estimation of Extended Mixed Models Using Latent Classes and Latent
Processes: The r Package Lcmm.” *Journal of Statistical Software* 78
(2): 1–56. <https://doi.org/10.18637/jss.v078.i02>.

</div>

</div>
