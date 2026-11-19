# fdaSP 1.1.4

* Corrected all four adaptive function-to-scalar DoF branches (with/without
  smoothing and unpenalized covariates) to use n * lambda1 with the
  unnormalized Gram matrix. Adaptive DoF and BIC values can change.
* Included the coordinate-weight factors in the adaptive penalty-gradient
  derivative in functional and linear grouped DoF routines. This matters
  for nonunit var_weights, including the default overlapping spline weights.
* Aligned function-to-scalar documentation with RSS/(2*n) and
  (lambda2/2) * ||D psi||^2, and corrected the displayed linear-loss scale.
  Solver updates and fitted paths at fixed parameters are unchanged.
* Added functional adaptive DoF regression tests against central finite
  differences and an independent KKT derivative, including nonunit weights,
  smoothing and unpenalized covariates.

* Fixed missing response forwarding in the adaptive Lasso degrees-of-freedom
  calculation and the internal grouped-penalty dispatcher.
* Used the fixed-weight degrees-of-freedom branch for Group Lasso.
* Passed unpenalized covariates to the Lasso degrees-of-freedom calculation
  and corrected the augmented penalty-matrix dimensions for Group Lasso.
* Matched the adaptive correction to the solver's normalized least-squares
  loss, including the sample-size factor, and used the partial least-squares
  pilot when unpenalized covariates are present.
* Corrected the ordering of coefficient-surface slices returned in
  `f2fSP()$sp.fun.path`.
* Added regression tests using finite-difference divergence checks and
  coefficient-to-surface reconstruction checks.

# fdaSP 1.1.3

* Updated the documentation for `f2sSP()` and `f2sSP_cv()` to describe the
  sparsity (`lambda1`) and smoothness (`lambda2`) regularization parameters
  more clearly.

* Standardized the names of the sparsity-parameter arguments and returned
  components to use the `lambda1` terminology consistently.

* Revised the examples and the documentation of the returned model-selection
  quantities for fits with and without a smoothness penalty.
