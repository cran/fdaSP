#
## last update: 30 september 2025
#

#
## Description: compute dual norms for several Lasso-type penalties
#

# group-lasso norm
#' @keywords internal
glasso_norm <- function(x, groups, weights = NULL) {
  
  # get dimensions
  p <- length(x)
  G <- dim(groups)[1]

  # check for null weights
  if (is.null(weights)) {
    weights <- rep(1, G)
  }
  
  # compute the group lasso norm
  norm2 <- 0.0
  for (g in 1:G) {
    idx   <- which(groups[g,] == 1)
    norm2 <- norm2 + weights[g] * norm(x[idx], type = "2")
  }
  
  # return output
  return(norm2)
}
# overlap group-lasso norm
#' @keywords internal
ovglasso_norm <- function(x, groups, weights = NULL) {
  
  # return output
  return(glasso_norm(x = x, groups = groups, weights = weights))
}
# group-lasso dual norm
#' @keywords internal
glasso_dual_norm <- function(x, groups, weights = NULL) {
  
  # get dimensions
  p <- length(x)
  G <- dim(groups)[1]
  
  # check for null weights
  if (is.null(weights)) {
    weights <- rep(1, G)
  }
  
  # compute the glasso dual norm
  dual_norm2 <- rep(0, G)
  for (g in 1:G) {
    idx           <- which(groups[g,] == 1)
    dual_norm2[g] <- norm(x[idx], type = "2") / weights[g]
  }
  
  # return output
  return(max(dual_norm2))
}
# This function compute the non-overlap approximation of overlap group lasso norm
# provided by Qi and Li (2024, JoMLR) in Algorithm 2
#
# Input:  G  = matrice (g x p) binaria (righe = gruppi originari, colonne = variabili)
# Output: matrice (k x p) con gruppi non-overlapping
#' @keywords internal
ovg2g <- function(groups, weights = NULL, return_list = TRUE) {
  
  # get dimensions
  groups <- as.matrix(groups)
  G      <- nrow(groups)
  p      <- ncol(groups)
  
  # pattern per colonna (stringa di 0/1) — uso as.integer per gestire anche valori logici
  patterns <- apply(groups, 2, function(col) paste(as.integer(col), collapse = ""))
  
  # unici nell'ordine di primo apparire
  uniq <- unique(patterns)
  k    <- length(uniq)
  
  # costruisco G1
  G1           <- matrix(0, nrow = k, ncol = p)
  rownames(G1) <- paste0("group", seq_len(k))
  if (!is.null(colnames(groups))) {
    colnames(G1) <- colnames(groups) 
  } else {
    colnames(G1) <- paste0("var", seq_len(p))
  }
  parts       <- vector("list", length = k)
  orig_groups <- vector("list", length = k)
  
  for (i in seq_along(uniq)) {
    cols        <- which(patterns == uniq[i])
    G1[i, cols] <- 1
    parts[[i]]  <- cols
    # quali gruppi originari partecipano a questo pattern (indici righe)
    bits             <- strsplit(uniq[i], "")[[1]]
    orig_groups[[i]] <- which(bits == "1")
  }
  names(parts)       <- rownames(G1)
  names(orig_groups) <- rownames(G1)
  
  # get weights
  weights_ <- rep(0, length(orig_groups))
  if (is.null(weights)) {
    weights <- rep(1, G)
    for (i in 1:length(orig_groups)) {
      weights_[i] <- sum(weights[orig_groups[[i]]])
    }
  } else {
    for (i in 1:length(orig_groups)) {
      weights_[i] <- sum(weights[orig_groups[[i]]])
    }
  }
  
  # return output
  if (return_list) {
    output             <- NULL
    output$G1          <- G1
    output$orig_groups <- orig_groups
    output$weights     <- weights_
  } else {
    output <- G1
  }
  return(output)
}
# This function compute the upper bound of the overlap group lasso dual norm
# provided by Qi and Li (2024, JoMLR), Proposition 1 in Appendix C
#' @keywords internal
ovglasso_dual_norm_upper_bound <- function(x, groups, weights) {
  
  # get the diagonal matrix of overlap degrees
  H <- diag(1.0 / colSums(groups))
  
  # compute the approximate dual norm of the overlap group lasso
  res <- glasso_dual_norm(x = H %*% x, groups = groups, weights = weights)
  
  # return output
  return(res)
}
# This function compute the approximation of the overlap group lasso dual norm
# provided by Qi and Li (2024, JoMLR)
#' @keywords internal
ovglasso_dual_norm_approx <- function(x, groups, weights) {
  
  # get the non-overlap approximation of overlap group lasso norm
  res      <- ovg2g(groups = groups, weights = weights, return_list = TRUE)
  weights_ <- res$weights
  groups_  <- res$G1

  # compute the approximate dual norm of the overlap group lasso
  output <- glasso_dual_norm(x = x, groups = groups_, weights = weights_)
  
  # return output
  return(output)
}



