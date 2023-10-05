# Authors: Mauro Bernardi 
#          Department Statistical Sciences
#       	 University of Padova
#      	   Via Cesare Battisti, 241
#      	   35121 PADOVA, Italy
#          E-mail: mauro.bernardi@unipd.it  

# Last change: July 20, 2021


# ================================
# Check functions
check_data_matrix <- function(A) {
  cond1 = (is.matrix(A))  # matrix
  cond2 = (!(any(is.infinite(A))||any(is.na(A))))
  if (cond1 && cond2) {
    return(TRUE)
  } else {
    return(FALSE)
  }
}

check_data_vector <- function(b) {
  cond1 = ((is.vector(b))||((is.matrix(b))&&
                              (length(b)==nrow(b))||(length(b)==ncol(b))))
  cond2 = (!(any(is.infinite(b))||any(is.na(b))))
  if (cond1&&cond2){
    return(TRUE)
  } else {
    return(FALSE)
  }
}

check_data_positive_vector <- function(b) {
  cond1 = ((is.vector(b))||((is.matrix(b))&&
                              (length(b)==nrow(b))||(length(b)==ncol(b))))
  cond2 = (!(any(is.infinite(b))||any(is.na(b))))
  cond3 = (!(any(b<=0)))
  if (cond1&&cond2&&cond3){
    return(TRUE)
  } else {
    return(FALSE)
  }
}

check_param_constant <- function(num, lowerbound=0){
  cond1 = (length(num)==1)
  cond2 = ((!is.infinite(num))&&(!is.na(num)))
  cond3 = (num > lowerbound)
  
  if (cond1&&cond2&&cond3){
    return(TRUE)
  } else {
    return(FALSE)
  }
}

check_param_constant_multiple <- function(numvec, lowerbound=0){
  for (i in 1:length(numvec)){
    if (!check_param_constant(numvec[i], lowerbound)){
      return(FALSE)
    }
  }
  return(TRUE)
}

check_param_integer <- function(num, lowerbound=0){
  cond1 = (length(num)==1)
  cond2 = ((!is.infinite(num))&&(!is.na(num)))
  cond3 = (num > lowerbound)
  cond4 = (abs(num-round(num)) < sqrt(.Machine$double.eps))
  
  if (cond1&&cond2&&cond3&&cond4){
    return(TRUE)
  } else {
    return(FALSE)
  }
}

check_weights <- function(x, n, algname, funname) {
  if (!is.null(x)) {
    if (!is.numeric(x)) {
      stop(paste0("*", algname, " : the vector ", funname, " is not a numeric vector."))
    }
    if (n != length(x)) {
      stop(paste0("*", algname, " : the length of the vector ", funname, " does not match the number of penalized groups."))
    }
    if (min(x) < 0.0) {
      stop(paste0("*", algname, " : the values in ", funname, " must be positive."))
    }
  }
}

check_group_weights <- function(x, n, algname, funname) {
  if (!is.null(x)) {
    if (!is.numeric(x)) {
      stop(paste0("*", algname, " : the vector ", funname, " is not a numeric vector."))
    }
    if ((n != length(x)) && (n != (length(x)+1))) {
      stop(paste0("*", algname, " : the length of the vector ", funname, " does not match the number of penalized predictors."))
    }
    if (min(x) < 0.0) {
      stop(paste0("*", algname, " : the values in ", funname, " must be positive."))
    }
  }
}

check_group_overlaps <- function(GRmat) {
  
  csum <- colSums(GRmat)
  if (any(csum != 1)) {
    res <- "ovglasso"
  } else if (any(rowSums(GRmat) != 1)) {
    res <- "glasso"
  } else {
    res <- "lasso"
  }
  return(res)
}

# ================================
# AUXILIARY COMPUTATIONS 

# 2. PseudoInverse using SVD and NumPy Scheme
aux_pinv <- function(A){
  svdA      <- svd(A)
  tolerance <- (.Machine$double.eps)*max(c(nrow(A),ncol(A)))*as.double(max(svdA$d))
  
  idxcut          <- which(svdA$d <= tolerance)
  invDvec         <- (1.0 / svdA$d)
  invDvec[idxcut] <- 0
  output          <- (svdA$v %*% diag(invDvec) %*% t(svdA$u))
  return(output)
}

logseq <- function(from = 1, to = 1000, length.out = 6, reverse = FALSE) {
  
  # ================================
  # logarithmic spaced sequence
  out <- exp(seq(log(from), log(to), length.out = length.out))
  
  # ================================
  # reverse the sequence
  if (reverse == TRUE) {
    res <- out[length(out):1]
  } else {
    res <- out
  }
  
  # ================================
  # Get output
  return(res)
}

linreg <- function(vY, mX) {
  
  # ================================
  # Get the relevant quantities
  mXX <- crossprod(mX)
  vXy <- crossprod(mX, vY)
  
  # ================================
  # Perform the QR decomposition of mXX
  mXX_QR <- qr(mXX)
  mQ     <- qr.Q(mXX_QR)
  mR     <- qr.R(mXX_QR)
  
  # ================================
  # Compute the estimate
  vXy_  <- crossprod(mQ, vXy)
  vRegP <- backsolve(mR, vXy_, k = ncol(mR), upper.tri = TRUE, transpose = FALSE)
  
  # ================================
  # Return output
  return(vRegP)
}

scaledata <- function(x) {
  if (is.vector(x) == TRUE) {
    x.scaled <- (x - mean(x)) / sd(x)
  } else {
    x.scaled <- apply(x, 2, function(x) (x-mean(x))/sd(x))
  }
  return(x.scaled)
}

# ================================
# Standardise response and design matrix
standardizemat <- function(X, y) {
  
  # ================================
  # Get dimensions
  n <- dim(X)[1]
  
  # ================================
  # Get std
  y  <- matrix(y, nrow = n, ncol = 1)
  mU <- diag(sqrt(diag(var(X))))
  if (ncol(y) > 1) {
    mV <- diag(sqrt(diag(var(y))))
  } else {
    mV <- sqrt(var(y))
  }
  y.ctr <- apply(y, 2, function(x) x - mean(x))
  y.std <- y.ctr %*% solve(mV)
  X.ctr <- apply(X, 2, function(x) x - mean(x))
  X.std <- X.ctr %*% solve(mU)
  
  # ================================
  # Get output
  out       <- NULL
  out$X.std <- X.std
  out$y.std <- y.std
  out$mU    <- mU
  out$mV    <- mV
  
  # ================================
  # Return output
  return(out)
}

# return the regression parameters in the original scale
# after standardization
invstdparms <- function(parms, mU, mV) {

  # get dimensions
  n <- dim(parms)[1]
  p <- dim(parms)[2]
  
  # standardize data
  if (p == 1) {
    parms.std <- solve(mU) %*% parms[1,] %*% mV
  } else {
    #parms.std <- t(apply(t(parms), 2, function(x) solve(mU) %*% x %*% mV))
    parms.std <- t(apply(parms, 1, function(x) solve(mU) %*% x %*% mV))
  }
  
  # convert to a mtrix object
  parms.std <- matrix(data = parms.std, nrow = n, ncol = p)
  
  # return output
  return(parms.std)
}

groups.cv <- function(n, k = 10) {
  if (k == 0) {
    k <- n
  }
  if (!is.numeric(k) || k < 0 || k > n) {
    stop("Invalid values of 'k'. Must be between 0 (for leave-one-out CV) and 'n'.")
  }
  dummyorder <- sample(1:n, size = n)
  f          <- ceiling(seq_along(dummyorder)/(n/k))
  
  groups     <- vector("list", length = max(f))
  groups.all <- NULL
  for (it in 1:max(f)) {
    indi         <- which(f == it)
    groups[[it]] <- dummyorder[indi]
    groups.all   <- c(groups.all, dummyorder[indi])
  }
  
  # groups cv
  groups.cv <- vector("list", length = max(f))
  for (it in 1:max(f)) {
    groups.cv[[it]] <- setdiff(groups.all, groups[[it]])
  }
  
  # define foldid
  foldid <- rep(0, n)
  for (i in 1:k) {
    idx         <- groups[[i]]
    foldid[idx] <- i
  }
  
  # get output
  ret             <- NULL
  ret$groups.pred <- groups
  ret$groups.cv   <- groups.cv
  ret$shuffle     <- groups.all
  ret$foldid      <- foldid
  
  # return output
  return(ret)
}

#' Function to solve the soft thresholding problem 
#'  
#' @param x the data value.
#' @param lambda the lambda value. 
#'
#' @return the solution to the soft thresholding operator. 
#' 
#' @references
#'  
#' \insertRef{hastie_etal.2015}{fdaSP}
#' 
#' @export softhresh
softhresh <- function(x, lambda) {
  return(sign(x) * pmax(rep(0, length(x)), abs(x) - lambda))
}

softhresh_group <- function(x, lam){
  return(max(0L, 1L - lam / normvec(x)) * x)
}

normvec <- function(va) {
  return(as.numeric(sqrt(crossprod(va))))
}

getsparsevec <- function(x, toler) {
  
  # ================================
  # Get dimensions
  n <- length(x)
  
  y <- x
  for (it in 1:n) {
    if (abs(x[it]) < toler) {
      y[it] <- 0.0
    }
  }
  
  # ================================
  # Return output
  return(y)
}

softthreshold <- function(xx, lambda) {

  # ================================
  # LASSO soft threshold operator
  zz <- abs(xx) - lambda
  if (zz > 0.0) {
    yy <- zz * sign(xx)
  } else {
    yy <- 0.0
  }
  # ================================
  # Return output
  return(yy)
}

getglassopenalty <- function(vRegP, mSelMat) {
  
  # ================================
  # Get indicators
  vPen <- sqrt(t(mSelMat) %*% (vRegP^2))
  
  # ================================
  # Return output
  return(vPen)
}

scaledata <- function(x) {
  if (is.vector(x) == TRUE) {
    x.scaled <- (x - mean(x)) / sd(x)
  } else {
    x.scaled <- apply(x, 2, function(x) (x-mean(x))/sd(x))
  }
  return(x.scaled)
}

# 6. fround
fround <- function(x, n) {
  
  y <- x
  for (it in 1:n) {
    if (abs(x[it]) < .Machine$double.eps) {
      y[it] <- 0.0
    }
  }
  return(y)
}

# 7. fthreshold
fthreshold <- function(x, n, toler) {
  
  y <- x
  for (it in 1:n) {
    if (abs(x[it]) < toler) {
      y[it] <- 0.0
    }
  }
  return(y)
}


# 8. set to a given tolerance
fSet2GivenTolerance <- function(x, n, toler) {
  
  y <- x
  for (it in 1:n) {
    if (abs(x[it]) < toler) {
      y[it] <- toler
    }
  }
  return(y)
}




