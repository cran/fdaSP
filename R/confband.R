### codice per bande ombreggiate
# xV:    vettore asse x (sequenza lambda)
# yVmax: vettore asse y (MSE + sd(MSE)))
# yVmin: vettore asse y (MSE - sd(MSE)))

#' Function to plot the confidence bands 
#'  
#' @param xV the values for the x-axis.
#' @param yVmin the minimum values for the y-axis. 
#' @param yVmax the maximum values for the y-axis.
#'
#' @return a polygon. 
#' 
#' @export confband
confband <- function(xV, yVmin, yVmax) {
  polygon(x = c(xV, rev(xV)), y = c(yVmin, rev(yVmax)),  col = rgb(0,0,1,.3), border = NA)
}
