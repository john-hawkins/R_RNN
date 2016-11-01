
# #################################################################################
#  RETURN THE ACTIVATION FUNCTIONS AND THEIR DERIVATIVES BY NAME
# #################################################################################

# PRE-DEFINED VECTORISED ACTIVATION FUNCTIONS
sigmoid.activation <- Vectorize(
	function(x) {
		1 / ( 1+exp(-(x)) )
	}
)
tanh.activation <- Vectorize(
	function(x) {
		( exp(x) - exp(-x) ) / ( exp(x) + exp(-x) )
	}
)
# PRE-DEFINED VECTORISED DERIVATIVES OF THE ACTIVATION FUNCTIONS
# NOTE: FOR PROCESSING EFFICIENCY THESE ARE DEFINIED IN TERMS OF 
#        THE OUTPUT RATHER THAN THE INPUT OF THE ACTIVATION FUNCTION
d.sigmoid.activation <- Vectorize(
	function(z) {
               z*(1-z)
	}
)
d.tanh.activation <- Vectorize(
	function(z) {
               1 - (z^2)
	}
)

# RETURN THE ACTIVATION FUNCTIONS BY NAME
get.activation.function <- function(actvn) {
	if(actvn=="tanh") {
		return(tanh.activation)
	} else {
		return(sigmoid.activation)
	}
}

get.activation.function.derivative <- function (actvn) {
        if(actvn=="tanh") {
                return(d.tanh.activation)
        } else {
                return(d.sigmoid.activation)
        }
}

# ####################################################################################
# RETURN THE COST FUNCTION AND ITS DERIVATIVE BY NAME
# ####################################################################################
get.cost.function <- function(cost.func){
	if(cost.func=="xent") {
		return(cross.entropy.loss)
	} else {
		return(quadratic.loss)
	}
}

get.cost.function.derivative <- function(cost.func) {
        if(cost.func=="xent") {
                return(d.cross.entropy.loss)
        } else {
                return(d.quadratic.loss)
        }
}

quadratic.loss <- Vectorize(
         function(y, o) {
            0.5 * (y - o)^2
         }
)

cross.entropy.loss <- Vectorize(
         function(y, o) {
            -y * log(o) + (1-y) * log(1-o) 
         }
)


d.quadratic.loss <- Vectorize(
         function(y, o) {
            y - o
         }
)

d.cross.entropy.loss <- Vectorize(
         function(y, o) {
            (y - o)/(o*(1+o)) 
         }
)

