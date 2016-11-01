#
# WARNING: THIS CODE DOES NOT YET RUN
#

require(data.table)
source("Common-Lib/rnn_functions.R")

elman.rnn <- function(
	data, 
	covars, 
	targ.vars, 
	time.var, 
	group.var = NULL, 
        model = NULL,
        h.actvn.func = "tanh" # Options: c("sig","tanh")
        o.actvn.func = "sig" # Options: c("sig")
        cost.func = "xent" # Options: c("xent","quad")
	l.rate = 0.1, 
	decay = 1, 
	momentum = 0, 
	h.dim = 10,
        t.steps = 10,
	epochs = 1,
	batch_size = 1,
	seed = 1L
 ) {
   set.seed(seed)
   # #######################################################################################################
   # WE ASSUME THAT THE DATA HAS BEEN PRE-ORDERED, FIRST BY THE GROUP DIMENSION, THEN BY THE TIME DIMENSION
   # I TRIED TO DO THIS HERE BUT I COULD NOT GET THE SORT FUNCTIONS TO DEREFENCE THE COLUMN NAMES VARS
   # RUN THE FOLLOWING:   setorder(data, group.var, time.var)
   #
   # WE USE DATA TABLE OPERATIONS, SO IF THE INPUT IS NOT A DATA TABLE THEN WE COERCE IT
   # #######################################################################################################
   if() {
      data <- data.table(data)
   }
   
   if(is.null(model)) {
      model 			= list()
      model$i.dim 		= length(covars)
      model$h.dim 		= h.dim 
      model$o.dim 	        = length(targ.vars)
      model$epochs	     	= epochs
      model$batch.size    	= batch_size
      model$l.rate 	  	= l.rate
      model$decay		= decay
      model$t.steps             = t.steps
      model$momentum    	= momentum
      model$i_h_weights         = array( rnorm(model$i.dim * model$h.dim), dim=c(model$i.dim, model$h.dim) )
      model$h_rec_weights       = array( rnorm(model$h.dim * model$h.dim), dim=c(model$h.dim, model$h.dim) )
      model$h_o_weights 	= array( rnorm(model$h.dim * model$o.dim), dim=c(model$h.dim, model$o.dim) )
      model$h_bias 		= array( rnorm(model$h.dim) )
      model$o_bias 		= array( rnorm(model$o.dim) )

      model$h.actvn.func	= h.actvn.func
      model$h.actvn		= get.activation.function(h.actvn.func)
      model$h.actvn.deriv       = get.activation.function.derivative(h.actvn.func)
      model$o.actvn             = get.activation.function(o.actvn.func)
      model$o.actvn.deriv       = get.activation.function.derivative(o.actvn.func)      

      model$cost.func  		= cost.func
      model$cost		= get.cost.function(cost.func)
      model$cost.deriv		= get.cost.function.derivative(cost.func)

   } else {
	# ######################################################################################
	# TODO -- CHECK THAT THE NEW DATA IS APPLICABLE TO THE EXISTING MODEL
	# ######################################################################################
   }

   # ###########################################################################################
   # TRAIN THE MODEL
   # ###########################################################################################
   # ITERATE OVER THE TOTAL NUMBER OF TRAINING EPOCHS 
   # - EACH EPOCH CORRESPONDS TO A SINGLE PASS THROUGH THE COMPLETE DATA SET
   # ###########################################################################################
   for(e in seq(model$epochs) ) {
      # MODEL STARTS THE EPOCH WITH BLANK ACTIVATIONS
      model <- reset.activations(model)

      # SET UP THE BATCH RELATED VARABLES
      # CAPTURE THE CURRENT GROUP
      batch.group <- data[1,][[group.var]]
      batch.index <- 0
      batch.error <- array(0, length(targ.vars))

      for(r in seq(nrow(data) ) ) {
         curr.group <- data[r,][[group.var]]
         if(curr.group != batch.group ) { # RESET ACTIVATIONS
            reset.activations(model)
            batch.group = curr.group
         }
         in.data <- as.numeric(data[r, covars, with=FALSE])
         targs <- as.numeric(data[r, targ.vars, with=FALSE])
         output <- rnn.predict(model, in.data)
         error <- model$cost(targs, output)
         batch.error <- batch.error + error
         backpropagate.error(model, targs)

         # INCREMENT BATCH INDEX AND CHECK IF WE NEED TO APPLY WEIGHT CHANGES
         batch.index = batch.index + 1
         if(batch.index == model$batch.size) {
            apply.weight.updates(model)
            reset.deltas(model)
            avg.error = sum(batch.error/batch.index)/model$o.dim
            print(paste("Iteration:", iters, " Average Batch Error:", avg.error)
            batch.index = 0
         }
      }
      # APPLY RESIDUAL WEIGHT UPDATES
      apply.weight.updates(model)
      reset.deltas(model)
      model$epoch <- e
   }

}


# ######################################################################################
# GIVEN A PARAMETERISED RNN MODEL - SET NODE ACTIVATIONS TO ZERO
# ######################################################################################
reset.activations <- function(model) {
   model$i_actvns = array( 0, model$i.dim )
   model$h_actvns <- array( 0, dim=c( model$t.steps, model$h.dim ) )
   model$o_actvns = array( 0, model$o.dim )
   model
}

# ##############################################################################3#######
# RESET THE CUMLATIVE WEIGHT CHANGES TO ZERO
# - THIS SHOULD BE CALLED AT THE END OF EVERY BATCH ONCE THE UPDATES ARE DONE
# ######################################################################################
reset.deltas <- function(model) {
   model$delta_i_h_weights         = array( 0, dim=c(model$i.dim, model$h.dim) )
   model$delta_h_rec_weights       = array( 0, dim=c(model$h.dim, model$h.dim) )
   model$delta_h_o_weights         = array( 0, dim=c(model$h.dim, model$o.dim) )
   model$delta_h_bias              = array( 0, model$h.dim )
   model$delta_o_bias              = array( 0, model$o.dim )
   model   
}

# #################################################################################
# GIVEN A MODEL - PUSH THE TEMPORAL ACTIVATIONS BACK ONE STEP IN TIME
# #################################################################################
push.activations <- function(model) {
   for(t in seq(model$t.steps,2)) {
      model$h_actvns[t,] <- model$h_actvns[t-1,]
   }
   model$h_actvns[1,] <- array( 0, model$h.dim )
   model
}

# #################################################################################
# GIVEN A MODEL AND A ROW OF DATA - MAKE A PREDICTION
# #################################################################################
rnn.predict <- function(model, newdata) {
   model			= push.activations(model)
   model$i_actvns 		= as.numeric(newdata)
   model$h_pre_sigmoid 		= model$i_actvns %*% model$i_h_weights + 
				  model$h_actvns[2,] %*% model$h_rec_weights +
				  as.vector(model$h_bias)
   model$h_post_sigmoid 	= model$h.actvn(model$h_pre_sigmoid)
   model$h_actvns[1,] 		= model$h_post_sigmoid
   model$o_pre_sigmoid 		= model$h_post_sigmoid %*% model$h_o_weights +
				  as.vector(model$o_bias)
   model$o_post_sigmoid 	= model$o.actvn(model$o_pre_sigmoid)
   model$o_post_sigmoid
}


# #################################################################################
# GIVEN A MODEL AND A TARGET OUTPUT - BACKPROPAGATE THE ERROR THROUGH THE NETWORK
# #################################################################################
backpropagate.error <- function(model, targs) {
   # Derivative of the cost function with respect to output
   d_C_d_O       		<- model$cost.deriv( targs, model$o_post_sigmoid )

   # Derivative of the output with respect to net (the sum of weighted inputs and bias)
   # - Convention is to define this function in terms of the output of the sigmoid
   #   because it is simpler to compute.
   d_O_d_onet 			<- model$o.actvn.deriv( model$o_post_sigmoid )
   
   # Chain previous two together
   d_C_d_onet 			<- d_C_d_O * d_O_d_onet

   # Derivative of Pre-Sigmoid Sum with respect to each of the h-to-o weights
   # is just the activation behind the weight. So to get dC/dW we matrix multiply the
   # derivative of the cost to the weighted sum, by the hidden node activations.
   d_C_d_h_o_w 			<- d_C_d_onet %*% model$h_post_sigmoid

   # Derivative of the cost with respect to the hidden node outputs
   d_C_d_h_act			<- d_C_d_onet %*% model$h_o_weights

   # Derivative of the hidden node activation with respect to the hidden node inputs
   d_h_act_h_net		<- model$h.actvn.deriv( model$h_post_sigmoid )

   # Derivative of the cost with respect to the hidden node inputs
   d_C_d_h_net			<- d_C_d_h_act %*% d_h_act_h_net

   # Derivative of the Cost with respect to the hidden node recurrent weights
   d_C_d_h_rec_w                <- d_C_d_h_net %*% model$h_actvns[2,]

   # TODO: ITERATE OVER PREVIOUS TIME STEPS AND BACKPROPAGATE
    
   # Derivative of the Cost with respect to the input to hidden node weights
   d_C_d_i_h_w  		<- d_C_d_h_net %*% model$i_actvns 

   # We accumulate the updates in buffers until the end of the batch   
   model$delta_i_h_weights  	= model$delta_i_h_weights + d_C_d_i_h_w 
   model$delta_h_rec_weights 	= model$delta_h_rec_weights + d_C_d_h_rec_w
   model$delta_h_o_weights  	= model$delta_h_o_weights + d_C_d_h_o_w
   model$delta_h_bias   	= model$delta_h_bias + d_C_d_h_net
   model$delta_o_bias   	= model$delta_o_bias + d_C_d_onet
}


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

