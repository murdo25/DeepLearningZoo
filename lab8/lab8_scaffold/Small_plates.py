#the small plates

import tensorflow as tf
import numpy as np

from textloader import TextLoader
from tensorflow.python.ops.rnn_cell import BasicLSTMCell



#
# -------------------------------------------
#
# Global variables

batch_size      = 50
sequence_length = 50

data_loader = TextLoader( ".", batch_size, sequence_length )

vocab_size = data_loader.vocab_size  # dimension of one-hot encodings
state_dim = 128

num_layers = 2

tf.reset_default_graph()

#
# ==================================================================
# 
'''                     INCOMPLETE GRU :'('''
from tensorflow.python.ops.rnn_cell import RNNCell
import numpy as np

class mygru( RNNCell ):
 
    def __init__( self, num_units):
    	self.num_units = num_units
    	#self.activation = tanh
 
    @property
    def state_size(self):
    	return self.num_units
    	
 
    @property
    def output_size(self):
    	return self.num_units
    	
 
    def __call__( self, inputs, state, scope=None ):
    	rt = _linear([inputs, h], 2 * self._num_units, True)

''''''
# ==================================================================
#

# define placeholders for our inputs.  
# in_ph is assumed to be [batch_size,sequence_length]
# targ_ph is assumed to be [batch_size,sequence_length]

in_ph = tf.placeholder( tf.int32, [ batch_size, sequence_length ], name='inputs' )
targ_ph = tf.placeholder( tf.int32, [ batch_size, sequence_length ], name='targets' )
in_onehot = tf.one_hot( in_ph, vocab_size, name="input_onehot" )

inputs = tf.split( 1, sequence_length, in_onehot )
inputs = [ tf.squeeze(input_, [1]) for input_ in inputs ]
targets = tf.split( 1, sequence_length, targ_ph )

# at this point, inputs is a list of length sequence_length
# each element of inputs is [batch_size,vocab_size]

# targets is a list of length sequence_length
# each element of targets is a 1D vector of length batch_size


# ------------------
# YOUR COMPUTATION GRAPH HERE
    
# create a BasicLSTMCell
cell = BasicLSTMCell( state_dim )
stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([cell]*num_layers, state_is_tuple=True)

initial_state = stacked_lstm.zero_state(batch_size, tf.float32)

# call seq2seq.rnn_decoder
outputs, final_state = tf.nn.seq2seq.rnn_decoder(inputs, initial_state, stacked_lstm)
    
# transform the list of state outputs to a list of logits.
    
W = tf.Variable(tf.truncated_normal( [state_dim,vocab_size], stddev=0.1 ))  


# use a linear transformation.
logits = [tf.matmul(i,W) for i in outputs] 

Logit_Weights = [tf.ones([batch_size,1],tf.float32) for i in logits]
    
# call seq2seq.sequence_loss
loss = tf.nn.seq2seq.sequence_loss(logits,targets,Logit_Weights)

# create a training op using the Adam optimizer
optim = tf.train.AdamOptimizer( 0.002, beta1=0.5 ).minimize( loss )    
    
    
# ------------------


# place your sampler graph here it will look a lot like your
# computation graph, except with a "batch_size" of 1.

# remember, we want to reuse the parameters of the cell and whatever
# parameters you used to transform state outputs to logits!
s_len   = 1
b_size  = 1

# YOUR SAMPLER GRAPH HERE

tf.get_variable_scope().reuse_variables()


s_initial_state = stacked_lstm.zero_state(b_size, tf.float32)
# call seq2seq.rnn_decoder
s_inputs = tf.placeholder( tf.int32, [s_len], name='s_input' )

s_in = [tf.one_hot( s_inputs, vocab_size, name="one_hot_inputs" )] 


 
s_outputs, s_final_state = tf.nn.seq2seq.rnn_decoder(s_in, s_initial_state, stacked_lstm, loop_function=None, scope=None)
    
# use a linear transformation.
s_logits = [tf.matmul(i,W) for i in s_outputs] 
    
s_probs = tf.nn.softmax(s_logits[0])
    


#
# ==================================================================
# ==================================================================
# ==================================================================
#

def sample( num=200, prime='ab' ):

    # prime the pump 

    # generate an initial state. this will be a list of states, one for
    # each layer in the multicell.
    s_state = sess.run( s_initial_state )

    # for each character, feed it into the sampler graph and
    # update the state.
    for char in prime[:-1]:
        x = np.ravel( data_loader.vocab[char] ).astype('int32')
        feed = { s_inputs:x }
        for i, s in enumerate( s_initial_state ):
            feed[s] = s_state[i]
        s_state = sess.run( s_final_state, feed_dict=feed )

    # now we have a primed state vector; we need to start sampling.
    ret = prime
    char = prime[-1]
    for n in range(num):
        x = np.ravel( data_loader.vocab[char] ).astype('int32')

        # plug the most recent character in...
        feed = { s_inputs:x }
        for i, s in enumerate( s_initial_state ):
            feed[s] = s_state[i]
        ops = [s_probs]
        ops.extend( list(s_final_state) )

        retval = sess.run( ops, feed_dict=feed )

        s_probsv = retval[0]
        s_state = retval[1:]

        # ...and get a vector of probabilities out!

        # now sample (or pick the argmax)
        # sample = np.argmax( s_probsv[0] )
        sample = np.random.choice( vocab_size, p=s_probsv[0] )

        pred = data_loader.chars[sample]
        ret += pred
        char = pred

    return ret

#
# ==================================================================
# ==================================================================
# ==================================================================
#




#
# ==================================================================
# ==================================================================
# ==================================================================
#
sess = tf.Session()
sess.run( tf.initialize_all_variables() )
summary_writer = tf.train.SummaryWriter( "./tf_logs", graph=sess.graph )

lts = []

print "FOUND %d BATCHES" % data_loader.num_batches

for j in range(10000):
    
    state = sess.run( initial_state )
    data_loader.reset_batch_pointer()
    
    for i in range( data_loader.num_batches ):
        
        x,y = data_loader.next_batch()

        # we have to feed in the individual states of the MultiRNN cell
        feed = { in_ph: x, targ_ph: y }
        for k, s in enumerate( initial_state ):
            feed[s] = state[k]

        ops = [optim,loss]
        ops.extend( list(final_state) )

        # retval will have at least 3 entries:
        # 0 is None (triggered by the optim op)
        # 1 is the loss
        # 2+ are the new final states of the MultiRNN cell
        retval = sess.run( ops, feed_dict=feed )

        lt = retval[1]
        state = retval[2:]

        if i%1000==0:
            print "%d %d\t%.4f" % ( j, i, lt )
            lts.append( lt )
    
    print sample( num=60, prime="Harry " )
#    print sample( num=60, prime="ababab" )
#    print sample( num=60, prime="foo ba" )
#    print sample( num=60, prime="abcdab" )

#summary_writer.close()

#
# ==================================================================
# ==================================================================
# ==================================================================
#

#import matplotlib
#import matplotlib.pyplot as plt
#plt.plot( lts )
#plt.show()


#gru _.linier
