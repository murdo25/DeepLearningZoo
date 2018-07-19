
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
# from dropout lab:

def weight_variable(shape):
  initial = tf.truncated_normal( shape, stddev=0.1 )
  return tf.Variable( initial )

def bias_variable(shape):
  initial = tf.constant( 0.1, shape=shape )
  return tf.Variable(initial)

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

with tf.variable_scope("Graph_") as scope:
    
    # create a BasicLSTMCell
    cell = BasicLSTMCell( state_dim,state_is_tuple=True )
    #   use it to create a MultiRNNCell
    #tf.nn.rnn_cell.MultiRNNCell.__init__(cells, state_is_tuple=False)
    stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([cell]*2, state_is_tuple=True)
    #stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([cell]*2)
    #   use it to create an initial_state
    initial_state = stacked_lstm.zero_state(batch_size, tf.float32)
    #note that initial_state will be a *list* of tensors!
    
    # call seq2seq.rnn_decoder
    # rnn_decoder(decoder_inputs, initial_state, cell, loop_function=None, scope=None):
    #outputs, state = tf.nn.seq2seq.rnn_decoder(inputs, initial_state, stacked_lstm, loop_function=None, scope=None)
    outputs, state = tf.nn.seq2seq.rnn_decoder(inputs, initial_state, stacked_lstm, loop_function=None, scope=None)
    
    #print("shape ",tf.shape(outputs))
    #print(outputs[0].get_shape())
    
    # transform the list of state outputs to a list of logits.
    # use a linear transformation.
    #out = W * output + b
    #W = weight_variable([sequence_length,batch_size,state_dim])
    #W = weight_variable([batch_size,state_dim])
    W = tf.Variable(tf.truncated_normal( [state_dim,batch_size], stddev=0.1 ))
    
    b = bias_variable([len(outputs)])
    
    #out = tf.matmul(outputs,W)
    logits = [tf.matmul(i,W)+b for i in outputs] 
    #logits += b


    #print(logits[0].get_shape())
    
    #logits = tf.pack(logits)
    #print(W.dtype)
    #print(targets[0].dtype)
    #print(logits[0].dtype)
    #print()
    
    
    # call seq2seq.sequence_loss
    #def sequence_loss(logits, targets, weights,
    #                  average_across_timesteps=True, average_across_batch=True,
    #                  softmax_loss_function=None, name=None):
    #out:A tuple of the form (outputs, state)
    #weights: List of 1D batch-sized float-Tensors of the same length as logits.
    Logit_Weights = [tf.ones([batch_size,1],tf.float32) for i in logits]
    
    #print(Logit_Weights.dtype)
    #print()
    #print(logits[0].get_shape())
    #print(targets[0].get_shape())
    #print(Logit_Weights.get_shape())
    loss = tf.nn.seq2seq.sequence_loss(logits,targets,Logit_Weights)
    
    # create a training op using the Adam optimizer
    Adam = tf.train.AdamOptimizer( 0.0002, beta1=0.5 ).minimize( loss )    
    
    
# ------------------
# YOUR SAMPLER GRAPH HERE

# place your sampler graph here it will look a lot like your
# computation graph, except with a "batch_size" of 1.

# remember, we want to reuse the parameters of the cell and whatever
# parameters you used to transform state outputs to logits!

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

sess = tf.Session()
sess.run( tf.initialize_all_variables() )
summary_writer = tf.train.SummaryWriter( "./tf_logs", graph=sess.graph )

lts = []

print "FOUND %d BATCHES" % data_loader.num_batches

for j in range(1000):
    
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

    print sample( num=60, prime="And " )
#    print sample( num=60, prime="ababab" )
#    print sample( num=60, prime="foo ba" )
#    print sample( num=60, prime="abcdab" )

summary_writer.close()

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
