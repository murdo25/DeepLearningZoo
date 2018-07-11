import tensorflow as tf







for i in range( 150 ):
  #make a dropout mask for this epoch
  mask1 = np.random.rand(1000,500) < p
  print("applying mask of size")
  print(mask1.shape)
  lst = mask1.tolist()
  training = 1
  
  _, acc = sess.run( [ train_step, accuracy ], feed_dict={ x: images, y_: labels, mask : lst,train_time : training  } )
  
  print( "step %d, training accuracy %g" % (i, acc) )
  
