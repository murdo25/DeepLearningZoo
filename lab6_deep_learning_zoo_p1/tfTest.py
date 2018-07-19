import tensorflow as tf

#x = tf.placeholder("float", 3)
x = tf.placeholder(3)

print(x[0])
if(x[0] == 1):
	y = x * 2
else:
	y = x * 400

with tf.Session() as session:
    result = session.run(y, feed_dict={x: [1, 2, 3]})
    print(result)
