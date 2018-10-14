import tensorflow as tf

X = tf.Variable(3, name='X')
Y = tf.Variable(4, name='Y')
f = X * X * Y + Y + 2

session = tf.Session()
session.run(X.initializer)
session.run(Y.initializer)
print(session.run(f))
session.close()


