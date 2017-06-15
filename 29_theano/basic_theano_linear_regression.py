# https://github.com/marekrei/theano-tutorial/blob/master/minimal_training_example.py
import theano
import numpy

x = theano.tensor.fvector('x')
target = theano.tensor.fscalar('target')

W = theano.shared(numpy.asarray([0.2, 0.7]), 'W')# init w1, w2
y = (x * W).sum()

cost = theano.tensor.sqr(target - y)
gradients = theano.tensor.grad(cost, [W])
W_updated = W - (0.1 * gradients[0])
updates = [(W, W_updated)]

f = theano.function([x, target], y, updates=updates)

for i in range(10):
    output = f([1.0, 1.0], 20.0)# x1=1, x2=1, y=20
    print(output)
