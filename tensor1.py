# numpy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
x = np.array([25, 2, 5])
y = np.array([0, 1, 2])
np.dot(x, y)  # 25*0 + 2*1 + 5*2

# TensorFlow
import tensorflow as tf
x_tf = tf.Variable([25, 2, 5])
y_tf = tf.Variable([0, 1, 2])
# dot product of x * y
print(tf.reduce_sum(tf.multiply(x_tf, y_tf)))

A_tf = tf.Variable([[3, 4.0], [5, 6.0], [7, 8.0]])
B_tf = tf.Variable([[1., 3.], [2., 4.]])

print("== Print A , B and shapes")
print(A_tf)
print(A_tf.shape)

print(B_tf)
print(B_tf.shape)

print("== Print output matrix with zeros")
print(tf.zeros((A_tf.shape[0], B_tf.shape[1])))

print("== Print output of tensordot(.. axes=1)")
print(tf.tensordot(A_tf, B_tf, axes=1))

print("== Print output of tf.matmul(A_tf, B_tf)")
print(tf.matmul(A_tf, B_tf))

print("== Print matrix * vector:  tf.linalg.matvec(A_tf, B_tf[:,0])")
print("B column[0]: ", B_tf[:,0])
print(tf.linalg.matvec(A_tf, B_tf[:,0]))

print("== Calc output entry wise, using tf.reduce_sum + tf.multiply of sum(row i of A  * column k of B)")
D_tf = tf.Variable([[ float(tf.reduce_sum(tf.multiply(A_tf[i, :], B_tf[:, k]))) for k in range(B_tf.shape[1])] for i in range(A_tf.shape[0])])
print(D_tf)
print("== Calc output entry wise, using direct sum and multiply: sum( [ (A_tf[i,j] * B_tf[j,k]) for j in range(A_tf.shape[1]) ]")
E_tf = tf.Variable([[ sum( [ (A_tf[i,j] * B_tf[j,k]) for j in range(A_tf.shape[1]) ] ) for k in range(B_tf.shape[1])] for i in range(A_tf.shape[0])])
print(E_tf)

print("== Calc output entry wise in a np array using tf.reduce_sum + tf.multiply of sum(row i of A  * column k of B)")
A = np.zeros((A_tf.shape[0], B_tf.shape[1]))

for i in range(A_tf.shape[0]):
    for k in range(B_tf.shape[1]):
        A[i,k] = float(tf.reduce_sum(tf.multiply(A_tf[i, :], B_tf[:, k])))

print(A)

print("== convert to tf.Variable (Tensor)")
F_ft = tf.Variable(A)
print(F_ft)


