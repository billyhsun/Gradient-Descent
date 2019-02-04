from starter import *
from starter_test import *
from Main import *
import time


trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
W = np.random.rand(784, 1)/1000
b_train = np.random.rand(1,1)
trainData = trainData.reshape(-1, 784)
y = trainTarget

iterations = 5000
alpha = 0.0001
reg = 0
EPS=1e-7




# Part 1.3

alpha = [0.005, 0.001, 0.0001]

f=open("results.txt","w+")

f.write("Part 1.3 \n\n")

for a in alpha:
        t = time.time()
        W, b, weights, biases = grad_descent(W, b_train, trainData, trainTarget, a, iterations, 0, EPS, "MSE")
        training_time = time.time()-t
        fig = plt.figure()
        losses_train, losses_val, losses_test, train_acc, val_acc, test_acc = calcLossesAcc(weights, biases, trainData, validData,
         testData, trainTarget, validTarget, testTarget, 0, "MSE")

        plt.plot(losses_train, label='train')
        plt.plot(losses_val, label='validation')
        plt.plot(losses_test, label='test')

        plt.title('Learning Losses (MSE) with alpha={}'.format(a))
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='best')
        fig.savefig('plots/p13_alpha_{}.png'.format(a))
        s="a={}   time={} s\n train \t {}\n validation \t {}\n test \t {}\n".format(a, training_time,
        losses_train[-1], losses_val[-1], losses_test[-1])
        r="train_acc \t {}\n validation_acc \t {}\n test_acc \t {}\n".format(train_acc, val_acc, test_acc)
        print(s)
        print(r)
        f.write(s)
        f.write(r)

# Part 1.4

f.write("\nPart 1.4 \n\n")

lambdas = [0.001, 0.1, 0.5]
a = 0.005

for l in lambdas:
        t = time.time()
        W_train, b_train, weights, biases = grad_descent(W, b_train, trainData, trainTarget, a, iterations, l, EPS, "MSE")
        training_time = time.time() - t
        losses_train, losses_val, losses_test, train_acc, val_acc, test_acc = calcLossesAcc(weights, biases, trainData, validData,
         testData, trainTarget, validTarget, testTarget, l, "MSE")
        fig = plt.figure()
        plt.plot(losses_train, label='train')
        plt.plot(losses_val, label='validation')
        plt.plot(losses_test, label='test')
        plt.title('Learning Losses with lambda={}'.format(l))
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='best')
        fig.savefig('plots/p14_lambda_{}.png'.format(l))
        f.write("lambda={}   time={} s\n train \t {}\n validation \t {}\n test \t {}\n".format(a, training_time,
        losses_train[-1], losses_val[-1], losses_test[-1]))
        f.write("train_acc \t {}\n validation_acc \t {}\n test_acc \t {}\n".format(train_acc, val_acc, test_acc))

# Part 2.2

f.write("\nPart 2.2 \n\n")
l = 0.1
t = time.time()
W_train, b_train, weights, biases = grad_descent(W, b_train, trainData, trainTarget, a, iterations, l, EPS, "CE")
t = time.time() - t
losses_train, losses_val, losses_test, train_acc, val_acc, test_acc = calcLossesAcc(weights, biases, trainData, validData,
         testData, trainTarget, validTarget, testTarget, l, "CE")
fig = plt.figure()
plt.plot(losses_train, label='train')
plt.plot(losses_val, label='validation')
plt.plot(losses_test, label='test')
plt.title('Learning Losses (CE)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='best')
fig.savefig('plots/p22_loss.png')
f.write("lambda={}   time={} s\n train \t {}\n validation \t {}\n test \t {}\n".format(a, training_time,losses_train[-1], losses_val[-1], losses_test[-1])) 
f.write("train_acc \t {}\n validation_acc \t {}\n test_acc \t {}\n".format(train_acc, val_acc, test_acc))

f.close()