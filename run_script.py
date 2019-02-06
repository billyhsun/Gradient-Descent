from starter import *
from starter_test import *
from Main import *
import time


trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
W = np.random.rand(784, 1)/1000
b_train = np.random.rand(1,1)
trainData = trainData.reshape(-1, 784)
validData = validData.reshape(-1, 784)
testData = testData.reshape(-1, 784)

iterations = 5000
reg = 0
EPS=1e-7
'''



# Part 1.3

alpha = [0.005, 0.001, 0.0001]

f=open("results.txt","w+")

f.write("Part 1.3 \n\n")
'''
'''
for a in alpha:
        t = time.time()
        W, b, weights, biases = grad_descent(W, b_train, trainData, trainTarget, a, iterations, 0, EPS, "MSE")
        training_time = time.time()-t
        fig = plt.figure()
        losses_train, losses_val, losses_test, train_acc, val_acc, test_acc = calcLossesAcc(W, b, weights, biases, trainData, validData,
         testData, trainTarget, validTarget, testTarget, 0, "MSE")

        plt.plot(losses_train, label='train')
        plt.plot(losses_val, label='validation')
        plt.plot(losses_test, label='test')
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlim(-200, 5000)
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
'''

'''
# Part 1.4

f.write("\nPart 1.4 \n\n")

lambdas = [0.001, 0.1, 0.5]
a = 0.005

for l in lambdas:
        t = time.time()
        W, b, weights, biases = grad_descent(W, b_train, trainData, trainTarget, a, iterations, l, EPS, "MSE")
        training_time = time.time() - t
        losses_train, losses_val, losses_test, train_acc, val_acc, test_acc = calcLossesAcc(W, b, weights, biases, trainData, validData,
         testData, trainTarget, validTarget, testTarget, l, "MSE")
        fig = plt.figure()
        plt.plot(losses_train, label='train')
        plt.plot(losses_val, label='validation')
        plt.plot(losses_test, label='test')
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlim(-200, 5000)
        plt.title('Learning Losses with lambda={}'.format(l))
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='best')
        fig.savefig('plots/p14_lambda_{}.png'.format(l))
        s = "lambda={}   time={} s\n train \t {}\n validation \t {}\n test \t {}\n".format(a, training_time,
        losses_train[-1], losses_val[-1], losses_test[-1])
        r = "train_acc \t {}\n validation_acc \t {}\n test_acc \t {}\n".format(train_acc, val_acc, test_acc)
        f.write(s)
        f.write(r)
        print(s)
        print(r)

#Analytical
t_a = time.time()
W_a, b_a = compareAnalytical(trainData, trainTarget)
time_a = time.time()-t_a

y_train = trainData.reshape(-1,784) @ W_a + b
y_valid = validData.reshape(-1, 784) @ W_a + b
y_test = testData.reshape(-1, 784) @ W_a + b

train_acc = np.sum(np.equal(trainTarget.squeeze() > 0.5, y_train.squeeze() > 0.5)) / np.shape(trainTarget)[0]
val_acc = np.sum(np.equal(validTarget.squeeze() > 0.5, y_valid.squeeze() > 0.5)) / np.shape(validTarget)[0]
test_acc = np.sum(np.equal(testTarget.squeeze() > 0.5, y_test.squeeze() > 0.5)) / np.shape(testTarget)[0]

s="time={} s\n train \t {}\n validation \t {}\n test \t {}\n".format(time_a,
        MSE(W_a, b_a, trainData, trainTarget, 0), MSE(W_a, b_a, validData, validTarget, 0), MSE(W_a, b_a, testData, testTarget, 0))
r="train_acc \t {}\n validation_acc \t {}\n test_acc \t {}\n".format(train_acc, val_acc, test_acc)
f.write(s)
f.write(r)
print(s)
print(r)
'''
'''
# Part 2.2

f.write("\nPart 2.2 \n\n")
l = 0
a = 0.005
t = time.time()
W, b_train, weights, biases = grad_descent(W, b_train, trainData, trainTarget, a, iterations, l, EPS, "CE")
t = time.time() - t
losses_train, losses_val, losses_test, train_acc, val_acc, test_acc = calcLossesAcc(W, b_train, weights, biases, trainData, validData,
         testData, trainTarget, validTarget, testTarget, l, "CE")
train_accs, val_accs, test_accs = getAccs(weights, biases, trainData, validData, testData, trainTarget, validTarget, testTarget)

fig = plt.figure()
plt.plot(losses_train, label='train')
plt.plot(losses_val, label='validation')
plt.plot(losses_test, label='test')
ax = fig.add_subplot(1, 1, 1)
ax.set_xlim(-200, 5000)
plt.title('Learning Losses (CE)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='best')
fig.savefig('plots/p22_loss.png')

fig = plt.figure()
plt.plot(train_accs, label='train')
plt.plot(val_accs, label='validation')
plt.plot(test_accs, label='test')
ax = fig.add_subplot(1, 1, 1)
ax.set_xlim(-200, 5000)
plt.title('Learning Accuracies (CE)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='best')
fig.savefig('plots/p22_acc.png')


s = "a = {}  lambda={}   time={} s\n train \t {}\n validation \t {}\n test \t {}\n".format(a, l, training_time,losses_train[-1], losses_val[-1], losses_test[-1])
r = "train_acc \t {}\n validation_acc \t {}\n test_acc \t {}\n".format(train_acc, val_acc, test_acc)
print(s)
print(r)
'''
'''
# Part 2.3
a = 0.005
reg = 0

W, b_train, weights, biases = grad_descent(W, b_train, trainData, trainTarget, a, iterations, 0, EPS, "MSE")
losses_train, losses_val, losses_test, train_acc, val_acc, test_acc = calcLossesAcc(W, b_train, weights, biases, trainData, validData,
         testData, trainTarget, validTarget, testTarget, 0, "MSE")

fig = plt.figure()
plt.plot(losses_train, label='train')
plt.plot(losses_val, label='validation')
plt.plot(losses_test, label='test')
ax = fig.add_subplot(1, 1, 1)
ax.set_xlim(-200, 5000)
plt.title('Learning Losses (MSE)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='best')
fig.savefig('plots/p23_loss_MSE.png')
s = "a = {}  lambda={} \n train \t {}\n validation \t {}\n test \t {}\n".format(a, 0,losses_train[-1], losses_val[-1], losses_test[-1])
r = "train_acc \t {}\n validation_acc \t {}\n test_acc \t {}\n".format(train_acc, val_acc, test_acc)
print(s)
print(r)

W, b_train, weights, biases = grad_descent(W, b_train, trainData, trainTarget, a, iterations, 0, EPS, "CE")
losses_train, losses_val, losses_test, train_acc, val_acc, test_acc = calcLossesAcc(W, b_train, weights, biases, trainData, validData,
         testData, trainTarget, validTarget, testTarget, 0, "CE")

fig = plt.figure()
plt.plot(losses_train, label='train')
plt.plot(losses_val, label='validation')
plt.plot(losses_test, label='test')
ax = fig.add_subplot(1, 1, 1)
ax.set_xlim(-200, 5000)
plt.title('Learning Losses (CE)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='best')
fig.savefig('plots/p23_loss_CE.png')
s = "a = {}  lambda={}   time={} s\n train \t {}\n validation \t {}\n test \t {}\n".format(a, 0, training_time,losses_train[-1], losses_val[-1], losses_test[-1])
r = "train_acc \t {}\n validation_acc \t {}\n test_acc \t {}\n".format(train_acc, val_acc, test_acc)
print(s)
print(r)

'''


# Part 3.2
'''
batch_size = 500
epochs = 700
eval_every = 1
lossfcn = "MSE"
beta1 = 0.9
beta2 = 0.999
regularization = 0
learningrate = 0.001
eps = 1e-8


W, b, predMSE, y, loss, optimizer, x, reg, train_step, predSig = buildGraph(beta1=beta1, beta2=beta2, learning_rate=learningrate, lossType=lossfcn, epsilon=eps)
sess = tf.Session()

init = tf.global_variables_initializer()
sess.run(init)

trainlosslist = []
testlosslist = []
validlosslist = []
trainacclist = []
testacclist = []
relevantepoch = []
validacclist = []

xx = trainData
yy = trainTarget

t = 0
for epoch in range(epochs):
        batches = iterate_minibatches(xx,yy, batch_size, shuffle=True)
        for batch in batches:
            x_batch, y_batch = batch
            x_batch = x_batch.reshape(batch_size, 784)
            y_batch = y_batch.reshape(batch_size)
            sess.run(train_step, feed_dict={x: x_batch, y: y_batch, reg: regularization})
        if (epoch + 1) % eval_every == 0:
                if lossfcn == "MSE":
                        training_preds = sess.run(predMSE, feed_dict={x: xx})
                        testing_preds = sess.run(predMSE, feed_dict = {x: testData})
                        validation_preds = sess.run(predMSE, feed_dict = {x: validData})
                elif lossfcn == "CE":
                        training_preds = sess.run(predSig, feed_dict={x: xx})
                        testing_preds = sess.run(predSig, feed_dict = {x: testData})
                        validation_preds = sess.run(predSig, feed_dict={x: validData})

                training_acc = round(((yy == (training_preds > 0.5)).sum() / yy.shape[0]),3)
                testing_acc = round(((testTarget == (testing_preds > 0.5)).sum() / testTarget.shape[0]),3)
                valid_acc = round(((validTarget == (validation_preds > 0.5)).sum() / validTarget.shape[0]),3)

                train_loss = round(sess.run(loss, feed_dict={x: xx, y: yy.reshape(3500), reg: regularization}),3)
                test_loss = round(sess.run(loss, feed_dict={x: testData, y: testTarget.reshape(testTarget.shape[0]), reg:regularization}),3)
                valid_loss = round(sess.run(loss, feed_dict={x: validData, y: validTarget.reshape(validTarget.shape[0]), reg:regularization}),3)

                trainlosslist.append(train_loss)
                testlosslist.append(test_loss)
                validlosslist.append(valid_loss)

                trainacclist.append(training_acc)
                testacclist.append(testing_acc)
                validacclist.append(valid_acc)

                relevantepoch.append(epoch)
                t = t + 1
                if(epoch==699):
                        print(batch_size)
                        print("Epoch {} | Training Accuracy = {} | Testing Accuracy = {} | Training Loss = {} | Testing Loss = {} | Val Loss = {} | Val Acc = {}".format(epoch,training_acc, testing_acc, train_loss, test_loss,valid_loss,valid_acc))

relevantepoch = np.array(relevantepoch)
fig = plt.figure()
plt.plot(relevantepoch, trainlosslist, label='Training Loss')
plt.plot(relevantepoch, testlosslist, label = 'Test Loss')
plt.plot(relevantepoch, validlosslist, label = 'Validation Loss')
ax = fig.add_subplot(1, 1, 1)
#ax.set_xlim(-200, relevantepoch.shape[0] + 10)
plt.title('SGD Learning Losses ({}), alpha={}, lambda={}, batch_size{}'.format(lossfcn, learningrate, regularization, batch_size))
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='best')
fig.savefig('plots/p32_sgd_loss.png')

fig = plt.figure()
plt.plot(relevantepoch, trainacclist, label='Training Accuracy')
plt.plot(relevantepoch, validacclist, label = 'Validation Accuracy')
plt.plot(relevantepoch, testacclist, label = 'Testing Accuracy')
ax = fig.add_subplot(1, 1, 1)
#ax.set_xlim(-200, relevantepoch.shape[0] + 10)
plt.title('SGD Learning Accuracies ({}), alpha={}, lambda={}, batch_size={}'.format(lossfcn, learningrate, regularization, batch_size))
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='best')
fig.savefig('plots/p32_sgd_acc.png')
'''
#f.close()
batch_size = 500
epochs = 700
eval_every = 1
lossfcn = "MSE"
beta1 = None
beta2 = None
regularization = 0
learningrate = 0.001
epss = [1e-9, 1e-4]

for eps in epss:
        W, b, predMSE, y, loss, optimizer, x, reg, train_step, predSig = buildGraph(beta1=beta1, beta2=beta2, learning_rate=learningrate, lossType=lossfcn, epsilon=eps)
        sess = tf.Session()

        init = tf.global_variables_initializer()
        sess.run(init)

        trainlosslist = []
        testlosslist = []
        validlosslist = []
        trainacclist = []
        testacclist = []
        relevantepoch = []
        validacclist = []

        xx = trainData
        yy = trainTarget

        t = 0
        for epoch in range(epochs):
                batches = iterate_minibatches(xx,yy, batch_size, shuffle=True)
                for batch in batches:
                        x_batch, y_batch = batch
                        x_batch = x_batch.reshape(batch_size, 784)
                        y_batch = y_batch.reshape(batch_size)
                        sess.run(train_step, feed_dict={x: x_batch, y: y_batch, reg: regularization})
                if (epoch + 1) % eval_every == 0:
                        if lossfcn == "MSE":
                                training_preds = sess.run(predMSE, feed_dict={x: xx})
                                testing_preds = sess.run(predMSE, feed_dict = {x: testData})
                                validation_preds = sess.run(predMSE, feed_dict = {x: validData})
                        elif lossfcn == "CE":
                                training_preds = sess.run(predSig, feed_dict={x: xx})
                                testing_preds = sess.run(predSig, feed_dict = {x: testData})
                                validation_preds = sess.run(predSig, feed_dict={x: validData})

                        training_acc = round(((yy == (training_preds > 0.5)).sum() / yy.shape[0]),3)
                        testing_acc = round(((testTarget == (testing_preds > 0.5)).sum() / testTarget.shape[0]),3)
                        valid_acc = round(((validTarget == (validation_preds > 0.5)).sum() / validTarget.shape[0]),3)

                        train_loss = round(sess.run(loss, feed_dict={x: xx, y: yy.reshape(3500), reg: regularization}),3)
                        test_loss = round(sess.run(loss, feed_dict={x: testData, y: testTarget.reshape(testTarget.shape[0]), reg:regularization}),3)
                        valid_loss = round(sess.run(loss, feed_dict={x: validData, y: validTarget.reshape(validTarget.shape[0]), reg:regularization}),3)

                        trainlosslist.append(train_loss)
                        testlosslist.append(test_loss)
                        validlosslist.append(valid_loss)

                        trainacclist.append(training_acc)
                        testacclist.append(testing_acc)
                        validacclist.append(valid_acc)

                        relevantepoch.append(epoch)
                        t = t + 1
                        if (epoch==699):
                                print()
                                print("Epoch {} | Training Accuracy = {} | Testing Accuracy = {} | Training Loss = {} | Testing Loss = {} | Val Loss = {} | Val Acc = {} ".format(epoch,training_acc, testing_acc, train_loss, test_loss, valid_loss, valid_acc))

        relevantepoch = np.array(relevantepoch)
        fig = plt.figure()
        plt.plot(relevantepoch, trainlosslist, label='Training Loss')
        plt.plot(relevantepoch, testlosslist, label = 'Test Loss')
        plt.plot(relevantepoch, validlosslist, label = 'Validation Loss')
        ax = fig.add_subplot(1, 1, 1)
        plt.title('SGD Learning Losses ({}), alpha={}, epsilon={}, batch_size={}'.format(lossfcn, learningrate, eps, batch_size))
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='best')
        fig.savefig('plots/p34c_sgd_loss_eps_{}.png'.format(eps))

        fig = plt.figure()
        plt.plot(relevantepoch, trainacclist, label='Training Accuracy')
        plt.plot(relevantepoch, validacclist, label = 'Validation Accuracy')
        plt.plot(relevantepoch, testacclist, label = 'Testing Accuracy')
        ax = fig.add_subplot(1, 1, 1)
        #ax.set_xlim(-200, relevantepoch.shape[0] + 10)
        plt.title('SGD Learning Accuracies ({}), alpha={}, epsilon={}, batch_size={}'.format(lossfcn, learningrate, eps, batch_size))
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc='best')
        fig.savefig('plots/p34c_sgd_acc_eps_{}.png'.format(eps))

