from starter import *
from starter_test import *




trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()

W = np.zeros((784, 1))
b_train = np.zeros((trainData.shape[0],1))
b_val = np.zeros((validData.shape[0],1))
b_test = np.zeros((testData.shape[0],1))
'''trainData2 = np.reshape(trainData, (trainData.shape[0], -1))
print("{}    {}\n".format(MSE(W, b_train, trainData, trainTarget, 0.1),
MSE_test(W, b_train, trainData2, trainTarget, 0.1)))

print("{}    {}\n".format(crossEntropyLoss(W, b_train, trainData, trainTarget, 0.1),
crossEntropyLoss_test(W, b_train, trainData2, trainTarget, 0.1)))

dw1, db1 = gradMSE(W, b_train, trainData, trainTarget, 0.5)
dw2, db2 = gradMSE_test(W, b_train, trainData2, trainTarget, 0.5)
print(np.linalg.norm(dw1-dw2))
print(db1-np.sum(db2))
#print("{}    {}\n".format(gradMSE(W, b_train, trainData, trainTarget, 0.1),
#radMSE_test(W, b_train, trainData2, trainTarget, 0.1)))

dw1, db1 = gradCE(W, b_train, trainData, trainTarget, 0.5)
dw2, db2 = gradCE_test(W, b_train, trainData2, trainTarget, 0.5)
print(np.linalg.norm(dw1-dw2))
print(db1-np.sum(db2))

#print("{}    {}\n".format(gradCE(W, b_train, trainData, trainTarget, 0.1),
#gradCE_test(W, b_train, trainData2, trainTarget, 0.1)))

'''
# Part 1.3

alpha = [0.005, 0.001, 0.0001]

f=open("results.txt","w+")

f.write("Part 1.3 \n\n")

for a in alpha:
        W_train, b_train, losses_train = grad_descent(W, b_train, trainData, trainTarget, a, 5000, 0, 1e-7, "MSE")
        W_valid, b_valid, losses_valid = grad_descent(W, b_val, validData, validTarget, a, 5000, 0, 1e-7, "MSE")
        W_test, b_test, losses_test = grad_descent(W, b_test, testData, testTarget, a, 5000, 0, 1e-7, "MSE")
        fig = plt.figure()
        plt.plot(losses_train, label='train')
        plt.plot(losses_valid, label='validation')
        plt.plot(losses_test, label='test')
        plt.title('Learning Losses (MSE) with alpha={}'.format(a))
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='best')
        fig.savefig('plots/p13_alpha_{}.png'.format(a))
        f.write("a={}\n train \t {}\n validation \t {}\n test \t {}\n".format(a,
        losses_train[-1], losses_valid[-1], losses_test[-1]))

# Part 1.4

f.write("\nPart 1.4 \n\n")

lambdas = [0.001, 0.1, 0.5]
a = 0.005

for l in lambdas:
        W_train, b_train, losses_train = grad_descent(W, b_train, trainData, trainTarget, a, 5000, l, 1e-7, "MSE")
        W_valid, b_valid, losses_valid = grad_descent(W, b_val, validData, validTarget, a, 5000, l, 1e-7, "MSE")
        W_test, b_test, losses_test = grad_descent(W, b_test, testData, testTarget, a, 5000, l, 1e-7, "MSE")
        fig = plt.figure()
        plt.plot(losses_train, label='train')
        plt.plot(losses_valid, label='validation')
        plt.plot(losses_test, label='test')
        plt.title('Learning Losses with lambda={}'.format(l))
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='best')
        fig.savefig('plots/p14_lambda_{}.png'.format(l))
        f.write("lambda={}\n train \t {}\n validation \t {}\n test \t {}\n".format(l,
        losses_train[-1], losses_valid[-1], losses_test[-1]))

# Part 2.2

l = 0.1

W_train, b_train, losses_train = grad_descent(W, b_train, trainData, trainTarget, a, 5000, l, 1e-7, "CE")
W_valid, b_valid, losses_valid = grad_descent(W, b_val, validData, validTarget, a, 5000, l, 1e-7, "CE")
W_test, b_test, losses_test = grad_descent(W, b_test, testData, testTarget, a, 5000, l, 1e-7, "CE")
fig = plt.figure()
plt.plot(losses_train, label='train')
plt.plot(losses_valid, label='validation')
plt.plot(losses_test, label='test')
plt.title('Learning Losses (CE)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='best')
fig.savefig('plots/p22_loss.png')
f.write("lambda={}\n train \t {}\n validation \t {}\n test \t {}\n".format(l, losses_train[-1], losses_valid[-1], losses_test[-1]))