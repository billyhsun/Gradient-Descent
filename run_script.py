import starter.py


trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()

W = np.zeros((784, 1))
b_train = np.zeros(trainData.shape[0],1)
b_val = np.zeros(validData.shape[0],1)
b_test = np.zeros(testData.shape[0],1)

# Part 1.3

alpha = [0.005, 0.001, 0.0001]

f=open("results.txt","w+")

f.write("Part 1.3 \n\n")

for a in alpha:
        W_train, b_train, losses_train = grad_descent(W, b_train, trainData, trainTarget, a, 5000, 0, 1e-7, "MSE")
        W_valid, b_valid, losses_valid = grad_descent(W, b_valid, validData, validTarget, a, 5000, 0, 1e-7, "MSE")
        W_test, b_test, losses_test = grad_descent(W, b_test, testData, testTarget, a, 5000, 0, 1e-7, "MSE")
        plt.plot(losses_train, label='train')
        plt.plot(losses_valid, label='validation')
        plt.plot(losses_test, label='test')
        plt.title('Learning Losses (MSE) with alpha={}'.format(a))
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='best')
        fig.savefig('plots/p13_alpha_{}.png'.format(a))
        f.write("a={}\n train \t {}\n validation \t {}\n test \t {}\n".format(a,
        losses_train[-1], losses_valid, losses_test)

# Part 1.4

f.write("\nPart 1.4 \n\n")

lambdas = [0.001, 0.1, 0.5]
a = 0.005

for l in lambdas:
        W_train, b_train, losses_train = grad_descent(W, b_train, trainData, trainTarget, a, 5000, l, 1e-7, "MSE")
        W_valid, b_valid, losses_valid = grad_descent(W, b_valid, validData, validTarget, a, 5000, l, 1e-7, "MSE")
        W_test, b_test, losses_test = grad_descent(W, b_test, testData, testTarget, a, 5000, l, 1e-7, "MSE")
        plt.plot(losses_train, label='train')
        plt.plot(losses_valid, label='validation')
        plt.plot(losses_test, label='test')
        plt.title('Learning Losses with lambda={}'.format(l))
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='best')
        fig.savefig('plots/p14_lambda_{}.png'.format(l))
        f.write("lambda={}\n train \t {}\n validation \t {}\n test \t {}\n".format(l,
        losses_train[-1], losses_valid[-1], losses_test[-1])

# Part 2.2

l = 0.1

W_train, b_train, losses_train = grad_descent(W, b_train, trainData, trainTarget, a, 5000, l, 1e-7, "CE")
W_valid, b_valid, losses_valid = grad_descent(W, b_valid, validData, validTarget, a, 5000, l, 1e-7, "CE")
W_test, b_test, losses_test = grad_descent(W, b_test, testData, testTarget, a, 5000, l, 1e-7, "CE")
plt.plot(losses_train, label='train')
plt.plot(losses_valid, label='validation')
plt.plot(losses_test, label='test')
plt.title('Learning Losses (CE)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='best')
fig.savefig('plots/p22_loss.png')
f.write("lambda={}\n train \t {}\n validation \t {}\n test \t {}\n".format(l,
losses_train[-1], losses_valid[-1], losses_test[-1])
