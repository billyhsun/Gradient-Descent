import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def loadData():
    with np.load('notMNIST.npz') as data:
        Data, Target = data['images'], data['labels']
        posClass = 2
        negClass = 9
        dataIndx = (Target == posClass) + (Target == negClass)
        Data = Data[dataIndx]/255.
        Target = Target[dataIndx].reshape(-1, 1)
        Target[Target == posClass] = 1
        Target[Target == negClass] = 0
        np.random.seed(421)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget


def MSE(W, b, x, y, reg):
    x = np.reshape(x, (x.shape[0], -1))
    N = y.shape[0]
    res = (1/(2*N))* (np.linalg.norm(np.matmul(x, W) + b - y))**2 + (reg/2)*np.matmul(W.T, W)
    return float(res)


def gradMSE(W, b, x, y, reg):
    x = np.reshape(x, (x.shape[0], -1))
    N = y.shape[0]
    dw = (1/N)*((((x@W).T)@x).T + (((b-y).T)@x).T) + reg * W
    db = np.sum((1/N)*(x@W + 2*(b-y)))
    return dw, db


def crossEntropyLoss(W, b, x, y, reg):
    x = np.reshape(x, (x.shape[0], -1))
    N = y.shape[0]
    a = x@W+b
    b = sigmoid(a)
    c = np.log(b)
    sum = 0
    for i in range(N):
        sum += -1*y[i]*np.log(sigmoid(W.T@x[i])) - (1-y[i])*np.log(1 - sigmoid(W.T@x[i]))
    sum = sum/N
    sum = sum + (reg/2) * (np.linalg.norm(W))**2
    return sum
    #print("{}, {}, {}".format(a[0], b[0], c[0]))
    #return (1/N)*tf.sum(-1*(y.T)@tf.log(tf.sigmoid(x@W + b)) - ((1-y).T)@tf.log(1-tf.sigmoid(x@W + b)))
    #+ (reg/2)*(np.linalg.norm(W))**2

def sigmoid(x):
    #print(np.exp(-1*x))
    #print(1/(1 + np.exp(-x)))
    return 1.0/(1.0 + np.exp(-x))

def gradCE(W, b, x, y, reg):
    x = np.reshape(x, (x.shape[0], -1))
    N = y.shape[0]
    db = np.sum((1/N)*(sigmoid(x@W+b)-y))
    dw = (1/N)*((x.T)@(sigmoid(x@W+b)-y))
    return dw, db


def grad_descent(W, b, trainingData, trainingLabels, alpha, iterations, reg, EPS, lossType="None"):
    # Your implementation here
    epoch = 0
    W_old = W
    b_old = b
    losses = []
    while(epoch < iterations):
        if(lossType == "MSE"):
            gradW, gradB = gradMSE(W_old, b_old, trainingData, trainingLabels, reg)
            #print(gradW)
        elif(lossType == "CE"):
            gradW, gradB = gradCE(W_old, b_old, trainingData, trainingLabels, reg)
            #print(gradW)
        else:
            print("Invalid loss function")
            return

        W_new = W_old - alpha * gradW
        b_new = b_old - alpha * gradB

        if(lossType == "MSE"):
            loss = MSE(W_new, b_new, trainingData, trainingLabels, reg)
            #print(loss)
        else:
            loss = crossEntropyLoss(W_new, b_new, trainingData, trainingLabels, reg)
            #print(loss)

        losses.append(loss)

        if((np.linalg.norm(W_new - W_old)) < EPS):
            return W_new, b_new, losses

        epoch = epoch + 1
        W_old = W_new
        b_old = b_new
    return W_new, b_new, losses


def buildGraph(beta1=None, beta2=None, epsilon=None, lossType=None, learning_rate=None):
    tf.set_random_seed(421)
    W = tf.Variable(tf.truncated_normal(stddev=0.5), name='weights')
    b = tf.Variable(tf.truncated_normal(stddev=0.5), name='biases')

    y = tf.placeholder(tf.float32, name='y')
    y_predicted = tf.matmul(X,W) + b

    l = tf.placeholder(tf.float32, name='lambda')
    x = tf.placeholder(tf.float32, name='x')

    optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
    regularizer = tf.nn.l2loss(weights)

    if(lossType=="MSE"):
        MSE = tf.reduce_mean(tf.mean_squared_error(labels=y, predictions=y_predicted) + epsilon*regularizer,name='cross_entropy_error')
        train = optimizer.minimize(loss=MSE)
    elif(lossType=="CE"):
        CE = tf.reduce_mean(tf.sigmoid_cross_entropy() + epsilon*regularizer,name='cross_entropy_error')
        train = optimizer.minimize(loss=CE)
    else:
        print("Invalid loss function")
        return

    #train = optimizer.minimize(loss=)


def solveLeastSquares(W, b, x, y):
    x = np.reshape(x, (x.shape[0], -1))
    ans = (np.inv(((x.T)@x)))
    return ()


if __name__ == "__main__":
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()

    W = np.zeros((784, 1))
    #W = (1/100)*W
    b_train = np.random.rand(trainData.shape[0],1)
    b_val = np.random.rand(validData.shape[0],1)
    b_test = np.random.rand(testData.shape[0],1)

    W_train, b_train, loss_train = grad_descent(W, b_train, trainData, trainTarget, 0.0001, 5000, 0, 1e-7, "CE")
    #W_val, b_val, loss_val = grad_descent(W, b_val, validData, validTarget, 0.000001, 1000, 0, 0.3)

    #W_test, b_test, loss_test = grad_descent(W, b_test, testData, testTarget, 0.000001, 1000, 0, 0.3)

    #crossEntropyLoss(W, b_train, trainData, trainTarget, 0)


    #plt.plot(np.arange(len(loss_train)), loss_train)

    #x = np.arange(10)
    fig = plt.figure()
    plt.plot(loss_train)
    plt.title('Training loss')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.show()
    fig.savefig('plots/train.png')
'''
    fig = plt.figure()
    plt.plot(loss_val)
    plt.title('Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.show()
    fig.savefig('plots/val.png')

    fig = plt.figure()
    plt.plot(loss_test)
    plt.title('Test loss')
    plt.xlabel('Epoch')
    plt.ylabel('Test Loss')
    plt.show()
    fig.savefig('plots/test.png')





    #plt.plot(loss_val)
    #plt.plot(loss_test)

    #print(np.matmul(x, W_final) + b_final - trainTarget)
    #print(np.linalg.norm(np.matmul(x, W_final) + b_final - trainTarget))




    #reg = 0
    #trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()

   #print("gradw: {} \n gradb: {}".format(gradMSE(W, b, x, y, reg)[0], gradMSE(W, b, x, y, reg)[1]))
'''
