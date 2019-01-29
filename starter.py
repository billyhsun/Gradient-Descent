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
    # Your implementation here
    x = np.reshape(x, (x.shape[0], -1))
    N = y.shape[1]
    res = (1/(2*N))*np.matmul((np.matmul(x, W) + b - y).transpose(), (np.matmul(x, W) + b - y)) + (reg/2)*np.matmul(W.transpose(), W)
    return float(res)


def gradMSE(W, b, x, y, reg):
    x = np.reshape(x, (x.shape[0], -1))
    return 2*np.matmul(np.matmul(x, W).T, x).T + 2*np.matmul((b-y).T, x).T + reg * W, 2*np.matmul(x, W) + 2*(b-y)
    # Your implementation here


def crossEntropyLoss(W, b, x, y, reg):
    # Your implementation here
    pass


def gradCE(W, b, x, y, reg):
    # Your implementation here
    pass


def grad_descent(W, b, trainingData, trainingLabels, alpha, iterations, reg, EPS):
    # Your implementation here
    epoch = 0
    W_old = W
    b_old = b
    loss = []
    while(epoch < iterations):
        gradW, gradB = gradMSE(W_old, b_old, trainingData, trainingLabels, reg)
        W_new = W_old - alpha * gradW
        b_new = b_old - alpha * np.sum(gradB)
        if(epoch % 10 == 0):
            loss.append(MSE(W_new, b_new, trainingData, trainingLabels, reg))
        # check EPS, break if it doesnt work out
        epoch = epoch + 1
        W_old = W_new
        b_old = b_new
    return W_new, b_new, loss


def buildGraph(beta1=None, beta2=None, epsilon=None, lossType=None, learning_rate=None):
    # Your implementation here
    pass


if __name__ == "__main__":
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    
    W = np.random.rand(784, 1)
    b_train = np.random.rand(trainData.shape[0],1)
    b_val = np.random.rand(validData.shape[0],1)
    b_test = np.random.rand(testData.shape[0],1)
    
    W_train, b_train, loss_train = grad_descent(W, b_train, trainData, trainTarget, 0.000001, 1000, 0, 0.3)

    W_val, b_val, loss_val = grad_descent(W, b_val, validData, validTarget, 0.000001, 1000, 0, 0.3)
    
    W_test, b_test, loss_test = grad_descent(W, b_test, testData, testTarget, 0.000001, 1000, 0, 0.3)



    #plt.plot(np.arange(len(loss_train)), loss_train)

    #x = np.arange(10)
    fig = plt.figure()
    plt.plot(loss_train)
    plt.title('Training loss')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.show()
    fig.savefig('plots/train.png')

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

