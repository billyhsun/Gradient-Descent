import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time


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
    N = y.shape[0]
    res = (1/(2*N))* (np.linalg.norm(np.matmul(x, W) + b - y))**2 + (reg/2)*np.matmul(W.T, W)
    return float(res)


def gradMSE(W, b, x, y, reg):
    N = y.shape[0]
    dw = (1/N)*((((x@W).T)@x).T + (((b-y).T)@x).T) + reg * W
    db = np.sum((1/N)*(x@W + (b-y)))
    return dw, db


def crossEntropyLoss(W, b, x, y, reg):
    N = y.shape[0]
    loss = (1/N)*(-1*y * np.log(sigmoid(x@W + b)) - (1-y)*np.log(1-sigmoid(x@W+b)))
    loss = np.sum(loss) + (reg/2) * (np.linalg.norm(W))**2
    return loss

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-1*x))

def gradCE(W, b, x, y, reg):
    N = y.shape[0]
    db = np.sum((1/N)*(sigmoid(x@W+b)-y))
    dw = (1/N)*((x.T)@(sigmoid(x@W+b)-y))
    dw += reg*W
    return dw, db


def grad_descent(W, b, trainingData, trainingLabels, alpha, iterations, reg, EPS, lossType="None"):
    # Your implementation here
    epoch = 0
    W_old = W
    b_old = b
    weights = np.zeros((1, 784, 1))
    biases = np.zeros((1, 1, 1))
    while(epoch < iterations):
        if(lossType == "MSE"):
            gradW, gradB = gradMSE(W_old, b_old, trainingData, trainingLabels, reg)
        elif(lossType == "CE"):
            gradW, gradB = gradCE(W_old, b_old, trainingData, trainingLabels, reg)
        else:
            print("Invalid loss function")
            return

        weights = np.concatenate((weights, W_old[np.newaxis,:,:]), axis=0)
        biases = np.concatenate((biases, b_old[np.newaxis,:,:]), axis=0)

        W_new = W_old - alpha * gradW
        b_new = b_old - alpha * gradB

        if(np.sqrt(np.linalg.norm(W_new - W_old)**2 + np.linalg.norm(b_new - b_old)**2) < EPS):
            return W_new, b_new, weights, biases

        epoch = epoch + 1
        W_old = W_new
        b_old = b_new

    return W_new, b_new, np.array(weights), np.array(biases)


def buildGraph(beta1=None, beta2=None, epsilon=None, lossType=None, learning_rate=None):
    tf.set_random_seed(421)
    W = tf.Variable(tf.truncated_normal(shape = [784, 1],
                                   mean = 0, stddev=0.5), name='weights')
    #b = tf.Variable(tf.truncated_normal(shape = [1, 1],
    #                               mean = 0, stddev=0.5), name='biases')

    b = tf.Variable(tf.zeros([1,1]), name='bias')
    x = tf.placeholder(tf.float32, shape = [None, 784], name="x")
    y = tf.placeholder(tf.float32, shape = [None], name="y")
    y_pred = tf.add(tf.matmul(x, W),b, name = "predMSE")
    y_sig = tf.sigmoid(y_pred, name = "predSig")

    l = tf.placeholder(tf.float32, name='lambda')
    reg = tf.placeholder(tf.float32, name='reg')

    if(lossType=="MSE"):
        '''loss = tf.losses.mean_squared_error(
            labels=tf.reshape(y, [tf.shape(y)[0],1]),
            predictions=y_pred,
        ) + tf.multiply(tf.reduce_sum(tf.square(W)), reg/2)'''
        loss = tf.losses.mean_squared_error(
            labels=tf.reshape(y, [tf.shape(y)[0],1]),
            predictions=y_pred,
        ) + tf.multiply(tf.reduce_sum(tf.square(W)), reg/2)
    elif(lossType=="CE"):
        '''loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels = tf.reshape(y, [tf.shape(y)[0],1]),
            logits = y_pred,
            name = "loss_fn")) + tf.multiply(tf.reduce_sum(tf.square(W)), reg/2, name = "reg_part")'''
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels = tf.reshape(y, [tf.shape(y)[0],1]),
        logits = y_pred,
        name = "loss_fn")) + tf.multiply(tf.reduce_sum(tf.square(W)), reg/2, name = "reg_part")

    else:
        print("Invalid loss function")
        return

    optimizer =  tf.train.AdamOptimizer(learning_rate=learning_rate,
                                                   #beta1 = beta1,
                                                   #beta2 = beta2,
                                                   epsilon = epsilon
                                                   )
    train_step = optimizer.minimize(loss)

    return W, b, y_pred, y, loss, optimizer, x, reg, train_step, y_sig#, normal_MSE, reg_loss

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert inputs.shape[0] == targets.shape[0]
    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]



def compareAnalytical(X, y):
    X = np.concatenate((np.ones((3500, 1)), X), axis=1)
    X_pinv = np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T)
    W = np.matmul(X_pinv, y)
    b = W[0]
    W_opt = W[1:]
    return W_opt, b

def calcLossesAcc(W, b, weights, biases, trainData, valData, testData, trainTarget, valTarget, testTarget, reg, lossType=None):
    losses_train = []
    losses_val = []
    losses_test = []
    for i in range(np.shape(weights)[0]):
        if(lossType=="MSE"):
            losses_train.append(MSE(weights[i], biases[i], trainData.reshape(-1, 784), trainTarget, reg))
            losses_val.append(MSE(weights[i], biases[i], valData.reshape(-1, 784), valTarget, reg))
            losses_test.append(MSE(weights[i], biases[i], testData.reshape(-1, 784), testTarget, reg))
        elif(lossType=="CE"):
            losses_train.append(crossEntropyLoss(weights[i], biases[i], trainData.reshape(-1, 784), trainTarget, reg))
            losses_val.append(crossEntropyLoss(weights[i], biases[i], valData.reshape(-1, 784), valTarget, reg))
            losses_test.append(crossEntropyLoss(weights[i], biases[i], testData.reshape(-1, 784), testTarget, reg))
        else:
            print("Invalid loss type")
            return
    if(lossType=="MSE"):
        y_train = trainData.reshape(-1,784) @ W + b
        y_valid = valData.reshape(-1, 784) @ W + b
        y_test = testData.reshape(-1, 784) @ W + b
    elif(lossType=="CE"):
        y_train = sigmoid(trainData.reshape(-1,784) @ W + b)
        y_valid = sigmoid(valData.reshape(-1, 784) @ W + b)
        y_test = sigmoid(testData.reshape(-1, 784) @ W + b)
    else:
        print("Invalid loss type")
        return

    train_acc = np.sum(np.equal(trainTarget.squeeze() > 0.5, y_train.squeeze() > 0.5)) / np.shape(trainTarget)[0]
    val_acc = np.sum(np.equal(valTarget.squeeze() > 0.5, y_valid.squeeze() > 0.5)) / np.shape(valTarget)[0]
    test_acc = np.sum(np.equal(testTarget.squeeze() > 0.5, y_test.squeeze() > 0.5)) / np.shape(testTarget)[0]

    return losses_train, losses_val, losses_test, train_acc, val_acc, test_acc

def getAccs(weights, biases, trainData, valData, testData, trainTarget, valTarget, testTarget):
    train_accs = []
    val_accs = []
    test_accs = []
    for i in range(weights.shape[0]):
        y_train = trainData.reshape(-1,784) @ weights[i] + biases[i]
        y_valid = valData.reshape(-1, 784) @ weights[i] + biases[i]
        y_test = testData.reshape(-1, 784) @ weights[i] + biases[i]
        train_accs.append(np.sum(np.equal(trainTarget.squeeze() > 0.5, y_train.squeeze() > 0.5)) / np.shape(trainTarget)[0])
        val_accs.append(np.sum(np.equal(valTarget.squeeze() > 0.5, y_valid.squeeze() > 0.5)) / np.shape(valTarget)[0])
        test_accs.append(np.sum(np.equal(testTarget.squeeze() > 0.5, y_test.squeeze() > 0.5)) / np.shape(testTarget)[0])
    return train_accs, val_accs, test_accs

if __name__ == "__main__":
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()

    W = np.zeros((784, 1))
    #W = (1/100)*W
    b_train = np.random.rand(trainData.shape[0],1)
    b_val = np.random.rand(validData.shape[0],1)
    b_test = np.random.rand(testData.shape[0],1)
    #x = np.reshape(x, (x.shape[0], -1))


    print("{}    {}".format(MSE()))
    #W_train, b_train, loss_train = grad_descent(W, b_train, trainData, trainTarget, 0.0001, 5000, 0, 1e-7, "CE")
    #W_val, b_val, loss_val = grad_descent(W, b_val, validData, validTarget, 0.000001, 1000, 0, 0.3)

    #W_test, b_test, loss_test = grad_descent(W, b_test, testData, testTarget, 0.000001, 1000, 0, 0.3)

    #crossEntropyLoss(W, b_train, trainData, trainTarget, 0)


    #plt.plot(np.arange(len(loss_train)), loss_train)

    #x = np.arange(10)
    '''fig = plt.figure()
    plt.plot(loss_train)
    plt.title('Training loss')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.show()
    fig.savefig('plots/train.png')'''
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
