import numpy as np
import matplotlib.pyplot as plt
import Dataset as YY

####################################

def sof_relu(x, thresh=15):
    res = x.copy()
    ind = np.abs(x) < thresh
    res[x < -thresh] = 0
    res[ind] = np.log(1 + np.exp(x[ind]))
    return res

def soft_relu_deriv(x, thresh=15):
    res = np.ones_like(x)
    ind = np.abs(x) < thresh
    res[x < -thresh] = 0
    res[ind] = np.exp(x[ind])/(1 + np.exp(x[ind]))
    return res

class SoftReLULayer(object):
    def forward(self, input):
        # remember the input for later backpropagation
        self.input = input
        # return the ReLU of the input
        return sof_relu(input)

    def backward(self, upstream_gradient):
        # compute the derivative of ReLU from upstream_gradient and the stored input
        #z_ij = relu(input_ij) -> z_ij' = relu'(input_ij)input_ij'
        downstream_gradient = upstream_gradient*soft_relu_deriv(self.input)
        return downstream_gradient

    def update(self, learning_rate):
        pass # parameter-free


####################################

class ReLULayer(object):
    def forward(self, input):
        # remember the input for later backpropagation
        self.input = input
        # return the ReLU of the input
        relu = np.maximum(np.zeros_like(input), input) #!!!!!!!!!! your code here
        return relu

    def backward(self, upstream_gradient):
        # compute the derivative of ReLU from upstream_gradient and the stored input
        #z_ij = relu(input_ij) -> z_ij' = relu'(input_ij)input_ij'
        relu_deriv = np.maximum(np.zeros_like(self.input), self.input)
        relu_deriv[relu_deriv>0] = 1
        downstream_gradient = upstream_gradient*relu_deriv #!!!!!!!!!!! your code here
        return downstream_gradient

    def update(self, learning_rate):
        pass # ReLU is parameter-free


####################################

class SigmoidLayer(object):
    def forward(self, input):
        # remember the input for later backpropagation
        self.input = input
        # return the ReLU of the input
        sigmoid = 1/(1+np.exp(-input))
        return sigmoid

    def backward(self, upstream_gradient):
        # compute the derivative of sigmoid from upstream_gradient and the stored input
        #z_ij = sigmoid(input_ij) -> z_ij' = sigmoid'(input_ij)input_ij'
        sig_deriv = np.exp(-self.input)/(1+np.exp(-self.input))**2
        downstream_gradient = upstream_gradient*sig_deriv
        return downstream_gradient

    def update(self, learning_rate):
        pass # parameter-free

####################################

class XEntropySoftmaxLayer(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def forward(self, input):
        # remember the input for later backpropagation
        self.input = input
        # return the softmax of the input
        diff = input - np.max(input, axis=1)[:, np.newaxis]
        softmax = np.exp(diff)/np.expand_dims(np.sum(np.exp(diff), axis=1), axis=1) #!!!!!!!!!! your code here
        return softmax

    def backward(self, predicted_posteriors, true_labels):
        # return the loss derivative with respect to the stored inputs
        # (use cross-entropy loss and the chain rule for softmax,
        #  as derived in the lecture)
        #loss = -sum_ik[y_ik*log(f_ik)], f_ik = exp(z_ik)/sum_j(exp(z_ij), z_ik = input_ik, f_ik = predicted_posterior_ik
        #     = -sum_ik[y_ik*z_ik - y_ik*log(sum_j(exp(z_ij)))]
        #loss' (wrt z_ik) = -y_ik + sum_j[y_ij*f_ik] = -y_ik + f_ik
        downstream_gradient = -true_labels + predicted_posteriors
        return +downstream_gradient

    def update(self, learning_rate):
        pass # softmax is parameter-free


####################################

class SquaredErrorSoftReLULayer(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def forward(self, input):
        # remember the input for later backpropagation
        self.input = input
        # return the softrelu of the input
        return sof_relu(input)

    def backward(self, predicted_output, true_values):
        # return the loss derivative with respect to the stored inputs
        # (squared error loss and the chain rule for softrelu)
        #loss = sum_ik[(y_ik-f_ik)**2], f_ik = log(1+exp(z_ik)), z_ik = input_ik, f_ik = predicted_output_ik
        #loss' (wrt z_ik) = = 2*(y_ik-f_ik)*exp(z_ik)/(1+exp(z_ik)
        downstream_gradient = 2*(predicted_output-true_values)*soft_relu_deriv(self.input)
        return downstream_gradient

    def update(self, learning_rate):
        pass # softmax is parameter-free


####################################

class LinearLayer(object):
    def __init__(self, n_inputs, n_outputs, bias_on=True):
        self.n_inputs  = n_inputs
        self.n_outputs = n_outputs
        self.bias_on = bias_on
        # randomly initialize weights and intercepts
        self.B = np.random.normal(0, 0.7, (n_outputs, n_inputs))
        if bias_on:
            self.b = np.random.normal(0, 0.7, (n_outputs))
        else:
            self.b = np.zeros(n_outputs)

    def forward(self, input):
        # remember the input for later backpropagation
        self.input = input
        # compute the scalar product of input and weights
        # (these are the preactivations for the subsequent non-linear layer)
        preactivations = np.dot(self.B, input.T).T+np.expand_dims(self.b, axis=0)
        return preactivations

    def backward(self, upstream_gradient):
        # compute the derivative of the weights from
        # upstream_gradient and the stored input
        #a_ij = sum_k (B_jk*input_ik + b_j), a_ij = preactivations_ij
        #d a_ij/ d b_l = d_jl
        self.grad_b = np.sum(upstream_gradient, axis=0)
        #d a_ij/ d B_lm = d_jl*input_im
        self.grad_B = np.dot(upstream_gradient.T, self.input)
        # compute the downstream gradient to be passed to the preceding layer
        downstream_gradient = np.dot(upstream_gradient, self.B)
        return downstream_gradient

    def update(self, learning_rate):
        # update the weights by batch gradient descent
        self.B = self.B - learning_rate * self.grad_B
        if self.bias_on:
            self.b = self.b - learning_rate * self.grad_b

####################################

class MLP(object):
    def __init__(self, n_features, layer_sizes, act="ReLU", bias_on=True, classify=True, track_weigths=False):
        # constuct a multi-layer perceptron
        # with sigmoid activation in the hidden layers and softmax output
        # (i.e. it predicts the posterior probability of a classification problem)
        #
        # n_features: number of inputs
        # len(layer_size): number of layers
        # layer_size[k]: number of neurons in layer k
        # (specifically: layer_sizes[-1] is the number of classes)
        self.n_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes
        self.layers   = []

        # create interior layers (linear + ReLU)
        n_in = n_features
        for n_out in layer_sizes[:-1]:
            self.layers.append(LinearLayer(n_in, n_out, bias_on=bias_on))
            if act=="SoftReLU":
                self.layers.append(SoftReLULayer())
            elif act=="Sigmoid":
                self.layers.append(SigmoidLayer())
            else:
                self.layers.append(ReLULayer())
            n_in = n_out

        # create last linear layer + output layer
        n_out = layer_sizes[-1]
        self.layers.append(LinearLayer(n_in, n_out))
        if classify:
            self.layers.append(XEntropySoftmaxLayer(n_out))
        else:
            self.layers.append(SquaredErrorSoftReLULayer(n_out))
        self.classify = classify
        self.track_weights = track_weigths

    def forward(self, X):
        # X is a mini-batch of instances
        batch_size = X.shape[0]
        # flatten the other dimensions of X (in case instances are images)
        X = X.reshape(batch_size, -1)

        # compute the forward pass
        # (implicitly stores internal activations for later backpropagation)
        result = X
        for layer in self.layers:
            result = layer.forward(result)
        return result

    def backward(self, predicted_posteriors, true_classes):
        # perform backpropagation w.r.t. the prediction for the latest mini-batch X
        downstream_gradient = self.layers[-1].backward(predicted_posteriors, true_classes) # !!!!!!!!!!! your code here
        for layer in reversed(self.layers[:-1]):
            downstream_gradient = layer.backward(downstream_gradient)
            
    def update(self, X, Y, learning_rate):
        posteriors = self.forward(X)
        if self.classify:
            y = np.zeros((len(Y), self.layers[-1].n_classes))
            y[np.arange(len(Y)),Y] = 1
            loss = -np.sum(np.sum(y * np.log(posteriors + np.ones_like(posteriors) * 10 ** -10), axis=1), axis=0) / len(
                X)
        else:
            y = Y
            loss = np.sum((y-posteriors)**2) / len(X)
        global loss_rec, weights_rec
        global idx
        loss_rec[idx] = loss
        if self.track_weights:
            for n in range(self.n_layers):
                weights_rec[n][idx] = self.layers[2*n].B
        idx += 1
        #print(loss)
        self.backward(posteriors, y)
        for layer in self.layers:
            layer.update(learning_rate)

    def train(self, x, y, n_epochs, batch_size, learning_rate):
        N = len(x)
        n_batches = N // batch_size
        global loss_rec
        global weights_rec
        global idx
        loss_rec = np.zeros(n_epochs * N // batch_size)
        if self.track_weights:
            weights_rec = [np.zeros((n_epochs * N // batch_size, self.layers[n].B.shape[0], self.layers[n].B.shape[1])) for n in range(len(self.layers)) if n%2==0]
        idx = 0
        for i in range(n_epochs):
            # print("Epoch", i)
            # reorder data for every epoch
            # (i.e. sample mini-batches without replacement)
            permutation = np.random.permutation(N)

            for batch in range(n_batches):
                # create mini-batch
                start = batch * batch_size
                x_batch = x[permutation[start:start+batch_size]]
                y_batch = y[permutation[start:start+batch_size]]

                # perform one forward and backward pass and update network parameters

                self.update(x_batch, y_batch, learning_rate)
            #print("%d/%d epochs done"%(i, n_epochs))

##################################

loss_rec = None
weights_rec = None
idx = 0

def ewma(data, window):
    alpha = 2 /(window + 1.0)
    alpha_rev = 1-alpha
    n = data.shape[0]
    pows = alpha_rev**(np.arange(n+1))
    scale_arr = 1/pows[:-1]
    offset = data[0]*pows[1:]
    pw0 = alpha*alpha_rev**(n-1)
    mult = data*pw0*scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums*scale_arr[::-1]
    return out

def run(N, n_epochs, eta, batch_size, i=0):

    # create training and test data
    X_train, Y_train = YY.YinYangDataset(bottom_left=0, top_right=1, size=N, flipped_coords=True)[:]
    X_test, Y_test = YY.YinYangDataset(bottom_left=0, top_right=1, size=N, flipped_coords=True)[:]

    n_features = 4
    n_classes = 3

    # standardize features to be in [-1, 1]
    # offset  = X_train.min(axis=0)
    # scaling = X_train.max(axis=0) - offset#!!!!!!!!!! your code here
    # X_train = ((X_train - offset) / scaling - 0.5) * 2.0
    # X_test  = ((X_test  - offset) / scaling - 0.5) * 2.0

    # set hyperparameters (play with these!)
    layer_sizes = [120, n_classes]
    learning_rate = eta

    # create network
    network = MLP(n_features, layer_sizes, act="SoftReLU", bias_on=False)

    # train
    network.train(X_train, Y_train, n_epochs, batch_size, learning_rate)

    # test
    predicted_posteriors = network.forward(X_test)
    # determine class predictions from posteriors by winner-takes-all rule
    predicted_classes = np.argmax(predicted_posteriors, axis=1).flatten()  # !!!!!!! your code here
    YY.plot(X_test, predicted_classes)
    plt.savefig("full grad res %d.png" % (i))
    plt.show()
    # compute and output the error rate of predicted_classes
    error_rate = np.sum(predicted_classes == Y_test) / len(Y_test)  # your code here
    print("accuracy: %f, eta: %f" % (error_rate, learning_rate))

    plt.semilogy(ewma(loss_rec, len(loss_rec) // 20))
    plt.title("full gradient descent, eta: %f" % (eta))
    plt.savefig("full grad %d.png" % (i))
    plt.show()

    return network, X_train, Y_train, X_test, Y_test


def mimic(N=5000, n_epochs=100, N_in=2, N_hidden=3, N_out=2, eta=8.0):
    # test task: Learn to mimic simple forward network
    W_21 = np.random.sample((N_out, N_hidden)) * 2 - 1
    W_10 = np.random.sample((N_hidden, N_in)) * 2 - 1

    # training set
    X_train, X_test = np.random.sample((N, N_in)), np.random.sample((20, N_in))
    teacher = lambda r_in: np.matmul(W_21, np.matmul(W_10, r_in.T)).T
    Y_train, Y_test = teacher(X_train), teacher(X_test)

    # set hyperparameters (play with these!)
    layer_sizes = [N_hidden, N_out]
    learning_rate = eta

    # create network
    network = MLP(N_in, layer_sizes, act="SoftReLU", bias_on=False, classify=False, track_weigths=True)

    # train
    network.train(X_train, Y_train, n_epochs, 1, learning_rate)

    # test
    prediction = network.forward(X_test)

    # compute and output the error rate of prediction
    error_rate = np.sum((prediction - Y_test)**2) / len(Y_test)
    print("average squared error: %f, eta: %f" % (error_rate, learning_rate))

    plt.semilogy(ewma(loss_rec, len(loss_rec) // 20))
    plt.title("full gradient descent, eta: %f" % (eta))
    plt.show()

    plt.figure(figsize=(12, 10))
    plt.title("Forward Weights Evolution")
    n_trials = N*n_epochs
    for i in range(N_in):
        for j in range(N_hidden):
            plt.plot(weights_rec[0][:, j, i], label="$W^{(0)}_{up; %d->%d}$" % (i, j), lw=1)
    for i in range(N_hidden):
        for j in range(N_out):
            plt.plot(weights_rec[1][:, j, i], label="$W^{(1)}_{up; %d->%d}$" % (i, j), ls="--", lw=1)
    for i in range(N_in):
        for j in range(N_hidden):
            plt.plot([int(n_trials * 0.98), n_trials], [W_10[j, i], W_10[j, i]], c="r", lw=1.2)
    for i in range(N_hidden):
        for j in range(N_out):
            plt.plot([int(n_trials * 0.98), n_trials], [W_21[j, i], W_21[j, i]], c="r", lw=1.2, ls=":")
    plt.xlabel("trial")
    plt.ylabel("weight")
    plt.xlim([-1, n_trials])
    ax = plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()




if __name__=="__main__":

    for i, eta in enumerate([0.08]):
        #run(5000, 100, eta, 1)
        mimic(eta=eta)


