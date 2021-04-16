# Library imports
import numpy as np
import matplotlib.pyplot as pl
from matplotlib import animation

# Generate dataset
class_1 = np.hstack([np.random.normal( 1, 1, size=(25, 2)),  np.ones(shape=(25, 1))])
class_2 = np.hstack([np.random.normal(-1, 1, size=(25, 2)), -np.ones(shape=(25, 1))])
dataset = np.vstack([class_1, class_2])



# Setting up the classes
class Activation(object):
    '''
    Setting up the Activation class, which allows us to easily refer to the activation function and its derivative
    based on the given activation function of the layer - by setting the properties f and f_deriv

    Instantiation parameter: activation function for layer
    '''

    def __tanh(self, x):
        return np.tanh(x)
    def __tanh_deriv(self, a):
        return 1.0 - a**2

    def __logistic(self, x):
        return 1.0 / (1.0 + np.exp(-x))
    def __logistic_deriv(self, a):
        return  a * (1 - a )

    def __relu(self,x):
        return np.maximum(0,x)
    def __relu_deriv(self,a):
        return np.heaviside(a, 0)

    def __softmax(self, x):
        z = z - np.max(z) # we're adding this such that it doesn't overflow with very large nums
        return np.divide(np.exp(z), np.sum(np.exp(z)))
    def __softmax_deriv(self, a):
        return
    
    def __init__(self,activation: str = 'tanh'):
        if activation == 'logistic':
            self.f = self.__logistic
            self.f_deriv = self.__logistic_deriv
        elif activation == 'tanh':
            self.f = self.__tanh
            self.f_deriv = self.__tanh_deriv
        elif activation == 'relu':
            self.f = self.__relu
            self.f_deriv = self.__relu_deriv
        elif activation == 'softmax':
            self.f = self.__softmax
            self.f_deriv = self.__softmax_deriv


class HiddenLayer(object):

    @staticmethod
    def __xavier_weight_init(n_in, n_out):
        ## This sets a layer's weights to values chosen from a random
        ## uniform distribution that's bounded between
        ## +- sqrt(6)/sqrt(number of input connections [fan-in] + number of output connections [fan-out]).

        weights = np.random.uniform( ## Draw random samples from a uniform distribution.
            low=-np.sqrt(6. / (n_in + n_out)), ## Lower boundary of the output interval
            high=np.sqrt(6. / (n_in + n_out)), ## Upper boundary of the output
            size=(n_in, n_out) ## Output shape
        )
        return weights

    def __init__(self,n_in: int, n_out: int,
                 activation_last_layer: str ='tanh',activation: str='tanh', W=None, b=None, weight_init_method='Xavier', batch_size=1):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: string
        :param activation: Non linearity to be applied in the hidden
                           layer
        """


        self.input=None
        self.activation=Activation(activation).f
        
        # activation deriv of last layer
        self.activation_deriv=None
        if activation_last_layer:
            self.activation_deriv=Activation(activation_last_layer).f_deriv
        
        # Chuck all the different weight init methods here
        if weight_init_method == 'Xavier':
            self.W = self.__xavier_weight_init(n_in, n_out)

        ## Note : optimal initialization of weights is dependent on the
        ##        activation function used (among other things).
        ##        For example, results presented in [Xavier10] suggest that you
        ##        should use 4 times larger initial weights for sigmoid compared
        ##        to tanh. We have no info for other function, so we use the
        ##        same as tanh.
        if activation == 'logistic':
            self.W *= 4

        self.b = np.zeros(n_out,)
        
        self.grad_W = [np.zeros(self.W.shape) for i in range(batch_size)]
        self.grad_b = [np.zeros(self.b.shape) for i in range(batch_size)]
        # Transforming it into NP Array instead of Python list
        self.grad_W = np.array(self.grad_W)
        self.grad_b = np.array(self.grad_b)
        
    def forward(self, input):
        '''
        :type input: numpy.array
        :param input: a symbolic tensor of shape (n_in,)
        '''
        lin_output = np.dot(input, self.W) + self.b
        self.output = (
            lin_output if self.activation is None
            else self.activation(lin_output)
        )
        self.input=input
        return self.output
    
    def backward(self, delta, observation_idx, output_layer=False):         
        self.grad_W[observation_idx] = np.atleast_2d(self.input).T.dot(np.atleast_2d(delta))
        ## Explanation on 'np.atleast_2d':
        ## View inputs as arrays with at least two dimensions.
        print("Grad_W:", self.grad_W)
        print("\n")

        self.grad_b[observation_idx] = delta
        print("Grad_b:", self.grad_b)
        print("\n")
        if self.activation_deriv:
            delta = delta.dot(self.W.T) * self.activation_deriv(self.input)
            print("Hidden Delta:", delta)
            print("\n")
        return delta
        ## Return delta for the next layer.


class MLP:
    """
    """      
    def __init__(self, layers, activation=[None,'tanh','tanh'], weight_init_method='Xavier', batch_size = 1):
        """
        :param layers: A list containing the number of units in each layer.
        Should be at least two values
        :param activation: The activation function to be used. Can be
        "logistic" or "tanh"
        """        
        self.batch_size = batch_size

        ### initialize layers
        self.layers=[]
        self.params=[]
        
        self.activation=activation
        for i in range(len(layers)-1):
            ## Added for you to see the output
            print("====", "Layer", str(i+1) + ":", "====")
            self.layers.append(HiddenLayer(layers[i],layers[i+1],activation[i],activation[i+1], weight_init_method = weight_init_method, batch_size = self.batch_size))
            ## Remember: HiddenLayer(n_input = dimensionality of input,
            ##                       n_output = number of hidden units,
            ##                       activation_last_layer,
            ##                       activation)

    def forward(self,input):
        for layer in self.layers:
            output=layer.forward(input)
            input=output
        return output
    
    def criterion_MSE(self,y,y_hat):
        activation_deriv=Activation(self.activation[-1]).f_deriv
        # MSE
        error = y-y_hat
        loss = error**2
        # calculate the delta of the output layer
        delta = -error*activation_deriv(y_hat)    
        # return loss and delta
        return loss,delta
        
    def backward(self,delta, observation_idx):
        delta = self.layers[-1].backward(delta, observation_idx, output_layer=True)
        ## Added for you to see the output
        print("====", "Layer", str(len(self.layers)) + ":", "====")
        print("Delta:", delta)
        print("\n")
        ##

        ## Since backpropagation starts from the last layer, we use 'reversed'.
        #for layer in reversed(self.layers[:-1]):
        for i in reversed(range(len(self.layers[:-1]))): ## changed
            layer = self.layers[i] ## added
            delta = layer.backward(delta, observation_idx)
            ## Added for you to see the output
            print("====", "Layer", str(i+1) + ":", "====")
            print("Delta:", delta)
            print("\n")
            ##
            
    def update(self,lr):
        for layer in self.layers:
            # Obtaining the averages
            grad_W_avg = np.average(layer.grad_W, axis=0)
            grad_b_avg = np.average(layer.grad_b, axis=0)
            layer.W -= lr * grad_W_avg
            layer.b -= lr * grad_b_avg

    def fit(self,X,y,learning_rate=0.1, epochs=100):
        """
        Online learning.
        :param X: Input data or features
        :param y: Input targets
        :param learning_rate: parameters defining the speed of learning
        :param epochs: number of times the dataset is presented to the network for learning
        """ 
        X=np.array(X)
        y=np.array(y)
        # to_return = np.zeros(epochs)
        to_return = []
        
        observation_idx_current: int = 0
        ready_for_exit: bool = False

        for k in range(epochs):
            ## Added for you to see the output
            print("******", "EPOCH", str(k+1) + ":", "******")
            ##

            loss=np.zeros(self.batch_size)
            
            ## Added for you to see the output
            print("******", "Epoch #" + str(k+1) + ":", "******")

            # Perform check, to ensure that there is enough data for next batch size
            if (observation_idx_current + self.batch_size) > X.shape[0]:
                self.batch_size = X.shape[0] - observation_idx_current
                ready_for_exit = True

            for observation_idx in range(self.batch_size):
                
                # forward pass
                y_hat = self.forward(X[observation_idx_current + observation_idx])
                
                # backward pass
                loss[observation_idx],delta=self.criterion_MSE(y[observation_idx_current + observation_idx],y_hat)
                self.backward(delta, observation_idx)

            observation_idx_current += self.batch_size

            # perform the update after average of Delta has been performed
            self.update(learning_rate)

            to_return.append(np.mean(loss))

            # To catch instances where X.shape modulo batch_size == 0, which results in extra append to the to_return list
            if observation_idx_current == X.shape[0]:
                ready_for_exit = True

            if ready_for_exit:
                break

        return to_return

    def predict(self, x):
        x = np.array(x)
        output = np.zeros(x.shape[0])
        for i in np.arange(x.shape[0]):
            output[i] = self.forward(x[i,:]) ## change from nn.forward(x[i,:])
        return output


class Preprocessing:
    '''
    Class used to hold the data and any pre-processing that may occur
    '''

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def shuffle(self):
        # create an index vector with the size of X
        shuffled_idx = np.arange(self.X.shape[0])
        np.random.shuffle(shuffled_idx)
        self.X = self.X[shuffled_idx]
        self.y = self.y[shuffled_idx]


# Testing out the NN
# Random seed important for evaluating model performance - ensure it's consistent
np.random.seed(101)

input_data = dataset[:,0:2]
output_data = dataset[:,2]

# Setting up preprocessing
data = Preprocessing(input_data, output_data)
data.shuffle()

nn = MLP([2,3,1], [None,'logistic','tanh'], 'Xavier', 25)
MSE = nn.fit(data.X, data.y, learning_rate=0.01, epochs=500)
print(f'Loss values are {MSE}')