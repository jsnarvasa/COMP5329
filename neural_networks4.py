# Library imports
import numpy as np
import matplotlib.pyplot as pl


# Read data
train_data = np.load("./Assignment1-Dataset/train_data.npy")
train_label = np.load("./Assignment1-Dataset/train_label.npy")
test_data = np.load("./Assignment1-Dataset/test_data.npy")
test_label = np.load("./Assignment1-Dataset/test_label.npy")


'''
# Generate dataset
class_1 = np.hstack([np.random.normal( 1, 1, size=(25, 2)),  np.ones(shape=(25, 1))])
class_2 = np.hstack([np.random.normal(-1, 1, size=(25, 2)), -np.ones(shape=(25, 1))])
dataset = np.vstack([class_1, class_2])

train_data = dataset[:,0:2]
train_label = dataset[:,2]
'''


##DROPOUT FUNCTIONS

drop_prob=0.5
#the drop_prob - value to control the factor of dropped neurons in the network. 
#drop_prob=0.5 means that 50% of neurons will be dropped.
def dropout_frwd(input, drop_prob): #function to perform dropout during the training of forward passs
    binomial_array = np.random.binomial(1, 1 - drop_prob, size=input.shape) / drop_prob  #A random array of size of input is made. The values in array are either 0 or 1.
    out = input * binomial_array #the binomial array is mulpits with the input(forward pass) in which some of th elayers are dropped as they are multiplied by 0
    return out, binomial_array


def dropout_backpass(delta, binomial_array): #function to perform dropout during the training of forward passs
    delta=delta*binomial_array #the nuerons which are droped out are removed or their value is made to 0 in the backward pass
    return delta


class Activation(object):
    '''
    Setting up the Activation class, which allows us to easily refer to the activation function and its derivative
    based on the given activation function of the layer - by setting the properties f and f_deriv

    Instantiation parameter: activation function for layer
    '''
    
    def __tanh(self, z):
        return np.tanh(z)
    
    def __tanh_deriv(self, a):
        #where a = np.tanh(z)
        return 1.0 - a**2
        
    def __logistic(self, z):
        return 1.0 /(1.0 + np.exp(-z))
    
    def __logistic_deriv(self, a):
        #where a = logistic(z)
        return a * (1-a)
        
    def __relu(self,z):
        return np.maximum(0,z)
  
    def __relu_deriv(self,a):
        return np.heaviside(a, 0)

    def __softmax(self, z):
        # return np.exp(z) / np.sum(np.exp(z))
        z = z - np.max(z) # we're adding this such that it doesn't overflow with very large nums - numerical stability
        return np.divide(np.exp(z), np.sum(np.exp(z)))

    def __softmax_deriv(self, a):
      for i in range(0, len(a)):
        for j in range(0, len(a)):
          if i == j:
            return a[i] * (1 - a[i])
          elif i != j:
            return -a([i] * a[j])
        
    #Initialise & set the default activation functions
    def __init__(self, activation = 'relu'):
        if activation == "logistic":
            self.f = self.__logistic
            self.f_deriv = self.__logistic_deriv
        elif activation == "tanh":
            self.f = self.__tanh
            self.f_deriv = self.__tanh_deriv    
        elif activation == 'relu': 
            self.f = self.__relu
            self.f_deriv= self.__relu_deriv
        elif activation == "softmax":
            self.f = self.__softmax
            self.f_deriv = self.__softmax_deriv


class HiddenLayer(object):
    def __init__(self, 
                 n_in: int, 
                 n_out: int, 
                 activation_last_layer: str = 'relu',
                 activation: str = 'relu',
                 W = None,
                 b = None,
                 v_W = None,
                 v_b = None,
                 weight_init_method = 'Xavier',
                 batch_size = 1,
                 last_hidden_layer = False):

        self.last_hidden_layer = last_hidden_layer
        self.output = []
        self.input = []
        self.activation = Activation(activation).f

        #Derivative of last layer activation
        #Note methods/links to Activation class
        self.activation_deriv = None
        if activation_last_layer:
            self.activation_deriv = Activation(activation_last_layer).f_deriv
    
        # Chuck all the different weight init methods here
        if weight_init_method == 'Xavier':
            self.W = self.__xavier_weight_init(n_in, n_out)
    

        if activation == 'logistic':
            self.W *= 4 #*= being similar to +=
    
        #size of bias = size of output
        self.b = np.zeros(n_out,)
    
        #size of weight gradient = size of weight
        self.grad_W = [np.zeros(self.W.shape) for i in range(batch_size)]
        self.grad_b = [np.zeros(self.b.shape) for i in range(batch_size)]
        # Transforming it into NP Array instead of Python list
        self.grad_W = np.array(self.grad_W)
        self.grad_b = np.array(self.grad_b)
        
        #Initialise velocities for Momentum SGD
        self.v_W = np.zeros_like(self.grad_W)
        self.v_b = np.zeros_like(self.grad_b)
        self.binomial_array=np.zeros(n_out) # array to store the binomial array used for dropout 
    

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
        

    # forward progress for training epoch:
    def forward(self, input):
        '''
        Feedforward for the neural network
        '''

        lin_output = np.dot(input, self.W) + self.b #simple perceptron output
        self.output.append(
            lin_output if self.activation is None #linear if no activation specified
            else self.activation(lin_output) #activation fn on w*I + b  (i.e. activation function on linear output)
        ) 
        
        #if not self.last_hidden_layer:
        #    output, self.binomial_array = dropout_frwd(self.output[-1], drop_prob)
        #    self.output[-1] = output

        self.input.append(input)
        return self.output[-1]

    #backpropagation
    def backward(self, delta, observation_idx, output_layer = False):
        '''
        Backpropagation for the neural network
        '''

        self.grad_W[observation_idx] = np.atleast_2d(self.input[observation_idx]).T.dot(np.atleast_2d(delta))
        self.grad_b[observation_idx] = delta

        if self.activation_deriv:
            delta = delta.dot(self.W.T) * self.activation_deriv(self.input[observation_idx])

        #delta=dropout_backpass(delta, self.binomial_array)

        return delta


class MLP:
    '''
    Main class holding the structure of the multi-layer Neural Network
    '''

    def __init__(self,
                 layers: list,
                 activation: list = [None, 'relu', 'relu'],
                 weight_init_method: str='Xavier', 
                 batch_size: int = 1,
                 gd_mode: str = 'batch'):
        
        # Initialise the layers
        self.layers = []
        self.params = []
        
        self.activation = activation
        self.gd_mode = gd_mode

        self.batch_size = batch_size
        
        for i in range(len(layers)-1):

            last_hidden_layer = False

            if i == len(layers) - 2: # -2 because -1 for output layer, and another -1 since it's index 0
                last_hidden_layer = True

            self.layers.append(HiddenLayer(layers[i],
                                           layers[i+1],
                                           activation[i],
                                           activation[i+1],
                                           weight_init_method = weight_init_method,
                                           batch_size = self.batch_size,
                                           last_hidden_layer=last_hidden_layer))
            
            
            #note HiddenLayer(n_in, n_out, activation_last_layer, activation) - type(layers) = int
        #Forward - pass info through layer, return result of final output
    def forward(self, input):
        for layer in self.layers: #iterate forward through layers
            output = layer.forward(input) #call forward method from HiddenLayer
            input = output #feedforward to next layer
        return output
         
    def criterion_MSE(self, y, y_hat):
        activation_deriv = Activation(self.activation[-1]).f_deriv #Derivate of last layer activation
        #MSE
        error = y - y_hat
        #Squared error
        loss = error**2
       # loss= *0.98
        #Delta of output
        delta = -error * activation_deriv(y_hat)
        return loss, delta

    def CE_loss(self, y, y_hat):
      softmax_y_hat = np.exp(y_hat) / np.sum(np.exp(y_hat))
      loss = -np.sum(y * np.log(softmax_y_hat)) 
     # loss=loss*0.98
      delta = y - softmax_y_hat
      return loss, delta

    
    def backward(self, delta, observation_idx):
        delta = self.layers[-1].backward(delta, observation_idx, output_layer=True)
        for i in reversed(range(len(self.layers[:-1]))):
            layer = self.layers[i]
            delta = layer.backward(delta, observation_idx)

                
    #Update weights after backward function - lr = learning rate. Added SGD_optim to allow flexibility choosing SGD optimiser (e.g. Momentum)    
    def update(self, lr, SGD_optim = None):
        if SGD_optim is None:
            for layer in self.layers:
                # Obtaining the averages
                grad_W_avg = np.average(layer.grad_W, axis=0)
                grad_b_avg = np.average(layer.grad_b, axis=0)
                layer.W -= lr * grad_W_avg
                layer.b -= lr * grad_b_avg

        # Need to make this compatible with batches
        elif SGD_optim['Type'] == 'Momentum':
            for layer in self.layers:
                layer.v_W = (SGD_optim['Parameter'] * layer.v_W) + (lr * layer.grad_W)
                layer.v_b = (SGD_optim['Parameter'] * layer.v_b) + (lr * layer.grad_b)
                layer.W = layer.W - layer.v_W
                layer.b = layer.b - layer.v_b

                
    #fit/training function - returns all losses within whole training process
    def fit(self, X, y, learning_rate = 0.1, epochs = 100, SGD_optim = None):
        #X = input data/features
        #y = input target
        #learning rate = param for speed of learning
        #epochs - # times dataset is presented to network for learning
            
        X = np.array(X)
        y = np.array(y)
        to_return = []

            
        for k in range(epochs):
            if k % 50 == 0:
                print('Training Epoch', k)

            # Shuffle the data, to ensure that each epoch will have different sequence of observations
            X, y = Utils.shuffle(X, y)

            if self.gd_mode == 'SGD':
                # since there will be 1 batch, of X.shape[0] observation per epoch in SGD
                num_batches = 1
            else:
                # Get the number of batches that we'll have given our dataset size and size of each batch
                num_batches = int(np.ceil(X.shape[0] / self.batch_size))

            # replacing the loss variable below, since the size should be based on the number of epochs and not the size of our data
            # loss = np.zeros(X.shape[0]) #initialise as zeros with same length as input
            batch_loss = np.zeros(num_batches)

            # May seem redundant, but is required when the remaining data observations is less than the batch size
            batch_size = self.batch_size

            observation_idx_current: int = 0


            # Iterate over each batch
            for batch in range(num_batches):

                obs_loss = np.zeros(batch_size)

                for observation_idx in range(batch_size):

                    if self.gd_mode == 'SGD':
                        observation_idx_current = np.random.randint(X.shape[0])
                        y_hat = self.forward(X[observation_idx_current])
                        obs_loss[observation_idx],delta=self.CE_loss(y[observation_idx_current],y_hat)

                    else:
                        # forward pass
                        y_hat = self.forward(X[observation_idx_current + observation_idx])                    
                        # backward pass
                        obs_loss[observation_idx],delta=self.CE_loss(y[observation_idx_current + observation_idx],y_hat)

                    self.backward(delta, observation_idx)

                observation_idx_current += batch_size

                # perform the update after average of Delta has been performed
                self.update(learning_rate)

                # Getting the average of the observation loss within the batch
                batch_loss[batch] = np.mean(obs_loss)

                if ((observation_idx_current + batch_size) > X.shape[0]) and not (self.gd_mode == 'SGD'):
                    # then we reduce the size of the variable batch_size, so we don't reach end of index in the last batch iteration
                    batch_size = X.shape[0] - observation_idx_current


            to_return.append(np.mean(batch_loss))

         
        return to_return
            
    def predict(self, x):
        '''
        Predict method
        '''

        x = np.array(x)
        output = np.zeros(x.shape[0])
        for i in np.arange(x.shape[0]):
            output[i] = self.forward(x[i, :])
        return output



class Preprocessing:
    '''
    Class used to contain any pre-processing that may occur
    '''

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def normalize(self, new_min=0, new_max=1):     
        #Min-Max Normalization- (x-xmin)/(xmax-xmin)
        norm_data = (self.X - np.min(self.X))/(np.max(self.X) - np.min(self.X))
        self.X = norm_data

    def label_encode(self):
        '''
        Encode the label vector of our dataset, such that we can use it for the computation of the MSE.
        This is because the labels are labelled as integers 0 to 9, whereas for MSE to work, we need to
        create a array vector of size 10 for every single observation, where the value 1 is set on the index of the correct observation
        '''
        
        num_classes = np.unique(self.y).size
        
        encoded_label_vector = []
        
        for label in self.y:
            encoded_label = np.zeros(num_classes)
            encoded_label[int(label)-1] = 1
            encoded_label_vector.append(encoded_label)
        
        self.y = np.array(encoded_label_vector)


class Utils:
    '''
    Class used to contain miscellaneous methods used for Neural Network
    '''

    @staticmethod
    def shuffle(X, y):
        # create an index vector with the size of X
        shuffled_idx = np.arange(X.shape[0])
        np.random.shuffle(shuffled_idx)
        X = X[shuffled_idx]
        y = y[shuffled_idx]

        return X, y



# Instantiating dataset into class
train_df = Preprocessing(train_data, train_label)

# Perform normalisation
train_df.normalize()
print(train_df.X.shape)

# One-hot encode labels
train_df.label_encode()

# Batch mode
# nn = MLP([2, 10, 15, 12, 2], [None, 'relu', 'relu', 'relu', 'softmax'], weight_init_method='Xavier', batch_size=25)

# SGD mode
nn = MLP([128, 10, 15, 12, 10], [None, 'relu', 'relu', 'relu', 'softmax'], weight_init_method='Xavier', batch_size=50, gd_mode='SGD')
trial1 = nn.fit(train_df.X, train_df.y, learning_rate = 0.005, epochs = 700, SGD_optim = {'Type': 'Momentum', 'Parameter': 0.5})
print(trial1)

test_df = Preprocessing(test_data, test_label)
test_df.normalize()

predictions = nn.predict(test_df.X)

print(predictions)