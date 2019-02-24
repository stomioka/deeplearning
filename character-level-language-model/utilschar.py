class LSTMbasedTextGenerator:
    
    def __init__(self, input_text_for_training, learning_rate ):
        self.bar = 1
        self.input_text_for_training = input_text_for_training
        self.learning_rate = learning_rate

            
    def read_text_file(self):
        input_text_for_training=self.input_text_for_training
        data = open(input_text_for_training, 'r').read()
        data= data.lower()
        chars = list(set(data))
        data_size, vocab_size = len(data), len(chars)
        #print('There are %d total characters and %d unique characters in your data.' % (data_size, vocab_size))  
        char_to_ix = { ch:i for i,ch in enumerate(sorted(chars)) }
        ix_to_char = { i:ch for i,ch in enumerate(sorted(chars)) }
        return chars, vocab_size, char_to_ix, ix_to_char, data
    
    def softmax(self,x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)
    
    def smooth(self,loss, cur_loss):
        return loss * 0.999 + cur_loss * 0.001
    
    def print_sample(self,sample_ix):
        ix_to_char=self.read_text_file()[3]
        txt = ''.join(ix_to_char[ix] for ix in sample_ix)
        txt = txt[0].upper() + txt[1:]  # capitalize first character 
        print ('%s' % (txt, ), end='')
    
    def get_initial_loss(self,seq_length):
        vocab_size=self.read_text_file()[1]
        return -np.log(1.0/vocab_size)*seq_length
    
    def softmax(self,x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)
    
    def initialize_parameters(self,n_a, n_x, n_y):
        """
        Initialize parameters with small random values
        
        Returns:
        parameters -- python dictionary containing:
                            Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                            Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                            Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                            b --  Bias, numpy array of shape (n_a, 1)
                            by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
        """
        np.random.seed(1)
        Wax = np.random.randn(n_a, n_x)*0.01 # input to hidden
        Waa = np.random.randn(n_a, n_a)*0.01 # hidden to hidden
        Wya = np.random.randn(n_y, n_a)*0.01 # hidden to output
        b = np.zeros((n_a, 1)) # hidden bias
        by = np.zeros((n_y, 1)) # output bias
        
        parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b,"by": by}
        
        return parameters
    
    def rnn_step_forward(self,parameters, a_prev, x):
        
        Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
        a_next = np.tanh(np.dot(Wax, x) + np.dot(Waa, a_prev) + b) # hidden state
        p_t = self.softmax(np.dot(Wya, a_next) + by) # unnormalized log probabilities for next chars # probabilities for next chars 
        
        return a_next, p_t
    
    def rnn_step_backward(self,dy, gradients, parameters, x, a, a_prev):
        
        gradients['dWya'] += np.dot(dy, a.T)
        gradients['dby'] += dy
        da = np.dot(parameters['Wya'].T, dy) + gradients['da_next'] # backprop into h
        daraw = (1 - a * a) * da # backprop through tanh nonlinearity
        gradients['db'] += daraw
        gradients['dWax'] += np.dot(daraw, x.T)
        gradients['dWaa'] += np.dot(daraw, a_prev.T)
        gradients['da_next'] = np.dot(parameters['Waa'].T, daraw)
        return gradients
    
    def update_parameters(self,parameters, gradients, lr):
    
        parameters['Wax'] += -lr * gradients['dWax']
        parameters['Waa'] += -lr * gradients['dWaa']
        parameters['Wya'] += -lr * gradients['dWya']
        parameters['b']  += -lr * gradients['db']
        parameters['by']  += -lr * gradients['dby']
        return parameters
    
    def rnn_forward(self,X, Y, a0, parameters):
        vocab_size=self.read_text_file()[1]
        # Initialize x, a and y_hat as empty dictionaries
        x, a, y_hat = {}, {}, {}
        
        a[-1] = np.copy(a0)
        
        # initialize your loss to 0
        loss = 0
        
        for t in range(len(X)):
            
            # Set x[t] to be the one-hot vector representation of the t'th character in X.
            # if X[t] == None, we just have x[t]=0. This is used to set the input for the first timestep to the zero vector. 
            x[t] = np.zeros((vocab_size,1)) 
            if (X[t] != None):
                x[t][X[t]] = 1
            
            # Run one step forward of the RNN
            a[t], y_hat[t] = self.rnn_step_forward(parameters, a[t-1], x[t])
            
            # Update the loss by substracting the cross-entropy term of this time-step from it.
            loss -= np.log(y_hat[t][Y[t],0])
            
        cache = (y_hat, a, x)
            
        return loss, cache
    
    def rnn_backward(self,X, Y, parameters, cache):
        # Initialize gradients as an empty dictionary
        gradients = {}
        
        # Retrieve from cache and parameters
        (y_hat, a, x) = cache
        Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
        
        # each one should be initialized to zeros of the same dimension as its corresponding parameter
        gradients['dWax'], gradients['dWaa'], gradients['dWya'] = np.zeros_like(Wax), np.zeros_like(Waa), np.zeros_like(Wya)
        gradients['db'], gradients['dby'] = np.zeros_like(b), np.zeros_like(by)
        gradients['da_next'] = np.zeros_like(a[0])
        
        ### START CODE HERE ###
        # Backpropagate through time
        for t in reversed(range(len(X))):
            dy = np.copy(y_hat[t])
            dy[Y[t]] -= 1
            gradients = self.rnn_step_backward(dy, gradients, parameters, x[t], a[t], a[t-1])
        ### END CODE HERE ###
        
        return gradients, a
    
    def clip(self,gradients, maxValue):
        '''
        Clips the gradients' values between minimum and maximum.
        
        Arguments:
        gradients -- a dictionary containing the gradients "dWaa", "dWax", "dWya", "db", "dby"
        maxValue -- everything above this number is set to this number, and everything less than -maxValue is set to -maxValue
        
        Returns: 
        gradients -- a dictionary with the clipped gradients.
        '''
        
        dWaa, dWax, dWya, db, dby = gradients['dWaa'], gradients['dWax'], gradients['dWya'], gradients['db'], gradients['dby']
       
        # clip to mitigate exploding gradients, loop over [dWax, dWaa, dWya, db, dby].
        for gradient in [dWax, dWaa, dWya, db, dby]:
             np.clip(gradient, -maxValue, maxValue, out=gradient)
        
        gradients = {"dWaa": dWaa, "dWax": dWax, "dWya": dWya, "db": db, "dby": dby}
        
        return gradients
    
    def sample(self,parameters,  seed):
        """
        Sample a sequence of characters according to a sequence of probability distributions output of the RNN
    
        Arguments:
        parameters -- python dictionary containing the parameters Waa, Wax, Wya, by, and b. 
        char_to_ix -- python dictionary mapping each character to an index.
        seed -- used for grading purposes. Do not worry about it.
    
        Returns:
        indices -- a list of length n containing the indices of the sampled characters.
        """
        char_to_ix=self.read_text_file()[2]
        # Retrieve parameters and relevant shapes from "parameters" dictionary
        Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
        vocab_size = by.shape[0]
        n_a = Waa.shape[1]
    
        # Step 1: Create the one-hot vector x for the first character (initializing the sequence generation). 
        x = np.zeros([vocab_size,1])
        # Step 1': Initialize a_prev as zeros (≈1 line)
        a_prev = np.zeros([n_a,1])
        
        # Create an empty list of indices, this is the list which will contain the list of indices of the characters to generate 
        indices = []
        
        # Idx is a flag to detect a newline character, we initialize it to -1
        idx = -1 
        
        # Loop over time-steps t. At each time-step, sample a character from a probability distribution and append 
        # its index to "indices". We'll stop if we reach 50 characters (which should be very unlikely with a well 
        # trained model), which helps debugging and prevents entering an infinite loop. 
        counter = 0
        newline_character = char_to_ix['\n']
        
        while (idx != newline_character and counter != 50):
            
            # Step 2: Forward propagate x using the equations (1), (2) and (3)
            a = np.tanh(np.dot(Wax,x)+np.dot(Waa,a_prev)+b)
            z = np.dot(Wya,a)+by
            y = self.softmax(z)
            
            # for grading purposes
            np.random.seed(counter+seed) 
            
            # Step 3: Sample the index of a character within the vocabulary from the probability distribution y
            idx = np.random.choice(range(len(y)),p=y.ravel())
    
            # Append the index to "indices"
            indices.append(idx)
            
            # Step 4: Overwrite the input character as the one corresponding to the sampled index.
            x = np.zeros((vocab_size,1))
            x[idx] = 1
            
            # Update "a_prev" to be "a"
            a_prev = a
            
            # for grading purposes
            seed += 1
            counter +=1
            
        if (counter == 50):
            indices.append(char_to_ix['\n'])
        
        return indices
    
    def optimize(self,X, Y, a_prev, parameters):
        """
        Execute one step of the optimization to train the model.
        
        Arguments:
        X -- list of integers, where each integer is a number that maps to a character in the vocabulary.
        Y -- list of integers, exactly the same as X but shifted one index to the left.
        a_prev -- previous hidden state.
        parameters -- python dictionary containing:
                            Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                            Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                            Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                            b --  Bias, numpy array of shape (n_a, 1)
                            by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
        learning_rate -- learning rate for the model.
        
        Returns:
        loss -- value of the loss function (cross-entropy)
        gradients -- python dictionary containing:
                            dWax -- Gradients of input-to-hidden weights, of shape (n_a, n_x)
                            dWaa -- Gradients of hidden-to-hidden weights, of shape (n_a, n_a)
                            dWya -- Gradients of hidden-to-output weights, of shape (n_y, n_a)
                            db -- Gradients of bias vector, of shape (n_a, 1)
                            dby -- Gradients of output bias vector, of shape (n_y, 1)
        a[len(X)-1] -- the last hidden state, of shape (n_a, 1)
        """
        learning_rate =self.learning_rate
        # Forward propagate through time 
        loss, cache = self.rnn_forward(X, Y, a_prev, parameters)
        
        # Backpropagate through time 
        gradients, a = self.rnn_backward(X, Y, parameters, cache)
        
        # Clip gradients between -5 (min) and 5 (max) 
        gradients = self.clip(gradients, 5)
        
        # Update parameters 
        parameters = self.update_parameters(parameters, gradients, learning_rate)
        
        return loss, gradients, a[len(X)-1]
    
    def model(self, num_iterations, n_a, seq_length):
        """
        Trains the model and generates text 
        
        Arguments:
        data -- text corpus
        ix_to_char -- dictionary that maps the index to a character
        char_to_ix -- dictionary that maps a character to an index
        num_iterations -- number of iterations to train the model for
        n_a -- number of units of the RNN cell
        seq_length -- number of lines to sample at each iteration. 
        vocab_size -- number of unique characters found in the text, size of the vocabulary
        
        Returns:
        parameters -- learned parameters
        """
        data=self.read_text_file()[4]
        ix_to_char=self.read_text_file()[3]
        char_to_ix=self.read_text_file()[2]
        vocab_size=self.read_text_file()[1]
        input_text_for_training=self.input_text_for_training
        # Retrieve n_x and n_y from vocab_size
        n_x, n_y = vocab_size, vocab_size
        
        # Initialize parameters
        parameters = self.initialize_parameters(n_a, n_x, n_y)
        
        # Initialize loss (this is required because we want to smooth our loss)
        loss = self.get_initial_loss(seq_length)
        
        # Build list of all lines (training examples).
        with open(input_text_for_training) as f:
            examples = f.readlines()
        examples = [x.lower().strip() for x in examples]
        
        # Shuffle list of all lines
        np.random.seed(0)
        np.random.shuffle(examples)
        
        # Initialize the hidden state of your LSTM
        a_prev = np.zeros((n_a, 1))
        
        # Optimization loop
        for j in range(num_iterations):
    
            index = j % len(examples)
            X = [None] + [char_to_ix[ch] for ch in examples[index]] 
            Y = X[1:] + [char_to_ix["\n"]]
            
            # Perform one optimization step: Forward-prop -> Backward-prop -> Clip -> Update parameters
            curr_loss, gradients, a_prev = self.optimize(X, Y, a_prev, parameters)
            
            
            # Use a latency trick to keep the loss smooth. It happens here to accelerate the training.
            loss = self.smooth(loss, curr_loss)
    
            # Every 2000 Iteration, generate "n" characters thanks to sample() to check if the model is learning properly
            if j % 2000 == 0:
                
                print('Iteration: %d, Loss: %f' % (j, loss) + '\n')
                
                # The number of dinosaur names to print
                seed = 0
                for name in range(seq_length):
                    
                    # Sample indices and print them
                    sampled_indices = self.sample(parameters, seed)
                    self.print_sample(sampled_indices)
                    
                    seed += 1  # To get the same result for grading purposed, increment the seed by one. 
          
                print('\n')
            
        return parameters