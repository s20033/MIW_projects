# -- coding: utf-8 --
"""
Created on Sun Apr  4 14:52:33 2021

@author: Suman Bhurtel
"""
import os
import glob
import numpy as np

# Reading files from the Language folders English, german and polish.
READ_FILES_ENGLISH = glob.glob(os.path.join("D:/PYTHON/Robotics/MIW/ANN/lang/English", "*.txt"))
READ_FILES_GERMAN = glob.glob(os.path.join("D:/PYTHON/Robotics/MIW/ANN/lang/German", "*.txt"))
READ_FILES_POLISH = glob.glob(os.path.join("D:/PYTHON/Robotics/MIW/ANN/lang/Polish", "*.txt"))
TOTAL_LIST_READ_FILES = READ_FILES_ENGLISH + READ_FILES_GERMAN + READ_FILES_POLISH

#create empty list ith 26 elements
EMPTY_LIST = [0]*26

def count_the_variable(filename, asci_value):
    """ It counts the number of alphabets"""
    asci_number = 97 + asci_value
    for i in range(26):
        with open(filename, encoding='utf-8')as _f:
            count = 0
            for line in _f:
                for char in line:
                    if char.isalpha():
                        if char.isupper():
                            char = char.lower()
                        if char == chr(asci_number):
                            count += 1
                            i += 1
            return count

def convert_text_to_vec(vector):
    """ Converts the texts in to vector"""
    input_vector = np.array([[0]*26])
    for k in range(len(vector)):
        #test = count_the_variable(read_files_english[k],0)
        for j in range(26):
            EMPTY_LIST[j] = count_the_variable(vector[k], j)
        #normolize the vector
        for l_1 in range(26):
            EMPTY_LIST[l_1] = EMPTY_LIST[l_1]/sum(EMPTY_LIST)
        input_l = np.array([EMPTY_LIST])
        input_vector = np.concatenate((input_vector, input_l), axis=0)
    input_vector_final = np.array(input_vector)
    input_vector_final = np.delete(input_vector_final, (0), axis=0)
    return input_vector_final

# Create a neural net
_X = convert_text_to_vec(TOTAL_LIST_READ_FILES)
Y_ARR = np.array(([0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1],
                  [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0],
                  [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0],
                  [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
                  [1, 0, 0], [1, 0, 0]), dtype=float)

def sigmoid(value):
    """  Activation function (unipolar)"""
    return np.exp(np.fmin(value, 0)) / (1 + np.exp(-np.abs(value)))

def sigmoid_derivative(derivative_function):
    """  Derivative of sigmoid"""
    return derivative_function * (1 - derivative_function)

class NeuralNetwork:
    """ Neural Class definition """
    def __init__(self, x, y_value):
        """ Constructor """
        self.input = x
# initialize the weights of the node connecting input vec and hidden layer i.e 26*4 = 104
        self.weights1 = np.random.rand(self.input.shape[1], 3)
# initialize the weights of the edge connecting hidden layer and the output layer i.e 4*3 = 12
        self.weights2 = np.random.rand(3, 3)
        self.weights3 = np.random.rand(3, 3)
        self.weights4 = np.random.rand(3, 3)
        self.y_val = y_value
        self.output = np.zeros(y_value.shape)
        self.layer1 = None
        self.layer2 = None
        self.layer3 = None
        self.layer4 = None

    def feedforward(self):
        """ Feed forward Calculation"""
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.layer2 = sigmoid(np.dot(self.layer1, self.weights2))
        self.layer3 = sigmoid(np.dot(self.layer2, self.weights3))
        self.layer4 = sigmoid(np.dot(self.layer3, self.weights4))
        return self.layer4

    def backprop(self):
        """  Back Propagation"""
        d_wght4 = np.dot(self.layer3.T, 2*(self.y_val -self.output)*sigmoid_derivative(self.output))
        d_wght3 = np.dot(self.layer2.T, 2*(self.y_val -self.output)*sigmoid_derivative(self.output))
        d_wght2 = np.dot(self.layer1.T, 2*(self.y_val -self.output)*sigmoid_derivative(self.output))
        d_wght1 = np.dot(self.input.T, np.dot(2*(self.y_val -self.output)*sigmoid_derivative
                                              (self.output), self.weights2.T)*sigmoid_derivative
                         (self.layer1))
        self.weights1 += d_wght1
        self.weights2 += d_wght2
        self.weights3 += d_wght3
        self.weights4 += d_wght4

    def train(self, _X, Y_ARR):
        """ Trains the neural network """
        self.output = self.feedforward()
        self.backprop()

    def predict(self, data):
        """ predicts the output."""
        self.input = data
        out = self.feedforward()
        return out

# trains the NN 1,500 times
NN = NeuralNetwork(_X, Y_ARR)
for epoch in range(1500):
    NN.train(_X, Y_ARR)
    error_value = (NN.predict(_X)- Y_ARR)
    print("Error Value for Each training: ", error_value)

TEST_FILE_ENGLISH = glob.glob(os.path.join("D:/PYTHON/Robotics/MIW/ANN/lang.test/English", "*.txt"))
TEST_FILE_GERMAN = glob.glob(os.path.join("D:/PYTHON/Robotics/MIW/ANN/lang.test/German", "*.txt"))
TEST_FILE_POLISH = glob.glob(os.path.join("D:/PYTHON/Robotics/MIW/ANN/lang.test/Polish", "*.txt"))
TEST_INPUT_COMBINED = TEST_FILE_ENGLISH + TEST_FILE_GERMAN + TEST_FILE_POLISH
TEST_INPUT = convert_text_to_vec(TEST_INPUT_COMBINED)
Y_TRUE = np.array(([0, 0, 1], [0, 0, 1], [0, 0, 1],
                   [0, 1, 0], [0, 1, 0], [0, 1, 0],
                   [1, 0, 0], [1, 0, 0], [1, 0, 0]))
PREDICTION = NN.predict(TEST_INPUT)
PREDICTION = (PREDICTION > 0.5).astype(int)
MEAN_ABS_TEST_ERROR = np.abs(Y_TRUE-PREDICTION)/len(Y_TRUE)
print("MAE: ", MEAN_ABS_TEST_ERROR)

# User Input
# D:/PYTHON/Robotics/MIW/ANN/user_input_to_check/English
USER_INPUT = input(str('please provide the path of the text that you want program to predict '))
READ_USER_INPUT = glob.glob(os.path.join(USER_INPUT, "*.txt"))
USER_INPUT = convert_text_to_vec(READ_USER_INPUT)
PREDICT_USER_TEXT = NN.predict(USER_INPUT)
print("first one: ", PREDICT_USER_TEXT)

def conversion(naya_kura):
    """ this function converts user predict text to new array with
    max elements of the array to be 1 and others 0."""
    #max_value = np.argmax(naya_kura)
    max_index = np.argmax(naya_kura, axis=1)
    ind = max_index[0]
    new_list = np.array([0, 0, 0])
    new_list[ind] = 1
    return new_list

# provide the predicted out put to the user input
PREDICT_USER_TEXT = conversion(PREDICT_USER_TEXT)
#PREDICT_USER_TEXT = (PREDICT_USER_TEXT > 0.4).astype(int)
print("second one: ", PREDICT_USER_TEXT)
if (PREDICT_USER_TEXT == [[0, 0, 1]]).all():
    print('your given text is in English language')
elif (PREDICT_USER_TEXT == [[0, 1, 0]]).all():
    print('your given text is in German language')
elif (PREDICT_USER_TEXT == [[1, 0, 0]]).all():
    print('your given text is in Polish language')
else:
    print('sorry could not predict may be I am a stupid Machine!! :( ')
    print(PREDICT_USER_TEXT)
