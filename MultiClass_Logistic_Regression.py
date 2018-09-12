import pandas as pd  # importing the Pandas library required for file reading and file importing into the code
import numpy as np   # mathematical library used for calling basic maths functions

#-----------------------------------------------------------------------------------------------------------------------------------------------

#BASIC FUNCTIONS REQUIRED

def sigmoid(x) :                                        #calculates the conditional probability of the prediction. 
    denom = (1.0 + np.exp(-x))                          #The greater the odds of a positive event occuring the greater the odds
    return 1.0/denom

def V(X,w) :                                            # Given an input feature vector.
    net = np.dot(X,w)
    return sigmoid(net)                                 #this function returns the sigmoid of weighted sum(the input vector and the weight vectors are inclusive of the bias term for this code)

def Error(X,w,y) :                                      # Cost Function
    f1 = np.sum(np.dot(y.T,np.log(V(X,w))))             # This is the main function that gives the information about how far away the parameters are from their locallly optimized values.
    f2 = np.sum(np.dot((1-y).T,np.log(1-V(X,w))))       # Also known as negative log likelihood function. This is obtained since the outcomes are conditional probabilities for each class and each feature vector is independent of the others.
    return -(f1 + f2)/y.size                            # The main idea of this implementation is the minimization of this cost function to obtain optimized parameters.

def gradError(X,w,y) :                                  # The partial derivative of cost function w.r.t the Weights. 
    prediction = V(X,w)                                 
    X_trans = X.T                                       # Transpose of feature vector
    return (np.dot(X_trans,(V(X,w) - y)))/(y.size)


# Gradient of Cost Function, X: feature vector, w: weight matrix, y: function class to be learned, V(X,w): predicted class

#-----------------------------------------------------------------------------------------------------------------------------------------------

# FUNCTION REQUIRED FOR NORMALIZATION OF INPUT DATA

def normalized(X):      
    X_mean=X.mean(axis=0)                                # Calculates the mean value for the input data set 
    X_std=X.std(axis=0)                                  # Calculates the standard deviation for the input data set 
    return (X-X_mean)/X_std                              # Returns the normalized data set 


# -----------------------------------------------------------------------------------------------------------------------------------------------

# DATA HANDLING PART OF THE CODE USING PANDAS LIBRARY

# The pandas library function "read" takes as argument the local file path to where the data was stored on my computer during training
# This needs to be essentially updated if the location of the data files is updated. 

data_train_features = pd.read_csv("/Users/vishalsharma/Documents/ELL409/Assignment1/dataset/train_data.csv", names=['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16'])   # Importing the training feature vectors and storing them in the corresponding variable
data_test_features = pd.read_csv("/Users/vishalsharma/Documents/ELL409/Assignment1/dataset/test_data.csv", names=['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16'])     # Importing the test feature vectors and storing them in the corresponding variable


data_train_features_matrix = data_train_features.as_matrix()            # Creating a matrix of the obtained features for both the training and the test data sets.
data_test_features_matrix = data_test_features.as_matrix()              # Trainging/Test feature matrix shape = (number of training/test inputs, 16) # Exclusive of the bias term '1'


data_train_labels = pd.read_csv("/Users/vishalsharma/Documents/ELL409/Assignment1/dataset/train_labels.csv", names=['y'])   # Importing the training labels and storing them in the corresponding variable
data_test_labels = pd.read_csv("/Users/vishalsharma/Documents/ELL409/Assignment1/dataset/test_labels.csv", names=['y'])     # Importing the test labels and storing them in the corresponding variable
#Y_df = pd.DataFrame(data_train_labels.y)
#print(Y_df.head())

data_train_labels_matrix = data_train_labels.as_matrix()                # Creating a matrix of the obtained labels for both the training and the test data sets.
data_test_labels_matrix = data_test_labels.as_matrix()                  # Trainging/Test label matrix shape = (number of training/test inputs, 1)



data_train_features_matrix[:,1:] = normalized(data_train_features_matrix[:,1:])     # Normalizing the training feature data set


X_train = np.zeros((data_train_features_matrix.shape[0],17))
X_train[:,16] = 1.0 
for i in range(16):
    X_train[:,i] = data_train_features_matrix[:,i]                                  # Training feature matrix shape = (number of training inputs, 17) # Inclusive of the bias term '1'


data_test_features_matrix[:,1:] = normalized(data_test_features_matrix[:,1:])       # Normalizing the test feature data set

X_test = np.zeros((data_test_features_matrix.shape[0],17))
X_test[:,16] = 1.0 
for i in range(16):
    X_test[:,i] = data_test_features_matrix[:,i]                                    # Test feature matrix shape = (number of test inputs, 17) # Inclusive of the bias term '1'





Y_train = np.zeros((data_train_labels_matrix.shape[0],10))                          # In this step an output matrix for each of the training and test sets is created based on the value of label for that data point
for i in range(10):                     
    Y_train[:,i] = np.where(data_train_labels_matrix[:,0]==i, 1,0)                  # The new matrix has the shape = (number of training/test labels , 10)

Y_test = np.zeros((data_test_labels_matrix.shape[0],10))                            # So, a new matrix is constructed having 10 coloumns with the coloumn number corresponding to the label value to be 1 and the rest to be zero.
for j in range(10):
    Y_test[:,j] = np.where(data_test_labels_matrix[:,0]==j, 1,0)                    


#------------------------------------------------------------------------------------------------------------------------------------------------

# MAIN LEARNING PART OF THE CODE. HERE I IMPLEMENT THE USUAL GRADIENT DESCENT ALGORITHM TO MAKE THE COST FUNCTION CONVERGE TO A LOCAL MINIMA

W_opt= np.zeros((X_train.shape[1],10))                      # The optimized weight matrix is stored in this variable.
W_opt2= np.zeros((X_train.shape[1],10))                     # Again, each coloumn of this matrix is a decision boundary separating that particular class from the rest of the classes.
                                                            # The shape of the optimized W_opt matrix = (17,10) in this case with 16 dimensional feature vectors that are required to be classified into either of the 10 distinct classes 

def grad_desc(X, w, y, Tolerance, LearningRate) :
    error = Error(X, w, y)                                  # Computing the value of the cost function right at the start of the gradient descent algorithm for the first step.
    iterations = 1                                          # Starting the counter for iterations with 1
    error_diff = 2                                          # difference in error between two consecutive iterations(intially set to a random value greater than convergence) (important for loop termination), will be updated inside the loop
    while(error_diff > Tolerance):
        error_prev = error                                  # assigns the value of the existing error to the variable error_prev
        w = w - (LearningRate * gradError(X, w, y))         # update the weights according to the equation (w(j+1) = w(j) - LearningRate(gradError)) # step towards parameter optimization
        error = Error(X, w, y)                              # new value of error will be equal to the newly calculated one with updated weights
        error_diff = error_prev - error                     # defintion of error_diff 
        iterations+=1                                       # updating the iteration number
    print('Total Interations required for learning this decision boundary: ', iterations)
    return w

for i in range(10):
    print('\nLearning the parameters for Class-{} versus the rest\n'.format(i))
    W_opt2[:,i] = grad_desc(X_train, W_opt[:,i], Y_train[:,i], Tolerance=1e-6, LearningRate=.001) # I have selected the convergence/tolerance and the learning rate to values that give best efficiency, but the learning is slow with these hyperparameters.
                                                                                                  # Taking between 35,000 - 55,000 iterations for learning each class. We can change these values for a trade-off between training time and efficiency
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# VARIOUS SCORING METHODS TO TEST FOR THE EFFICIENCY OF THE LEARNED ALGORITHM


def Prob_list(X,w,y):                                       # A function that calculates the probability of a feature vector belonging to a given class 
    h_prob_list = np.zeros(y.shape)                         # Simply by computing the sigmoid of the weighted sum over the input vector 
    for CLASS in range(10):
        h_prob_list[:,CLASS]= V(X,w[:,CLASS])
    return h_prob_list

def Pred_list(X,w,y):                                       # Converts the probability of the highest coloumn to 1 and the rest to zero. 
    h_prob_list2 = Prob_list(X,w,y)                         # This is classification based on the maximum probability corresponding to a class.
    pred_list = np.zeros(y.shape)
    for Class in range(10): 
        for i in range(y[:,[1]].shape[0]):
            if h_prob_list2[i,Class] == np.amax(h_prob_list2[i,:]):
                pred_list[i,Class] = 1
            else:
                pred_list[i,Class] = 0
    return pred_list                                        # This function does the classification based on the probability distributions from the previous function





def true_Pos(pred_list, y, Class):                          # As the name suggests, gives the total number of true Positives for a class in train/test data
    totalTruePos = 0
    for i in range(y.shape[0]):
        if (pred_list[i,Class] == 1 and y[i] == 1):
            totalTruePos += 1
    return totalTruePos

def false_Pos(pred_list, y, Class):                         # As the name suggests, gives the total number of false Positives for a class in train/test data
    totalFalsePos = 0
    for i in range(y.shape[0]):
        if (pred_list[i,Class] == 1 and y[i] == 0):
            totalFalsePos += 1
    return totalFalsePos

def false_Neg(pred_list, y, Class):                         # As the name suggests, gives the total number of false Negatives for a class in train/test data
    totalFalseNeg = 0
    for i in range(y.shape[0]):
        if (pred_list[i,Class] == 0 and y[i] == 1):
            totalFalseNeg += 1
    return totalFalseNeg

def true_Neg(pred_list, y, Class):                          # As the name suggests, gives the total number of true Negatives for a class in train/test data
    totalTrueNeg = 0
    for i in range(y.shape[0]):
        if (pred_list[i,Class] == 0 and y[i] == 0):
            totalTrueNeg += 1
    return totalTrueNeg


#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# A FEW SCORING METHODS WITH THEIR MATHEMATICAL DEFINITIONS


def accuracy(pred_list, y, Class):
    acc = (true_Pos(pred_list, y, Class) + true_Neg(pred_list, y, Class))/y.size
    return acc

def precision(pred_list, y, Class):
    prec = true_Pos(pred_list, y,Class)/(false_Pos(pred_list, y, Class) + true_Pos(pred_list, y, Class))
    return prec

def recall(pred_list, y, Class):
    recall = true_Pos(pred_list, y, Class)/(true_Pos(pred_list, y,Class)+false_Neg(pred_list, y, Class))
    return recall

def f1_score(pred_list, y, Class):
    score = 2*recall(pred_list, y, Class)*precision(pred_list, y,Class)/(recall(pred_list, y,Class)+precision(pred_list, y,Class))
    return score


#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# PART OF THE CODE THAT COMPUTES THE SCORES VIA AFOREMENTIONED METHODS FOR BOTH TRAINING AND TEST DATA. 


def scoringMethods(X,w,y):
    pred_list = Pred_list(X,w,y)
    
    ACCURACY = np.zeros(10)
    PRECISION = np.zeros(10)
    RECALL = np.zeros(10)
    F_SCORE = np.zeros(10)


    for Class in range(10):
        
        pos_TRUE = true_Pos(pred_list, y[:,Class],Class)
        
        pos_FALSE = false_Pos(pred_list, y[:,Class], Class)
        
        neg_FALSE = false_Neg(pred_list, y[:,Class], Class)
        
        neg_TRUE = true_Neg(pred_list, y[:,Class], Class)
        
        ACCURACY[Class] = accuracy(pred_list, y[:,Class],Class)*100
        
        PRECISION[Class] = precision(pred_list, y[:,Class], Class)
        
        RECALL[Class] = recall(pred_list, y[:,Class], Class)
        
        F_SCORE[Class] = f1_score(pred_list, y[:,Class], Class)

    return ACCURACY, PRECISION, RECALL, F_SCORE



ACCURACY_train = np.zeros(10)
PRECISION_train = np.zeros(10)
RECALL_train = np.zeros(10)
F_SCORE_train = np.zeros(10)

ACCURACY_test = np.zeros(10)
PRECISION_test = np.zeros(10)
RECALL_test = np.zeros(10)
F_SCORE_test = np.zeros(10)


# SCORING ANALYSIS OF THE TRAINING DATA
ACCURACY_train, PRECISION_train, RECALL_train, F_SCORE_train = scoringMethods(X_train, W_opt2, Y_train)
print('Accuracy on train data: ', ACCURACY_train)
print('Precision on train data: ', PRECISION_train)
print('Recall on train data: ', RECALL_train)
print('F Score on train data: ', F_SCORE_train)

# SCORING ANALYSIS OF THE TEST DATA
ACCURACY_test, PRECISION_test, RECALL_test, F_SCORE_test = scoringMethods(X_test, W_opt2, Y_test)
print('Accuracy on test data: ', ACCURACY_test)
print('Precision on test data: ', PRECISION_test)
print('Recall on test data: ', RECALL_test)
print('F Score on test data: ', F_SCORE_test)