
# coding: utf-8

# ## Load MNIST on Python 3.x

# In[4]:


import pickle
import gzip


# In[5]:


filename = 'mnist.pkl.gz'
f = gzip.open(filename, 'rb')
training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
f.close()


# ## Load USPS on Python 3.x

# In[6]:


from PIL import Image
import os
import numpy as np
from tqdm import tqdm_notebook
from matplotlib import pyplot as plt


# In[7]:


#Loading USPS Data
USPSMat  = []
USPSTar  = []
curPath  = 'USPSdata/Numerals'
savedImg = []

for j in range(0,10):
    curFolderPath = curPath + '/' + str(j)
    imgs =  os.listdir(curFolderPath)
    for img in imgs:
        curImg = curFolderPath + '/' + img
        if curImg[-3:] == 'png':
            img = Image.open(curImg,'r')
            img = img.resize((28, 28))
            savedImg = img
            imgdata = (255-np.array(img.getdata()))/255
            USPSMat.append(imgdata)
            USPSTar.append(j)


# In[77]:


#Defined Logistic Regression functions
def get_activation(X,W):
    return np.dot(X,W);

def get_model(A):
    exp_A = np.exp(A);
    for i in exp_A:
        sum_exp = np.sum(i); 
        for j in range(len(i)):
            i[j] = i[j]/sum_exp;
    return exp_A;
def get_hot_target(t,y):
    i=0;
    for j in t:
        a = y[i]
        j[a] = j[a] + 1;
        i = i+1
    return t
def get_cross_entropy(X,W,Y):
    target = []
    a = np.dot(X,W);
    exp_a = np.exp(a);
    sum_ex = 0;
    for i in exp_a:
        sum_ex = np.sum(i);
        for j in range(len(i)):
            i[j] = i[j]/sum_ex;
    k=0;
    count = 0;
    for i in exp_a:
        j = np.argmax(i);
        target.append(j)
        if(j == Y[k]):
            count = count+1;
        k = k+1;
        
    Loss = 0
    ln_y = -np.log(exp_a);
    i=0;
    for j in ln_y:
        x = Y[i];
        Loss = Loss + j[x];
        i=i+1;
    return (1/len(X))*Loss,float((count*100)/len(X)),np.asarray(target);


# In[9]:


#Loading MNIST Data
Mtraining_data = np.array(training_data[0]);
Mtraining_target = np.array(training_data[1]);
Mvalidation_data = np.array(validation_data[0]);
Mvalidation_target = np.array(validation_data[1]);
Mtest_data = np.array(test_data[0]);
Mtest_target = np.array(test_data[1]);


# In[134]:


Weights = np.random.rand(784,10);
t_values = np.zeros((50000,10));
La = 2;
learning_rate = 0.000012;
t_values = get_hot_target(t_values,Mtraining_target);
L = [];
A = [];
ite = [];
finalTarget = []
for i in tqdm_notebook(range(0,1000)):
    Activation = get_activation(Mtraining_data,Weights);
    Model = get_model(Activation);
    Delta_EW = np.matmul((Model - t_values).T,Mtraining_data);
    La_Delta_EW  = np.dot(La,Weights);
    Delta_E = np.add(Delta_EW, np.transpose(La_Delta_EW));
    Delta_W = -np.dot(learning_rate,Delta_E);
    W_next = Weights + np.transpose(Delta_W);
    Weights = W_next;
    Loss,accuracy,finalTarget = get_cross_entropy(Mtraining_data,Weights,Mtraining_target);
    A.append(accuracy);
    L.append(Loss);
    ite.append(i);


# In[136]:


print("Accuracy for MNIST Logistics: ", A[999])
from sklearn.metrics import confusion_matrix
print("Confusion Matrix for MNIST Logistics")
print(confusion_matrix(Mtraining_target, finalTarget))


# In[138]:


import pandas as pd
pd.crosstab(Mtraining_target, finalTarget,rownames=['Actual'],colnames=['Predicted'],margins=True)


# In[71]:


#Testing Logistic on USPS Dataset
Weights = np.random.rand(784,10);
t_values = np.zeros((19999,10));
La = 2;
learning_rate = 0.000012;
t_values = get_hot_target(t_values,np.asarray(USPSTar));
print("Shape of tvalues :");
print(np.shape(t_values));
L = [];
A = [];
ite = [];
for i in tqdm_notebook(range(0,1000)):
    Activation = get_activation(np.asarray(USPSMat),Weights);

    Model = get_model(Activation);

    Delta_EW = np.matmul((Model - t_values).T,np.asarray(USPSMat));
    #print(np.shape(Delta_EW));
    
    La_Delta_EW  = np.dot(La,Weights);
    Delta_E = np.add(Delta_EW, np.transpose(La_Delta_EW));
    #print(np.shape(Delta_E));
    Delta_W = -np.dot(learning_rate,Delta_E);
    #print(Weights);
    #print(Delta_EW);
    W_next = Weights + np.transpose(Delta_W);
    #print(np.shape(W_next));
    Weights = W_next;
    #print(W_next);
    Loss,accuracy = get_cross_entropy(np.asarray(USPSMat),Weights,np.asarray(USPSTar));
    #L.append(Loss);
    #print(Loss);
    A.append(accuracy);
    L.append(Loss);
    ite.append(i);


# In[74]:


print("Accuracy for Logistics on USPS: ", A[999])


# In[139]:


#Neural Networks
import keras
from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential
(x_train, y_train), (x_test, y_test) = mnist.load_data()
YTestOrg = y_test
num_classes=10
image_vector_size=28*28
x_train = x_train.reshape(x_train.shape[0], image_vector_size)
x_test = x_test.reshape(x_test.shape[0], image_vector_size)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
USPS_test = keras.utils.to_categorical(USPSTar, num_classes)
USPS_xtest = np.asarray(USPSMat)
USPS_xtest = USPS_xtest.reshape(USPS_xtest.shape[0], image_vector_size)
print(np.shape(USPS_xtest))


# In[140]:


image_size = 784
model = Sequential()
model.add(Dense(units=32, activation='sigmoid', input_shape=(image_size,)))
model.add(Dense(units=num_classes, activation='softmax'))
model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=256, epochs=90, verbose=False, validation_split=.02)

loss,accuracy = model.evaluate(x_test, y_test, verbose=False)
print("Accuracy for Neural Network for MNIST Dataset: ",accuracy)


# In[147]:


NeuralNetYTest = model.predict_classes(x_test)
print(NeuralNetYTest.shape)
print(y_test.shape)


# In[148]:



from sklearn.metrics import confusion_matrix
print("Confusion Matrix for Neural Networks")
print(confusion_matrix(YTestOrg, NeuralNetYTest))
pd.crosstab(YTestOrg, NeuralNetYTest,rownames=['Actual'],colnames=['Predicted'],margins=True)


# In[28]:


loss,accuracy = model.evaluate(USPS_xtest, USPS_test, verbose=False)
print("Accuracy for Neural Network on USPS Datset: ",accuracy)


# In[150]:


#Random Forest
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_mldata
from sklearn import metrics
mnist = fetch_mldata('MNIST original')
n_train = 60000
n_test = 10000
indices = np.arange(len(mnist.data))
train_idx = np.arange(0,n_train)
test_idx = np.arange(n_train+1,n_train+n_test)
X_train, y_train = mnist.data[train_idx], mnist.target[train_idx]
X_test, y_test = mnist.data[test_idx], mnist.target[test_idx]


# In[151]:


classifier2 = RandomForestClassifier(n_estimators=60);
classifier2.fit(X_train, y_train) 
RFy_pred = classifier2.predict(X_test)
print("Accuracy for Random Forest for MNIST Dataset: ", metrics.accuracy_score(y_test, RFy_pred))


# In[152]:


print("Confusion Matrix for RF")
print(confusion_matrix(y_test, RFy_pred))
pd.crosstab(y_test, RFy_pred,rownames=['Actual'],colnames=['Predicted'],margins=True)


# In[31]:


USPSXtest = np.asarray(USPSMat)
USPS_pred = classifier2.predict(USPSXtest)
print("Accuracy for Random Forest for USPS Dataset: ", metrics.accuracy_score(USPSTar, USPS_pred))


# In[131]:


n_train = 20000
n_test = 10000
indices = np.arange(len(mnist.data))
train_idx = np.arange(0,n_train)
test_idx = np.arange(n_train+1,n_train+n_test)
X_train, y_train = mnist.data[train_idx], mnist.target[train_idx]
X_test, y_test = mnist.data[test_idx], mnist.target[test_idx]

classifier1 = SVC(kernel='linear')
classifier1.fit(X_train, y_train)
SVMy_pred = classifier1.predict(X_test)
print("Accuracy for SVM for MNIST Dataset: ", metrics.accuracy_score(y_test, SVMy_pred))

USPS_pred = classifier1.predict(USPSXtest)
print("Accuracy for SVM for USPS Dataset: ", metrics.accuracy_score(USPSTar, USPS_pred))


# In[127]:



print("Accuracy for SVM for USPS Dataset: ", metrics.accuracy_score(USPSTar, USPS_pred))


# In[153]:


print("Confusion Matrix for SVM")
print(confusion_matrix(y_test, SVMy_pred))
pd.crosstab(y_test, SVMy_pred,rownames=['Actual'],colnames=['Predicted'],margins=True)


# In[110]:


NeuralNetYTest = model.predict_classes(X_test)


# In[115]:


Weights = np.random.rand(784,10);
t_values = np.zeros((9999,10));
La = 2;
learning_rate = 0.000012;
y = y_test.astype(int)
t_values = get_hot_target(t_values,y);
print("Shape of tvalues :");
print(np.shape(t_values));
L = [];
A = [];
ite = [];
finalTarget = []
for i in tqdm_notebook(range(0,1000)):
    Activation = get_activation(X_test,Weights);
    #print("Shape of Activation :");
    #print(np.shape(Activation));
    Model = get_model(Activation);
    #print("Shape of Model :");
    #print(np.shape(Model));
    #cost = get_cross_entropy(Model, Mtraining_target);
    #print("Cross entropy loss:")
    #print(cost);
    #print(np.shape(np.transpose(Mtraining_data[i])));
    #x = np.transpose(np.array([Mtraining_data[i]]));
    #y = np.transpose(np.array([Model-t_values[i]]));
    #Delta_EW =  np.dot(y,np.transpose(x));
    Delta_EW = np.matmul((Model - t_values).T,X_test);
    #print(np.shape(Delta_EW));
    
    La_Delta_EW  = np.dot(La,Weights);
    Delta_E = np.add(Delta_EW, np.transpose(La_Delta_EW));
    #print(np.shape(Delta_E));
    Delta_W = -np.dot(learning_rate,Delta_E);
    #print(Weights);
    #print(Delta_EW);
    W_next = Weights + np.transpose(Delta_W);
    #print(np.shape(W_next));
    Weights = W_next;
    #print(W_next);
    Loss,accuracy,finalTarget = get_cross_entropy(X_test,Weights,y);
    #L.append(Loss);
    #print(Loss);
    A.append(accuracy);
    L.append(Loss);
    ite.append(i);


# In[116]:


print(finalTarget)
print(A[999])


# In[120]:


def find_max_mode(list1):
    list_table = statistics._counts(list1)
    len_table = len(list_table)

    if len_table == 1:
        max_mode = statistics.mode(list1)
    else:
        new_list = []
        for i in range(len_table):
            new_list.append(list_table[i][0])
        max_mode = max(new_list) # use the max value here
    return max_mode
import statistics
final =[]
print(finalTarget.shape)
print(NeuralNetYTest.shape)
print(RFy_pred.shape)
print(SVMy_pred.shape)
for i in range(0,len(RFy_pred)):
    try:
        final.append(statistics.mode([finalTarget[i],NeuralNetYTest[i],RFy_pred[i],SVMy_pred[i]]))
    except:
        final.append(find_max_mode([finalTarget[i],NeuralNetYTest[i],RFy_pred[i],SVMy_pred[i]]))
print("Accuracy after Voting for MNIST Dataset: ", metrics.accuracy_score(y_test, final))


# In[156]:


print("Confusion Matrix for Ensembler")
print(y_test.shape)
#print(final.shape)
print(confusion_matrix(y_test, final))
pd.crosstab(y_test, np.asarray(final),rownames=['Actual'],colnames=['Predicted'],margins=True)

