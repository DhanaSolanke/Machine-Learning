
# coding: utf-8

# In[130]:


import numpy as np
import csv
import math
import pandas as pd
import matplotlib.pyplot
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd


# In[131]:


def HumanDataSetSubstractionMethod(samepairs,diffpairs,humandat):
    dfval = pd.DataFrame(columns=['img_id_A','img_id_B',"FA1","FA2", "FA3",  "FA4",  "FA5", "FA6",  "FA7", "FA8", "FA9","Out"])
    for index, row in samepairs.iterrows():
        #print(row['img_id_A'])
        arr = [];
        arr1 =[];
        df1 = humdat.loc[row['img_id_A']]
        arr.append(row['img_id_A'])
        arr.append(row['img_id_B'])
        for m in df1:
            arr.append(m)

        #l1 = df1[:][0]

        #d = df1.rename(columns={"f1": "FA1", "f2": "FA2", "f3": "FA3", "f4": "FA4", "f5": "FA5", "f6": "FA6", "f7": "FA7", "f8": "FA8", "f9": "FA9"}, axis='columns')
        #print(df1.T[1])
        df2 = humdat.loc[row['img_id_B']]
        i = 2
        for m1 in df2:
            arr[i] = int(arr[i]) - int(m1)
            i = i + 1
        #l = pd.concat([df1,df2])
        #print(arr)
        arr.append(row['target'])
        s = pd.Series(arr, index=['img_id_A','img_id_B',"FA1","FA2", "FA3",  "FA4",  "FA5", "FA6",  "FA7", "FA8", "FA9","Out"])
        dfval = dfval.append(s, ignore_index=True)
        
    k = 0
    for index, row in diffpairs.iterrows():
        k =k+1
        if k > 791:
            #print(k)
            break
        arr2 = [];
        df1 = humdat.loc[row['img_id_A']]
        arr2.append(row['img_id_A'])
        arr2.append(row['img_id_B'])
        for m in df1:
            arr2.append(m)
        df2 = humdat.loc[row['img_id_B']]
        i = 2
        for m1 in df2:
            arr[i] = int(arr[i]) - int(m1)
            i = i + 1
        arr2.append(row['target'])
        s = pd.Series(arr2,
                      index=['img_id_A', 'img_id_B', "FA1", "FA2", "FA3", "FA4", "FA5", "FA6", "FA7", "FA8", "FA9", "Out"])
        dfval = dfval.append(s, ignore_index=True)
    return dfval

def humanDataSetConcantenate(samepairs,diffpairs,humandat):
    dfval = pd.DataFrame(columns=['img_id_A','img_id_B',"FA1","FA2", "FA3",  "FA4",  "FA5", "FA6",  "FA7", "FA8", "FA9","FB1","FB2", "FB3",  "FB4",  "FB5", "FB6",  "FB7", "FB8", "FB9","Out"])
    k = 0
    for index, row in samepairs.iterrows():
        #print(row['img_id_A'])
        #k = k + 1
        #if k > 10:
            #break
        arr = [];
        arr1 =[];

        df1 = humdat.loc[row['img_id_A']]
        arr.append(row['img_id_A'])
        arr.append(row['img_id_B'])
        for m in df1:
            arr.append(m)
        df2 = humdat.loc[row['img_id_B']]
        for m1 in df2:
            arr.append(m1)
        #l = pd.concat([df1,df2])
        #print(arr)
        arr.append(row['target'])
        s = pd.Series(arr, index=['img_id_A','img_id_B',"FA1","FA2", "FA3",  "FA4",  "FA5", "FA6",  "FA7", "FA8", "FA9","FB1","FB2", "FB3",  "FB4",  "FB5", "FB6",  "FB7", "FB8", "FB9","Out"])
        dfval = dfval.append(s, ignore_index=True)
    
    k = 0
    for index, row in diffpairs.iterrows():
        k =k+1
        if k > 791:
            break
        arr2 = [];
        df1 = humdat.loc[row['img_id_A']]
        arr2.append(row['img_id_A'])
        arr2.append(row['img_id_B'])
        for m in df1:
            arr2.append(m)
        df2 = humdat.loc[row['img_id_B']]
        for m1 in df2:
            arr2.append(m1)
        arr2.append(row['target'])
        s = pd.Series(arr2,
                      index=['img_id_A', 'img_id_B', "FA1", "FA2", "FA3", "FA4", "FA5", "FA6", "FA7", "FA8", "FA9", "FB1",
                             "FB2", "FB3", "FB4", "FB5", "FB6", "FB7", "FB8", "FB9", "Out"])
        dfval = dfval.append(s, ignore_index=True)
    return dfval


# In[195]:


def GenerateTrainingTarget(rawTraining, TrainingPercent=80):
    TrainingLen = int(math.ceil(len(rawTraining) * (TrainingPercent * 0.01)))
    t = rawTraining[:TrainingLen]
    # print(str(TrainingPercent) + "% Training Target Generated..")
    return t

def GenerateTrainingDataMatrix(rawData, TrainingPercent = 80):
    T_len = int(math.ceil(len(rawData['FA1'])*(0.01*TrainingPercent)))
    #print(T_len)
    d2 = rawData[:T_len]
    #print(str(TrainingPercent) + "% Training Data Generated..")
    return d2

def GenerateTrainingDataMatrixGSC(rawData, TrainingPercent = 80):
    T_len = int(math.ceil(len(rawData[3])*(0.01*TrainingPercent)))
    #print(T_len)
    d2 = rawData[:T_len]
    #print(str(TrainingPercent) + "% Training Data Generated..")
    return d2

def GetPhiMatrix(Data, MuMatrix, BigSigma, TrainingPercent):
    DataT = np.transpose(Data)
    TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))
    #print(TrainingLen)
    PHI = np.zeros((int(TrainingLen),len(MuMatrix)))
    BigSigInv = np.linalg.inv(BigSigma)
    for  C in range(0,len(MuMatrix)):
        for R in range(0,int(TrainingLen)):
            PHI[R][C] = GetRadialBasisOut(DataT[R], MuMatrix[C], BigSigInv)
    #print ("PHI Generated..")
    return PHI

def GetRadialBasisOut(DataRow,MuRow, BigSigInv):
    phi_x = math.exp(-0.5*GetScalar(DataRow,MuRow,BigSigInv))
    return phi_x

def GetScalar(DataRow,MuRow, BigSigInv):
    R = np.subtract(DataRow,MuRow)
    T = np.dot(BigSigInv,np.transpose(R))
    L = np.dot(R,T)
    return L

def GetValTest(VAL_PHI, W):
    Y = np.dot(W, np.transpose(VAL_PHI))
    #print ("Test Out Generated..")
    #print(Y)
    return Y


def GetErms(VAL_TEST_OUT, ValDataAct):
    sum = 0.0
    t = 0
    accuracy = 0.0
    counter = 0
    val = 0.0
    for i in range(0, len(VAL_TEST_OUT)):
        sum = sum + math.pow((ValDataAct[i] - VAL_TEST_OUT[i]), 2)
        if (int(np.around(VAL_TEST_OUT[i], 0)) == ValDataAct[i]):
            counter += 1
    accuracy = (float((counter * 100)) / float(len(VAL_TEST_OUT)))
    #print ("Accuracy Generated..")
    #print ("E_RMS : " + str(math.sqrt(sum/len(VAL_TEST_OUT))))
    return (str(accuracy) + ',' + str(math.sqrt(sum / len(VAL_TEST_OUT))))

def GenerateValData(rawData, ValPercent, TrainingCount):
    valSize = int(math.ceil(len(rawData[0]) * ValPercent * 0.01))
    V_End = TrainingCount + valSize
    dataMatrix = rawData[:, TrainingCount + 1:V_End]
    #print (str(ValPercent) + "% Val Data Generated..")
    return dataMatrix


def GenerateValTargetVector(rawData, ValPercent, TrainingCount):
    valSize = int(math.ceil(len(rawData) * ValPercent * 0.01))
    V_End = TrainingCount + valSize
    t = rawData[TrainingCount + 1:V_End]
    #print (str(ValPercent) + "% Val Target Data Generated..")
    return t

def GenerateBigSigma(Data, MuMatrix,TrainingPercent,IsSynthetic):
    BigSigma    = np.zeros((len(Data),len(Data)))
    DataT       = np.transpose(Data)
    TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))        
    varVect     = []
    for i in range(0,len(DataT[0])):
        vct = []
        for j in range(0,int(TrainingLen)):
            vct.append(Data[i][j])    
        varVect.append(np.var(vct))
    
    for j in range(len(Data)):
        BigSigma[j][j] = varVect[j]
    if IsSynthetic == True:
        BigSigma = np.dot(3,BigSigma)
    else:
        BigSigma = np.dot(200,BigSigma)
    ##print ("BigSigma Generated..")
    return BigSigma

def GetPhiMatrixGCS(Data, MuMatrix, BigSigma, TrainingPercent):
    DataT = np.transpose(Data)
    TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))
    #print(TrainingLen)
    PHI = np.zeros((int(TrainingLen),len(MuMatrix)))
    BigSigInv = np.linalg.inv(BigSigma)
    for  C in range(0,len(MuMatrix)):
        for R in range(0,int(TrainingLen)):
            PHI[R][C] = GetRadialBasisOut(DataT[R], MuMatrix[C], BigSigInv)
    #print ("PHI Generated..")
    return PHI


# In[133]:


def performRegressionHumDat(dfval):
    #print("DFVAL")
    #print(dfval.shape)
    humandfcopy = dfval.copy()
    traindata = humandfcopy.drop(columns=['img_id_A', 'img_id_B','Out'])
    target = dfval[['Out']].copy()
    B = np.array([0, 0, 0])
    
    TrainingPercent = 80
    TrainingTarget = np.array(GenerateTrainingTarget(target,TrainingPercent))
    TrainingData   = GenerateTrainingDataMatrix(traindata,TrainingPercent)
    TrainingData = TrainingData.T
    #print(TrainingTarget.shape)
    #print(TrainingData.shape)
    
    M = 10
    kmeans = KMeans(n_clusters=M, random_state=0).fit(np.transpose(TrainingData))
    Mu = kmeans.cluster_centers_
    
    RawData = np.array(traindata.T)
    #print(RawData.shape)
    BigSigma     = GenerateBigSigma(RawData, Mu, TrainingPercent,False)
    #print(TrainingTarget.shape)
    TRAINING_PHI = GetPhiMatrix(RawData, Mu, BigSigma, TrainingPercent)
    
    ValDataAct = np.array(GenerateValTargetVector(target,10, (len(TrainingTarget))))
    ValData    = GenerateValData(RawData,10, (len(TrainingTarget)))
    #print(ValDataAct.shape)
    #print(ValData.shape)
    
    TestDataAct = np.array(GenerateValTargetVector(target,10, (len(TrainingTarget)+len(ValDataAct))))
    TestData = GenerateValData(RawData,10, (len(TrainingTarget)+len(ValDataAct)))
    #print(TEST_PHI.shape)
    #print(ValData.shape)
    
    TEST_PHI = GetPhiMatrix(TestData, Mu, BigSigma, 100)
    VAL_PHI = GetPhiMatrix(ValData, Mu, BigSigma, 100)
    #BigSigma     = GenerateBigSigma(RawData, Mu, TrainingPercent,IsSynthetic)
    #TRAINING_PHI = GetPhiMatrix(RawData, Mu, BigSigma, TrainingPercent)

    w= [0,0,0,0,0,0,0,0,0,0]
    W_Now        = np.dot(220, np.array(w))
    La           = 2
    learningRate = 0.01
    L_Erms_Val   = []
    L_Erms_TR    = []
    L_Erms_Test  = []
    W_Mat        = []
    
    for i in range(0, 400):
        # print ('---------Iteration: ' + str(i) + '--------------')
        Delta_E_D = -np.dot((TrainingTarget[i][0] - np.dot(np.transpose(W_Now), TRAINING_PHI[i])), TRAINING_PHI[i])
        La_Delta_E_W = np.dot(La, W_Now)
        Delta_E = np.add(Delta_E_D, La_Delta_E_W)
        Delta_W = -np.dot(learningRate, Delta_E)
        W_T_Next = W_Now + Delta_W
        W_Now = W_T_Next

        # -----------------TrainingData Accuracy---------------------#
        TR_TEST_OUT = GetValTest(TRAINING_PHI, W_T_Next)
        Erms_TR = GetErms(TR_TEST_OUT, TrainingTarget)
        L_Erms_TR.append(float(Erms_TR.split(',')[1]))

        # -----------------ValidationData Accuracy---------------------#
        VAL_TEST_OUT = GetValTest(VAL_PHI, W_T_Next)
        Erms_Val = GetErms(VAL_TEST_OUT, ValDataAct)
        L_Erms_Val.append(float(Erms_Val.split(',')[1]))

        # -----------------TestingData Accuracy---------------------#
        TEST_OUT = GetValTest(TEST_PHI, W_T_Next)
        Erms_Test = GetErms(TEST_OUT, TestDataAct)
        L_Erms_Test.append(float(Erms_Test.split(',')[1]))
        
    print('----------Gradient Descent Solution--------------------')
    print("E_rms Training   = " + str(np.around(min(L_Erms_TR), 5)))
    print("E_rms Validation = " + str(np.around(min(L_Erms_Val), 5)))
    print("E_rms Testing    = " + str(np.around(min(L_Erms_Test), 5)))


# In[134]:


humdat = pd.read_csv('HumanObserved-Features-Data.csv')
samepairs =  pd.read_csv('same_pairs.csv')
diffpairs = pd.read_csv('diffn_pairs.csv')

humdat = humdat.drop(humdat.columns[[0]],axis=1)
humdat.set_index('img_id', inplace=True)
dfval = humanDataSetConcantenate(samepairs,diffpairs,humdat)
print("Linear Regrssion: Concantenation (Human Observed Dataset)")
performRegressionHumDat(dfval)

print("Linear Regrssion: Substraction (Human Observed Dataset)")
dfvalSub = HumanDataSetSubstractionMethod(samepairs,diffpairs,humdat)
performRegressionHumDat(dfvalSub)


# In[159]:


def createDataSetGSCConcat(gcsfeat,gcssame,gcsdiff):
    k = 0
    PHI = []
    IsSynthetic = False
    dfgcs =  pd.DataFrame()

    for index, row in gcssame.iterrows():
        arrgcs = [];
        k = k + 1
        if k > 2000:
            break
        val = np.array(row).T
        #print(val)
        gcsdf =np.array(gcsfeat.loc[gcsfeat['img_id'] == str(val[0])])[0]
        #gcsfeat.loc[str(val[0])] #gcsfeat.loc[val['img_id']]
        gcsdf = np.delete(gcsdf, 0, 0)
        #print(gcsdf.T)
        #gcsdf = gcsdf.drop(gcsdf.columns[0], axis=1)
        #v = gcsdf.T
        #v = v.iloc[0:]
        #d = gcsdf[1]
        #print(np.array(row).T)
        arrgcs.append(val[0])
        arrgcs.append(val[1])
        for m in gcsdf:
            arrgcs.append(m)
        gcsdf2 = np.array(gcsfeat.loc[gcsfeat['img_id'] == str(val[1])])[0]
        #print(gcsdf2)
        gcsdf2 = np.delete(gcsdf2, 0, 0)
       # print(gcsdf2)
        for l in gcsdf2:
            arrgcs.append(l)
        #print(arrgcs)
        #for m1 in arrgcs:
        #    arrgcs.append(m1)
        arrgcs.append(val[2])
        s = pd.Series(arrgcs)#index=['img_id_A','img_id_B',"FA1","FA2", "FA3",  "FA4",  "FA5", "FA6",  "FA7", "FA8", "FA9","FB1","FB2", "FB3",  "FB4",  "FB5", "FB6",  "FB7", "FB8", "FB9","Out"])
        dfgcs = dfgcs.append(s, ignore_index=True)
    
    for index, row in gcsdiff.iterrows():
        arrgcs = [];
        k = k + 1
        if k > 2000:
            break
        val = np.array(row).T
        #print(val)
        gcsdf =np.array(gcsfeat.loc[gcsfeat['img_id'] == str(val[0])])[0]
        #gcsfeat.loc[str(val[0])] #gcsfeat.loc[val['img_id']]
        gcsdf = np.delete(gcsdf, 0, 0)
        #print(gcsdf.T)
        #gcsdf = gcsdf.drop(gcsdf.columns[0], axis=1)
        #v = gcsdf.T
        #v = v.iloc[0:]
        #d = gcsdf[1]
        #print(np.array(row).T)
        arrgcs.append(val[0])
        arrgcs.append(val[1])
        for m in gcsdf:
            arrgcs.append(m)
        gcsdf2 = np.array(gcsfeat.loc[gcsfeat['img_id'] == str(val[1])])[0]
        #print(gcsdf2)
        gcsdf2 = np.delete(gcsdf2, 0, 0)
       # print(gcsdf2)
        for l in gcsdf2:
            arrgcs.append(l)
        #print(arrgcs)
        #for m1 in arrgcs:
        #    arrgcs.append(m1)
        arrgcs.append(val[2])
        s = pd.Series(arrgcs)#index=['img_id_A','img_id_B',"FA1","FA2", "FA3",  "FA4",  "FA5", "FA6",  "FA7", "FA8", "FA9","FB1","FB2", "FB3",  "FB4",  "FB5", "FB6",  "FB7", "FB8", "FB9","Out"])
        dfgcs = dfgcs.append(s, ignore_index=True)

    #dfgcs = pd.DataFrame(np.array(arrgcs))
    return dfgcs


# In[166]:


def createDataSetGSCSub(gcsfeat,gcssame,gcsdiff):
    k = 0
    PHI = []
    IsSynthetic = False
    dfgcs =  pd.DataFrame()

    for index, row in gcssame.iterrows():
        arrgcs = [];
        k = k + 1
        if k > 2000:
            break
        val = np.array(row).T
        #print(val)
        gcsdf =np.array(gcsfeat.loc[gcsfeat['img_id'] == str(val[0])])[0]
        #gcsfeat.loc[str(val[0])] #gcsfeat.loc[val['img_id']]
        gcsdf = np.delete(gcsdf, 0, 0)
        arrgcs.append(val[0])
        arrgcs.append(val[1])
        for m in gcsdf:
            arrgcs.append(m)
        gcsdf2 = np.array(gcsfeat.loc[gcsfeat['img_id'] == str(val[1])])[0]
        #print(gcsdf2)
        gcsdf2 = np.delete(gcsdf2, 0, 0)
       # print(gcsdf2)
        i = 2
        for l in gcsdf2:
            arrgcs[i] = abs(int(arrgcs[i]) - int(l))
            i = i + 1
            #arrgcs.append(l)
        #print(arrgcs)
        #for m1 in arrgcs:
        #    arrgcs.append(m1)
        arrgcs.append(val[2])
        s = pd.Series(arrgcs)
        #index=['img_id_A','img_id_B',"FA1","FA2", "FA3",  "FA4",  "FA5", "FA6",  "FA7", "FA8", "FA9","FB1","FB2", "FB3",  "FB4",  "FB5", "FB6",  "FB7", "FB8", "FB9","Out"])
        dfgcs = dfgcs.append(s, ignore_index=True)
    
    for index, row in gcsdiff.iterrows():
        arrgcs = [];
        k = k + 1
        if k > 2000:
            break
        val = np.array(row).T
        #print(val)
        gcsdf =np.array(gcsfeat.loc[gcsfeat['img_id'] == str(val[0])])[0]
        #gcsfeat.loc[str(val[0])] #gcsfeat.loc[val['img_id']]
        gcsdf = np.delete(gcsdf, 0, 0)
        arrgcs.append(val[0])
        arrgcs.append(val[1])
        for m in gcsdf:
            arrgcs.append(m)
        gcsdf2 = np.array(gcsfeat.loc[gcsfeat['img_id'] == str(val[1])])[0]
        #print(gcsdf2)
        gcsdf2 = np.delete(gcsdf2, 0, 0)
       # print(gcsdf2)
        i = 2
        for l in gcsdf2:
            arrgcs[i] = abs(int(arrgcs[i]) - int(l))
            i = i + 1
        #print(arrgcs)
        #for m1 in arrgcs:
        #    arrgcs.append(m1)
        arrgcs.append(val[2])
        s = pd.Series(arrgcs)#index=['img_id_A','img_id_B',"FA1","FA2", "FA3",  "FA4",  "FA5", "FA6",  "FA7", "FA8", "FA9","FB1","FB2", "FB3",  "FB4",  "FB5", "FB6",  "FB7", "FB8", "FB9","Out"])
        dfgcs = dfgcs.append(s, ignore_index=True)

    #dfgcs = pd.DataFrame(np.array(arrgcs))
    return dfgcs


# In[196]:


def performRegressionGCS(dfval,case):
    #print("DFVAL")
    #print(dfval.shape)
    humandfcopy = dfval.copy()
    if case == "Sub":
        traindata = humandfcopy.drop(columns=[0,1,514])
        target = dfval[[514]].copy()
    else:
        traindata = humandfcopy.drop(columns=[0,1,1026])
        target = dfval[[1026]].copy()
        
    print(traindata.shape)
    print(target.shape)

    TrainingPercent = 80
    TrainingTarget = np.array(GenerateTrainingTarget(target,TrainingPercent))
    TrainingData   = GenerateTrainingDataMatrixGSC(traindata,TrainingPercent)
    TrainingData = TrainingData.T
    #print(TrainingTarget.shape)
    #print(TrainingData.shape)
    
    M = 10
    kmeans = KMeans(n_clusters=M, random_state=0).fit(np.transpose(TrainingData))
    Mu = kmeans.cluster_centers_
    #print(Mu.shape)
    RawData = np.array(traindata.T)
    #print(RawData.shape)
    BigSigma     = GenerateBigSigma(RawData, Mu, TrainingPercent,False)
    np.fill_diagonal(BigSigma,0.2)
    #print(BigSigma)
    #print(BigSigma.shape)
    #print(TrainingTarget.shape)
    TRAINING_PHI = GetPhiMatrixGCS(RawData, Mu, BigSigma, TrainingPercent)
    
    ValDataAct = np.array(GenerateValTargetVector(target,10, (len(TrainingTarget))))
    ValData    = GenerateValData(RawData,10, (len(TrainingTarget)))
    #print(ValDataAct.shape)
    #print(ValData.shape)
    
    TestDataAct = np.array(GenerateValTargetVector(target,10, (len(TrainingTarget)+len(ValDataAct))))
    TestData = GenerateValData(RawData,10, (len(TrainingTarget)+len(ValDataAct)))
    #print(TEST_PHI.shape)
    #print(ValData.shape)
    
    TEST_PHI = GetPhiMatrix(TestData, Mu, BigSigma, 100)
    VAL_PHI = GetPhiMatrix(ValData, Mu, BigSigma, 100)
    #BigSigma     = GenerateBigSigma(RawData, Mu, TrainingPercent,IsSynthetic)
    #TRAINING_PHI = GetPhiMatrix(RawData, Mu, BigSigma, TrainingPercent)

    w= [0,0,0,0,0,0,0,0,0,0]
    W_Now        = np.dot(220, np.array(w))
    La           = 2
    learningRate = 0.01
    L_Erms_Val   = []
    L_Erms_TR    = []
    L_Erms_Test  = []
    W_Mat        = []
    
    for i in range(0, 400):
        # print ('---------Iteration: ' + str(i) + '--------------')
        Delta_E_D = -np.dot((TrainingTarget[i][0] - np.dot(np.transpose(W_Now), TRAINING_PHI[i])), TRAINING_PHI[i])
        La_Delta_E_W = np.dot(La, W_Now)
        Delta_E = np.add(Delta_E_D, La_Delta_E_W)
        Delta_W = -np.dot(learningRate, Delta_E)
        W_T_Next = W_Now + Delta_W
        W_Now = W_T_Next

        # -----------------TrainingData Accuracy---------------------#
        TR_TEST_OUT = GetValTest(TRAINING_PHI, W_T_Next)
        Erms_TR = GetErms(TR_TEST_OUT, TrainingTarget)
        L_Erms_TR.append(float(Erms_TR.split(',')[1]))

        # -----------------ValidationData Accuracy---------------------#
        VAL_TEST_OUT = GetValTest(VAL_PHI, W_T_Next)
        Erms_Val = GetErms(VAL_TEST_OUT, ValDataAct)
        L_Erms_Val.append(float(Erms_Val.split(',')[1]))

        # -----------------TestingData Accuracy---------------------#
        TEST_OUT = GetValTest(TEST_PHI, W_T_Next)
        Erms_Test = GetErms(TEST_OUT, TestDataAct)
        L_Erms_Test.append(float(Erms_Test.split(',')[1]))
        
    print('----------Gradient Descent Solution--------------------')
    print("E_rms Training   = " + str(np.around(min(L_Erms_TR), 5)))
    print("E_rms Validation = " + str(np.around(min(L_Erms_Val), 5)))
    print("E_rms Testing    = " + str(np.around(min(L_Erms_Test), 5)))


# In[185]:


gcsfeat = pd.read_csv('gsc/GSC-Features.csv')
gcsdiff = pd.read_csv('gsc/diffn_pairs.csv')
gcssame = pd.read_csv('gsc/same_pairs.csv')
gcsfeat1 = gcsfeat.drop(gcsfeat.columns[0],axis=1)


# In[197]:



dfgcs = createDataSetGSCConcat(gcsfeat,gcsdiff,gcssame)
print("Linear Regression: Concantenation (GSC Dataset)")
performRegressionGCS(dfgcs,"")
print("Linear Regression: Substraction (GSC Dataset)")
dfgcsSub = createDataSetGSCSub(gcsfeat,gcsdiff,gcssame)
performRegressionGCS(dfgcsSub,"Sub")


# In[208]:


def sigmoid(z):
    return 1.0 / (1 + np.exp(-1.0*z))

def thresh(val):
    if (val >= 0.5):
        return 1
    else:
        return 0


# In[261]:


def performLogisticRegression(Data, Target, state):
    if state == "Con":
        w1 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    else:
        w1 = [0,0,0,0,0,0,0,0,0]
    #w1= [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    from scipy import optimize as op
    w1  = np.dot(220, np.array(w1))

    predict =[]
    for i in range(0, len(Data)):
        #print("Data[i]",Data[i].shape)
        # print ('---------Iteration: ' + str(i) + '--------------')
        pdv = np.dot(w1.T,Data[i])
        sig = 1/(1 + np.exp(-pdv))
        Delta_E_D = -np.dot(Target[i][0] - sig ,Data[i])
        La_Delta_E_W = np.dot(La, w1)
        Delta_E = np.add(Delta_E_D, La_Delta_E_W)
        Delta_W = -np.dot(learningRate, Delta_E)
        W_T_Next = w1 + Delta_W
        w1 = W_T_Next
        TR_TEST_OUT= GetValTest(Data.T, W_T_Next)#np.dot(W_T_Next, TrainingData)#GetValTest(TrainingData, W_T_Next)
        #preds = np.round(1/(1 + np.exp(-TR_TEST_OUT)))
        #x = (TrainingTarget == preds).sum().astype(float) / len(TrainingTarget)
        #print(x)
        #z = np.dot(w1.T,TrainingData[i])
        #ds = np.dot(TrainingData.T,w1)
        ds = TR_TEST_OUT.astype('float64')
        z1 = sigmoid(ds)
    d = np.array(z1)
    cnt =0
    for i in range(0, len(Data)):
          if Target[i] == thresh(d[i]):
                cnt=cnt+1

    x = cnt / len(Target)
    return x

def preLogisticOp(t,tr,state,typeOp):
    
    t = dfval.drop(columns=['img_id_A', 'img_id_B','Out'])
    tr = dfval[['Out']].copy()
    TrainingPercent = 80
    TrainingTarget = np.array(GenerateTrainingTarget(tr,TrainingPercent))
    TrainingData   = GenerateTrainingDataMatrix(t,TrainingPercent)
    TrainingData = TrainingData.T

    RawData = np.array(t.T)

    ValDataAct = np.array(GenerateValTargetVector(tr,10, (len(TrainingTarget))))
    ValData    = GenerateValData(RawData,10, (len(TrainingTarget)))
    ValData = ValData.T

    TestDataAct = np.array(GenerateValTargetVector(target,10, (len(TrainingTarget)+len(ValDataAct))))
    TestData = GenerateValData(RawData,10, (len(TrainingTarget)+len(ValDataAct)))
    TestData = TestData.T

    print("Logistic Regression for ",state)
    print("Accuracy TrainingData: ",performLogisticRegression(TrainingData,TrainingTarget,typeOp))
    print("Accuracy on ValidationData: ",performLogisticRegression(ValData,ValDataAct,typeOp))
    print("Accuracy on Testing Data: ",performLogisticRegression(TestData,TestDataAct,typeOp))


# In[262]:



#Human Concantenation
traindataHumanC = dfval.drop(columns=['img_id_A', 'img_id_B','Out'])
targetHumanC = dfval[['Out']].copy()
preLogisticOp(traindataHumanC,targetHumanC,"Human Observed Dataset Concatenation","Con")

#Human Substraction
traindataHumanS = dfvalSub.drop(columns=['img_id_A', 'img_id_B','Out'])
targetHumanS = dfvalSub[['Out']].copy()
preLogisticOp(traindataHumanS,targetHumanS,"Human Observed Dataset Substraction","Sub")

#GSC Concantenate
traindataGSCS = dfgcs.drop(columns=[0,1,1026])
targetGSCS = dfgcs[[1026]].copy()
preLogisticOp(traindataGSCS,targetGSCS,"GSC Dataset Concatenation","Con")

#GSC Substraction
traindataGSCC = dfgcsSub.drop(columns=[0,1,514])
targetGSCC = dfgcsSub[[514]].copy()
preLogisticOp(traindataGSCC,targetGSCC,"GSC Dataset Substraction","Sub")

