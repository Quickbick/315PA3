
import pandas as pd
import numpy as np
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

#Stopwords
stop = stopwords.words('english')

#Read Files
x = pd.read_csv(r"./fortune-cookie-data/traindata.txt", header=None)
y = pd.read_csv(r"./fortune-cookie-data/trainlabels.txt", header=None)
z = pd.read_csv(r"./fortune-cookie-data/stoplist.txt", header=None)
predData = pd.read_csv(r"fortune-cookie-data/testdata.txt", header=None)
predLabels = pd.read_csv(r"fortune-cookie-data/testlabels.txt", header=None)

outfile = open("./output.txt", "w")

#Training Data Size
rangeX = x.size
#Test Data Size
rangePred = predData.size

#Rename Columns
x.columns = ['Data']
y.columns = ['Label']
z.columns = ['Stop']
predData.columns = ['Data']
predLabels.columns = ['Label']

#Combine Train and Test
dX = pd.concat([x, predData])

#List of Stopwords
stopwords =z["Stop"].tolist()

#Creating Tokenized Data
dX['Filt_Data'] = dX['Data'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))
del dX['Data']
dX['Tokenized_Data'] = dX.apply(lambda row: nltk.word_tokenize(row['Filt_Data']), axis=1)

#Train Labels to List
y = y['Label'].tolist()
#Test Labels to List
predLabels = predLabels['Label'].tolist()

#Training Data
v = TfidfVectorizer()

Tfidf = v.fit_transform(dX['Filt_Data'])

df1 = pd.DataFrame(Tfidf.toarray(), columns=v.get_feature_names())
#print(df1)

#Seperating Train and Test
x = df1[0:rangeX]
predData = df1[rangeX:rangeX+rangePred]

#Perceptron Implementation in Scikit Learn
ppn = Perceptron(max_iter=20, eta0=1, random_state=0, verbose=1)
#Fitting
ppn.fit(x,y)

#Using Test Data
x_pred = ppn.predict(x)
y_pred = ppn.predict(predData)

#Accuracy Calculation
outfile.write('Accuracy: %.2f' % accuracy_score(y, x_pred))
outfile.write(' %.2f \n' % accuracy_score(predLabels, y_pred))

########################
#OCR
########################

#Create List From String
def split(word):
    return list(word)

#Pre-Process Data Files
def pre_process(file):
    
    #Read All Lines
    with open(file, 'r') as f:
        lines  = f.readlines()

    #Preprocessing Steps
    #Filter Empty Lines
    #Seperate Index, Data, Label
    #Preprocess Data
    #Create Labels

    i = 0
    for line in lines:
        l = re.split(r'\t+', line)

        if len(l) > 2:

            #print(l)
            #print(l[0])
            l1 = split(l[1][2:len(l[1])])
            dsl = len(l1)
            #print("data string length: ", len(l1))
            if i == 0:
                k1 = np.array(l1)
                k = k1
                label1 = np.array(l[2])
                label = label1
            else:
                k1 = np.array(l1)
                k = np.append(k, k1)
                label1 = np.array(l[2])
                label = np.append(label, label1)
            #print(y)
            i = i + 1

        #print(i)

    #Reshape
    k2 = k.reshape(i, dsl)

    #Data Frame from Array
    dataframe = pd.DataFrame.from_records(k2)
    return(dataframe, label)

#Implement Perceptron
def implement_perceptron(train_data, train_labels, test_data, test_labels):
    #Perceptron Implementation
    ppn = Perceptron(max_iter=20, eta0=0.1, random_state=0, verbose=1)
    #Fitting
    ppn.fit(train_data, train_labels)
    #Reading Test Data
    #Using model created previously to calculate results
    tr_pred = ppn.predict(train_data)
    t_pred = ppn.predict(test_data)
    outfile.write('Accuracy: %.2f' % accuracy_score(train_labels, tr_pred))
    outfile.write(' %.2f' % accuracy_score(test_labels, t_pred))


#Read Train Data
train = r"./OCR-data/ocr_test.txt"
#Read Test Data
test = r"./OCR-data/ocr_train.txt"

#Preprocess Train Data
train_data, train_labels = pre_process(train)
#Preprocess Test Data
test_data, test_labels = pre_process(test)

#Implemet Perceptron
implement_perceptron(train_data, train_labels, test_data, test_labels)
outfile.close()