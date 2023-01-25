#Find-s
import pandas as pd
import numpy as np
data = pd.read_csv("data.csv")
print(data)

#making an array of all the attributes
concepts = np.array(data)[:,:-1]
print(" The attributes are: ",concepts)

#segragating the target that has positive and negative examples
target = np.array(data)[:,-1]
print("The target is: ",target)

#training function to implement find-s algorithm
def train(c,t):
    for i, val in enumerate(t):
        if val == "Yes":
            specific_hypothesis = c[i]
            break
             
    for i, val in enumerate(c):
        if t[i] == "Yes":
            for x in range(len(specific_hypothesis)):
                if val[x] != specific_hypothesis[x]:
                    specific_hypothesis[x] = '?'
                else:
                    pass
                 
    return specific_hypothesis

#obtaining the final hypothesis
print(" The final hypothesis is:",train(concepts,target))

------------------------------------------------------------------------------------------------------


#CANDIDATE ELIMINATION ALGORITHM
import numpy as np 
import pandas as pd

# Reading the data from CSV file
data = pd.read_csv('data.csv')
concepts = np.array(data.iloc[:,:-1])
print("\nInstances are:\n",concepts)

target = np.array(data.iloc[:,-1])
print("\nTarget Values are: ",target)

def train(concepts, target):
  for i, h in enumerate(concepts):
    if target[i]=="Yes":
      specific_h=concepts[i]
      break

  print("Intialization of specific_h and genral_h")
  print(specific_h)     
  general_h = [["?" for i in range(len(specific_h))] for i in range(len(specific_h))]
  print(general_h) 
# Checking if the hypothesis has a positive target 
  for i,h in enumerate(concepts):
    if target[i]=="Yes":
      for x in range(len(specific_h)):
        if h[x]!=specific_h[x]:
          #Change values S and G
          specific_h[x]='?'
          general_h[x][x]='?'
## Checking if the hypothesis has a negative target 
    if target[i]=='No':
       for x in range(len(specific_h)):
         # for negative hypothisis change values only in G
         if h[x]!=specific_h[x]:
           general_h[x][x]=specific_h[x]
         else:
           general_h[x][x]='?'
  print("steps of Candidate Elimination Algorithm",i+1) 
  print(specific_h)  
  print(general_h)
  # find indices where we have empty rows, meaning that are unchanged 
  indices = [i for i, val in enumerate(general_h) if val == ['?', '?', '?', '?', '?', '?']] 
  for i in indices:
    # remove those rows from general_h
    general_h.remove(['?', '?', '?', '?', '?', '?'])
# return final values 
  return specific_h, general_h       
s_final, g_final = train(concepts, target)
# displaying Specific_hypothesis
print("Final Specific_h: ", s_final, sep="\n")
# displaying Generalized_Hypothesis
print("Final General_h: ", g_final, sep="\n") 
------------------------------------------------------------------------------------------------------

SIMPLE LINEAR REGRESSION
#.1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#2.

data=pd.read_csv("headbrain6.csv")
data.head()

x=data.iloc[:,2:3].values  #INPUT VARIABLE(Head Size(cm^3))

y=data.iloc[:,3:4].values   #OUTPUT VARIABLE(Brain Weight(grams))
  

#3.
from sklearn .model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=1/4, random_state=0)

#4.
from sklearn.linear_model import LinearRegression
reg=LinearRegression().fit(x_train,y_train)
reg

#5.

y_pred=reg.predict(x_test)
y_pred


#6.

plt.scatter(x_train,y_train,c='red')
plt.xlabel("Head Size")
plt.ylabel("Brain Weight")
plt.show()

# To See the relationship between the predicted brain weight values using scatter plot.(

plt.plot(x_test,y_pred)
plt.scatter(x_test,y_test,c='red')
plt.xlabel("Head Size")
plt.ylabel("Brain Weight")
plt.show()


#7.

print("Final RMSE Value",np.sqrt(np.mean((y_test-y_pred)**2)))

------------------------------------------------------------------------------------------------------

#Decision tree Classifier

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
from sklearn.tree import export_text


iris = load_iris()
iris.data

iris.target

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.5, random_state=0)


print("x_train\n",X_train)
print("y_train\n",y_train)
print("x_test\n",X_test)
print("y_test\n",y_test)



clf = DecisionTreeClassifier(random_state=0,max_depth=2)
clf.fit(X_train,y_train)

clf.predict(X_test)

print("Avg. Accuracy",clf.score(X_train,y_train))



plot_tree(clf)
r = export_text(clf, feature_names=iris['feature_names'])
print(r)




  
------------------------------------------------------------------------------------------------------


#SVM
#Data Pre-processing Step  
# importing libraries  
import numpy as nm  
import matplotlib.pyplot as plt  
import pandas as pd  
#importing datasets  
data_set= pd.read_csv('user_data.csv')
data_set
#Extracting Independent and dependent Variable  
x= data_set.iloc[:, [2,3]].values  
y= data_set.iloc[:, 4].values   

print(x)
print(y)

#feature Scaling  
from sklearn.preprocessing import StandardScaler    
st_x= StandardScaler()    
x_train= st_x.fit_transform(x_train)  

x_test= st_x.transform(x_test)


print(x_train)
print(x_test)

#Fitting the SVM classifier to the training set: 
from sklearn.svm import SVC # "Support vector classifier"  
classifier = SVC(kernel='linear', random_state=0)  
classifier.fit(x_train, y_train)  	

#Predicting the test set result  
y_pred= classifier.predict(x_test)  
y_pred


#Creating the Confusion matrix  
from sklearn.metrics import confusion_matrix  
cm= confusion_matrix(y_test, y_pred)  
cm



---------------------------------------------------
#NBA

import numpy as np
import pandas as pd
tennis = pd.read_csv('Tennisdataset.csv')
tennis


ct_outlook=pd.crosstab(tennis['outlook'], tennis['play'], margins = True)
ct_outlook

ct_temp=pd.crosstab(tennis['temp'], tennis['play'], margins = True)
ct_temp
#ct_temp.iloc[1,1]/ ct_temp.iloc[3,1]= 2/9

ct_humidity=pd.crosstab(tennis['humidity'], tennis['play'], margins = True)
ct_humidity

play windy


# calculation of P(yes/(sunny,hot,high,true))= p(yes) *
#[ P(sunny/yes)* p(hot/yes) * p(high/yes)* p(true/yes)  ]

# X'= (Outlook= sunny, temp= hot,humidity =high, windy= true) = NO

# P(yes/X') = p(yes) 9/14 *  [ P(sunny/yes)* p(hot/yes) * p(high/yes)* p(true/yes)] =0.004
# P(no/X')

p_yes = ct_play.iloc[1,0] /ct_play.iloc[2,0] * ct_outlook.iloc[2,1]/ct_outlook.iloc[3,1]* ct_temp.iloc[1,1]/ct_temp.iloc[3,1]*ct_humidity.iloc[0,1]/ct_humidity.iloc[2,1]*ct_windy.iloc[1,1]/ct_windy.iloc[2,1] 

print('The probability of tennis played with given conditions is', '%.3f'%p_yes)


# calculation of P(no/(sunny,hot,high,true))= p(no) 
#*  [ P(sunny/no)* p(hot/no) * p(high/no)* p(true/no)  ]

p_no = ct_play.iloc[0,0] /ct_play.iloc[2,0] * ct_outlook.iloc[2,0]/ct_outlook.iloc[3,0] * ct_temp.iloc[1,0]/ct_temp.iloc[3,0]*ct_humidity.iloc[0,0]/ct_humidity.iloc[2,0]*ct_windy.iloc[1,0]/ct_windy.iloc[2,0] 

print('The probability of tennis not played with given conditions is', '%.3f'%p_no)



#MAP 
if p_yes > p_no:
      print('tennis match will be conducted when the outlook is sunny, 
            the temperature is hot, there is high humidity and windy is higher')
else:
      print('tennis match will not be conducted when the outlook is sunny, 
            the temperature is hot, there is high humidity and windy is higher')


P(h=yes/X')= P(yes) * P(X1,X2,X3,X4/yes)

P(h=No/X')= P(No) * P(X1,X2,X3,X4/No)



=------------------------------------------------------------

#BBN
import pandas as pd
import numpy as np

# import bayespy as bp
#import warnings
#warnings.filterwarnings('ignore')

!pip install pgmpy
!pip install bayespy


heart_disease=pd.read_csv("data7_heart.csv")
print(heart_disease)


print('Columns in the dataset')
heart_disease.columns



from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
model=BayesianModel([('age','trestbps'), ('age', 'fbs'), ('sex', 'trestbps'), ('exang',
'trestbps'),('trestbps','heartdisease'),('fbs','heartdisease'),('heartdisease','restecg'),
('heartdisease','thalach'), ('heartdisease','chol')])
model.fit(heart_disease, estimator=MaximumLikelihoodEstimator)


# Inferencing with Bayesian Network

from pgmpy.inference import VariableElimination
HeartDisease_infer = VariableElimination(model)
q = HeartDisease_infer.query(variables=['heartdisease'], evidence={'chol':250 })
print(q)



--------------------------------------------------------------------------
IRIs

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("Iris.csv")
data

x=data.iloc[:, [1,2,3,4]].values
y=data.iloc[:,-1]


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20)


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train)

y_pred=knn.predict(x_test)

from sklearn.metrics import classification_report, confusion_matrix 
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))



=--------------------------------------------------------------------
#K-MEANS


#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv("Mall_Customers.csv")
x = dataset.iloc[:,[3,4]].values
x



#Using the elbow method
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
  kmeans= KMeans(n_clusters=i , init= 'k-means++',random_state=42)
  kmeans.fit(x)
  wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.xlabel('n of clusters')
plt.ylabel('WCSS')
plt.show()



#Training the K-Means model 
kmeans= KMeans(n_clusters=5 , init= 'k-means++',random_state=42)
y_kmeans = kmeans.fit_predict(x)
print(y_kmeans)




#Visualising the clusters
plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1], s=100,c='red',label = 'cluster1')
plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1], s=100,c='blue',label = 'cluster2')
plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1], s=100,c='green',label = 'cluster3')
plt.scatter(x[y_kmeans==3,0],x[y_kmeans==3,1], s=100,c='pink',label = 'cluster4')
plt.scatter(x[y_kmeans==4,0],x[y_kmeans==4,1], s=100,c='cyan',label = 'cluster5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='magenta',label='CENTROIDS')
plt.title('Clusters')
plt.xlabel('Annual_Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()

==============================================================================4

#HAC

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('Wholesale customers data.csv')
print(data.head())
data.shape


from sklearn.preprocessing import normalize
data_scaled = normalize(data)
data_scaled = pd.DataFrame(data_scaled, columns=data.columns)
data_scaled.head()

import scipy.cluster.hierarchy as shc
plt.figure(figsize=(10, 7))  
plt.title("Dendrograms")  

dend = shc.dendrogram(shc.linkage(data_scaled, method='ward'))

import scipy.cluster.hierarchy as shc
plt.figure(figsize=(10, 7))  
plt.title("Dendrograms")  

dend = shc.dendrogram(shc.linkage(data_scaled, method='ward'))


from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')  
cluster.fit_predict(data_scaled)


plt.figure(figsize=(10, 7))  
plt.scatter(data_scaled['Milk'], data_scaled['Grocery'], c=cluster.labels_) 


