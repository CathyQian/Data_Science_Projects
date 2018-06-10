import csv as csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

########################## step 1. upload train and test data##################
# For .read_csv, always use header=0 when you know row 0 is the header row
train = pd.read_csv('train.csv', header=0)
test = pd.read_csv('test.csv', header=0)

########################## step 2. train data analysis ########################
# analyze train data  
print(train.head(10)) # check data format
print(train.info()) # check missing values, age and embarked (missing 2 values)
print(train.describe()) # numerical data descriptions

# histograms
train.hist(figsize=(12,8))
plt.show()

# class distribution
print(train.groupby('Sex').size())
print(train.groupby('Pclass').size())
print(train.groupby('SibSp').size())
print(train.groupby('Parch').size())
print(train.groupby('Survived').size())
print(train.groupby('Embarked').size())

##################### step 3. preliminary data cleaning  ######################
# map Sex into numerical values
train['Sex'] = train['Sex'].map({'male':1,'female':0})

# fill age NUll with average values based on Gender and Pclass
train['AgeFill'] = train['Age']
median_train_ages = np.zeros((2,3))

for i in range(0, 2):
    for j in range(0, 3):
        median_train_ages[i,j] = train[(train['Sex'] == i) & (train['Pclass'] 
                                            == j+1)]['Age'].dropna().median()

for i in range(0, 2):
    for j in range(0, 3):
        train.loc[(train.Age.isnull()) & (train.Sex == i) & (train.Pclass ==
                                        j+1), 'AgeFill'] = median_train_ages[i,j]

train = train.drop(['Age'], axis=1)

######################### step 4. visualize/analyze train data  ###############

# correlation matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(train.corr(), vmin = -1, vmax = 1, interpolation = 'none')
fig.colorbar(cax)
plt.show()
# negatively correlated with Gender, Pclass, positivelyy correlated with Fare, Age

# visualize correlations to verify
# gender: women are more likely to survive
survived_sex = train[train['Survived']==1]['Sex'].value_counts()
dead_sex = train[train['Survived']==0]['Sex'].value_counts()
df = pd.DataFrame([survived_sex,dead_sex])
df.index = ['Survived','Dead']
df.plot(kind='bar',stacked=True, figsize=(8,4))

# Pclass
figure = plt.figure(figsize=(8, 4))
plt.hist([train[train['Survived']==1]['Pclass'],train[train['Survived']==0]['Pclass']], stacked=True, color = ['g','r'],
         bins = 30,label = ['Survived','Dead'])
plt.xlabel('Pclass')
plt.ylabel('Number of passengers')
plt.legend()

# fare: low fare less likely to survive
figure = plt.figure(figsize=(8, 4))
plt.hist([train[train['Survived']==1]['Fare'],train[train['Survived']==0]['Fare']], stacked=True, color = ['g','r'],
         bins = 30,label = ['Survived','Dead'])
plt.xlabel('Fare')
plt.ylabel('Number of passengers')
plt.legend()

# age: kids below 10 are more likely to survive, older than 65 are less likely.
figure = plt.figure(figsize=(8,4))
plt.hist([train[train['Survived']==1]['AgeFill'],train[train['Survived']==0]['AgeFill']],
         stacked=True, color = ['g','r'],bins = 30,label = ['Survived','Dead'])
plt.xlabel('Age')
plt.ylabel('Number of passengers')
plt.legend()

# ticket fare corelate with class
ax = plt.subplot()
ax.set_ylabel('Average fare')
train.groupby('Pclass').mean()['Fare'].plot(kind='bar',figsize=(8,4), ax = ax)

########################## step 5. feature engineering ########################
# combine train and test data
def get_combined_data():
    # reading train data
    train = pd.read_csv('train.csv', header=0)
    test = pd.read_csv('test.csv', header=0)
    
    # extracting and then removing the targets from the training data 
    target = train['Survived']
    train.drop('Survived', 1, inplace=True)

    # merging train data and test data for future feature engineering
    combined = train.append(test)
    combined.reset_index(inplace=True)
    combined.drop('index',inplace=True,axis=1)
    
    return combined, target
    
combined, train_y = get_combined_data()

# extract tittle from name
def get_titles():
    global combined
    
    # we extract the title from each name
    combined['Title'] = combined['Name'].map(lambda name:name.split(',')[1].
                        split('.')[0].strip())
    
    # a map of more aggregated titles
    Title_Dictionary = {
                        "Capt":       "Officer",
                        "Col":        "Officer",
                        "Major":      "Officer",
                        "Jonkheer":   "Royalty",
                        "Don":        "Royalty",
                        "Sir" :       "Royalty",
                        "Dr":         "Officer",
                        "Rev":        "Officer",
                        "the Countess":"Royalty",
                        "Dona":       "Royalty",
                        "Mme":        "Mrs",
                        "Mlle":       "Miss",
                        "Ms":         "Mrs",
                        "Mr" :        "Mr",
                        "Mrs" :       "Mrs",
                        "Miss" :      "Miss",
                        "Master" :    "Master",
                        "Lady" :      "Royalty"

                        }
    
    # we map each title
    combined['Title'] = combined.Title.map(Title_Dictionary)

get_titles()
combined.info()

# Let's create a function that fills in the missing age in combined based on 
# gender, Pclass, Title
grouped = combined.groupby(['Sex','Pclass','Title'])
grouped.median()

def process_age():    
    global combined   
    # a function that fills the missing values of the Age variable   
    def fillAges(row):
        if row['Sex']== 'female' and row['Pclass'] == 1:
            if row['Title'] == 'Miss':
                return 30
            elif row['Title'] == 'Mrs':
                return 45
            elif row['Title'] == 'Officer':
                return 49
            elif row['Title'] == 'Royalty':
                return 39

        elif row['Sex']== 'female' and row['Pclass'] == 2:
            if row['Title'] == 'Miss':
                return 20
            elif row['Title'] == 'Mrs':
                return 30

        elif row['Sex']== 'female' and row['Pclass'] == 3:
            if row['Title'] == 'Miss':
                return 18
            elif row['Title'] == 'Mrs':
                return 31

        elif row['Sex']== 'male' and row['Pclass'] == 1:
            if row['Title'] == 'Master':
                return 6
            elif row['Title'] == 'Mr':
                return 41.5
            elif row['Title'] == 'Officer':
                return 52
            elif row['Title'] == 'Royalty':
                return 40

        elif row['Sex']== 'male' and row['Pclass'] == 2:
            if row['Title'] == 'Master':
                return 2
            elif row['Title'] == 'Mr':
                return 30
            elif row['Title'] == 'Officer':
                return 41.5

        elif row['Sex']== 'male' and row['Pclass'] == 3:
            if row['Title'] == 'Master':
                return 6
            elif row['Title'] == 'Mr':
                return 26    
    combined.Age = combined.apply(lambda r : fillAges(r) if np.isnan(r['Age'])
                                                        else r['Age'], axis=1)

process_age()

# process fare by the mean value
def process_fares():
    
    global combined
    combined.Fare.fillna(combined.Fare.mean(),inplace=True) 

process_fares()

# map sex to numerical values (0, 1)
def process_sex():
    global combined
  
    # mapping string values to numerical one 
    combined['Sex'] = combined['Sex'].map({'male':1,'female':0})
    
process_sex()

# process name: drop Name, Title
def process_names():
    
    global combined
    # we clean the Name variable
    combined.drop('Name',axis=1,inplace=True)
    
    # encoding in dummy variable
    titles_dummies = pd.get_dummies(combined['Title'],prefix='Title')
    combined = pd.concat([combined,titles_dummies],axis=1)
    
    # removing the title variable
    combined.drop('Title',axis=1,inplace=True)

process_names()

# fill in two missing values of Embarked with the most frequent Embarked value.
def process_embarked():
    
    global combined
    # two missing embarked values - filling them with the most frequent one (S)
    combined.Embarked.fillna('S',inplace=True)
    
    # dummy encoding 
    embarked_dummies = pd.get_dummies(combined['Embarked'],prefix='Embarked')
    combined = pd.concat([combined,embarked_dummies],axis=1)
    combined.drop('Embarked',axis=1,inplace=True)
    
process_embarked()

# This function replaces NaN values with U (for Unknow). It then maps each Cabin
# value to the first letter. Then it encodes the cabin values using dummy 
# encoding again.
def process_cabin():   
    global combined   
    # replacing missing cabins with U (for Uknown)
    combined.Cabin.fillna('U',inplace=True)
    
    # mapping each Cabin value with the cabin letter
    combined['Cabin'] = combined['Cabin'].map(lambda c : c[0])
    
    # dummy encoding ...
    cabin_dummies = pd.get_dummies(combined['Cabin'],prefix='Cabin')    
    combined = pd.concat([combined,cabin_dummies],axis=1)    
    combined.drop('Cabin',axis=1,inplace=True)

process_cabin()

# tickets
# This function preprocess the tikets first by extracting the ticket prefix
def process_ticket():    
    global combined
    
    def cleanTicket(ticket):
        ticket = ticket.replace('.','')
        ticket = ticket.replace('/','')
        ticket = ticket.split()
        ticket = map(lambda t : t.strip() , ticket)
        ticket = list(filter(lambda t : not t.isdigit(), ticket))
        if len(ticket) > 0:
            return ticket[0]
        else: 
            return 'XXX'

    # Extracting dummy variables from tickets:
    combined['Ticket'] = combined['Ticket'].map(cleanTicket)
    tickets_dummies = pd.get_dummies(combined['Ticket'],prefix='Ticket')
    combined = pd.concat([combined, tickets_dummies],axis=1)
    combined.drop('Ticket',inplace=True,axis=1)
    
process_ticket()

#v4: def process_pclass() remain, v5 deleted
def process_pclass():
    
    global combined
    # encoding into 3 categories:
    pclass_dummies = pd.get_dummies(combined['Pclass'],prefix="Pclass")
    
    # adding dummy variables
    combined = pd.concat([combined,pclass_dummies],axis=1)
    
    # removing "Pclass"
    
    combined.drop('Pclass',axis=1,inplace=True)
    
    
process_pclass()

# singleton, small family, large family, large families are more likely to be rescued
def process_family():
    
    global combined
    # introducing a new feature : the size of families (including the passenger)
    combined['FamilySize'] = combined['Parch'] + combined['SibSp'] + 1
    combined.drop('Parch',inplace=True,axis=1)
    combined.drop('SibSp',inplace=True,axis=1)
   # status('family')

process_family()    
    
####################### step 6. feature scaling ############################### 
# scale features except PassengerId to 0 ~ 1

def scale_all_features():
    
    global combined
    
    features = list(combined.columns)
    features.remove('PassengerId')
    combined[features] = combined[features].apply(lambda x: x/x.max(), axis=0)
    
    print('Features scaled successfully !')      

scale_all_features()    

#################### step 7. recover test and train data ######################
# recovering the train set and the test set from the combined dataset 

def recover_train_test_target():
    global combined
    train = combined.ix[0:890]
    test = combined.ix[891:]
    
    return train,test

train, test = recover_train_test_target()

#################### step 8. feature weights and selection ####################
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

clf = ExtraTreesClassifier(n_estimators=200)
clf = clf.fit(train, train_y)

features = pd.DataFrame()
features['feature'] = train.columns
features['importance'] = clf.feature_importances_
features.sort(['importance'],ascending=False)

# choosing features by settign weight threshold
model = SelectFromModel(clf, threshold = 0.005, prefit=True)

train_new = model.transform(train) 
test_new = model.transform(test)

############################ step 9. building models ##########################
from sklearn.model_selection import train_test_split

validation_size = 0.30
seed = 6
X_train, X_validation, Y_train, Y_validation = train_test_split(train_new, train_y, test_size 
                                               = validation_size, random_state = seed)
# standardize the dataset
pipelines = []
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()), ('LR', LogisticRegression())])))
pipelines.append(('ScaledLDA', Pipeline([('Scaler', StandardScaler()), ('LDA', LinearDiscriminantAnalysis())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()), ('KNN', KNeighborsClassifier())])))
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()), ('CART', DecisionTreeClassifier())])))
pipelines.append(('ScaledNB', Pipeline([('Scaler', StandardScaler()), ('NB', GaussianNB())])))
pipelines.append(('ScaledSVM', Pipeline([('Scaler', StandardScaler()), ('SVM', SVC())])))

from sklearn.metrics import accuracy_score
results = []
names = []
scoring = 'accuracy'
num_folds = 10
seed = 7

for name, model in pipelines:
    kfold = KFold(n_splits = num_folds, random_state = seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv = kfold, scoring = 'accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)    

# compare algorithms
fig = plt.figure()
fig.suptitle('Scaled Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# LR and SVM gives best results
# ScaledLR: 0.812007 (0.051506)
# ScaledLDA: 0.810292 (0.060409)
# ScaledKNN: 0.791142 (0.055939)
# ScaledCART: 0.739785 (0.035478)
# ScaledNB: 0.699821 (0.062842)
# ScaledSVM: 0.818356 (0.044654)

# tune scaled LR
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
penalty_value = ['l1', 'l2']
c_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0, 10, 20]
param_grid = dict(penalty = penalty_value, C = c_values)

model = LogisticRegression()

kfold = KFold(n_splits = num_folds, random_state = seed)
grid = GridSearchCV(estimator = model, param_grid = param_grid, scoring = 'accuracy', cv = kfold)
grid_result = grid.fit(rescaledX, Y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("results from scaled CART are %f (%f) with: %r" % (mean, stdev, params))
# Best: 0.826645 using {'C': 0.1, 'penalty': 'l2'}

# tune scaled SVM
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
c_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0]
kernel_values = ['linear', 'poly', 'rbf', 'sigmoid']
param_grid = dict(C=c_values, kernel= kernel_values)

model = SVC()

kfold = KFold(n_splits = num_folds, random_state = seed)
grid = GridSearchCV(estimator = model, param_grid = param_grid, scoring = 'accuracy', cv = kfold)
grid_result = grid.fit(rescaledX, Y_train)
print("Best: %f using %s" %(grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("results from scaled SVM are %f (%f) with: %r" % (mean, stdev, param))
# Best: 0.826645 using {'kernel': 'rbf', 'C': 1.5}
    
# ensemble methods
ensembles = []
ensembles.append(('AB', AdaBoostClassifier()))
ensembles.append(('GBM', GradientBoostingClassifier()))
ensembles.append(('RF', RandomForestClassifier()))
ensembles.append(('ET', ExtraTreesClassifier()))

results = []
names = []

for name, model in ensembles:
    kfold = KFold(n_splits = num_folds, random_state = seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring = scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" %(name, cv_results.mean(), cv_results.std())
    print(msg)

# compare algorithms
fig = plt.figure()
fig.suptitle('Ensemble Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
# results of v4
#AB: 0.779852 (0.057075)
#GBM: 0.802227 (0.060652)
#RF: 0.808858 (0.045576)
#ET: 0.807245 (0.038523)

# tuned GMB
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)

loss_values = ['deviance', 'exponential']
n_estimators_value = [100, 200, 300, 400, 500]
min_samples_split_value = [1, 2]
max_features_value = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
param_grid = dict(loss = loss_values, n_estimators = n_estimators_value,
                  min_samples_split = min_samples_split_value, max_features = max_features_value)

model = GradientBoostingClassifier()

kfold = KFold(n_splits = num_folds, random_state = seed)
grid = GridSearchCV(estimator = model, param_grid = param_grid, scoring = 'accuracy', cv = kfold)
grid_result = grid.fit(rescaledX, Y_train)
print("Best: %f using %s" %(grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("results from scaled SVM are %f (%f) with: %r" % (mean, stdev, param))
# Best: 0.837881 using {'max_features': 2, 'n_estimators': 200, 'min_samples_split': 2, 'loss': 'deviance'}
   
# tuned Random Forest
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
parameter_grid = {
                 'max_depth' : [4,5,6,7,8],
                 'n_estimators': [50, 100, 150, 200, 210, 240, 250, 300, 400, 500],
                 'criterion': ['gini','entropy']
                 }

model = RandomForestClassifier(max_features='sqrt')

grid = GridSearchCV(model, param_grid=parameter_grid, cv=kfold)
grid_result = grid.fit(rescaledX, Y_train)
print("Best: %f using %s" %(grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("results from scaled SVM are %f (%f) with: %r" % (mean, stdev, param))
# Best: 0.834671 using {'n_estimators': 150, 'max_depth': 7, 'criterion': 'gini'}

############################# step 10, finalize models ########################
# finalize logistic regression
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)

scaler = StandardScaler().fit(X_validation)
rescaledX_validation = scaler.transform(X_validation)

model = LogisticRegression(penalty = 'l2', C = 0.3)

model.fit(rescaledX, Y_train)
y_pred = model.predict(rescaledX_validation)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(Y_validation, predictions)
print("Accuracy: %.2f%%"%(accuracy*100.0)) 
# v4 84.70% v5 85.45%


# finalize model_SVC
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)

scaler = StandardScaler().fit(X_validation)
rescaledX_validation = scaler.transform(X_validation)

model = SVC(C = 1.3, kernel = 'rbf')

model.fit(rescaledX, Y_train)
y_pred = model.predict(rescaledX_validation)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(Y_validation, predictions)
print("Accuracy: %.2f%%"%(accuracy*100.0)) 
# Accuracy = 83.96%

# finalize model_GBM
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)

scaler = StandardScaler().fit(X_validation)
rescaledX_validation = scaler.transform(X_validation)

model = GradientBoostingClassifier(loss = 'deviance', n_estimators = 100, 
                                   min_samples_split = 1, max_features = 4)

model.fit(rescaledX, Y_train)
y_pred = model.predict(rescaledX_validation)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(Y_validation, predictions)
print("Accuracy: %.2f%%"%(accuracy*100.0)) 
# Accuracy = 84.33%

# finalize model_random Forest
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)

scaler = StandardScaler().fit(X_validation)
rescaledX_validation = scaler.transform(X_validation)

model = RandomForestClassifier(criterion = 'entropy', n_estimators = 100, max_depth = 6)

model.fit(rescaledX, Y_train)
y_pred = model.predict(rescaledX_validation)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(Y_validation, predictions)
print("Accuracy: %.2f%%"%(accuracy*100.0)) 
# Accuracy = 84.33%

#################### step 11, finalize model ##################################
# estimate accuracy on validation dataset
rescaledtest = scaler.transform(test_new)
predictions = model.predict(rescaledtest).astype(int)
ids = test['PassengerId'].values

predictions_file = open("v5LR_dec2016.csv", "w", newline='')
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, predictions))
predictions_file.close()
print('Done.')

