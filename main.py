import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn import metrics, cross_validation

def scorer(estimator, X, Y):
    return metric(Y, estimator.predict_proba(X)[:, 1])


def accurancy(y_test, y_pred):
    return (y_test == y_pred).sum() / y_test.size if y_test.size else 1.0

def statistics(InputFile):
    print(InputFile.groupby('Component').mean())
    print(InputFile.groupby('Environment').mean())
    print(InputFile.groupby('Description').mean())
    print(InputFile.groupby('Steps_to_Reproduce').mean())
    print(InputFile.groupby('Priority').mean())
    print(InputFile.groupby('Number_of_attachments').mean())

# Cool bar chart. Just in case
def bar_chart(InputFile, Name, x, y):
    InputFile.Name.hist()
    plt.title('Name')
    plt.xlabel('x')
    plt.ylabel('y')



InputFile = pd.read_excel("/home/natalia/example1.xlsx")
result = pd.DataFrame(InputFile.Key)
InputFile = InputFile.drop(['Summary','Status','Resolution','Created', 'Resolved', 'Images'],axis=1)
InputFile = InputFile.rename(columns={'Steps to Reproduce': 'Steps_to_Reproduce',
                                      'Number of attachments': 'Number_of_attachments',
                                      'USEFUL Description': 'USEFUL_Description',
                                      'Component/s' : 'Component'})

InputFile['Component'] = (InputFile.Component.notnull()).astype(bool)
InputFile['Environment'] = (InputFile.Environment.notnull()).astype(bool)
InputFile['Description'] = (InputFile.Description.notnull()).astype(bool)
InputFile['Steps_to_Reproduce'] = (InputFile.Steps_to_Reproduce.notnull()).astype(bool)


statistics(InputFile)
#Reduce str

label = LabelEncoder()
dicts = {}

label.fit(InputFile.Priority.drop_duplicates())
dicts['Priority'] = list(label.classes_)
InputFile.Priority = label.transform(InputFile.Priority)


#Division into groups(test and train)
target = InputFile.USEFUL_Description
train = InputFile.drop(['USEFUL_Description', 'Key'], axis=1)
kfold = 12 #количество подвыборок для валидации -- лишнее




#LEARNING_OPTION_2

#X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=0.25)
model_lr2 = LogisticRegression()
for train_index, test_index in cross_validation.StratifiedShuffleSplit(target, n_iter=20, test_size=0.25):
    y_train, y_test = target[train_index], target[test_index]
    X_train, X_test = train.iloc[train_index], train.iloc[test_index]
    model_lr2.fit(X_train, y_train.reshape(len(y_train)))
    print(model_lr2.coef_)

#metrics and accuracy - still not sure about that
metric = metrics.roc_auc_score
Y_pred = model_lr2.predict_proba(X_test)[:, 1]
score = metric(y_test, Y_pred)
print('Score: ', score)

predicted = model_lr2.predict(X_test)
print(predicted)

print('accurancy = {accurancy}'.format(accurancy = accurancy(y_test, predicted)))

probs = model_lr2.predict_proba(X_test)
print (probs)
print(model_lr2.coef_)

#Some more metrics

print (metrics.accuracy_score(y_test, predicted))
print (metrics.confusion_matrix(y_test, predicted))
print (metrics.classification_report(y_test, predicted))

scores = cross_val_score(LogisticRegression(), train, target , scoring='accuracy', cv=50)
print (scores)
print (scores.mean())


scores = cross_val_score(model_lr, train, target, cv = kfold)
itog_val = scores.mean()
print(itog_val)

label.fit(InputFile.Key.drop_duplicates())
dicts['Key'] = list(label.classes_)
InputFile.Key = label.transform(InputFile.Key)



#LEARNING_OPTION_3

cls = linear_model.SGDClassifier(loss='log')
cls.fit(X_train, y_train)


print(cls.coef_)

#again some metrics

metric = metrics.roc_auc_score
Y_pred = cls.predict_proba(X_test)[:, 1]
score = metric(y_test, Y_pred)
print('Score: ', score)


#and now with cross-validation
for train_index, test_index in cross_validation.StratifiedShuffleSplit(target, n_iter=200, test_size=0.25):
    y_train, y_test = target[train_index], target[test_index]
    X_train, X_test = train.iloc[train_index], train.iloc[test_index]
    cls.fit(X_train, y_train.reshape(len(y_train)))

    print(cls.coef_)

metric = metrics.roc_auc_score
Y_pred = cls.predict_proba(X_test)[:, 1]
score = metric(y_test, Y_pred)
print('Score: ', score)








