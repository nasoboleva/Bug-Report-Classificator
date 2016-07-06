import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics, cross_validation
from sklearn.metrics import roc_curve, auc
from sklearn.naive_bayes import GaussianNB
import pylab as pl


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

#Reduce str
label = LabelEncoder()
dicts = {}

label.fit(InputFile.Priority.drop_duplicates())
dicts['Priority'] = list(label.classes_)
InputFile.Priority = label.transform(InputFile.Priority)


#Division into groups(test and train)
target = InputFile.USEFUL_Description
train = InputFile.drop(['USEFUL_Description', 'Key'], axis=1)

#LEARNING_OPTION_2 --- LogisticRegression && LEARNING_OPTION_3 --- LogisticRegression with gradient desent

model_lr = LogisticRegression()
cls = linear_model.SGDClassifier(loss='log')

#StratifiedShuffleSplit
for train_index, test_index in cross_validation.StratifiedShuffleSplit(target, n_iter=10, test_size=0.25):
    y_train, y_test = target[train_index], target[test_index]
    X_train, X_test = train.iloc[train_index], train.iloc[test_index]

    #training
    model_lr.fit(X_train, y_train.reshape(len(y_train)))
    cls.fit(X_train, y_train.reshape(len(y_train)))
    #print(model_lr2.coef_)

    # proving
    predicted_lr = model_lr.predict(X_test)
    predicted_cls = cls.predict(X_test)
    print(metrics.classification_report(y_test, predicted_lr))
    print(metrics.confusion_matrix(y_test, predicted_lr))
    #print(metrics.accuracy_score(y_test, predicted_lr), metrics.accuracy_score(y_test, predicted_cls))


#BAYES
bayes = GaussianNB()

#Kfolds
for train_index, test_index in cross_validation.KFold(train.shape[0], n_folds=5):
    y_train, y_test = target[train_index], target[test_index]
    X_train, X_test = train.iloc[train_index], train.iloc[test_index]

    #training
    bayes.fit(X_train, y_train.reshape(len(y_train)))
    #print(model_lr2.coef_)

    # proving
    predicted_b = bayes.predict(X_test)
    #print(metrics.accuracy_score(y_test, predicted_lr2), metrics.accuracy_score(y_test, predicted_cls2))


#Same models(LogisticRegression && SGDClassifier) other validation
model_lr3 = LogisticRegression()
cls3 = linear_model.SGDClassifier(loss='log')

#StratifiedKfolds
for train_index, test_index in cross_validation.StratifiedKFold(target, n_folds=5, shuffle=True, random_state=0):
    y_train, y_test = target[train_index], target[test_index]
    X_train, X_test = train.iloc[train_index], train.iloc[test_index]

    #training
    model_lr3.fit(X_train, y_train.reshape(len(y_train)))
    cls3.fit(X_train, y_train.reshape(len(y_train)))
    #print(model_lr2.coef_)

    # proving
    predicted_lr3 = model_lr3.predict(X_test)
    predicted_cls3 = cls3.predict(X_test)
    print(metrics.classification_report(y_test, predicted_lr3))
    print(metrics.confusion_matrix(y_test, predicted_lr3))
    #print(metrics.accuracy_score(y_test, predicted_lr3), metrics.accuracy_score(y_test, predicted_cls3))


#Accurancy
AX_train, AX_test, Ay_train, Ay_test = cross_validation.train_test_split(train, target, test_size=0.25)

TestModels = pd.DataFrame()
tmp = {}


#LogisticRegression
tmp['Model'] = 'LogisticRegression'
model_lr.fit(AX_train, Ay_train)
predicted = model_lr.predict(AX_test)
tmp['Accurancy'] = metrics.accuracy_score(Ay_test, predicted)

TestModels = TestModels.append([tmp])

#SGDClassifier
tmp['Model'] = 'SGDClassifier'
cls.fit(AX_train, Ay_train)
predicted = cls.predict(AX_test)
tmp['Accurancy'] = metrics.accuracy_score(Ay_test, predicted)

TestModels = TestModels.append([tmp])

#Bayes
tmp['Model'] = 'Naive Bayes'
bayes.fit(AX_train, Ay_train)
predicted = bayes.predict(AX_test)
tmp['Accurancy'] = metrics.accuracy_score(Ay_test, predicted)

TestModels = TestModels.append([tmp])

TestModels.set_index('Model', inplace=True)
TestModels.plot(kind='bar', legend= False)
plt.show()


#Cross validation
kfold = 5 #количество подвыборок для валидации
itog_val = {}

scores = cross_validation.cross_val_score(model_lr, train, target, cv = kfold)
itog_val['LogisticRegression1'] = scores.mean()
scores = cross_validation.cross_val_score(cls, train, target, cv = kfold)
itog_val['SGDClassifier1'] = scores.mean()
scores = cross_validation.cross_val_score(model_lr3, train, target, cv = kfold)
itog_val['LogisticRegression3'] = scores.mean()
scores = cross_validation.cross_val_score(cls3, train, target, cv = kfold)
itog_val['SGDClassifier3'] = scores.mean()
scores = cross_validation.cross_val_score(bayes, train, target, cv = kfold)
itog_val['Naive Bayes'] = scores.mean()

pd.DataFrame.from_dict(data = itog_val, orient='index').plot(kind='bar', legend=False)
plt.show()

#ROC curve
ROCtrainTRN, ROCtestTRN, ROCtrainTRG, ROCtestTRG = cross_validation.train_test_split(train, target, test_size=0.25)

plt.figure(figsize=(8,6))

#LogisticRegression1

probas = model_lr.fit(ROCtrainTRN, ROCtrainTRG).predict_proba(ROCtestTRN)
false_positive_rate, true_positive_rate, _ = roc_curve(ROCtestTRG, probas[:, 1])
roc_auc  = auc(false_positive_rate, true_positive_rate)
pl.plot(false_positive_rate, true_positive_rate, label='%s ROC (area = %0.2f)' % ('LogisticRegression',roc_auc))

#Naive Bayes

probas = bayes.fit(ROCtrainTRN, ROCtrainTRG).predict_proba(ROCtestTRN)
false_positive_rate, true_positive_rate, _ = roc_curve(ROCtestTRG, probas[:, 1])
roc_auc  = auc(false_positive_rate, true_positive_rate)
pl.plot(false_positive_rate, true_positive_rate, label='%s ROC (area = %0.2f)' % ('Naive Bayes',roc_auc))


#SGDClassifier1

probas = cls.fit(ROCtrainTRN, ROCtrainTRG).predict_proba(ROCtestTRN)
false_positive_rate, true_positive_rate, _ = roc_curve(ROCtestTRG, probas[:, 1])
roc_auc  = auc(false_positive_rate, true_positive_rate)
pl.plot(false_positive_rate, true_positive_rate, label='%s ROC (area = %0.2f)' % ('SGDClassifier1', roc_auc))


#SGDClassifier3

probas = cls3.fit(ROCtrainTRN, ROCtrainTRG).predict_proba(ROCtestTRN)
false_positive_rate, true_positive_rate, _ = roc_curve(ROCtestTRG, probas[:, 1])
roc_auc  = auc(false_positive_rate, true_positive_rate)
pl.plot(false_positive_rate, true_positive_rate, label='%s ROC (area = %0.2f)' % ('SGDClassifier3', roc_auc))



pl.plot([0, 1], [0, 1], 'k--')
pl.xlim([-0.1, 1.1])
pl.ylim([-0.1, 1.1])
pl.xlabel('False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('ROC curve')
pl.legend(loc=0, fontsize='small')
pl.show()




