import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics, cross_validation
from sklearn.metrics import roc_curve, auc
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


TestModels = pd.DataFrame()
tmp1 = {}
tmp2 = {}
#LEARNING_OPTION_2 --- LogisticRegression && LEARNING_OPTION_3 --- LogisticRegression with gradient desent

model_lr = LogisticRegression()
cls = linear_model.SGDClassifier(loss='log')
tmp1['Model'] = 'LogisticRegression1'
tmp2['Model'] =  'SGDClassifier1'
tmp1['Accurancy'] = 0
tmp2['Accurancy'] = 0

#StratifiedShuffleSplit
for train_index, test_index in cross_validation.StratifiedShuffleSplit(target, n_iter=100, test_size=0.25):
    y_train, y_test = target[train_index], target[test_index]
    X_train, X_test = train.iloc[train_index], train.iloc[test_index]

    #training
    model_lr.fit(X_train, y_train.reshape(len(y_train)))
    cls.fit(X_train, y_train.reshape(len(y_train)))
    #print(model_lr2.coef_)

    # proving
    predicted_lr = model_lr.predict(X_test)
    predicted_cls = cls.predict(X_test)
    tmp1['Accurancy'] += metrics.accuracy_score(y_test, predicted_lr)
    tmp2['Accurancy'] += metrics.accuracy_score(y_test, predicted_cls)
    print(metrics.accuracy_score(y_test, predicted_lr), metrics.accuracy_score(y_test, predicted_cls))

tmp1['Accurancy'] /= 100
tmp2['Accurancy'] /= 100
TestModels = TestModels.append([tmp1])
TestModels = TestModels.append([tmp2])



#Same models(LogisticRegression && SGDClassifier) other validation
tmp3 = {}
tmp4 = {}

model_lr2 = LogisticRegression()
cls2 = linear_model.SGDClassifier(loss='log')
tmp3['Model'] = 'LogisticRegression2'
tmp4['Model'] =  'SGDClassifier2'
tmp3['Accurancy'] = 0
tmp4['Accurancy'] = 0
iter = 0
#Kfolds
for train_index, test_index in cross_validation.KFold(train.shape[0], n_folds=5):
    iter += 1
    y_train, y_test = target[train_index], target[test_index]
    X_train, X_test = train.iloc[train_index], train.iloc[test_index]

    #training
    model_lr2.fit(X_train, y_train.reshape(len(y_train)))
    cls2.fit(X_train, y_train.reshape(len(y_train)))
    #print(model_lr2.coef_)

    # proving
    predicted_lr2 = model_lr2.predict(X_test)
    predicted_cls2 = cls2.predict(X_test)
    tmp3['Accurancy'] += metrics.accuracy_score(y_test, predicted_lr2)
    tmp4['Accurancy'] += metrics.accuracy_score(y_test, predicted_cls2)
    print(metrics.accuracy_score(y_test, predicted_lr2), metrics.accuracy_score(y_test, predicted_cls2))

tmp3['Accurancy'] /= iter
tmp4['Accurancy'] /= iter
TestModels = TestModels.append([tmp3])
TestModels = TestModels.append([tmp4])


#Same models(LogisticRegression && SGDClassifier) other validation
tmp5 = {}
tmp6 = {}

model_lr3 = LogisticRegression()
cls3 = linear_model.SGDClassifier(loss='log')
tmp5['Model'] = 'LogisticRegression3'
tmp6['Model'] =  'SGDClassifier3'
tmp5['Accurancy'] = 0
tmp6['Accurancy'] = 0
iter = 0
#StratifiedKfolds
for train_index, test_index in cross_validation.StratifiedKFold(target, n_folds=5, shuffle=True, random_state=0):
    iter += 1
    y_train, y_test = target[train_index], target[test_index]
    X_train, X_test = train.iloc[train_index], train.iloc[test_index]

    #training
    model_lr3.fit(X_train, y_train.reshape(len(y_train)))
    cls3.fit(X_train, y_train.reshape(len(y_train)))
    #print(model_lr2.coef_)

    # proving
    predicted_lr3 = model_lr3.predict(X_test)
    predicted_cls3 = cls3.predict(X_test)
    tmp5['Accurancy'] += metrics.accuracy_score(y_test, predicted_lr3)
    tmp6['Accurancy'] += metrics.accuracy_score(y_test, predicted_cls3)
    print(metrics.accuracy_score(y_test, predicted_lr3), metrics.accuracy_score(y_test, predicted_cls3))
tmp5['Accurancy'] /= iter
tmp6['Accurancy'] /= iter
TestModels = TestModels.append([tmp5])
TestModels = TestModels.append([tmp6])

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
scores = cross_validation.cross_val_score(model_lr2, train, target, cv = kfold)
itog_val['LogisticRegression2'] = scores.mean()
scores = cross_validation.cross_val_score(cls2, train, target, cv = kfold)
itog_val['SGDClassifier2'] = scores.mean()
scores = cross_validation.cross_val_score(model_lr3, train, target, cv = kfold)
itog_val['LogisticRegression3'] = scores.mean()
scores = cross_validation.cross_val_score(cls3, train, target, cv = kfold)
itog_val['SGDClassifier3'] = scores.mean()

pd.DataFrame.from_dict(data = itog_val, orient='index').plot(kind='bar', legend=False)
plt.show()

#ROC curve
ROCtrainTRN, ROCtestTRN, ROCtrainTRG, ROCtestTRG = cross_validation.train_test_split(train, target, test_size=0.25)

plt.figure(figsize=(8,6))
#SGDClassifier1

probas = cls.fit(ROCtrainTRN, ROCtrainTRG).predict_proba(ROCtestTRN)
false_positive_rate, true_positive_rate, _ = roc_curve(ROCtestTRG, probas[:, 1])
roc_auc  = auc(false_positive_rate, true_positive_rate)
pl.plot(false_positive_rate, true_positive_rate, label='%s ROC (area = %0.2f)' % ('SGDClassifier1', roc_auc))


#LogisticRegression1

probas = model_lr.fit(ROCtrainTRN, ROCtrainTRG).predict_proba(ROCtestTRN)
false_positive_rate, true_positive_rate, _ = roc_curve(ROCtestTRG, probas[:, 1])
roc_auc  = auc(false_positive_rate, true_positive_rate)
pl.plot(false_positive_rate, true_positive_rate, label='%s ROC (area = %0.2f)' % ('LogisticRegression1',roc_auc))


#SGDClassifier2

probas = cls2.fit(ROCtrainTRN, ROCtrainTRG).predict_proba(ROCtestTRN)
false_positive_rate, true_positive_rate, _ = roc_curve(ROCtestTRG, probas[:, 1])
roc_auc  = auc(false_positive_rate, true_positive_rate)
pl.plot(false_positive_rate, true_positive_rate, label='%s ROC (area = %0.2f)' % ('SGDClassifier2', roc_auc))


#LogisticRegression2

probas = model_lr2.fit(ROCtrainTRN, ROCtrainTRG).predict_proba(ROCtestTRN)
false_positive_rate, true_positive_rate, _ = roc_curve(ROCtestTRG, probas[:, 1])
roc_auc  = auc(false_positive_rate, true_positive_rate)
pl.plot(false_positive_rate, true_positive_rate, label='%s ROC (area = %0.2f)' % ('LogisticRegression2',roc_auc))

#SGDClassifier3

probas = cls3.fit(ROCtrainTRN, ROCtrainTRG).predict_proba(ROCtestTRN)
false_positive_rate, true_positive_rate, _ = roc_curve(ROCtestTRG, probas[:, 1])
roc_auc  = auc(false_positive_rate, true_positive_rate)
pl.plot(false_positive_rate, true_positive_rate, label='%s ROC (area = %0.2f)' % ('SGDClassifier3', roc_auc))


#LogisticRegression3

probas = model_lr3.fit(ROCtrainTRN, ROCtrainTRG).predict_proba(ROCtestTRN)
false_positive_rate, true_positive_rate, _ = roc_curve(ROCtestTRG, probas[:, 1])
roc_auc  = auc(false_positive_rate, true_positive_rate)
pl.plot(false_positive_rate, true_positive_rate, label='%s ROC (area = %0.2f)' % ('LogisticRegression3',roc_auc))


pl.plot([0, 1], [0, 1], 'k--')
pl.xlim([-0.1, 1.1])
pl.ylim([-0.1, 1.1])
pl.xlabel('False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('ROC curve')
pl.legend(loc=0, fontsize='small')
pl.show()




