import pandas
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
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


def DataProcessing(path):
    file = pandas.read_excel(path) #read file
    file = file[file.Key.notnull()] #reduce empty rows
    #reduce useless columns and rename others
    file = file.drop(['Summary','Status','Resolution','Created', 'Resolved', 'Images'],axis=1)
    file = file.rename(columns={'Steps to Reproduce': 'Steps_to_Reproduce',
                                          'Number of attachments': 'Number_of_attachments',
                                          'USEFUL Description': 'USEFUL_Description',
                                          'Component/s': 'Component'})
    #converting to boolean type
    file['Component'] = (file.Component.notnull()).astype(bool)
    file['Environment'] = (file.Environment.notnull()).astype(bool)
    file['Description'] = (file.Description.notnull()).astype(bool)
    file['Steps_to_Reproduce'] = (file.Steps_to_Reproduce.notnull()).astype(bool)
    #reduce strings
    label = LabelEncoder()
    dicts = {}

    label.fit(file.Priority.drop_duplicates())
    dicts['Priority'] = list(label.classes_)
    file.Priority = label.transform(file.Priority)

    return file

def LearnModel(model, file, folds=5):

    # Division into groups(test and train)
    target = file.USEFUL_Description
    train = file.drop(['USEFUL_Description', 'Key'], axis=1)

    # Learning
    for train_index, test_index in cross_validation.StratifiedKFold(target, n_folds=folds, shuffle=True,
                                                                        random_state=0):
        y_train, y_test = target[train_index], target[test_index]
        X_train, X_test = train.iloc[train_index], train.iloc[test_index]

        # training
        model.fit(X_train, y_train.reshape(len(y_train)))
        # proving
        predicted = model.predict(X_test)
        print(metrics.classification_report(y_test, predicted))

def TestByAccurancy(model, TestModels, TestFile):

    test_target = TestFile.USEFUL_Description
    test = TestFile.drop(['USEFUL_Description', 'Key'], axis=1)

    tmp = {}
    name = str(model)
    tmp['Model'] = name[:name.index('(')]
    predicted = model.predict(test)
    tmp['Accurancy'] = metrics.accuracy_score(test_target, predicted)

    TestModels = TestModels.append([tmp])
    return TestModels


def TestByCrossValidation(model, value, TestFile, fold=5):

    test_target = TestFile.USEFUL_Description
    test = TestFile.drop(['USEFUL_Description', 'Key'], axis=1)

    scores = cross_validation.cross_val_score(model, test, test_target, cv=fold)
    name = str(model)
    value[name[:name.index('(')]] = scores.mean()
    return value

def ROCAnalysis(model, TestFile, pl):

    test_target = TestFile.USEFUL_Description
    test = TestFile.drop(['USEFUL_Description', 'Key'], axis=1)

    probas = model.predict_proba(test)
    false_positive_rate, true_positive_rate, _ = roc_curve(test_target, probas[:, 1])
    roc_auc = auc(false_positive_rate, true_positive_rate)
    name = str(model)
    pl.plot(false_positive_rate, true_positive_rate, label='%s ROC (area = %0.2f)' % (name[:name.index('(')], roc_auc))


InputFile = DataProcessing("/home/natalia/example1.xlsx")
TestFile = DataProcessing("/home/natalia/PycharmProjects/Logit/test.xlsx")

RegressionModel = LogisticRegression()
LearnModel(RegressionModel, InputFile)

SGDModel = SGDClassifier(loss='log')
LearnModel(SGDModel, InputFile)

BayesModel = GaussianNB()
LearnModel(BayesModel, InputFile)

#Analizing the differences
#...
#Accurancy on Test sample

TestModels = pandas.DataFrame()
TestModels = TestByAccurancy(RegressionModel, TestModels, TestFile)
TestModels = TestByAccurancy(SGDModel, TestModels, TestFile)
TestModels = TestByAccurancy(BayesModel, TestModels, TestFile)

TestModels.set_index('Model', inplace=True)
TestModels.plot(kind='bar', legend= False)
plt.show()


#Cross validation on test sample
ResultValue  = {}

ResultValue = TestByCrossValidation(RegressionModel, ResultValue, TestFile)
ResultValue = TestByCrossValidation(SGDModel, ResultValue, TestFile)
ResultValue = TestByCrossValidation(BayesModel, ResultValue, TestFile)


pandas.DataFrame.from_dict(data = ResultValue, orient='index').plot(kind='bar', legend=False)
plt.show()

#ROC curve
plt.figure(figsize=(8,6))

ROCAnalysis(RegressionModel, TestFile, pl)
ROCAnalysis(SGDModel, TestFile, pl)
ROCAnalysis(BayesModel, TestFile, pl)

pl.plot([0, 1], [0, 1], 'k--')
pl.xlim([-0.1, 1.1])
pl.ylim([-0.1, 1.1])
pl.xlabel('False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('ROC curve')
pl.legend(loc=0, fontsize='small')
pl.show()

print(RegressionModel.coef_)
print(SGDModel.coef_)


