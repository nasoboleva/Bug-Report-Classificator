import pandas
import numpy
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import metrics, cross_validation
from sklearn.metrics import roc_curve, auc

#sorting data by different parametrs
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
    file = file.drop(['Summary','Status','Resolution','Created', 'Resolved', 'Images', 'Number of comments'],axis=1)
    file = file.rename(columns={'Steps to Reproduce': 'Steps_to_Reproduce',
                                          'Number of attachments': 'Number_of_attachments',
                                          'USEFUL Description': 'USEFUL_Description',
                                          'Component/s': 'Component'})
    #converting some data to boolean type
    file['Component'] = (file.Component.notnull()).astype(bool)
    file['Environment'] = (file.Environment.notnull()).astype(bool)
    file['Description'] = (file.Description.notnull()).astype(bool)
    file['Steps_to_Reproduce'] = (file.Steps_to_Reproduce.notnull()).astype(bool)
    print(file.head())

    # reduce strings columns
    label = LabelEncoder()
    dicts = {}

    label.fit(file.Priority.drop_duplicates())
    dicts['Priority'] = list(label.classes_)
    file.Priority = label.transform(file.Priority)

    #normalization
    scaler = StandardScaler()
    file['Number_of_attachments'] = scaler.fit_transform(file['Number_of_attachments'])


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


def ExamCoefficients(model, InputFile):
    InputFile = InputFile.drop(['USEFUL_Description', 'Key'], axis=1)
    file = pandas.DataFrame(list(zip(InputFile.columns, numpy.transpose(model.coef_))))
    return file

def GetResult(model, test, name):

    result = pandas.DataFrame(test.Key)
    test = test.drop(['USEFUL_Description', 'Key'], axis=1)
    result.insert(1, 'Useful', model.predict(test))
    result.to_excel(name, index=False)


# reading and processing data
InputFile = DataProcessing("/home/natalia/example1.xlsx") #for the input data
TestFile = DataProcessing("/home/natalia/PycharmProjects/Logit/test.xlsx") # and test data

print(TestFile.head())

# show some statictics
statistics(InputFile)
statistics(TestFile)

# learning models
RegressionModel = LogisticRegression() #sklearn LogisticRegression
LearnModel(RegressionModel, InputFile)

SGDModel = SGDClassifier(loss='log') #sklearn Classifier with stohaick gradient descent
LearnModel(SGDModel, InputFile)

BayesModel = GaussianNB() #sklearn Naive Bayes
LearnModel(BayesModel, InputFile)

#showing coefficients learnt by models
print (ExamCoefficients(RegressionModel, InputFile))
print (ExamCoefficients(SGDModel, InputFile))

#Analizing the differences

#Accurancy, using test data

TestModels = pandas.DataFrame()
TestModels = TestByAccurancy(RegressionModel, TestModels, TestFile)
TestModels = TestByAccurancy(SGDModel, TestModels, TestFile)
TestModels = TestByAccurancy(BayesModel, TestModels, TestFile)

TestModels.set_index('Model', inplace=True) #drawing bar chart by accurancy
TestModels.plot(kind='bar', legend= False)
plt.show()


#Cross validation, using test data
ResultValue  = {}

ResultValue = TestByCrossValidation(RegressionModel, ResultValue, TestFile)
ResultValue = TestByCrossValidation(SGDModel, ResultValue, TestFile)
ResultValue = TestByCrossValidation(BayesModel, ResultValue, TestFile)

#drawing bar chart on validation
pandas.DataFrame.from_dict(data = ResultValue, orient='index').plot(kind='bar', legend=False)
plt.show()

#ROC curve, using the test data
plt.figure(figsize=(8,6))

ROCAnalysis(RegressionModel, TestFile, plt)
ROCAnalysis(SGDModel, TestFile, plt)
ROCAnalysis(BayesModel, TestFile, plt)

#drawing ROC-curve

plt.plot([0, 1], [0, 1], 'k--')
plt.axis([-0.1, 1.1, -0.1, 1.1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend(loc=0, fontsize='small')
plt.show()

GetResult(RegressionModel, TestFile, '/home/natalia/PycharmProjects/Logit/predict_r.xlsx')
GetResult(SGDModel, TestFile, '/home/natalia/PycharmProjects/Logit/predict_cls.xlsx')
GetResult(BayesModel, TestFile, '/home/natalia/PycharmProjects/Logit/predict_bayes.xlsx')