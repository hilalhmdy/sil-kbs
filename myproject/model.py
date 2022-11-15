import pandas
import numpy
import pickle

from os.path import exists

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

MODEL_FILE_PATH = 'DIABETES_AI'
DATA_FILE_PATH ='diabetes.csv'

def predict(data):
    to_predict = numpy.array(data).reshape(1, 21)
    loaded_model = pickle.load(open(MODEL_FILE_PATH, "rb"))
    result = loaded_model.predict(to_predict)
    return result[0]

def init():
    isModelExist = exists(MODEL_FILE_PATH)
    if (isModelExist):
        print(">> Model is ready to action:", MODEL_FILE_PATH)
        print()
        return

    print("Begin to train DIABETES KBS Model")
    print()
    df = pandas.read_csv(DATA_FILE_PATH)
    df = df.drop(['id'], axis=1)
    df.head()
    cols = df.columns
    print("Data Columns:", cols)
    print()

    category_col = ['race', 'gender', 'age', 'admission_source_id', 'medical_specialty', 'primary_diagnosis','max_glu_serum', 'A1Cresult', 'insulin', 'change', 'diabetesMed', 'readmit_30_days']

    labelEncoder = preprocessing.LabelEncoder()

    mapping_dict = {}
    for col in category_col:
        df[col] = labelEncoder.fit_transform(df[col])

        le_name_mapping = dict(zip(labelEncoder.classes_,
                                  labelEncoder.transform(labelEncoder.classes_)))

        mapping_dict[col] = le_name_mapping
    print("Category Mapping:", mapping_dict)
    print()

    X = df.values[:, 0:21]
    Y = df.values[:, 21]
    Y=Y.astype('int')

    X_train, X_test, y_train, y_test = train_test_split(
              X, Y, test_size = 0.3, random_state = 100)
    dt_clf_gini = DecisionTreeClassifier(criterion = "gini",
                                        random_state = 100,
                                        max_depth = 5,
                                        min_samples_leaf = 5)

    dt_clf_gini.fit(X_train, y_train)
    y_pred_gini = dt_clf_gini.predict(X_test)

    print ("Decision Tree using Gini Index\nAccuracy is ",
                accuracy_score(y_test, y_pred_gini)*100 )
    print()

    print("Dumping model...")
    print()
    pickle.dump(dt_clf_gini, open(MODEL_FILE_PATH, 'wb'))
    print("Done. Model saved to:", MODEL_FILE_PATH)
    print()