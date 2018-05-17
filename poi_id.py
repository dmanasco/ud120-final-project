#!/usr/bin/python
import pandas as pd
import sys
import pickle
import warnings
warnings.filterwarnings("ignore")
sys.path.append("../tools/")


from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedShuffleSplit
from numpy import mean
from sklearn import cross_validation
from sklearn.metrics import accuracy_score, precision_score, recall_score

### Task 1: Select what features you'll use.

### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
target_label = 'poi'
email_features_list = [
    'from_messages',
    'from_poi_to_this_person',
    'from_this_person_to_poi',
    'shared_receipt_with_poi',
    'to_messages',
    ]
financial_features_list = [
    'bonus',
    'deferral_payments',
    'deferred_income',
    'director_fees',
    'exercised_stock_options',
    'expenses',
    'loan_advances',
    'long_term_incentive',
    'other',
    'restricted_stock',
    'restricted_stock_deferred',
    'salary',
    'total_payments',
    'total_stock_value',
]

features_list = [target_label] + financial_features_list + email_features_list


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

print('# Exploratory Data Analysis #')
data_dict.keys()
print('Total number of data points: %d' % len(data_dict.keys()))
num_poi = 0
for name in data_dict.keys():
    if data_dict[name]['poi'] == True:
        num_poi += 1
print('Number of Persons of Interest: %d' % num_poi)
print('Number of people without Person of Interest label: %d' % (len(data_dict.keys()) - num_poi))

all_features = data_dict['ALLEN PHILLIP K'].keys()
print('Each person has %d features available' %  len(all_features))
missing_values = {}
for feature in all_features:
    missing_values[feature] = 0
for person in data_dict.keys():
    records = 0
    for feature in all_features:
        if data_dict[person][feature] == 'NaN':
            missing_values[feature] += 1
        else:
            records += 1

print('Number of Missing Values for Each Feature:')
for feature in all_features:
    print("%s: %d" % (feature, missing_values[feature]))

### Identified two records that were not correct vs the PDF
def fix_records(data_dict):
    data_dict['BELFER ROBERT'] = {'bonus': 'NaN',
                              'deferral_payments': 'NaN',
                              'deferred_income': -102500,
                              'director_fees': 102500,
                              'email_address': 'NaN',
                              'exercised_stock_options': 'NaN',
                              'expenses': 3285,
                              'from_messages': 'NaN',
                              'from_poi_to_this_person': 'NaN',
                              'from_this_person_to_poi': 'NaN',
                              'loan_advances': 'NaN',
                              'long_term_incentive': 'NaN',
                              'other': 'NaN',
                              'poi': False,
                              'restricted_stock': -44093,
                              'restricted_stock_deferred': 44093,
                              'salary': 'NaN',
                              'shared_receipt_with_poi': 'NaN',
                              'to_messages': 'NaN',
                              'total_payments': 3285,
                              'total_stock_value': 'NaN'}

    data_dict['BHATNAGAR SANJAY'] = {'bonus': 'NaN',
                                 'deferral_payments': 'NaN',
                                 'deferred_income': 'NaN',
                                 'director_fees': 'NaN',
                                 'email_address': 'sanjay.bhatnagar@enron.com',
                                 'exercised_stock_options': 15456290,
                                 'expenses': 137864,
                                 'from_messages': 29,
                                 'from_poi_to_this_person': 0,
                                 'from_this_person_to_poi': 1,
                                 'loan_advances': 'NaN',
                                 'long_term_incentive': 'NaN',
                                 'other': 'NaN',
                                 'poi': False,
                                 'restricted_stock': 2604490,
                                 'restricted_stock_deferred': -2604490,
                                 'salary': 'NaN',
                                 'shared_receipt_with_poi': 463,
                                 'to_messages': 523,
                                 'total_payments': 137864,
                                 'total_stock_value': 15456290} 
    return data_dict


data_dict = fix_records(data_dict)
### Task 2: Remove outliers
def remove_outlier(dict_object, keys):
    """ removes list of outliers keys from dict object """
    for key in keys:
        dict_object.pop(key, 0)


outliers = ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK', 'LOCKHART EUGENE E']
remove_outlier(data_dict, outliers)

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

def compute_fraction(poi_messages, all_messages):
    """ return fraction of messages from/to that person to/from POI"""    
    if poi_messages == 'NaN' or all_messages == 'NaN':
        return 0.
    fraction = float(poi_messages) / float(all_messages)
    return fraction*100

for name in my_dataset:
    data_point = my_dataset[name]
    data_point["fraction_from_poi"] = compute_fraction(data_point["from_poi_to_this_person"], data_point["to_messages"])
    data_point["fraction_to_poi"] = compute_fraction(data_point["from_this_person_to_poi"], data_point["from_messages"])
    data_point["combined_poi_communications"] = data_point["fraction_from_poi"] + data_point["fraction_to_poi"]

my_feature_list = features_list+['to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi',
                                 'shared_receipt_with_poi', 'fraction_from_poi', 'fraction_to_poi', 'combined_poi_communications']
my_features_list = my_feature_list


def get_k_best(data_dict, features_list, k):
    """ runs scikit-learn's SelectKBest feature selection
        returns dict where keys=features, values=scores
    """
    data = featureFormat(data_dict, features_list)
    labels, features = targetFeatureSplit(data)

    k_best = SelectKBest(k=k)
    k_best.fit(features, labels)
    scores = k_best.scores_
    unsorted_pairs = zip(features_list[1:], scores)
    sorted_pairs = list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))
    k_best_features = dict(sorted_pairs[:k])
    print "{0} best features: ".format(k)
    return k_best_features

def min_evaluate_clf(clf, features, labels, num_iters=1000, test_size=0.3):
    accuracy = []
    precision = []
    recall = []
    first = True
    for trial in range(num_iters):
        features_train, features_test, labels_train, labels_test =\
            cross_validation.train_test_split(features, labels, test_size=test_size)
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        accuracy.append(accuracy_score(labels_test, predictions))
        precision.append(precision_score(labels_test, predictions))
        recall.append(recall_score(labels_test, predictions))

    print "accuracy: {} precision: {} recall:    {}".format(mean(accuracy),mean(precision),mean(recall))
    return mean(accuracy),mean(precision), mean(recall)

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
k = 1
a =[["K","accuracy","precision","recall"]]
while (k<15):
  best_features = get_k_best(my_dataset, my_features_list, k)
  my_feature_list = [target_label] + best_features.keys()
  data = featureFormat(my_dataset, my_feature_list, sort_keys = True)
  labels, features = targetFeatureSplit(data)
  scaler = preprocessing.MinMaxScaler()
  features = scaler.fit_transform(features)
  l1_clf = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(tol = 0.00001, C = 0.0002, penalty = 'l2', random_state = 42))])
  acc, pres, rec = min_evaluate_clf(l1_clf, features, labels)
  a += [[k, acc, pres, rec]]
  k += 1

bestk = 1
maxprecrec = 0
for x in a[1:]:
    if x[2] > 0.3 and x[3] > 0.3:
        if x[2]+x[3] > maxprecrec:
            maxprecrec = x[2]+x[3]
            bestk = x[0]

print "The Best K is {}".format(bestk)
print "Precision plus Recall = {}".format(maxprecrec)

best_features = get_k_best(my_dataset, my_feature_list, bestk)

print best_features

my_feature_list = [target_label] + best_features.keys()

print "{0} selected features: {1}\n".format(len(my_feature_list) - 1, my_feature_list[1:])


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, my_feature_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)



l_clf = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(tol = 0.001, C = 10**-8, penalty = 'l2', random_state = 42))])

from sklearn.cluster import KMeans
k_clf = KMeans(n_clusters=2, tol=0.001)

from sklearn.svm import SVC
s_clf = SVC(kernel='rbf', C=1000,gamma = 0.0001,random_state = 42,class_weight='balanced')

from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(max_depth = 2,max_features = 'sqrt',n_estimators = 10, random_state = 42)

from sklearn.neighbors import KNeighborsClassifier
nn_clf = KNeighborsClassifier(n_neighbors=6,algorithm='ball_tree', weights = 'distance', leaf_size = 5)

from sklearn.neural_network import MLPClassifier
mpl_clf = MLPClassifier(alpha=1)

from sklearn.ensemble import AdaBoostClassifier
qd_clf = AdaBoostClassifier()


def evaluate_clf(clf, features, labels, num_iters=1000, test_size=0.3):
    print clf
    accuracy = []
    precision = []
    recall = []
    first = True
    for trial in range(num_iters):
        features_train, features_test, labels_train, labels_test =\
            cross_validation.train_test_split(features, labels, test_size=test_size)
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        accuracy.append(accuracy_score(labels_test, predictions))
        precision.append(precision_score(labels_test, predictions))
        recall.append(recall_score(labels_test, predictions))
        if trial % 10 == 0:
            if first:
                sys.stdout.write('\nProcessing')
            sys.stdout.write('.')
            sys.stdout.flush()
            first = False

    print "done.\n"
    print "accuracy: {}".format(mean(accuracy))
    print "precision: {}".format(mean(precision))
    print "recall:    {}".format(mean(recall))
    return mean(precision), mean(recall)

### Test the Different Classifiers

# evaluate_clf(l_clf, features, labels)
# evaluate_clf(k_clf, features, labels)
# evaluate_clf(s_clf, features, labels)
# evaluate_clf(rf_clf, features, labels)
# evaluate_clf(nn_clf, features, labels)
# evaluate_clf(mpl_clf, features, labels)
# evaluate_clf(qd_clf, features, labels)



### Please name your classifier clf for easy export below.

### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
### Determined that Logistic Regression was the best baseline
### Further tune the algorithm
data = featureFormat(my_dataset, my_feature_list)
labels, features = targetFeatureSplit(data)

l1_clf = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(tol = 0.00001, C = 0.0002, random_state = 42))])
evaluate_clf(l1_clf, features, labels)


clf = l1_clf
# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)