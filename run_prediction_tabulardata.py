import pandas as pd
import pickle
import numpy as np
import random
import os

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import KFold, StratifiedKFold



def train_ml_model(X_train, X_test, y_train, y_test, alg="RF"):

    if alg == "SVM":
        clf = SVC()

    elif alg == "RF":
        clf = RandomForestClassifier()

    elif alg == "DT":
        clf = DecisionTreeClassifier()

    elif alg == "MLP":
        clf = MLPClassifier()

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    pr = metrics.precision_score(y_test, y_pred) 
    rec = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred) 
    acc = metrics.accuracy_score(y_test, y_pred)  
    waf = metrics.f1_score(y_test, y_pred, average="weighted")
    auc = metrics.roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])
    return clf, pr, rec, f1, acc, waf, auc


def process_labels_file(path_entities_label):

    entities, dic_labels = [], {}
    with open(path_entities_label, 'r') as file_entities_label:
        for line in file_entities_label:
            ent, label = line.strip().split('\t')
            entities.append(ent)
            dic_labels[ent] = int(label)
    return entities, dic_labels


def split_dataset(path_entities_label, path_train_entities, path_test_entities, cv_folds, shuffle=False):
    entities, dic_labels = process_labels_file(path_entities_label)

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=shuffle)
    for cv, (train_index, test_index) in enumerate(skf.split(entities, [dic_labels[ent] for ent in entities])):

        with open(path_train_entities + '_' + str(cv) + '.tsv', 'w') as train_entities_file:
            for ind in train_index:
                train_entities_file.write(entities[ind] + '\n')
        with open(path_test_entities + '_' + str(cv) + '.tsv', 'w') as test_entities_file:
            for ind in test_index:
                test_entities_file.write(entities[ind] + '\n')


def run_ml_model(path_features_file, path_ml_model, path_metrics, cv_folds, alg, path_entities_label, path_train_entities, path_test_entities):

    entities, dic_labels = process_labels_file(path_entities_label)
    print('processing labels file - DONE!!')

    with open(path_features_file, "rb") as features_file:
        dic_features = pickle.load(features_file)
    print('reading KGE model - DONE!!')

    for cv in range(cv_folds):

        train_entities = [ent.strip() for ent in open(path_train_entities + '_' + str(cv) + '.tsv', 'r').readlines()]
        test_entities = [ent.strip() for ent in open(path_test_entities + '_' + str(cv) + '.tsv', 'r').readlines()]
        print('processing test and train files - DONE!!')

        X_train = [list(dic_features[ent]) for ent in train_entities]
        X_test = [list(dic_features[ent]) for ent in test_entities]
        y_train = [dic_labels[ent] for ent in train_entities]
        y_test = [dic_labels[ent] for ent in test_entities]

        clf, pr, rec, f1, acc, waf, auc =  train_ml_model(X_train, X_test, y_train, y_test, alg)

        print('training and predicting ML model - DONE!!')
        
        with open(path_ml_model + '_' + str(cv) + '.pickle', 'wb') as file_clf:
            pickle.dump(clf, file_clf)
        print('saving ML model - DONE!!')

        with open(path_metrics + '_' + str(cv) + '.tsv', 'w') as file_metrics:
            file_metrics.write('Metric\tValue\n')
            file_metrics.write('Accuracy\t' + str(acc) + '\n')
            file_metrics.write('Precision\t' + str(pr) + '\n')
            file_metrics.write('Recall\t' + str(rec) + '\n')
            file_metrics.write('F1-Score\t' + str(f1) + '\n')
            file_metrics.write('WAF\t' + str(waf) + '\n')
            file_metrics.write('AUC\t' + str(auc) + '\n')
        print('computing metrics - DONE!!')


