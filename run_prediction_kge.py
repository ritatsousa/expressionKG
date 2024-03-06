import pandas as pd
import pickle

from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.embedders import Word2Vec
from pyrdf2vec.graphs import KG
from pyrdf2vec.walkers import RandomWalker
from pyrdf2vec.samplers import WeightedEdgeSampler

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import KFold, StratifiedKFold

import os



from multiprocessing import cpu_count
n_jobs = cpu_count()


def ensure_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def generate_rdfvec_embeddings(entities, path_kg, size_emb, max_walks, max_depth):

    kg = KG(location=path_kg, is_remote=False, mul_req=False)
    print('creating graph -> DONE!')

    transformer = RDF2VecTransformer(
    Word2Vec(vector_size=size_emb),
    walkers=[RandomWalker(max_depth=max_depth, max_walks=max_walks, n_jobs=n_jobs)],
    verbose=2
    )

    # Get our embeddings.
    embeddings, literals = transformer.fit_transform(kg, entities)
    kge_model = transformer
    return kge_model, {ent:emb for ent, emb in zip(entities, embeddings)}


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


def split_dataset(path_entities_label, path_train_entities, path_test_entities, cv_folds):
    entities, dic_labels = process_labels_file(path_entities_label)

    skf = StratifiedKFold(n_splits=cv_folds)
    for cv, (train_index, test_index) in enumerate(skf.split(entities, [dic_labels[ent] for ent in entities])):

        with open(path_train_entities + '_' + str(cv) + '.tsv', 'w') as train_entities_file:
            for ind in train_index:
                train_entities_file.write(entities[ind] + '\n')
        with open(path_test_entities + '_' + str(cv) + '.tsv', 'w') as test_entities_file:
            for ind in test_index:
                test_entities_file.write(entities[ind] + '\n')


def run_ml_model(path_kg, path_kge_model, size_emb, max_walks, max_depth, path_ml_model, path_metrics, cv_folds, alg, path_entities_label, path_train_entities, path_test_entities):

    entities, dic_labels = process_labels_file(path_entities_label)
    print('processing labels file - DONE!!')

    if os.path.exists(path_kge_model):
        with open(path_kge_model, "rb") as file_kge:
            kge_model = pickle.load(file_kge)
        print('reading KGE model - DONE!!')

        dic_features = {ent:emb for ent, emb in zip(entities, kge_model.embedder.transform(entities))}

    else:
        kge_model, dic_features = generate_rdfvec_embeddings(entities, path_kg, size_emb, max_walks, max_depth)
        print('generating RDF2vec embeddings - DONE!!')

        with open(path_kge_model, "wb") as file_kge:
            pickle.dump(kge_model, file_kge)
        print('saving KGE model - DONE!!')

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


