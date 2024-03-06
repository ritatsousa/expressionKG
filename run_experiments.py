import run_prediction_composite_kge
import run_prediction_kge
import run_prediction_tabulardata

import os 
def ensure_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


################################################ GSE30208 #################################################
path_entities_label = "data/GSE30208/Patients.tsv" 
path_train_entities = "prediction/GSE30208/Train_Entities"
path_test_entities = "prediction/GSE30208/Test_Entities"
path_features_file = "data/GSE30208/Gene_expression_features.pickle"
cv_folds = 5
alg = "DT"

path_ml_model = "prediction/semwebmeda/GSE30208/tabulardata/" + alg + "_model_with_expression_counts.pickle"
path_metrics= "prediction/semwebmeda/GSE30208/tabulardata/Metrics_" + alg + "_model_with_expression_counts"
run_prediction_tabulardata.run_ml_model(path_features_file, path_ml_model, path_metrics, cv_folds, alg, path_entities_label, path_train_entities, path_test_entities)


# ################################################ GSE30208_GSE15932_GSE55098 #################################################
path_entities_label = "data/GSE30208_GSE15932_GSE55098/Patients.tsv" 
path_train_entities = "prediction/GSE30208_GSE15932_GSE55098/Train_Entities"
path_test_entities = "prediction/GSE30208_GSE15932_GSE55098/Test_Entities"
path_features_file = "data/GSE30208_GSE15932_GSE55098/Gene_expression_features.pickle"
cv_folds = 5
alg = "DT"

path_ml_model = "prediction/semwebmeda/GSE30208_GSE15932_GSE55098/tabulardata/" + alg + "_model_with_expression_counts.pickle"
path_metrics= "prediction/semwebmeda/GSE30208_GSE15932_GSE55098/tabulardata/Metrics_" + alg + "_model_with_expression_counts"
run_prediction_tabulardata.run_ml_model(path_features_file, path_ml_model, path_metrics, cv_folds, alg, path_entities_label, path_train_entities, path_test_entities)    

size_emb = 500 
max_walks = None
max_depth = 4

path_kg = "./data/GSE30208_GSE15932_GSE55098/kg_10binningexpression_data.nt"
path_kge_model = "./prediction/GSE30208_GSE15932_GSE55098/embeddings_" + str(size_emb) + "_" + str(max_walks) + "_" + str(max_depth) + "_10binningexpression/rdf2vec_model.pickle"
path_ml_model = "./prediction/semwebmeda/GSE30208_GSE15932_GSE55098/embeddings_" + str(size_emb) + "_" + str(max_walks) + "_" + str(max_depth) + "_10binningexpression/" + alg + "_model_with_rdf2vec_embeddings.pickle"
path_metrics= "./prediction/semwebmeda/GSE30208_GSE15932_GSE55098/embeddings_" + str(size_emb) + "_" + str(max_walks) + "_" + str(max_depth) + "_10binningexpression/Metrics_" + alg + "_model_with_rdf2vec_embeddings"
run_prediction_kge.run_ml_model(path_kg, path_kge_model, size_emb, max_walks, max_depth, path_ml_model, path_metrics, cv_folds, alg, path_entities_label, path_train_entities, path_test_entities)

path_kg = "./data/GSE30208_GSE15932_GSE55098/kg_avg_linkpatientgene.nt"
path_kge_model = "./prediction/GSE30208_GSE15932_GSE55098/embeddings_" + str(size_emb) + "_" + str(max_walks) + "_" + str(max_depth) + "_avg_linkpatientgene/rdf2vec_model.pickle"
path_ml_model = "./prediction/semwebmeda/GSE30208_GSE15932_GSE55098/embeddings_" + str(size_emb) + "_" + str(max_walks) + "_" + str(max_depth) + "_avg_linkpatientgene/" + alg + "_model_with_rdf2vec_embeddings.pickle"
path_metrics= "./prediction/semwebmeda/GSE30208_GSE15932_GSE55098/embeddings_" + str(size_emb) + "_" + str(max_walks) + "_" + str(max_depth) + "_avg_linkpatientgene/Metrics_" + alg + "_model_with_rdf2vec_embeddings"
run_prediction_kge.run_ml_model(path_kg, path_kge_model, size_emb, max_walks, max_depth, path_ml_model, path_metrics, cv_folds, alg, path_entities_label, path_train_entities, path_test_entities)

size_emb = 500 
max_walks = 5000
max_depth = 4

path_kg = "./data/GSE30208_GSE15932_GSE55098/kg_10binningexpression_data_enriched_GO_PPI.nt"
path_kge_model = "./prediction/GSE30208_GSE15932_GSE55098/embeddings_" + str(size_emb) + "_" + str(max_walks) + "_" + str(max_depth) + "_10binningexpression_GO_PPI/rdf2vec_model.pickle"
path_ml_model = "./prediction/semwebmeda/GSE30208_GSE15932_GSE55098/embeddings_" + str(size_emb) + "_" + str(max_walks) + "_" + str(max_depth) + "_10binningexpression_GO_PPI/" + alg + "_model_with_rdf2vec_embeddings.pickle"
path_metrics= "./prediction/semwebmeda/GSE30208_GSE15932_GSE55098/embeddings_" + str(size_emb) + "_" + str(max_walks) + "_" + str(max_depth) + "_10binningexpression_GO_PPI/Metrics_" + alg + "_model_with_rdf2vec_embeddings"
run_prediction_kge.run_ml_model(path_kg, path_kge_model, size_emb, max_walks, max_depth, path_ml_model, path_metrics, cv_folds, alg, path_entities_label, path_train_entities, path_test_entities)

path_kg = "./data/GSE30208_GSE15932_GSE55098/kg_avg_linkpatientgene_enriched_GO_PPI.nt"
path_kge_model = "./prediction/GSE30208_GSE15932_GSE55098/embeddings_" + str(size_emb) + "_" + str(max_walks) + "_" + str(max_depth) + "_avg_linkpatientgene_GO_PPI/rdf2vec_model.pickle"
path_ml_model = "./prediction/semwebmeda/GSE30208_GSE15932_GSE55098/embeddings_" + str(size_emb) + "_" + str(max_walks) + "_" + str(max_depth) + "_avg_linkpatientgene_GO_PPI/" + alg + "_model_with_rdf2vec_embeddings.pickle"
path_metrics= "./prediction/semwebmeda/GSE30208_GSE15932_GSE55098/embeddings_" + str(size_emb) + "_" + str(max_walks) + "_" + str(max_depth) + "_avg_linkpatientgene_GO_PPI/Metrics_" + alg + "_model_with_rdf2vec_embeddings"
run_prediction_kge.run_ml_model(path_kg, path_kge_model, size_emb, max_walks, max_depth, path_ml_model, path_metrics, cv_folds, alg, path_entities_label, path_train_entities, path_test_entities)

size_emb = 100 
max_walks = 500
max_depth = 4
path_features_file = "./data/GSE30208_GSE15932_GSE55098/Gene_expression_features.pickle"
path_genes_file = "./data/GSE30208_GSE15932_GSE55098/Genes.tsv"
type_representation = "weighthedavg_generepresentations"
path_kg = "./data/GSE30208_GSE15932_GSE55098/kg_enriched_GO_PPI.nt"
path_kge_model = "./prediction/GSE30208_GSE15932_GSE55098/embeddings_genes_" + str(size_emb) + "_" + str(max_walks) + "_" + str(max_depth) + "/rdf2vec_model.pickle"
path_ml_model = "./prediction/semwebmeda/GSE30208_GSE15932_GSE55098/embeddings_genes_" + str(size_emb) + "_" + str(max_walks) + "_" + str(max_depth) + "/" + type_representation + "/" + alg + "_model_with_rdf2vec_embeddings.pickle"
path_metrics= "./prediction/semwebmeda/GSE30208_GSE15932_GSE55098/embeddings_genes_" + str(size_emb) + "_" + str(max_walks) + "_" + str(max_depth) + "/" + type_representation + "/Metrics_" + alg + "_model_with_rdf2vec_embeddings"
path_representations = "./prediction/semwebmeda/GSE30208_GSE15932_GSE55098/embeddings_genes_" + str(size_emb) + "_" + str(max_walks) + "_" + str(max_depth) + "/" + type_representation + "/Patients_representations.pickle" 
run_prediction_composite_kge.run_ml_model(path_features_file, path_genes_file, path_kg, path_kge_model, size_emb, max_walks, max_depth, path_representations, path_ml_model, path_metrics, cv_folds, alg, path_entities_label, path_train_entities, path_test_entities)

