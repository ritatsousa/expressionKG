import requests
import pickle
import csv
import statistics
import numpy as np
import pandas as pd

import rdflib
from rdflib import Graph, URIRef, XSD, Literal, Namespace, BNode, Literal, RDF


def process_gene_expression_file(path_gene_expression_file, path_output):

    genes = []

    with open(path_gene_expression_file, "r") as gene_expression_file:

        genes = gene_expression_file.readline().strip().split("	")[:-1]
        dic_features, patients, labels = {}, [], []

        for line in gene_expression_file:
            elements_line = line.strip().split("	")
            patients.append(elements_line[0])
            dic_features[elements_line[0]] = [float(i) for i in elements_line[1:-1]]
            labels.append(elements_line[-1])
    print('Reading Gene expression File - DONE!')

    with open(path_output + "Genes_code.tsv", "w") as genes_file:
        for gene in genes:
            genes_file.write(gene + '\n')
    print('Writing Genes File - DONE!')

    with open(path_output + "Patients.tsv", "w") as entities_file:
        for j in range(len(patients)):
            label_str = labels[j]
            pat = patients[j] 
            if "Control" in label_str:
                label = 0 
            else: 
                label = 1
            entities_file.write('http://patients/' + pat + '\t' + str(label) + '\n')
    print('Writing Patients File - DONE!')
    
    with open(path_output + "Gene_expression_features.pickle", "wb") as features_file:
        pickle.dump(dic_features, features_file)
    print('Writing Genes Expression Features - DONE!')


def prepare_genes_mapping(path_gene_codes_file, path_colsname_2_gene_name, path_cols_names, path_output):

    with open(path_gene_codes_file, "r") as gene_codes_file:
        genes_code = [line.strip() for line in gene_codes_file.readlines()]
    print('Processing Genes File - DONE!')

    dic_genes_code_2_colsname, dic_colsname_2_gene_name = {}, {}

    with open(path_colsname_2_gene_name, "r") as colsname_2_gene_name_file:
        colsname_2_gene_name_file.readline()
        for line in colsname_2_gene_name_file:
            line_elements = line.strip().split("	")
            if len(line_elements) == 2:
                dic_colsname_2_gene_name[line_elements[0]] = line_elements[1]

    with open(path_cols_names, "r") as cols_names_file:
        cols_names_file.readline()
        for line in cols_names_file:
            col_num, col_name = line.strip().split("	")
            dic_genes_code_2_colsname["M" + col_num] = col_name

    with open(path_output + "Genes.tsv", "w") as genes_file:
        for gene in genes_code:
            genes_file.write(dic_colsname_2_gene_name[dic_genes_code_2_colsname[gene]] + '\n')
    print('Writing Genes File - DONE!')


def mapping_genes_2_proteins(path_mapping_file, path_output):

    dic_GO_annotations, list_ppi, dic_gene2protein = {}, set(), {}

    with open(path_mapping_file, 'r') as mapping_file:

        reader =  csv.DictReader(mapping_file, delimiter='\t')
        for row in reader:
            gene, protein, GO_annotations, ppis = row['From'], row['Entry'], row['Gene Ontology (GO)'], row['Interacts with']

            if gene in dic_gene2protein:
                dic_gene2protein[gene] = dic_gene2protein[gene] + [protein]
            else:
                dic_gene2protein[gene] = [protein]

            if GO_annotations != "":
                if protein not in dic_GO_annotations:
                    GO_annotations = GO_annotations.split(";")
                    dic_GO_annotations[protein] = [annot[:-1].split("[")[-1] for annot in GO_annotations]

            if ppis != "":
                for ppi in ppis.split("; "):
                    if "[" in ppi:
                        ppi = ppi.split("[")[-1].replace("]","")
                    list_ppi.add((protein,ppi))
                    list_ppi.add((ppi,protein))

    with open(path_output + "dic_GO_annotations.pickle", "wb") as dic_GO_annotations_file:
        pickle.dump(dic_GO_annotations, dic_GO_annotations_file)

    with open(path_output + "dic_gene2protein.pickle", "wb") as dic_gene2protein_file:
        pickle.dump(dic_gene2protein, dic_gene2protein_file)

    with open(path_output + "PPIs.tsv", "w") as ppi_file:
        for (p1, p2) in list_ppi:
            ppi_file.write(p1 + '\t' + p2 + '\n')


def create_bins(data, percentage=0.1):
    unique_values = np.unique(data)
    num_bins = int(len(unique_values) * percentage)
    bins = np.linspace(unique_values.min(), unique_values.max(), num_bins + 1)
    return bins


def find_bin(value, bins):
    bin_index = np.digitize(value, bins)
    return bin_index

    
def build_expression_data_kg_binningexpression(path_gene_expression_features, path_genes, path_output_kg):

    with open(path_gene_expression_features, 'rb') as gene_expression_features_file:
        dic_features = pickle.load(gene_expression_features_file)
    genes = [gene.strip() for gene in open(path_genes, 'r').readlines()]

    dic_expression_genes = {gene:[] for gene in genes}
    for patient in dic_features:
        for i in range(len(dic_features[patient])):
            dic_expression_genes[genes[i]].append(dic_features[patient][i])

    ns = Namespace("http://")
    ns_patient = Namespace("http://patients/")
    ns_gene = Namespace("https://www.genecards.org/cgi-bin/carddisp.pl?gene=")
    ns_expression = Namespace("https://ExpressionLevelBin/")
    graph = rdflib.Graph()

    dic_expression_genes_bin = {}
    for gene in dic_expression_genes:
        gene_uri = ns_gene[gene]
        graph.add((gene_uri, RDF.type, ns.Gene))
        dic_expression_genes_bin[gene] = create_bins(dic_expression_genes[gene])

    for patient in dic_features:
        patient_uri = ns_patient[patient]
        graph.add((patient_uri, RDF.type, ns.Patient))

        for i in range(len(dic_features[patient])):
            
            expression_gene = ns_gene[genes[i]]
            expression_level = dic_features[patient][i]

            bin = find_bin(expression_level, dic_expression_genes_bin[genes[i]])

            expression_node = BNode()
            graph.add((patient_uri, ns.hasGeneExpression, expression_node))
            graph.add((expression_node, ns.hasExpressionLevel, ns_expression[str(bin)] ))
            graph.add((expression_node, ns.hasGeneExpression, expression_gene))
            
    graph.serialize(path_output_kg, format='nt')
    

def build_expression_data_kg_linkpatientgene(path_gene_expression_features, path_genes, path_output_kg):

    with open(path_gene_expression_features, 'rb') as gene_expression_features_file:
        dic_features = pickle.load(gene_expression_features_file)
    genes = [gene.strip() for gene in open(path_genes, 'r').readlines()]

    dic_expression_genes = {gene:[] for gene in genes}
    for patient in dic_features:
        for i in range(len(dic_features[patient])):
            dic_expression_genes[genes[i]].append(dic_features[patient][i])

    ns = Namespace("http://")
    ns_patient = Namespace("http://patients/")
    ns_gene = Namespace("https://www.genecards.org/cgi-bin/carddisp.pl?gene=")
    graph = rdflib.Graph()

    for gene in dic_expression_genes:
        gene_uri = ns_gene[gene]
        graph.add((gene_uri, RDF.type, ns.Gene))
        dic_expression_genes[gene] = statistics.mean(dic_expression_genes[gene])
    
    for patient in dic_features:
        patient_uri = ns_patient[patient]
        graph.add((patient_uri, RDF.type, ns.Patient))

        for i in range(len(dic_features[patient])):
            expression_level = dic_features[patient][i]
            if expression_level >= dic_expression_genes[genes[i]]:
                expression_gene = ns_gene[genes[i]]
                graph.add((patient_uri, ns.hasGeneExpression, expression_gene))
  
    graph.serialize(path_output_kg, format='nt')


def unite_datasets(path_output, list_patient_files, list_dic_features_file, list_n_feat):

    with open(path_output + "Patients.tsv", "w") as output_patient_file:
        for path_patient_file in list_patient_files:
            with open(path_patient_file, "r") as patient_file:
                for line in patient_file:
                    output_patient_file.write(line)
    
    dic_features = {}
    n_total_feat = sum(list_n_feat)

    list_input_features = []
    for path_dic_features_file in list_dic_features_file:
        with open(path_dic_features_file, 'rb') as dic_features_file:
            list_input_features.append(pickle.load(dic_features_file))
    
    n_zeros_before = 0
    i = 0
    for input_dic_feature in list_input_features:
        n_zeros_after = n_total_feat - (list_n_feat[i]+n_zeros_before)
        for patient in input_dic_feature:
            new_representation = [0]*n_zeros_before + input_dic_feature[patient] + [0]*n_zeros_after
            dic_features[patient] = new_representation
        n_zeros_before = n_zeros_before+list_n_feat[i]
        i = i+1
        

    with open(path_output + "Gene_expression_features.pickle", "wb") as features_file:
        pickle.dump(dic_features, features_file)
    print('Writing Genes Expression Features - DONE!')


def join_kgs(path_output_kg, list_kg_files):
    kg =  rdflib.Graph()
    for kg_file in list_kg_files:
        kg.parse(kg_file)
    kg.serialize(path_output_kg, format='nt')


def transform_expression_data(path_input_features, path_output_features):

    with open(path_input_features, 'rb') as input_features:
        dic_features_raw = pickle.load(input_features)
    dic_features = {"http://patients/" + pat: dic_features_raw[pat] for pat in dic_features_raw}

    with open(path_output_features, "wb") as dic_output_features_file:
        pickle.dump(dic_features, dic_output_features_file)


def build_enriched_expression_data_kg(path_kg, path_GO, path_dic_GO_annotations, path_gene2protein, path_ppi_file, path_output_kg_enriched):

    kg =  rdflib.Graph()
    kg.parse(path_kg)
    kg.parse(path_GO)

    ns = Namespace("http://")
    ns_protein = Namespace("https://www.uniprot.org/uniprotkb/")
    ns_go = Namespace("http://purl.obolibrary.org/obo/GO_")
    ns_gene = Namespace("https://www.genecards.org/cgi-bin/carddisp.pl?gene=")

    with open(path_dic_GO_annotations, 'rb') as dic_GO_annotations_file:
        dic_GO_annotations = pickle.load(dic_GO_annotations_file)

    with open(path_gene2protein, 'rb') as gene2protein_file:
        gene2protein = pickle.load(gene2protein_file)
        
    for prot in dic_GO_annotations:
        protein_uri = ns_protein[prot]
        kg.add((protein_uri, RDF.type, ns.Protein))
        for annot in dic_GO_annotations[prot]:
            go_uri = ns_go[annot.split(":")[-1]]
            kg.add((protein_uri, ns.hasAnnotation, go_uri))

    for gene in gene2protein:
        gene_uri = ns_gene[gene]
        for prot in gene2protein[gene]:
            protein_uri = ns_protein[prot]
            kg.add((gene_uri, ns.Codificates, protein_uri))

    with open(path_ppi_file, 'r') as ppi_file:
        for line in ppi_file:
            p1 ,p2 = line.strip().split('\t')
            p1_uri = ns_protein[p1]
            p2_uri = ns_protein[p2]
            kg.add((p1_uri, RDF.type, ns.Protein))
            kg.add((p2_uri, RDF.type, ns.Protein))
            kg.add((p2_uri, ns.hasInteraction, p1_uri))
            kg.add((p1_uri, ns.hasInteraction, p2_uri))
           
    kg.serialize(path_output_kg_enriched, format='nt')


def process_GO_annotations_file(path_GO_annotations):
    dic_GO_annotations = {}

    with open(path_GO_annotations, 'r') as file:
        for line in file:
            # Skip comment lines
            if line.startswith('!'):
                continue

            # Split the line into fields
            fields = line.strip().split('\t')
            protein_id, go_term = fields[1], fields[4]

            # Store the information in a dictionary
            if protein_id not in dic_GO_annotations:
                dic_GO_annotations[protein_id] = []

            dic_GO_annotations[protein_id].append(go_term)
    return dic_GO_annotations


def build_GO_PPI_kg(path_genes_file, path_GO, path_dic_GO_annotations, path_gene2protein, path_ppi_file, path_output_kg_enriched):

    kg =  rdflib.Graph()
    kg.parse(path_GO)

    ns = Namespace("http://")
    ns_protein = Namespace("https://www.uniprot.org/uniprotkb/")
    ns_go = Namespace("http://purl.obolibrary.org/obo/GO_")
    ns_gene = Namespace("https://www.genecards.org/cgi-bin/carddisp.pl?gene=")

    with open(path_genes_file, 'r') as gene_file:
        genes = [line.strip() for line in gene_file.readlines()]
    for gene in genes:
        gene_uri = ns_gene[gene]
        kg.add((gene_uri, RDF.type, ns.gene))

    with open(path_dic_GO_annotations, 'rb') as dic_GO_annotations_file:
        dic_GO_annotations = pickle.load(dic_GO_annotations_file)

    with open(path_gene2protein, 'rb') as gene2protein_file:
        gene2protein = pickle.load(gene2protein_file)
        
    for prot in dic_GO_annotations:
        protein_uri = ns_protein[prot]
        kg.add((protein_uri, RDF.type, ns.Protein))
        for annot in dic_GO_annotations[prot]:
            go_uri = ns_go[annot.split(":")[-1]]
            kg.add((protein_uri, ns.hasAnnotation, go_uri))

    for gene in gene2protein:
        gene_uri = ns_gene[gene]
        for prot in gene2protein[gene]:
            protein_uri = ns_protein[prot]
            kg.add((gene_uri, ns.Codificates, protein_uri))

    with open(path_ppi_file, 'r') as ppi_file:
        for line in ppi_file:
            p1 ,p2 = line.strip().split('\t')
            p1_uri = ns_protein[p1]
            p2_uri = ns_protein[p2]
            kg.add((p1_uri, RDF.type, ns.Protein))
            kg.add((p2_uri, RDF.type, ns.Protein))
            kg.add((p2_uri, ns.hasInteraction, p1_uri))
            kg.add((p1_uri, ns.hasInteraction, p2_uri))
           
    kg.serialize(path_output_kg_enriched, format='nt')


def generate_input_features(path_patient_file, path_merged_file, path_input_features):

    dic_features, patients = {}, []

    with open(path_patient_file, "r") as patient_file:
        for line in patient_file:
            patients.append(line.strip().split("	")[0].replace("http://patients/", ""))

    with open(path_merged_file, 'rb') as merged_features_file:
        dic_merged_features = pickle.load(merged_features_file)
    
    for pat in dic_merged_features:
        if pat in patients:
            dic_features[pat] = dic_merged_features[pat]

    with open(path_input_features, "wb") as features_file:
        pickle.dump(dic_features, features_file)
    


###################################################################
#                             GSE30208                            #
###################################################################

#################### PROCESS GENE EXPRESSIONS FILE #################### 
# path_output = "./data/GSE30208/"
# path_gene_expression_file = "./data/GSE30208/GSE30208.tsv"
# process_gene_expression_file(path_gene_expression_file, path_output)

################### PREPARE GENE MAPPING #################### 
# path_gene_codes_file = "./data/GSE30208/Genes_code.tsv"
# path_colsname_2_gene_name = "./data/GSE30208/TargetMarkers.tsv"
# path_cols_names = "./data/GSE30208/colnamesGSE30208.tsv"
# prepare_genes_mapping(path_gene_codes_file, path_colsname_2_gene_name, path_cols_names, path_output)

################### MAPPING GENE 2 PROTEIN #################### 
# path_mapping_file= "./data/GSE30208/Gene2protein.tsv"
# mapping_genes_2_proteins(path_mapping_file, path_output)

################### TRANSFORM EXPRESSION DATA ####################
# path_input_features = "./data/GSE30208/Gene_expression_features.pickle"
# path_output_features = "./data/GSE30208/Gene_expression_features_2.pickle"
# transform_expression_data(path_input_features, path_output_features)

################### BUILD EXPRESSION DATA KG ####################
# path_gene_expression_features = "./data/GSE30208/Gene_expression_features.pickle"
# path_genes = "./data/GSE30208/Genes_manual_check.tsv"
    
# path_output_kg = "./data/GSE30208/kg_10binningexpression_data.nt"
# build_expression_data_kg_binningexpression(path_gene_expression_features, path_genes, path_output_kg)

# path_output_kg = "./data/GSE30208/kg_avg_linkpatientgene.nt"
# build_expression_data_kg_linkpatientgene(path_gene_expression_features, path_genes, path_output_kg)

################### BUILD ENRICHED EXPRESSION DATA KG ####################
# path_gene2protein = "./data/GSE30208_/dic_gene2protein.pickle"
# path_ppi_file = "./data/GSE30208/PPIs.tsv"
# path_GO = "./data/GO/go.owl"
# path_dic_GO_annotations = "./data/GSE30208/dic_GO_annotations.pickle"
# path_GO_annotations = "./data/GO/goa_human.gaf"

# path_output_kg = "./data/GSE30208/kg_10binningexpression_data.nt"
# path_output_kg_enriched = "./data/GSE30208/kg_10binningexpression_data_enriched_GO_PPI.nt"
# build_enriched_expression_data_kg(path_output_kg, path_GO, path_dic_GO_annotations, path_gene2protein, path_ppi_file, path_output_kg_enriched)

# path_output_kg = "./data/GSE30208/kg_avg_linkpatientgene.nt"
# path_output_kg_enriched = "./data/GSE30208/kg_avg_linkpatientgene_enriched_GO_PPI.nt"
# build_enriched_expression_data_kg(path_output_kg, path_GO, path_dic_GO_annotations, path_gene2protein, path_ppi_file, path_output_kg_enriched)

       
###################################################################
#                    GSE30208_GSE15932_GSE55098                   #
###################################################################

# path_output = "./data/GSE30208_GSE15932_GSE55098(corrected)/"
# list_patient_files = ["./data/GSE30208/Patients.tsv", "./data/GSE15932/Patients.tsv", "./data/GSE55098/Patients.tsv"]
# list_dic_features_file = ["./data/GSE30208/Gene_expression_features.pickle", 
#                           "./data/GSE15932/Gene_expression_features.pickle",
#                            "./data/GSE55098/Gene_expression_features.pickle"]
# list_n_feat = [368, 754, 754]
# unite_datasets(path_output, list_patient_files, list_dic_features_file, list_n_feat)

################### TRANSFORM EXPRESSION DATA ####################
# path_input_features = "./data/GSE30208_GSE15932_GSE55098/Gene_expression_features.pickle"
# path_output_features = "./data/GSE30208_GSE15932_GSE55098/Gene_expression_features_2.pickle"
# transform_expression_data(path_input_features, path_output_features)

# path_input_features = "./data/GSE30208_GSE15932_GSE55098(corrected)/Gene_expression_features.pickle"
# path_output_features = "./data/GSE30208_GSE15932_GSE55098(corrected)/Gene_expression_features_2.pickle"
# transform_expression_data(path_input_features, path_output_features)

################## BUILD EXPRESSION DATA KG ####################

# path_output_kg = "./data/GSE30208_GSE15932_GSE55098/kg_10binningexpression_data.nt" 
# list_kg_files = ["./data/GSE30208/kg_10binningexpression_data.nt", 
#                  "./data/GSE15932_GPL570_GSE55098_GPL570/kg_10binningexpression_data.nt"]
# join_kgs(path_output_kg, list_kg_files)

# path_output_kg = "./data/GSE30208_GSE15932_GSE55098/kg_avg_linkpatientgene.nt" 
# list_kg_files = ["./data/GSE30208/kg_avg_linkpatientgene.nt", 
#                  "./data/GSE15932_GPL570_GSE55098_GPL570/kg_avg_linkpatientgene.nt"]
# join_kgs(path_output_kg, list_kg_files)

# ################## BUILD ENRICHED EXPRESSION DATA KG ####################
# path_output_kg = "./data/GSE30208_GSE15932_GSE55098/kg_10binningexpression_data_enriched_GO_PPI.nt" 
# list_kg_files = ["./data/GSE30208/kg_10binningexpression_data_enriched_GO_PPI.nt", 
#                  "./data/GSE15932_GPL570_GSE55098_GPL570/kg_10binningexpression_data_enriched_GO_PPI.nt"]
# join_kgs(path_output_kg, list_kg_files)

# path_output_kg = "./data/GSE30208_GSE15932_GSE55098/kg_avg_linkpatientgene_enriched_GO_PPI.nt" 
# list_kg_files = ["./data/GSE30208/kg_avg_linkpatientgene_enriched_GO_PPI.nt", 
#                  "./data/GSE15932_GPL570_GSE55098_GPL570/kg_avg_linkpatientgene_enriched_GO_PPI.nt"]
# join_kgs(path_output_kg, list_kg_files)

# path_output_kg = "./data/GSE30208_GSE15932_GSE55098/kg_enriched_GO_PPI.nt" 
# list_kg_files = ["./data/GSE30208/kg_enriched_GO_PPI.nt", 
#                  "./data/GSE15932_GPL570_GSE55098_GPL570/kg_enriched_GO_PPI.nt"]
# join_kgs(path_output_kg, list_kg_files)