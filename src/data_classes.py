import numpy as np
import pandas as pd
import scipy.sparse as sp
import os
import json


class FAERSData:
    reports_by_drugs: sp.csr_matrix
    reports_by_reactions: sp.csr_matrix
    reports_by_indications: sp.csr_matrix
    drug2index: dict
    reaction2index: dict
    indication2index: dict
    
    def __init__(self, data_dir):
        
        self.reports_by_drugs = sp.load_npz(os.path.join(data_dir, 'reports_by_drugs.npz'))
        self.reports_by_reactions = sp.load_npz(os.path.join(data_dir, 'reports_by_reactions.npz'))
        self.reports_by_indications = sp.load_npz(os.path.join(data_dir, 'reports_by_indications.npz'))
        
        self.drug2index = json.load(open(os.path.join(data_dir, 'drug2index.json')))
        self.reaction2index = json.load(open(os.path.join(data_dir, 'reaction2index.json')))
        self.indication2index = json.load(open(os.path.join(data_dir, 'indication2index.json')))


def remap_offsides_data(offsides_df, p_to_mapping_dict='/Users/kivelsons/offsides/dconvae/data/offsides_to_meddra.json'):

    # Load the mapping
    with open(p_to_mapping_dict, 'r') as f:
        offsides_to_meddra = json.load(f)

    mapping = pd.Series(offsides_to_meddra)

    # Replace condition_concept_name with mapped MedDRA ID where possible and drop rows with no mapping
    offsides_df = offsides_df[offsides_df['condition_concept_name'].isin(mapping.index)]
    offsides_df['condition_meddra_id'] = offsides_df['condition_concept_name'].map(mapping)
    offsides_df = offsides_df.drop('condition_concept_name', axis=1)
    return offsides_df


def get_fears_and_offsides_data(offsides_data_dir='/Users/kivelsons/offsides/dconvae/data/', faers_data_dir='/Users/kivelsons/offsides/dconvae/data/drug_2023-2023/'):
    offsides_data = pd.read_csv(os.path.join(offsides_data_dir, 'OFFSIDES.csv.gz'))
    offsides_data = remap_offsides_data(offsides_data)
    faers_data = FAERSData(faers_data_dir)
    return faers_data, offsides_data