import os
import sys
import csv
import copy
import json
import tqdm
import string
import random
import argparse
import numpy as np
import scipy as sp
import pandas as pd

from collections import Counter, defaultdict

import seaborn as sns
import matplotlib.pyplot as plt

import sklearn.metrics

# TODO: Refactor this so that the datasets are created separately from when they're saved to disk.
# TODO: This will support if/when we want to call the generation directly instead of using the file
# TODO: system as an intermediary. 

# TODO: Add confounding by co-prescription. 
# TODO: Add confounding by age.
# TODO: Add confounding by sex. 

def numpy_safe_json_dump(data, filepath, indent=2):
    """
    Safely saves dictionary containing NumPy types to a JSON file by converting them to Python native types.
    
    Args:
        data (dict): Dictionary to save
        filepath (str): Path to save the JSON file
    """
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NumpyEncoder, self).default(obj)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, cls=NumpyEncoder, indent=indent)

def compute_matrices(m1, m2, num_reports):
    #print(m1.shape)
    #print(m2.shape)
    A = (m1.T @ m2)
    B = (m1.sum(0).reshape((m1.shape[1],1)) - A)
    B[B==0] = 1
    C = (m2.sum(0) - A)
    C[C==0] = 1
    D = (num_reports-(A+B+C))
    
    PRR = ((A/B)/(C/D))
    Tc = A/(A+B+C)

    return {
        'A': A,
        'B': B,
        'C': C,
        'D': D,
        'PRR': PRR,
        'Tc': Tc
    }

def build_dataframe(matrices, ordered_m1, m1_name, ordered_m2, m2_name, m12label=None, m22label=None, minA=10):
    
    mask = matrices['A']>=minA
    indices = np.where(mask)
    dfdata = {}

    dfdata[m1_name] = [ordered_m1[i] for i in indices[0]]
    if m12label is not None:
        dfdata[f"{m1_name}_label"] = [m12label.get(ordered_m1[i], i) for i in indices[0]]
    
    dfdata[m2_name] = [ordered_m2[i] for i in indices[1]]
    if m22label is not None:
        dfdata[f"{m2_name}_label"] = [m22label.get(ordered_m2[i], i) for i in indices[1]]

    for key, mat in matrices.items():
        dfdata[key] = mat[mask]

    return pd.DataFrame(dfdata)

def parse_args():
    parser = argparse.ArgumentParser(description='Simulate and analyze drug-reaction relationships')
    
    # Data generation parameters
    parser.add_argument('--ndrugs', type=int, default=25,
                        help='Number of unique drugs (default: 25)')
    parser.add_argument('--nreactions', type=int, default=50,
                        help='Number of unique reactions (default: 50)')
    parser.add_argument('--nindications', type=int, default=30,
                        help='Number of unique indications (default: 30)')
    parser.add_argument('--nreports', type=int, default=10000,
                        help='Number of reports to generate (default: 10000)')
    parser.add_argument('--proportion-true', type=float, default=0.10,
                        help='Proportion of true drug-reaction relationships (default: 0.10)')
    parser.add_argument('--nexamples', type=int, default=100,
                        help='Number of confounded datasets to create (default: 100)')
    
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Directory for saving output files (default: current directory)')
    
    return parser.parse_args()



if __name__ == "__main__":

    args = parse_args()

    run_name  = ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(8))

    output_path = os.path.join(args.data_dir, f"{run_name}")
    print(f"Saving data to: {output_path}")
    os.makedirs(output_path, exist_ok=True)

    for arg_name, arg_value in args.__dict__.items():
        print(f"{arg_name}: {arg_value}")

    config = copy.deepcopy(args.__dict__)
    config['output_path'] = output_path

    ndrugs = args.ndrugs

    drugs = list()
    for drug in range(ndrugs):
        for _ in range(random.randint(1, 10)):
            drugs.append(f"D{drug}")

    drug_probs = pd.Series(drugs).value_counts(normalize=True)
    
    nreactions = args.nreactions

    reactions = list()
    for reaction in range(nreactions):
        for _ in range(random.randint(2,10)):
            reactions.append(f"Rxn{reaction}")

    rxn_probs = pd.Series(reactions).value_counts(normalize=True)
    
    nindications = args.nindications
    indications = list()
    for indication in range(nindications):
        for _ in range(random.randint(1, 10)):
            indications.append(f"I{indication}")

    ind_probs = pd.Series(indications).value_counts(normalize=True)

    drug_rxn_factors = dict()
    drug_rxn_truth = dict()

    proportion_true = args.proportion_true

    for drug in drug_probs.index:
        for rxn in rxn_probs.index:
            if np.random.rand() <= proportion_true:
                # true drug->reactions relationships
                factor = np.exp(np.random.normal(1, 1))
                drug_rxn_truth[(drug,rxn)] = 1.0
            else:
                # not true relatinoship 
                factor = np.exp(np.random.normal(0, 0.1))
                drug_rxn_truth[(drug,rxn)] = 0.0
            drug_rxn_factors[(drug,rxn)] = factor

    drf = list()
    for drug in drug_probs.index:
        drf.append([drug_rxn_factors[(drug, rxn)] for rxn in rxn_probs.index])

    plt.figure(figsize=(5,2))
    plt.hist(drug_rxn_factors.values(), bins=100)
    plt.title('Distribution of Drug Reaction Factors')
    plt.ylabel('Count')
    plt.xlabel('Factor')
    sns.despine()
    plt.savefig(os.path.join(output_path, 'drug_rxn_factors_hist.png'))
    plt.close()

    nreports = 10000

    reports = list()

    for rid in range(nreports):
        drugid = np.random.choice(len(drug_probs), 1, True, p=drug_probs)[0]
        adjusted_probs = drf[drugid]*rxn_probs
        adjusted_probs /= (adjusted_probs).sum()
        rxnid = np.random.choice(len(rxn_probs), 1, True, p=adjusted_probs)[0]
        reports.append([rid, drugid, rxnid])

    fh = open(os.path.join(output_path, 'true_reports.csv'), 'w')
    writer = csv.writer(fh)
    writer.writerow(['report_id', 'drug_id', 'reaction_id'])
    writer.writerows(reports)
    fh.close()

    ind_output_path = os.path.join(output_path, 'datasets')
    os.makedirs(ind_output_path, exist_ok=True)

    for i in tqdm.tqdm(range(args.nexamples)):

        config[f'dataset_{i}'] = dict()
        
        ind_rxn_factors = dict()

        for ind in ind_probs.index:
            # randomly select one reaction that this indication will cause
            rxnid = np.random.choice(len(rxn_probs), 1, True)[0]
            rxn = rxn_probs.index[rxnid]
            # indications cause big reactions
            factor = np.exp(np.random.normal(5, 1))
            ind_rxn_factors[(ind,rxn)] = factor

        irf = list()
        for ind in ind_probs.index:
            irf.append([ind_rxn_factors.get((ind, rxn), 1.0) for rxn in rxn_probs.index])

        plt.figure(figsize=(5,2))
        plt.hist(ind_rxn_factors.values(), bins=100)
        plt.title('Distribution of Indication Reaction Factors')
        plt.ylabel('Count')
        plt.xlabel('Factor')
        sns.despine()
        plt.savefig(os.path.join(ind_output_path, f'{i}_ind_rxn_factors_hist.png'))
        plt.close()

        # establish correlations between drugs and indications

        drug_indid_probs = defaultdict(dict)
        for drug in drug_probs.index:
            num_inds = max(1, np.random.poisson(lam=2))
            random_inds = np.random.choice(len(ind_probs), num_inds, False)
            inds = list()
            for ind in random_inds:
                for _ in range(random.randint(2,10)):
                    inds.append(ind)
            
            diprobs = pd.Series(inds).value_counts(normalize=True)
            for ind, prob in diprobs.items():
                drug_indid_probs[drug][ind] = prob
        
        # build report x drug, report by rxn, and report by indication matrices

        # reports by drugs
        drugs = np.zeros(shape=(nreports, len(drug_probs)))
        # reports by reactions
        rxns = np.zeros(shape=(nreports, len(rxn_probs)))
        # reports by indications
        inds = np.zeros(shape=(nreports, len(ind_probs)))

        for rid, drugid, rxnid in reports:
            
            # first add the drug and the reaction
            drugs[rid, drugid] = 1
            rxns[rid, rxnid] = 1

            # check to see if the drug's indications add any reactions to this report
            indids, probs = zip(*drug_indid_probs[drug_probs.index[drugid]].items())
            indid = np.random.choice(indids, 1, True, p=probs)[0]
            ind = ind_probs.index[indid][0]

            adjusted_probs = irf[indid]*rxn_probs
            adjusted_probs /= (adjusted_probs).sum()
            rxnid = np.random.choice(len(rxn_probs), 1, True, p=adjusted_probs)[0]
            inds[rid, indid] = 1
            rxns[rid, rxnid] = 1
        
        # to load use sparse.load_npz(filepath).toarray()
        sp.sparse.save_npz(os.path.join(ind_output_path, f'{i}_drugs.npz'), sp.sparse.csr_matrix(drugs))
        sp.sparse.save_npz(os.path.join(ind_output_path, f'{i}_reactions.csv'), sp.sparse.csr_matrix(rxns))
        sp.sparse.save_npz(os.path.join(ind_output_path, f'{i}_indications.csv'), sp.sparse.csr_matrix(inds))

        # np.savetxt(os.path.join(ind_output_path, f'{i}_drugs.csv'), drugs, delimiter=',')
        # np.savetxt(os.path.join(ind_output_path, f'{i}_reactions.csv'), rxns, delimiter=',')
        # np.savetxt(os.path.join(ind_output_path, f'{i}_indications.csv'), inds, delimiter=',')

        # run some basic evals on the data we just generated
        matrices = compute_matrices(drugs, rxns, num_reports=nreports)
        drug_rxn = build_dataframe(matrices, drug_probs.index, 'drug', rxn_probs.index, 'reaction', minA=1)
        drug_rxn['factor'] = [drug_rxn_factors[(d,r)] for _, (d, r) in drug_rxn[['drug', 'reaction']].iterrows()]
        drug_rxn['truth'] = [drug_rxn_truth[(d, r)] for _, (d, r) in drug_rxn[['drug', 'reaction']].iterrows()]

        matrices = compute_matrices(inds, rxns, num_reports=nreports)
        ind_rxn = build_dataframe(matrices, ind_probs.index, 'indication', rxn_probs.index, 'reaction', minA=1)

        matrices = compute_matrices(inds, drugs, num_reports=nreports)
        ind_ing = build_dataframe(matrices, ind_probs.index, 'indication', drug_probs.index, 'drug', minA=1)

        # build datafraem to look at confounding by indication
        ind_keep = ['drug', 
                'reaction', 
                'indication',
                'PRR_ing_rxn',
                'PRR_ind_ing',
                'PRR_ind_rxn']
        ind_merged = drug_rxn.merge(ind_ing, 
                                    on='drug',
                                    suffixes=('_ing_rxn', '_ind_ing')
                                ).merge(ind_rxn,
                                    on=('indication', 'reaction',),
                                    suffixes=('', '_ind_rxn'))

        ind_merged.rename(columns={'PRR': 'PRR_ind_rxn'}, inplace=True)
        ind_reduced = ind_merged[ind_keep]
        
        config[f'dataset_{i}']['IndPRR>10'] = (ind_reduced['PRR_ind_rxn'] > 10).sum()
        #print(f"Ind PRRs greater than 10: {(ind_reduced['PRR_ind_rxn'] > 10).sum()}")

        # we only care about whent the relationship between the indication and the reaction is high
        ind_rxn_high = ind_reduced[ind_reduced['PRR_ind_rxn'] > 10]
        # print(ind_rxn_high.shape)

        plt.figure(figsize=(5,3))
        plt.scatter(ind_rxn_high['PRR_ind_ing'], ind_rxn_high['PRR_ing_rxn'], alpha=0.4)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('PRR for Ingredient and Indication')
        plt.ylabel('PRR for Ingredient and Confounded Reaction')
        sns.despine()
        plt.savefig(os.path.join(ind_output_path, f'{i}_drug_ind_rxn_scatter.png'))
        plt.close()

        config[f'dataset_{i}']['ind_drug_prr_spearman'] = sp.stats.spearmanr(ind_rxn_high['PRR_ind_ing'], ind_rxn_high['PRR_ing_rxn'])

        plt.figure(figsize=(10,3))
        plt.subplot(1,2,1)
        plt.scatter(drug_rxn['factor'], drug_rxn['Tc'], alpha=0.4)
        plt.xlabel('Factor')
        # plt.xscale('log')
        plt.ylabel('Tc')
        sns.despine()
        plt.subplot(1,2,2)
        plt.scatter(drug_rxn['factor'], drug_rxn['PRR'], alpha=0.4)
        plt.xlabel('Factor')
        plt.xscale('log')
        plt.ylabel('PRR')
        plt.yscale('log')
        sns.despine()
        plt.savefig(os.path.join(ind_output_path, f'{i}_drug_rxn_factor_scatter.png'))
        plt.close()

        config[f'dataset_{i}']['Tc_spearman'] = sp.stats.spearmanr(drug_rxn['factor'], drug_rxn['Tc'])
        config[f'dataset_{i}']['PRR_spearman'] = sp.stats.spearmanr(drug_rxn['factor'], drug_rxn['PRR'])
        config[f'dataset_{i}']['Tc_AUROC'] = sklearn.metrics.roc_auc_score(drug_rxn['truth'], drug_rxn['Tc'])
        config[f'dataset_{i}']['Tc_AUPR'] = sklearn.metrics.average_precision_score(drug_rxn['truth'], drug_rxn['Tc'])
        config[f'dataset_{i}']['PRR_AUROC'] = sklearn.metrics.roc_auc_score(drug_rxn['truth'], drug_rxn['PRR'])
        config[f'dataset_{i}']['PRR_AUPR'] = sklearn.metrics.average_precision_score(drug_rxn['truth'], drug_rxn['PRR'])
        
        # print(f"Tc: {sp.stats.spearmanr(drug_rxn['factor'], drug_rxn['Tc'])}")
        # print(f"PRR: {sp.stats.spearmanr(drug_rxn['factor'], drug_rxn['PRR'])}")
        # print(f"Tc AUROC: {sklearn.metrics.roc_auc_score(drug_rxn['truth'], drug_rxn['Tc']):.3f} AUPR: {sklearn.metrics.average_precision_score(drug_rxn['truth'], drug_rxn['Tc']):.3f}")
        # print(f"PRR AUROC: {sklearn.metrics.roc_auc_score(drug_rxn['truth'], drug_rxn['PRR']):.3f} AUPR: {sklearn.metrics.average_precision_score(drug_rxn['truth'], drug_rxn['PRR']):.3f}")
    
    numpy_safe_json_dump(config, os.path.join(output_path, 'config.json'))



            

    




