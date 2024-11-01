import numpy as np
import pandas as pd
import itertools
import random
from scipy.special import expit
from sklearn.preprocessing import normalize
from data_classes import get_fears_and_offsides_data
from tqdm import tqdm
import scipy.sparse as sp
from scipy.special import softmax
"""
Some questions and thoughts that aren't tied to one function

1. is covariance actually the best way to handle sparse binary data like this? do we want use something like jaccard similarity or MI to measure the associations between the various varibles?

2. i am assuming independence between the trios of variables which is probably not reasonable. 
    - should we make this better by adding joint distributions into the model? or fit some sklearn model to learn the dependencies?
    - or do want to do some PGM or other sampling technique to learn the dependencies?

3. are there some kinds of negative covar we want to capture too?

3. def check on numerical stability with avoiding division by zero but also having sparse data

3. this is definitely lacking validation


"""

class PharmacovigSyntheticGenerator:
    def __init__(self, faers_data, offsides_data):
        """
        Initialize with both FAERS and OFFSIDES data
        
        faers_data: dict containing sparse matrices:
            - reports_by_drugs
            - reports_by_reactions
            - reports_by_indications 
            - mapping *2index dictionaries
            
        offsides_data: DataFrame with columns:
            - drug_rxnorn_id
            - condition_meddra_id
            - condition_concept_name (<- has overlap with indications keys)
            - PRR
            - mean_reporting_frequency
        """
        print("Initializing PharmacovigSyntheticGenerator...")
        self.faers = faers_data
        self.offsides = offsides_data
        
        # Compute key covariance matrices
        self._compute_covariance_structure_in_faers()
        print("Initialization complete")
        
    def _compute_covariance_structure_in_faers(self):
        """
        Compute all relevant first and second order covariances

        TODO: Note that for the sake of efficency I am not itterating through the df right now to see which drugs where logged for which indications, 
        indication drug associates are just derived from co-occurence in reports - is this okay?
        """
        print("Computing covariance structures...")
        
        print("Computing first order covariances...")
        # First order covariances
        self.drug_drug_cov = self._compute_normalized_covariance(
            self.faers.reports_by_drugs, self.faers.reports_by_drugs
        )
        print("Drug-drug covariance computed")
        
        self.reaction_reaction_cov = self._compute_normalized_covariance(
            self.faers.reports_by_reactions, self.faers.reports_by_reactions
        )
        print("Reaction-reaction covariance computed")
        
        self.indication_indication_cov = self._compute_normalized_covariance(
            self.faers.reports_by_indications, self.faers.reports_by_indications
        )
        print("Indication-indication covariance computed")
        
        print("Computing second order covariances...")
        # Second order covariances
        self.drug_drug_reaction_cov = self._compute_triple_covariance(
            self.faers.reports_by_drugs,
            self.faers.reports_by_drugs,
            self.faers.reports_by_reactions
        )
        print("Drug-drug-reaction covariance computed")
        
        self.drug_indication_reaction_cov = self._compute_triple_covariance(
            self.faers.reports_by_drugs,
            self.faers.reports_by_indications,
            self.faers.reports_by_reactions
        )
        print("Drug-indication-reaction covariance computed")
        
        self.indication_indication_reaction_cov = self._compute_triple_covariance(
            self.faers.reports_by_indications,
            self.faers.reports_by_indications,
            self.faers.reports_by_reactions
        )
        print("Indication-indication-reaction covariance computed")
        print("All covariance structures computed")

    def _compute_normalized_covariance(self, X, Y):
        """
        Compute normalized covariance between two sparse matrices using efficient sparse operations.
        """
        # Co-occurrence counts as a sparse matrix
        co_occurrence = X.T.dot(Y)
        
        # Marginal sums (1D arrays)
        x_sum = np.asarray(X.sum(axis=0)).ravel()
        y_sum = np.asarray(Y.sum(axis=0)).ravel()
        n = float(X.shape[0])
        
        # Expected co-occurrence under independence
        expected = x_sum[:, None] * y_sum[None, :] / n
        
        # Compute covariance only for non-zero co-occurrences
        coo = co_occurrence.tocoo()
        cov_data = (coo.data - expected[coo.row, coo.col]) / n
        
        # Compute standard deviations
        std_X = np.sqrt(np.maximum((x_sum * (n - x_sum)) / (n * n), 1e-10))
        std_Y = np.sqrt(np.maximum((y_sum * (n - y_sum)) / (n * n), 1e-10))
        
        # Compute correlations only for non-zero covariances
        corr_data = cov_data / (std_X[coo.row] * std_Y[coo.col])
        
        # Build sparse correlation matrix using scipy.sparse instead of sparse
        corr = sp.coo_matrix((corr_data, (coo.row, coo.col)), shape=co_occurrence.shape)
        
        # Optionally convert to CSR format for faster arithmetic operations
        return corr.tocsr()


    def _compute_triple_covariance(self, X, Y, Z):
        """
        Compute three-way covariance tensor using efficient sparse operations.
        """
        n = float(X.shape[0])
        
        # Compute co-occurrences
        XY = X.T.dot(Y)  # Sparse matrix
        YZ = Y.multiply(Z)  # Element-wise multiplication
        XYZ = X.T.dot(YZ)  # Sparse matrix
        
        # Marginal sums
        x_sum = np.asarray(X.sum(axis=0)).ravel()
        y_sum = np.asarray(Y.sum(axis=0)).ravel()
        z_sum = np.asarray(Z.sum(axis=0)).ravel()
        
        # Expected counts under independence
        # For each non-zero in XYZ, compute expected value
        XYZ_coo = XYZ.tocoo()
        expected_data = (XY[XYZ_coo.row, XYZ_coo.col] * z_sum[XYZ_coo.data.astype(int)]) / n
        
        # Compute covariance
        cov_data = (XYZ_coo.data - expected_data) / n
        
        # Handle negative covariances if desired
        cov_data = np.maximum(cov_data, 0)
        
        # Normalize to probabilities using softmax
        # Group by (row, col) pairs
        from collections import defaultdict
        tensor = defaultdict(list)
        for idx, (i, j, cov) in enumerate(zip(XYZ_coo.row, XYZ_coo.col, cov_data)):
            tensor[(i, j)].append((XYZ_coo.data[idx], cov))
        
        # Build the tensor
        tensor_dict = {}
        for (i, j), items in tqdm(tensor.items(), desc="Building 3 way tensor"):
            indices, cov_values = zip(*items)
            # Apply softmax to cov_values
            probs = np.exp(cov_values - np.max(cov_values))
            probs /= probs.sum()
            tensor_dict[(i, j)] = (indices, probs)
    
        return tensor_dict

    def _compute_jaccard_similarity(self, X, Y):
        """
        Compute Jaccard similarity between two sparse binary matrices X and Y.
        """
        # Co-occurrence counts as a sparse matrix
        intersection = X.T.dot(Y)
        
        # Marginal sums
        x_sum = np.asarray(X.sum(axis=0)).ravel()
        y_sum = np.asarray(Y.sum(axis=0)).ravel()
        
        # Compute union counts
        x_sum_matrix = sp.csr_matrix(x_sum)
        y_sum_matrix = sp.csr_matrix(y_sum)
        union = x_sum_matrix.T + y_sum_matrix - intersection
        
        # Element-wise division
        with np.errstate(divide='ignore', invalid='ignore'):
            jaccard = intersection.multiply(1 / union)
            jaccard.data[np.isnan(jaccard.data)] = 0
        return jaccard.tocsr()




    def generate_true_relationships_offsides(self):
        """
        Generate true drug-reaction relationships with strength scores based on OFFSIDES 
        
        could also be done with onsides addtionally or instead, but for now this should be enough to evaluate the structure of this code 
        and sholuldnt be too hard to hot swap in onsides as the basis for  true relationships

        function returns relationships dict with keys as tuples of drug and reaction indices 
        (indicies corresponding to self.faers.drug2idx and self.faers.reaction2idx)
        values are the PRR, filtered for PRR > 2 - hopefully this counts as strong enough weak supervision?
        also 2 is pretty arbitarty here, if we stick with this we should do some valildation on different thresholds

        """
        print("Generating true relationships from OFFSIDES...")
        relationships = {}
        
        # Map OFFSIDES concepts to our vocabulary
        for _, row in tqdm(self.offsides.iterrows(), total=len(self.offsides), desc="Processing OFFSIDES data"):
            if row.PRR > 2:
                drug_idx = self.faers.drug2idx.get(row.drug_rxnorn_id)
                reaction_idx = self.faers.reaction2idx.get(row.condition_meddra_id)
                
                if drug_idx is not None and reaction_idx is not None:
                    # Use PRR as effect size
                    relationships[(drug_idx, reaction_idx)] = row.PRR
        
        print(f"Generated {len(relationships)} true relationships")
        return relationships
    
    #TODO implement onsides derived true relationships - effect sizes could come from ehr

    def generate_report(self, true_relationships):
        """
        Generate a single report with both clean and confounded versions.
        right now, I have clean data including only the primary drug and its direct reactions. 
        TODO how do we want to expand this? multiple drugs and their reactions and all asssociated indications? 
        realzing I am not clear on where the line between fleshed out clean data and confounded data should be drawn here
        Confounded data now includes additional drugs, indications, and reactions but any of these attributes can easily be moved to the clean data.
        """
        print("\nGenerating synthetic report...")
        
        # Sample primary drug at random, weighted by drug frequency
        drug_freqs = np.array(self.faers.reports_by_drugs.sum(axis=0)).flatten()
        drug_probs = drug_freqs / drug_freqs.sum()
        primary_drug = np.random.choice(self.faers.reports_by_drugs.shape[1], p=drug_probs)
        print(f"Selected primary drug index: {primary_drug}")
        
        # Generate clean reactions using true reaction relationships for the primary drug
        clean_reactions = set()
        print("Generating clean reactions...")
        for (d, r), effect in tqdm(true_relationships.items(), desc="Processing true relationships"):
            if d == primary_drug and np.random.random() < expit(effect):
                clean_reactions.add(r)
        print(f"Generated {len(clean_reactions)} clean reactions")
        
        # Prepare clean data
        clean_data = {
            'drugs': [primary_drug],
            'reactions': list(clean_reactions)
        }
        
        print("\nGenerating confounded data...")
        # Generate confounded data

        # Sample co-prescribed drugs using drug-drug covariance
        drug_cov = self.drug_drug_cov[primary_drug]
        drug_cov = np.maximum(drug_cov, 0)
        co_drug_probs = softmax(drug_cov)
        
        # TODO I think this is a reasonable distribution but lmk, not something I have a lot of conviction on
        n_co_drugs = np.random.poisson(2)  # Average of 2 co-prescribed drugs
        co_drugs = np.random.choice(
            len(co_drug_probs),
            size=n_co_drugs,
            p=co_drug_probs,
            replace=False
        )
        drugs = [primary_drug] + list(co_drugs)
        print(f"Added {len(co_drugs)} co-prescribed drugs")
        
        # Generate reactions from true relationships for all drugs
        confounded_reactions = set(clean_reactions)
        print("Adding reactions from co-prescribed drugs...")
        for drug in tqdm(co_drugs, desc="Processing co-prescribed drugs"):
            for (d, r), effect in true_relationships.items():
                if d == drug and np.random.random() < expit(effect):
                    confounded_reactions.add(r)
        
        # Sample indications using drug-indication covariance
        print("\nGenerating indications...")
        indications = set()
        for drug in tqdm(drugs, desc="Processing drug indications"):
            drug_ind_cov = self.drug_indication_cov[drug]
            drug_ind_cov = np.maximum(drug_ind_cov, 0)
            if drug_ind_cov.sum() > 0:
                ind_probs = softmax(drug_ind_cov)
                sampled_inds = np.random.choice(
                    len(ind_probs),
                    size=1,  
                    p=ind_probs,
                    replace=False
                )
                indications.update(sampled_inds)
        print(f"Generated {len(indications)} indications")
        
        # Add reactions arising from indications
        print("\nAdding reactions from indications...")
        for ind in tqdm(indications, desc="Processing indication reactions"):
            ind_rxn_cov = self.indication_reaction_cov[ind]
            ind_rxn_cov = np.maximum(ind_rxn_cov, 0)
            if ind_rxn_cov.sum() > 0:
                rxn_probs = softmax(ind_rxn_cov)
                n_reactions = np.random.poisson(1)
                reactions = np.random.choice(
                    len(rxn_probs),
                    size=n_reactions,
                    p=rxn_probs,
                    replace=False
                )
                confounded_reactions.update(reactions)
        
        # Add reactions from drug-drug interactions
        print("\nAdding reactions from drug-drug interactions...")
        for i, drug1 in enumerate(drugs):
            for drug2 in drugs[i+1:]:
                dd_rxn_cov = self.drug_drug_reaction_cov[drug1, drug2]
                dd_rxn_cov = np.maximum(dd_rxn_cov, 0)
                if dd_rxn_cov.sum() > 0:
                    rxn_probs = softmax(dd_rxn_cov)
                    n_reactions = np.random.poisson(0.5)
                    reactions = np.random.choice(
                        len(rxn_probs),
                        size=n_reactions,
                        p=rxn_probs,
                        replace=False
                    )
                    confounded_reactions.update(reactions)
        print(f"Final number of confounded reactions: {len(confounded_reactions)}")
        
        # Prepare confounded data
        confounded_data = {
            'drugs': drugs,
            'reactions': list(confounded_reactions),
            'indications': list(indications)
        }
        
        print("Report generation complete")
        return {
            'clean': clean_data,
            'confounded': confounded_data
        }
    

if __name__ == '__main__':
    print("Loading FAERS and OFFSIDES data...")
    faers_data, offsides_data = get_fears_and_offsides_data()
    generator = PharmacovigSyntheticGenerator(faers_data, offsides_data)
    true_relationships = generator.generate_true_relationships_offsides()
    report = generator.generate_report(true_relationships)
    print("\nGenerated report:")
    print(report)






    # def generate_report(self, true_relationships):
    #     """
    #     Generate a single report with both clean and confounded versions
    #     """
    #     # weight this by drug frequency
    #     drug_freqs = self.faers.reports_by_drugs.sum(axis=0).A1
    #     primary_drug = np.random.choice(self.faers.reports_by_drugs.shape[1], p=normalize(drug_freqs))
        
    #     # Sample co-prescribed drugs using drug-drug covariance
    #     drug_probs = self.drug_drug_cov[primary_drug]
    #     #SHOULD I ADD SOFTMAX HERE?
    #     n_drugs = np.random.poisson(2)  # Average of 2 additional drugs
    #     other_drugs = np.random.choice(
    #         len(drug_probs),
    #         size=n_drugs,
    #         p=normalize(np.maximum(drug_probs, 0)),
    #         replace=False
    #     )
    #     drugs = [primary_drug] + list(other_drugs)
        
    #     # Generate clean reactions using true relationships
    #     clean_reactions = set()
    #     for drug in drugs:
    #         # Get reactions from OFFSIDES relationships
    #         for (d, r), effect in true_relationships.items():
    #             #TODO could do this for onsides but maybe effect would be random?
    #             if d == drug and np.random.random() < expit(effect):
    #                 clean_reactions.add(r)
            
    #         # Sample additional reactions using drug-drug-reaction tensor
    #         #TODO should this be in the clean data or the confounded data?
    #         for other_drug in drugs:
    #             if other_drug != drug:
    #                 rxn_probs = self.drug_drug_reaction_cov[drug, other_drug]
    #                 n_reactions = np.random.poisson(1)
    #                 reactions = np.random.choice(
    #                     len(rxn_probs),
    #                     size=n_reactions,
    #                     p=normalize(rxn_probs),
    #                     replace=False
    #                 )
    #                 clean_reactions.update(reactions)
        
    #     # Sample indications using drug-indication covariances
    #     indications = set()
    #     for drug in drugs:
    #         ind_probs = self.drug_indication_reaction_cov[drug].sum(axis=1)
    #         n_indications = np.random.poisson(1)
    #         drug_indications = np.random.choice(
    #             len(ind_probs),
    #             size=n_indications,
    #             p=normalize(np.maximum(ind_probs, 0)),
    #             replace=False
    #         )
    #         indications.update(drug_indications)
        
        
    #         # Sample co-occurring indications
    #         #TODO should this be in the clean data or the confounded data? do we need this to be more involved in the other sampling?
    #         #TODO is there some more cyclical pruning way we could this or use monte carlo some how?
    #         for ind in drug_indications:
    #             coind_probs = self.indication_indication_cov[ind]
    #             n_coinds = np.random.poisson(0.5)
    #             coindications = np.random.choice(
    #                 len(coind_probs),
    #                 size=n_coinds,
    #                 p=normalize(np.maximum(coind_probs, 0)),
    #                 replace=False
    #             )
    #             indications.update(coindications)
        
    #     # Generate confounded reactions
    #     confounded_reactions = clean_reactions.copy()
        
    #     # Add reactions from indications using indication-reaction patterns
    #     # what else should go here?
    #     for indication in indications:
    #         # Direct indication effects
    #         rxn_probs = self.drug_indication_reaction_cov[:, indication, :].sum(axis=0)
    #         n_reactions = np.random.poisson(1)
    #         reactions = np.random.choice(
    #             len(rxn_probs),
    #             size=n_reactions,
    #             p=normalize(np.maximum(rxn_probs, 0)),
    #             replace=False
    #         )
    #         confounded_reactions.update(reactions)
            
    #         # Indication pair effects
    #         for other_ind in indications:
    #             if other_ind != indication:
    #                 pair_rxn_probs = self.indication_indication_reaction_cov[indication, other_ind]
    #                 n_reactions = np.random.poisson(0.5)
    #                 reactions = np.random.choice(
    #                     len(pair_rxn_probs),
    #                     size=n_reactions,
    #                     p=normalize(np.maximum(pair_rxn_probs, 0)),
    #                     replace=False
    #                 )
    #                 confounded_reactions.update(reactions)
        
    #     return {
    #         'clean': {
    #             'drugs': list(drugs),
    #             'reactions': list(clean_reactions)
    #         },
    #         'confounded': {
    #             'drugs': list(drugs),
    #             'reactions': list(confounded_reactions),
    #             'indications': list(indications)
    #         }
    #     }


