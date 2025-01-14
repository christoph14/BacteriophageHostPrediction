import warnings
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import RBP_functions as RBP_f
from utils import construct_features, construct_protein_embeddings

column_names = [
    'id', 'tax_id', 'embl_id', 'name', 'protein_sequence', 'dna_sequence', 'org', 'host', 'protein_length', 'protein_200'
]
rbp_database = pd.read_csv('RBP_database.csv', delimiter=',', header=0, names=column_names)

dummies = []
fibers_test_index = []
# Select host
for i, item in enumerate(rbp_database['host']):
    if 'Staphylococcus aureus' in item:
        dummies.append(0)
    elif 'Klebsiella pneumoniae' in item:
        dummies.append(1)
    elif 'Acinetobacter baumannii' in item:
        dummies.append(2)
    elif 'Pseudomonas aeruginosa' in item:
        dummies.append(3)
    elif 'Escherichia coli' in item:
        dummies.append(4)
    elif 'Salmonella enterica' in item:
        dummies.append(5)
    elif 'Clostridium difficile' in item:
        dummies.append(6)
    else:
        dummies.append(7)
        fibers_test_index.append(i)

# Restrict dataset to classes 0-6 (all bot 'others' class -> seven-class classification)
rbp_database['class'] = np.asarray(dummies)
rbp_database = rbp_database.loc[rbp_database['class'] != 7]
dummies = [item for item in dummies if item != 7]
rbp_database.reset_index(drop=True, inplace=True)
print("Shape of RBP database:", rbp_database.shape)

# Load alignment scores
alignment_scores = np.loadtxt('RBP_alignmentscores.txt')
np.fill_diagonal(alignment_scores, 0)

# Remove duplicates from fibers, dummies and alignment scores
_, unique_indices = np.unique(rbp_database['protein_sequence'], return_index=True)
remove_indices = np.setdiff1d(np.arange(len(rbp_database)), unique_indices)
dummies = list(np.array(dummies)[sorted(unique_indices)])
rbp_database = rbp_database.drop(index=remove_indices)
rbp_database.reset_index(drop=True, inplace=True)
alignment_scores = np.delete(alignment_scores, remove_indices, 0)
alignment_scores = np.delete(alignment_scores, remove_indices, 1)
print("Shape of cleaned RBP database:", rbp_database.shape)

# TODO REMOVE (select only short proteins for faster computation)
small_idx = [len(x) < 300 for x in rbp_database['protein_sequence']]
rbp_database = rbp_database[small_idx]
rbp_database.reset_index()
dummies = list(np.array(dummies)[small_idx])
alignment_scores = alignment_scores[small_idx][:, small_idx]
print("Number of short proteins:", len(rbp_database))

# Construct features
print("Constructing features ...")
# features = construct_features(fibers)
features = construct_protein_embeddings(rbp_database)
print('done')

# Evaluate classifier
random_state = 52641332
classes, class_counts = np.unique(dummies, return_counts=True)

# LDA (no param_grid needed)
pipe_lda = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(shrinkage='auto', solver='lsqr'))

# Logistic regression
pipe_lr_l1 = make_pipeline(StandardScaler(), LogisticRegression(penalty='l1', class_weight='balanced', solver='saga', random_state=random_state))
param_grid_lr = {'logisticregression__C': np.logspace(-1, 3, 5)}

# Random Forests
pipe_rf = make_pipeline(StandardScaler(), RandomForestClassifier(class_weight='balanced', random_state=random_state))
param_grid_rf = {
    'RandomForestClassifier__n_estimators'.lower(): [10, 100, 500],
    'RandomForestClassifier__max_features'.lower(): [800, 1000]
}

# TODO maybe add methods

# Estimate performance with nested GroupKFold
performances = {}
features['class'] = np.asarray(dummies)  # add target column
X = features.values[:, :-1]
y = features.values[:, -1]

models = {
    # 'LDA': pipe_lda,
    # 'logistic (L1)': pipe_lr_l1,
    'RF': pipe_rf,
}

grids = {
    'LDA': {'lineardiscriminantanalysis__shrinkage': ['auto']},
    'logistic (L1)': param_grid_lr,
    'RF': param_grid_rf,
}

identity_threshold = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1]
f1_scores = np.empty((len(identity_threshold), len(models)))

COMPUTE = True

for iteration, threshold in enumerate(identity_threshold):
    if not COMPUTE:
        continue
    print(f"Computing f1 score for threshold {threshold} ...")
    # Compute groups and check result
    groups_array = RBP_f.define_groups(alignment_scores, threshold)
    similar_instances = np.unique(np.sort(np.array(np.where(alignment_scores >= threshold)).T, 1), axis=0)
    assert np.all([groups_array[i] == groups_array[j] for i, j in similar_instances]), "The groups are not correct"

    outer_cv = StratifiedGroupKFold(n_splits=4)
    inner_cv = StratifiedGroupKFold(n_splits=4)

    t = time()
    for model_name, model in models.items():
        parameters = grids[model_name]
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ConvergenceWarning)
            perf = RBP_f.NestedGroupKFold(model, X, y, parameters, groups_array, class_counts, scorer=make_scorer(f1_score, average='weighted', labels=classes), inner_cv=inner_cv, outer_cv=outer_cv)
        performances[model_name] = perf
        # print(f"Model {model_name} evaluated.")
    print(f"Computed in {time() - t:.0f} seconds")

    performance_db = pd.DataFrame(performances)
    performance_db.index = ['accuracy', 'precision', 'recall', 'f1']

    f1_scores[iteration, :] = pd.DataFrame(performances).values[-1, :]

if COMPUTE:
    f1_scores_db = pd.DataFrame(f1_scores)
    f1_scores_db.index = identity_threshold
    f1_scores_db = f1_scores_db.set_axis(models.keys(), axis=1)
    f1_scores_db.to_csv('results_megaDNA.csv')
else:
    f1_scores_db = pd.read_csv('results_megaDNA.csv', index_col=0)
print(f1_scores_db)

sns.relplot(f1_scores_db, kind='line')
plt.show()
