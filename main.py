import warnings
from time import time

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import RBP_functions as RBP_f

column_names = [
    'id', 'tax_id', 'embl_id', 'name', 'protein_sequence', 'dna_sequence', 'org', 'host', 'protein_length', 'protein_200'
]
fibers = pd.read_csv(
    'RBP_database.csv',
    delimiter=',',
    header=0,
    names=column_names,
)

dummies = []
fibers_test_index = []
for i, item in enumerate(fibers.iloc[:, 7]):
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

# Restrict dataset to classes 0-6 (all but 'others' class -> seven-class classification)
fibers['class'] = np.asarray(dummies)
fibers = fibers.loc[fibers['class'] != 7]
dummies = [item for item in dummies if item != 7]
fibers.reset_index(drop=True, inplace=True)
print("Shape of fibers database:", fibers.shape)

# Load alignment scores
alignment_scores = np.loadtxt('RBP_alignment_scores.txt')
np.fill_diagonal(alignment_scores, 0)

# Remove duplicates from fibers, dummies and alignment_scores
_, unique_indices = np.unique(fibers['protein_sequence'], return_index=True)
remove_indices = np.setdiff1d(np.arange(len(fibers)), unique_indices)
dummies = list(np.array(dummies)[sorted(unique_indices)])
fibers = fibers.drop(index=remove_indices)
fibers.reset_index(drop=True, inplace=True)
alignment_scores = np.delete(alignment_scores, remove_indices, 0)
alignment_scores = np.delete(alignment_scores, remove_indices, 1)
print("Shape of cleaned fibers database:", fibers.shape)

# Construct features
print("Constructing features... ", end="", flush=True)
dna_list = list(fibers.iloc[:, 5])
dna_feats = RBP_f.dna_features(dna_list)

protein_list = list(fibers.iloc[:, 4])
protein_feats = RBP_f.protein_features(protein_list)

# protein features: CTD & Z-scale
extra_feats = np.zeros((len(dummies), 47))

for i, item in enumerate(protein_list):
    feature_lst = []
    feature_lst += RBP_f.CTDC(item)
    feature_lst += RBP_f.CTDT(item)
    feature_lst += RBP_f.zscale(item)
    extra_feats[i, :] = feature_lst

extra_feats_df = pd.DataFrame(extra_feats, columns=[
    'CTDC1', 'CTDC2', 'CTDC3', 'CTDT1', 'CTDT2', 'CTDT3', 'CTDT4', 'CTDT5', 'CTDT6', 'CTDT7', 'CTDT8', 'CTDT9',
    'CTDT10', 'CTDT11', 'CTDT12', 'CTDT13', 'CTDT14', 'CTDT15', 'CTDT16', 'CTDT17', 'CTDT18', 'CTDT19', 'CTDT20',
    'CTDT21', 'CTDT22', 'CTDT23', 'CTDT24', 'CTDT25', 'CTDT26', 'CTDT27', 'CTDT28', 'CTDT29', 'CTDT30', 'CTDT31',
    'CTDT32', 'CTDT33', 'CTDT34', 'CTDT35', 'CTDT36', 'CTDT37', 'CTDT38', 'CTDT39', 'Z1', 'Z2', 'Z3', 'Z4', 'Z5'
])
print("done.")


# Evaluate classifiers
random_state = 52641332
classes, class_counts = np.unique(dummies, return_counts=True)
features = pd.concat([dna_feats, protein_feats, extra_feats_df], axis=1)

# LDA (no param_grid needed)
pipe_lda = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(shrinkage='auto', solver='lsqr'))

# Logistic Regression
pipe_lr_l1 = make_pipeline(
    StandardScaler(),
    LogisticRegression(penalty='l1', class_weight='balanced', solver='saga', multi_class='multinomial', random_state=random_state)
)
from sklearn.preprocessing import FunctionTransformer
transformer = FunctionTransformer(lambda x: x)
pipe_lr_l2 = make_pipeline(
    # transformer,
    StandardScaler(),
    LogisticRegression(penalty='l2', class_weight='balanced', solver='saga', multi_class='multinomial', random_state=random_state)
)
param_grid_lr = {'logisticregression__C': np.logspace(-1, 3, 5)}

# Random Forests
pipe_rf = make_pipeline(StandardScaler(), RandomForestClassifier(class_weight='balanced', random_state=random_state))
param_grid_rf = {
    'RandomForestClassifier__n_estimators'.lower(): [10, 100, 500],
    'RandomForestClassifier__max_features'.lower(): ['sqrt', 0.1, 0.25, 0.5]
}

# Gradient Boosting
pipe_gb = make_pipeline(StandardScaler(), GradientBoostingClassifier(random_state=random_state))
param_grid_gb = {'gradientboostingclassifier__n_estimators': [10, 100, 500]}

# MLP
pipe_mlp = make_pipeline(StandardScaler(), MLPClassifier(random_state=random_state))
param_grid_mlp = {
    'MLPClassifier__hidden_layer_sizes'.lower(): [10, 100, (100, 200, 100), (200, 500, 100)],
    'MLPClassifier__max_iter'.lower(): [100, 200, 1000],
}

# Estimate performance with nested GroupKFold
performances = {}
features['class'] = np.asarray(dummies)  # add target column
features_array = features.values
X = features_array[:, :-1]
y = features_array[:, -1]

models = {
    # 'LDA': make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver='lsqr')),
    # 'logistic (L1)': pipe_lr_l1,
    # 'logistic (L2)': pipe_lr_l2,
    'MLP': pipe_mlp,
    'RF': pipe_rf,
    # 'GB': pipe_gb,
}
grids = {
    'LDA': {'lineardiscriminantanalysis__shrinkage': ['auto']},
    'logistic (L1)': param_grid_lr,
    'logistic (L2)': param_grid_lr,
    'MLP': param_grid_mlp,
    'RF': param_grid_rf,
    'GB': param_grid_gb,
}

identity_threshold = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 1]
f1_scores = np.empty((len(identity_threshold), len(models)))

for iteration, threshold in enumerate(identity_threshold):
    print(f"Computing f1 score for threshold {threshold} ...")
    # Compute groups and check result
    groups_array = RBP_f.define_groups(alignment_scores, threshold=threshold)
    similar_instances = np.unique(np.sort(np.array(np.where(alignment_scores >= threshold)).T, 1), axis=0)
    assert np.all([groups_array[i] == groups_array[j] for i, j in similar_instances]), "The groups are not correct!"

    outer_cv = StratifiedGroupKFold(n_splits=4)
    inner_cv = StratifiedGroupKFold(n_splits=4)

    t = time()
    for model_name, model in models.items():
        parameters = grids[model_name]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            perf = RBP_f.NestedGroupKFold(model=model, X=X, y=y, parameter_grid=parameters, groups=groups_array,
                                          outer_cv=outer_cv, inner_cv=inner_cv, class_weights=class_counts,
                                          scorer=make_scorer(f1_score, average='weighted', labels=classes))
        performances[model_name] = perf
        print('Model ', model_name, ' evaluated. On to the next.')
    print(f"Computed in {time() - t:.0f} seconds.")

    performance_db = pd.DataFrame(performances)
    performance_db.index = ['accuracy', 'precision', 'recall', 'f1']

    f1_scores[iteration, :] = pd.DataFrame(performances).values[-1, :]
f1_scores_db = pd.DataFrame(f1_scores)
f1_scores_db.index = identity_threshold
f1_scores_db = f1_scores_db.set_axis(models.keys(), axis=1)
f1_scores_db.to_csv("results_mlp.csv")
print(f1_scores_db)
