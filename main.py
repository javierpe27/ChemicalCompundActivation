import pandas as pd
import numpy as np
import rdkit
from rdkit import Chem
import rdkit.Chem.rdMolDescriptors as d
import rdkit.Chem.Fragments as f
import rdkit.Chem.Lipinski as l
from rdkit.Chem import AllChem
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.feature_selection import SelectKBest, chi2


train_fn = "training_smiles.csv"
test_fn = "test_smiles.csv"


def load_data(fn, **kwargs):
    return pd.read_csv(fn, **kwargs)



train = pd.read_csv(train_fn, **{"index_col": 0})
test = pd.read_csv(test_fn, **{"index_col": 0})

print("Look at the data:\n", train.head)

print("Columns:", ", ".join(train.columns))

print("Number of observations:", len(train))

print("Class distribution:\n", train.groupby("ACTIVE").count())
#Imbalanced dataset

def get_mols(smiles):
    return smiles.apply(lambda x: Chem.MolFromSmiles(x))


train["mols"] = get_mols(train["SMILES"])
test["mols"] = get_mols(test["SMILES"])


####################FEATURE EXTRACTION########################

lib_funs = {
d: dir(d),
f: dir(f),
l: dir(l)
}

lib_funs = {key: [f for f in value if f[0] != "_"] for key, value in lib_funs.items()}

def add_features(df, lib_funs):
    for lib, funs in lib_funs.items():
        for fun in funs:
            if not fun.startswith("_"):
                try:
                    method = getattr(lib, fun)
                    values = df.mols.apply(lambda x: method(x))
                    if (values.dtype == np.float64) | (values.dtype == np.int64):
                        df[fun] = values
                        print("%s addedd successfully\n\n" % (fun))
                    else:
                        print("%s not added due to type:%s\n\n" % (fun, values.dtype))
                except Exception as e:
                    print("Could not process function %s.%s due to:\n%s\n Continuouing\n\n" % (lib.__name__, fun, e))
    return df


train = add_features(train, lib_funs)
test = add_features(test, lib_funs)

if len(train.columns) - 1 != len(test.columns):
    raise ValueError("Number of columns differ in training - excluding label - (%d) and test set (%d)" % (
    len(train.columns) - 1, len(test.columns)))



train.to_csv("train_features.csv", index=None)
test.to_csv("test_features.csv", index=None)

train = pd.read_csv("train_features.csv")
test = pd.read_csv("test_features.csv")

def getClassShare(df, class_column):
    return list(df.groupby(class_column).size().items())


def sample(df, class_column, over_sample=True):
    """
    Consider using this library instead: https://imbalanced-learn.readthedocs.io/en/stable/under_sampling.html
    """
    class_share = getClassShare(df, class_column)
    tmp_df = pd.DataFrame([], columns=df.columns)
    target_count = max(class_share, key=lambda x: x[1])[1] if over_sample else min(class_share, key=lambda x: x[1])[1]
    for class_value, class_count in class_share:
        if class_count == target_count:
            tmp_df = tmp_df.append(df[df[class_column] == class_value])
        else:
            tmp_df = tmp_df.append(df[df[class_column] == class_value].sample(target_count, replace=over_sample))
    return tmp_df



X = train.drop(["SMILES", "ACTIVE", "mols"], axis = 1)
Y = train["ACTIVE"].astype(np.int64)



####################FEATURE CLEANING BASED ON CORRELATION#########################
corr = X.corr()
#sns.heatmap(corr)
#plt.show()


threshold = 0.7
columns = np.full((corr.shape[0],), True, dtype=bool)
for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        if corr.iloc[i,j] >= threshold:
            if columns[j]:
                columns[j] = False

selected_columns = X.columns[columns]
newdataSet = X[selected_columns]

sns.heatmap(newdataSet.corr())
plt.show()

############ DROP COLUMNS WITH CONSTANT VALUES ###################

for column in newdataSet.columns:
    if len(np.unique(newdataSet[column])) == 1:
        newdataSet = newdataSet.drop([column], axis=1)
sns.heatmap(newdataSet.corr())


############## DECISION TREE - FEATURE IMPORTANCE ####################
############## DROP COLUMNS IF THEY ARE NOT IMPORTANT ################


dt_reduced = tree.DecisionTreeClassifier()
dt_reduced = dt_reduced.fit(newdataSet, Y)

fi = dt_reduced.feature_importances_
fi_names = {feature_name : fi_value for feature_name, fi_value in zip(newdataSet.columns,fi)}
order_fi = sorted(fi_names.items(), key = lambda kv : kv[1], reverse=True)
plt.bar([x[0] for x in order_fi], [x[1] for x in order_fi])
plt.xticks([x[0] for x in order_fi], rotation='vertical')
plt.show()

for column, value in order_fi:
    if value< 0.005:
        print(column, value)
        ReducedNewdataSet = newdataSet.drop([column], axis=1)



#################### Randomized search VS Grid Search for RANDOM FOREST#########################


# Number of trees in random forest
n_estimators = [50, 200, 300,500, 1000]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [10, 50, 100, 150, 200]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

rfGrid = RandomForestClassifier()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 4, verbose=2, n_jobs = -1)
rf_random.fit(newdataSet, Y)
print(rf_random.best_params_)
bestEstimator = rf_random.best_estimator_

#applyCrossValidation(newdataSet, Y, bestEstimator,5)

from sklearn.model_selection import GridSearchCV
baseRF = RandomForestClassifier()
grid_search = GridSearchCV (estimator = baseRF, param_grid = random_grid, cv = 3, n_jobs = -1, verbose = 2)
grid_search.fit(newdataSet, Y)
best_grid = grid_search.best_estimator_
applyCrossValidation(newdataSet, Y,best_grid,5 )



def split_x_y(df):
    return (df.drop("ACTIVE", axis = 1).astype(np.float64), df["ACTIVE"].astype(np.int64))


from sklearn.model_selection import KFold, StratifiedKFold

def newCrossValidation(data, k, model):
    results = {
        "test_accuracy": [],
        "test_roc_auc": []
    }
    kf = StratifiedKFold(n_splits=k)
    x = data.drop("ACTIVE", axis = 1)
    y = data["ACTIVE"].astype(np.int64)
    for train_index, test_index in kf.split(x, y):
        train_set = sample(data.iloc[train_index], "ACTIVE", False)
        test_set = data.iloc[test_index]
        x_train, y_train = split_x_y(train_set)
        x_test, y_test = split_x_y(test_set)
        model = model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        results["test_accuracy"].append(metrics.accuracy_score(y_test, predictions))
        results["test_roc_auc"].append(metrics.roc_auc_score(y_test, predictions))
    return results


############## Gridsearch #################3
import itertools


def createParameters(params):
	keys = params.keys()
	vals = params.values()
	tmp = []
	for instance in itertools.product(*vals):
		tmp.append(dict(zip(keys, instance)))
	return tmp


def gridsearch(modelClass, params):
    tmp = {}
    for param in createParameters(params):
        print(str(param))
        model = modelClass(**param)
        cv_results = newCrossValidation(train[np.append(selected_columns, ["ACTIVE"])], 5, model)
        tmp[str(param)] = {key: np.mean(values) for key, values in cv_results.items()}
    return tmp


params = {
"n_estimators": [50, 500],
"max_features": ["auto", "sqrt"],
"max_depth": [10, 100],
"min_samples_split": [2, 10],
"min_samples_leaf": [1, 5],
"bootstrap": [True, False]
}

gridsearch_model = RandomForestClassifier

grid_search_results = gridsearch(gridsearch_model, params)

max(grid_search_results.items(), key = lambda x: x[1]["test_roc_auc"])



#knn
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn_results = newCrossValidation(train[np.append(selected_columns, ["ACTIVE"])], 5, knn)

knn_results

#Logistic regression
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr_results = newCrossValidation(train[np.append(selected_columns, ["ACTIVE"])], 5, lr)

lr_results


#decision tree
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
dt_results = newCrossValidation(train[np.append(selected_columns, ["ACTIVE"])], 5, dt)

dt_results

#RF
rf = RandomForestClassifier()
rf_results = newCrossValidation(train[np.append(selected_columns, ["ACTIVE"])], 5, rf)

rf_results

#MLP
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier()
mlp_results = newCrossValidation(train[np.append(selected_columns, ["ACTIVE"])], 5, mlp)

mlp_results


#xgboost
from xgboost import XGBClassifier

xgb = XGBClassifier()
xgb_results = newCrossValidation(train[np.append(selected_columns, ["ACTIVE"])], 5, xgb)

xgb_results

#Tuned RF

rf_tuned_results = newCrossValidation(train[np.append(selected_columns, ["ACTIVE"])], 5, bestEstimator)


#PLOT

methods = ["KNN", "Logistic Regresion", "Decision Tree", "Random Forest", "MLP", "XGBoost","Random Forest Tunned"]
Accuracy = []
AUC =[]

Accuracy.append(np.mean(knn_results["test_accuracy"]))
AUC.append(np.mean(knn_results["test_roc_auc"]))

Accuracy.append(np.mean(lr_results["test_accuracy"]))
AUC.append(np.mean(lr_results["test_roc_auc"]))

Accuracy.append(np.mean(dt_results["test_accuracy"]))
AUC.append(np.mean(dt_results["test_roc_auc"]))

Accuracy.append(np.mean(rf_results["test_accuracy"]))
AUC.append(np.mean(rf_results["test_roc_auc"]))

Accuracy.append(np.mean(mlp_results["test_accuracy"]))
AUC.append(np.mean(mlp_results["test_roc_auc"]))

Accuracy.append(np.mean(xgb_results["test_accuracy"]))
AUC.append(np.mean(xgb_results["test_roc_auc"]))

Accuracy.append(np.mean(rf_tuned_results["test_accuracy"]))
AUC.append(np.mean(rf_tuned_results["test_roc_auc"]))


x = np.arange(len(methods))  # the label locations
width = 0.35  # the width of the bars



fig, ax = plt.subplots(figsize = (15, 7))
rects1 = ax.bar(x - width/2, Accuracy, width, label="Accuracy")
rects2 = ax.bar(x + width/2, AUC, width, label="AUC")

ax.set_ylabel("Accuracy/AUC")
ax.set_title("")
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.legend()

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate("", xy = (rect.get_x() + rect.get_width() / 2, height))

autolabel(rects1)
autolabel(rects2)
fig.tight_layout()

plt.show()
############# Fingerpint util functions ################3

def ExplicitBitVect_to_NumpyArray(bitvector):
    """
    Convert ExplicitBitVect object to Numpy array
    Source: Elton, 2017
    """
    bitstring = bitvector.ToBitString()
    intmap = map(int, bitstring)
    return np.array(list(intmap))


def convert_fingerprint(obj):
    """
    Convert fingerprints to numpy arrays
    Source: Elton, 2017
    """
    if isinstance(obj, tuple):
        return np.array(list(obj[0]))
    if isinstance(obj, rdkit.DataStructs.cDataStructs.ExplicitBitVect):
        return ExplicitBitVect_to_NumpyArray(obj)
    if isinstance(obj, rdkit.DataStructs.cDataStructs.IntSparseIntVect):
        return np.array(list(obj))

# Fingerpints to test

fingerprint_funcs = {
    "AtomPairFP": lambda x, nBits=1024: AllChem.GetHashedAtomPairFingerprintAsBitVect(x, nBits=nBits),
    "TopologicalTorsionFP": lambda x, nBits=1024: AllChem.GetHashedTopologicalTorsionFingerprintAsBitVect(x, nBits=nBits),
    "MACCSKeysFP": lambda x, nBits=None: AllChem.GetMACCSKeysFingerprint(x),
    "MorganFP": lambda x, nBits=1024: AllChem.GetMorganFingerprintAsBitVect(x, radius=2, nBits=nBits)
}

# Fingerprint test

train_sample = sample(train, "ACTIVE", False)

rf = RandomForestClassifier(n_estimators=100)

train = pd.read_csv("train_features.csv")
test = pd.read_csv("test_features.csv")

train["mols"] = get_mols(train["SMILES"])
test["mols"] = get_mols(test["SMILES"])


k = 5
fp_param_results = pd.DataFrame([],
                                columns=["feature", "nBits", "accuracy", "roc_auc"])
for nBits in [1024, 2048, 4096]:
    for fp_name, fp_func in fingerprint_funcs.items():
        print(fp_name)
        totrain = train.mols.apply(lambda x: pd.Series(convert_fingerprint(fp_func(x, nBits))))
        Y = train["ACTIVE"].astype(np.int64)
        totrain["ACTIVE"] = Y
        results = newCrossValidation(totrain, k, rf)
        tmp = pd.DataFrame({
            "accuracy": results["test_accuracy"],
             "roc": results["test_roc_auc"]}
            #[results["test_accuracy"], results["test_roc_auc"]],
            )
        tmp["feature"] = fp_name
        tmp["nBits"] = np.nan if fp_name == "MACCSKeysFP" else nBits
        fp_param_results = fp_param_results.append(tmp)


X = train.drop(["SMILES", "ACTIVE", "mols"], axis = 1)
Y = train["ACTIVE"].astype(np.int64)
X["ACTIVE"] = Y
results = newCrossValidation(X, k, rf)
tmp = pd.DataFrame({
            "accuracy": results["test_accuracy"],
             "roc": results["test_roc_auc"]}
            #[results["test_accuracy"], results["test_roc_auc"]],
            )


tmp["feature"] = "other"
tmp["nBits"] = np.nan
fp_param_results = fp_param_results.append(tmp)

aggregate_results = fp_param_results.groupby(["nBits", "feature"]).agg(["mean", "std"])

fig, ax = plt.subplots(figsize=(15, 7))
aggregate_results["roc"]["mean"].unstack().plot(ax=ax)
ax.set_ylabel("AUC")
ax.set_xlabel("nBits")
ax.set_title("Fingerprint - number of bits comparison")
plt.show()


colors = ["red", "green", "blue", "yellow"]
fig, ax = plt.subplots(figsize=(10, 7))
for i, fp in enumerate(set([x[1] for x in aggregate_results.index.values])):
    tmp = aggregate_results["accuracy"]["mean"].reset_index()
    sub_tmp = tmp[tmp["feature"] == fp]
    ax.scatter(sub_tmp["nBits"], sub_tmp["mean"], c=colors[i], label=fp)

ax.set_xlabel("nBits")
ax.set_ylabel("AUC")
ax.set_title("Random Forest CV AUC with different nBits parameters")
plt.legend(loc = "lower right")
plt.show()

chosen_nbits = 4096
aggregate_results = fp_param_results[(fp_param_results["nBits"] == chosen_nbits) | (fp_param_results["nBits"].isna())].groupby("feature").agg(["min", "mean", "max"])
print(aggregate_results["roc"])

aggregate_results = fp_param_results.groupby("feature").agg(["min", "mean", "max"])
print(aggregate_results["roc"])


fps = train.mols.apply(lambda x: pd.Series(convert_fingerprint(AllChem.GetMACCSKeysFingerprint(x))))
train_selected_columns = list(selected_columns)
train_selected_columns.append("ACTIVE")
totrain = pd.concat([train[train_selected_columns], fps], axis = 1)
rf = RandomForestClassifier(100)
results_all = newCrossValidation(totrain, 5, rf)

results_all

undersampled_df = sample(totrain, "ACTIVE", False)

finalModel = rf.fit(undersampled_df.drop("ACTIVE", axis = 1), undersampled_df["ACTIVE"].astype(np.int64))
test_fps = test.mols.apply(lambda x: pd.Series(convert_fingerprint(AllChem.GetMACCSKeysFingerprint(x))))
totest = pd.concat([test[selected_columns], test_fps], axis = 1)
predictions = rf.predict_proba(totest)
pd.DataFrame(predictions[:,1]).to_csv("13.txt", index=None, header = False)
plt.hist(predictions[:,1])

