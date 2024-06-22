from pathlib import Path
import pandas as pd
import numpy as np
import datetime
import scipy
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from BayesianOpt import BayesOpt
from Genetic import Genetic
from settings import domain, mcnemar_test, get_contingency_table
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import interp
from sklearn.metrics import matthews_corrcoef, roc_curve, auc

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH_NASA = ROOT_DIR / 'datasets' / 'nasa.csv'
# https://www.kaggle.com/datasets/ritwikb3/heart-disease-statlog?resource=download
DATASET_PATH_HEART = ROOT_DIR / 'datasets' / 'Heart_disease_statlog.csv'
# https://www.kaggle.com/datasets/matinmahmoudi/sales-and-satisfaction?select=Sales_without_NaNs_v1.3.csv
DATASET_PATH_SALES = ROOT_DIR / 'datasets' / 'Sales_without_NaNs_v1.3.csv'

# def score(model, X_test_split, y_test_split, model_name):
#     y_pred = model.predict(X_test_split)
# 
#     accuracy = accuracy_score(y_test_split, y_pred)
#     precision = precision_score(y_test_split, y_pred)
#     f1 = f1_score(y_test_split, y_pred)
#     sensitivity = recall_score(y_test_split, y_pred) 
#     
#     y_pred_proba = model.predict_proba(X_test_split)[:, 1]
#     auc_roc = roc_auc_score(y_test_split, y_pred_proba)
#     
#     print(f"Matthews: {matthews_corrcoef(y_test_split, model.predict(X_test_split))}")
#     print(f"Accuracy: {accuracy}")
#     print(f"Precision: {precision}")
#     print(f"F1 Score: {f1}")
#     print(f"Sensitivity (Recall): {sensitivity}")
#     print(f"{model_name} AUC-ROC: {auc_roc}")
#     
#     # Optional: Print confusion matrix for more insights
#     tn, fp, fn, tp = confusion_matrix(y_test_split, y_pred).ravel()
#     print(f"\n{model_name} Confusion Matrix:")
#     print(f"True Negatives: {tn}, False Positives: {fp}")
#     print(f"False Negatives: {fn}, True Positives: {tp}")
# 
#     # Plot AUC-ROC curve
#     fpr, tpr, _ = roc_curve(y_test_split, y_pred_proba)
#     plt.figure()
#     plt.plot(fpr, tpr, color='blue', label=f'AUC = {auc_roc:.2f}')
#     plt.plot([0, 1], [0, 1], color='red', linestyle='--')
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title(f'{model_name} AUC-ROC Curve')
#     plt.legend(loc='lower right')
#     plt.savefig(f'{model_name}_aucroc.png')
#     plt.show()
#     plt.close()
#     
#     # Plot confusion matrix
#     plt.figure()
#     cm = confusion_matrix(y_test_split, y_pred)
#     sns.heatmap(cm, 
#                 annot=True,
#                 fmt='g', 
#                 xticklabels=['True','False'],
#                 yticklabels=['True','False'])
#     plt.xlabel('Prediction', fontsize=13)
#     plt.ylabel('Actual', fontsize=13)
#     plt.title(f'{model_name} Confusion Matrix', fontsize=17)
#     plt.savefig(f'{model_name}_matrix.png')
#     plt.show()
#     plt.close()

# Dropped because they are identifiers, duplicate data or single value columns:
dropped_columns_nasa = [
    'Neo Reference ID',
    'Name',
    'Est Dia in KM(min)',
    'Est Dia in KM(max)',
    'Est Dia in Miles(min)',
    'Est Dia in Miles(max)',
    'Est Dia in Feet(min)',
    'Est Dia in Feet(max)',
    'Close Approach Date',
    'Relative Velocity km per hr',
    'Miles per hour',
    'Miss Dist.(miles)',
    'Miss Dist.(kilometers)',
    'Miss Dist.(Astronomical)',
    'Orbiting Body',
    'Orbit ID',
    'Equinox',
]

# Potentially unnecessary columns:
unnecessary_columns_nasa = [
    'Absolute Magnitude',
    'Orbit Determination Date',
]

def main(query, query_dataset='nasa'):
    # query_dataset = input("Which dataset do you want to use? (nasa | heart | sales):")
    if query_dataset == 'nasa':
        df = pd.read_csv(DATASET_PATH_NASA)
        df_X = df.drop('Hazardous', axis=1)
        df_y = df['Hazardous']
    elif query_dataset == 'heart':
        df = pd.read_csv(DATASET_PATH_HEART)
        df_X = df.drop('target', axis=1)
        df_y = df['target']
    elif query_dataset == 'sales':
        df = pd.read_csv(DATASET_PATH_SALES)
        df.replace({'Yes': 1, 'No': 0, 'Control': 0, 'Treatment': 1, 'High Value': 2, 'Medium Value': 1, 'Low Value': 0}, inplace=True)
        print(df.head())
        df_X = df.drop('Purchase_Made', axis=1)
        df_y = df['Purchase_Made']
    else:
        raise ValueError("Invalid dataset")

    # query = input("What do you want to do? (bayes | genetic):")

    # RepeatedStratifiedKFold here
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
    accuracies = []
    precisions = []
    f1s = []
    sensitivities = []
    matthews_corrcoef_scores = []

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    fold_counter = 0
    plt.figure(figsize=(10, 10), dpi=400)

    predictions_nemar = []
    predictions_y_test = []

    for (train_index, test_index) in rskf.split(df_X, df_y):
        df_X_train = df_X.iloc[train_index]
        df_y_train = df_y.iloc[train_index]
        df_X_test = df_X.iloc[test_index]
        df_y_test = df_y.iloc[test_index]

        if query_dataset == 'nasa':
            df_X_train.drop(columns=(dropped_columns_nasa+unnecessary_columns_nasa), inplace=True)
            df_X_test.drop(columns=(dropped_columns_nasa+unnecessary_columns_nasa), inplace=True)

        if query == 'bayes':
            bayes_opt1 = BayesOpt(
                init_guesses=3,
                n_iter=4,
                kernel=lambda X1, X2: np.exp(
                    -0.5 * scipy.spatial.distance.cdist(
                        X1, X2, 'sqeuclidean'
                    )
                ),
                noise=0.1,
                acquisition='PI'
            )

            bayes_opt1.fit(df_X_train, df_y_train, model=RandomForestClassifier)
            print(bayes_opt1.best_params)

            clf1 = bayes_opt1.get_model()
            clf1.fit(df_X_train, df_y_train)
            y_prob = clf1.predict_proba(df_X_test)[:, 1]
            print(clf1.score(df_X_test, df_y_test))
            y_pred = clf1.predict(df_X_test)
            print(matthews_corrcoef(df_y_test, y_pred))
            print((y_pred == df_y_test).value_counts())

            predictions_nemar.append(y_pred)
            predictions_y_test.append(df_y_test)

            fpr, tpr, thresholds = roc_curve(df_y_test, y_prob)

            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            print(fold_counter, tpr, fpr)
            plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (fold_counter, roc_auc))

            fold_counter += 1

            accuracy = accuracy_score(df_y_test, y_pred)
            precision = precision_score(df_y_test, y_pred)
            f1 = f1_score(df_y_test, y_pred)
            sensitivity = recall_score(df_y_test, y_pred)
            matthews_corrcoef_score = matthews_corrcoef(df_y_test, y_pred)

            print(f"Accuracy: {accuracy}")
            print(f"Precision: {precision}")
            print(f"F1 Score: {f1}")
            print(f"Sensitivity (Recall): {sensitivity}")
            print(f"Matthews: {matthews_corrcoef_score}")

            accuracies.append(accuracy)
            precisions.append(precision)
            f1s.append(f1)
            sensitivities.append(sensitivity)
            matthews_corrcoef_scores.append(matthews_corrcoef_score)

        elif query == 'genetic':
            genetic = Genetic(
                population_size=3,
                offspring_size=3,
                n_generations=3,
                mutation_prob=0.1,
                crossover=lambda x, y: {
                    key: x[key] if np.random.rand() > 0.5 else y[key]
                    for key in x.keys()
                },
                mutation=lambda x: {
                    key: np.clip(
                        int(x[key] + np.random.normal(0, 1)),
                        domain[0][0],
                        domain[0][1]
                    )
                    if key == 'max_depth'
                    else np.clip(
                        int(x[key] + np.random.normal(0, 1)),
                        domain[1][0],
                        domain[1][1]
                    )
                    if key == 'min_samples_split'
                    else np.clip(
                        int(x[key] + np.random.normal(0, 1)),
                        domain[2][0],
                        domain[2][1]
                    )
                    if key == 'min_samples_leaf'
                    else np.clip(
                        int(x[key] + np.random.normal(0, 1)),
                        domain[3][0],
                        domain[3][1]
                    )
                    if key == 'max_features'
                    else np.clip(
                        int(x[key] + np.random.normal(0, 1)),
                        domain[4][0],
                        domain[4][1]
                    )
                    if key == 'max_leaf_nodes'
                    else np.clip(
                        x[key] + np.random.normal(0, 1),
                        domain[5][0],
                        domain[5][1]
                    )
                    if key == 'min_impurity_decrease'
                    else x[key]
                    for key in x.keys()
                },
                random_state=42
            )
            genetic.fit(df_X_train, df_y_train, model=RandomForestClassifier)
            print(genetic.best_params)
            clf2 = genetic.get_model()
            clf2.fit(df_X_train, df_y_train)
            y_prob = clf2.predict_proba(df_X_test)[:, 1]
            print(clf2.score(df_X_test, df_y_test))
            y_pred = clf2.predict(df_X_test)
            print(matthews_corrcoef(df_y_test, y_pred))
            print((y_pred == df_y_test).value_counts())

            predictions_nemar.append(y_pred)
            predictions_y_test.append(df_y_test)

            fpr, tpr, thresholds = roc_curve(df_y_test, y_prob)

            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            print(fold_counter, tpr, fpr)
            plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (fold_counter, roc_auc))

            fold_counter += 1

            accuracy = accuracy_score(df_y_test, y_pred)
            precision = precision_score(df_y_test, y_pred)
            f1 = f1_score(df_y_test, y_pred)
            sensitivity = recall_score(df_y_test, y_pred)
            matthews_corrcoef_score = matthews_corrcoef(df_y_test, y_pred)

            print(f"Accuracy: {accuracy}")
            print(f"Precision: {precision}")
            print(f"F1 Score: {f1}")
            print(f"Sensitivity (Recall): {sensitivity}")
            print(f"Matthews: {matthews_corrcoef_score}")

            accuracies.append(accuracy)
            precisions.append(precision)
            f1s.append(f1)
            sensitivities.append(sensitivity)
            matthews_corrcoef_scores.append(matthews_corrcoef_score)
    
    print(f"Average Accuracy: {sum(accuracies) / len(accuracies)}") 
    print(f"Average Precision: {sum(precisions) / len(precisions)}")
    print(f"Average F1 Score: {sum(f1s) / len(f1s)}")
    print(f"Average Sensitivity (Recall): {sum(sensitivities) / len(sensitivities)}")
    print(f"Average Matthews: {sum(matthews_corrcoef_scores) / len(matthews_corrcoef_scores)}")

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.title(f'Cross-Validation ROC of {query.title()}', fontsize=18)
    plt.legend(loc="lower right", prop={'size': 5})
    plt.savefig(fname=f'AUC-ROC_{query.title()}-{query_dataset}-{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}')
    plt.show()

    # plotting ROC curve source:
    # https://stackoverflow.com/questions/57708023/plotting-the-roc-curve-of-k-fold-cross-validation

    predictions_nemar_con = np.concatenate(predictions_nemar)
    predictions_y_con = np.concatenate(predictions_y_test)

    return predictions_nemar_con, predictions_y_con

    # print("Random Search vs Random Search Modified")
    # Y_rs_rsm = pd.DataFrame(predictions_y_con, columns=['Hazardous'])
    # Y_rs_rsm['model_1'] = np.where(predictions_nemar_con1 > 0.5, 1, 0)
    # Y_rs_rsm['model_2'] = np.where(predictions_nemar_con2 > 0.5, 1, 0)
    # 
    # mcnemar_test(get_contingency_table(Y_rs_rsm, 'Hazardous', 'model_1', 'model_2'))

if __name__ == '__main__':
    predictions_nemar_con_bayes, predictiopns_y_con_bayes = main('bayes', 'nasa')
    predictions_nemar_con_genetic, predictiopns_y_con_genetic = main('genetic', 'nasa')

    Y_bayes_genetic = pd.DataFrame(predictiopns_y_con_bayes, columns=['Hazardous'])
    Y_bayes_genetic['bayes'] = np.where(predictions_nemar_con_bayes > 0.5, 1, 0)
    Y_bayes_genetic['genetic'] = np.where(predictions_nemar_con_genetic > 0.5, 1, 0)

    mcnemar_test(get_contingency_table(Y_bayes_genetic, 'Hazardous', 'bayes', 'genetic'))
