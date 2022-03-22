import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.colors import LinearSegmentedColormap
from pythermalcomfort.models import pmv
from pythermalcomfort.utilities import clo_dynamic, v_relative

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.model_selection import KFold, StratifiedKFold


def save_variable(file_name, variable):
    pickle.dump(variable, open(file_name, "wb"))


def load_variable(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def clf_metrics(test_labels, pred_labels, conf_matrix_print=False, scorer="f1_micro"):
    """Compute the confusion matrix and a particular score based on `scorer`."""
    if scorer == "f1_micro":  # [0, 1]
        metric = f1_score(test_labels, pred_labels, average="micro", zero_division=0)

    # classification report
    class_report = classification_report(
        test_labels, pred_labels, output_dict=True, zero_division=0
    )

    if conf_matrix_print:
        print(f"Confusion Matrix: \n {confusion_matrix(test_labels, pred_labels)} \n")

    return metric, class_report


def choose_tree_depth(
    clf,
    X,
    y,
    k_fold,
    fig_name="",
    scorer="f1_micro",
    save_fig=False,
    verbose=False,
):
    """Choose the optimal depth of a tree model"""
    depths = list(range(1, 11))
    cv_scores = []

    if verbose:
        print("Finding optimal tree depth")

    for d in depths:
        # keep same params but depth
        clf_depth = clf.set_params(max_depth=d)

        if scorer == "f1_micro":
            scorer = "accuracy"  # accuracy = f1-micro

        scores = cross_val_score(clf_depth, X, y, cv=k_fold, scoring=scorer)
        cv_scores.append(scores.mean())

    # changing to misclassification error and determining best depth
    error = [1 - x for x in cv_scores]  # error = 1 - scorer
    optimal_depth = depths[error.index(min(error))]

    if save_fig:
        plt.figure(figsize=(12, 10))
        plt.plot(depths, error)
        plt.xlabel("Tree Depth", fontsize=40)
        plt.ylabel("Misclassification Error", fontsize=40)
        plt.savefig(f"{fig_name}_depth.png")
        plt.close()

    if verbose:
        print(
            f"The optimal depth is: {optimal_depth} with error of {min(error)} and score {max(cv_scores)}"
        )

    return optimal_depth, max(cv_scores)


def cv_model_param(X, y, model, parameters, k_fold, scorer="f1_micro", verbose=False):
    """Choose the best combination of parameters for a given model"""

    grid_search = GridSearchCV(model, parameters, cv=k_fold, scoring=scorer)
    grid_search.fit(X, y)

    if verbose:
        print(
            f"Best parameters set found on CV set: {grid_search.best_params_} with score of {grid_search.best_score_:.2f}"
        )

    return grid_search.best_estimator_, grid_search.best_score_


def train_model(
    dataframe,
    stratified=False,
    model="rdf",
    scorer="f1_micro",
    use_val=False,
    fig_name="",
):
    """
    Finds best set of param with K-fold CV and returns trained model and accuracy
    Assumes the label is the last column.

    Returns
    -------
        clf_cv: object
            Best performing lassification model from CV
        model_acc: dictionary
            Dictionary where the keys are the scorers used and values is the metric itself
        class_report:
            Dictionary where the keys are the scorers used and values is the classifiation report
    """
    model_acc = {}  # TODO: can be extended to more metrics
    model_acc["f1_micro"] = {}
    class_report = {}
    class_report["f1_micro"] = {}

    # create feature matrix X and target vector y
    X = np.array(
        dataframe.iloc[:, 0 : dataframe.shape[1] - 1]
    )  # minus 1 for the target column
    y = np.array(dataframe.iloc[:, -1]).astype(
        int
    )  # casting in case the original variable was a float

    if model == "rdf":
        parameters = {
            "n_estimators": [100, 300, 500],
            "criterion": ["gini"],
            "min_samples_split": [2, 3, 4],
            "min_samples_leaf": [1, 2, 3],
            "class_weight": ["balanced"],
        }
        clf = RandomForestClassifier(
            random_state=100, warm_start=False
        )  # warm_start=true allows for partial_fit

    # cross-validation
    kf = (
        StratifiedKFold(n_splits=5, shuffle=True)
        if stratified
        else KFold(n_splits=5, shuffle=True)
    )

    if use_val:
        dev_size_percentage = 0.2
        X_cv, X_dev, y_cv, y_dev = train_test_split(
            X, y, test_size=dev_size_percentage, random_state=100
        )  # , stratify=y)
        # find params with f1_micro
        clf_cv, cv_score_f1_micro = cv_model_param(
            X_cv, y_cv, clf, parameters, kf, scorer
        )
    else:
        # find params with f1_micro
        clf_cv, cv_score_f1_micro = cv_model_param(X, y, clf, parameters, kf, scorer)

    # plot depth for rdf and update model
    if model == "rdf":
        # find depth
        optimal_depth, cv_score_f1_micro = (
            choose_tree_depth(clf_cv, X_cv, y_cv, kf, fig_name, "f1_micro")
            if use_val
            else choose_tree_depth(clf_cv, X, y, kf, fig_name, "f1_micro")
        )
        clf_cv = clf_cv.set_params(max_depth=optimal_depth)

    # fit the model and get accuracy
    if use_val:
        clf_cv.fit(X_cv, y_cv)
        y_pred = clf_cv.predict(X_dev)
        model_acc["f1_micro"], class_report["f1_micro"] = clf_metrics(
            y_dev, y_pred, conf_matrix_print=False, scorer="f1_micro"
        )
    else:  # no dev_set (use_val=False) average cv_score will be the model_acc
        model_acc["f1_micro"] = cv_score_f1_micro
        class_report["f1_micro"] = ""

    return clf_cv, model_acc, class_report


def find_pcm(dataframe, model, scorer, use_val, folder_str, verbose=False):
    """
    Find the personal comfort model of each user based on CV.
    Assumes a column `user_id` exists.

    Parameters
    ----------
        dataframe: dataframe
            A DataFrame with all data and labels as last column
        model: str
            Name of the classification model to be used
        scorer: str
            Scoring metric for cross-validation performance
        use_val: boolean
            Whether to use a validation set
        folder_str: str
            Name for generated figures

    Returns
    -------
        user_pcm: dictionary
            Dictionary with the model (value) for each user (key)
        user_pcm_acc: dictionary
            Dictionary with the model accuracy (value) for each user (key)
    """

    df = dataframe.copy()

    user_list = df["user_id"].unique()
    if verbose:
        print(
            f"Features used for modeling (`user_id` and the last feature are not used): {df.columns.values}"
        )

    user_pcm = {}
    user_pcm_acc = {}
    user_pcm_acc["f1_micro"] = {}
    # TODO: other metrics can be added

    # for every user, do CV
    for user in user_list:
        df_user = df[df["user_id"] == user]
        df_user = df_user.drop(["user_id"], axis=1)

        fig_name = folder_str + str(user)
        model_user, model_user_acc, _ = train_model(
            dataframe=df_user,
            stratified=True,
            model=model,
            scorer=scorer,
            use_val=use_val,
            fig_name=fig_name,
        )
        user_pcm[user] = model_user
        user_pcm_acc["f1_micro"][user] = model_user_acc["f1_micro"]

    return user_pcm, user_pcm_acc


def simplified_pmv_model(data):
    data = data[["rh-env", "t-env", "clothing", "met", "thermal"]].copy()
    data["met"] = data["met"].map(
        {
            "Sitting": 1.1,
            "Resting": 0.8,
            "Standing": 1.4,
            "Exercising": 3,
        }
    )
    data["clothing"] = data["clothing"].map(
        {
            "Very light": 0.3,
            "Light": 0.5,
            "Medium": 0.7,
            "Heavy": 1,
        }
    )

    arr_pmv_grouped = []
    arr_pmv = []
    for _, row in data.iterrows():
        val = pmv(
            row["t-env"],
            row["t-env"],
            v_relative(0.1, row["met"]),
            row["rh-env"],
            row["met"],
            clo_dynamic(row["clothing"], row["met"]),
        )
        if val < -1.5:
            arr_pmv_grouped.append("Warmer")
        elif -1.5 <= val <= 1.5:
            arr_pmv_grouped.append("No Change")
        else:
            arr_pmv_grouped.append("Cooler")

        arr_pmv.append(val)

    data["PMV"] = arr_pmv
    data["PMV_grouped"] = arr_pmv_grouped

    return data["PMV_grouped"]

def vote_by_user(
    dataframe,
    dataset="dorn",
    show_percentages=False,
    preference_label="thermal_cozie",
    fontsize=40,
):
    """
    Original code by Dr. Federico Tartarini
    https://github.com/FedericoTartarini
    """

    df = dataframe.copy()
    df[preference_label] = df[preference_label].map(
        {9.0: "Warmer", 10.0: "No Change", 11.0: "Cooler"}
    )
    _df = (
        df.groupby(["user_id", preference_label])[preference_label]
        .count()
        .unstack(preference_label)
    )
    _df.reset_index(inplace=True)

    df_total = _df.sum(axis=1)
    df_rel = _df[_df.columns[1:]].div(df_total, 0) * 100
    df_rel["user_id"] = _df["user_id"]

    # sort properly
    df_rel["user_id"] = df_rel["user_id"].str.replace(dataset, "").astype(int)
    df_rel = df_rel.sort_values(by=["user_id"], ascending=False)
    df_rel["user_id"] = dataset + df_rel["user_id"].astype(str)
    df_rel = df_rel.reset_index(drop=True)

    # plot a Stacked Bar Chart using matplotlib
    rc("text.latex", preamble=r"\usepackage{cmbright}")
    rc("text", usetex=True)

    df_rel.plot(
        x="user_id",
        kind="barh",
        stacked=True,
        mark_right=True,
        cmap=LinearSegmentedColormap.from_list(
            preference_label,
            [
                "tab:blue",
                "tab:green",
                "tab:red",
            ],
            N=3,
        ),
        width=0.95,
        figsize=(16, 16),
    )

    plt.legend(
        bbox_to_anchor=(0.5, 1.02),
        loc="center",
        borderaxespad=0,
        ncol=3,
        frameon=False,
        fontsize=fontsize,
    )
    sns.despine(left=True, bottom=True, right=True, top=True)

    plt.tick_params(labelsize=fontsize * 0.75)
    plt.xlabel(r"Percentage [\%]", size=fontsize)
    plt.ylabel("User ID", size=fontsize)

    if show_percentages:
        # add percentages
        for index, row in df_rel.drop(["user_id"], axis=1).iterrows():
            cum_sum = 0
            for ix, el in enumerate(row):
                if ix == 1:
                    plt.text(
                        cum_sum + el / 2 if not np.isnan(cum_sum) else el / 2,
                        index,
                        str(int(np.round(el, 0))) + "\%",
                        va="center",
                        ha="center",
                        size=fontsize * 0.6,
                    )
                cum_sum += el

    plt.tight_layout()
    plt.savefig(f"img/{dataset}_vote_dist.png", pad_inches=0, dpi=300)
    plt.show()
