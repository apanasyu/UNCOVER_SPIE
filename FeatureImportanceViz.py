'''
To perform 5-fold cross-validation, you can use scikit-learn’s `cross_val_score` or `cross_validate` function. Below, I'm going to modify your code to use `cross_validate`, which will perform 5-fold cross-validation for each classifier and provide a more detailed performance assessment for each fold. I'll also accumulate the performance metrics so that you can save them to a CSV file at the end.
To save the best-performing model for making predictions later, you will need to determine which model performs the best based on the cross-validation metrics and then retrain that model on the entire dataset before saving it.
1. Collect average scores from cross-validation for each model.
2. Retrain the best model on all available data.
3. Save the best model using `joblib`.

In this implementation:
- We loop through each feature name provided in `featureList`, removing one feature at a time.
- For each reduced feature set, we run cross-validation on each classifier and calculate an average precision, recall, and F1-score.
- If we find a model with a better score, we update our records of the best score, model, and feature set.
- After the feature elimination loop, we refit the best classifier with all the data and the best features selected.
- We save the best classifier to a file in the specified directory, and we also save the list of best features to a text file.


This code generates a line plot where the mean Precision, Recall, and F1-Score are visualized as functions of the total number of features, displayed in decreasing order on the x-axis.
'''

def plot_classifier_metrics(filepath, fileToSaveTo, maxFeatures = 1000):
    import pandas as pd
    import matplotlib.pyplot as plt

    #if metric not in ['Precision', 'Recall', 'F1']:
    #    raise ValueError("Metric should be one of 'Precision', 'Recall', or 'F1-Score'")

    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(filepath)

    # Filter rows by 'MEAN' in the 'Fold' column
    if 'Fold' in list(df.columns):
        mean_df = df[df['Fold'] == 'MEAN']
    else:
        mean_df = df

    # Get unique classifiers
    classifiers = mean_df['Classifier'].unique()

    # Iterate over each classifier and plot the specified metric
    for classifier in classifiers:
        # Filter the DataFrame for the current classifier
        classifier_df = mean_df[mean_df['Classifier'] == classifier]
        classifier_df = classifier_df[classifier_df['TotalFeatures'] <= maxFeatures]

        # Sort the DataFrame by 'TotalFeatures' in descending order
        classifier_df.sort_values('TotalFeatures', ascending=False, inplace=True)

        # Plot
        plt.figure(figsize=(7, 5))
        for metric in ['Precision', 'Recall', 'F1']:
            plt.plot(classifier_df['TotalFeatures'], classifier_df[metric+"_class_0"], marker='o', label=f'{metric+"_class_0"}')
            plt.plot(classifier_df['TotalFeatures'], classifier_df[metric+"_class_1"], marker='*', label=f'{metric+"_class_1"}')

        # Inverse x-axis to show decreasing TotalFeatures
        plt.gca().invert_xaxis()

        # Labeling the axes and title
        plt.xlabel('Number of Features', fontsize=9)
        plt.ylabel('Precision, Recall, and F1', fontsize=9)
        plt.title(f'{classifier}: Mean performance of each metric as total number of features decreases\n Class 0 = Contains Propaganda, Class 1 = No Propaganda', fontsize=11)

        # Show legend
        plt.legend()

        # Additional customization
        plt.grid(True)
        plt.tight_layout()

        # Show plot
        plt.savefig(fileToSaveTo+"_"+classifier+'.png', dpi=300)
        plt.clf()

    plt.figure(figsize=(7, 5))
    for classifier in classifiers:
        # Filter the DataFrame for the current classifier
        classifier_df = mean_df[mean_df['Classifier'] == classifier]
        classifier_df = classifier_df[classifier_df['TotalFeatures'] <= maxFeatures]

        # Sort the DataFrame by 'TotalFeatures' in descending order
        classifier_df.sort_values('TotalFeatures', ascending=False, inplace=True)

        # Plot
        for metric in ['F1']:
            plt.plot(classifier_df['TotalFeatures'], classifier_df[metric+"_class_0"], marker='o', label=f'{classifier+"_"+metric+"_class_0"}')
            plt.plot(classifier_df['TotalFeatures'], classifier_df[metric+"_class_1"], marker='*', label=f'{classifier+"_"+metric+"_class_1"}')
            print(classifier_df['TotalFeatures'])
            print(classifier_df[metric+"_class_0"])
            print(classifier_df[metric + "_class_1"])
        # Inverse x-axis to show decreasing TotalFeatures
        plt.gca().invert_xaxis()

        # Labeling the axes and title
        plt.xlabel('Number of Features', fontsize=11)
        plt.ylabel('F1', fontsize=11)
        plt.title(f'Mean performance of each classifier as total number of features decreases\n Class 0 = Contains Propaganda, Class 1 = No Propaganda', fontsize=11)

        # Show legend
        plt.legend()

        # Additional customization
        plt.grid(True)
        plt.tight_layout()

        # Show plot
    plt.savefig(fileToSaveTo+'allClass.png', dpi=300)


from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_validate
import joblib

def performCrossValidationEliminatingFeaturesAlongTheWay(X, y, labelName, featureList, out_path1):
    # Prepare the data (remove NaN values)
    y_label = y[labelName].dropna()
    X_filtered = X.loc[y_label.index]

    from sklearn.metrics import f1_score
    classifiers = {
        'LogisticRegression': LogisticRegression(solver='liblinear', random_state=42),
        'RandomForest': RandomForestClassifier(random_state=42),
        'SVC': SVC(random_state=42)
    }

    # Define metrics for evaluation
    scoring_metrics = {
        'precision_weighted': make_scorer(precision_score, average='weighted'),
        'recall_weighted': make_scorer(recall_score, average='weighted'),
        'f1_weighted': make_scorer(f1_score, average='weighted'),
        'precision_class_0': make_scorer(precision_score, average=None, labels=[0]),
        'precision_class_1': make_scorer(precision_score, average=None, labels=[1]),
        'recall_class_0': make_scorer(recall_score, average=None, labels=[0]),
        'recall_class_1': make_scorer(recall_score, average=None, labels=[1]),
        'f1_class_0': make_scorer(f1_score, average=None, labels=[0]),
        'f1_class_1': make_scorer(f1_score, average=None, labels=[1]),
    }

    rows_list = []
    totalFeatures = len(X_filtered.columns.tolist())
    for classifier_name, classifier in classifiers.items():
        print(f"Training {classifier_name} classifier for label: {labelName}...")

        # Perform cross-validation
        cv_results = cross_validate(classifier, X_filtered, y_label, cv=5,
                                    scoring=scoring_metrics, return_train_score=False)

        # Store the results
        for fold_index in range(5):
            print(cv_results['test_precision_class_0'][fold_index])
            print(cv_results['test_precision_class_1'][fold_index])
            print(cv_results['test_recall_class_0'][fold_index])
            row = {
                'TotalFeatures': totalFeatures,
                'Classifier': classifier_name,
                'Fold': fold_index + 1,
                'Precision_weighted': cv_results['test_precision_weighted'][fold_index],
                'Recall_weighted': cv_results['test_recall_weighted'][fold_index],
                'F1_weighted': cv_results['test_f1_weighted'][fold_index],
                'Precision_class_0': cv_results['test_precision_class_0'][fold_index],
                'Precision_class_1': cv_results['test_precision_class_1'][fold_index],
                'Recall_class_0': cv_results['test_recall_class_0'][fold_index],
                'Recall_class_1': cv_results['test_recall_class_1'][fold_index],
                'F1_class_0': cv_results['test_f1_class_0'][fold_index],
                'F1_class_1': cv_results['test_f1_class_1'][fold_index],
            }
            rows_list.append(row)

        row = {
            'TotalFeatures': totalFeatures,
            'Classifier': classifier_name,
            'Fold': "MEAN",
            'Precision_weighted': cv_results['test_precision_weighted'].mean(),
            'Recall_weighted': cv_results['test_recall_weighted'].mean(),
            'F1_weighted': cv_results['test_f1_weighted'].mean(),
            'Precision_class_0': cv_results['test_precision_class_0'].mean(),
            'Precision_class_1': cv_results['test_precision_class_1'].mean(),
            'Recall_class_0': cv_results['test_recall_class_0'].mean(),
            'Recall_class_1': cv_results['test_recall_class_1'].mean(),
            'F1_class_0': cv_results['test_f1_class_0'].mean(),
            'F1_class_1': cv_results['test_f1_class_1'].mean(),
        }
        rows_list.append(row)

        print("Done with cross-validation for", classifier_name)
    for i in range(len(featureList) - 1, -1, -1):
        feature = featureList[i]

        # Remove the feature from X
        print(f"Removing feature {feature} from X")
        X_filtered = X_filtered.drop(columns=[feature])

        totalFeatures = len(X_filtered.columns.tolist())
        print(f"{i} {X_filtered.columns.tolist()}")
        for classifier_name, classifier in classifiers.items():
            print(f"Training {classifier_name} classifier for label: {labelName}...")

            # Perform cross-validation
            cv_results = cross_validate(classifier, X_filtered, y_label, cv=5,
                                        scoring=scoring_metrics, return_train_score=False)

            # Store the results
            for fold_index in range(5):
                print(cv_results['test_precision_class_0'][fold_index])
                print(cv_results['test_precision_class_1'][fold_index])
                print(cv_results['test_recall_class_0'][fold_index])
                row = {
                    'TotalFeatures': totalFeatures,
                    'Classifier': classifier_name,
                    'Fold': fold_index + 1,
                    'Precision_weighted': cv_results['test_precision_weighted'][fold_index],
                    'Recall_weighted': cv_results['test_recall_weighted'][fold_index],
                    'F1_weighted': cv_results['test_f1_weighted'][fold_index],
                    'Precision_class_0': cv_results['test_precision_class_0'][fold_index],
                    'Precision_class_1': cv_results['test_precision_class_1'][fold_index],
                    'Recall_class_0': cv_results['test_recall_class_0'][fold_index],
                    'Recall_class_1': cv_results['test_recall_class_1'][fold_index],
                    'F1_class_0': cv_results['test_f1_class_0'][fold_index],
                    'F1_class_1': cv_results['test_f1_class_1'][fold_index],
                }
                rows_list.append(row)

            row = {
                'TotalFeatures': totalFeatures,
                'Classifier': classifier_name,
                'Fold': "MEAN",
                'Precision_weighted': cv_results['test_precision_weighted'].mean(),
                'Recall_weighted': cv_results['test_recall_weighted'].mean(),
                'F1_weighted': cv_results['test_f1_weighted'].mean(),
                'Precision_class_0': cv_results['test_precision_class_0'].mean(),
                'Precision_class_1': cv_results['test_precision_class_1'].mean(),
                'Recall_class_0': cv_results['test_recall_class_0'].mean(),
                'Recall_class_1': cv_results['test_recall_class_1'].mean(),
                'F1_class_0': cv_results['test_f1_class_0'].mean(),
                'F1_class_1': cv_results['test_f1_class_1'].mean(),
            }
            rows_list.append(row)

            print("Done with cross-validation for", classifier_name)

        # Create a DataFrame from the list of dictionaries
        results_df = pd.DataFrame(rows_list)
        results_df.to_csv(out_path1, index=False)
        print(f"Classification results have been saved to: {out_path1}")

        if totalFeatures == 1:
            break

def fitClassifierEliminatingFeaturesAlongTheWay(X, y, labelName, featureList, out_path1):
    # Prepare the data (remove NaN values)
    y_label = y[labelName].dropna()
    X_filtered = X.loc[y_label.index]

    from sklearn.metrics import f1_score
    classifiers = {
        'LogisticRegression': LogisticRegression(solver='liblinear', random_state=42),
        'RandomForest': RandomForestClassifier(random_state=42),
        'SVC': SVC(random_state=42)
    }

    rows_list = []
    totalFeatures = len(X_filtered.columns.tolist())
    # Split the data into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_label,
                                                        test_size=0.2, random_state=42)


    for classifier_name, classifier in classifiers.items():
        print(f"Training {classifier_name} classifier...")

        # Fit the classifier
        classifier.fit(X_train, y_train)

        # Make predictions on the test set
        predictions = classifier.predict(X_test)

        # Evaluate the classifier
        report = classification_report(y_test, predictions, output_dict=True)

        # Print the classification report
        print(f"Classification Report for {classifier_name}:")
        print(report)

        row = {
            'TotalFeatures': totalFeatures,
            'Classifier': classifier_name,
            'Precision_class_0': report['0.0']['precision'],
            'Precision_class_1': report['1.0']['precision'],
            'Recall_class_0': report['0.0']['recall'],
            'Recall_class_1': report['1.0']['recall'],
            'F1_class_0': report['0.0']['f1-score'],
            'F1_class_1': report['1.0']['f1-score'],
            'support_class_0': report['0.0']['support'],
            'support_class_1': report['1.0']['support'],
        }
        rows_list.append(row)

    for i in range(len(featureList) - 1, -1, -1):
        feature = featureList[i]

        # Remove the feature from X
        print(f"Removing feature {feature} from X")
        X_filtered = X_filtered.drop(columns=[feature])
        X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_label,
                                                            test_size=0.2, random_state=42)

        totalFeatures = len(X_filtered.columns.tolist())
        print(f"{i} {X_filtered.columns.tolist()}")
        for classifier_name, classifier in classifiers.items():
            print(f"Training {classifier_name} classifier...")

            # Fit the classifier
            classifier.fit(X_train, y_train)

            # Make predictions on the test set
            predictions = classifier.predict(X_test)

            # Evaluate the classifier
            report = classification_report(y_test, predictions, output_dict=True)

            # Print the classification report
            print(f"Classification Report for {classifier_name}:")
            print(report)

            row = {
                'TotalFeatures': totalFeatures,
                'Classifier': classifier_name,
                'Precision_class_0': report['0.0']['precision'],
                'Precision_class_1': report['1.0']['precision'],
                'Recall_class_0': report['0.0']['recall'],
                'Recall_class_1': report['1.0']['recall'],
                'F1_class_0': report['0.0']['f1-score'],
                'F1_class_1': report['1.0']['f1-score'],
                'support_class_0': report['0.0']['support'],
                'support_class_1': report['1.0']['support'],
            }
            rows_list.append(row)

        # Create a DataFrame from the list of dictionaries
        results_df = pd.DataFrame(rows_list)
        results_df.to_csv(out_path1, index=False)
        print(f"Classification results have been saved to: {out_path1}")

        if totalFeatures == 1:
            break

    # Create a DataFrame from the list of dictionaries
    results_df = pd.DataFrame(rows_list)
    results_df.to_csv(out_path1, index=False)
    print(f"Classification results have been saved to: {out_path1}")


def ApplyClassifierOnSelectFeatures(X, y, labelName, featureList, out_path1):
    # Prepare the data (remove NaN values)
    y_label = y[labelName].dropna()
    X_filtered = X.loc[y_label.index]
    if len(featureList) > 0:
        X_filtered = X_filtered.filter(featureList)

    from sklearn.metrics import f1_score
    classifiers = {
        'LogisticRegression': LogisticRegression(solver='liblinear', random_state=42),
        'RandomForest': RandomForestClassifier(random_state=42),
        'SVC': SVC(random_state=42)
    }

    rows_list = []
    totalFeatures = len(X_filtered.columns.tolist())
    # Split the data into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_label,
                                                        test_size=0.2, random_state=42)

    for classifier_name, classifier in classifiers.items():
        print(f"Training {classifier_name} classifier...")

        # Fit the classifier
        classifier.fit(X_train, y_train)

        # Make predictions on the test set
        predictions = classifier.predict(X_test)

        # Evaluate the classifier
        report = classification_report(y_test, predictions, output_dict=True)

        # Print the classification report
        print(f"Classification Report for {classifier_name}:")
        print(report)

        row = {
            'TotalFeatures': totalFeatures,
            'Classifier': classifier_name,
            'Precision_class_0': report['0.0']['precision'],
            'Precision_class_1': report['1.0']['precision'],
            'Recall_class_0': report['0.0']['recall'],
            'Recall_class_1': report['1.0']['recall'],
            'F1_class_0': report['0.0']['f1-score'],
            'F1_class_1': report['1.0']['f1-score'],
            'support_class_0': report['0.0']['support'],
            'support_class_1': report['1.0']['support'],
            'Avg_Precision': report['macro avg']['precision'],
            'Avg_Recall': report['macro avg']['recall'],
            'Avg_F1': report['macro avg']['f1-score'],
        }
        rows_list.append(row)

    # Create a DataFrame from the list of dictionaries
    results_df = pd.DataFrame(rows_list)
    results_df.to_csv(out_path1, index=False)
    print(f"Classification results have been saved to: {out_path1}")

def apply_saved_model(model_path, new_X):
    """
    Applies a saved model to new data for prediction.

    Parameters:
    model_path (str): path to the saved model file.
    new_X (array-like): new data to predict on, shape = [n_samples, n_features].

    Returns:
    array: predicted classes or regression values for each instance in new_X.
    """

    # Load the saved model from the file
    model = joblib.load(model_path)

    # Make predictions on the new data
    predictions = model.predict(new_X)

    # You can return predictions here if you'd like to use them directly.
    return predictions

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report

if __name__ == '__main__':

    directoryFolderSemEvalQ = "SemEvalQuestionOntology/"
    fileCSVPath = directoryFolderSemEvalQ + "FeatureMatrix.csv"
    columnsToSkipList = ["QueryID"]
    dTree = directoryFolderSemEvalQ + 'random_forest_feature_importances.csv'
    dAnova = directoryFolderSemEvalQ + 'anova_feature_importance.csv'
    from StatisticalAnalysis import getXy
    X, y = getXy(fileCSVPath, columnsToSkipList)

    from QuestionRepository import getRepositoryOfQuestions

    dirOut = "QuestionRelatedOut/"
    fileNames = ["SemEval.txt"]
    systemRoles, questionCountToQuestion, questionCountToQuestionCategory = getRepositoryOfQuestions(dirOut, fileNames,
                                                                                                     directoryFolderSemEvalQ + "SemEvalIndividualQuestionIDs.csv")
    from StatisticalAnalysis import getTopXFeaturesForLabel

    topFeatures_ANOVA = getTopXFeaturesForLabel(dAnova, questionCountToQuestion, questionCountToQuestionCategory,
                                                label="None")
    topFeatures_DTree = getTopXFeaturesForLabel(dTree, questionCountToQuestion, questionCountToQuestionCategory,
                                                label="None")

    #performCrossValidationEliminatingFeaturesAlongTheWay(X, y, "None", topFeatures_ANOVA, directoryFolderSemEvalQ+"ClassifierCrossFoldEliminatingFeaturesUsingANOVA.csv")
    #performCrossValidationEliminatingFeaturesAlongTheWay(X, y, "None", topFeatures_DTree, directoryFolderSemEvalQ+"ClassifierCrossFoldEliminatingFeaturesUsingRandomTree.csv")
    #fitClassifierEliminatingFeaturesAlongTheWay(X, y, "None", topFeatures_DTree, directoryFolderSemEvalQ+"ClassifierEliminatingFeaturesUsingRandomTree.csv")
    #fitClassifierEliminatingFeaturesAlongTheWay(X, y, "None", topFeatures_ANOVA, directoryFolderSemEvalQ + "ClassifierEliminatingFeaturesUsingANOVA.csv")

    filepath0 = directoryFolderSemEvalQ+"ClassifierCrossFoldEliminatingFeaturesUsingANOVA.csv"  # Path to your CSV file
    filepath1 = directoryFolderSemEvalQ + "ClassifierCrossFoldEliminatingFeaturesUsingRandomTree.csv"  # Path to your CSV file
    #filepath2 = directoryFolderSemEvalQ + "ClassifierEliminatingFeaturesUsingANOVA.csv"  # Path to your CSV file
    #filepath3 = directoryFolderSemEvalQ + "ClassifierEliminatingFeaturesUsingRandomTree.csv"  # Path to your CSV file

    import os
    dirOutPlot = directoryFolderSemEvalQ+"TopFeatureSelectionPlots/"
    if not os.path.exists(dirOutPlot):
        os.makedirs(dirOutPlot)
        print(f"The new directory {dirOutPlot} is created!")

    if True:
        for maxFeatures in [30, 50, 500]:
            plot_classifier_metrics(filepath0, dirOutPlot+"ANOVAFeatureElim_MaxF"+str(maxFeatures)+"_", maxFeatures)
            plot_classifier_metrics(filepath0, dirOutPlot+"RandTreeFeatureElim_MaxF"+str(maxFeatures)+"_", maxFeatures)
            plot_classifier_metrics(filepath1, dirOutPlot+"ANOVAFeatureElim_MaxF"+str(maxFeatures)+"_", maxFeatures)
            plot_classifier_metrics(filepath1, dirOutPlot+"RandTreeFeatureElim_MaxF"+str(maxFeatures)+"_", maxFeatures)
            #plot_classifier_metrics(filepath2, dirOutPlot+"ANOVAFeatureElim_MaxF"+str(maxFeatures)+"_", maxFeatures)
            #plot_classifier_metrics(filepath2, dirOutPlot+"RandTreeFeatureElim_MaxF"+str(maxFeatures)+"_", maxFeatures)
            #plot_classifier_metrics(filepath3, dirOutPlot+"ANOVAFeatureElim_MaxF"+str(maxFeatures)+"_", maxFeatures)
            #plot_classifier_metrics(filepath3, dirOutPlot+"RandTreeFeatureElim_MaxF"+str(maxFeatures)+"_", maxFeatures)

    '''top features based on technique'''
    classifiers = ['LogisticRegression', 'RandomForest', 'SVC']
    featuresElim = ['ANOVAFeatureElim', 'RandTreeFeatureElim']
    topFeatures = {}
    topFeatureCount = {}
    for classifier in classifiers:
        for featuresElimT in featuresElim:
            topFeatures[classifier+"_"+featuresElimT] = []
            topFeatureCount[classifier+"_"+featuresElimT] = 0
    print(topFeatureCount)

    #this is generated by manually analyzing each graph
    topFeatureCount = {'LogisticRegression_ANOVAFeatureElim': 0, 'LogisticRegression_RandTreeFeatureElim': 0,
     'RandomForest_ANOVAFeatureElim': 0, 'RandomForest_RandTreeFeatureElim': 0, 'SVC_ANOVAFeatureElim': 0,
     'SVC_RandTreeFeatureElim': 0}

    topFeatures1 = topFeatures_ANOVA[:8]
    topFeatures2 = topFeatures_DTree[:8]
    topFeatures = set(topFeatures1).union(set(topFeatures2))
    print(topFeatures)
    count = 0
    rows_list = []
    for feature in topFeatures:
        index1 = "n/a"
        index2 = "n/a"
        if feature in topFeatures_ANOVA:
            index1 = topFeatures_ANOVA.index(feature)
        if feature in topFeatures_DTree:
            index2 = topFeatures_DTree.index(feature)
        print(f"{count} {feature} ({index1}, {index2}) {questionCountToQuestionCategory[feature]} {questionCountToQuestion[feature]}")
        count += 1

        row = {
            'Qid': feature,
            'Rank via ANOVA/Random Forest': f"({index1}, {index2})",
            'Q Category': questionCountToQuestionCategory[feature],
            'Question': questionCountToQuestion[feature]
        }
        rows_list.append(row)

    results_df = pd.DataFrame(rows_list)
    results_df.to_csv(directoryFolderSemEvalQ+"TopFeaturesForLabelNone.csv", index=False)

    from StatisticalAnalysis import getXy
    XOld, yOld = getXy(directoryFolderSemEvalQ + "FeatureMatrixOld.csv", columnsToSkipList)
    ApplyClassifierOnSelectFeatures(X, y, "None", ['Q215', 'Q20', 'Q211', 'Q295', 'Q210', 'Q212', 'Q216', 'Q217', 'Q92', 'Q88', 'Q213', 'Q258'], directoryFolderSemEvalQ + "ClassifierUsingTopFeatureComb.csv")
    ApplyClassifierOnSelectFeatures(X, y, "None",
                                    topFeatures_ANOVA[:8], directoryFolderSemEvalQ + "ClassifierUsingTopFeatureAnova.csv")
    ApplyClassifierOnSelectFeatures(X, y, "None",
                                    topFeatures_DTree[:8], directoryFolderSemEvalQ + "ClassifierUsingTopFeatureRTree.csv")
    ApplyClassifierOnSelectFeatures(X, y, "None",
                                    topFeatures_DTree, directoryFolderSemEvalQ + "ClassifierUsingAllFeatures.csv")

    ApplyClassifierOnSelectFeatures(XOld, yOld, "None", [], directoryFolderSemEvalQ + "ClassifierUsingAllFeatureOld.csv")
