'''
If you have performed one-hot encoding on the `GoldLabel` column, the next step is to perform feature selection to determine the influence or importance of each feature (Q0, Q1, ..., Q323) on the prediction task. There are multiple approaches to perform statistical analysis for feature selection:

1. **Chi-Square**: Use the Chi-Square test to select features that have the strongest relationship with the response variable. However, the Chi-Square test requires all input values to be positive and is typically used for categorical data.

2. **ANOVA F-test**: For numerical input and categorical output, the ANOVA F-test can be applied, and it is provided in scikit-learn via the `f_classif` function.

3. **Feature Importance from Tree-based models**: Decision Trees and ensemble algorithms such as Random Forest or Gradient Boosting can provide feature importances based on how helpful each feature is at reducing uncertainty (entropy or Gini impurity).
'''

'''
This code will generate a series of feature rankings for each of the `GoldLabel` categories, after ANOVA F-tests. By observing which features consistently score higher across different categories, you can identify the most significant features.
You can tweak `k` in `SelectKBest` function to only select the top `k` features instead of all features (`k='all'`). If `k` is set to a specific number, `SelectKBest` will only return that number of features with the highest scores.
If the dataset is large, or there are many features, ranking all of them could be computationally expensive, and selecting a subset might be preferable.
Remember that the F-test assumes that the features follow a normal distribution and it should be applied considering this. If you're not sure about the assumptions, you might want to use other feature selection techniques like feature importance from tree-based models.
'''
#Here's a sample Python code that uses the `f_classif` for ANOVA F-test from scikit-learn:
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

def getXy(fileCSVPath, columnsToSkipList):
    # Load the CSV file
    # Load the CSV file
    data = pd.read_csv(fileCSVPath)
    data = data.drop(columnsToSkipList, axis=1)
    labelName = "None"
    data.dropna(subset=[labelName], inplace=True)


    # Features
    X = data.filter(regex='^Q\d+$').copy()  # Using regex to select columns starting with 'Q'
    X.fillna(0, inplace=True)

    # Labels - Assuming 'GoldLabel' is now a one-hot encoded matrix
    y = data.filter(regex='^(?!Q\d+$).*$').copy()  # Selecting non-'Q' columns
    y.dropna(axis=0, inplace=True)

    return X, y

def anovaF_Test(X, y, out_path='anova_feature_importance.csv'):
    # Perform feature selection separately for each label
    feature_scores = {}
    for label in y:
        selector = SelectKBest(f_classif, k='all')
        selector.fit(X.loc[y.dropna().index], y[label].dropna())
        feature_scores[label] = selector.scores_

    # Turn feature scores into a DataFrame
    features_df = pd.DataFrame(feature_scores, index=X.columns)
    features_df.fillna(0, inplace=True)  # Fill any remaining NaNs with zeros

    # Save to CSV
    features_df.to_csv(out_path)

'''
Feature importance from tree-based models, such as a Random Forest, can help identify which features are most influential in predicting the outcomes of a given model. These models measure feature importance by looking at how much each feature decreases the impurity of the split (e.g., Gini impurity for classification tasks).

This code prepares the feature data `X` and target `y` to be used by the RandomForestClassifier. We iterate over each label column, fit a Random Forest classifier, and store feature importances. An average importance across all labels is then computed to rank features.
Keeping the `random_state` parameter fixed ensures that the results are reproducible, as the randomness of the Random Forests will be the same on each run. Moreover, you might want to tune the `n_estimators` parameter in the RandomForestClassifier initialization to adjust the number of trees in the forest; more trees typically lead to better performance but increase computational cost.
Remember to explore the parameters of RandomForestClassifier and potentially perform hyperparameter tuning to achieve better performance. More sophisticated methods such as using a MultiOutputClassifier, or problem transformation methods that handle multi-label classification more explicitly, can also be considered depending on the complexity and specifics of your problem.
'''
def decisionTree(X, y, out_pathDir='random_forest_feature_importance.csv'):
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier

    # Initialize the RandomForestClassifier
    rf_models = {}
    feature_importance_df = pd.DataFrame(index=X.columns)  # DataFrame to hold feature importances

    # Loop over each label and fit a model
    for label in y:
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X.loc[y[label].dropna().index], y[label].dropna())
        feature_importance_df[label] = rf.feature_importances_
        rf_models[label] = rf  # Save the fitted model if needed

    # Compute the mean of the feature importances and save to CSV
    feature_importance_df['mean_importance'] = feature_importance_df.mean(axis=1)
    feature_importance_df.to_csv(out_pathDir)

def printTopXFeatures(filePath, qIDtoQuestion, questionCountToQuestionCategory):
    import pandas as pd
    scores_df = pd.read_csv(filePath)
    print(scores_df.columns)

    labelsWithNoData = []
    for label in scores_df.columns:
        if label != "Unnamed: 0":
            top_10_features = scores_df.nlargest(10, label)
            featureNames = list(top_10_features["Unnamed: 0"])
            scores = list(top_10_features[label])
            if scores[0] != 0:
                print(label)
                print(featureNames)
                questionCategories = []
                for qID in featureNames:
                    questionCategories.append(questionCountToQuestionCategory[qID])
                print(questionCategories)

                for qID in featureNames:
                    print(qIDtoQuestion[qID])

                print(scores)
            else:
                print(f"No data available for: {label}")
                labelsWithNoData.append(label)

    print(labelsWithNoData)
    return labelsWithNoData

def getTopXFeaturesForLabel(filePath, qIDtoQuestion, questionCountToQuestionCategory, label):
    import pandas as pd
    scores_df = pd.read_csv(filePath)
    print(scores_df.columns)

    top_x_features = scores_df.nlargest(1000, label)
    featureNames = list(top_x_features["Unnamed: 0"])
    scores = list(top_x_features[label])
    print(featureNames)
    print(scores)

    return list(top_x_features["Unnamed: 0"])