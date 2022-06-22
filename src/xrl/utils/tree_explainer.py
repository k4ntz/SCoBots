# class for decision tree to approximate trained policy

import graphviz

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

class TreeExplainer():
    def __init__(self, x, y, cols, actions):
        # init tree
        self.actions = actions
        self.tree = DecisionTreeClassifier()
        self.X = pd.DataFrame(x[:], columns=cols)
        self.y = pd.DataFrame(y, columns=["action"])
        print(self.X.describe())


    # function to generate decision tree classifier
    def train(self, test_size=0.3):
        # Split the dataset in two equal parts
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=0)    
        # parameters for tree
        min_samples_node = int(len(y_train) * 0.02)
        parameters = {'max_depth':range(3,6), 'criterion':["gini", "entropy"], 'min_samples_leaf': np.arange(0.005, 0.05, 0.005)} #min_samples_leaf':[1], 'ccp_alpha': np.arange(0.01, 0.05, 0.01)
        # grid search CV for paramter search
        clf = GridSearchCV(self.tree, parameters, n_jobs=-1, verbose=1, cv=10)
        clf.fit(X_train, y_train)
        # apply best tree
        self.tree = clf.best_estimator_
        print("\nBest parameters set found on development set:")
        print(clf.best_params_) 
        print("\nGrid scores on development set:")
        means = clf.cv_results_["mean_test_score"]
        stds = clf.cv_results_["std_test_score"]
        for mean, std, params in zip(means, stds, clf.cv_results_["params"]):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        print("\nDetailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_pred = self.tree.predict(X_test)
        print(classification_report(y_test, y_pred))
        print()
    
    
    # function to visualize tree
    def visualize(self):
        #plt.figure(figsize=(16,10))
        #plot_tree(self.tree, 
        #feature_names=self.X.columns, class_names=self.actions, 
        #filled=True, rounded=True)
        #plt.show()

        dot_data = export_graphviz(self.tree, out_file=None, 
        feature_names=self.X.columns, class_names=self.actions, 
        filled=True, rounded=True, special_characters=True)  
        graph = graphviz.Source(dot_data)  
        graph.render("visualisation/xdt")

    

    