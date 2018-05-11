""" Same random forest classifier implementation as in RandomForestTutorial.ipynb,
	but just the code without all the fluff.
"""

import numpy as np


class DecisionNode:
    """ Decision node for decision tree classification models.
        Attributes:
            col (int) : column index, corresponding to what feature to split on
            val (float) : value to test split against
            child_t (DecisionNode) : node to traverse to if sample evaulates true
            child_f (DecisionNode) : node to traverse to if sample evaulates false
            label (int) : predicted class label
    """
    
    def __init__(self):
        """ Initialize a decision node."""
        self.col = None
        self.val = None
        self.child_t = None
        self.child_f = None
        self.label = None
    
    def is_leaf(self):
        """ Check if decision node is leaf.
            Return:
                (bool) : True if is leaf, else False
        """
        if self.label == None:
            return False
        return True


def gini(d1, d2):
    """ Loss function for determining best decision rule when growing 
        decision trees, based on the Gini Impurity.        
        Args:
            d1 (array like) : vector of class labels for data group 1
            d2 (array like) : vector of class labels for data group 2
        Return:
            (float) : loss value
    """
    
    n1, n2 = d1.shape[0], d2.shape[0]
    g1 = 1 - np.sum((np.unique(d1, return_counts=True)[1] / n1)**2)
    g2 = 1 - np.sum((np.unique(d2, return_counts=True)[1] / n2)**2)
    return (g1*n1 + g2*n2) / (n1 + n2)


def best_split(data, loss_fxn):
    """ Given a set of training data and a loss function, determines the best feature 
        to split on and best value to test on for growing a decision tree. 
        
        Args:
            data (np array) : array of training data with class labels as last column
            loss_fxn (function) : python function for computing data split loss        
        Return:
            (int) : best column to split on
            (float) :  best value to test on
            (np array) : data subset that evaluates true
            (np array) : data subset that evaluates false                    
    """
    
    class_vals = np.unique(data[:,-1])
    b_loss = float('Inf')
    b_col = b_val = None
    b_data_t = b_data_f = np.array([])

    for col in range(data.shape[1]-1):
        feature_vals = np.sort(np.unique(data[:,col]))
        midpoints = (feature_vals[1:] + feature_vals[:-1]) / 2.

        for val in midpoints:
            data_t = data[data[:,col] < val]
            data_f = data[data[:,col] >= val]
            loss = loss_fxn(data_t[:,-1], data_f[:,-1])
            if loss < b_loss:
                b_loss, b_col, b_val, b_data_t, b_data_f = loss, col, val, data_t, data_f

    return (b_col, b_val, b_data_t, b_data_f) 


class DecisionTree:
    """ Build decision tree classification model and predict class labels on unseen examples.
        Attributes:
            max_depth (int) : maximum depth tree is allowed to grow to
            loss_fxn (funtion) : function for evaluating data split loss
            split_fxn (function) : function for determing best feature and value to split on
            root (DecisionNode) : root of learned decision tree model            
    """
    
    def __init__(self, max_depth=float('Inf'), loss=gini, split=best_split):
        """ Initialize a decision tree classifier model.
            Args:
                max_depth (int) : (optional) maximum depth tree is allowed to grow to
                loss (function) : (optional) function for evaluating data split loss
                split (function) : (optional) function for determining best feature and value to split on
        """
        self.max_depth = max_depth
        self.loss_fxn = loss
        self.split_fxn = split
        self.root = None   
    
    def fit(self, X, y):
        """ Fit a decision tree classification model to training data.
            Args:
                X (np array) : array of training data, n_samples x n_features
                y (array like) : vector of class labels, n_samples
        """
        self.root = self.add_child(np.c_[X, y], 0)
    
    def predict(self, X):
        """ Predict class labels for unseen examples.
            Args:
                X (np array) : array of test data, n_samples x n_features
            Return:
                (np array) : vector of predicted class labels
        """
        y = np.array([self.node_search(self.root, row) for row in X])
        return y
    
    def add_child(self, data, depth):
        """ Add a child node to a decision tree, called recursively.
            Args:
                data (np array) : array of training data with class labels as last column
                depth (int) : current depth in tree
            Return:
                (DecisionNode or None) : DecisionNode of child to add to tree, None if no data passed in
        """
        if data.shape[0]==0:
            return None
        if depth >= self.max_depth:
            return self.make_leaf(data)

        col, val, data_t, data_f = self.split_fxn(data, self.loss_fxn)
        child_t = self.add_child(data_t, depth+1)
        child_f = self.add_child(data_f, depth+1)
        
        if (child_t == None) and (child_f != None):
            return self.make_leaf(data_f)
        if (child_f == None) and (child_t != None):
            return self.make_leaf(data_t)
        if (child_t == None) and (child_f == None):
            return self.make_leaf(data)

        node = DecisionNode()
        if child_t.is_leaf() and child_f.is_leaf() and child_t.label==child_f.label:
            node.label = child_t.label
        else:
            node.col, node.val, node.child_t, node.child_f = col, val, child_t, child_f
        return node 
    
    def make_leaf(self, data):
        """ Makes a leaf decision node, with predicted class label as mode of example labels.
            Args:
                data (np array) : array of training data with class labels as last column
            Return:
                (DecisionNode) : leaf node
        """
        labels = data[:,-1].tolist()
        node = DecisionNode()
        node.label = max(set(labels), key=labels.count)
        return node
    
    def node_search(self, node, sample):
        """ Traverse decision tree by evaluating samples at each node, called recursively.
            Args:
                node (DecisionNode) : decision containing decision criteria, or class label if leaf
                sample (array like) : vector of feature values
            Return: 
                (int) : predicted class label
        """
        if node.is_leaf():
            return node.label
            
        if sample[node.col] < node.val:
            return self.node_search(node.child_t, sample)
        else:
            return self.node_search(node.child_f, sample)           


def print_tree(node, depth, flag):
    """ Prints a decision tree, displaying decision rules and class labels, called recursively.
        Args:
            node (DecisionNode) : current decision node
            depth (int) : current tree depth
            flag (int) : 0=root, 1=True, 2=False
    """
    if flag==1:
        prefix = 'T->'
    elif flag==2:
        prefix = 'F->'
    else:
        prefix = ''

    if node.is_leaf():
        print('{}{}[{}]'.format(depth*'   ', prefix, node.label))
    else:  
        print('{}{}(X{} < {:0.3f})?'.format(depth*'   ', prefix, node.col+1, node.val))
        print_tree(node.child_t, depth+1, 1)
        print_tree(node.child_f, depth+1, 2)
        
        
def accuracy(model, X_test, y_test):
    """ Computes the accuracy of a given model as the percentage of correctly classified examples.
        Args:
            model (model like) : trained classification to be evaluated
            X_test (np array) : test data, n_samples x n_features
            y_test (array like) : true class labels, n_samples
        Return:
            (float) : accuracy of classification model in range [0.0, 1.0]
    """
    predictions = model.predict(X_test)
    return (np.array(predictions) == np.array(y_test)).mean()


def best_split_rf(data, loss_fxn):
    """ Given a set of training data and a loss function, determines the best feature to split on 
        and best value to test on for growing a decision tree with stochastic feature selection.
        
        Args:
            data (np array) : array of training data with class labels as last column
            loss_fxn (function) : python function for computing data split loss        
        Return:
            (int) : best column to split on
            (float) :  best value to test on
            (np array) : data subset that evaluates true
            (np array) : data subset that evaluates false                    
    """
    
    class_vals = np.unique(data[:,-1])
    b_loss = float('Inf')
    b_col = b_val = None
    b_data_t = b_data_f = np.array([])
    
    n_cols = int(np.sqrt(data.shape[1]-1))
    cols = np.random.choice(np.arange(data.shape[1]-1), n_cols, replace=False)

    for col in cols:
        feature_vals = np.sort(np.unique(data[:,col]))
        midpoints = (feature_vals[1:] + feature_vals[:-1]) / 2.

        for val in midpoints:
            data_t = data[data[:,col] < val]
            data_f = data[data[:,col] >= val]
            loss = loss_fxn(data_t[:,-1], data_f[:,-1])
            if loss < b_loss:
                b_loss, b_col, b_val, b_data_t, b_data_f = loss, col, val, data_t, data_f

    return (b_col, b_val, b_data_t, b_data_f)


class RandomForest:
    """ Build random forest classification model and predict class labels on unseen examples.
        Attributes:
            max_depth (int) : maximum depth tree is allowed to grow to
            n_trees (int) : number of decision trees in ensemble
            loss_fxn (funtion) : function for evaluating data split loss
            split_fxn (function) : function for determing best feature and value to split on
            trees (list) : list of learned decision tree models             
    """
    
    def __init__(self, n_trees=50, max_depth=float('Inf'), loss=gini, split=best_split_rf):
        """ Initialize a decision tree classifier model.
            Args:
                n_trees (int) : (optional) number of decision trees in ensemble
                max_depth (int) : (optional) maximum depth tree is allowed to grow to
                loss (function) : (optional) function for evaluating data split loss
                split (function) : (optional) function for determining best feature and value to split on
        """
        self.max_depth = max_depth
        self.n_trees = n_trees
        self.loss_fxn = loss
        self.split_fxn = split
        self.trees = []        
    
    def fit(self, X, y):
        """ Fit a random forest classification model to training data.
            Args:
                X (np array) : array of training data, n_samples x n_features
                y (array like) : vector of class labels, n_samples
        """
        for i in range(self.n_trees):
            sample_idx = np.random.choice(X.shape[0], X.shape[0], replace=True)
            tree = DecisionTree(max_depth=self.max_depth, loss=self.loss_fxn, split=self.split_fxn)
            tree.fit(X[sample_idx], y[sample_idx])
            self.trees.append(tree)        
            
    def predict(self, X):
        """ Predict class labels for unseen examples.
            Args:
                X (np array) : array of test data, n_samples x n_features
            Return:
                (np array) : vector of predicted class labels
        """
        y = []
        for row in X:
            predictions = [t.predict([row])[0] for t in self.trees]
            y.append(max(set(predictions), key=predictions.count))
        return np.array(y)         