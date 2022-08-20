import numpy as np
from ete3 import Tree, TreeStyle, TextFace, add_face_to_node # Just a library that helps draw trees

class tree_node:
    def __init__(self, class_problem, data, labels, diagram_node, available_features, entropy_func):
        self.class_problem = class_problem
        self.data = data # holds the data which the tree node will use to find the best feature
        self.labels = labels # holds the corresponding labels for each data point
        self.diagram_node = diagram_node # Used for creating the tree diagram
        self.available_features = available_features # Holds the list of names for the remaining features available at a tree node
        self.feature_index = None # Holds the index of the feature in the list of features which provides the most information gain
        self.feature_name = None # Holds the name of the feature which provides the most information gain
        self.node_values = None # Holds the unique values of the feature which provides the most information gain
        self.is_leaf = False # Reflects if the node is a leaf node
        self.class_value = None # Is set to True or False if a node classifies a data point (it is a leaf node)
        self.entropy_func = entropy_func
        
        # Three cases to consider when adding a node to the tree.
        # 1. The node entropy is not 0, so the node still needs to split the data further, and there is still
        #    an unused feature to split the data with
        if not (entropy_func(labels[:,self.class_problem]) == 0.0 or\
                       entropy_func(labels[:,self.class_problem]) == -0.0 or self.data.shape[1] == 0):
            self.children = np.array([])
            self.find_feature()
            self.descend_tree(self.data, self.feature_index)
        # 2. If the entropy is not 0 but we have already used all features to split the data
        elif self.data.shape[1] == 0:
            self.is_leaf = True
            unique, counts = np.unique(labels[:,class_problem], return_counts=True)
            majority_class = np.argmax(counts)
            self.class_value = unique[majority_class]
            self.feature_name = str(self.class_value)
            self.diagram_node.name = self.feature_name
        # 3. The entropy of the data coming into the node is 0 and so the node must just pick a classification
        #    (ie: True or False)
        else:
            self.is_leaf = True
            self.class_value = labels[0,class_problem]
            self.feature_name = str(labels[0,class_problem])
            self.diagram_node.name = self.feature_name


    def pick_feature(self, data, initial_entropy, labels):
        pass
            
    # This function is used to calculate the information gain of each feature and pick the best feature using the
    # "pick_feature" function and then just manages the class information afterwards.
    def find_feature(self):
        print("Finding feature for new node")
        chosen_feature = self.pick_feature(self.data, self.labels)
        self.feature_index = chosen_feature # update features of the node class
        self.feature_name = self.available_features[self.feature_index]
        self.diagram_node.name = self.feature_name
        print("Found feature: ", self.feature_name)
        
    # This just adds the tree node to the diagram and then calls the tree_node class constructor. This is the
    # recursive part of the algorithm. Inside of the ``descend_tree`` function in the jupyter notebook we call this 
    # function.
    def add_child(self, data_for_feature_value, remaining_features):
        new_child_diagram_node = self.diagram_node.add_child(name="Temp") # used for making the tree diagram
        self.children = np.append(self.children, 
                tree_node(self.class_problem, self.data[data_for_feature_value][:,remaining_features],
                self.labels[data_for_feature_value],
                new_child_diagram_node, self.available_features[remaining_features], self.entropy_func))
        
    # This function is used to add nodes to the tree once the best feature to split the data is found.
    # Its output is passed into ``add_child'' in the jupyter notebook.
    def descend_tree(self, data, chosen_feature):
        pass

    # This function infers (predicts) the class of a new/unseen data point. We call this on the test data points
    def infer(self, data_point):
        if not self.is_leaf: # if the node we are looking at is not a leaf node (can't classify the data point)
            for i in range(self.node_values.shape[0]): # look through the set of values the node looks for
                if self.node_values[i] == data_point[self.feature_index]: # to find which branch to descend down
                    allocated_class = self.children[i].infer(data_point[np.arange(data_point.shape[0])!=self.feature_index]) # recursively run the infer function on the child node
                    return allocated_class # return back up the tree
            print("Error found new value, can't classify")
        else:
            return(self.class_value) # If it is a leaf node then we just return the classification given by the leaf node

# Loads the data so that we don't have to worry about it in the jupyter notebook
def read_data():
    data_file = open('card_data.txt', 'r')
    data_string = data_file.read()
    full_data = np.array(data_string.split(','))[:-1].astype(np.float64).reshape((1100,5))
    print("full_data has ", full_data.shape[0], " data points with ", full_data.shape[1], "features")
    # Reading in the 4 different labels for each data point
    labels_file = open('card_categories.txt', 'r')
    labels_string = labels_file.read()
    full_y_values = np.array(labels_string.split(','))[:-1]
    full_y_values = np.where(full_y_values == 'True', True, False).reshape((1100,4))
    print("full_y_values has labels for ", full_y_values.shape[0], " data points with ", full_y_values.shape[1], "labels per data points")
    return full_data, full_y_values

# This creates the tree diagram
def create_tree(class_problem, data, y_values, feature_names, entropy_func, render):
    t = Tree() # Used for diagram, creates tree
    diagram_root = t.add_child(name="root") # adds root node to tree diagram
    root = tree_node(class_problem, data, y_values, diagram_root, feature_names, entropy_func) # Use our class to train a decision tree on the training data
    print(t.get_ascii(show_internal=True)) # prints a diagram of the decision tree
    # The remainder of the window is use to draw the decsision tree. Note the last line can be removed to
    # avoid rendering the image as it can look quite bad with large trees. The printed ascii version can still be seen
    # in this case
    ts = TreeStyle()
    ts.show_leaf_name = False
    def my_layout(node):
        F = TextFace(node.name, tight_text=True)
        F.rotable = True
        F.border.width = 0
        F.margin_top = 5
        F.margin_bottom = 5
        F.margin_left = 5
        F.margin_right = 5
        add_face_to_node(F, node, column=0, position="branch-right")
    ts.layout_fn = my_layout
    ts.mode = 'r'
    ts.arc_start = 270
    ts.arc_span = 185
    ts.draw_guiding_lines = True
    ts.scale = 100
    ts.branch_vertical_margin = 100
    ts.min_leaf_separation = 100
    ts.show_scale = False
    if render == True:
        t.render(file_name="%%inline", w=500, h=500, tree_style=ts)
    return root
