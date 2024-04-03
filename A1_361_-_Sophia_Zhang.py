import numpy as np
import pandas as pd

df = pd.read_csv('data.csv')
df_clear = df.dropna()

def split_data(dataset, train_size, val_size):
    dataset = dataset.sample(frac=1, random_state=50).reset_index(drop=True)
    train_end = int(len(dataset) * train_size)
    val_end = train_end + int(len(dataset) * val_size)

    train_data = dataset.iloc[:train_end]
    val_data = dataset.iloc[train_end:val_end]
    test_data = dataset.iloc[val_end:]

    return train_data, val_data, test_data

train_data, val_data, test_data = split_data(df_clear, train_size=0.7, val_size=0.1)

train_features = train_data.drop(columns=['poisonous'])
train_labels = train_data['poisonous'].map({'p': 1, 'e': 0}).astype(str)

val_features = val_data.drop(columns=['poisonous'])
val_labels = val_data['poisonous'].map({'p': 1, 'e': 0}).astype(str)

test_features = test_data.drop(columns=['poisonous'])
test_labels = test_data['poisonous'].map({'p': 1, 'e': 0}).astype(str)

class DecisionTree:
    def __init__(self, dataset, labels, features, depth=None, threshold=None, parent=None):
        self.dataset = dataset
        self.labels = labels
        self.features = features
        self.depth = depth if depth is not None else 0
        self.threshold = threshold
        self.parent = parent
        self.children = []
        
    def entropy(self, column_name, labels):
        total_rows = len(labels)
        entropy = 0
        unique_categories = self.features[column_name].unique()
            
        for category in unique_categories:
            category_rows = labels[self.features[column_name] == category]
            category_length = len(category_rows)
            category_counts = category_rows.value_counts()
            category_entropy = 0
            for count in category_counts:
                category_entropy -= (count/category_length) * np.log2(count/category_length)

            entropy += (category_length/total_rows)*category_entropy
        
        return entropy
    
    def information_gain(self, labels, features):
        total_rows = len(labels)
        total_entropy = 0
        _, counts = np.unique(labels, return_counts=True)
        for count in counts:
            total_entropy -= (count/total_rows)*np.log2(count/total_rows)
        
        threshold = None
        max_ig = float("-inf")
        for column in features.columns:
            column_entropy = self.entropy(column, labels)
            information_gain = total_entropy - column_entropy
            if information_gain > max_ig:
                max_ig = information_gain
                threshold = column
        self.threshold = threshold
        return max_ig
        
    def split(self, feature, labels, features):
        split_data = {}
        unique_values = features[feature].unique()
        
        for value in unique_values:
            indices = features.index[features[feature] == value]
            split_data[value] = indices
            
        return split_data
        
    def train(self, stopping_depth):
        if self.depth >= stopping_depth or len(self.labels.unique()) == 1:
            return self.labels.mode()[0]
        
        max_ig = self.information_gain(self.labels, self.features)
        split_data = self.split(self.threshold, self.labels, self.features)

        for value, indices in split_data.items():
            child_dataset = self.dataset.loc[indices]
            child_labels = self.labels.loc[indices]
            child_features = self.features.loc[indices]
            child_tree = DecisionTree(child_dataset, child_labels, child_features, depth=self.depth + 1, threshold=value, parent=self)
            self.children.append(child_tree)
            child_tree.train(stopping_depth)
        
        return self.children
        
    def is_leaf(self):
        return self.children == []
        

    def __str__(self, indent=0):
        if self is None:
            return ""
    
        if self.is_leaf():
            return f"{'    ' * indent}return {self.labels.mode()[0]}\n"
    
        string = ""
        for child in self.children:
            string += f"{'    ' * indent}if {self.threshold} == {child.dataset[self.threshold].iloc[0]}:\n"
            string += child.__str__(indent + 1)
        string += f"{'    ' * indent}else:\n"
        string += f"{'    ' * (indent + 1)}return {self.labels.mode()[0]}\n"
        return string


    def traverse(self, row):
        if self.is_leaf():
            return self.labels.mode()[0]
    
        feature_value = row[self.threshold]  
        
        child_node = None

        for child in self.children:
           
            if child.dataset[self.threshold].iloc[0] == feature_value:
                child_node = child
                break
    
        if child_node is None:
            return self.labels.mode()[0]
            
        return child_node.traverse(row)

        
    def test(self,val_features):
        results = []
        for index, row in val_features.iterrows(): 
            results.append(self.traverse(row))
        return results

    def accuracy(self,results,labels):
        count = 0
        correct = 0
        for label in labels:
            if results[count] == label:
                correct += 1
            count+=1
        return f"{round((correct/count)*100,2)}%"
        
    

tree = DecisionTree(train_data, train_labels, train_features)
tree.train(2)
results = tree.test(train_features)
print(tree.accuracy(results, train_labels))
print(tree)

