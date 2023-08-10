import json
from sklearn.tree import DecisionTreeClassifier


def convert_tree_to_json(tree: DecisionTreeClassifier) -> str:
    def traverse_tree(node):
        if tree.tree_.children_left[node] == -1 and tree.tree_.children_right[node]:  # reached a leaf node
            return {"class": int(tree.tree_.value[node].argmax())}

        return {
            "feature_index": int(tree.tree_.feature[node]),
            "threshold": round(tree.tree_.threshold[node], 4),
            "left": traverse_tree(tree.tree_.children_left[node]),
            "right": traverse_tree(tree.tree_.children_right[node])
        }

    tree_as_dict = traverse_tree(0)
    tree_as_json = json.dumps(tree_as_dict, indent=2)
    return tree_as_json
