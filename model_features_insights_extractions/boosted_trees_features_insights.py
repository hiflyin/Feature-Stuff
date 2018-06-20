import pandas as pd
from graph_search_algorithms import *
from array_algorithms import *


def get_xgboost_trees(xgboost_model):
    '''
    Summary: takes a trained xgboost model and returns the trees as a list of pandas dfs
    '''

    trees = xgboost_model.get_dump()
    trees_list = []

    for tree in trees:

        tree_text = tree.split(":")

        tree_struct = []
        next_level_id = 0

        if len(tree_text) > 2:
            for level in tree_text:
                level_text = level.split(" ")

                if len(level_text) > 1:
                    level_tokens = level_text[1].split(",")
                    a = level_text[0]

                    a = a[1:]
                    if "<=" in a:
                        split_char = "<="
                    elif "<" in a:
                        split_char = "<"
                    elif ">=" in a:
                        split_char = ">="
                    elif ">" in a:
                        split_char = ">"
                    elif "==" in a:
                        split_char = "=="

                    tree_struct.append({"level_id": int(next_level_id),
                                        "var_name": a.split(split_char)[0],
                                        "yes": level_tokens[0].split("=")[1],
                                        "no": level_tokens[1].split("=")[1]
                                        })
                    next_level_id = level_tokens[2].split("\t")[-1]
                else:
                    next_level_id = level_text[0].split("\t")[-1]

                    #
            tree_struct = pd.DataFrame(tree_struct)

            tree_struct = tree_struct.set_index(tree_struct.level_id.values)
            trees_list.append(tree_struct)

    return trees_list


def get_xgboost_interactions(xgboost_model):
    '''
    Summary: takes a trained xgboost model and returns a list of interactions between features, to the order of maximum
        depth of all trees.
    '''

    return [ x for x in get_paths_from_trees(get_xgboost_trees(xgboost_model)) if len(x)>1]
