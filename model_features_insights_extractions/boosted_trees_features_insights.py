import pandas as pd
from graph_search_algorithms import *
from array_algorithms import *

'''
Summary: takes a trained xgboost model and returns the trees as a list of pandas dfs
'''

def get_xgboost_trees(xgboost_model):

    trees = xgboost_model.get_dump()
    print trees
    trees_list = []
    #print trees
    for tree in trees:
        #print(tree)
        tree_text = tree.split(":")

        tree_struct = []
        next_level_id = 0

        if len(tree_text) > 2:
            for level in tree_text:
                level_text = level.split(" ")
                #print(level_text)
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
            # print(tree_struct)
            tree_struct = tree_struct.set_index(tree_struct.level_id.values)
            trees_list.append(tree_struct)

    return trees_list


'''
Summary: takes a trained xgboost model and returns the trees as a list of pandas dfs
'''

def get_xgboost_interactions(xgboost_model):

    return [ x for x in get_paths_from_trees(get_xgboost_trees(xgboost_model)) if len(x)>1]
