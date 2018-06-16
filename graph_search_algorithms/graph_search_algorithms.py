
import  array_algorithms



def get_paths_from_tree(tree_struct):
    '''
    Summary: simple depth first search implementation to return all paths in a binary tree
    Inputs:
        tree_struct: a data frame containing the tree structure with columns: yes (one branch indicator),
         no (another branch indicator), level_id (current level id), var_name (node name)
    Outputs:
        a list of paths in which each path is a list of node names

    '''

    paths = []
    cand_path_stack = [[0]]

    while (len(cand_path_stack) > 0):

        cand_path = cand_path_stack.pop()
        last_node_id = cand_path[-1]
        row = tree_struct[tree_struct.level_id == last_node_id]

        if len(row) > 0:
            cand_path_stack.append(cand_path + [int(row.yes)])
            cand_path_stack.append(cand_path + [int(row.no)])
        else:
            del cand_path[-1]
            paths.append(tree_struct.var_name[cand_path].tolist())

    return array_algorithms.removeDuplicateRows(paths)



def get_paths_from_trees(trees_list):
    '''
    Summary: takes a list of trees as dfs and returns a set of paths as a list - set meaning each path occurs only once

    Inputs:
        trees_list: a list of  trees in which is tree is a data frame containing the tree structure with columns:
        yes (one branch indicator), no (another branch indicator), level_id (current level id), var_name (node name)
    Outputs:
        a list of paths in which each path is a list of node names

    '''

    paths = []

    for tree_struct in trees_list:

        paths = paths + get_paths_from_tree(tree_struct)

    return sorted(array_algorithms.removeDuplicateRows(paths))