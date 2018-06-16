
import graph_search_algorithms
import pandas as pd

def generate_mock_binary_tree():

    df = pd.DataFrame([[0, "is_sunny",1,2]], columns=["level_id", "var_name", "yes", "no"])
    df.loc[1] = [1,"choose_park",3, 4] # if it is sunny
    df.loc[2] = [2,"choose_movie",5, 6] # if it is not sunny

    return df

def test_get_paths_from_tree():

    assert sorted(graph_search_algorithms.get_paths_from_tree(generate_mock_binary_tree()))==\
           sorted([['choose_park', 'is_sunny'], ['choose_movie', 'is_sunny']])

