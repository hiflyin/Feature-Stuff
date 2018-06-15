

import array_algorithms

def test_removeDuplicateRows():

    data = [["a", "c", "d"], ["a", "b", "d"], ["a", "c", "d"]]

    assert array_algorithms.removeDuplicateRows(data) == [data[x] for x in [1,0]]

    data = [["a", "c", "d"], ["a", "b"], ["a", "c", "d"]]

    assert array_algorithms.removeDuplicateRows(data) == [data[x] for x in [1,0]]
