

import array_algorithms

def test_removeDuplicateRows():

    data = [["a", "c", "d"], ["a", "b", "d"], ["a", "c", "d"]]

    assert sorted(array_algorithms.removeDuplicateRows(data)) == sorted([data[x] for x in [1,0]])

    data = [["a", "c", "d"], ["a", "b"], ["a", "c", "d"]]

    assert sorted(array_algorithms.removeDuplicateRows(data)) == sorted([data[x] for x in [1,0]])
