
import numpy as np

'''

Input:  takes an list of lists of variable length
Output: returns a set of it by removing duplicate rows

'''

def removeDuplicateRows(ndarr):

    return [list(y) for y in set( [tuple(sorted(x)) for x in ndarr])]