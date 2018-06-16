


def removeDuplicateRows(ndarr):
    '''
    Input:  takes an list of lists of variable length
    Output: returns a set of it by removing duplicate rows
    '''

    return [list(y) for y in set( [tuple(sorted(x)) for x in ndarr])]