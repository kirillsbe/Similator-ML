from typing import List


def extend_matches(groups: List[tuple]) -> List[tuple]:
    result = []
    tuple_list = groups
    for tup in tuple_list:
        for idx, already in enumerate(result):
            # check if any items are equal
            if any(item in already for item in tup):
                # tuples are immutable so we need to set the result item directly
                result[idx] = already + tuple(item for item in tup if item not in already)
                break
        else:
            # else in for-loops are executed only if the loop wasn't terminated by break
            result.append(tup)
    result = sorted([tuple(sorted(i)) for i in (result)])
    return result