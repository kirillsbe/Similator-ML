from typing import List
from typing import Tuple
import itertools


def extend_matches(pairs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    listOfSets = [set(i) for i in pairs]
    symmDiff = []
    symmDiffOld = [1]
    for x in listOfSets:
        for y in listOfSets:
            if (x ^ y) not in symmDiff and x.intersection(y) != set():
                symmDiff.append(x ^ y)
    symmDiff = [i for i in symmDiff if i != set()]
    for x in symmDiff + listOfSets:
        for y in symmDiff + listOfSets:
            if (x ^ y) not in (symmDiff + listOfSets) and x.intersection(y) != set():
                symmDiff.append(x ^ y)
    symmDiff = [i for i in symmDiff if i != set()]
    res = sorted([tuple(i) for i in (symmDiff + listOfSets)])
    return res
