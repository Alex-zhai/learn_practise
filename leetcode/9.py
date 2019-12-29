import heapq
class Solution:
    def GetLeastNumbers_Solution(self, tinput, k):
        if not tinput or not k or len(tinput) < k:
            return []
        return heapq.nsmallest(k, tinput)