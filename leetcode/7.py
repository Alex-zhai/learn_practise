from collections import deque

class Solution:
    def reOrderArray(self, array):
        sortedInt = deque()
        x = len(array)
        for i in range(x):
            if array[x - i - 1] % 2 != 0:
                sortedInt.appendleft(array[x-i-1])
            if array[i] % 2 == 0:
                sortedInt.append(array[i])
        return list(sortedInt)

a = Solution()
print(a.reOrderArray([1,2,3,4,5]))



