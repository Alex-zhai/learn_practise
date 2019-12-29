def findMaxSum(nums):
    sum = 0
    maxSum = nums[0]
    for num in nums:
        if sum >= 0:
            sum += num
        else:
            sum = num
        if sum > maxSum:
            maxSum = sum
    return maxSum

while True:
    try:
        n = input()
        nums = list(map(int, input().strip().split()))
        print(findMaxSum(nums))
    except:
        break