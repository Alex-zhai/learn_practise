n = int(input())
zero_num = 0
index = -1
result = 1
for i in range(1, n+1):
    result = result * i
result_str = str(result)
while True:
    if result_str[index] == '0':
        zero_num += 1
        index = index - 1
    else:
        break
print(zero_num)