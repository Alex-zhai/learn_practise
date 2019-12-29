# while True:
#     try:
#         n = int(input())
#         flag = int(input())
#         if flag == 0:
#             flag = True
#         else:
#             flag = False
#         grad_list = []
#         strings = [input() for i in range(n)]
#         for i in range(n):
#             grad_list.append(strings[i].split())
#         sort_grad = sorted(grad_list, key = lambda x: int(x[1]), reverse=flag)
#         for i in range(n):
#             print(" ".join(sort_grad[i]))
#     except:
#         break

while True:
    try:
        n = int(input())
        strs = [input() for i in range(n)]
        grad_dict = {}
        for i in range(n):
            line = strs[i].split()
            grad_dict[line[0]] = int(line[1])

        sort_list = sorted(grad_dict.items(), key=lambda x: x[1])

        for i in sort_list:
            print(str(i[0]) + " " + str(i[1]))

    except:
        break