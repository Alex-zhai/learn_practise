def sum_m():
    n_m = input()
    n, m = n_m.split()
    n = int(n)
    m = int(m)
    if n <= 0 or m <= 0:
        return
    arr = range(1, n + 1)
    result = list()
    path = list()
    pos = 0
    recursion(arr, pos, m, path, result)
    for i in result[:-1]:
        print(' '.join([str(num) for num in i]))
    print(' '.join(str(num) for num in result[-1]),)


def recursion(arr, pos, m, path, result):
    if pos >= len(arr):
        return
    count = 1
    for i in range(pos, len(arr)):
        path.append(arr[i])
        if sum(path) == m:
            result.append(path[:])
        recursion(arr, pos + count, m, path, result)
        path.pop()
        count += 1


if __name__ == "__main__":
    sum_m()

