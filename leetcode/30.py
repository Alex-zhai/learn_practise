import math
def coding(s):
    s_len = len(s)
    bit1 = math.pow(25, 3) + math.pow(25, 2) + math.pow(25, 1) + 1
    bit2 = math.pow(25, 2) + math.pow(25, 1) + 1
    bit3 = math.pow(25, 1) + 1
    bit4 = 1
    index = 0
    b = [bit1, bit2, bit3, bit4]
    for i in range(s_len):
        index += b[i] * int(ord(s[i]) - ord('a'))
    return index + s_len - 1

while True:
    try:
        s = input()
        print(int(coding(s)))
    except:
        break