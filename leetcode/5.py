import re
while True:
    try:
        str1 = input().strip()
        str2 = input().strip()
        str1 = str1.replace('*', '[1-9a-zA-z]*').replace('?','.')
        print(str1)
        str3 = re.findall(str1, str2)
        str3 = ''.join(str3)
        if str3 == str2:
            print('true')
        else:
            print('false')
    except:
        break