import re
str = input()
str_list = re.findall(r'\d+', str)
print(max(str_list, key=len))