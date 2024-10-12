# Say "Hello, World!" With Python - 1
print("Hello, World!")

# Python If-Else - 2
import math
import os
import random
import re
import sys

if __name__ == '__main__':
    n = int(input().strip())
    if (n % 2 == 1) or ((n % 2 == 0) and (n >= 6 and n<= 20)):
        print("Weird")
    elif ((n % 2 == 0) and (n > 20)) or ((n % 2 == 0) and (n >= 2 and n<= 5)):
        print("Not Weird")

# Arithmetic Operators - 3
if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print(a+b)
    print(a-b)
    print(a*b)

# Python: Division - 4
if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print(a//b)
    print(a/b)

# Loops - 5
if __name__ == '__main__':
    n = int(input())
    for i in range(n):
        print(i*i)

# Write a function - 6
def is_leap(year):
    leap = False
    if year % 400 == 0:
        leap = True
    elif year % 100 == 0:
        leap = False
    elif year % 4 == 0:
        leap = True
    return leap

year = int(input())
print(is_leap(year))

# Print Function - 7
if __name__ == '__main__':
    n = int(input())
    for i in range(n):
        print(i + 1, end='')

# List Comprehensions - 8
if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())
    print([[i, j, k] for i in range(x + 1) for j in range(y + 1) for k in range(z + 1) if i + j + k != n])

# Find the Runner-Up Score! - 9
if __name__ == '__main__':
    n = int(input())
    arr = list(set(map(int, input().split())))
    arr.sort(reverse=True)
    print(arr[1])


# Nested Lists - 10
if __name__ == '__main__':
    
    my_list = []
    for _ in range(int(input())):
        name = input()
        score = float(input())
        my_list.append([name, score])
        
    unique_sorted = sorted(list(set([score for name, score in my_list])), reverse=False)
    second_lowest_score = unique_sorted[1]

    second_lowest_students = [name for name, score in my_list if score == second_lowest_score]
    second_lowest_students.sort()
    
    for student in second_lowest_students:
        print(student)

# Finding the percentage - 11
if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()
    average = (sum(student_marks[query_name]) / len(student_marks[query_name]))
    print(f'{average:.2f}')

# Tuples - 12 -------- (The compiler raised error for this question but I am still attaching code)
if __name__ == '__main__':
    n = int(input())
    t = tuple(map(int, input().split()))
    print(hash(t))

# Lists - 13
if __name__ == '__main__':
    N = int(input())
    thelist = []
    for _ in range(N):
        command = list(input().strip().split())
        if command[0] == "insert":
            thelist.insert(int(command[1]), int(command[2]))
        elif command[0] == "print":
            print(thelist)
        elif command[0] == "remove":
            thelist.remove(int(command[1]))
        elif command[0] == "append":
            thelist.append(int(command[1]))
        elif command[0] == "sort":
            thelist.sort()
        elif command[0] == "pop":
            thelist.pop()
        elif command[0] == "reverse":
            thelist.reverse()

# sWAP cASE - 14
def swap_case(s):
    swapped = ''.join([char.lower() if char.isupper() else char.upper() for char in s])
    return swapped

# String Split and Join - 15
def split_and_join(line):
    words = list(line.split())
    newstr = "-".join(words)
    return newstr

# What's Your Name? - 16
# Complete the 'print_full_name' function below.
#
# The function is expected to return a STRING.
# The function accepts following parameters:
#  1. STRING first
#  2. STRING last
#
def print_full_name(first, last):
    # Write your code here
    print(f'Hello {first} {last}! You just delved into python.')


# Mutations - 17
def mutate_string(string, position, character):
    modifiable_string = list(string)
    modifiable_string[position] = character
    string_modified = ''.join(modifiable_string)
    return string_modified

# Find a string - 18
def count_substring(main_string, substring):
    total = 0
    for i in range(len(main_string)):
        if main_string[i:len(main_string)].startswith(substring):
            total += 1
    return total

# String Validators - 19
if __name__ == '__main__':
    s = input()

    if any(char.isalnum() for char in s):
        print("True")
    else:
        print("False")

    if any(char.isalpha() for char in s):
        print("True")
    else:
        print("False")

    if any(char.isdigit() for char in s):
        print("True")
    else:
        print("False")

    if any(char.islower() for char in s):
        print("True")
    else:
        print("False")

    if any(char.isupper() for char in s):
        print("True")
    else:
        print("False")

# Text Wrap - 20
import textwrap

def wrap(string, max_width):
    textwrap.fill.wrap()
    return

if __name__ == '__main__':
    string, max_width = input(), int(input())
    result = wrap(string, max_width)
    print(result)

# String Formatting - 21
def print_formatted(number):
    
    width = len(bin(number)) - 2
    for i in range(1, number + 1):
        print(f"{i:{width}d} {i:{width}o} {i:{width}X} {i:{width}b}")
# d: decimal, o: octal, X: Hexadecimal, b: binary

# Capitalize! - 22
# Complete the solve function below.
def solve(s):
    return ' '.join([word.capitalize() if word else '' for word in s.split(' ')])

# The Minion Game - 23
def minion_game(string):
    vowels = "AEIOU"
    length = len(string)
    kevin_score = 0
    stuart_score = 0

    for i in range(length):
        if string[i] in vowels:
            kevin_score += length - i
        else:
            stuart_score += length - i
    
    if kevin_score > stuart_score:
        print("Kevin", kevin_score)
    elif kevin_score < stuart_score:
        print("Stuart", stuart_score)
    else:
        print("Draw")
       
# Merge the Tools! - 24
def merge_the_tools(string, k):
    for i in range(0, len(string), k):
        substring = string[i:i + k]
        unique_chars = []
        for char in substring:
            if char not in unique_chars:
                unique_chars.append(char)
        print(''.join(unique_chars))


# Introduction to Sets - 25
def average(array):
    # your code goes here
    return (sum(set(array)) / len(set(array)))

# No Idea! - 26

# Enter your code here. Read input from STDIN. Print output to STDOUT
happiness = 0
first = list(map(int, input().split()))
n = first[0]
m = first[1]
Arr = list(map(int, input().split()))
setA = set(map(int, input().split()))
setB = set(map(int, input().split()))

for elem in Arr:
    if elem in setA:
        happiness += 1
    if elem in setB:
        happiness -= 1
print(happiness)

# Symmetric Difference - 27
# Enter your code here. Read input from STDIN. Print output to STDOUT
first = int(input())
first_set = set(map(int, input().split()))
second = int(input())
second_set = set(map(int, input().split()))

[print(x) for x in sorted(list(first_set.symmetric_difference(second_set)))]

# Set .add() - 28
# Enter your code here. Read input from STDIN. Print output to STDOUT
N = int(input())
my_stamps = list()
for _ in range(N):
    stamp = input()
    my_stamps.append(stamp)
print(len(set(my_stamps)))

# Set .discard(), .remove() & .pop() - 29
n = int(input())
s = set(map(int, input().split()))


N = int(input())
for _ in range(N):
    command = list(input().split())
    if command[0] == 'pop':
        s.pop()
    elif command[0] == 'remove':
        s.remove(int(command[1]))
    elif command[0] == 'discard':
        s.discard(int(command[1]))

print(sum(s))

# Set .intersection() Operation - 30
# Enter your code here. Read input from STDIN. Print output to STDOUT
M = int(input())
setA = set(map(int, input().split()))
N = int(input())
setB = set(map(int, input().split()))
INTER = setA.intersection(setB)
print(len(INTER))

# Set .union() Operation - 31
# Enter your code here. Read input from STDIN. Print output to STDOUT
M = int(input())
setA = set(map(int, input().split()))
N = int(input())
setB = set(map(int, input().split()))
Union = setA.union(setB)
print(len(Union))

# Set .difference() Operation - 32
# Enter your code here. Read input from STDIN. Print output to STDOUT
M = int(input())
setA = set(map(int, input().split()))
N = int(input())
setB = set(map(int, input().split()))
Diff = setA.difference(setB)
print(len(Diff))

# Set .symmetric_difference() Operation - 33
# Enter your code here. Read input from STDIN. Print output to STDOUT
M = int(input())
setA = set(map(int, input().split()))
N = int(input())
setB = set(map(int, input().split()))
sym_dif = setA.symmetric_difference(setB)
print(len(sym_dif))

# Set Mutations - 34
# Enter your code here. Read input from STDIN. Print output to STDOUT

A = int(input())
set_A = set(map(int, input().split()))
N = int(input())
for _ in range(N):
    cmd, *args = input().split()
    setB = set(map(int, input().split()))
    if cmd == 'intersection_update':
        set_A.intersection_update(setB)
    elif cmd == 'update':
        set_A.update(setB)
    elif cmd == 'symmetric_difference_update':
        set_A.symmetric_difference_update(setB)
    elif cmd == 'difference_update':
        set_A.difference_update(setB)

print(sum(set_A))

# Check Subset - 35
# Enter your code here. Read input from STDIN. Print output to STDOUT
T = int(input())
for i in range(T):
    LenA = int(input())
    setA = set(map(int, input().split()))
    LenB = int(input())
    setB = set(map(int, input().split()))

    if len(setA - setB) == 0:
        print("True")
    else:
        print("False")

# The Captain's Room - 36 - Examined the Discussions
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import Counter
K = input()
mylist = input().split()
countdict = Counter(mylist)
for k,v in countdict.items():
    if v == 1:
        print (k)

# Check Strict Superset - 37
# Enter your code here. Read input from STDIN. Print output to STDOUT
setA = set(map(int, input().split()))
N = int(input())
superset = True 

for _ in range(N):
    other_set = set(map(int, input().split()))
    if not setA.issuperset(other_set):
        superset = False
        break 

print(superset)

# collections.Counter() - 38
from collections import Counter
X = int(input())
sizes = Counter(map(int, input().split()))
nofcustomers = int(input())
amount = 0
for _ in range(nofcustomers):
    size, price = map(int, input().split())
    if sizes[size]:
        amount += price
        sizes[size] -= 1
print(amount)

# DefaultDict Tutorial - 39
from collections import defaultdict
n, m = map(int, input().split())
group_A = defaultdict(list)

for i in range(1, n + 1):  # 1 - indexed
    word = input().strip()
    group_A[word].append(i)

for _ in range(m):
    word = input().strip()
    if word in group_A:
        print(' '.join(map(str, group_A[word])))
    else:
        print(-1)

# Collections.namedtuple() - 40
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import namedtuple
N = int(input())
columns = input().split()
students = namedtuple('students', columns)
total_marks = 0
for _ in range(N):
    MARKS, CLASS, NAME, ID = input().split()
    student = students(MARKS, CLASS, NAME, ID)
    total_marks += int(student.MARKS)
print(round(total_marks/N, 2))

# Collections.OrderedDict() - 41
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import OrderedDict
order = OrderedDict()
listofitems = int(input())
for _ in range(listofitems):
    input_line = input()
    item, price = input_line.rsplit(' ', 1)
    order[item] = order.get(item, 0) + int(price)
    
for item, price in order.items():
    print(item, price)

# Collections.deque() - 42
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import deque
N = int(input())
d = deque()
for _ in range(N):
    commands = input().strip().split()
    if (commands[0] == 'append'):
        d.append(commands[1])
    elif (commands[0] == 'pop'):
        d.pop()
    elif (commands[0] == 'popleft'):
        d.popleft()
    elif (commands[0] == 'appendleft'):
        d.appendleft(commands[1])
result = ' '.join(d)
print(result)

# Word Order - 43
from collections import Counter
n = int(input())
words_list = []

for i in range(n):
    words_list.append(input().strip())

counts = Counter(words_list)
print(len(counts))
print(*counts.values())

# Company Logo - 44
from collections import Counter

if __name__ == '__main__':
    S = input()
    S = sorted(S)
    freq = Counter(list(S))
    for x, y in freq.most_common(3):
        print(x, y)

# Calendar Module - 45
# Enter your code here. Read input from STDIN. Print output to STDOUT
import calendar
mm, dd, yy = map(int, input().strip().split())
day_of_week = calendar.weekday(yy, mm, dd)
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
print(days[day_of_week].upper())

# Time Delta - 46
from datetime import datetime
def time_delta(t1, t2):
# a: abb weekday, d: day of the month, b: abb month name, Y: year, H: hours, M: minutes, S: seconds, z: zone
    time_format = '%a %d %b %Y %H:%M:%S %z'
    t1 = datetime.strptime(t1, time_format) # parse_time
    t2 = datetime.strptime(t2, time_format)
    return str(int(abs((t1-t2).total_seconds()))) 

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    t = int(input())
    for t_itr in range(t):
        t1 = input()
        t2 = input()
        delta = time_delta(t1, t2)
        fptr.write(delta + '\n')
    fptr.close()

# Exceptions - 47
T = int(input())
for _ in range(T):
    try:
        a,b = map(int,input().split())
        print(int(a/b))
    except ZeroDivisionError:
        print("Error Code:"+" integer division or modulo by zero")
    except ValueError as ve:
        print("Error Code:", ve)

# Zipped! - 48
N, X = map(int, input().split())
score = []
for _ in range(X):
    score.append(list(map(float, input().split())))
for stud in list(zip(*score)):
    print("%.1f" % (sum(stud) / len(stud)))

# Athlete Sort - 49
rows, cols = map(int, input().split())
table = []

for i in range(rows):
    line = input().split()
    table.append([int(x) for x in line])
    
sort_key = int(input())
table.sort(key = lambda x: x[sort_key])

for row in table:
    print(' '.join(str(x) for x in row))

# ginortS - 50

string = input().strip()
lowercase_letters = []
uppercase_letters = []
odd_digits = []
even_digits = []

for x in string:
    if x.islower():
        lowercase_letters.append(x)
    elif x.isupper():
        uppercase_letters.append(x)
    elif x.isnumeric():
        if int(x) % 2 == 0:
            even_digits.append(x)
        else:
            odd_digits.append(x)

lowercase_letters.sort()
uppercase_letters.sort()
odd_digits.sort()
even_digits.sort()

sorted_str = ''.join(lowercase_letters + uppercase_letters + odd_digits + even_digits)
print(sorted_str)

# Map and Lambda Function - 51
cube = lambda x: x ** 3

def fibonacci(n):
    a, b = 0, 1
    result = []
    for _ in range(n):
        result.append(a)
        a, b = b, a + b
    return result

# Detect Floating Point Number - 52
import re
T = int(input())
for _ in range(T):
    string = str(input())
    pattern = r"^[\.\+\-\d]\d*\.\d+$"
    match = re.match(pattern, string)
    if match:
        print("True")
    else:
        print("False")

# Re.split() - 53
regex_pattern = r"[.,]"

# Re.findall() & Re.finditer() - 54
string = input()
vowels = "aeiou"
consonants = "qwrtypsdfghjklzxcvbnm"

# ?<= a look behind; ?= a look ahead
pattern = rf'(?<=[{consonants}])([{vowels}]{{2,}})(?=[{consonants}])'
matches = re.findall(pattern, string, re.IGNORECASE)
if matches:
    print("\n".join(matches))
else:
    print("-1")

# Re.start() & Re.end() - 55
string = input()
substring = input()

pattern = re.compile(substring)
match = pattern.search(string)

if not match:
    print('(-1, -1)')

while match:
    print('({0}, {1})'.format(match.start(), match.end() - 1))
# The end() method returns the index just after the last character of the match, 
# To get the index of the last character, match.end() - 1 is used.
    match = pattern.search(string, match.start() + 1)

# Regex Substitution - 56
# Enter your code here. Read input from STDIN. Print output to STDOUT

n = int(input())
pattern = re.compile(r'(?<= )(&&|\|\|)(?= )')

def replace_operator(match):
    if match.group() == '&&':
        return 'and'
    elif match.group() == '||':
        return 'or'

for i in range(n):
    s = input()
    newstr = pattern.sub(replace_operator, s)
    print(newstr)

# Validating Roman Numerals - 57 - used support of ChatGPT
regex_pattern = r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$"

# Validating phone numbers - 58
N = int(input())
for _ in range(N):
    number = input()
    if re.match(r'^[789]\d{9}$', number):
        print('YES')
    else:
        print('NO')

# Validating and Parsing Email Addresses - 58
# Enter your code here. Read input from STDIN. Print output to STDOUT
pattern = r'^<[a-zA-Z][a-zA-Z0-9._-]*@[a-zA-Z]+\.[a-zA-Z]{1,3}>$'
for _ in range(int(input())):
    name, email = input().split(' ')
    if re.match(pattern, email):
        print(name, email)

# Hex Color Code - 59 - took help from tutorials on regex for python and gemini
# Enter your code here. Read input from STDIN. Print output to STDOUT
N = int(input())
for _ in range(N):
    css_code = input()
    # ?i case-insensitive; ?: non-capturing group
    pattern = r'(?i)#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})(?=[;,)])'
    matches = re.findall(pattern, css_code)
    if matches:
        print("\n".join(list(matches)))


# Group(), Groups() & Groupdict() - 60
# Search for consecutive characters that are the same
# ([a-zA-Z0-9]) caputures a character as group 1 and (?=\1) is a look ahead assertion that the same 
# character is ahead
found = re.search(r'([a-zA-Z0-9])(?=\1)', input().strip())
if found != None:
    print(found.group(1))
else:
    print('-1')

# HTML Parser - Part 1 - 61 - Took help from discussions on HackerRank
# Enter your code here. Read input from STDIN. Print output to STDOUT
from html.parser import HTMLParser
class Parse_HTML(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print("Start :", tag)
        self.print_tag_attrs(attrs=attrs)
    def handle_endtag(self, tag):
        print("End   :", tag)
    def handle_startendtag(self, tag, attrs):
        print("Empty :", tag)
        self.print_tag_attrs(attrs=attrs)
    def print_tag_attrs(self, attrs):
        [print(f"-> {x[0]} > {x[1]}") for x in attrs]

parser = Parse_HTML()
parser.feed(''.join([input() for _ in range(int(input()))]))


# HTML Parser - Part 2 - 62
class MyHTMLParser(HTMLParser):
    def handle_comment(self, data):
        if '\n' in data:
            print(">>> Multi-line Comment")
            print(data)
        else:
            print(">>> Single-line Comment")
            print(data)
            
    def handle_data(self, data):
        if '\n' not in data:
            print(">>> Data")
            print(data)

html = ""
for i in range(int(input())):
    html += input().rstrip()
    html += '\n'
    
parser = MyHTMLParser()
parser.feed(html)
parser.close()

# Detect HTML Tags, Attributes and Attribute Values - 63
# Enter your code here. Read input from STDIN. Print output to STDOUT
class ParseHTML(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print(tag)
        if attrs:
            [print(f'-> {attr[0]} > {attr[1]}') for attr in attrs]
    
    def handle_startendtag(self, tag, attrs):
        print(tag)
        if attrs:
            [print(f'-> {attr[0]} > {attr[1]}') for attr in attrs]
  
html = ""
for _ in range(int(input())):
    html += input().rstrip()
    html += '\n'
    
parser = ParseHTML()
parser.feed(html)
parser.close()

# Validating UID - 64
# Enter your code here. Read input from STDIN. Print output to STDOUT
def is_valid_uid(uid):
    if len(set(uid)) != 10:
        return False
    if len(re.findall(r'[A-Z]', uid)) < 2:
        return False
    if len(re.findall(r'[0-9]', uid)) < 3:
        return False
    if not re.match(r'^[a-zA-Z0-9]*$', uid):
        return False
    if len(set(uid)) != 10:
        return False
    return True

t = int(input())

for _ in range(t):
    uid = input().strip()
    if is_valid_uid(uid):
        print('Valid')
    else:
        print('Invalid')

# Validating Credit Card Numbers - 65
# Enter your code here. Read input from STDIN. Print output to STDOUT

def is_valid_card(card_number):
    pattern = r'^[456]\d{3}-?\d{4}-?\d{4}-?\d{4}$'
    if not re.match(pattern, card_number):
        return False
    card_number_digits = card_number.replace('-', '')
    if re.search(r'(\d)\1{3,}', card_number_digits):
        return False
    return True

N = int(input())
for _ in range(N):
    card_data = input().strip()
    if is_valid_card(card_data):
        print("Valid")
    else:
        print("Invalid")

# Validating Postal Codes - 66
regex_integer_in_range = r"^[100000-999999]{6}$"
regex_alternating_repetitive_digit_pair = r"(\d)(?=\d\1)"

# XML 1 - Find the Score - 67
def get_attr_number(node):
    a=0
    for child in node:
        a+=(get_attr_number(child))
    return (len(node.attrib) + a)


# XML2 - Find the Maximum Depth - 68
maxdepth = 0
def depth(elem, level):
    global maxdepth
    level += 1  # because initial level is -1 in the already given code
    if level > maxdepth:
        maxdepth = level
    for child in elem:
        depth(child, level)

# Arrays - 68
import numpy as np
def arrays(arr):
    newarray = np.array(arr[::-1], float)
    return newarray

# Shape and Reshape - 69
list_ints = list(map(int, input().strip().split()))
array = np.array(list_ints)
print(np.reshape(array, (3, 3)))

# Transpose and Flatten - 70
n,m = map(int,input().split())
arr = []
for i in range(n):
    row = list(map(int,input().split()))
    arr.append(row)
nparr = np.array(arr)
print(np.transpose(nparr))
print(nparr.flatten())

# Concatenate - 71
n, m, p = map(int, input().split())

lista1=[list(map(int, input().split())) for i in range(n)]
lista2=[list(map(int, input().split())) for i in range(m)]

a1=np.array(lista1)
a2=np.array(lista2)

print(np.concatenate((a1, a2)))

# Zeros and Ones - 72
size = list(map(int, input().split()))

zeros_array = np.zeros(size, dtype= int)
ones_array = np.ones(size, dtype= int)

print(zeros_array)
print(ones_array)

# Eye and Identity - 73
np.set_printoptions(legacy='1.13')
N,M = map(int, input().split())

print(np.eye(N,M))

# Array Mathematics - 74
N, M = list(map(int, input().split()))
A, B = [], []

for _ in range(N):
    A.append([int(i) for i in input().strip().split()])
for _ in range(N):
    B.append([int(i) for i in input().strip().split()])

A = np.array(A)
B = np.array(B)

print(A + B)
print(A - B)
print(A * B)
print(A // B)
print(A % B)
print(A ** B)

# Floor, Ceil and Rint - 75
np.set_printoptions(legacy='1.13')
array_a = np.array(list(map(float, input().split())))

print(np.floor(array_a))
print(np.ceil(array_a))
print(np.rint(array_a))

# Sum and Prod - 76
n = list(map(int, input().split()))
x = []
for i in range(n[0]):
    x.append(list(map(int, input().split())))
nparr = np.array(x, int)
thesum = np.sum(nparr, axis=0)
print(np.prod(thesum))

# Min and Max - 77
N,M = map(int, input().split())

arr = list()
for _ in range(N):
    arr.append(list(map(int, input().split())))
np_arr = np.array(arr)

print( np.max(np.min(np_arr, axis = 1) ))

# Mean, Var, and Std - 78
n, m = tuple(map(int, input().split()))

A = np.array([list(map(int, input().split())) for j in range(n)])

print(np.mean(A,axis=1))
print(np.var(A,axis=0))
stdnp = np.std(A)
print(round(stdnp,11))

# Dot and Cross - 79
N = int(input())
arr_a = list()
arr_b = list()
for _ in range(N):
    arr_a.append(list(map(int, input().split())))
for _ in range(N):
    arr_b.append(list(map(int, input().split())))

np_a = np.array(arr_a)
np_b = np.array(arr_b)

print(np.matmul(np_a, np_b))

# Inner and Outer - 80
A=list(map(int,input().split()))
B=list(map(int,input().split()))
nparr = np.array(A,int)
nparr2 = np.array(B,int)
print(np.inner(nparr,nparr2))
print(np.outer(nparr,nparr2))

# Polynomials - 81
N = list(map(float,input().split()))
x = float(input())

result = np.polyval(N,x)
print(result)

# Linear Algebra - 82
N = int(input())
l = []
for _ in range(N):
    l.append(list(map(float, input().strip().split())))

nparr = np.array(l)
nparr = nparr.reshape((N, N))
result = np.linalg.det(nparr)
print(round(result, 2))

# Birthday Cake Candles - 83

# Complete the 'birthdayCakeCandles' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER_ARRAY candles as parameter.

def birthdayCakeCandles(candles):
    # write your code here
    counter = Counter(candles)
    most_common_item = counter.most_common(1)
    return most_common_item[0][1]

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    candles_count = int(input().strip())

    candles = list(map(int, input().rstrip().split()))

    result = birthdayCakeCandles(candles)

    fptr.write(str(result) + '\n')

    fptr.close()

# Number Line Jumps - 84

def kangaroo(x1, v1, x2, v2):
    # same veloc but diff start
    if v1 == v2:
        return "YES" if x1 == x2 else "NO"
    
    # difference in positions divisible by (?) the difference in velocities
    if (x2 - x1) % (v1 - v2) == 0 and (x2 - x1) / (v1 - v2) >= 0:
        return "YES"
    else:
        return "NO"

 
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    x1V1X2V2 = input().split()

    x1 = int(x1V1X2V2[0])

    v1 = int(x1V1X2V2[1])

    x2 = int(x1V1X2V2[2])

    v2 = int(x1V1X2V2[3])

    result = kangaroo(x1, v1, x2, v2)

    fptr.write(result + '\n')

    fptr.close()

# Viral Advertising - 85

# Complete the 'viralAdvertising' function below.
# The function is expected to return an INTEGER.
# The function accepts INTEGER n as parameter.

def viralAdvertising(n):
    count = 0
    k = 5//2
    
    for i in range(n):
        count = count + k
        k = (k*3)//2
    
    return count

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input().strip())

    result = viralAdvertising(n)

    fptr.write(str(result) + '\n')

    fptr.close()

# Recursive Digit Sum - 86

# Complete the 'superDigit' function below.
# The function is expected to return an INTEGER.
# The function accepts following parameters:
#  1. STRING n
#  2. INTEGER k


def superDigit(n, k):
    
    numbers = map(int, list(n))
    short = sum(numbers)
    
    def recurse(summed):
        if summed < 10:
            return summed
        else:
            string = str(summed)
            numbers = map(int, list(string))
            return recurse(sum(numbers))
    
    return recurse(short * k)



if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    n = first_multiple_input[0]

    k = int(first_multiple_input[1])

    result = superDigit(n, k)

    fptr.write(str(result) + '\n')

    fptr.close()


# Insertion Sort - Part 1 - 87
# Complete the 'insertionSort1' function below.
# The function accepts following parameters:
#  1. INTEGER n
#  2. INTEGER_ARRAY arr

def insertionSort1(n, arr):
    # write your code here
    
    # Loop starts from the last element of the array and moves backward
    for i in range(n - 1, 0, -1):
        val = arr[i]  # Store the value of the current element
        j = i - 1     # Set j to be the index just before i
        while j >= 0 and val < arr[j]:
            arr[j+1] = arr[j]  # Shift the larger element to the right
            print(*arr)  # Print the array after every shift to see the process
            j -= 1       # Move one step back to continue checking
    
        # Place the value (val) in the correct position after the shifting
        arr[j + 1] = val
    print(*arr)

if __name__ == '__main__':
    n = int(input().strip())
    arr = list(map(int, input().rstrip().split()))
    insertionSort1(n, arr)
