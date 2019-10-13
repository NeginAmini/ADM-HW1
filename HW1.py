#exercise 1 - print Hello World!
print("Hello, World!")


#exercise 2 - Python If-Else
n = int(input(""))
if n % 2 != 0 :
    print("Weird")
else :
    if n in range(2, 6) or n > 20 :
        print("Not Weird")
    else:
        print("Weird")


#exercise 3 - Arithmetic Operators
a = int(input())
b = int(input())
print(a+b)
print(a-b)
print(a*b)

#exercise 4 - Division
a = int(input())
b = int(input())
print(a//b)  #integer division
print(a/b)

#exercise 5 - Loops
N = int(input())
for i in range(0,N) :
    print(i**2)

#exercise 6 - write a function(leap year)
def is_leap(year):
    if year % 4 == 0 :
        if year % 100 ==0:
            if year % 400 == 0 :
                return(True) #if devisible by 400
            else:
                return(False) #if divisible by 100 but not 400
        else :
            return(True) #if divisible by 4 and not by 100
    else:
        return(False) #if not divisible by 4
year = int(input())
print(is_leap(year))


#exercise 7 - print function - read integer N, print 123..N
n = int(input())
i = 1
while i <= n  :
    print(i , end ="")  #end function ends the output with"" instead of new line
    i += 1


#exercise 8 - list comprehension
x = int(input())
y = int(input())
z = int(input())
n = int(input())
L = [ [ i , j , k] for i in range(0,x+1) for j in range(0,y+1) for k in range(0,z+1) if i + j + k != n  ]
print(L)


#exercise 9 - find the runner-up score

n = int(input())
S = set(map(int , input().split()))   # to avoid repeated values, we put input to set not list
# to find the second max we can find the first max, eliminate it , and find
# max of the new set
a = max(S)
S.remove(a)
print(max(S))


#exercise 10 -  Nested lists
N = int(input())
L = [[input(),float(input())] for i in range(0 , N)]
marks = [x[1] for x in L] #put the second element of each argument(grades) in a list
set_marks = set(marks) #to find the second min make a set of the grades to avoid repeatable values
set_marks.remove(min(set_marks))
a = min(set_marks)
names = [x[0] for x in L if x[1] == a]
s = sorted(names)
for i in s :
    print(i)



# exercise 11 - finding the percentage of grades
if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for i in range(n):
        L = list(input().split())
        name = L[0]
        scores = list(map(float , L[1:]))
        student_marks[name] = scores
percentage = sum(student_marks[input()])/3
print ("%2.2f" % percentage)  # "%2.2f" means 2 characters and 2 decimal places


#exercise 12 - lists - remove/append/pop/sort
L = []
result = []
N = int(input())
for i in range(N):
    L = input().split()
    if L[0] == "insert" :
        position = int(L[1])
        integer = int(L[2])
        result.insert(position,integer)
    if L[0] == "remove" :
        rem_int = int(L[1])
        result.remove(rem_int)
    if L[0] == "append" :
        append_int = int(L[1])
        result.append(append_int)
    if L[0] == "sort" :
        result.sort()
    if L[0] == "pop" :
        result.pop()
    if L[0] == "reverse" :
        result.reverse()
    if L[0] == "print" :
        print(result)





# exercise 13 _ tuple - hash
n = int(input())
integer_list = map(int, input().split()) # make a list of integers
t = tuple(integer_list)  # convert list to tuple
print(hash(t))



# exercise 14 swap case

def swap_case(s):
    return s.swapcase()

if __name__ == '__main__':
    s = input()
    result = swap_case(s)
    print(result)



#exercise 15 - string split and join

def split_and_join(line):
    a = line.split() #put every character of string in a list
    b = "-".join(a)
    return b

if __name__ == '__main__':
    line = input()
    result = split_and_join(line)
    print(result)

#exercise 16 - what's your name

def print_full_name(a, b):
    print("Hello " + first_name + " " + last_name +"! You just delved into python.")
if __name__ == '__main__':
    first_name = input()
    last_name = input()
    print_full_name(first_name, last_name)


#exercise 17 - Mutations

def mutate_string(string, position, character):
    L = list(string)
    L[position] = character
    return "".join(L)


if __name__ == '__main__':
    s = input()
    i, c = input().split()
    s_new = mutate_string(s, int(i), c)
    print(s_new)


#exercise 18 - find a string

def count_substring(string, sub_string):
    n = 0
    for i in range(0, len(string)):
        if string.find(sub_string, i, i + len(sub_string)) != -1:
            n += 1
    return (n)


if __name__ == '__main__':
    string = input().strip()
    sub_string = input().strip()

    count = count_substring(string, sub_string)
    print(count)



#exercise 19 - string validators

if __name__ == '__main__':
    s = input()
m = 0
k = 0
p = 0
j = 0
f = 0
for i in range(0 , len(s)) :
    if s[i].isalnum() == True :
        m += 1
        break
print(bool(m))


for i in range(0 , len(s)) :
    if s[i].isalpha() == True :
        k += 1
        break
print(bool(k))



for i in range(0 , len(s)) :
    if s[i].isdigit() == True :
        p += 1
        break
print(bool(p))


for i in range(0 , len(s)) :
    if s[i].islower() == True :
        j += 1
        break
print(bool(j))


for i in range(0 , len(s)) :
    if s[i].isupper() == True :
        f += 1
        break
print(bool(f))



#exercise 20 - Text alignment
thickness = int(input()) #This must be an odd number
c = 'H'

#Top Cone
for i in range(thickness):
    print((c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1))

#Top Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))

#Middle Belt
for i in range((thickness+1)//2):
    print((c*thickness*5).center(thickness*6))

#Bottom Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))

#Bottom Cone
for i in range(thickness):
    print(((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6))



#exercise 21 -  text wrap

def wrap(string, max_width):
    return ('\n'.join(string[i * max_width:i * max_width + max_width] for i in range(len(string) // max_width + 1)))
if __name__ == '__main__':
    string, max_width = input(), int(input())
    result = wrap(string, max_width)
    print(result)


#exercise 22 - designing door mat
N , M = map( int , input().split())
for i in range((N-1)//2) :
    print((((M-3)//2)-3*i)* "-" + (2*i +1) * ".|." + (((M-3)//2)-3*i) * "-")

print(((M-7)//2)*"-" + "WELCOME" + ((M-7)//2)*"-")

for i in range( (N-1)//2)[::-1] :
    print((((M-3)//2)-3*i)* "-" + (2*i +1) * ".|." + (((M-3)//2)-3*i) * "-")
# I splited the problem into 3 parts: parts before welcome and after welcome have the same pattern so I just reversed the counter for them


#exercise 23 - string formatting

def print_formatted(number):
    m = len(bin(n)) - 2  # we should calculate the lenght of the binary format, the two first charecters (0b) should be eliminated
    for i in range(1,n+1) :
        print (str(i).rjust(m) , str(oct(i))[2:].rjust(m) , str(hex(i)[2:]).rjust(m).upper() ,  str(bin(i)[2:]).rjust(m) )  #use rjust to format the output
if __name__ == '__main__':
    n = int(input())
    print_formatted(n)


#exercise 24 - capitaliza
def solve(s):
    for i in s.split():  # if we don't use split() it is gonna capitalize every letter
        s = s.replace(i,
                      i.capitalize())  # we should use replace function to get spaces in ouput string exactly like input
    return s
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    s = input()

    result = solve(s)

    fptr.write(result + '\n')

    fptr.close()



#exercise 25 - The minion game
def minion_game(string):
    vowels = "AEIOU"
    s_num = 0
    k_num = 0
    for i in range(len(
            string)):  # the number of sub strings which starts with vowels is equal to (len(string) - index of vowel) for each vowel
        if s[i] in vowels:
            k_num += len(string) - i

        else:
            s_num += len(string) - i
    if k_num > s_num:
        print("Kevin " + str(k_num))
    elif s_num > k_num:
        print("Stuart " + str(s_num))
    else:
        print("Draw")
if __name__ == '__main__':
        s = input()
        minion_game(s)


#exercise 26 - merge the tools
def merge_the_tools(string, k):
    for i in range(len(string) // k):  # we should repeat the process len(string)/k times to reach all the string
        s = ""
        for j in string[i * k: i * k + k]:

            if j not in s:
                s += j
        print(s)
if __name__ == '__main__':
    string, k = input(), int(input())
    merge_the_tools(string, k)

#exercise 27 - intro to sets
def average(array):
    s = set(array)  #to avoid repeated values we put it in a set
    return sum(s) / len(s)
if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().split()))
    result = average(arr)
    print(result)

#exercise 28 - No idea
n , m = map(int, input().split())
L = list(map(int, input().split()))
A = set(map(int, input().split()))
B = set(map(int, input().split()))
num_happy = 0
for i in L :
    if i in A :
        num_happy += 1
    if i in B :
        num_happy -= 1
print(num_happy)

#exercise 29 - set.add()

N = int(input())
s = set() #to avoid repeated values we put it in a set
for i in range(0,N) :
    s.add(input())

print(len(s))


#exercise 30 - set remove() discard() pop()

n = int(input())
s = set(map(int, input().split()))
N = int(input())

for i in range(0, N):
    L = list(input().split())
    if L[0] == "pop"  and len(s) > 0 :
        s.pop()

    if L[0] == "remove" and int(L[1]) in s:
        s.remove(int(L[1]))


    if  L[0] == "discard":
        s.discard(int(L[1]))
print(sum(s))


#exercise 31 - set.uninion()
NE = int(input())
E = set(map(int, input().split()))
NF = int(input())
F = set(map(int, input().split()))
union = set(E.union(F))
print(len(union))


#exercise 32 - set.intersection()
NE = int(input())
E = set(map(int, input().split()))
NF = int(input())
F = set(map(int, input().split()))
union = set(E.intersection(F))
print(len(union))

#exercise 33 - set.difference()

NE = int(input())
E = set(map(int, input().split()))
NF = int(input())
F = set(map(int, input().split()))
union = set(E.difference(F))
print(len(union))


#exercise 34 - set.symmetric difference
NE = int(input())
E = set(map(int, input().split()))
NF = int(input())
F = set(map(int, input().split()))
union = set(E.symmetric_difference(F))
print(len(union))

#exercise 35 - set mutations
num_A = int(input())
A = set(map(int, input().split()))
N = int(input())
for i in range(0 , N) :
    L = input().split()
    S = set(map(int, input().split()))
    if L[0] == "intersection_update" :
        A.intersection_update(S)
    if L[0] == "update" :
        A.update(S)
    if L[0] == "symmetric_difference_update" :
        A.symmetric_difference_update(S)
    if L[0] == "difference_update" :
        A.difference_update(S)
print(sum(A))


#exercise 36 - capitan's room
k = int(input())
L = list(map(int, input().split()))
s = set(L)
print((sum(s)*k - sum(L))// (k-1))


#exercise 37 - check subset
N = int(input())
for i in range(0, N):
    NA= int(input())
    A = set(map(int, input().split()))
    NB= int(input())
    B = set(map(int, input().split()))
    print(A.issubset(B))

#exercise 38 - chech strict superset
A = set(map(int, input().split()))
N = int(input())
m = 1
for i in range(0, N):
    s = set(map(int, input().split()))
    if( s.issubset(A) == False) or  (len(A) - len(s) < 1) : # s should be a subset of A and A should have at least1 member more than s
        m *= 0
print(bool(m))


#exercise 39 - collection.counter()
X = int(input())
L = list(map(int, input().split()))
N = int(input())
money = 0
for i in range(0 , N) :
    customer = list(map(int, input().split()))
    if customer[0] in L :
        money += customer[1]
        L.remove(customer[0])
print(money)



#exercise 40 - Defualt dict
from collections import defaultdict
N , M = map(int, input().split())
A = defaultdict(list)
B = []
for i in range(0,N):
    A[input()].append(i+1)
for i in range (0 ,M ) :
    B.append(input())
for i in B :
    if i in A :
        print(" ".join(map(str, A[i])))
    else :
        print("-1")


#exercise 40 -  collections.namedtuple()

from collections import namedtuple
sum = 0
N = int(input())
catg = input().split()
student_tup = namedtuple('student',catg)
for i in range(0, N):
    L = list(input().split())
    m = student_tup(L[0], L[1], L[2], L[3])
    sum += int(m.MARKS)

print(sum/N)



#exercise 41 - collections.deque()
from collections import deque
d = deque()
N = int(input())
for i in range(N) :
    L = list((input().split()))
    if L[0] == "append" :
        d.append(L[1])
    if L[0] == "appendleft" :
        d.appendleft(L[1])
    if L[0] == "pop" :
        d.pop()
    if L[0] == "popleft" :
        d.popleft()
[print(x, end=' ') for x in d] # to put output in one line should use end = " "



#exercise 42 - company Lego
from collections import Counter
import math
import os
import random
import re
import sys
s = sorted(input()) #instead of sorting the output which ic hard we sort input!
a = dict((Counter(s).most_common(3)))
for key , value in a.items() :
    print(" ".join((str(key) , str(value)) ))


#exercise 43 - calendar module

import calendar
L = list(map(int, input().split()))
a = (calendar.day_name)[calendar.weekday(L[2],L[0],L[1])]
print(a.upper())


#exercise 44 - exception
for i in range(0 , int(input())):
    try:
        a, b = map(int, input().split())
        m = a // b
        print(m)
    except ZeroDivisionError as e:
        print("Error Code:", e)
    except ValueError as e:
        print("Error Code:", e)


#exercise 45 - incorrect Regex

import re

n = int(input())
for i in range(0, n):
    try:
        is_reg = True
        re.compile(input())  # if re.compile raise an error is_reg get False value

    except:
        is_reg = False
    print(is_reg)



#exercise 46 - zipped
L = []
N, M = map(int, input().split())
for i in range(M):
    L.append(list(map(float, input().split())))
for i in zip(*L) :
    print(sum(i)/len(i))


#exercise 47 - input()

x , k = map(int,input().split())
p = input()
if k == eval(p) :
    print(True)
else :
    print(False)


#exercise 48 -  python evaluation
eval(input())


#exercise 49 - Any or All
N = int(input())
L =  input().split()


print( all([int(i) > 0 for i in L]) and any([i[::-1] == i for i in L]))


#exercise 50 - map and lambda function

cube = lambda x: x**3

def fibonacci(n):
    L = [0,1]
    for i in range(2,n) :
        L.append(L[i-2] + L[i-1])
    return L[:n] # put index to make the code work for n =1


#exercise 51 - reduce function
from fractions import Fraction
from functools import reduce

def product(fracs):
    t = reduce(lambda x , y : x * y , fracs)
    return t.numerator, t.denominator

if __name__ == '__main__':
    fracs = []
    for _ in range(int(input())):
        fracs.append(Fraction(*map(int, input().split())))
    result = product(fracs)
    print(*result)


#exercise 52 - detect floating point number
import re
T = int(input())
for i in range(T) :
    N = input()
    print (bool(re.match('^[-+]?[0-9]*\.[0-9]+$' , N))) # ? means zero or one * means zero or more +means one or more


#exercise 53 - Re.split()
regex_pattern = r"[,.]"

import re
print("\n".join(re.split(regex_pattern, input())))



#exercise 54 - group() groupdict()
import re
m = re.findall(r"([A-Za-z0-9])\1+",input())  #\1 refers to first match group
if bool(m) :
    print(m[0])
else:
    print(-1)


#exercise 55 - re.start() re.rnd()

import re

s, k = input(), input()
if not re.search(k, s):  # if we could not find the string k in s
    print('(-1, -1)')
else:
    i = 0
    while re.search(k, s[i:]):
        i += re.search(k, s[i:]).start() + 1
        print('(', i - 1, ', ', i + len(k) - 2, ')',
              sep='')  # to skip the difficulty of putting values
        # in tuple and then print tuple, we print parantheses seprated


#exercise 56 - validating phone numbers
import re
n = int(input())
for i in range(n):
    if re.match(r'^[789]\d{9}$',input()):  # ^ means start  \d means integer  {9} means 9 times
        print("YES")

    else:
        print("NO")

#exercise 57 - numpy array
import numpy


def arrays(arr):
    a = numpy.array(arr, float)
    return a[::-1]


arr = input().strip().split(' ')
result = arrays(arr)
print(result)



#exercise 58 - shape and reshape
import numpy
L = numpy.array(list(map(int, input().split())))
L.shape = (3,3)
print(L)


#exercise 59 - transpose and flatten
import numpy
L = []
N , M = map(int, input().split())
for i in range(0 ,N) :
    L.append(  input().split())

arr = numpy.array(L, int)

print(arr.transpose())
print(arr.flatten())

# exercise 60 - concatenate
import numpy

N, M, P = map(int, input().split())

A = numpy.array([input().split() for i in range(0,N)], int)

B = numpy.array([input().split() for i in range(0,M)], int)
print (numpy.concatenate((A,B))) #axis = 0 by default


#exercise 61 - zeros and ones
import numpy
L = list(map(int, input().split()))
print(numpy.zeros(L, dtype = numpy.int))
print(numpy.ones(L, dtype = numpy.int))


#exercise 62 - identity and eye
import numpy
N , M = map(int, input().split())
numpy.set_printoptions(sign=' ')
print(numpy.eye(N,M))

#exercise 63 - Array mathematics
import numpy
N , M = map(int, input().split())
A = numpy.array([input().split() for i in range(0, N)] , int)
B = numpy.array([input().split() for i in range(0, N)] , int)
print (A + B)
print (A - B)
print (A * B)
print (A // B)
print (A % B)
print (A ** B)

#exercise 64 - floor ceil and rent
import numpy
arr = list(map(float, input().split()))
print(str(numpy.floor(arr)).replace('.', '. ').replace('[', '[ ').replace('. ]', '.]'))
print(str(numpy.ceil(arr)).replace('.', '. ').replace('[', '[ ').replace('. ]', '.]'))
print(str(numpy.rint(arr)).replace('.', '. ').replace('[', '[ ').replace('. ]', '.]'))
# for making the output exactly like the expected output, I used replace fuction and add added space in specefic positions


#exercise 66 - sum and prod
import numpy
N , M = map(int, input().split())
arr = numpy.array([input().split() for i in range(0,N)] , int)
print(numpy.prod(numpy.sum(arr, axis = 0))) #put the sum of the array around axis 0 as the input of prod function


#exercise 67 - min and max
import numpy
N , M = map(int, input().split())
arr = numpy.array([input().split() for i in range(0, N)],int)
print(numpy.max(numpy.min(arr, axis = 1)))


#exercise 68 - mean , var and std
import numpy
N , M = map(int,input().split())
arr = numpy.array([input().split() for i in range(0,N)], int)
numpy.set_printoptions(legacy='1.13') # to fix the format of print
print(numpy.mean(arr, axis = 1))
print(numpy.var(arr , axis = 0))
print(numpy.std(arr))


#exercise 69 - dot and cross
import numpy
N = int(input())
A = numpy.array([input().split() for i in range(0,N)], int)
B = numpy.array([input().split() for i in range(0,N)], int)
print(numpy.dot(A,B))

#exercise 70 - inner & outer
import numpy
A = numpy.array([input().split()], int)
B = numpy.array([input().split()], int)
print (numpy.inner(A,B)[0][0]) # put [0][0] to remove barckets
print(numpy.outer(A,B))

#exercise 71 - polynomials
import numpy
L = list(map(float, input().split()))
x = float(input())
print (numpy.polyval(L , x))


#exercise 72 - linear algebra
import numpy

N = int(input())
arr = numpy.array([input().split() for i in range(0, N)], float)
numpy.set_printoptions(legacy='1.13')
print(numpy.linalg.det(arr))











QUESTION NUMBER 2




#2.1) Birthday Cake Candles
def count_highest_candle(arr) :
    s = set(arr) # TO avoid repeated values we make a set
    num= arr.count(max(s))
    print(num)




N = int(input())
candle_height = list(map(int,input().split()))
count_highest_candle(candle_height)



#2.2) kangaroo
L = list(map(int, input().split()))
if (L[1]-L[3] != 0 ) : #if they have same speed they never gonna meet
    if ((L[2] - L[0]) % (L[1]- L[3]) == 0) & ((L[1] - L[3]) * (L[2] - L[0]) >= 0) :
        #second condition is because if one of them is in a lower position and has lower speed they never gonna meet
        print("YES")
    else :
        print("NO")
else :
    print("NO")

#2.3) viral advertising
n = int(input())
shared_count = 5
liked_count = 2
cum = 2
for i in range(0, n-1) :
    shared_count = (shared_count//2) * 3
    liked_count = shared_count // 2
    cum = cum + liked_count
print(cum)


#2.4) insertion sort - part 1
n = int(input())
arr = list(map(int, input().split()))
m = arr[n - 1]
i = n - 1
while i > 0:
    if m < arr[i - 1]:
        arr[i] = arr[i - 1]
        arr2 = list(map(str, arr))
        print(' '.join(arr2))
        i -= 1

    else:
        arr[i] = m
        arr2 = list(map(str, arr))
        print(' '.join(arr2))
        i -= 1
        break

if m < arr[0]:  # to make the code work when m is the lowest number of the array
    arr[0] = m
    arr2 = list(map(str, arr))
    print(' '.join(arr2))




#insertion sort - part 2
import math
import os
import random
import re
import sys


def insertionSort2(n, arr):
    for i in range(1,n) :
        for j in range(0,i):
            if arr[i] < arr[j] :
                m = arr[j]     #switch position
                arr[j] = arr[i]
                arr[i] = m
        print(*arr)

if __name__ == '__main__':
    n = int(input())

    arr = list(map(int, input().rstrip().split()))

    insertionSort2(n, arr)




#2.6) recursive digit sum
x , k = map(int, input().split())
if k * x % 9 == 0 :
    m = 9
else :
    m = k * x % 9

while m >= 10 :

      m = m % 9
# this question was solved based on the fact that sum of elements in an integer is equal to
# the residual of diversion of that integer to 9

print(m)
