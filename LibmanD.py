#Решение задачи Дирихле методом Либмана для области произвольного очертания
#-------------------------Библиотеки--------------------------
import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d;
import numpy as np
#-------------------------------------------------------------
#------------Функции заданные внутри границы------------------
def f1 (x, y): 
    return 100

def f(x, y): 
    return 1000

def f3(x, y): 
    return 8*(math.cos(x+y)**2)-4
#-------------------------------------------------------------
#------------Функции заданные на границе----------------------
def Gran_Func(x, y): 
    return 100

def Gran_Func2(x, y):
    if x==0:
        return math.sin(y)**2
    elif x==2:
        return math.sin(y+2)**2
    if y==0:
        return math.sin(x)**2
    elif y==3:
        return math.sin(x+3)**2
    return 0

def Gran_Func3(x, y): 
    if x==0:
        return math.sin(y)
    elif x==1:
        return math.exp(1)*math.sin(y)
    if y==0:
        return 0
    elif y==1:
        return math.exp(x)*math.sin(1)
    return 0
#-------------------------------------------------------------
#------------Функции задания границы--------------------------
#необходимо задавать функции так ij если впуклая по оси ox и
# ji если впуклая по оси oy

def Search_Granica(m,h, n,l): #Окружность
    #m = 100
    #n = 100
    #h = 0.5
    #l = 0.5
    Granica=[]
    for i in range(m+1):
        for j in range(n+1):
            if 450 < ((i*h)-25)**2 + ((j*l)-25)**2< 500:
                Granica.append((i*h, j*l))
    return Granica

def Search_Granica2(m,h, n,l): #Окружность2 > ji
    #m = 100
    #n = 100
    #h = 0.5
    #l = 0.5
    Granica=[]
    for j in range(n+1):
        for i in range(m+1):
            if (i<80) and (j<=50) and 150 < ((i*h)-25)**2 + ((j*l)-15)**2< 200:
                Granica.append((i*h, j*l))
            if (i<80) and (j>=50) and 150 < ((i*h)-25)**2 + ((j*l)-35)**2< 200:
                Granica.append((i*h, j*l))
    return Granica

def Search_Granica3(m,h, n,l): #Окружность2 > ij
    #m = 100
    #n = 100
    #h = 0.5
    #l = 0.5
    Granica=[]
    for i in range(m+1):
        for j in range(n+1):
            if (i<80) and (j<=50) and 150 < ((i*h)-25)**2 + ((j*l)-15)**2< 200:
                Granica.append((i*h, j*l))
            if (i<80) and (j>=50) and 150 < ((i*h)-25)**2 + ((j*l)-35)**2< 200:
                Granica.append((i*h, j*l))
    return Granica

def Search_Granica4(m,h, n,l): #Окружность2 ^ ji
    #m = 100
    #n = 100
    #h = 0.5
    #l = 0.5
    Granica=[]
    for j in range(n+1):
        for i in range(m+1):
            if (j<80) and (i<=50) and 150 < ((j*h)-25)**2 + ((i*l)-15)**2< 200:
                Granica.append((i*h, j*l))
            if (j<80) and (i>=50) and 150 < ((j*h)-25)**2 + ((i*l)-35)**2< 200:
                Granica.append((i*h, j*l))
    return Granica

def Search_Granica5(m,h, n,l): #Окружность2 ^ ij
    #m = 100
    #n = 100
    #h = 0.5
    #l = 0.5
    Granica=[]
    for i in range(m+1):
        for j in range(n+1):
            if (j<80) and (i<=50) and 150 < ((j*h)-25)**2 + ((i*l)-15)**2< 200:
                Granica.append((i*h, j*l))
            if (j<80) and (i>=50) and 150 < ((j*h)-25)**2 + ((i*l)-35)**2< 200:
                Granica.append((i*h, j*l))
    return Granica 

def Search_Granica6(m,h, n,l): #Окружность4 > ij
    #m = 100
    #n = 100
    #h = 0.5
    #l = 0.5
    Granica=[]
    for i in range(m+1):
        for j in range(n+1):
            if (i<=50) and (j<=50) and ((40>i) or (45>j)) and 150 < ((i*h)-15)**2 + ((j*l)-15)**2< 170:
                Granica.append((i*h, j*l))
            if (i<=50) and (j>=50) and ((40>i) or (55<j)) and 150 < ((i*h)-15)**2 + ((j*l)-35)**2< 170:
                Granica.append((i*h, j*l))
            if (i>=50) and (j<=50) and ((60<i) or (45>j)) and 150 < ((i*h)-35)**2 + ((j*l)-15)**2< 170:
                Granica.append((i*h, j*l))
            if (i>=50) and (j>=50) and ((60<i) or (55<j)) and 150 < ((i*h)-35)**2 + ((j*l)-35)**2< 170:
                Granica.append((i*h, j*l))
    return Granica

def Search_Granica7(m,h, n,l): #Окружность2.5 > ij
    #m = 100
    #n = 100
    #h = 0.5
    #l = 0.5
    Granica=[]
    for i in range(m+1):
        for j in range(n+1):
            if (i<38 or j<47) and 130 < ((i*h)-15)**2 + ((j*l)-16)**2< 150:
                Granica.append((i*h, j*l))
            if (i>48 or j>53) and 170 < ((i*h)-30)**2 + ((j*l)-35)**2< 190:
                Granica.append((i*h, j*l))
    return Granica


def Search_Granica8(m,h, n,l): #Окружность2.5 > ji
    #m = 100
    #n = 100
    #h = 0.5
    #l = 0.5
    Granica=[]
    for j in range(n+1):
        for i in range(m+1):
            if (i<38 or j<47) and 130 < ((i*h)-15)**2 + ((j*l)-16)**2< 150:
                Granica.append((i*h, j*l))
            if (i>48 or j>53) and 170 < ((i*h)-30)**2 + ((j*l)-35)**2< 190:
                Granica.append((i*h, j*l))
    return Granica

def Search_Granica9(m,h, n,l): #Окружность4 > ji
    #m = 100
    #n = 100
    #h = 0.5
    #l = 0.5
    Granica=[]
    for j in range(n+1):
        for i in range(m+1):
            if (i<=50) and (j<=50) and ((40>i) or (45>j)) and 150 < ((i*h)-15)**2 + ((j*l)-15)**2< 170 and not((i*h, j*l) in Granica):
                Granica.append((i*h, j*l))
            if (i<=50) and (j>=50) and ((40>i) or (55<j)) and 150 < ((i*h)-15)**2 + ((j*l)-35)**2< 170 and not((i*h, j*l) in Granica):
                Granica.append((i*h, j*l))
            if (i>=50) and (j<=50) and ((60<i) or (45>j)) and 150 < ((i*h)-35)**2 + ((j*l)-15)**2< 170 and not((i*h, j*l) in Granica):
                Granica.append((i*h, j*l))
            if (i>=50) and (j>=50) and ((60<i) or (55<j)) and 150 < ((i*h)-35)**2 + ((j*l)-35)**2< 170 and not((i*h, j*l) in Granica):
                Granica.append((i*h, j*l))
    return Granica
    

def Search_Granica10(m, h, n, l): #Парабола2 ^ ij
    #m = 100
    #n = 100
    #h = 0.5
    #l = 0.5
    Parabola=[]
    a = 9
    for i in range(m+1):
        for j in range(n+1):
            if (i==80 and 12<=j<=98):
                Parabola.append((i*h, j*l))
                continue
            if (i<80) and (j<=55) and 15 < (((j*l)-20)**2 - a*(i*h-20)) < 30:
                Parabola.append((i*h, j*l))
            if (i<80) and (j>=55) and 15 < (((j*l)-35)**2 - a*(i*h-20)) < 30:
                Parabola.append((i*h, j*l))
    return Parabola

def Search_Granica11(m, h, n, l): #Парабола2 ^ ji
    #m = 100
    #n = 100
    #h = 0.5
    #l = 0.5
    Parabola=[]
    a = 9
    for j in range(n+1):
        for i in range(m+1):
            if (i==80 and 12<=j<=98):
                Parabola.append((i*h, j*l))
                continue
            if (i<80) and (j<=55) and 15 < (((j*l)-20)**2 - a*(i*h-20)) < 30:
                Parabola.append((i*h, j*l))
            if (i<80) and (j>=55) and 15 < (((j*l)-35)**2 - a*(i*h-20)) < 30:
                Parabola.append((i*h, j*l))
    return Parabola

def Search_Granica12(m, h, n, l): #Парабола2 > ij
    #m = 100
    #n = 100
    #h = 0.5
    #l = 0.5
    Parabola=[]
    a = 9
    for i in range(m+1):
        for j in range(n+1):
            if (j==80 and 12<=i<=98):
                Parabola.append((i*h, j*l))
                continue
            if (j<80) and (i<=55) and 15 < (((i*h)-20)**2 - a*(j*l-20)) < 30:
                Parabola.append((i*h, j*l))
            if (j<80) and (i>=55) and 15 < (((i*h)-35)**2 - a*(j*l-20)) < 30:
                Parabola.append((i*h, j*l))
    return Parabola

def Search_Granica13(m, h, n, l): #Парабола ^
    #m = 100
    #n = 100
    #h = 0.5
    #l = 0.5
    Parabola=[]
    a = 9
    for i in range(m+1):
        for j in range(n+1):
            if (i==80 and 21<j<79):
                Parabola.append((i*h, j*l))
                continue
            if (i<80) and 15 < (((j*l)-25)**2 - a*(i*h-20)) < 30:
                Parabola.append((i*h, j*l))
    return Parabola

def Search_Granica14(m,h, n,l): #Прямоугольник по краям графика
    Granica=[]
    for i in range(m+1):
        for j in range(n+1):
            if (i==0 or i==m) and (j>=0 and j<=n):
                Granica.append((i*h, j*l))
            if (j==0 or j==n) and (i>=0 and i<=m):
                Granica.append((i*h, j*l))    
    return Granica

def Search_Granica15(m,h, n,l): #Прямоугольник внутри графика 1
    Granica=[]
    for i in range(m+1):
        for j in range(n+1):
            if (i==20 or i==70) and (j>=20 and j<=50):
                Granica.append((i*h, j*l))
            if (j==20 or j==50) and (i>=20 and i<=70):
                Granica.append((i*h, j*l))
    return Granica

def Search_Granica16(m,h, n,l): #Прямоугольник внутри графика 2 lh
    Granica=[]
    for i in range(m+1):
        for j in range(n+1):
            if (i*h==10 or i*h==40) and (j*l>=10 and j*l<=40):
                Granica.append((i*h, j*l))
            if (j*l==10 or j*l==40) and (i*h>=10 and i*h<=40):
                Granica.append((i*h, j*l))
    return Granica


def ReserchX(Turtle):
    Turtle2=[]
    n=len(Turtle)
    for j in range(0,n):
        p1x,p1y=Turtle[j]
        Turtle2.append((p1x,p1y))
        for i in range(j+1,n):
             p2x,p2y=Turtle[i]
             if p2x==p1x:
                 Turtle2.append((p2x,p2y))
    li = []
    for i in Turtle2:
        if i not in li:
            li.append(i)
    return li

def ReserchY(Turtle):
    Turtle2=[]
    n=len(Turtle)
    for j in range(0,n):
        p1x,p1y=Turtle[j]
        Turtle2.append((p1x,p1y))
        for i in range(j+1,n):
             p2x,p2y=Turtle[i]
             if p2y==p1y:
                 Turtle2.append((p2x,p2y))
    li = []
    for i in Turtle2:
        if i not in li:
            li.append(i)
    return li

def ReserchY2(Turtle):
    li=[]
    n=len(Turtle)
    for j in range(0,n):
        p1x,p1y=Turtle[j]
        li.append((p1y,p1x))
    return li
#------------------------------------------------------------------
#-Функции определения где находиться точка(внутри/снаружи) границы-
def in_polygon1(x, y, polygon): #1.0
    n = len(polygon)
    
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n+1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y) and x <= max(p1x, p2x):
                if p1y != p2y:
                    xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x, p1y = p2x, p2y

    inside2 = 0
    for i in range(n):
        px1, py1 = polygon[i]
        px2, py2 = polygon[(i + 1) % n]
        if py1 == py2:
            continue
        if min(py1, py2) <= y < max(py1, py2):
            xints2 = (y - py1) * (px2 - px1) / (py2 - py1) + px1
            if xints2 <= x:
                inside2 += 1

    return inside or inside2 % 2 == 1

def in_polygon2(x, y, polygon): #1.1
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]

    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if min(p1y, p2y) < y <= max(p1y, p2y):
            if x <= max(p1x, p2x):
                if p1y != p2y:
                    xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                if p1x == p2x or x <= xinters:
                    inside = not inside
        p1x, p1y = p2x, p2y

    return inside

def in_polygon3(x, y, polygon): #1.2
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        # Проверка, находится ли y в пределах границ полигона
        if min(p1y, p2y) < y <= max(p1y, p2y):
            # Проверка, находится ли x в пределах границ полигона
            if x <= max(p1x, p2x):
                # Вычисляем x-координату точки пересечения ребра полигона с лучом
                if p1y != p2y:
                    xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                if p1x == p2x or x <= xinters:
                    inside = not inside
        p1x, p1y = p2x, p2y

    return inside

def in_polygon4(x, y, polygon): #Идеально для ji по оси оу^ и ij ля оси ox^ и выпуклых
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(1, n):
        p2x, p2y = polygon[i]
        if (p1y >= y) == (p2y <= y):
            if (p1x >= x) == (p2x <= x):
                inside = not inside
        p1x, p1y = p2x, p2y
    polygon2=ReserchY(polygon)
    n2 = len(polygon2)
    inside2 = False
    p1x, p1y = polygon2[0]
    for i in range(1, n2):
        p2x, p2y = polygon2[i]
        if (p1y >= y) == (p2y <= y):
            if (p1x >= x) == (p2x <= x):
                inside2 = not inside2
        p1x, p1y = p2x, p2y
    return (inside and inside2) 
#--------------------------------#Работает+оптимизацая+краткость-------------------
def in_polygon1(x, y, polygon): #только для впуклых по оси ох^ с ij и выпуклых
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(1, n):   
        p2x, p2y = polygon[i % n]
        if (p1y > y) != (p2y > y):
            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
            if x == xinters:  # Луч пересекает ребро
                inside = not inside 
        p1x, p1y = p2x, p2y
    return inside


def in_polygon2(x, y, polygon): #только для впуклых по оси ох^ с ij и выпуклых
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(1, n):
        p2x, p2y = polygon[i % n]
        if (p1x >= x) != (p2x > x):
            if min(p1y, p2y) < y < max(p1y, p2y): 
                inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def in_polygon2(x, y, polygon): #Идеально для ji по оси оу^ и ij ля оси ox^ и выпуклых
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(1, n):
        p2x, p2y = polygon[i]
        if (p1x >= x) == (p2x <= x):
            if (p1y >= y) == (p2y <= y):
                inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def in_polygon(x, y, polygon): 
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(1, n):
        p2x, p2y = polygon[i]
        if (p1y >= y) == (p2y <= y):
            if (p1x >= x) == (p2x <= x):
                inside = not inside
        p1x, p1y = p2x, p2y
    return (inside) 
#------------------------------------------------------------------
#-------------------------Методы Либмана---------------------------
def Libman1(h, l, coordinata, u0, u1, E): # Пяти точечный + f 
    dobavka = np.zeros((m+1, n+1))
    for i in range(m+1):
        for j in range(n+1):
            dobavka[i][j] = (((h**2)*(l**2))/(2*(h**2+l**2)))*f(i*h, j*l)
    iteration = 0
    max_val = 0
    while max_val >= E:     
        max_val = 0
        for i in range(1, m):
            for j in range(1, n):
                if ((((i-1)* h, j* l) in coordinata or in_polygon((i-1) * h, j * l, coordinata)) and
                (((i+1)* h, j * l) in coordinata or in_polygon((i+1) * h, j * l, coordinata)) and
                ((i* h, (j+1)* l) in coordinata or in_polygon(i * h, (j+1) * l, coordinata)) and
                ((i* h, (j-1)* l) in coordinata or in_polygon(i * h, (j-1) * l, coordinata))):
                    u1[i][j] = (l**2*(u0[i+1][j]+u0[i-1][j]) + h**2*(u0[i][j+1]+u0[i][j-1])) / (2*(h**2+l**2)) - dobavka[i][j]
                    diff = abs(u1[i][j] - u0[i][j])
                    if diff > max_val:
                        max_val = diff
                        print ('diff:',diff)
        u0 = u1.copy()
        iteration += 1
        print ('iteration',iteration)
    return u0, iteration

def Libman(h, l, coordinata, u0, u1, E): # Пяти точечное Усреднение (для констант)
    iteration = 0
    max_val = E+1
    while max_val >= E:     
        max_val = 0
        for i in range(1, m):
            for j in range(1, n):
                if ((((i-1)* h, j* l) in coordinata or in_polygon((i-1) * h, j * l, coordinata)) and
                (((i+1)* h, j * l) in coordinata or in_polygon((i+1) * h, j * l, coordinata)) and
                ((i* h, (j+1)* l) in coordinata or in_polygon(i * h, (j+1) * l, coordinata)) and
                ((i* h, (j-1)* l) in coordinata or in_polygon(i * h, (j-1) * l, coordinata))):
                    u1[i][j] = (l**2*(u0[i+1][j]+u0[i-1][j]) + h**2*(u0[i][j+1]+u0[i][j-1])) / (2*(h**2+l**2))
                    diff = abs(u1[i][j] - u0[i][j])
                    if diff > max_val:
                        max_val = diff
                        print ('diff:',diff)
        u0 = u1.copy()
        iteration += 1
        print ('iteration',iteration)
    return u0, iteration


def Libman3(h, l, coordinata, u0, u, E): #Двенадцати точечное не ограниченное с f
    dobavka = np.zeros((m+1, n+1))
    for i in range(m+1):
        for j in range(n+1):
            dobavka[i][j] = ((h**4)*f(i*h, j*l))/20
    iteration = 0
    max_val = E+1
    while max_val >= E:     
        max_val = 0
        for i in range (2, m-1):
            for j in range (2, n-1):
                if ((((i-1)* h, j* l) in coordinata or in_polygon((i-1) * h, j * l, coordinata)) and
                (((i+1)* h, j * l) in coordinata or in_polygon((i+1) * h, j * l, coordinata)) and
                ((i* h, (j+1)* l) in coordinata or in_polygon(i * h, (j+1) * l, coordinata)) and
                ((i* h, (j-1)* l) in coordinata or in_polygon(i * h, (j-1) * l, coordinata)) and
                (((i-2)* h, j* l) in coordinata or in_polygon((i-2) * h, j * l, coordinata)) and
                (((i+2)* h, j * l) in coordinata or in_polygon((i+2) * h, j * l, coordinata)) and
                ((i* h, (j+2)* l) in coordinata or in_polygon(i * h, (j+2) * l, coordinata)) and
                ((i* h, (j-2)* l) in coordinata or in_polygon(i * h, (j-2) * l, coordinata)) and
                (((i-1)* h, (j-1)* l) in coordinata or in_polygon((i-1) * h, (j-1) * l, coordinata)) and
                (((i+1)* h, (j+1) * l) in coordinata or in_polygon((i+1) * h, (j+1) * l, coordinata)) and
                (((i-1)* h, (j+1)* l) in coordinata or in_polygon((i-1) * h, (j+1) * l, coordinata)) and
                (((i+1)* h, (j-1)* l) in coordinata or in_polygon((i+1) * h, (j-1) * l, coordinata))):
                    u[i][j] = dobavka[i][j] - (u[i + 2][j] + u[i - 2][j] + u[i][j + 2] + u[i][j - 2] - 8 * u[i + 1][j]- 8 * u[i - 1][j]- 8 * u[i][j + 1]- 8 * u[i][j - 1]+ 2 * u[i + 1][j + 1]+ 2 * u[i + 1][j - 1]+ 2 * u[i - 1][j - 1] + 2 * u[i - 1][j + 1])/ 20
                    diff = abs(u[i][j] - u0[i][j])
                    if diff > max_val:
                        max_val = diff
                        print (max_val)
        u0 = u.copy()
        iteration += 1
        print ('iteration',iteration)
    return u0, iteration

def Libman3(h, l, coordinata, u0, u, E): #Двенадцати точечное более быстрое ограниченное с f
    dobavka = np.zeros((m+1, n+1))
    for i in range(m+1):
        for j in range(n+1):
            dobavka[i][j] = ((h**4)*f(i*h, j*l))/20
    iteration = 0
    max_val = E+1
    while max_val >= E:     
        max_val = 0
        for i in range (2, m-1):
            for j in range (2, n-1):
                if ((((i+2)* h, j * l) in coordinata or in_polygon((i+2) * h, j * l, coordinata)) and
                ((i* h, (j+2)* l) in coordinata or in_polygon(i * h, (j+2) * l, coordinata)) and
                ((i* h, (j-2)* l) in coordinata or in_polygon(i * h, (j-2) * l, coordinata)) and
                (((i-2)* h, j* l) in coordinata or in_polygon((i-2) * h, j * l, coordinata))):
                    u[i][j] = dobavka[i][j]-(u[i+2][j]+u[i-2][j]+u[i][j+2]+u[i][j-2]-8*u[i+1][j]-8*u[i-1][j]-8*u[i][j+1]-8*u[i][j-1]+2*u[i+1][j+1]+2*u[i+1][j-1]+2*u[i-1][j-1]+2*u[i-1][j+1])/20
                    diff = abs(u[i][j] - u0[i][j])
                    if diff > max_val:
                        max_val = diff
                        print (max_val)
        u0 = u.copy()
        iteration += 1
        print ('iteration',iteration)
    return u0, iteration

def Libman5(h, l, coordinata, u0, u1, E): # Пяти точечный + f ################################
    dobavka = np.zeros((m+1, n+1))
    for i in range(m+1):
        for j in range(n+1):
            dobavka[i][j] = (((h**2)*(l**2))/(2*(h**2+l**2)))*f(i*h, j*l)
    iteration = 0
    max_val = E+1
    while max_val >= E:     
        max_val = 0
        for i in range(1, m):
            for j in range(1, n):
                if ((((i-1)* h, j* l) in coordinata or in_polygon((i-1) * h, j * l, coordinata)) and
                (((i+1)* h, j * l) in coordinata or in_polygon((i+1) * h, j * l, coordinata)) and
                ((i* h, (j+1)* l) in coordinata or in_polygon(i * h, (j+1) * l, coordinata)) and
                ((i* h, (j-1)* l) in coordinata or in_polygon(i * h, (j-1) * l, coordinata))):
                    u1[i][j] = (l**2*(u0[i+1][j]+u0[i-1][j]) + h**2*(u0[i][j+1]+u0[i][j-1])) / (2*(h**2+l**2)) + dobavka[i][j]
                    diff = abs(u1[i][j] - u0[i][j])
                    if diff > max_val:
                        max_val = diff
                        print ('diff:',diff)
        u0 = u1.copy()
        iteration += 1
        print ('iteration',iteration)
    return u0, iteration
#------------------------------------------------------------------------------------------------------
#-------------------------------------------ОСНОВНАЯ ЧАСТЬ---------------------------------------------
#------------------------------------------------------------------------------------------------------
#-------------------------Задание размера сетки и точности--------------------------
m = 100 #количество узлов сетки по оси х (максимальное значение по оси х / h) Произвольные
n = 100 #количество узлов сетки по оси у (максимальное значение по оси у / l)
h = 0.5 #длина шага по оси х 
l = 0.5 #длина шага по оси у 
E = 10 #необходимая точность (епсилон)
 
#m = 10 #количество узлов сетки по оси х (максимальное значение по оси х / h) Градиент
#n = 10 #количество узлов сетки по оси у (максимальное значение по оси у / l)
#h = 0.1 #длина шага по оси х
#l = 0.1 #длина шага по оси у
#E = 0.0001 #необходимая точность (епсилон)

#m = 10 #количество узлов сетки по оси х (максимальное значение по оси х / h) 8cos
#n = 10 #количество узлов сетки по оси у (максимальное значение по оси у / l)
#h = 0.2 #длина шага по оси х
#l = 0.3 #длина шага по оси у
#E = 0.0001 #необходимая точность (епсилон)
#--------------------------------------------------------------------
#-------------------Создание массивов--------------------------------
u0 = np.zeros((m+1, n+1))
coordinata=Search_Granica(m,h, n,l)
for i in range(m + 1):
    for j in range(n + 1):
        if ((i * h,j*l)in coordinata):
            u0[i][j] = Gran_Func(i*h,j*l)
        elif in_polygon(i * h, j * l, coordinata):
            u0[i][j] = f(i*h,j*l)
u1 = u0.copy()
#----------------------------------------------------------------
#--------------------Метод Либмана-------------------------------
u0,iteration=Libman(h, l, coordinata, u0, u1, E)
#----------------------------------------------------------------
#--------------Вывод полученных значений узлов сетки-------------
#print('Iterations: ', iteration) 
#print('*\t', end='') #Вывод сетки
#for i in range(m+1):
#    print(f'{i:.0f}', end='\t')
#print()
#for j in range(n+1):
#    print(f'{j:.0f}', end='\t')
#    for i in range(m+1):
#        print(f'{u1[i][j]:.3f}', end='\t')
#    print()
#----------------------------------------------------------------
#----------------Построение графика------------------------------
#plt.ion()
#fig, ax = plt.subplots()
#fig. suptitle('Решение задачи Дирихле методом Либмана')
#p1=ax.imshow(u0, cmap='plasma')
#fig.colorbar(p1)
#ax.invert_yaxis()
#ax.text(0.5, 1.01, f'Число итераций: {iteration}', transform=ax.transAxes, ha='center')
#plt.draw()



u = [[0 for _ in range(m + 1)] for _ in range(n + 1)]
for i in range(m + 1):
    for j in range(n + 1):
        u[i][j] = u0[i, j]

Z = np.array(u)
x = [h * i for i in range(n + 1)]
X, Y = np.meshgrid(x, x)

# Настройка визуализации
plt.ion()
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})  # создание фигуры и осей для 3D-графика
fig.suptitle(f'Решение задачи Дирихле методом Либмана\nЧисло итераций: {iteration}')

#ax.plot_surface(X, Y, Z, cmap='plasma')
ax.plot_wireframe(X, Y, Z, color="green")
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
 

plt.show()
plt.ioff()  


#----------------------------------------------------------------
