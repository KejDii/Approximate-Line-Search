import numpy as np
import matplotlib.pyplot as plt


def opt_booth(x0, stop, max_iter, r): #x0 - punkt początkowy, stop - zbieżność rozwiązania, max_iter - maksymalna liczba iteracji, r - zmienna ważąca karę
    class Booth:
        def __init__(self):
            self.calls = 0

        def f_val(self, x):
            self.calls += 1
            return (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2

        def derivative(self, x):
            dx1 = 2 * (x[0] + 2 * x[1] - 7) + 4 * (2 * x[0] + x[1] - 5)
            dx2 = 4 * (x[0] + 2 * x[1] - 7) + 2 * (2 * x[0] + x[1] - 5)
            return dx1, dx2

        def print_calls(self):
            print("Liczba wywołań funkcji Booth")
            print(self.calls)

        def penalized_f_val(self,x,r):
            self.calls += 1
            g = x[0] ** 2 + 2 * x[0] - x[1]
            if g < -1e-8 or r == 0:
                return self.f_val(x) - r / g #wartość funkcji z uwzględnieniem kary
            else:
                raise ValueError(f"Punkt narusza ograniczenie: g(x) = {g} >= 0")

    def alfa_search(x, r, function): #x - punkt, r - zmienna ważąca karę, function - obiekt klasy funkcji Booth
        c1 = 1e-4
        alpha_init = 0.05
        g = x[0] ** 2 + 2 * x[0] - x[1] #ograniczenie
        if g < -1e-8 or r == 0: #jeżeli ograniczenie spełnione
            dx = function.derivative(x)
            dx_r = [0, 0]
            dx_r[0] = dx[0] + 2*r*(x[0]+1)/ g ** 2
            dx_r[1] = dx[1] + (-r/ g ** 2)
        else:
            raise ValueError(f"Punkt narusza ograniczenie: g(x) = {g} >= 0") #jeżeli nie, wyrzuć błąd
        # kierunek spadku (ujemny gradient)
        direction = [-dx_r[0], -dx_r[1]]

        f_x = function.penalized_f_val(x,r)
        grad_dot_dir = dx_r[0] * direction[0] + dx_r[1] * direction[1]  # iloczyn skalarny gradient, kierunek

        alpha = alpha_init
        while True: #backtracking line search
            # x - alpha * grad
            x_now = [x[0] + alpha * direction[0], x[1] + alpha * direction[1]]
            f_now = function.penalized_f_val(x_now,r)
            # warunek Armijo
            if f_now <= f_x + c1 * alpha * grad_dot_dir:
                break
            alpha *= 0.5
            # Zabezpieczenie żeby alpha != 0
            if alpha < 1e-8:
                break
        return alpha

    g = x0[0] ** 2 + 2 * x0[0] - x0[1]
    if g >= -1e-8 and r != 0:
        raise ValueError("x0 nie spełnia ograniczenia — wybierz punkt wewnątrz obszaru dopuszczalnego.")
    i = 0
    booth = Booth()
    history_x0 = []
    history_x1 = []
    history_f = []
    dx_r = [0, 0]
    x1 = x0 #x1 - zmienny punkt w trakcie optymalizacji
    f_prev = 0
    f_now = booth.penalized_f_val(x0,r)
    history_f.append(f_now)  # zapisz dane
    history_x0.append(x0[0])
    history_x1.append(x0[1])

    while abs(f_now - f_prev) > stop and i < max_iter:
        g = x1[0] ** 2 + 2 * x1[0] - x1[1]
        if g < -1e-8 or r == 0:
            dx = booth.derivative(x1)
            dx_r[0] = dx[0] + 2 * r * (x1[0] + 1) / g ** 2
            dx_r[1] = dx[1] + (-r / g ** 2)
        else:
            raise ValueError(f"Punkt narusza ograniczenie: g(x) = {g} >= 0")
        f_prev = f_now

        alfa = alfa_search(x1, r, booth) #znajdź krok alfa
        x1[0] = x1[0] - dx_r[0] * alfa #przypisz nowy punkt
        x1[1] = x1[1] - dx_r[1] * alfa

        g = x1[0] ** 2 + 2 * x1[0] - x1[1]
        if g < -1e-8 or r == 0:
            f_now = booth.f_val(x1) - r / g
        else:
            raise ValueError(f"Punkt narusza ograniczenie: g(x) = {g} >= 0")
        history_f.append(f_now) #zapisz dane
        history_x0.append(x1[0])
        history_x1.append(x1[1])
        i = i + 1 #przejdź do kolejnej iteracji

    #koniec przebiegu optymalizacji
    if i == max_iter:
        print("Warunek stopu - maksymalna ilość iteracji osiągnięta")
    else:
        print("Warunek stopu - zbieżność rozwiązania osiągnięta")
    print("Punkt optimum:")
    print(x1)
    print("Iteracje:")
    print(i)
    booth.print_calls()
    print("Ograniczenie(x)")
    print(x1[0] ** 2 + 2 * x1[0] - x1[1])
    print("Wartosć samej funkcji booth(x1)")
    print(booth.f_val(x1))
    plt.figure(1)
    plt.plot(history_f)
    plt.title("Przebieg wartości funkcji")
    plt.xlabel("Iteracje")
    plt.ylabel("Booth(x)")
    plt.ylim([0, history_f[0]])
    plt.show()
    plt.figure(2)
    plt.plot(history_x0, history_x1)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Przebieg zmiennych x1,x2")
    plt.show()
    plt.figure(3)
    plt.plot(history_x0, 'b')
    plt.plot(history_x1, 'r')
    plt.xlabel("Itearcje")
    plt.ylabel("x1,x2")
    plt.title("Przebieg zmiennych x1,x2")
    plt.show()


    # rysowanie wykresu contour
    x = np.linspace(-5, 3, 400)
    x = x[x != 0]
    y = np.linspace(-2, 6, 400)
    y = y[y != 0]

    X, Y = np.meshgrid(x, y)
    # ograniczenie
    constraint = X ** 2 + 2 * X - Y
    mask = constraint <= 0
    # obliczenie wartości funkcji dla każdego punktu
    if r > 0:
       # Z = np.vectorize(lambda x, y: booth.penalized_f_val([x, y], r))(X, Y)
        Z = np.full_like(X, np.nan, dtype=np.float64)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                if mask[i, j]:
                    Z[i, j] = booth.penalized_f_val([X[i, j], Y[i, j]], r)
        levels = 500
    else:
        Z = np.vectorize(lambda x,y: booth.f_val([x,y]))(X,Y)
        levels = 50

    # rysowanie wykresu
    plt.figure(figsize=(8, 6))
    cp = plt.contour(X, Y, Z, levels=levels, cmap='viridis')
    plt.colorbar(cp, label='Wartość funkcji Booth')

    # dodanie ograniczenia
    if r > 0:
        plt.contour(X, Y, constraint, levels=[0], colors='red')
        plt.title('Funkcja Booth z ograniczeniem $x1^2 + 2x1 - x2 \\leq 0$')
    else:
        plt.title('Funkcja Booth')

    plt.plot(history_x0, history_x1, marker='o', color='black', label='Przebieg optymalizacji')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid(True)
    plt.legend()
    plt.show()


def opt_rosenbrock(x0, stop, max_iter, eq_con, r):
    class Rosenbrock:
        def __init__(self):
            self.calls = 0

        def f_val(self, x):
            self.calls += 1
            return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

        def derivative(self, x):
            dx1 = 2 * (200 * x[0] ** 3 - 200 * x[0] * x[1] + x[0] - 1)
            dx2 = 2 * 100 * (x[1] - x[0] ** 2)
            return dx1, dx2

        def print_calls(self):
            print("Liczba wywołań funkcji Rosenbrock")
            print(self.calls)

        def penalized_f_val(self,x,eq_con,r):
            return self.f_val(x) + r * (eq_con[0] * x[0] + eq_con[1] * x[1] + eq_con[2]) ** 2

    def alfa_search(x, eq_con, r, function):
        c1 = 1e-4
        alpha_init = 0.5
        dx = function.derivative(x)
        dx_r = [0, 0]
        g = eq_con[0] * x[0] + eq_con[1] * x[1] + eq_con[2]
        dx_r[0] = dx[0] + 2 * r * eq_con[0] * g
        dx_r[1] = dx[1] + 2 * r * eq_con[1] * g

        # kierunek spadku (ujemny gradient)
        direction = [-dx_r[0], -dx_r[1]]
        f_x = function.penalized_f_val(x,eq_con,r)
        grad_dot_dir = dx_r[0] * direction[0] + dx_r[1] * direction[1]  # iloczyn skalarny

        alpha = alpha_init
        #backtracking line search
        while True:
            # x - alpha * grad
            x_now = [x[0] + alpha * direction[0], x[1] + alpha * direction[1]]
            f_now = function.penalized_f_val(x_now,eq_con,r)
            # warunek Armijo
            if f_now <= f_x + c1 * alpha * grad_dot_dir:
                break
            alpha *= 0.5
            # zabezpieczenie żeby alpha != 0
            if alpha < 1e-5:
                break
        return alpha

    i = 0
    history_x0 = []
    history_x1 = []
    history_f = []
    x1 = x0
    f_prev = 0
    rosenbrock = Rosenbrock()
    f_now = rosenbrock.penalized_f_val(x0,eq_con,r)
    history_x0.append(x1[0])
    history_x1.append(x1[1])
    history_f.append(f_now)
    while abs(f_now - f_prev) > stop and i < max_iter:
        dx = rosenbrock.derivative(x1)
        dx_r = [0, 0]
        g = eq_con[0] * x1[0] + eq_con[1] * x1[1] + eq_con[2]
        dx_r[0] = dx[0] + 2 * r * eq_con[0] * g
        dx_r[1] = dx[1] + 2 * r * eq_con[1] * g
        f_prev = f_now

        alfa = alfa_search(x1, eq_con, r, rosenbrock)
        x1[0] = x1[0] - dx_r[0] * alfa
        x1[1] = x1[1] - dx_r[1] * alfa
        history_x0.append(x1[0])
        history_x1.append(x1[1])
        f_now = rosenbrock.penalized_f_val(x1,eq_con,r)
        history_f.append(f_now)
        i = i + 1

    if i == max_iter:
        print("Warunek stopu - maksymalna ilość iteracji osiągnięta")
    else:
        print("Warunek stopu - zbieżność rozwiązania osiągnięta")
    print("Punkt optimum:")
    print(x1)
    print("Iteracje:")
    print(i)
    rosenbrock.print_calls()
    print("Ograniczenie(x)")
    print(x1[0] * eq_con[0] + x1[1] * eq_con[1] + eq_con[2])
    print("Wartosć samej funkcji  rosenbrock(x1) z karą")
    print(rosenbrock.penalized_f_val(x1,eq_con,r))
    plt.figure(1)
    plt.plot(history_f)
    plt.title("Przebieg wartości funkcji")
    plt.xlabel("Iteracje")
    plt.ylabel("Rosenbrock(x)")
    plt.ylim([0, history_f[0]])
    plt.show()
    plt.figure(2)
    plt.plot(history_x0, history_x1)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Przebieg zmiennych x1,x2")
    plt.show()
    plt.figure(3)
    plt.plot(history_x0, 'b')
    plt.plot(history_x1, 'r')
    plt.xlabel("Itearcje")
    plt.ylabel("x1,x2")
    plt.title("Przebieg zmiennej x1,x2")
    plt.show()

    # contury
    x = np.linspace(-2, 2, 400)
    y = np.linspace(-5, 5, 400)
    X, Y = np.meshgrid(x, y)
    # obliczenie wartości funkcji dla każdego punktu
    Z = np.vectorize(lambda x, y: rosenbrock.penalized_f_val([x,y],eq_con,r))(X, Y)

    # oarametry ograniczenia
    constraint = X*eq_con[0]+Y*eq_con[1]+eq_con[2]

    # rysowanie wykresu
    plt.figure(figsize=(8, 6))
    cp = plt.contour(X, Y, Z, levels=100, cmap='viridis')
    plt.colorbar(cp, label='Wartość funkcji Rosenbrock')

    if r > 0:
        plt.title(f'Funkcja Rosenbrock z ograniczeniem ${eq_con[0]}x1 + {eq_con[1]}x2 + {eq_con[2]} = 0$')
        # dodanie ograniczenia
        plt.contour(X, Y, constraint, levels=[0], colors='red')
    else:
        plt.title('Funkcja Rosenbrock')
    plt.plot(history_x0, history_x1, marker='o', color='black', label='Przebieg optymalizacji')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid(True)
    #plt.axis('equal')
    plt.show()


def opt_camel(x0, stop, max_iter,radius,r):
    class Camel:
        def __init__(self):
            self.calls = 0

        def f_val(self, x):
            self.calls += 1
            return 2 * x[0] ** 2 - 1.05 * x[0] ** 4 + (1 / 6) * x[0] ** 6 + x[0] * x[1] + x[1] ** 2

        def derivative(self, x):
            dx1 = 4 * x[0] - 4 * 1.05 * x[0] ** 3 + x[0] ** 5 + x[1]
            dx2 = x[0] + 2 * x[1]
            return dx1, dx2

        def penalized_f_val(self,x,radius,r):
            if (x[0]**2+x[1]**2-radius) > 0:
                return self.f_val(x) + r*(x[0]**2+x[1]**2-radius)
            else:
                return self.f_val(x)

        def penalized_derivative(self,x,radius,r):
            dx1 = 4 * x[0] - 4 * 1.05 * x[0] ** 3 + x[0] ** 5 + x[1] + r*(2*x[0])
            dx2 = x[0] + 2 * x[1] + r*(2*x[1])
            return dx1, dx2

        def print_calls(self):
            print("Liczba wywołań funkcji Camel")
            print(self.calls)

    def alfa_search(x, function,radius,r):
        c1 = 1e-4
        alpha_init = 0.1
        #jeżeli poza ograniczeniem - dodaj karę
        if x[0]**2 + x[1]**2 < radius:
            dx = function.derivative(x)
            f_x = function.f_val(x)
        else:
            dx=function.penalized_derivative(x,radius,r)
            f_x=function.penalized_f_val(x,radius,r)

        # kierunek spadku (ujemny gradient)
        direction = [-dx[0], -dx[1]]
        grad_dot_dir = dx[0] * direction[0] + dx[1] * direction[1]  # iloczyn skalarny

        alpha = alpha_init
        #backtracking line search
        while True:
            # x - alpha * grad
            x_now = [x[0] + alpha * direction[0], x[1] + alpha * direction[1]]
            if x_now[0] ** 2 + x_now[1] ** 2 < radius:
                f_now= function.f_val(x_now)
            else:
                f_now = function.penalized_f_val(x_now, radius, r)
            # Warunek Armijo
            if f_now <= f_x + c1 * alpha * grad_dot_dir:
                break
            alpha *= 0.5
            # Zabezpieczenie żeby alpha != 0
            if alpha < 1e-5:
                break
        return alpha

    i = 0
    camel = Camel()
    history_x0 = []
    history_x1 = []
    history_f = []
    history_x0.append(x0[0])
    history_x1.append(x0[1])
    x1 = x0
    f_prev = 0
    if x0[0] ** 2 + x0[1] ** 2 < radius:
        f_now = camel.f_val(x0)
    else:
        f_now = camel.penalized_f_val(x0, radius, r)
    history_f.append(f_now)
    while abs(f_now - f_prev) > stop and i < max_iter:
        if x1[0] ** 2 + x1[1] ** 2 < radius:
            dx=camel.derivative(x1)
        else:
            dx=camel.penalized_derivative(x1,radius,r)
        f_prev = f_now

        alfa = alfa_search(x1, camel,radius, r)
        x1[0] = x1[0] - dx[0] * alfa
        x1[1] = x1[1] - dx[1] * alfa
        history_x0.append(x1[0])
        history_x1.append(x1[1])
        if x1[0] ** 2 + x1[1] ** 2 < radius:
            f_now = camel.f_val(x1)
        else:
            f_now = camel.penalized_f_val(x1,radius,r)
        history_f.append(f_now)
        i = i + 1

    if i == max_iter:
        print("Warunek stopu - maksymalna ilość iteracji osiągnięta")
    else:
        print("Warunek stopu - zbieżność rozwiązania osiągnięta")
    print("Punkt optimum:")
    print(x1)
    print("Iteracje:")
    print(i)
    camel.print_calls()
    print("Wartosć funkcji camel(x1) z karą")
    print(camel.penalized_f_val(x1,radius,r))
    plt.figure(1)
    plt.plot(history_f)
    plt.title("Przebieg wartości funkcji")
    plt.xlabel("Iteracje")
    plt.ylabel("Camel(x)")
    plt.ylim([0, history_f[0]])
    plt.show()
    plt.figure(2)
    plt.plot(history_x0, history_x1)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Przebieg zmiennych x1,x2")
    plt.show()
    plt.figure(3)
    plt.plot(history_x0, 'b')
    plt.plot(history_x1, 'r')
    plt.xlabel("Itearcje")
    plt.ylabel("x1,x2")
    plt.title("Przebieg zmiennych x1,x2")
    plt.show()

    #contury
    x = np.linspace(-2, 2, 400)
    y = np.linspace(-2, 2, 400)
    X, Y = np.meshgrid(x, y)
    # Obliczenie wartości funkcji dla każdego punktu
    Z = np.vectorize(lambda x, y: camel.penalized_f_val([x, y],radius,r))(X, Y)

    # Parametry ograniczenia
    constraint = X ** 2 + Y ** 2

    # Rysowanie wykresu
    plt.figure(figsize=(8, 6))
    cp = plt.contour(X, Y, Z, levels=80, cmap='viridis')
    plt.colorbar(cp, label='Wartość funkcji Camel')

    # Dodanie ograniczenia: kontur dla koła
    if r > 0:
        plt.contour(X, Y, constraint, levels=[radius], colors='red')
        plt.title('Funkcja Camel z ograniczeniem $x1^2 + x2^2 <= {}$'.format(radius))
    else:
        plt.title('Funkcja Camel')
    plt.plot(history_x0, history_x1, marker='o', color='black', label='Przebieg optymalizacji')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid(True)
    #plt.axis('equal')
    plt.show()

#funkcja opt_camel2 była stosowana do rysowania wykresów dla wielu punktów poczatkowych
def opt_camel2(x0, stop, max_iter,radius,r):
    class Camel:
        def __init__(self):
            self.calls = 0

        def f_val(self, x):
            self.calls += 1
            return 2 * x[0] ** 2 - 1.05 * x[0] ** 4 + (1 / 6) * x[0] ** 6 + x[0] * x[1] + x[1] ** 2

        def derivative(self, x):
            dx1 = 4 * x[0] - 4 * 1.05 * x[0] ** 3 + x[0] ** 5 + x[1]
            dx2 = x[0] + 2 * x[1]
            return dx1, dx2

        def penalized_f_val(self,x,radius,r):
            if (x[0]**2+x[1]**2-radius) > 0:
                return self.f_val(x) + r*(x[0]**2+x[1]**2-radius)
            else:
                return self.f_val(x)

        def penalized_derivative(self,x,radius,r):
            dx1 = 4 * x[0] - 4 * 1.05 * x[0] ** 3 + x[0] ** 5 + x[1] + r*(2*x[0])
            dx2 = x[0] + 2 * x[1] + r*(2*x[1])
            return dx1, dx2

        def print_calls(self):
            print("Liczba wywołań funkcji Camel")
            print(self.calls)

    def alfa_search(x, function,radius,r,alf,c):
        c1 = c
        alpha_init = alf
        #jeżeli poza ograniczeniem - dodaj karę
        if x[0]**2 + x[1]**2 < radius:
            dx = function.derivative(x)
            f_x = function.f_val(x)
        else:
            dx=function.penalized_derivative(x,radius,r)
            f_x=function.penalized_f_val(x,radius,r)

        # kierunek spadku (ujemny gradient)
        direction = [-dx[0], -dx[1]]
        grad_dot_dir = dx[0] * direction[0] + dx[1] * direction[1]  # iloczyn skalarny

        alpha = alpha_init
        #backtracking line search
        while True:
            # x - alpha * grad
            x_now = [x[0] + alpha * direction[0], x[1] + alpha * direction[1]]
            if x_now[0] ** 2 + x_now[1] ** 2 < radius:
                f_now= function.f_val(x_now)
            else:
                f_now = function.penalized_f_val(x_now, radius, r)
            # Warunek Armijo
            if f_now <= f_x + c1 * alpha * grad_dot_dir:
                break
            alpha *= 0.5
            # Zabezpieczenie żeby alpha != 0
            if alpha < 1e-5:
                break
        return alpha

    i = 0
    j=0
    camel = Camel()
    history_x0 = []
    history_x1 = []
    history_x2 = []
    history_x3 = []
    history_x4 = []
    history_x5 = []
    history_x6 = []
    history_x7 = []
    history_x8 = []
    history_x9 = []
    history_f = []
    for j in range(1,6):
        if j == 1:
            x1 = x0
            f_now = camel.f_val(x0)
            si = 0.9
            history_x0.append(x1[0])
            history_x1.append(x1[1])
        elif j == 2:
            x1=[0.2,1.9]
            f_now = camel.f_val(x1)
            si=0.7
            history_x2.append(x1[0])
            history_x3.append(x1[1])
        elif j == 3:
            x1 = [1.9,0.7]
            f_now = camel.f_val(x1)
            si=1e-4
            history_x4.append(x1[0])
            history_x5.append(x1[1])
        elif j == 4:
            x1 = [-1.5, -1.9]
            f_now = camel.f_val(x1)
            si = 1e-4
            history_x6.append(x1[0])
            history_x7.append(x1[1])
        elif j == 5:
            x1 = [0.5, -1.7]
            f_now = camel.f_val(x1)
            si = 1e-4
            history_x8.append(x1[0])
            history_x9.append(x1[1])
        f_prev = 0



        history_f.append(f_now)
        while abs(f_now - f_prev) > stop and i < max_iter:
            if x1[0] ** 2 + x1[1] ** 2 < radius:
                dx=camel.derivative(x1)
            else:
                dx=camel.penalized_derivative(x1,radius,r)
            f_prev = f_now

            alfa = alfa_search(x1, camel,radius,r,0.1,1e-4)
            x1[0] = x1[0] - dx[0] * alfa
            x1[1] = x1[1] - dx[1] * alfa
            if j == 1:
                history_x0.append(x1[0])
                history_x1.append(x1[1])
            elif j == 2:
                history_x2.append(x1[0])
                history_x3.append(x1[1])
            elif j == 3:
                history_x4.append(x1[0])
                history_x5.append(x1[1])
            elif j == 4:
                history_x6.append(x1[0])
                history_x7.append(x1[1])
            elif j == 5:
                history_x8.append(x1[0])
                history_x9.append(x1[1])

            if x1[0] ** 2 + x1[1] ** 2 < radius:
                f_now = camel.f_val(x1)
            else:
                f_now = camel.penalized_f_val(x1,radius,r)
            history_f.append(f_now)
            i = i + 1

        if i == max_iter:
            print("Warunek stopu - maksymalna ilość iteracji osiągnięta")
        else:
            print("Warunek stopu - zbieżność rozwiązania osiągnięta")


    #contury
    x = np.linspace(-2, 2, 400)
    y = np.linspace(-2, 2, 400)
    X, Y = np.meshgrid(x, y)
    # Obliczenie wartości funkcji dla każdego punktu
    Z = np.vectorize(lambda x, y: camel.penalized_f_val([x, y],radius,r))(X, Y)

    # Parametry ograniczenia
    constraint = X ** 2 + Y ** 2

    # Rysowanie wykresu
    plt.figure(figsize=(8, 6))
    cp = plt.contour(X, Y, Z, levels=80, cmap='viridis')
    plt.colorbar(cp, label='Wartość funkcji Camel')

    # Dodanie ograniczenia: kontur dla koła
    if r > 0:
        plt.contour(X, Y, constraint, levels=[radius], colors='red')
        plt.title('Funkcja Camel z ograniczeniem $x1^2 + x2^2 <= {}$'.format(radius))
    else:
        plt.title('Funkcja Camel')
    plt.plot(history_x0, history_x1, marker='o', color='black', label='x0=[-1.5,1.9]')
    plt.plot(history_x2, history_x3, marker='o', color='red', label='x0=[0.2,1.9]')
    plt.plot(history_x4, history_x5, marker='o', color='green', label='x0=[1.9,0.7]')
    plt.plot(history_x6, history_x7, marker='o', color='yellow', label='x0=[-1.5, -1.9]')
    plt.plot(history_x8, history_x9, marker='o', color='purple', label='x0=[0.5, -1.7]')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.grid(True)
    #plt.axis('equal')
    plt.show()

#funkcja x2 jest funkcją kwadratową - do rysowania wykresów na prezentację
def x2(x0):
    def alfa_search(x):
        ro = 0.5
        c = 0.1
        alpha_init = 1
        f_x = x ** 2
        # kierunek spadku (ujemny gradient)
        direction = -2*x # ujemna pochodna x^2
        grad_dot_dir = direction * 2 * x #iloczyn skalarny direction * pochodna

        alpha = alpha_init
        while True:
            # x - alpha * grad
            x_now = x + alpha * direction
            f_now = x_now ** 2
            # zeżeli warunek Armijo spełniony
            if f_now <= f_x + c * alpha * grad_dot_dir:
                break # zatrzymaj pętlę i zwróć alpha
            alpha *= ro
            # zabezpieczenie żeby alpha != 0
            if alpha < 1e-5:
                break
        return alpha

    ro = 0.5
    c = 0.1
    i=0
    history_x = []
    history_f = []
    history_x.append(x0)
    x1 = x0
    f_prev = 0
    f_now = x0 ** 2
    history_f.append(f_now)
    while abs(f_now - f_prev) > 0.001 and i < 100:
        dx = 2 * x1
        f_prev = f_now
        alfa = alfa_search(x1)
        x1 = x1 - dx * alfa
        history_x.append(x1)
        f_now = x1 ** 2
        history_f.append(f_now)
        i = i + 1
    if i == 100:
        print("Warunek stopu - maksymalna ilość iteracji osiągnięta")
    else:
        print("Warunek stopu - zbieżność rozwiązania osiągnięta")
    print("Punkt optimum:")
    print(x1)
    print("Iteracje:")
    print(i)

    # Funkcja celu
    f = lambda x: x ** 2

    # Zakres do rysowania funkcji
    x_vals = np.linspace(-5, 5, 400)
    y_vals = f(x_vals)

    # Rysowanie wykresu
    plt.figure(figsize=(8, 6))
    plt.plot(x_vals, y_vals, label=r'$f(x) = x^2$', color='blue')

    # Dodanie punktów optymalizacji
    plt.plot(history_x, history_f, marker='o', color='black', label='Przebieg optymalizacji')

    plt.title(
        'Funkcja $f(x) = x^2$ z przebiegiem optymalizacji dla $αinit = {}$, $c = {}$, $ρ = {}$, $x początkowe = {}$'.format(alfa, c, ro,x0))
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True)
    plt.legend()
    plt.show()

#opt_booth([0.25,0.8], 0.00000001,100, 1)
#opt_rosenbrock([-1,-1], 0.00000001,5000, [-0.5, -1, 1.5], 0)
#opt_rosenbrock([-1.5,-1.5], 0.00000001,5000, [1,1,0], 30)
#opt_camel([-0.2,0],0.0000000001,100,1,0)
#opt_camel([-2,2],0.0000000001,200,0.4,0.39) #fajne
#opt_camel2([-1.5,1.9],1e-8,500,0.4,0)
#x2(1)
