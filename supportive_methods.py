# Подключение библиотек
from collections import defaultdict
from sympy.ntheory import factorint
from primesieve import Iterator
import gmpy2 as gm


# Функция для поиска ord
def find_ord_single(a, n):
    order = n - 1
    factors = factorint(order)
    
    for p, e in factors.items():
        for _ in range(e):
            if gm.powmod(a, order // p, n) == 1: 
                order //= p
            else:
                break
            
    return order


# Функция для поиска ord для всех простых чисел - primes
# и множества оснований v
def find_ord(primes, v):
    ord = []
    for a in v:
        ord_cur_a = []
        for p in primes:
            x = find_ord_single(a, p)
            ord_cur_a.append(x)
        ord.append(ord_cur_a)
    return ord


# Функция для вычисления лямбда
def find_lambda(v, primes):
    ord = find_ord(primes, v)
    return gm.lcm(*sum(ord, []))


# Функция для вычисления сигнатуы обычным методом
def single_default_sign_search(a, p):
    ord = find_ord_single(a, p)
    _, degree = gm.remove(ord, 2)
    return degree


# Вычисление символа Лежандра для всех оснований
def find_leg_tup(v, p):
    sign = []
    for a in v:
        sign.append(int(gm.legendre(a, p) != 1))
    return tuple(sign)


# Функция для вычисления сигнатуры простого числа
def find_signature(v, p):
    sign = []

    if p % 4 == 3:
        sign = find_leg_tup(v, p) 
    else:
        for a in v:
            if gm.legendre(a, p) == -1:
                _, degree = gm.remove(p - 1, 2)
            else:
                degree = single_default_sign_search(a, p)
            sign.append(degree)

    return tuple(sign)


# Решение системы сравнений через китайскую теорема об остатках
def crt(eqn1, eqn2):
    x1, m1 = eqn1
    x2, m2 = eqn2
    d = gm.gcd(m1, m2)
    m = gm.lcm(m1, m2)
    l = m1 // d
    x = (x1 + l * (x2 - x1) * gm.invert(l, m2 // d)) % m
    return x, m


# Генерация коэффициентов для сравнений
def create_coeff_wheel(v, residue, module):
    coeff_wheel = defaultdict(list)
    for r in residue:
        for a in v:
            m = 4 * a
            coeff_a = [m, [[],[]]]
            for p in range(r, m, module):
                if p % a:
                    coeff_a[1][gm.legendre(a, p) != 1].append(p)
            coeff_wheel[(r, a)] = coeff_a
    return coeff_wheel


# Рекурсивная функция для создания и решения различных систем сравнений
def wheel(coeff_wheel, v_sieve, leg_tup, coeff, deep, last_solution, residues):
    m, cur_coeff = coeff_wheel[(coeff, v_sieve[deep])]

    if deep != 0:
        for p in cur_coeff[leg_tup[deep]]:
            cur_solution = crt((p, m), last_solution)
            wheel(coeff_wheel, v_sieve, leg_tup, coeff, deep - 1, cur_solution, residues)
    else:
        for p in cur_coeff[leg_tup[deep]]:
            cur_solution = crt((p, m), last_solution) 
            residues.append(cur_solution)



# Функция для генерации простых чисел
# в заданном промежутке
def get_prime(start, stop):
    it = Iterator()
    it.skipto(start)
    p = it.next_prime()
    while p <= stop:
        yield p
        p = it.next_prime()


# Фунция теста числа на простоту 
# с помощью алгоритма Миллера-Рабина.
def miller_rabin(v, n):
    t, s = n - 1, 0
    while t % 2 == 0:
        t, s = t // 2, s + 1

    label = False
    for i in range(len(v)):
        x = gm.powmod(v[i], t, n)
    
        if x == 1 or x == n - 1: 
            continue
        label = True

        for _ in range(s, 0, -1):
            x = gm.powmod(x, 2, n)
            if x == 1:
                break
            if x == n - 1: 
                label = False
                break
        if label:
            break

    return not label


# Функция для создания имени директории
def create_name(g):
    beaut_number = ''
    coeff, degree = gm.remove(g, 10)
    if coeff != 1:
        beaut_number += str(coeff)
    if degree != 0:
        if len(beaut_number):
            beaut_number += ' ⋅ ' 
        beaut_number += f'10 ^ {degree}'
    return beaut_number


# Функция для сохранения результатов поиска spsp
def save_result(spsp, dirname):
    for t, primes in spsp.items():
        if len(primes):
            with open(f'{dirname}/t = {t}, count = {len(primes)}.txt' , 'w') as file:
                spsp_dict = {}
                for p in primes:
                    factors = factorint(p, multiple=True)
                    spsp_dict[p] = f"{p} = {' * '.join(map(str, factors))}\n"

                for key in sorted(spsp_dict.keys()):
                    file.write(spsp_dict[key])