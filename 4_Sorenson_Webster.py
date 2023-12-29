# Подключение бибилотек
from collections import defaultdict
from supportive_methods import *
from sympy import primefactors
from primesieve import n_primes
from os import path, makedirs
from math import prod
import gmpy2 as gm
import logging
import time


# Определение классов остатков для сигнатурного просеивания
def generate_residue_classes(coeff_wheel, w, leg_tup, coeff, lambda_siev_coeff):
    residues = []
    wheel(coeff_wheel, w, leg_tup, coeff, len(w) - 1, lambda_siev_coeff, residues)
    return residues


# Функция для нахождения сигнатур всех простых чисел 
# до заданной границы и их группировка
def get_signatures(low_border, up_border):
    global v, g
    sorted_primes_by_sign = defaultdict(list)

    for p in get_prime(low_border, up_border):
        lambd = find_lambda(v, [p])
        cur_sign = find_signature(v, p)
        sorted_primes_by_sign[cur_sign].append((p, lambd))

    return sorted_primes_by_sign


# Функция для определения оснований сигнатурного просеивания
def find_w(k, lambd_k):
    global v, g, w_border

    lambd_k_fact = primefactors(lambd_k)
    w = 1
    for i in range(1, len(v)):
        if k * lambd_k * v[i] * w >  w_border:
            break
        if v[i] not in lambd_k_fact:
            w = w * v[i]
    w *= 2 ** (lambd_k % 4 == 0) 
    return tuple(primefactors(w))


# Функция для определения массива чисел h
def find_h(sign, k):
    global v
    u, _ = gm.remove(k - 1, 2)
    h = []
    for c, b in zip(sign, v):
        if c > 0:
            h.append(pow(b, u * pow(2, c - 1)) + 1)
        else:
            h.append(pow(b, u) - 1)
    return h


# Поиск pt методом gcd
def search_pt_GCD(sign, pt_1, k, m):
    h = find_h(sign, k)
    h_b = min(h)
    b_i = h.index(h_b)
    i = 1 if b_i == 0 else 0
    x = h[i] % h_b
    x = gm.gcd(h_b, x)

    while x > pt_1 and i < (m - 1):
        i = i + 1
        if i == b_i:
            continue
        y = h[i] % x
        x = gm.gcd(x, y)
    
    if x <= pt_1:
        return
    
    global g, cur_spsp
    border = g / k

    for p_t in primefactors(x):
        if pt_1 < p_t < border and miller_rabin(v, k * p_t):
            cur_spsp.append(k * p_t)


# Функция для просеивания подходящих чисел
# по заданному классу остатков
def sieve(k, pt_1, r, mod):
    global g, cur_spsp

    low_border = (pt_1 - r) // mod + 1
    up_border = (g - k * r) // (k * mod) + 1

    for step in range(low_border, up_border):
        pt = r + step * mod
        if gm.is_prime(pt) and miller_rabin(v, k * pt):
            cur_spsp.append(k * pt)


# Функция для определения итоговых классов остатков для просеивания
# с учетом лямбды и классов остатков сигнатурного просеивания
def search_pt(use_leg_tup, coeff, add_cond, lambda_siev_coeff, pt_1, k, w):
    global leg_tup_one, coeff_wheel

    sol = crt(add_cond, lambda_siev_coeff)
    leg_tup = find_leg_tup(w, pt_1) if use_leg_tup else leg_tup_one
    cur_residues = generate_residue_classes(coeff_wheel, w, leg_tup, coeff, sol)
    for res in set(cur_residues):   
        sieve(k, pt_1, *res)   


# Функция для проверки кортежа для случая t>3
def tup_check_t_more_3(k, tup, sign):
    global v, X

    lambd = gm.lcm(*map(lambda el: el[1], tup))
    if gm.gcd(lambd, k) > 1:
        return

    c = gm.powmod(k, -1, lambd)
    sieve(k, tup[0][0], c, lambd)


# Функция для проверки кортежа для случая t=3
# и определения метода поиска pt
def tup_check_t_equal_3(k, tup, sign):
    global v, X

    lambd = gm.lcm(*map(lambda el: el[1], tup))
    if gm.gcd(lambd, k) > 1:
        return

    if k <= X:
        search_pt_GCD(sign, tup[0][0], k, len(v))
    else:
        c = gm.powmod(k, -1, lambd)
        lambda_signature_sieving(tup[0][0], (c, lambd), k)


# Рекурсивная функция для генерации подходящих кортежей для t>=3
def rec_comp_feas_tup(sign, primes, deep, up_border, tup, k=1, low_border=0):
    global g, X, v, tup_check
    i = low_border

    if deep != 0:
        while i < up_border:
            tup[deep] = primes[i]
            if rec_comp_feas_tup(sign, primes, deep - 1, up_border, tup, k * primes[i][0], i + 1) == 0:
                break
            i += 1
    else:
        while i < up_border:
            kf = k * primes[i][0]
            if kf * primes[i][0] > g:
                return i - low_border
            tup[deep] = primes[i]
            tup_check(kf, tup, sign)       
            i += 1
        
    return i - low_border


# Рекурсивная функция для генерации подходящих кортежей для t>=3
# с заранее известным значением pt-1
def rec_comp_feas_tup_with_known_p_t_1(sign, primes, deep, up_border, tup, k=1, low_border=0):
    global g, X, v, tup_check
    i = low_border

    if deep != 1:
        while i < up_border:
            tup[deep] = primes[i]
            if rec_comp_feas_tup_with_known_p_t_1(sign, primes, deep - 1, up_border, tup, k * primes[i][0], i + 1) == 0:
                break
            i += 1
    else:
        while i < up_border:
            kf = k * primes[i][0]
            if kf * tup[0][0] > g:
                break
            tup[deep] = primes[i]   
            tup_check(kf, tup, sign) 
            i += 1

    return i - low_border


# Функция для определения параметров сигнатурного просеивания для случая t=2,3
def lambda_signature_sieving(pt_1, lambda_siev_coeff, k):
    w = find_w(k, lambda_siev_coeff[1])
    if len(w) == 0:
        sieve(k, pt_1, *lambda_siev_coeff)
    elif pt_1 % 4 == 3:
        search_pt(True, 3, (0, 1), lambda_siev_coeff, pt_1, k, w)
        search_pt(False, 1, (0, 1), lambda_siev_coeff, pt_1, k, w)
    elif pt_1 % 8 == 5:
        search_pt(True, 1, (5, 8), lambda_siev_coeff, pt_1, k, w)
        search_pt(False, 1, (1, 8), lambda_siev_coeff, pt_1, k, w)
    else:
        _, e = gm.remove(pt_1 - 1, 2)
        _, f = gm.remove(lambda_siev_coeff[1], 2)
        if e == f:
            r = 2 ** e
            m1 = r * 2
            search_pt(True, 1, (1 + r, m1), lambda_siev_coeff, pt_1, k, w)
            search_pt(False, 1, (1, m1), lambda_siev_coeff, pt_1, k, w)
        elif f < e: 
            sieve(k, pt_1, *lambda_siev_coeff)


# Функция для поиска spsp с количеством простых множителей >= 3
def find_spsp_with_more_than_two_factors(sorted_primes_by_sign, t):
    start_time = time.perf_counter()
    for sign, primes in sorted_primes_by_sign.items():
        if len(primes) >= t - 1:
            rec_comp_feas_tup(sign, primes, t - 2, len(primes), [0] * (t - 1))
    logging.info(f'spsp {t} | part 1: {time.perf_counter() - start_time:.4f} seconds')

    start_time = time.perf_counter()
    global X, g, v
    border_up = gm.sqrt((g / prod(n_primes(t - 2, v[-1] + 2))))
    for prime in get_prime(X, border_up): 
        sign = find_signature(v, prime)
        primes = sorted_primes_by_sign[sign]
        if len(primes) >= t - 2:
            lambd = find_lambda(v, [prime])
            tup = [0] * (t - 1)
            tup[0] = (prime, lambd) 
            rec_comp_feas_tup_with_known_p_t_1(sign, primes, t - 2, len(primes), tup, prime)
    logging.info(f'spsp {t} | part 2: {time.perf_counter() - start_time:.4f} seconds')

            
# Функция для поиска spsp с количеством простых множителей = 2
def find_spsp_with_two_factors(sorted_primes_by_sign, up_border_p1):
    global g, v, X
    
    start_time = time.perf_counter()
    for sign, primes in sorted_primes_by_sign.items():
        for p1, _ in primes:
            search_pt_GCD(sign, p1, p1, len(v))
    logging.info(f'spsp 2 GCD : {time.perf_counter() - start_time:.4f} seconds')

    start_time = time.perf_counter()
    for p1 in get_prime(X, up_border_p1):
        lambda_p = find_lambda(v, [p1])
        lambda_signature_sieving(p1, (1, lambda_p), p1)
    logging.info(f'spsp 2 Lambda and sign sieve: {time.perf_counter() - start_time:.4f} seconds')


# Основная функция
def main():
    start_time_all = time.perf_counter()

    global v, g, cur_spsp, X, leg_tup_one, coeff_wheel, tup_check, w_border

    # Параметры поиска
    v = list(n_primes(12))
    g = 3186_65857_83403_11511_67461 + 1
    X = gm.root(g, 3) 
    coeff = 1000
    w_border = g / coeff
    leg_tup_one = tuple([0] * (len(v)))

    # Параметры для сохранения результатов
    spsp = defaultdict(list)
    dirname = f'result/{path.basename(__file__).split(".")[0]}/g = {create_name(g)}, v={len(v)}, coeff={coeff}'
    if not path.exists(dirname):
        makedirs(dirname)
    logging.basicConfig(filename=f'{dirname}/time.log',
                         filemode='w', 
                         level=logging.INFO,
                         format='%(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())
    
    # Генерация коэффициентов для сравнений 
    start_time = time.perf_counter()
    coeff_wheel = create_coeff_wheel(v, (1, 3), 4)
    logging.info(f'coeff_wheel : {time.perf_counter() - start_time:.4f} seconds')
    
    # Предварительное вычисление сигнатур
    start_time = time.perf_counter()
    sorted_primes_by_sign = get_signatures(v[-1], X) 
    logging.info(f'get_signatures : {time.perf_counter() - start_time:.4f} seconds')

    # Запуск поиска spsp для t > 3
    tup_check = tup_check_t_more_3
    for t in range(6, 3, -1):
        cur_spsp = spsp[t]
        start_time = time.perf_counter()
        find_spsp_with_more_than_two_factors(sorted_primes_by_sign, t=t)
        logging.info(f'spsp {t} : {time.perf_counter() - start_time:.4f} seconds')

    # Запуск поиска spsp для t = 3
    t = 3
    tup_check = tup_check_t_equal_3
    cur_spsp = spsp[t]
    start_time = time.perf_counter()
    find_spsp_with_more_than_two_factors(sorted_primes_by_sign, t=t)
    logging.info(f'spsp {t} : {time.perf_counter() - start_time:.4f} seconds')

    # Запуск поиска spsp для t = 2
    cur_spsp = spsp[2]
    start_time = time.perf_counter()
    find_spsp_with_two_factors(sorted_primes_by_sign, gm.sqrt(g))
    logging.info(f'spsp 2 : {time.perf_counter() - start_time:.4f} seconds')
    
    # Сохранение результата
    start_time = time.perf_counter()
    save_result(spsp, dirname)
    logging.info(f'save : {time.perf_counter() - start_time:.4f} seconds')

    logging.info(f'all : {time.perf_counter() - start_time_all:.4f} seconds')


main()