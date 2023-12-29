# Подключение бибилотек
from collections import defaultdict
from supportive_methods import *
from primesieve import n_primes
from sympy import primefactors
from os import path, makedirs
import gmpy2 as gm
import logging
import time


# Определение классов остатков для сигнатурного просеивания
def generate_residue_classes():
    global v_sieve
    r = (1, 3)
    wcoeff = create_coeff_wheel(v_sieve, r, 4)
    residues = defaultdict(dict)

    for dec in range(2 ** len(v_sieve)):
        leg_str = bin(dec)[2:].zfill(len(v_sieve))
        leg_tup = tuple(map(int, leg_str))

        for cur_r in r:
            cur_res = []
            wheel(wcoeff, v_sieve, leg_tup, cur_r, len(v_sieve) - 1, (0, 1), cur_res)
            residues[cur_r][leg_tup] = cur_res

    return residues


# Функция для нахождения сигнатур всех простых чисел 
# до заданной границы и их группировка
def get_signatures(low_border, up_border):
    global v, g
    sorted_primes_by_sign = defaultdict(list)

    for p in get_prime(low_border, up_border):
        cur_sign = find_signature(v, p)
        sorted_primes_by_sign[cur_sign].append(p)

    return sorted_primes_by_sign


# Поиск pt методом gcd
def search_pt_GCD(pt_1, k, up_border_pt):
    global v, cur_spsp 
    gcd = gm.gcd(*[pow(a, k - 1) - 1 for a in [2, 3, 5]]) 
    for p_t in primefactors(gcd): 
        if pt_1 < p_t < up_border_pt and miller_rabin(v, k * p_t):
            cur_spsp.append(k * p_t)  


# Функция для просеивания подходящих чисел pt
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
def search_pt(k, leg_tup, coeff, add_cond, lamb_siev_coeff, pt_1):
    global residues, v_sieve

    cur_residues = residues[coeff][leg_tup]

    sol = crt(add_cond, lamb_siev_coeff)
    solved = []
    for res in cur_residues:
        sol2 = crt(sol, res) 
        solved.append(sol2)

    for res in set(solved):   
        sieve(k, pt_1, *res)   


# Функция для определения особенностей сигнатурного просеивания pt
def search_pt_lambda_signature_sieving(k, pt_1, lambda_siev_coeff, leg_tup, max_sign=None, p1_sign=None):
    global v_sieve, leg_tup_zero

    if pt_1 % 4 == 3:
        search_pt(k, leg_tup, 3, (0, 1), lambda_siev_coeff, pt_1)
        search_pt(k, leg_tup_zero, 1, (0, 1), lambda_siev_coeff, pt_1)
    else:
        if 1 in leg_tup:
            if p1_sign is None:
                max_sign = gm.remove(pt_1 - 1, 2)[1]
            degree_2 = pow(2, max_sign)
            search_pt(k, leg_tup, 1, (degree_2 + 1, degree_2 * 2), lambda_siev_coeff, pt_1)
            search_pt(k, leg_tup_zero, 1, (1, degree_2 * 2), lambda_siev_coeff, pt_1)
        else:
            sieve(k, pt_1, *lambda_siev_coeff)


# Функция для проверки кортежа для случая t>3
def tup_check_t_more_3(k, tup, sign):
    lambd = find_lambda([2], tup)

    if gm.gcd(lambd, k) > 1:
        return

    c = gm.powmod(k, -1, lambd)
    sieve(k, tup[0], c, lambd)


# Функция для проверки кортежа для случая t=3
# и определения метода поиска pt
def tup_check_t_equal_3(k, tup, sign):
    global gcd_border

    lambd = find_lambda([2], tup)

    if gm.gcd(lambd, k) > 1:
        return
    
    if k < gcd_border:
        search_pt_GCD(tup[0], k, g / k)
    else:
        c = gm.powmod(k, -1, lambd)
        params = [k, tup[0], (c, lambd), sign[:len(v_sieve)], None, None]
        if tup[0] % 4 == 1:
            params[3] = find_leg_tup(v_sieve, tup[0])
            params[4] = max(sign)
            params[5] = sign
        search_pt_lambda_signature_sieving(*params)


# Рекурсивная функция для генерации подходящих кортежей
def rec_comp_feas_tup(sign, primes, deep, up_border, tup, k=1, low_border=0):
    global g, tup_check
    i = low_border

    if deep != 0:
        while i < up_border:
            tup[deep] = primes[i]
            if rec_comp_feas_tup(sign, primes, deep - 1, up_border, tup, k * primes[i], i + 1) == 0:
                break
            i += 1
    else:
        while i < up_border:
            kf = k * primes[i]
            if kf * primes[i] > g:
                break
            tup[deep] = primes[i]
            tup_check(kf, tup, sign)       
            i += 1
    return i - low_border


# Функция для поиска spsp с количеством простых множителей > 2
def find_spsp_with_more_than_two_factors(sorted_primes_by_sign, t):
    for sign, primes in sorted_primes_by_sign.items():
        if len(primes) >= t - 1:
            rec_comp_feas_tup(sign, primes, t - 2, len(primes), [0] * (t - 1))


# Функция для поиска spsp с количеством простых множителей = 2
def find_spsp_with_two_factors(low_border_p1, up_border_p1, border_methods):
    global g, v, v_sieve
    
    start_time = time.perf_counter()
    for p1 in get_prime(low_border_p1, min(border_methods, up_border_p1)):
        search_pt_GCD(p1, p1, g / p1)
    logging.info(f'spsp 2 GCD : {time.perf_counter() - start_time:.4f} seconds')

    start_time = time.perf_counter()
    for p1 in get_prime(border_methods, up_border_p1):
        lambda_p = gm.lcm(2, find_lambda(v, [p1]))
        leg_tup = find_leg_tup(v_sieve, p1)
        search_pt_lambda_signature_sieving(p1, p1, (1, lambda_p), leg_tup)
    logging.info(f'spsp 2 Lambda and sign sieve: {time.perf_counter() - start_time:.4f} seconds')


# Основная функция
def main():
    start_time_all = time.perf_counter()

    global v,v_sieve, g, cur_spsp, leg_tup_zero, residues, gcd_border, tup_check

    # Параметры поиска
    v = list(n_primes(8))
    v_sieve = v[:4]
    g = 34155_00717_28321 + 1
    gcd_border = 10 ** 5
    leg_tup_zero = tuple([0] * len(v_sieve))

    # Параметры для сохранения результатов
    spsp = defaultdict(list)
    dirname = f'result/{path.basename(__file__).split(".")[0]}/g = {create_name(g)}, v={len(v)}, v_sieve={len(v_sieve)}'
    if not path.exists(dirname):
        makedirs(dirname)
    logging.basicConfig(filename=f'{dirname}/time.log',
                         filemode='w', 
                         level=logging.INFO,
                         format='%(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())
    
    # Определение классов остатков для сигнатурного просеивания
    start_time = time.perf_counter()
    residues = generate_residue_classes()
    logging.info(f'residues : {time.perf_counter() - start_time:.4f} seconds')
    
    # Предварительное вычисление сигнатур
    start_time = time.perf_counter()
    sorted_primes_by_sign = get_signatures(v[-1], gm.sqrt(g / n_primes(1, v[-1] + 2)[0])) 
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
    sorted_primes_by_sign.clear()

    # Запуск поиска spsp для t = 2
    cur_spsp = spsp[2]
    start_time = time.perf_counter()
    find_spsp_with_two_factors(v[-1], gm.sqrt(g), gcd_border)
    logging.info(f'spsp 2 : {time.perf_counter() - start_time:.4f} seconds')
    
    # Сохранение результатов
    start_time = time.perf_counter()
    save_result(spsp, dirname)
    logging.info(f'save : {time.perf_counter() - start_time:.4f} seconds')

    logging.info(f'all : {time.perf_counter() - start_time_all:.4f} seconds')


main()