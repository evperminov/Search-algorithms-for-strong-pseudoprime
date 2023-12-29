# Подключение бибилотек
from collections import defaultdict
from supportive_methods import *
from primesieve import n_primes
from sympy import primefactors
from os import path, makedirs
import gmpy2 as gm
import logging
import time
import math


# Определение классов остатков для сигнатурного просеивания
def generate_residue_classes(v_sieve):
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
def get_signatures(low_border, up_border, v):
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


# Функции условия для просеивания
def cond_sieve_pt(pt, k):
    global v
    return gm.is_prime(pt) and miller_rabin(v, k * pt)

def cond_sieve_pt_1(pt, k):
    return gm.is_prime(pt)


#  Функции для определения границ просеивания 
def find_border_sieve_pt_1(pt_1, r, mod, k, g):
    return (pt_1 - r) // mod + 1, int((gm.sqrt(g / pt_1) - r) // mod) + 1

def find_border_sieve_pt(pt_1, r, mod, k, g):
    return (pt_1 - r) // mod + 1, (g - k * r) // (k * mod) + 1 


# Функция действия для просеивания
def act_add_spsp(k, tup):
    global cur_spsp
    cur_spsp.append(k)


# Функция для просеивания подходящих чисел
# по заданному классу остатков
def sieve(pt_1, r, mod, k, cfunc, afunc, bfunc):
    global g, cur_spsp

    for step in range(*bfunc(pt_1, r, mod, k, g)):
        pt = r + step * mod
        if cfunc(pt, k):
            afunc(k * pt, [pt, pt_1])


# Функция для определения итоговых классов остатков для просеивания
# с учетом лямбды и классов остатков сигнатурного просеивания
def search_pt(leg_tup, coeff, add_cond, lamb_siev_coeff, pt_1, params):
    global residues
    cur_residues = residues[coeff][leg_tup]

    sol = crt(add_cond, lamb_siev_coeff)
    solved = []
    for res in cur_residues:
        sol2 = crt(sol, res) 
        solved.append(sol2)

    for res in set(solved):   
        sieve(pt_1, *res, *params)   


# Функция для проверки кортежа для случая t>2
# и определения метода поиска pt
def tup_check(k, tup, use_gcd=True):
    global g, v, gcd_border

    lambd = find_lambda(v, tup)
    if gm.gcd(lambd, k) > 1:
        return
    
    if use_gcd and k < gcd_border: 
        search_pt_GCD(tup[0], k, g / k)
        return
    
    c = gm.powmod(k, -1, lambd)
    params = (k, cond_sieve_pt, act_add_spsp, find_border_sieve_pt)
    sieve(tup[0], c, lambd, *params)


# Рекурсивная функция для генерации подходящих кортежей
def rec_comp_feas_tup(sign, primes, deep, up_border, tup, k=1, low_border=0):
    global g
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
            tup_check(kf, tup, False)
            i += 1

    return i - low_border


# Функция для определения параметров сигнатурного просеивания для случая t=3
def lambda_signature_sieving_t3(leg_tup_zero, pt_1, lambd, leg_tup, params):
    if pt_1 % 4 == 3:
        search_pt(leg_tup, 3, (0, 1), (0, 1), pt_1, params)
        search_pt(leg_tup_zero, 1, (0, 1), (0, 1), pt_1, params)
    elif pt_1 % 8 == 5:
        search_pt(leg_tup, 1, (5, 8), (0, 1), pt_1, params)
        search_pt(leg_tup_zero, 1, (1, 8), (0, 1), pt_1, params)
    else:
        _, e = gm.remove(pt_1 - 1, 2)
        _, f = gm.remove(lambd, 2)
        if e == f: 
            r = 2 ** e
            m1 = r * 2
            m2 = m1 * 2
            search_pt(leg_tup, 1, (1 + r, m1), (0, 1), pt_1, params)
            search_pt(leg_tup_zero, 1, (1, m2), (0, 1), pt_1, params)
            search_pt(leg_tup_zero, 1, (1 + m1, m2), (0, 1), pt_1, params)
        elif f < e: 
            m3 = 2 ** f
            condition = crt((0, 1), (pt_1 % m3, m3))
            sieve(pt_1, *condition, *params)


# Функция для определения параметров сигнатурного просеивания для случая t=2
def lambda_signature_sieving_t2(leg_tup_zero, pt_1, lambda_siev_coeff, leg_tup, params):
    if pt_1 % 4 == 3:
        search_pt(leg_tup, 3, (0, 1), lambda_siev_coeff, pt_1, params)
        search_pt(leg_tup_zero, 1, (0, 1), lambda_siev_coeff, pt_1, params)
    elif pt_1 % 8 == 5:
        search_pt(leg_tup, 1, (5, 8), lambda_siev_coeff, pt_1, params)
        search_pt(leg_tup_zero, 1, (1, 8), lambda_siev_coeff, pt_1, params)
    else:
        _, e = gm.remove(pt_1 - 1, 2)
        _, f = gm.remove(lambda_siev_coeff[1], 2)
        if e == f: 
            r = 2 ** e
            m = r * 2
            search_pt(leg_tup, 1, (1 + r, m), lambda_siev_coeff, pt_1, params)
            search_pt(leg_tup_zero, 1, (1, m), lambda_siev_coeff, pt_1, params)
        elif f < e:
            sieve(pt_1, *lambda_siev_coeff, *params)
            

# Функция для поиска spsp с количеством простых множителей >= 4
def find_spsp_with_four_or_more_factors(sorted_primes_by_sign, t):
    for sign, primes in sorted_primes_by_sign.items():
        if len(primes) >= t - 1:
            rec_comp_feas_tup(sign, primes, t - 2, len(primes), [0] * (t - 1))


# Функция для поиска spsp с количеством простых множителей = 3
def find_spsp_with_three_factors(low_border_p1, up_border_p1, v_sieve):
    global residues, v
    params = [None, cond_sieve_pt_1, tup_check, find_border_sieve_pt_1]

    start_time = time.perf_counter()
    residues = generate_residue_classes(v_sieve)
    logging.info(f'residues t=3 : {time.perf_counter() - start_time:.4f} seconds')

    leg_tup_zero = tuple([0] * (len(v_sieve)))

    for p1 in get_prime(low_border_p1, up_border_p1):
        lambda_p = find_lambda(v, [p1])
        leg_tup = find_leg_tup(v_sieve, p1)
        params[0] = p1
        lambda_signature_sieving_t3(leg_tup_zero, p1, lambda_p, leg_tup, params)
    residues.clear()


# Функция для поиска spsp с количеством простых множителей = 2
def find_spsp_with_two_factors(low_border_p1, up_border_p1, border_methods, v_sieve):
    global g, v, residues
    params = [None, cond_sieve_pt, act_add_spsp, find_border_sieve_pt]

    cur_up_border = min(border_methods[0], up_border_p1)
    if low_border_p1 < cur_up_border:
        start_time = time.perf_counter()
        for p1 in get_prime(low_border_p1, cur_up_border):
            search_pt_GCD(p1, p1, g / p1)
        logging.info(f'spsp 2 Method GCD : {time.perf_counter() - start_time:.4f} seconds')

    cur_up_border = min(border_methods[1], up_border_p1)
    if border_methods[0] < cur_up_border:
        start_time = time.perf_counter()
        residues = generate_residue_classes(v_sieve)
        leg_tup_zero = tuple([0] * (len(v_sieve)))
        for p1 in get_prime(border_methods[0], cur_up_border):
            lambda_p = find_lambda(v, [p1])
            leg_tup = find_leg_tup(v_sieve, p1)
            params[0] = p1
            lambda_signature_sieving_t2(leg_tup_zero, p1, (1, lambda_p), leg_tup, params)
        logging.info(f'spsp 2 Method lambda and sign sieve: {time.perf_counter() - start_time:.4f} seconds')

    if border_methods[1] < up_border_p1:
        start_time = time.perf_counter()
        for p1 in get_prime(border_methods[1], up_border_p1):
            lambda_p = find_lambda(v, [p1])
            params[0] = p1
            sieve(p1, 1, lambda_p, *params)
        logging.info(f'spsp 2 Method lambda sieve : {time.perf_counter() - start_time:.4f} seconds')


# Основная функция
def main():

    start_time_all = time.perf_counter()

    global v, g, cur_spsp, gcd_border
    
    # Параметры поиска
    v = list(n_primes(9))
    t3_sign_siev_v_count = 8
    t2_sign_siev_v_count = 5
    g = 3825_12305_65464_13051 + 1
    gcd_border = 10 ** 6

    # Параметры для сохранения результатов
    spsp = defaultdict(list)
    dirname = f'result/{path.basename(__file__).split(".")[0]}/g = {create_name(g)}, v={len(v)}, v_t2={t2_sign_siev_v_count} v_t3={t3_sign_siev_v_count}'
    if not path.exists(dirname):
        makedirs(dirname) 
    logging.basicConfig(filename=f'{dirname}/time.log',
                         filemode='w', 
                         level=logging.INFO,
                         format='%(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())
        
    # Предварительное вычисление сигнатур
    start_time = time.perf_counter()
    up_border_sign = gm.sqrt(g  / math.prod(n_primes(2, v[-1] + 2)))
    sorted_primes_by_sign = get_signatures(v[-1], up_border_sign, v) 
    logging.info(f'get_signatures : {time.perf_counter() - start_time:.4f} seconds')

    # Запуск поиска spsp для t > 3
    for t in range(6, 3, -1):
        cur_spsp = spsp[t]
        start_time = time.perf_counter()
        find_spsp_with_four_or_more_factors(sorted_primes_by_sign, t)
        logging.info(f'spsp {t} : {time.perf_counter() - start_time:.4f} seconds')
    sorted_primes_by_sign.clear()
    
    # Запуск поиска spsp для t = 3
    cur_spsp = spsp[3]
    start_time = time.perf_counter()
    find_spsp_with_three_factors(v[-1], gm.root(g, 3), v[:t3_sign_siev_v_count])
    logging.info(f'spsp 3 : {time.perf_counter() - start_time:.4f} seconds')

    # Запуск поиска spsp для t = 2
    cur_spsp = spsp[2]
    start_time = time.perf_counter()
    find_spsp_with_two_factors(v[-1], gm.sqrt(g), (gcd_border, 10**8), v[:t2_sign_siev_v_count])
    logging.info(f'spsp 2 All time: {time.perf_counter() - start_time:.4f} seconds')
    
    # Сохранение результата
    start_time = time.perf_counter()
    save_result(spsp, dirname)
    logging.info(f'save : {time.perf_counter() - start_time:.4f} seconds')

    logging.info(f'all : {time.perf_counter() - start_time_all:.4f} seconds')


main() 