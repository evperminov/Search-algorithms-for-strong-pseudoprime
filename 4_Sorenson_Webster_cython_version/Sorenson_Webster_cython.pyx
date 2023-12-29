# cython: language_level=3
# Библиотеки
from libc.stdint cimport uint64_t, uint32_t, uint8_t, int8_t
from libc.stdlib cimport malloc, free
from libcpp.map cimport map as mapcpp
from sympy.ntheory import factorint
from collections import defaultdict
import logging, gmpy2 as gm, time
from libcpp.vector cimport vector
from cpython.array cimport array
from os import path, makedirs
from math import prod
cimport cython


# Работа с библиотеками из С++/C


# Primesieve - просеивание простых чисел
# Интерфейс взаимодействия с функциями
cdef extern from "primesieve.h":
    void* primesieve_generate_n_primes(uint64_t n, uint64_t start, int type)
    void* primesieve_generate_primes(uint64_t start, uint64_t stop, size_t* size, int type)
    void primesieve_free(void*)


cdef extern from "primesieve/iterator.hpp" namespace "primesieve":
    cdef cppclass iterator:
        iterator()
        iterator(uint64_t start, uint64_t stop_hint)
        uint64_t next_prime()
        uint64_t prev_prime()


cdef extern from *:
    '''
    #if PRIMESIEVE_VERSION_MAJOR >= 11
    #define iterator_jumpto(it, start, hint) it.jump_to(start, hint)
    #else
    #define iterator_jumpto(it, start, hint) it.skipto(start-1, hint)
    #endif
    '''
    void iterator_jumpto(iterator & it, uint64_t start, uint64_t stop_hint)


cdef extern from "primesieve.h":
    cdef enum:
        INT64_PRIMES
        ULONG_PRIMES
        ULONGLONG_PRIMES


cdef extern from 'errno.h':
    int errno


cdef array primes(uint64_t from_limit, uint64_t to_limit = 0) except +:
    """Generate a primes array from from_limit to to_limit (or up to
    from_limit if to_limit is unspecified or 0.)"""
    from_limit = max(from_limit, 0)
    to_limit = max(to_limit, 0)
    if to_limit == 0:
        (from_limit,to_limit) = (0,from_limit)

    # Rest errno
    global errno
    errno = 0

    cdef size_t size = 0
    cdef void* c_primes = primesieve_generate_primes(from_limit, to_limit, &size, ULONGLONG_PRIMES)

    if errno != 0:
        raise RuntimeError("Failed to generate primes, most likely due to insufficient memory.")

    cdef array primes = array("Q", (<char*>c_primes)[:(size*sizeof(unsigned long long))])
    primesieve_free(c_primes)
    return primes


cdef n_primes(uint64_t n, uint64_t start = 0) except +:
    """List the first n primes >= start."""
    n = max(n, 0)
    start = max(start, 0)

    # Rest errno
    global errno
    errno = 0

    cdef void* c_primes = primesieve_generate_n_primes(n, start, ULONGLONG_PRIMES)

    if errno != 0:
        raise RuntimeError("Failed to generate primes, most likely due to insufficient memory.")

    cdef array primes = array("Q", (<char*>c_primes)[:(n*sizeof(unsigned long long))])
    primesieve_free(c_primes)

    return primes


cdef class Iterator:
    cdef iterator _iterator
    def __cinit__(self):
        self._iterator = iterator()
    cdef void skipto(self, uint64_t start, uint64_t stop_hint = 2**62) except +:
        iterator_jumpto(self._iterator, start+1, stop_hint)
    cdef uint64_t next_prime(self) except +:
        return self._iterator.next_prime()
    cdef uint64_t prev_prime(self) except +:
        return self._iterator.prev_prime()


# Работа с большими числами
from gmpy2 cimport *


cdef extern from "gmp.h":
   #### MPZ ####
   # Initialization Functions
   void mpz_clear (mpz_t)
   void mpz_clears (mpz_t x, ...)
   void mpz_init (mpz_t)
   void mpz_inits (mpz_ptr, ...)

   # Assignment Functions
   void mpz_set_ui (mpz_t, unsigned long int)

   # Combined Initialization and Assignment Functions
   void mpz_init_set_ui (mpz_t, unsigned long int)
   void mpz_init_set (mpz_t, const mpz_t)

   # Conversion Functions
   unsigned long int mpz_get_ui (const mpz_t)

   # Arithmetic Functions
   void mpz_sub (mpz_t, const mpz_t, const mpz_t)
   void mpz_sub_ui (mpz_t, const mpz_t, unsigned long int)
   void mpz_mul (mpz_t, const mpz_t, const mpz_t)
   void mpz_mul_ui (mpz_t, const mpz_t, unsigned long int)
   void mpz_add (mpz_t, const mpz_t, const mpz_t)
   void mpz_add_ui (mpz_t, const mpz_t, unsigned long int)

   # Division Functions
   void mpz_fdiv_q (mpz_t, const mpz_t, const mpz_t)
   void mpz_mod (mpz_t, const mpz_t, const mpz_t)
   unsigned long int mpz_mod_ui (mpz_t, const mpz_t, unsigned long int)
   unsigned long int mpz_fdiv_q_ui (mpz_t, const mpz_t, unsigned long int)

   # Exponentiation Functions
   void mpz_powm(mpz_t, const mpz_t, const mpz_t, const mpz_t)
   void mpz_pow_ui (mpz_t, const mpz_t, unsigned long int)
   void mpz_powm_ui (mpz_t, const mpz_t, unsigned long int, const mpz_t)

   # Root Extraction Functions
   void mpz_sqrt (mpz_t, const mpz_t)
   int mpz_root (mpz_t, const mpz_t, unsigned long int n)

   # Number Theoretic Functions
   void mpz_gcd (mpz_t, const mpz_t, const mpz_t)
   int mpz_legendre (const mpz_t, const mpz_t)
   void mpz_lcm (mpz_t, const mpz_t, const mpz_t)
   void mpz_lcm_ui (mpz_t, const mpz_t, unsigned long)
   int mpz_probab_prime_p (const mpz_t, int reps)
   int mpz_invert (mpz_t, const mpz_t, const mpz_t)
   int mpz_remove (mpz_t, const mpz_t, const mpz_t)

   # Comparison Functions
   int mpz_cmp_ui (const mpz_t, unsigned long int)
   int mpz_cmp (const mpz_t, const mpz_t)
   

import_gmpy2() 


# Факторизация чисел
cdef extern from "factorize.c":
    struct factors:
        mpz_t         *p
        unsigned long *e
        long nfactors

ctypedef factors cfactors

cdef extern from "factorize.c":
    void factor(mpz_t, cfactors *)
    void factor_clear (cfactors *)
    void mpz_set (mpz_t, const mpz_t)


# Программа для поиска spsp


# Глобальные переменные
cdef mapcpp[vector[uint8_t], vector[vector[uint32_t]]] sorted_primes_by_sign
cdef mapcpp[vector[uint8_t], vector[vector[uint8_t]]] coeff_wheel
cdef vector[uint8_t] leg_tup_one
cdef mpz_t w_border_z
cdef vector[uint8_t] v
cdef mpz_t* v_z
cdef mpz_t g_z
cdef uint64_t X
cdef mpz_t X_z


############################################################################################
#### supportive_methods ####


# Функция для поиска ord
cdef void find_ord_single(mpz_t order_z, mpz_t a_z, mpz_t n_z):
    ### Инциализация
    cdef cfactors fact_cf
    cdef mpz_t n_1_z, degree_z  
    mpz_inits(n_1_z, degree_z, NULL)
    mpz_sub_ui(n_1_z, n_z, 1)
    mpz_set(order_z, n_1_z)

    ### Поиск X
    factor(n_1_z, &fact_cf)
    for i in range(fact_cf.nfactors):
        for _ in range(fact_cf.e[i]):
            mpz_fdiv_q(degree_z, order_z, fact_cf.p[i])
            mpz_powm(n_1_z, a_z, degree_z, n_z)
            if mpz_cmp_ui(n_1_z, 1) == 0:
                mpz_fdiv_q(order_z, order_z, fact_cf.p[i])
            else:
                break

    ### Очистка памяти
    mpz_clears(n_1_z, degree_z, NULL)
    factor_clear(&fact_cf) 


# Функция для вычисления лямбда
cdef void find_lambda(mpz_t lambda_z, mpz_t p_z):
    ### Инициализация
    global v_z, v
    cdef mpz_t tmp_z
    mpz_init(tmp_z)
    mpz_set_ui(lambda_z, 1)

    ### LCM(ORD)
    for i in range(v.size()):
        find_ord_single(tmp_z, v_z[i], p_z)
        mpz_lcm(lambda_z, tmp_z, lambda_z)

    ### Очистка памяти
    mpz_clears(tmp_z, NULL)

    
# Вычисление символа Лежандра для всех оснований
cdef vector[uint8_t] find_leg_tup(mpz_t* v_z, uint8_t length, mpz_t p_z):
    cdef vector[uint8_t] sign
    cdef uint8_t leg
    for i in range(length):
        leg = mpz_legendre(v_z[i], p_z) != 1
        sign.push_back(leg)
    return sign


# Функция для вычисления сигнатуры простого числа
cdef vector[uint8_t] find_signature(mpz_t p_z, uint64_t p):
    ### Инициализация
    global v_z, v
    cdef vector[uint8_t] sign
    cdef mpz_t p_1_z, f_z, ord_z
    cdef int8_t leg
    cdef uint64_t degree

    ### Определение метода для вычисления сигнатуры и ее нахождение
    if p % 4 == 3:
        sign = find_leg_tup(v_z, v.size(), p_z)
    else:
        ## Инициализация
        mpz_init_set_ui(p_1_z, p - 1)
        mpz_inits(f_z, ord_z, NULL)

        for i in range(v.size()):
            leg = mpz_legendre(v_z[i], p_z)
            if leg == -1:
                degree = mpz_remove(f_z, p_1_z, v_z[0])
            else:
                find_ord_single(ord_z, v_z[i], p_z)
                degree = mpz_remove(f_z, ord_z, v_z[0])
            sign.push_back(degree)
        
        # Очистка памяти
        mpz_clears(p_1_z, f_z, ord_z, NULL)

    return sign


# Решение системы сравнений через китайскую теорема об остатках
cdef void crt(mpz_t x_z, mpz_t m_z, mpz_t x1_z, mpz_t m1_z, mpz_t x2_z, mpz_t m2_z):
    ### Инициализация
    cdef mpz_t d_z, l_z, tmp_z
    mpz_inits(d_z, l_z, tmp_z, NULL)

    ### Решение системы сравнений
    mpz_gcd(d_z, m1_z, m2_z)
    mpz_lcm(m_z, m1_z, m2_z)
    mpz_fdiv_q(l_z, m1_z, d_z)
    mpz_fdiv_q(tmp_z, m2_z, d_z)
    mpz_invert(x_z, l_z, tmp_z)
    mpz_sub(tmp_z, x2_z, x1_z)
    mpz_mul(x_z, tmp_z, x_z)
    mpz_mul(x_z, x_z, l_z)
    mpz_add(x_z, x_z, x1_z)
    mpz_mod(x_z, x_z, m_z)

    ### Очистка памяти
    mpz_clears(d_z, l_z, tmp_z, NULL)


# Генерация коэффициентов для сравнений
cdef void create_coeff_wheel(vector[uint8_t]& residue, uint8_t module):
    ### Инициализация
    global v, v_z, coeff_wheel
    cdef uint8_t m, p
    cdef vector[vector[uint8_t]] coeff_a
    coeff_a.resize(2)
    cdef vector[uint8_t] keyy
    keyy.resize(2)
    cdef mpz_t p_z
    mpz_init(p_z)

    ## Определение коэффициентов и сохранение
    for i in range(residue.size()):
        for j in range(v.size()):
            m = 4 * v[j]
            coeff_a[0] = vector[uint8_t]()
            coeff_a[1] = vector[uint8_t]()

            p = residue[i]
            while p < m:
                if p % v[j]:
                    mpz_set_ui(p_z, p)
                    coeff_a[mpz_legendre(v_z[j], p_z) != 1].push_back(p)
                p = p + module

            keyy[0] = residue[i]
            keyy[1] = v[j]
            coeff_wheel[keyy] = coeff_a

    ### Очистка памяти
    mpz_clear(p_z)


# Рекурсивная функция для создания и решения различных систем сравнений
cdef void wheel(cfactors v_sieve, vector[uint8_t] leg_tup, uint8_t coeff, uint8_t deep, mpz_t r_z, mpz_t m_z, vector[vector[uint64_t]]& residues):
    ### Инициализация
    global coeff_wheel 
    cdef uint64_t base = mpz_get_ui(v_sieve.p[deep])

    cdef vector[uint8_t] keyy
    keyy.push_back(coeff)
    keyy.push_back(base)

    cdef vector[uint64_t] valuee
    valuee.resize(2)

    cdef vector[uint8_t] cur_coeff = coeff_wheel[keyy][leg_tup[deep]]
    cdef mpz_t cur_m_z, cur_r_z, sol_r_z, sol_m_z
    mpz_inits(cur_r_z, sol_r_z, sol_m_z, NULL)
    mpz_init_set_ui(cur_m_z, 4 * base )
    
    ### Решение систем сравнений для len(w) - 1 оснований
    if deep != 0:
        for i in range(cur_coeff.size()):
            mpz_set_ui(cur_r_z, cur_coeff[i])
            crt(sol_r_z, sol_m_z, cur_r_z, cur_m_z, r_z, m_z)
            wheel(v_sieve, leg_tup, coeff, deep - 1, sol_r_z, sol_m_z, residues)
    else:
    ### Решение последней системы сравнений и сохранение классов остаков
        for i in range(cur_coeff.size()):
            mpz_set_ui(cur_r_z, cur_coeff[i])
            crt(sol_r_z, sol_m_z, cur_r_z, cur_m_z, r_z, m_z)
            valuee[0] = mpz_get_ui(sol_r_z)
            valuee[1] = mpz_get_ui(sol_m_z)
            residues.push_back(valuee)

    ### Очистка памяти
    mpz_clears(cur_m_z, cur_r_z, sol_r_z, sol_m_z, NULL)


# Фунция теста числа на простоту 
# с помощью алгоритма Миллера-Рабина
cdef bint miller_rabin(mpz_t n):
    ### Инициализация 
    global v, v_z
    cdef uint64_t s
    cdef bint label = False
    cdef mpz_t t, x, n_1
    mpz_inits(x, n_1, t, NULL)

    ### Проверка 
    mpz_sub_ui(n_1, n, 1)
    s = mpz_remove(t, n_1, v_z[0])

    for i in range(v.size()):
        mpz_powm(x, v_z[i], t, n)

        if mpz_cmp_ui(x, 1) == 0 or mpz_cmp(x, n_1) == 0:
            continue
        label = True
        for _ in range(s, 0, -1):
            mpz_powm_ui(x, x, 2, n)
            if mpz_cmp_ui(x, 1) == 0:
                break
            if mpz_cmp(x, n_1) == 0:
                label = False
                break
        if label:
            break

    ### Очистка памяти
    mpz_clears(t, x, n_1, NULL)

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


############################################################################################
#### main ####


# Определение классов остатков для сигнатурного просеивания
cdef vector[vector[uint64_t]] generate_residue_classes(cfactors v_sieve, vector[uint8_t] leg_tup, uint8_t coeff, mpz_t r_z, mpz_t m_z):
    cdef vector[vector[uint64_t]] residues
    wheel(v_sieve, leg_tup, coeff, v_sieve.nfactors - 1, r_z,  m_z, residues)
    return residues


# Функция для нахождения сигнатур всех простых чисел 
# до заданной границы и их группировка
cdef void get_signatures():
    ### Инициализация
    global v, sorted_primes_by_sign, X
    cdef vector[uint8_t] sign 
    it = Iterator()
    it.skipto(v[v.size() - 1])
    cdef uint64_t lambd, p = it.next_prime()
    cdef mpz_t p_z, lambda_z
    mpz_init_set_ui(p_z, p)
    mpz_init(lambda_z)
    cdef vector[uint32_t] cur_part
    cur_part.resize(2)

    ### Вычисление сигнатуры и lamda для каждого p
    while p < X:
        find_lambda(lambda_z, p_z)
        sign = find_signature(p_z, p)
        cur_part[0] = p
        cur_part[1] = mpz_get_ui(lambda_z)
        sorted_primes_by_sign[sign].push_back(cur_part)
        p = it.next_prime()
        mpz_set_ui(p_z, p)

    ### Очистка памяти
    mpz_clears(p_z, lambda_z, NULL)


# Функция для определения оснований сигнатурного просеивания
cdef cfactors find_w(mpz_t k_z, mpz_t lambd_k_z):
    ### Инициализация
    global v, v_z, w_border_z
    cdef mpz_t w_z, tmp_z, ratio_z, lambd_k_copy_z
    cdef cfactors lambd_k_fact
    cdef bint flag
    mpz_inits(ratio_z, tmp_z, NULL)
    mpz_init_set_ui(w_z, 1)
    mpz_init_set(lambd_k_copy_z, lambd_k_z)
    mpz_mul(ratio_z, k_z, lambd_k_z)
    factor(lambd_k_copy_z, &lambd_k_fact)

    ### Отбор a для w
    for i in range(1, v.size()):
        mpz_set(tmp_z, ratio_z)
        mpz_mul(tmp_z, tmp_z, v_z[i])
        mpz_mul(tmp_z, tmp_z, w_z)

        if mpz_cmp(tmp_z, w_border_z) > 0:
            break
        
        flag = True
        for j in range(lambd_k_fact.nfactors):
            if mpz_cmp(lambd_k_fact.p[j], v_z[i]) == 0:
                flag = False
                break
        if flag:
            mpz_mul(w_z, w_z, v_z[i])

    mpz_mod_ui(ratio_z, lambd_k_z, 4)
    if mpz_cmp_ui(ratio_z, 0) == 0:
        mpz_mul_ui(w_z, w_z, 2)

    factor_clear(&lambd_k_fact) 
    factor(w_z, &lambd_k_fact)

    ### Очистка памяти 
    mpz_clears(w_z, tmp_z, ratio_z, lambd_k_copy_z, NULL)

    return lambd_k_fact 


# Функция для определения массива чисел h
cdef void find_h(mpz_t* h_z, vector[uint8_t]& sign, mpz_t k):
    ### Инициализация
    global v_z, v
    cdef mpz_t u_z, k_1_z, h_i_z, tmp_z
    mpz_inits(u_z, k_1_z, h_i_z, tmp_z, NULL)
    mpz_sub_ui(k_1_z, k, 1)
    mpz_remove(u_z, k_1_z, v_z[0])
    cdef uint64_t u = mpz_get_ui(u_z)

    ### Вычисление h_i
    for i in range(v.size()):
        if sign[i] > 0:
            mpz_pow_ui(tmp_z, v_z[0], sign[i] - 1)
            mpz_pow_ui(h_i_z, v_z[i], u * mpz_get_ui(tmp_z))
            mpz_add_ui(h_i_z, h_i_z, 1)
        else:
            mpz_pow_ui(h_i_z, v_z[i], u)
            mpz_sub_ui(h_i_z, h_i_z, 1)
        mpz_set(h_z[i], h_i_z)

    ### Очистка памяти
    mpz_clears(u_z, k_1_z, h_i_z, tmp_z, NULL)


# Поиск pt методом gcd
cdef void search_pt_GCD(vector[uint8_t]& sign, mpz_t pt_1_z, mpz_t k_z): 
    ### Инициализация 
    global v, g_z, cur_spsp
    cdef mpz_t x_z, y_z, border_z, output_z
    mpz_inits(x_z, y_z, border_z, output_z, NULL)
    cdef uint8_t b_i = 0, i
    cdef cfactors fact_cf
    mpz_fdiv_q(border_z, g_z, k_z)
    cdef mpz_t* h_z = <mpz_t*>malloc(v.size() * sizeof(mpz_t))
    for i in range(v.size()):
        mpz_init(h_z[i])

    ### Поиск h 
    find_h(h_z, sign, k_z)

    ### Определение минимального h_i
    for i in range(1, v.size()):
        if mpz_cmp(h_z[b_i], h_z[i]) > 0:
            b_i = i
    i = 1 if b_i == 0 else 0

    ### Вычисление x
    mpz_mod(x_z, h_z[i], h_z[b_i])
    mpz_gcd(x_z, h_z[b_i], x_z)
    while mpz_cmp(x_z, pt_1_z) > 0 and i < (v.size() - 1):
        i = i + 1
        if i == b_i:
            continue
        mpz_mod(y_z, h_z[i], x_z)
        mpz_gcd(x_z, x_z, y_z)
    
    ### Если условие не выполняется, то прерываем
    if mpz_cmp(x_z, pt_1_z) > 0:
        ### Факторизуем и ищемп подходящих pt
        factor(x_z, &fact_cf)
        for i in range(fact_cf.nfactors):
            if mpz_cmp(fact_cf.p[i], pt_1_z) > 0 and mpz_cmp(border_z, fact_cf.p[i]) > 0:
                mpz_mul(output_z, k_z, fact_cf.p[i])
                if miller_rabin(output_z):
                    cur_spsp.append(int(GMPy_MPZ_From_mpz(output_z)))

        ### Очистка памяти 
        factor_clear(&fact_cf)

    ### Очистка памяти     
    mpz_clears(x_z, y_z, output_z, border_z, NULL)
    for i in range(v.size()):
        mpz_clear(h_z[i])
    free(h_z)


# Функция для просеивания подходящих чисел
# по заданному классу остатков
cdef void sieve(mpz_t k_z, mpz_t pt_1_z, mpz_t r_z, mpz_t mod_z):
    ### Инициализация
    global g_z, cur_spsp
    cdef mpz_t low_border_z, up_border_z, tmp_z
    cdef mpz_t step_z, pt_z, output
    mpz_inits(tmp_z, pt_z, output, NULL)

    ### Определение нижней границы
    mpz_init_set(low_border_z, pt_1_z)
    mpz_sub(low_border_z, low_border_z, r_z)
    mpz_fdiv_q(low_border_z, low_border_z, mod_z)
    mpz_add_ui(low_border_z, low_border_z, 1)

    ### Определение верней границы
    mpz_init_set(up_border_z, g_z)
    mpz_mul(tmp_z, k_z, r_z)
    mpz_sub(up_border_z, up_border_z, tmp_z)
    mpz_mul(tmp_z, k_z, mod_z) 
    mpz_fdiv_q(up_border_z, up_border_z, tmp_z)
    mpz_add_ui(up_border_z, up_border_z, 1)
    
    # Генерация подходящих pt
    mpz_init_set(step_z, low_border_z)
    while mpz_cmp(up_border_z, step_z) > 0:
        mpz_mul(pt_z, step_z, mod_z)
        mpz_add(pt_z, pt_z, r_z)

        if mpz_probab_prime_p(pt_z, 15) > 0:
            mpz_mul(output, k_z, pt_z)
            if miller_rabin(output):
                cur_spsp.append(int(GMPy_MPZ_From_mpz(output)))
        
        mpz_add_ui(step_z, step_z, 1)

    ### Очистка памяти
    mpz_clears(low_border_z, up_border_z, tmp_z, step_z, pt_z, output, NULL)


# Функция для определения итоговых классов остатков для просеивания
# с учетом лямбды и классов остатков сигнатурного просеивания
cdef void search_pt(bint use_leg_tup, uint8_t coeff, mpz_t r_d_z, mpz_t modd_d_z, mpz_t r_l_z, mpz_t lambd_z, mpz_t pt_1_z, mpz_t k_z, cfactors w_cf):
    ### Инициализация
    global leg_tup_one
    cdef vector[uint8_t] leg_tup 
    cdef vector[vector[uint64_t]] cur_residues
    cdef mpz_t sol_r_z, sol_m_z, cur_r_z, cur_m_z
    mpz_inits(sol_r_z, sol_m_z, cur_r_z, cur_m_z, NULL)

    ### Определение классов остатков для просеивания
    crt(sol_r_z, sol_m_z, r_d_z, modd_d_z, r_l_z, lambd_z)
    leg_tup = find_leg_tup(w_cf.p, w_cf.nfactors, pt_1_z) if use_leg_tup else leg_tup_one
    cur_residues = generate_residue_classes(w_cf, leg_tup, coeff, sol_r_z, sol_m_z)
    
    ### Запуск лямбда-сигнатурного просеивания 
    for i in range(cur_residues.size()): 
        mpz_set_ui(cur_r_z, cur_residues[i][0])
        mpz_set_ui(cur_m_z, cur_residues[i][1])
        sieve(k_z, pt_1_z, cur_r_z, cur_m_z)

    ### Очистка памяти
    mpz_clears(cur_r_z, cur_m_z, sol_r_z, sol_m_z, NULL)


## Функция для проверки кортежа и запуска просеивания для t > 3
cdef void tup_check_t_more_3(mpz_t k_z, vector[vector[uint32_t]] & tup, vector[uint8_t] & sign):
    ### Инициализация
    cdef mpz_t lambda_z, tmp_z, pt_1_z, c_z
    mpz_init_set_ui(lambda_z, 1)
    mpz_inits(tmp_z, c_z, NULL)
    mpz_init_set_ui(pt_1_z, tup[0][0])

    ### Вычисление lambd = lcm(ord...)
    for i in range(tup.size()):
        mpz_lcm_ui(lambda_z, lambda_z, tup[i][1])

    ### Проверка условия
    mpz_gcd(tmp_z, lambda_z, k_z)
    if mpz_cmp_ui(tmp_z, 1) > 0:
        ### Очистка памяти
        mpz_clears(lambda_z, tmp_z, pt_1_z, c_z, NULL)
        return

    ### Запуск лямбда-просеивания
    mpz_invert(c_z, k_z, lambda_z)
    sieve(k_z, pt_1_z, c_z, lambda_z)

    ### Очистка памяти
    mpz_clears(lambda_z, tmp_z, pt_1_z, c_z, NULL)


## Функция для проверки кортежа, выбора метода поиска p_t и его запуск для t=3
cdef void tup_check_t_equal_3(mpz_t k_z, vector[vector[uint32_t]] & tup, vector[uint8_t] & sign):
    ### Инициализация
    global X_z
    cdef mpz_t lambda_z, tmp_z, pt_1_z, c_z
    mpz_init_set_ui(lambda_z, 1)
    mpz_inits(tmp_z, c_z, NULL)
    mpz_init_set_ui(pt_1_z, tup[0][0])

    ### Вычисление lambd = lcm(ord...)
    for i in range(tup.size()):
        mpz_lcm_ui(lambda_z, lambda_z, tup[i][1])

    ### Проверка условия
    mpz_gcd(tmp_z, lambda_z, k_z)
    if mpz_cmp_ui(tmp_z, 1) > 0:
        ### Очистка памяти
        mpz_clears(lambda_z, tmp_z, pt_1_z, c_z, NULL)
        return

    ### Выбор метода поиска и его запуск
    if mpz_cmp(k_z, X_z) <= 0:
        search_pt_GCD(sign, pt_1_z, k_z)
    else:
        mpz_invert(c_z, k_z, lambda_z)
        lambda_signature_sieving(pt_1_z, c_z, lambda_z, k_z)

    ### Очистка памяти
    mpz_clears(lambda_z, tmp_z, pt_1_z, c_z, NULL)


# Рекурсивная функция для генерации подходящих кортежей для t>3
cdef uint32_t rec_comp_feas_tup_more_t3(vector[uint8_t] & sign, vector[vector[uint32_t]] & primes, uint8_t deep, uint32_t up_border, vector[vector[uint32_t]] & tup, mpz_t k_z, uint32_t low_border):
    ### Инициализация
    global g_z, tup_check
    cdef uint32_t i = low_border
    cdef mpz_t cur_k_z, cur_k_check_z
    mpz_inits(cur_k_z, cur_k_check_z, NULL)

    ### Перебор чисел для подходящего кортежа на позиции i < t - 1
    if deep !=0:
        while i < up_border:
            tup[deep] = primes[i] 
            mpz_mul_ui(cur_k_z, k_z, primes[i][0])
            if rec_comp_feas_tup_more_t3(sign, primes, deep - 1, up_border, tup, cur_k_z, i + 1) == 0:
                break
            i += 1
    else:
        ### Перебор на позицию i = t - 1 
        while i < up_border:
            mpz_mul_ui(cur_k_z, k_z, primes[i][0])
            mpz_mul_ui(cur_k_check_z, cur_k_z, primes[i][0])

            if mpz_cmp(cur_k_check_z, g_z) > 0:
                break
            
            tup[deep] = primes[i]
            tup_check_t_more_3(cur_k_z, tup, sign)
            i = i + 1

    ### Очистка памяти
    mpz_clears(cur_k_z, cur_k_check_z, NULL)

    return i - low_border

# Рекурсивная функция для генерации подходящих кортежей для t>3
# с заранее известным значением pt-1
cdef uint32_t rec_comp_feas_tup_more_t3_with_known_p_t_1(vector[uint8_t] & sign, vector[vector[uint32_t]] & primes, uint8_t deep, uint32_t up_border, vector[vector[uint32_t]] & tup, mpz_t k_z, uint32_t low_border):
    ### Инициализация 
    global g_z, tup_check
    cdef uint32_t i = low_border
    cdef mpz_t cur_k_z, cur_k_check_z
    mpz_inits(cur_k_z, cur_k_check_z, NULL)

    ###  Перебор чисел для подходящего кортежа на позиции i < t - 2
    if deep != 1:
        while i < up_border:
            tup[deep] = primes[i]
            mpz_mul_ui(cur_k_z, k_z, primes[i][0])
            if rec_comp_feas_tup_more_t3_with_known_p_t_1(sign, primes, deep - 1, up_border, tup, cur_k_z, i + 1) == 0:
                break
            i += 1
    else:
        ### Перебор на позицию i = t - 2
        while i < up_border:
            mpz_mul_ui(cur_k_z, k_z, primes[i][0])
            mpz_mul_ui(cur_k_check_z, cur_k_z, tup[0][0])

            if mpz_cmp(cur_k_check_z, g_z) > 0:
                break
            
            tup[deep] = primes[i]
            tup_check_t_more_3(cur_k_z, tup, sign)
            i = i + 1

    ### Очистка памяти
    mpz_clears(cur_k_z, cur_k_check_z, NULL)

    return i - low_border

# Рекурсивная функция для генерации подходящих кортежей для t=3
cdef uint32_t rec_comp_feas_tup_t3(vector[uint8_t] & sign, vector[vector[uint32_t]] & primes, uint8_t deep, uint32_t up_border, vector[vector[uint32_t]] & tup, mpz_t k_z, uint32_t low_border):
    ### Инициализация
    global g_z, tup_check
    cdef uint32_t i = low_border
    cdef mpz_t cur_k_z, cur_k_check_z
    mpz_inits(cur_k_z, cur_k_check_z, NULL)

    ### Перебор чисел для подходящего кортежа на позиции i < t - 1
    if deep !=0:
        while i < up_border:
            tup[deep] = primes[i]
            mpz_mul_ui(cur_k_z, k_z, primes[i][0])
            if rec_comp_feas_tup_t3(sign, primes, deep - 1, up_border, tup, cur_k_z, i + 1) == 0:
                break
            i += 1
    else:
        ### Перебор на позицию i = t - 1 
        while i < up_border:
            mpz_mul_ui(cur_k_z, k_z, primes[i][0])
            mpz_mul_ui(cur_k_check_z, cur_k_z, primes[i][0])

            if mpz_cmp(cur_k_check_z, g_z) > 0:
                break
            
            tup[deep] = primes[i]
            tup_check_t_equal_3(cur_k_z, tup, sign)
            i = i + 1

    ### Очистка памяти
    mpz_clears(cur_k_z, cur_k_check_z, NULL)

    return i - low_border

# Рекурсивная функция для генерации подходящих кортежей для t=3
# с заранее известным значением pt-1
cdef uint32_t rec_comp_feas_tup_t3_with_known_p_t_1(vector[uint8_t] & sign, vector[vector[uint32_t]] & primes, uint8_t deep, uint32_t up_border, vector[vector[uint32_t]] & tup, mpz_t k_z, uint32_t low_border):
    ### Инициализация 
    global g_z, tup_check
    cdef uint32_t i = low_border
    cdef mpz_t cur_k_z, cur_k_check_z
    mpz_inits(cur_k_z, cur_k_check_z, NULL)

    ###  Перебор чисел для подходящего кортежа на позиции i < t - 2
    if deep != 1:
        while i < up_border:
            tup[deep] = primes[i]
            mpz_mul_ui(cur_k_z, k_z, primes[i][0])
            if rec_comp_feas_tup_t3_with_known_p_t_1(sign, primes, deep - 1, up_border, tup, cur_k_z, i + 1) == 0:
                break
            i += 1
    else:
        ### Перебор на позицию i = t - 2
        while i < up_border:
            mpz_mul_ui(cur_k_z, k_z, primes[i][0])
            mpz_mul_ui(cur_k_check_z, cur_k_z, tup[0][0])

            if mpz_cmp(cur_k_check_z, g_z) > 0:
                break
            
            tup[deep] = primes[i]
            tup_check_t_equal_3(cur_k_z, tup, sign)
            i = i + 1

    ### Очистка памяти
    mpz_clears(cur_k_z, cur_k_check_z, NULL)

    return i - low_border


# Функция для определения параметров сигнатурного просеивания для случая t=2,3
cdef void lambda_signature_sieving(mpz_t pt_1_z, mpz_t r_l_z, mpz_t lambd_z, mpz_t k_z):
    ## Инициализация
    global v_z
    cdef uint64_t e, f
    cdef cfactors w_cf = find_w(k_z, lambd_z)
    cdef mpz_t temp_z, r_d_z, modd_d_z, tmp2_z
    mpz_inits(temp_z, r_d_z, modd_d_z, tmp2_z, NULL)

    ## Если оснований w для сигнатурного просеивания 0, то 
    ## запускаем только лямбда просеивание
    if w_cf.nfactors == 0:
        sieve(k_z, pt_1_z, r_l_z, lambd_z)

        mpz_clears(temp_z, r_d_z, modd_d_z, tmp2_z, NULL)
        factor_clear(&w_cf)
        return
     
    ## Если pt_1 % 4 == 3
    mpz_mod_ui(temp_z, pt_1_z, 4)
    if mpz_cmp_ui(temp_z, 3) == 0:
        # p1 = 3 mod 4 and p2 = 3 mod 4
        mpz_set_ui(r_d_z, 0)
        mpz_set_ui(modd_d_z, 1)
        search_pt(True, 3, r_d_z, modd_d_z, r_l_z, lambd_z, pt_1_z, k_z, w_cf)
        # p1 = 3 mod 4 and p2 = 1 mod 4 (bin sign)
        search_pt(False, 1, r_d_z, modd_d_z, r_l_z, lambd_z, pt_1_z, k_z, w_cf)   

        mpz_clears(temp_z, r_d_z, modd_d_z, tmp2_z, NULL)
        factor_clear(&w_cf)   
        return
    
    ## Если pt_1 % 8 == 5
    mpz_mod_ui(temp_z, pt_1_z, 8)
    if mpz_cmp_ui(temp_z, 5) == 0:
        # p1 = 5 mod 8 and p2 = 5 mod 8
        mpz_set_ui(r_d_z, 5)
        mpz_set_ui(modd_d_z, 8)      
        search_pt(True, 1, r_d_z, modd_d_z, r_l_z, lambd_z, pt_1_z, k_z, w_cf)  
        # p1 = 5 mod 8 and p2 = 1 mod 8
        mpz_set_ui(r_d_z, 1)
        mpz_set_ui(modd_d_z, 8)
        search_pt(False, 1, r_d_z, modd_d_z, r_l_z, lambd_z, pt_1_z, k_z, w_cf)
        
        mpz_clears(temp_z, r_d_z, modd_d_z, tmp2_z, NULL)
        factor_clear(&w_cf)
        return

    ## Если pt_1 % 8 == 1
    mpz_sub_ui(temp_z, pt_1_z, 1)
    e = mpz_remove(tmp2_z, temp_z, v_z[0])
    f = mpz_remove(tmp2_z, lambd_z, v_z[0])

    if e == f:
        mpz_pow_ui(r_d_z, v_z[0], e) 
        mpz_mul(modd_d_z, r_d_z, v_z[0])
        mpz_add_ui(r_d_z, r_d_z, 1)
        # p1 = 1 mod 8 and p2 = 1 + 2 ^ e mod 2 ^(e + 1)
        search_pt(True, 1, r_d_z, modd_d_z, r_l_z, lambd_z, pt_1_z, k_z, w_cf)  
        # p1 = 1 mod 8 and p2 = 1 mod 2 ^ (e + 1)
        mpz_set_ui(r_d_z, 1)
        search_pt(False, 1, r_d_z, modd_d_z, r_l_z, lambd_z, pt_1_z, k_z, w_cf)  

    elif f < e: # 1 not in sign
        # p1 = 1 mod 8 and p2 = 1 mod lambda_p1
        sieve(k_z, pt_1_z, r_l_z, lambd_z)

    ### Очистка памяти
    mpz_clears(temp_z, r_d_z, modd_d_z, tmp2_z, NULL)
    factor_clear(&w_cf)


## Функция для поиска spsp с количеством простых множителей > 3
cdef void find_spsp_with_more_than_three_factors(uint8_t t):
    ### Инициализация 
    global sorted_primes_by_sign, v, g_z, X
    cdef mpz_t p_z, k_z, border_up_z, lambda_z
    mpz_inits(p_z, k_z, border_up_z, lambda_z, NULL)

    cdef vector[vector[uint32_t]] tup
    tup.resize(t - 1)
    for i in range(t - 1):
        tup[i].resize(2)

    cdef vector[uint8_t] sign
    cdef vector[vector[uint32_t]] primes

    mpz_fdiv_q_ui(border_up_z, g_z, prod(n_primes(t - 2, v[v.size() - 1] + 2))) 
    mpz_sqrt(border_up_z, border_up_z)
    cdef uint64_t border_up = mpz_get_ui(border_up_z)

    it = Iterator()
    it.skipto(X)
    cdef uint64_t p = it.next_prime()
    mpz_set_ui(p_z, p)

    ## Составление подходящих кортежей только на основе
    ## предварительно сохраненных сигнатур
    start_time = time.perf_counter()
    for cur_sign in sorted_primes_by_sign:
        if cur_sign.second.size() >= 2:
            mpz_set_ui(k_z, 1)
            rec_comp_feas_tup_more_t3(cur_sign.first, cur_sign.second, t - 2, cur_sign.second.size(), tup, k_z, 0)
    logging.info(f'spsp {t} | part 1: {time.perf_counter() - start_time:.4f} seconds')

    # Составление подходящих кортежей на основе 
    # предварительно сохраненных сигнатур и только что найденных
    start_time = time.perf_counter()
    while p < border_up:
        sign = find_signature(p_z, p)
        primes = sorted_primes_by_sign[sign]
        if primes.size() >= t - 2:
            find_lambda(lambda_z, p_z)
            tup[0][0] = p
            tup[0][1] = mpz_get_ui(lambda_z)
            rec_comp_feas_tup_more_t3_with_known_p_t_1(sign, primes, t - 2, primes.size(), tup, p_z, 0)
        p = it.next_prime()
        mpz_set_ui(p_z, p)  
    logging.info(f'spsp {t} | part 2: {time.perf_counter() - start_time:.4f} seconds')

    ### Очистка памяти
    mpz_clears(p_z, k_z, border_up_z, lambda_z, NULL)



# Функция для поиска spsp с количеством простых множителей = 3
cdef void find_spsp_with_three_factors():
    ### Инициализация 
    global sorted_primes_by_sign, v, g_z, X
    cdef mpz_t p_z, k_z, border_up_z, lambda_z
    mpz_inits(p_z, k_z, border_up_z, lambda_z, NULL)

    cdef vector[vector[uint32_t]] tup
    tup.resize(2)
    for i in range(2):
        tup[i].resize(2)

    cdef vector[uint8_t] sign
    cdef vector[vector[uint32_t]] primes

    mpz_fdiv_q_ui(border_up_z, g_z, n_primes(1, v[v.size() - 1] + 2)[0])
    mpz_sqrt(border_up_z, border_up_z)
    cdef uint64_t border_up = mpz_get_ui(border_up_z)

    it = Iterator()
    it.skipto(X)
    cdef uint64_t p = it.next_prime()
    mpz_set_ui(p_z, p)

    ### Составление подходящих кортежей только на основе
    ### предварительно сохраненных сигнатур
    start_time = time.perf_counter()
    for cur_sign in sorted_primes_by_sign:
        if cur_sign.second.size() >= 2:
            mpz_set_ui(k_z, 1)
            rec_comp_feas_tup_t3(cur_sign.first, cur_sign.second, 1, cur_sign.second.size(), tup, k_z, 0)
    logging.info(f'spsp 3 | part 1: {time.perf_counter() - start_time:.4f} seconds')

    ## Составление подходящих кортежей на основе 
    ## предварительно сохраненных сигнатур и только что найденных  
    start_time = time.perf_counter()
    while p < border_up:
        sign = find_signature(p_z, p)
        primes = sorted_primes_by_sign[sign]
        if primes.size() >= 1:
            find_lambda(lambda_z, p_z)
            tup[0][0] = p
            tup[0][1] = mpz_get_ui(lambda_z)
            rec_comp_feas_tup_t3_with_known_p_t_1(sign, primes, 1, primes.size(), tup, p_z, 0)
        p = it.next_prime()
        mpz_set_ui(p_z, p)  
    logging.info(f'spsp 3 | part 2: {time.perf_counter() - start_time:.4f} seconds')


    ### Очистка памяти
    mpz_clears(p_z, k_z, border_up_z, lambda_z, NULL)
    

# Функция для поиска spsp с количеством простых множителей = 2
cdef void find_spsp_with_two_factors(uint64_t up_border_p1):
    ### Инициализация
    global sorted_primes_by_sign, X
    cdef mpz_t p_z, r_z, lambda_z
    mpz_init_set_ui(r_z, 1)
    mpz_inits(p_z, lambda_z, NULL)
    it = Iterator()
    it.skipto(X)
    
    cdef uint64_t p = it.next_prime()

    ### Поиск с помощью метода GCD
    start_time = time.perf_counter()
    for sign in sorted_primes_by_sign:
        for i in range(sign.second.size()):
            mpz_set_ui(p_z, sign.second[i][0])
            search_pt_GCD(sign.first, p_z, p_z)
    logging.info(f'spsp 2 GCD : {time.perf_counter() - start_time:.4f} seconds')

    # Поиск с помощью лямбда-сигнатурного просеивания
    start_time = time.perf_counter()
    mpz_set_ui(p_z, p)

    while p < up_border_p1:
        find_lambda(lambda_z, p_z)
        lambda_signature_sieving(p_z, r_z, lambda_z, p_z)
        p = it.next_prime()
        mpz_set_ui(p_z, p)
    logging.info(f'spsp 2 Lambda and sign sieve: {time.perf_counter() - start_time:.4f} seconds')

    ### Очистка памяти
    mpz_clears(p_z, r_z, lambda_z, NULL)


# Основная функция
def sorenson(mpz g, uint8_t v_count, uint64_t coeff):
    start_time_all = time.perf_counter()

    global v_z, v, g_z, X, w_border_z, X_z
    global cur_spsp, leg_tup_one

    # Параметры для сохранения результатов
    spsp = defaultdict(list)
    dirname = f'result/g = {create_name(g)}, v={v_count}, coeff={coeff}'
    if not path.exists(dirname):
        makedirs(dirname)
    logging.basicConfig(filename=f'{dirname}/time.log',
                         filemode='w', 
                         level=logging.INFO,
                         format='%(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())

    ## Нулевой бинарный вектор
    for i in range(v_count):
        leg_tup_one.push_back(0)

    # Основания
    cdef array v_temp = n_primes(v_count)
    v_z = <mpz_t*>malloc(v_count * sizeof(mpz_t))
    for i in range(v_count):
        v.push_back(v_temp[i])
        mpz_init_set_ui(v_z[i], v_temp[i])

    # Граница поиска spsp
    mpz_init_set(g_z, MPZ(g))
    
    # Граница поиска сигнатур, работы метода GCD
    mpz_init(X_z)
    mpz_root(X_z, g_z, 3)
    X = mpz_get_ui(X_z)

    # w_border
    mpz_init(w_border_z)
    mpz_fdiv_q_ui(w_border_z, g_z, coeff)

    # Генерация коэффициентов для сравнений 
    cdef vector[uint8_t] residue_wh
    residue_wh.push_back(1)
    residue_wh.push_back(3)
    create_coeff_wheel(residue_wh, 4)

    # Предварительное вычисление сигнатур
    start_time = time.perf_counter()
    get_signatures() 
    logging.info(f'get_signatures : {time.perf_counter() - start_time:.4f} seconds')

    # Запуск поиска spsp для t > 3
    for t in range(6, 3, -1):
        cur_spsp = spsp[t]
        start_time = time.perf_counter()
        find_spsp_with_more_than_three_factors(t)
        logging.info(f'spsp {t} : {time.perf_counter() - start_time:.4f} seconds')

    # Запуск поиска spsp для t = 3
    cur_spsp = spsp[3]
    find_spsp_with_three_factors()

    # Запуск поиска spsp для t = 2
    start_time = time.perf_counter()
    cur_spsp = spsp[2]
    cdef mpz_t up_border_p1_z
    mpz_init(up_border_p1_z)
    mpz_sqrt(up_border_p1_z, g_z)
    cdef uint64_t up_border_p1 = mpz_get_ui(up_border_p1_z)
    find_spsp_with_two_factors(up_border_p1)
    logging.info(f'spsp 2 : {time.perf_counter() - start_time:.4f} seconds')

    # Сохранение результатов
    start_time = time.perf_counter()
    save_result(spsp, dirname)
    logging.info(f'save : {time.perf_counter() - start_time:.4f} seconds')

    # Освобождение памяти
    for i in range(v.size()):
        mpz_clear(v_z[i])
    free(v_z)
    mpz_clears(g_z, X_z, up_border_p1_z, w_border_z, NULL)

    logging.info(f'all : {time.perf_counter() - start_time_all:.4f} seconds')
