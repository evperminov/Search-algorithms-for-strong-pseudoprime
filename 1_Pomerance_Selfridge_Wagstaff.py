# Подключение библиотек
from collections import defaultdict
from supportive_methods import *
from primesieve import n_primes
from os import path, makedirs
import logging
import time


# Рекурсивная функция, генерирующая все возможные подходящие составные числа и проверяющие их на spsp
def recursive_enumeration(start, deep, lambd=1, k=1):
    global cur_spsp, g, v

    if deep > 1:
        for p in get_prime(start, gm.rootn(g / k, deep)):
            cur_lambd = gm.lcm(*sum(find_ord([p], v), []), lambd)
            recursive_enumeration(p, deep - 1, cur_lambd, k * p)
    else:
        for p in get_prime(start, g / k):
            output = k * p
            if output % lambd == 1 and miller_rabin(v, output):
                cur_spsp.append(output)  


# Основная функция
def main():
    start_time_all = time.perf_counter()  

    global v, g, cur_spsp

    # Параметры поиска
    v = list(n_primes(4))
    g = 32150_31751
   
    # Параметры для сохранения результатов
    spsp = defaultdict(list)
    dirname = f'result/{path.basename(__file__).split(".")[0]}/g = {create_name(g)}, v={v}'
    if not path.exists(dirname):
        makedirs(dirname)
    logging.basicConfig(filename=f'{dirname}/time.log',
                         filemode='w', 
                         level=logging.INFO,
                         format='%(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())

    # Запуск поиска spsp для t >= 2
    start_time = time.perf_counter()
    for t in range(6, 1, -1):
        start_time = time.perf_counter()
        cur_spsp = spsp[t]
        recursive_enumeration(v[-1], t)
        logging.info(f'spsp {t} : {time.perf_counter() - start_time:.4f} seconds')

    # Сохранение результатов
    start_time = time.perf_counter()
    save_result(spsp, dirname)
    logging.info(f'save : {time.perf_counter() - start_time:.4f} seconds')

    logging.info(f'all : {time.perf_counter() - start_time_all:.4f} seconds')
    

main()