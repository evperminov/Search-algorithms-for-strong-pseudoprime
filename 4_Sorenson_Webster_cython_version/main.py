import gmpy2  as gm
from Sorenson_Webster_cython import *

g = 10 ** 15
v_count = 5
coeff = 1

sorenson(gm.mpz(g), v_count, coeff)
 
