import gmpy2  as gm
from Sorenson_Webster_cython import *

g = 318665857834031151167461 + 1
v_count = 12
coeff = 1000

sorenson(gm.mpz(g), v_count, coeff)
 
