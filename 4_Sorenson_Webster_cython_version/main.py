import gmpy2  as gm
from Sorenson_Webster_cython import *

g = 3186_65857_83403_11511_67461 + 1
v_count = 12
coeff = 1000

sorenson(gm.mpz(g), v_count, coeff)
 
