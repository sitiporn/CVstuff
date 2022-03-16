from scipy import linalg
from scipy import linalg
import  numpy as np
#rng = np.random.default_rng()

a = np.array([[338.7, 6205.5, -3582.8, -9206.1],[0, 1796.3, -49050.0, 8103.8],[0, 6.4641, -3.73205, 1]])

a = a.reshape((3, 4)) 
m = a[:3,:3]
print("M:",m)
r, q = linalg.rq(m)
print("R:",r)
print("Q:",q)



