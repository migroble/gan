import numpy as np

mat = np.random.uniform(-1, 1, size=[64, 150]).astype(np.float32)
pos = mat.tolist()

types = np.random.uniform(0, 1, size=[64, 5]).astype(np.float32)
types_arr = types.tolist()
"""
types = []

for i in range(5):
	a = [ 0, 0, 0, 0, 0 ]
	a[i] = 1
	types.append(a)
	types.append(a)
	types.append(a)
	types.append(a)
	types.append(a)
"""
f = open("types.txt", "w+")
f.write("{\"types\": " + str(types_arr) + "}")
f.close()

f = open("pos.txt", "w+")
f.write("{\"positions\": " + str(pos) + "}")
f.close()