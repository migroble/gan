import numpy as np

STEPS = 64
INPUT_DIM = 150
TYPE_DIM = 5

A_p = np.random.uniform(-1, 1, size=INPUT_DIM).astype(np.float32)
B_p = np.random.uniform(-1, 1, size=INPUT_DIM).astype(np.float32)

A_t = np.random.uniform(0, 1, size=TYPE_DIM).astype(np.float32)
B_t = np.random.uniform(0, 1, size=TYPE_DIM).astype(np.float32)

dist_p = np.linalg.norm(A_p - B_p)
dist_t = np.linalg.norm(A_t - B_t)

vector_p = (B_p - A_p) / dist_p
vector_t = (B_t - A_t) / dist_t

step_dist_p = dist_p / STEPS
step_dist_t = dist_t / STEPS

pos = np.zeros((STEPS, INPUT_DIM))
types = np.zeros((STEPS, TYPE_DIM))
for i in range(STEPS):
	pos[i] = A_p + vector_p * step_dist_p * i
	types[i] = A_t + vector_t * step_dist_t * i

pos_arr = pos.tolist()
types_arr = types.tolist()

f = open("types_lerp.txt", "w+")
f.write("{\"types\": " + str(types_arr) + "}")
f.close()

f = open("pos_lerp.txt", "w+")
f.write("{\"positions\": " + str(pos_arr) + "}")
f.close()