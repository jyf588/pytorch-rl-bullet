import pybullet as p
import numpy as np

# run 20 tests
for test in range(2000):
    a1 = np.random.uniform(-2 * np.pi, 2 * np.pi)  # dummy dont care
    a2 = np.random.uniform(-2 * np.pi, 2 * np.pi)
    a3 = np.random.uniform(-2 * np.pi, 2 * np.pi)
    quat_test = p.getQuaternionFromEuler([a1, a2, a3])
    uz_2_solve = p.multiplyTransforms([0, 0, 0], quat_test, [0, 0, 1], [0, 0, 0, 1])[0]

    x, y, z = uz_2_solve
    a1_solved = np.arcsin(-y)
    a2_solved = np.arctan2(x, z)
    # a3_solved is zero since equation has under-determined
    quat_solved = p.getQuaternionFromEuler([a1_solved, a2_solved, 0])
    uz_check = p.multiplyTransforms([0, 0, 0], quat_solved, [0, 0, 1], [0, 0, 0, 1])[0]

    print(uz_2_solve, uz_check)
    assert np.linalg.norm(np.array(uz_2_solve) - np.array((uz_check))) < 1e-3
