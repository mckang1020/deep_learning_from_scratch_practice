import numpy as np


def function_2(x):
    return x[0]**2 + x[1]**2


def numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)  # x와 형상이 같은 배열을 생성

    for idx in range(x.size):
        tmp_val = x[idx]

        x[idx] = tmp_val + h
        fhx1 = f(x)
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fhx1 - fxh2) / (2*h)
        x[idx] = tmp_val  # 값 복원

    return grad


print(numerical_gradient(function_2, np.array([3.0, 4.0])))



