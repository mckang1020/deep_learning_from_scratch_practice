import numpy as np


def numerical_diff(f, x):
    h = 1e-4 # 0.0001
    return (f(x+h) - f(x-h)) / (2*h)

def function_2(x):
    return x[0]**2 + x[1]**2

def function_tmp1(x0):
    return x0**x0 + 4.0**2.0  # x1=4로 고정된 새로운 함수 생성


print(numerical_diff(function_tmp1, 3.0))  # x1이 상수로 고정된 상황에서 x0=3일때의 미분 값

