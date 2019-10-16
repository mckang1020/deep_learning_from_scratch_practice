import numpy as np


# def mean_squared_error(y, t):
#     return 0.5 * np.sum((y - t) ** 2)
#
#
# y1 = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
# y2 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
# t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  # 정답이 2인 경우
# # 리스트인 값을 행렬계산을 위해 array로 변환
# print(mean_squared_error(np.array(y1), np.array(t)))  # 0.0975
# print(mean_squared_error(np.array(y2), np.array(t)))  # 0.5975
# # 정답을 맞춘 신경망 출력값 y1이 더 작은 오차


def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t*np.log(y+delta))


t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  # 정답이 2인 경우
y1 = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
y2 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]

print(cross_entropy_error(np.array(y1), np.array(t)))  # 0.5108
print(cross_entropy_error(np.array(y2), np.array(t)))  # 2.3025

# 정답을 맞춘 신경망 출력값 y1이 더 작은 오차



