import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

dt = 0.1  # 시간 간격

def kalman_filter(z_meas, x_esti, P):
    """Kalman Filter Algorithm."""
    # (1) Prediction.
    x_pred = A @ x_esti
    P_pred = A @ P @ A.T + Q

    # (2) Kalman Gain.
    K = P_pred @ H.T @ inv(H @ P_pred @ H.T + R)

    # (3) Estimation.
    x_esti = x_pred + K @ (z_meas - H @ x_pred)

    # (4) Error Covariance.
    P = P_pred - K @ H @ P_pred

    return x_esti, P

# Matrix: A, H, Q, R, P_0
A = np.array([[1, dt],
              [0, 1]])
H = np.array([[0, 1]])
Q = np.array([[1, 0],
              [0, 3]])
R = np.array([[10]])

# Initialization for estimation.
x_0 = np.array([0, 20])  # 초기 위치와 속도
P_0 = 5 * np.eye(2)

# 사용자로부터 end_time과 각 시간의 좌표 입력받기
end_time = float(input("끝나는 시간 (초 단위): "))
time_steps = int(end_time / dt) + 1

positions = []
for i in range(time_steps):
    pos = tuple(map(float, input(f"{i * dt:.1f}초 뒤의 위치 x, y: ").split(',')))
    positions.append(pos)

# 칼만 필터 적용
x_esti, P = x_0, P_0
pos_true_save = np.zeros(time_steps)
pos_esti_save = np.zeros(time_steps)
vel_meas_save = np.zeros(time_steps)
vel_esti_save = np.zeros(time_steps)

for i in range(1, time_steps):
    # 속도 측정값 계산
    z_vel_meas = np.linalg.norm(np.array(positions[i]) - np.array(positions[i-1])) / dt
    x_esti, P = kalman_filter(z_vel_meas, x_esti, P)
    
    pos_true_save[i] = positions[i][0]
    pos_esti_save[i] = x_esti[0]
    vel_meas_save[i] = z_vel_meas
    vel_esti_save[i] = x_esti[1]

# 결과 출력
print(f"Estimated position: {x_esti[0]}")
print(f"Estimated velocity: {x_esti[1]}")

# 그래프 그리기
time = np.arange(0, end_time + dt, dt)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

axes[0].plot(time, vel_esti_save, 'bo-', label='Estimation (KF)')
axes[0].plot(time, vel_meas_save, 'r*--', label='Measurements', markersize=10)
axes[0].legend(loc='lower right')
axes[0].set_title('Velocity: Meas. v.s. Esti. (KF)')
axes[0].set_xlabel('Time [sec]')
axes[0].set_ylabel('Velocity [m/s]')

axes[1].plot(time, pos_esti_save, 'bo-', label='Estimation (KF)')
axes[1].plot(time, pos_true_save, 'g*--', label='True', markersize=10)
axes[1].legend(loc='upper left')
axes[1].set_title('Position: True v.s. Esti. (KF)')
axes[1].set_xlabel('Time [sec]')
axes[1].set_ylabel('Position [m]')

plt.tight_layout()
plt.show()
