import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class PowerCouplingSystem:
    def __init__(self, num_outer_cores, h0, h_levels):
        """
        初始化耦合系统。
        
        参数:
            num_outer_cores (int): 外围光纤芯的数量。
            h0 (float): 中心光纤芯与周围光纤芯的耦合系数。
            h_levels (list): 周围光纤芯之间的耦合系数列表，按距离级别排序。
        """
        self.num_outer_cores = num_outer_cores
        self.num_cores = num_outer_cores + 1
        self.h0 = h0
        self.h_levels = h_levels
        self.K = self.generate_coupling_matrix()

    def generate_coupling_matrix(self):
        """
        生成耦合系数矩阵。
        
        返回:
            numpy.ndarray: (num_outer_cores + 1)x(num_outer_cores + 1)的耦合系数矩阵。
        """
        K = np.zeros((self.num_cores, self.num_cores))
        
        # 中心光纤芯与周围光纤芯的耦合
        center = 0
        for i in range(1, self.num_outer_cores + 1):
            K[center][i] = self.h0
            K[i][center] = self.h0
        
        # 周围光纤芯之间的耦合
        for i in range(1, self.num_outer_cores + 1):
            for j in range(i + 1, self.num_outer_cores + 1):
                distance = min(abs(i - j), self.num_outer_cores - abs(i - j) + 1)
                if distance <= len(self.h_levels):
                    h = self.h_levels[distance - 1]
                    K[i][j] = h
                    K[j][i] = h
        
        return K

    def power_coupling_eq(self, z, P):
        """
        功率耦合方程。
        
        参数:
            z (float): 传播距离。
            P (numpy.ndarray): 各光纤芯的功率数组。
        
        返回:
            numpy.ndarray: 各光纤芯功率的变化率。
        """
        dP_dz = np.zeros_like(P)
        for i in range(len(P)):
            for j in range(len(P)):
                if i != j:
                    dP_dz[i] += self.K[i, j] * (P[j] - P[i])
        return dP_dz

    def solve(self, z_span, P0):
        """
        求解功率耦合方程。
        
        参数:
            z_span (tuple): 传播距离范围 (z0, zf)。
            P0 (numpy.ndarray): 初始功率分布。
        
        返回:
            tuple: 传播距离数组 z 和各光纤芯的功率数组 P。
        """
        sol = solve_ivp(self.power_coupling_eq, z_span, P0, t_eval=np.linspace(z_span[0], z_span[1], 500))
        return sol.t, sol.y

def main():
    # 参数设置
    num_outer_cores = 6  # 周围光纤芯的数量
    h0 = 0.05  # 中心光纤芯与周围光纤芯的耦合系数
    h_levels = [0.03, 0.02, 0.01]  # 周围光纤芯之间的耦合系数列表，按距离级别排序
    
    # 创建功率耦合系统实例
    system = PowerCouplingSystem(num_outer_cores, h0, h_levels)
    
    # 初始条件
    P0 = np.ones(system.num_cores)
    P0[0] = 0.8  # 假设仅中心光纤芯的初始功率为1
    
    # 定义传播距离范围
    z_span = (0, 10e1)
    
    # 求解功率耦合方程
    z, P = system.solve(z_span, P0)
    
    # 绘制结果
    plt.figure(figsize=(10, 6))
    for i in range(system.num_cores):
        plt.plot(z, P[i], label=f'Power in Core {i}')
    plt.xlabel('Propagation Distance (z)')
    plt.ylabel('Power')
    plt.title(f'Power Coupling in {system.num_cores}-Core Fiber')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()