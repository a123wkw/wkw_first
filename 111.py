#!/usr/bin/env python3
"""
量子-经典混合计算系统模拟器

功能：
1. 使用量子电路模拟进行特征提取
2. 结合经典机器学习模型进行预测
3. 实现量子梯度下降优化
4. 可视化整个计算流程

技术栈：
- Qiskit (量子计算)
- PyTorch (深度学习)
- NumPy (数值计算)
- Matplotlib (可视化)
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter
from qiskit.visualization import plot_histogram
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class QuantumFeatureExtractor:
    """量子特征提取器"""
    def __init__(self, n_qubits=4):
        self.n_qubits = n_qubits
        self.circuit = QuantumCircuit(n_qubits)
        self.theta = Parameter('θ')
        
        # 构建参数化量子电路
        for i in range(n_qubits):
            self.circuit.h(i)
            self.circuit.ry(self.theta, i)
        
        # 添加纠缠门
        for i in range(n_qubits-1):
            self.circuit.cx(i, i+1)
    
    def run(self, inputs, shots=1024):
        """执行量子电路"""
        backend = Aer.get_backend('qasm_simulator')
        results = []
        
        for x in inputs:
            # 绑定参数值 (将经典数据映射到量子空间)
            param_binds = {self.theta: 2 * np.pi * x}
            qc = self.circuit.bind_parameters(param_binds)
            
            # 测量所有量子位
            qc.measure_all()
            
            # 执行模拟
            job = execute(qc, backend, shots=shots)
            counts = job.result().get_counts()
            
            # 将测量结果转换为特征向量
            feature_vec = np.zeros(2**self.n_qubits)
            for k, v in counts.items():
                feature_vec[int(k, 2)] = v/shots
            results.append(feature_vec)
        
        return np.array(results)

class HybridModel(nn.Module):
    """量子-经典混合模型"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.quantum = QuantumFeatureExtractor()
        self.classical = nn.Sequential(
            nn.Linear(2**self.quantum.n_qubits, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # 量子特征提取
        with torch.no_grad():
            x_quantum = self.quantum.run(x.numpy())
            x_quantum = torch.tensor(x_quantum, dtype=torch.float32)
        
        # 经典神经网络处理
        return self.classical(x_quantum)

def quantum_gradient_descent(model, X, y, epochs=100, lr=0.01):
    """量子梯度下降优化器"""
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.classical.parameters(), lr=lr)
    
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
    
    return losses

def visualize_results(X, y, preds, losses):
    """可视化结果"""
    plt.figure(figsize=(18, 5))
    
    # 原始数据分布
    plt.subplot(1, 3, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm')
    plt.title("Original Data Distribution")
    
    # 预测结果
    plt.subplot(1, 3, 2)
    plt.scatter(X[:, 0], X[:, 1], c=preds, cmap='coolwarm')
    plt.title("Model Predictions")
    
    # 损失曲线
    plt.subplot(1, 3, 3)
    plt.plot(losses)
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    
    plt.tight_layout()
    plt.show()

def main():
    # 1. 准备数据
    X, y = make_moons(n_samples=500, noise=0.1, random_state=42)
    X = StandardScaler().fit_transform(X)
    y = y.reshape(-1, 1).astype(np.float32)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 转换为PyTorch张量
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    
    # 2. 初始化混合模型
    model = HybridModel(input_dim=2, hidden_dim=8)
    
    # 3. 训练模型
    losses = quantum_gradient_descent(model, X_train_t, y_train_t)
    
    # 4. 评估模型
    with torch.no_grad():
        test_preds = model(torch.tensor(X_test, dtype=torch.float32)).numpy()
        accuracy = np.mean((test_preds > 0.5) == y_test)
        print(f"\nTest Accuracy: {accuracy*100:.2f}%")
    
    # 5. 可视化
    with torch.no_grad():
        all_preds = model(torch.tensor(X, dtype=torch.float32)).numpy()
    visualize_results(X, y, all_preds, losses)

if __name__ == "__main__":
    main()
