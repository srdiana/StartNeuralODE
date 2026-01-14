import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchdiffeq



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
T = 1.0  # Конечное время для ODE
SOLVER = 'dopri5'  # Решатель ODE (метод Дорманда-Принса 4/5)

class ODEFunc(nn.Module):
    """
    Нейронная сеть, определяющая правую часть ODE:
    dz/dt = f_theta(z(t), t)
    """
    def __init__(self, dim):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, dim),
        )
        
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)
    
    def forward(self, t, x):
        return self.net(x)


class ODEBlock(nn.Module):
    """
    Блок Neural ODE, который интегрирует ODE от 0 до T
    """
    def __init__(self, odefunc, T=1.0, solver='dopri5', rtol=1e-3, atol=1e-4):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.T = T
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        self.nfe = 0  # Счетчик вызовов функции (Number of Function Evaluations)
    
    def forward(self, x):
        self.nfe = 0
        
        def odefunc_wrapper(t, x):
            self.nfe += 1
            return self.odefunc(t, x)
        
        out = torchdiffeq.odeint(
            odefunc_wrapper,
            x,
            torch.tensor([0, self.T]).float().to(x.device),
            method=self.solver,
            rtol=self.rtol,
            atol=self.atol
        )
        
        return out[-1]  # Возвращаем состояние в момент T

class NeuralODEModel(nn.Module):
    """
    Полная модель для MNIST:
    1. Энкодер (сверточный) для извлечения признаков
    2. Neural ODE для непрерывной трансформации
    3. Классификатор
    """
    def __init__(self, ode_dim=64, T=1.0, solver='dopri5'):
        super(NeuralODEModel, self).__init__()
        
        # Энкодер: преобразует изображение 28x28 в вектор признаков
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, ode_dim),
            nn.Tanh()  # Ограничиваем значения для стабильности ODE
        )
        
        # Neural ODE блок
        self.odefunc = ODEFunc(ode_dim)
        self.odeblock = ODEBlock(self.odefunc, T=T, solver=solver)
        
        self.classifier = nn.Sequential(
            nn.Linear(ode_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 10)  # 10 классов для MNIST
        )
        
    def forward(self, x, return_trajectory=False):
       
        z0 = self.encoder(x)
        
        if return_trajectory:
            trajectory = self.get_trajectory(z0)
            zT = trajectory[-1]
        else:
            zT = self.odeblock(z0)
        
        logits = self.classifier(zT)
        
        if return_trajectory:
            return logits, trajectory
        return logits
    
    def get_trajectory(self, z0, n_points=20):
        """Получаем траекторию для визуализации"""
        t = torch.linspace(0, self.odeblock.T, n_points).to(z0.device)
        with torch.no_grad():
            trajectory = torchdiffeq.odeint(
                self.odefunc,
                z0,
                t,
                method=self.odeblock.solver,
                rtol=self.odeblock.rtol,
                atol=self.odeblock.atol
            )
        return trajectory
    
    def get_nfe(self):
        """Возвращает количество вызовов функции в ODE"""
        return self.odeblock.nfe

