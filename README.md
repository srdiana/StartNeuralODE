# Example of using NeuralODE for MNIST dataset

**Нейронные обыкновенные дифференциальные уравнения (Neural ODE) для классификации рукописных цифр MNIST с использованием adjoint метода и полной визуализации процесса обучения.**

## Обзор

Этот проект реализует Neural ODE — революционный подход к машинному обучению, который моделирует непрерывную динамику данных с помощью обыкновенных дифференциальных уравнений (ОДУ). Вместо дискретных слоев нейросети мы обучаем ОДУ, где производная определяется нейронной сетью.

### Структура модели

```
NeuralODEModel(
  (encoder): Sequential(
    (0): Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (3): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): ReLU()
    (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (6): Flatten(start_dim=1, end_dim=-1)
    (7): Linear(in_features=1568, out_features=128, bias=True)
    (8): ReLU()
    (9): Linear(in_features=128, out_features=64, bias=True)
    (10): Tanh()
  )
  (odefunc): ODEFunc(
    (net): Sequential(
      (0): Linear(in_features=64, out_features=128, bias=True)
      (1): ReLU()
      (2): Linear(in_features=128, out_features=128, bias=True)
      (3): ReLU()
      (4): Linear(in_features=128, out_features=128, bias=True)
      (5): ReLU()
      (6): Linear(in_features=128, out_features=64, bias=True)
    )
  )
  (odeblock): ODEBlock(
    (odefunc): ODEFunc(
      (net): Sequential(
        (0): Linear(in_features=64, out_features=128, bias=True)
        (1): ReLU()
        (2): Linear(in_features=128, out_features=128, bias=True)
        (3): ReLU()
        (4): Linear(in_features=128, out_features=128, bias=True)
        (5): ReLU()
        (6): Linear(in_features=128, out_features=64, bias=True)
      )
    )
  )
  (classifier): Sequential(
    (0): Linear(in_features=64, out_features=64, bias=True)
    (1): ReLU()
    (2): Linear(in_features=64, out_features=10, bias=True)
  )
)
```


<!-- 
## 📈 Визуализации

Проект включает 6 типов визуализаций: -->
<!-- 
### 1. История обучения
![Training History](docs/images/training_history.png)
*Loss, accuracy и NFE (Number of Function Evaluations) по эпохам*

### 2. Траектории ODE в PCA пространстве
![ODE Trajectories](docs/images/ode_trajectories.png)
*Непрерывные траектории состояний для разных цифр*

### 3. Латентное пространство (t-SNE)
![Latent Space](docs/images/latent_space.png)
*Сравнение распределений до и после ODE* -->
<!-- 
### 4. Динамика ODE
![ODE Dynamics](docs/images/ode_dynamics.png)
*Эволюция норм состояний и корреляционная матрица*

### 5. Confusion Matrix
![Confusion Matrix](docs/images/confusion_matrix.png)
*Матрица ошибок классификации*

### 6. Предсказания модели
![Predictions](docs/images/predictions.png)
*Визуализация предсказаний с вероятностями* -->


```python
BATCH_SIZE = 64          # Размер батча
EPOCHS = 10              # Количество эпох
LEARNING_RATE = 0.001    # Скорость обучения
T = 1.0                  # Конечное время интегрирования ODE
ODE_DIM = 64             # Размерность латентного пространства
SOLVER = 'dopri5'        # Решатель ODE (dopri5/rk4/euler/midpoint)
RTOL = 1e-3              # Относительная погрешность
ATOL = 1e-4              # Абсолютная погрешность
```

### Доступные решатели ODE

| Решатель | Точность | Скорость | Стабильность |
|----------|----------|----------|--------------|
| `dopri5` | Высокая | Медленная | Высокая |
| `rk4` | Средняя | Средняя | Средняя |
| `euler` | Низкая | Быстрая | Низкая |
| `midpoint` | Средняя | Средняя | Средняя |

## 🧠 Математические детали

### Neural ODE Формализация

**Прямой проход**:
```
z(t₀) = Encoder(x)
dz/dt = f_θ(z(t), t)  для t ∈ [t₀, T]
ŷ = Classifier(z(T))
```

**Обратный проход (Adjoint метод)**:
```
a(t) = ∂L/∂z(t)  # adjoint состояние
da/dt = -a(t)ᵀ ∂f_θ/∂z
dL/dθ = -∫ₜ₀ᵀ a(t)ᵀ ∂f_θ/∂θ dt
```

1. **Neural Ordinary Differential Equations** (NeurIPS 2018)
   - Авторы: Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, David Duvenaud
   - [arXiv:1806.07366](https://arxiv.org/abs/1806.07366)

2. **FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models** (ICLR 2019)
   - [arXiv:1810.01367](https://arxiv.org/abs/1810.01367)

3. **Latent ODEs for Irregularly-Sampled Time Series** (NeurIPS 2019)
   - [arXiv:1907.03907](https://arxiv.org/abs/1907.03907)

### Ключевые концепции

- **Adjoint Sensitivity Method**: Эффективное вычисление градиентов через ODE
- **Continuous Normalizing Flows**: Нормализующие потоки как ODE
- **Neural Differential Equations**: Обобщение на SDE и PDE
