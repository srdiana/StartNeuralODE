import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def visualize_training_history(history, device, model, test_loader):
    """Визуализация истории обучения"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # График потерь
    axes[0, 0].plot(history['train_loss'], label='Train', linewidth=2)
    axes[0, 0].plot(history['test_loss'], label='Test', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Test Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # График точности
    axes[0, 1].plot(history['train_acc'], label='Train', linewidth=2)
    axes[0, 1].plot(history['test_acc'], label='Test', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Training and Test Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # График NFE
    axes[0, 2].plot(history['train_nfe'], label='Train NFE', linewidth=2, color='red')
    axes[0, 2].plot(history['test_nfe'], label='Test NFE', linewidth=2, color='blue')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('NFE per batch')
    axes[0, 2].set_title('Number of Function Evaluations')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Learning rate
    axes[1, 0].plot(history['learning_rate'], linewidth=2, color='green')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Confusion matrix (последняя эпоха)
    # Создаем confusion matrix для тестового набора
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    im = axes[1, 1].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    axes[1, 1].set_title('Confusion Matrix')
    plt.colorbar(im, ax=axes[1, 1])
    axes[1, 1].set_xlabel('Predicted')
    axes[1, 1].set_ylabel('True')
    
    # Пустой для следующей визуализации
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()


def visualize_ode_trajectories(model, test_loader, device, n_samples=100):
    """Визуализация траекторий ODE для разных классов"""
    model.eval()
    
    # Собираем данные
    trajectories = []
    labels = []
    images = []
    
    with torch.no_grad():
        count = 0
        for data, target in test_loader:
            if count >= n_samples:
                break
                
            data, target = data.to(device), target.to(device)
            
            # Получаем траекторию для каждого образца отдельно
            for i in range(data.shape[0]):
                if count >= n_samples:
                    break
                
                single_data = data[i:i+1]  # Берем один образец
                _, trajectory = model(single_data, return_trajectory=True)
                
                # trajectory shape: (n_time_points, batch_size=1, ode_dim)
                # Убираем batch dimension
                trajectory_np = trajectory.cpu().numpy().squeeze(1)  # (n_time_points, ode_dim)
                
                trajectories.append(trajectory_np)
                labels.append(target[i].cpu().numpy())
                images.append(data[i].cpu().numpy())
                
                count += 1
    
    # Визуализация 2D проекций с помощью PCA
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Выбираем несколько примеров для визуализации
    sample_indices = np.random.choice(len(trajectories), min(6, len(trajectories)), replace=False)
    
    for idx, sample_idx in enumerate(sample_indices):
        ax = axes[idx // 3, idx % 3]
        traj = trajectories[sample_idx]  # [time, features]
        label = labels[sample_idx]
        
        # Применяем PCA для снижения размерности до 2D
        pca = PCA(n_components=2)
        traj_2d = pca.fit_transform(traj)  # (n_time_points, 2)
        
        # Визуализируем траекторию
        scatter = ax.scatter(traj_2d[:, 0], traj_2d[:, 1], 
                           c=np.linspace(0, 1, len(traj_2d)), 
                           cmap='viridis', s=50, alpha=0.6)
        
        # Начальная и конечная точки
        ax.scatter(traj_2d[0, 0], traj_2d[0, 1], c='red', 
                  s=100, marker='o', label='t=0', edgecolors='black')
        ax.scatter(traj_2d[-1, 0], traj_2d[-1, 1], c='blue', 
                  s=100, marker='s', label=f't={model.odeblock.T}', edgecolors='black')
        
        # Соединяем точки линиями
        ax.plot(traj_2d[:, 0], traj_2d[:, 1], 'gray', alpha=0.3, linewidth=1)
        
        ax.set_title(f'Digit: {label}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('ODE Trajectories in PCA Space', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Визуализация изображений и их траекторий
    fig, axes = plt.subplots(2, min(5, len(trajectories)), figsize=(15, 6))
    
    for i in range(min(5, len(trajectories))):
        # Изображение
        ax_img = axes[0, i]
        img = images[i][0]  # Берем первый канал
        ax_img.imshow(img, cmap='gray')
        ax_img.set_title(f'Digit: {labels[i]}')
        ax_img.axis('off')
        
        # Траектория в PCA
        ax_traj = axes[1, i]
        traj = trajectories[i]
        pca = PCA(n_components=2)
        traj_2d = pca.fit_transform(traj)
        
        ax_traj.scatter(traj_2d[:, 0], traj_2d[:, 1], 
                       c=np.linspace(0, 1, len(traj_2d)), 
                       cmap='coolwarm', s=30, alpha=0.7)
        ax_traj.plot(traj_2d[:, 0], traj_2d[:, 1], 'k-', alpha=0.3)
        ax_traj.scatter(traj_2d[0, 0], traj_2d[0, 1], c='g', s=100, marker='*')
        ax_traj.scatter(traj_2d[-1, 0], traj_2d[-1, 1], c='r', s=100, marker='s')
        ax_traj.set_xlabel('PC1')
        ax_traj.set_ylabel('PC2')
        ax_traj.set_title('Trajectory')
        ax_traj.grid(True, alpha=0.3)
    
    plt.suptitle('Input Images and Their ODE Trajectories', fontsize=16)
    plt.tight_layout()
    plt.show()

def visualize_ode_dynamics(model, test_loader, device):
    """Визуализация динамики ODE для одного батча"""
    model.eval()
    
    # Берем один батч
    data, target = next(iter(test_loader))
    data, target = data.to(device), target.to(device)
    
    # Получаем траектории для всего батча
    with torch.no_grad():
        # Сначала получаем z0 для всего батча
        z0 = model.encoder(data)
        
        # Получаем траекторию для одного образца (для визуализации)
        # Берем первый образец из батча
        single_z0 = z0[0:1]
        trajectory = model.get_trajectory(single_z0, n_points=50)
        
        # Получаем z_T для всего батча (для классификации)
        z_T = model.odeblock(z0)
    
    # Вычисляем нормы состояний во времени для одного образца
    norms = torch.norm(trajectory.squeeze(1), dim=1)  # [time] - убираем batch dimension
    
    # Визуализация
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # График норм для разных классов во всем батче
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    time_points = np.linspace(0, model.odeblock.T, norms.size(0))
    
    # Для одного образца показываем норму во времени
    axes[0, 0].plot(time_points, norms.cpu().numpy(), 
                   color='blue', label=f'Digit {target[0].item()}', linewidth=2)
    
    axes[0, 0].set_xlabel('Time t')
    axes[0, 0].set_ylabel('Norm of z(t)')
    axes[0, 0].set_title(f'Evolution of State Norm for Digit {target[0].item()}')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Гистограмма конечных норм для всего батча
    final_norms = torch.norm(z_T, dim=1).cpu().numpy()
    axes[0, 1].hist(final_norms, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 1].set_xlabel('Final State Norm')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Distribution of Final State Norms')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Визуализация нескольких траекторий из батча
    sample_trajectories = []
    for i in range(min(5, data.shape[0])):
        single_z = z0[i:i+1]
        traj = model.get_trajectory(single_z, n_points=20)
        sample_trajectories.append(traj.cpu().numpy().squeeze(1))  # (n_points, ode_dim)
    
    for i in range(len(sample_trajectories)):
        # PCA для каждой траектории
        pca = PCA(n_components=2)
        traj_2d = pca.fit_transform(sample_trajectories[i])
        
        axes[1, 0].plot(traj_2d[:, 0], traj_2d[:, 1], 
                       linewidth=2, alpha=0.7, 
                       label=f'Digit {target[i].item()}')
        
        # Начальная и конечная точки
        axes[1, 0].scatter(traj_2d[0, 0], traj_2d[0, 1], 
                          s=100, marker='o', edgecolors='black')
        axes[1, 0].scatter(traj_2d[-1, 0], traj_2d[-1, 1], 
                          s=100, marker='s', edgecolors='black')
    
    axes[1, 0].set_xlabel('PC1')
    axes[1, 0].set_ylabel('PC2')
    axes[1, 0].set_title('Sample Trajectories in PCA Space')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Тепловая карта корреляций между измерениями (для конечных состояний)
    final_states = z_T.cpu().numpy()  # (batch_size, ode_dim)
    corr_matrix = np.corrcoef(final_states.T)  # (ode_dim, ode_dim)
    
    im = axes[1, 1].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    axes[1, 1].set_title('Correlation Matrix of Final States')
    axes[1, 1].set_xlabel('Dimension')
    axes[1, 1].set_ylabel('Dimension')
    plt.colorbar(im, ax=axes[1, 1])
    
    plt.suptitle('ODE Dynamics Analysis', fontsize=16)
    plt.tight_layout()
    plt.show()


def visualize_ode_dynamics(model, test_loader, device, T):
    """Визуализация динамики ODE для одного батча"""
    model.eval()
    
    data, target = next(iter(test_loader))
    data, target = data.to(device), target.to(device)
    
    with torch.no_grad():
        z0 = model.encoder(data)
        trajectory = model.get_trajectory(z0, n_points=50)
    
    # Вычисляем нормы состояний во времени
    norms = torch.norm(trajectory, dim=2)  # [time, batch]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    time_points = np.linspace(0, T, norms.size(0))
    
    for digit in range(10):
        mask = (target == digit).cpu().numpy()
        if mask.any():
            digit_norms = norms[:, mask].mean(dim=1).cpu().numpy()
            axes[0, 0].plot(time_points, digit_norms, 
                           color=colors[digit], label=f'Digit {digit}', linewidth=2)
    
    axes[0, 0].set_xlabel('Time t')
    axes[0, 0].set_ylabel('Norm of z(t)')
    axes[0, 0].set_title('Evolution of State Norm by Digit')
    axes[0, 0].legend(loc='upper right', fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)
    
    final_norms = norms[-1].cpu().numpy()
    axes[0, 1].hist(final_norms, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 1].set_xlabel('Final State Norm')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Distribution of Final State Norms')
    axes[0, 1].grid(True, alpha=0.3)
    
    sample_trajectories = trajectory[:, :5].cpu().numpy() 
    for i in range(5):
        pca = PCA(n_components=2)
        traj_2d = pca.fit_transform(sample_trajectories[:, i, :])
        
        axes[1, 0].plot(traj_2d[:, 0], traj_2d[:, 1], 
                       linewidth=2, alpha=0.7, 
                       label=f'Digit {target[i].item()}')
        
        axes[1, 0].scatter(traj_2d[0, 0], traj_2d[0, 1], 
                          s=100, marker='o', edgecolors='black')
        axes[1, 0].scatter(traj_2d[-1, 0], traj_2d[-1, 1], 
                          s=100, marker='s', edgecolors='black')
    
    axes[1, 0].set_xlabel('PC1')
    axes[1, 0].set_ylabel('PC2')
    axes[1, 0].set_title('Sample Trajectories in PCA Space')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    final_states = trajectory[-1].cpu().numpy()
    corr_matrix = np.corrcoef(final_states.T)
    
    im = axes[1, 1].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    axes[1, 1].set_title('Correlation Matrix of Final States')
    axes[1, 1].set_xlabel('Dimension')
    axes[1, 1].set_ylabel('Dimension')
    plt.colorbar(im, ax=axes[1, 1])
    
    plt.suptitle('ODE Dynamics Analysis', fontsize=16)
    plt.tight_layout()
    plt.show()


def visualize_predictions(model, test_loader, device, n_examples=10):
    """Визуализация предсказаний модели"""
    model.eval()
    
    data, target = next(iter(test_loader))
    data, target = data[:n_examples], target[:n_examples]
    data, target = data.to(device), target.to(device)
    
    with torch.no_grad():
        outputs, trajectories = model(data, return_trajectory=True)
        probs = torch.softmax(outputs, dim=1)
        _, preds = outputs.max(1)
    
    fig, axes = plt.subplots(2, n_examples, figsize=(n_examples*2, 4))
    
    for i in range(n_examples):
       
        ax_img = axes[0, i]
        img = data[i].cpu().numpy()[0]  # Берем первый канал
        ax_img.imshow(img, cmap='gray')
        
        # Цвет заголовка: зеленый если правильно, красный если нет
        color = 'green' if preds[i] == target[i] else 'red'
        ax_img.set_title(f'True: {target[i].item()}\nPred: {preds[i].item()}', 
                        color=color, fontsize=10)
        ax_img.axis('off')
        
        # График вероятностей
        ax_prob = axes[1, i]
        probs_i = probs[i].cpu().numpy()
        bars = ax_prob.bar(range(10), probs_i, color=plt.cm.tab10(range(10)))
        
        # Подсвечиваем предсказанный класс
        bars[preds[i]].set_edgecolor('red')
        bars[preds[i]].set_linewidth(3)
        
        # Подсвечиваем истинный класс
        bars[target[i]].set_alpha(0.7)
        bars[target[i]].set_edgecolor('green')
        bars[target[i]].set_linewidth(2)
        
        ax_prob.set_ylim([0, 1])
        ax_prob.set_xticks(range(10))
        ax_prob.set_xlabel('Digit')
        if i == 0:
            ax_prob.set_ylabel('Probability')
    
    plt.suptitle('Model Predictions', fontsize=16)
    plt.tight_layout()
    plt.show()



def visualize_latent_space(model, test_loader, device, n_samples=1000):
    """Визуализация латентного пространства до и после ODE"""
    model.eval()
    
    z0_list = []
    zT_list = []
    labels_list = []
    
    with torch.no_grad():
        count = 0
        for data, target in test_loader:
            if count >= n_samples:
                break
            
            data = data.to(device)
            batch_size = data.size(0)
            
            # Получаем z0 (перед ODE)
            z0 = model.encoder(data)
            
            # Получаем zT (после ODE)
            zT = model.odeblock(z0)
            
            z0_list.append(z0.cpu().numpy())
            zT_list.append(zT.cpu().numpy())
            labels_list.append(target.cpu().numpy())
            
            count += batch_size
    
    z0_all = np.vstack(z0_list)[:n_samples]
    zT_all = np.vstack(zT_list)[:n_samples]
    labels_all = np.hstack(labels_list)[:n_samples]
    
    # Применяем t-SNE для визуализации
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    for idx, (z, title) in enumerate([(z0_all, 'Before ODE (z0)'), 
                                       (zT_all, 'After ODE (zT)')]):
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        z_tsne = tsne.fit_transform(z)
        
        ax = axes[idx]
        scatter = ax.scatter(z_tsne[:, 0], z_tsne[:, 1], 
                           c=labels_all, cmap='tab10', 
                           s=20, alpha=0.6, edgecolors='none')
        
        ax.set_title(f'{title} - t-SNE')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.grid(True, alpha=0.3)
        
        # Легенда
        legend1 = ax.legend(*scatter.legend_elements(), 
                           title="Digits", loc="upper right")
        ax.add_artist(legend1)
    
    plt.suptitle('Latent Space Visualization with t-SNE', fontsize=16)
    plt.tight_layout()
    plt.show()