from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from neural_ode import NeuralODEModel
def train_epoch(model, train_loader, criterion, optimizer, device):
    """Одна эпоха обучения"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    nfe_total = 0
    batch_count = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
       
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        nfe_total += model.get_nfe()
        batch_count += 1
        
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100.*correct/total:.2f}%',
            'NFE': model.get_nfe()
        })
    
    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total
    avg_nfe = nfe_total / batch_count
    
    return avg_loss, accuracy, avg_nfe


def test_epoch(model, test_loader, criterion, device):
    """Тестирование модели"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    nfe_total = 0
    batch_count = 0
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing")
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            
            outputs = model(data)
            loss = criterion(outputs, target)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            nfe_total += model.get_nfe()
            batch_count += 1
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
    
    avg_loss = running_loss / len(test_loader)
    accuracy = 100. * correct / total
    avg_nfe = nfe_total / batch_count
    
    return avg_loss, accuracy, avg_nfe


def train_full_model(SOLVER, LEARNING_RATE, EPOCHS, T, device, train_loader, test_loader):
    """Полный цикл обучения и тестирования"""
  
    model = NeuralODEModel(ode_dim=64, T=T, solver=SOLVER).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    history = {
        'train_loss': [], 'train_acc': [], 'train_nfe': [],
        'test_loss': [], 'test_acc': [], 'test_nfe': [],
        'learning_rate': []
    }
    
    print("=" * 60)
    print(f"Starting training for {EPOCHS} epochs")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("=" * 60)
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 60)
        
        train_loss, train_acc, train_nfe = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        test_loss, test_acc, test_nfe = test_epoch(
            model, test_loader, criterion, device
        )
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_nfe'].append(train_nfe)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['test_nfe'].append(test_nfe)
        history['learning_rate'].append(current_lr)
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, NFE: {train_nfe:.1f}")
        print(f"Test Loss:  {test_loss:.4f}, Test Acc:  {test_acc:.2f}%, NFE: {test_nfe:.1f}")
        print(f"Learning Rate: {current_lr:.6f}")
    
    return model, history