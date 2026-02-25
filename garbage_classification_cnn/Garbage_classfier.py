import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, precision_score, recall_score, f1_score, accuracy_score
import seaborn as sns
from PIL import Image
import os
import pandas as pd

ROOT_DIR = './garbage_classification'
BATCH_SIZE = 64
EPOCHS = 40
NUM_FOLDS = 10
SEED = 42

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data transformations
transform = transforms.Compose([
    transforms.RandomResizedCrop(196, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])

# Load
dataset = datasets.ImageFolder(root=ROOT_DIR, transform=transform)
num_samples = len(dataset)
num_classes = len(dataset.classes)

per_class_indices = [[] for _ in range(num_classes)]
for idx, label in enumerate(dataset.targets):
    per_class_indices[label].append(idx)
for indices in per_class_indices:
    random.shuffle(indices)

fold_indices = [[] for _ in range(NUM_FOLDS)]
for class_indices in per_class_indices:
    for i, sample_idx in enumerate(class_indices):
        fold_indices[i % NUM_FOLDS].append(sample_idx)

# CNN Model Definition
class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512 * 3 * 3, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        x = self.pool(F.relu(self.bn6(self.conv6(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)

if __name__ == '__main__':
    all_metrics = []
    results = [] 
    show_folds = [0, 8]
    fig, axs = plt.subplots(len(show_folds), 2, figsize=(15, 5 * len(show_folds)))
    if len(show_folds) == 1:
        axs = [axs] 
    for fold in range(NUM_FOLDS):
        save_dir = 'models'
        model_path = os.path.join(save_dir, f'model_fold{fold+1}.pth')
        loss_path = os.path.join(save_dir, f'loss_fold{fold+1}.pt')
        
        train_idx = []
        for i in range(NUM_FOLDS):
            if i != fold:
                train_idx.extend(fold_indices[i])
        val_idx = fold_indices[fold]
        
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False)
        criterion = nn.CrossEntropyLoss()
        if os.path.exists(model_path):
            print(f"Skipping Fold {fold+1}")
            model = CNN(num_classes).to(device)
            model.load_state_dict(torch.load(model_path))
            model.eval()
            
            running_loss = 0.0
            confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    running_loss += loss.item() * images.size(0)
                    preds = outputs.argmax(dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    for t, p in zip(labels.cpu(), preds.cpu()):
                        confusion[t, p] += 1
            val_loss = running_loss / len(val_loader.dataset)
            
            if os.path.exists(loss_path):
                losses = torch.load(loss_path)
                train_losses = losses.get('train_losses', [])
                val_losses = losses.get('val_losses', [val_loss])
            else:
                train_losses = []
                val_losses = [val_loss]
            
            fold_result = {
                "fold": fold,
                "train_losses": train_losses,
                "val_losses": val_losses,
                "confusion_matrix": confusion,
                "preds": all_preds,
                "labels": all_labels
            }
        else:
            train_loader = DataLoader(
                Subset(dataset, train_idx), 
                batch_size=BATCH_SIZE, 
                shuffle=True, 
                num_workers=6, 
                pin_memory=True
            )
            
            model = CNN(num_classes).to(device)
            optimizer = torch.optim.SGD(
                model.parameters(), 
                lr=1e-3, 
                momentum=0.9, 
                weight_decay=1e-4
            )
            best_val_loss = float('inf')
            train_losses = []
            val_losses = []
            start_time = time.time()
            
            for epoch in range(EPOCHS):
                # Train epoch
                model.train()
                running_loss = 0.0
                for images, labels in train_loader:
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item() * images.size(0)
                
                train_loss = running_loss / len(train_loader.dataset)
                train_losses.append(train_loss)
                
                model.eval()
                val_running_loss = 0.0
                confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)
                all_preds = []
                all_labels = []
                with torch.no_grad():
                    for images, labels in val_loader:
                        images, labels = images.to(device), labels.to(device)
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        val_running_loss += loss.item() * images.size(0)
                        preds = outputs.argmax(dim=1)
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
                        for t, p in zip(labels.cpu(), preds.cpu()):
                            confusion[t, p] += 1
                
                val_loss = val_running_loss / len(val_loader.dataset)
                val_losses.append(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    torch.save(model.state_dict(), model_path)
                    print(f"Saved best model for Fold {fold+1} with Val Loss = {val_loss:.4f}")

                torch.save({'train_losses': train_losses, 'val_losses': val_losses}, loss_path)
            
            fold_result = {
                "fold": fold,
                "train_losses": train_losses,
                "val_losses": val_losses,
                "confusion_matrix": confusion,
                "preds": all_preds,
                "labels": all_labels
            }
        
        results.append(fold_result)
        
        if fold in show_folds:
            idx = show_folds.index(fold)
            train_losses = fold_result['train_losses']
            val_losses = fold_result['val_losses']
            confusion_matrix = fold_result['confusion_matrix']
            preds = fold_result['preds']
            labels = fold_result['labels']
            
            tp = confusion_matrix.diag().float()
            precision = torch.where(confusion_matrix.sum(0) == 0, torch.zeros(num_classes), tp / confusion_matrix.sum(0).float())
            recall = torch.where(confusion_matrix.sum(1) == 0, torch.zeros(num_classes), tp / confusion_matrix.sum(1).float())
            f1 = torch.where((precision + recall) == 0, torch.zeros(num_classes), 2 * precision * recall / (precision + recall))
            acc = tp.sum().item() / confusion_matrix.sum().item()
            mean_prec, mean_rec, mean_f1 = precision.mean().item(), recall.mean().item(), f1.mean().item()
            mse = mean_squared_error(labels, preds)
            r2 = r2_score(labels, preds)
            
            axs[idx][0].plot(train_losses, marker='o', label='Train Loss')
            axs[idx][0].plot(val_losses, marker='o', label='Val Loss')
            axs[idx][0].set_title(f'Fold {fold+1}  Acc={acc:.3f}  F1={mean_f1:.3f}')
            axs[idx][0].set_ylabel('Loss')
            axs[idx][0].legend()
            
            sns.heatmap(confusion_matrix.numpy(), annot=True, fmt="d", cmap='Blues', 
                        square=True, ax=axs[idx][1], cbar=False)
            axs[idx][1].set_title(f"Confusion Matrix Fold {fold+1}")
            axs[idx][1].set_xlabel("Predicted")
            axs[idx][1].set_ylabel("True")
            
            metrics_text = (f"Acc={acc:.3f}\nPrec={mean_prec:.3f}\nRec={mean_rec:.3f}\n"
                            f"F1={mean_f1:.3f}\nMSE={mse:.3f}\nR2={r2:.3f}")
            axs[idx][1].text(1.05, 0.5, metrics_text, transform=axs[idx][1].transAxes, 
                            fontsize=10, verticalalignment='center',
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=1))
            
            all_metrics.append([acc, mean_prec, mean_rec, mean_f1, mse, r2])
    y
    all_metrics_arr = np.array(all_metrics)
    mean_metrics = all_metrics_arr.mean(axis=0)
    std_metrics = all_metrics_arr.std(axis=0)
    print(f'Mean:    Acc={mean_metrics[0]:.3f}  Prec={mean_metrics[1]:.3f}  Rec={mean_metrics[2]:.3f}  F1={mean_metrics[3]:.3f}  MSE={mean_metrics[4]:.3f}  R2={mean_metrics[5]:.3f}')
    print(f'Std:     Acc={std_metrics[0]:.3f}  Prec={std_metrics[1]:.3f}  Rec={std_metrics[2]:.3f}  F1={std_metrics[3]:.3f}  MSE={std_metrics[4]:.3f}  R2={std_metrics[5]:.3f}')
    
    plt.tight_layout()
    plt.show()
    
    
    # Ensemble evaluation
    print("ensemble")
    eval_transform = transforms.Compose([
        transforms.Resize((196, 196)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    
    image_paths = []
    labels = []
    for root, _, files in os.walk(ROOT_DIR):
        for f in files:
            if f.lower().endswith(('jpg', 'jpeg', 'png')):
                path = os.path.join(root, f)
                label = os.path.basename(os.path.dirname(path))
                image_paths.append(path)
                labels.append(label)
    
    models = []
    for i in range(1, NUM_FOLDS+1):
        model = CNN(num_classes).to(device)
        model.load_state_dict(torch.load(f'models/model_fold{i}.pth'))
        model.eval()
        models.append(model)
    
    preds = []
    true_indices = []
    class_to_idx = dataset.class_to_idx
    
    for path, label in tqdm(zip(image_paths, labels), total=len(image_paths)):
        image = Image.open(path).convert('RGB')
        input_tensor = eval_transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = [model(input_tensor) for model in models]
            votes = [torch.argmax(output, dim=1).item() for output in outputs]
            pred = max(set(votes), key=votes.count)
            preds.append(pred)
            true_indices.append(class_to_idx[label])
    
    acc = accuracy_score(true_indices, preds)
    prec = precision_score(true_indices, preds, average='macro', zero_division=0)
    rec = recall_score(true_indices, preds, average='macro', zero_division=0)
    f1 = f1_score(true_indices, preds, average='macro', zero_division=0)
    
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
