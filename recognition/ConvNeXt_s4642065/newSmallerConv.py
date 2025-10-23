#smaller conv to see it it can learn better

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time as time
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
import os
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from timm.layers import trunc_normal_, DropPath

from torch.cuda.amp import autocast, GradScaler
import math



YOUR_MEAN = 0.1156
YOUR_STD = 0.2198

# og transforms
#train_transform = transforms.Compose([
#    transforms.Grayscale(num_output_channels=1),  
#    transforms.Resize((224,224)),
#    transforms.RandomHorizontalFlip(p=0.5),
#    transforms.RandomRotation(8),
#    transforms.RandomAffine(degrees=0, translate=(0.05,0.05), scale=(0.95,1.05)),
#    transforms.ToTensor(),
#    transforms.Normalize(mean=[YOUR_MEAN], std=[YOUR_STD])
#])

# new transforms
# Og 100 epoch: 75.28
# Change -> add colorjitter, increase rotation to 10 degrees



# new drop out in model forward at 0.2


train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  
    transforms.ColorJitter(brightness=0.17, contrast=0.17),
    #transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(14),
    transforms.RandomAffine(degrees=0, translate=(0.08,0.08), scale=(0.9,1.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[YOUR_MEAN], std=[YOUR_STD]),
    #transforms.Lambda(lambda x: x + 0.01 * torch.randn_like(x))  # light Gaussian noise
])


def build_transforms(level):
    """Return a transform pipeline of different randomization strength."""
    if level == "weak":
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ColorJitter(brightness=0.05, contrast=0.05),
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(6),
            transforms.RandomAffine(degrees=0, translate=(0.03,0.03), scale=(0.97,1.03)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[YOUR_MEAN], std=[YOUR_STD])
        ])
    elif level == "medium":
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ColorJitter(brightness=0.10, contrast=0.10),
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.05,0.05), scale=(0.95,1.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[YOUR_MEAN], std=[YOUR_STD])
        ])
    else:  # "strong"
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ColorJitter(brightness=0.17, contrast=0.17),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(14),
            transforms.RandomAffine(degrees=0, translate=(0.08,0.08), scale=(0.9,1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[YOUR_MEAN], std=[YOUR_STD])
        ])

test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1), 
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[YOUR_MEAN], std=[YOUR_STD])
])

def make_loaders(aug_level):
    train_t = build_transforms(aug_level)
    test_t  = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[YOUR_MEAN], std=[YOUR_STD])
    ])
    train_ds = datasets.ImageFolder("/home/groups/comp3710/ADNI/AD_NC/train", transform=train_t)
    test_ds  = datasets.ImageFolder("/home/groups/comp3710/ADNI/AD_NC/test",  transform=test_t)
    train_ld = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=1)
    test_ld  = DataLoader(test_ds,  batch_size=16, shuffle=False, num_workers=1)
    return train_ld, test_ld


# Datasets
train_dataset = datasets.ImageFolder(root="/home/groups/comp3710/ADNI/AD_NC/train", transform=train_transform)
test_dataset  = datasets.ImageFolder(root="/home/groups/comp3710/ADNI/AD_NC/test",  transform=test_transform)

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=1)
test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=1)

print(f"Train length: {len(train_dataset)}, Test length: {len(test_dataset)}")
print(train_dataset[0][0].mean(), train_dataset[0][0].std())


class SmallBlock(nn.Module):
    def __init__(self, dim, layer_scale_init_value=1e-5, drop_path_rate=0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pw1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pw2 = nn.Linear(4 * dim, dim)

        # ✅ Stochastic depth (DropPath)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

        if layer_scale_init_value > 0:
            self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
        else:
            self.gamma = None

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)             # N,C,H,W
        x = x.permute(0, 2, 3, 1)      # N,H,W,C
        x = self.norm(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)      # N,C,H,W
        return shortcut + self.drop_path(x)

class MiniConvNeXt(nn.Module):
    def __init__(self, in_chans=1, num_classes=2,
                 depths=(1, 1, 2, 2),
                 dims=(32, 64, 128, 256),
                 layer_scale_init_value=1e-5,
                 drop_path_rate=0.01): 
        super().__init__()
        assert len(depths) == 4 and len(dims) == 4

        self.dropout = nn.Dropout(p=0.25)
        # stem
        self.downsamples = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            # Use LayerNorm over channels_first by wrapping below in forward_features
        )
        self.downsamples.append(stem)

        for i in range(3):
            self.downsamples.append(
                nn.Sequential(
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2)
                )
            )

       # ✅ distribute drop path rate linearly across all blocks
        total_blocks = sum(depths)
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

        # build stages
        self.stages = nn.ModuleList()
        block_idx = 0
        for i in range(4):
            blocks = []
            for _ in range(depths[i]):
                blocks.append(
                    SmallBlock(
                        dims[i],
                        layer_scale_init_value=layer_scale_init_value,
                        drop_path_rate=dp_rates[block_idx],
                    )
                )
                block_idx += 1
            self.stages.append(nn.Sequential(*blocks))

        self.final_norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                trunc_normal_(m.weight, std=.02)
                if getattr(m, "bias", None) is not None:
                    nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsamples[i](x)
            x = self.stages[i](x)
        x = x.mean([-2, -1])
        x = self.final_norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.dropout(x)
        x = self.head(x)
        return x
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")    
#model = MiniConvNeXt(in_chans=1, num_classes=2,
#                 depths=(1,1,2,1), dims=(32,64,128,256)).to(device).add_module
    
    
EPOCHS = 200
    
#criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)  # AdamW is preferable
# One-cycle LR often helps for from-scratch training:


from torch.optim.lr_scheduler import OneCycleLR
#scheduler = OneCycleLR(optimizer, max_lr=3e-3, steps_per_epoch=len(train_loader), epochs=EPOCHS)

#optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
#
#def cosine_lr_with_warmup(optimizer, base_lr, epochs, warmup_epochs, steps_per_epoch):
#    def lr_lambda(step):
#        total_steps = epochs * steps_per_epoch
#        warmup_steps = warmup_epochs * steps_per_epoch
#        if step < warmup_steps:
#            return step / warmup_steps
#        progress = (step - warmup_steps) / (total_steps - warmup_steps)
#        return 0.5 * (1.0 + math.cos(math.pi * progress))
#    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
#
#scheduler = cosine_lr_with_warmup(optimizer, base_lr=6e-4, epochs=EPOCHS, warmup_epochs=3, steps_per_epoch=len(train_loader))

#scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
#
#print("\n Beginning Training:")
#start_time = time.time()
#
#for epoch in range(EPOCHS):
#    epoch_start = time.time()
#    model.train()
#    running_loss = 0.0
#    image_count = 0
#    prev_100_images_start = time.time()
#
#    for images, labels in train_loader:
#        images, labels = images.to(device), labels.to(device)
#        optimizer.zero_grad()
#
#        #  Forward pass in mixed precision
#        with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
#            outputs = model(images)
#            loss = criterion(outputs, labels)
#
#        #  Scaled backward pass
#        scaler.scale(loss).backward()
#        
#        if not torch.isfinite(loss):
#            print("Non-finite loss, stopping training")
#            break
#
#        #  Unscale + clip gradients
#        scaler.unscale_(optimizer)
#        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
#
#        #  Step with scaler
#        scaler.step(optimizer)
#        scaler.update()
#
#        #  Step LR scheduler once per batch
#        scheduler.step()
#
#        running_loss += loss.item()
#        image_count += images.size(0)
#
#        # Occasional progress report
#        if image_count % 500 == 0:
#            print(f"Trained {image_count} images, Time: {time.time() - prev_100_images_start:.1f}s")
#            prev_100_images_start = time.time()
#
#    avg_loss = running_loss / len(train_loader)
#    print(f"Epoch {epoch+1:02d}/{EPOCHS} | Loss: {avg_loss:.4f} | Time: {time.time()-epoch_start:.1f}s")
#
#
#print(f"Finished Training on {image_count} images in {time.time() - start_time} seconds")
#
#print("")
#model.eval()
#correct, total = 0, 0
#start_test_time = time.time()
#print("Beginning Testing:")
#with torch.no_grad():
#    for images, labels in test_loader:
#        images, labels = images.to(device), labels.to(device)
#        outputs = model(images)
#        _, preds = torch.max(outputs, 1)
#        total += labels.size(0)
#        correct += (preds == labels).sum().item()
#
#print(f"Test Accuracy: {100 * correct / total:.2f}%")
#print(f"Finished Testing in {time.time() - start_test_time} seconds")

# -------------------------------
# Validation helper
# -------------------------------
def validate(model, loader, criterion, device):
    model.eval(); loss_sum, correct, total = 0,0,0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            out = model(x); loss = criterion(out,y)
            loss_sum += loss.item()*x.size(0)
            _,pred = torch.max(out,1)
            correct += (pred==y).sum().item(); total += y.size(0)
    return loss_sum/total, correct/total


# -------------------------------
# Training loop for one model
# -------------------------------
def train_one(model, optimizer, scheduler, train_ld, val_ld, epochs, device):
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)  # ✅ move model to GPU or CPU once here
    best_acc = 0
    best_loss = float('inf')

    for ep in range(epochs):
        model.train()
        run_loss, correct, total = 0, 0, 0
        t0 = time.time()

        for x, y in train_ld:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            # no autocast or AMP — pure FP32
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            run_loss += loss.item() * x.size(0)
            _, pred = torch.max(out, 1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        if scheduler:
            scheduler.step()

        val_loss, val_acc = validate(model, val_ld, criterion, device)
        train_acc = correct / total
        best_acc = max(best_acc, val_acc)
        best_loss = min(best_loss, val_loss)

        print(f"Epoch[{ep+1:02d}/{epochs}] | TrainAcc:{train_acc:.3f} | "
              f"ValAcc:{val_acc:.3f} | ValLoss:{val_loss:.3f} | "
              f"Time:{time.time()-t0:.1f}s")

    return best_acc, best_loss

# -------------------------------
# Define experiment variants
# -------------------------------
def build_variants():
    base={"depths":(1,1,2,1),"dims":(32,64,128,256),"drop_path_rate":0.0}
    return [
        {"name":"weak_0.15","dropout":0.15,"aug":"weak",**base},
        {"name":"weak_0.25","dropout":0.25,"aug":"weak",**base},
        {"name":"medium_0.15","dropout":0.15,"aug":"medium",**base},
        {"name":"medium_0.25","dropout":0.25,"aug":"medium",**base},
        {"name":"strong_0.15","dropout":0.15,"aug":"strong",**base},
        {"name":"strong_0.25","dropout":0.25,"aug":"strong",**base},
    ]



# -------------------------------
# Run all variants
# -------------------------------
device='cuda' if torch.cuda.is_available() else 'cpu'
epochs=30
results=[]
for v in build_variants():
    print(f"\n=== {v['name']} ===")
    train_loader,val_loader = make_loaders(v["aug"])
    model = MiniConvNeXt(in_chans=1,num_classes=2,
                         depths=v["depths"],dims=v["dims"],
                         drop_path_rate=v["drop_path_rate"])
    model.dropout = nn.Dropout(p=v["dropout"])
    opt = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs)
    best_acc,best_loss = train_one(model,opt,sch,train_loader,val_loader,epochs,device)
    results.append((v["name"],best_acc,best_loss))

print("\n--- Summary of Transform × Dropout Runs ---")
for name,acc,loss in results:
    print(f"{name:18s} | Best Val Acc: {acc:.3f} | Best Val Loss: {loss:.3f}")