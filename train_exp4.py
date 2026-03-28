import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from utils import setup_directories, train_model_exp4, evaluate_and_save, plot_metrics, get_dataloaders_improved

EXP_NAME = "Exp4_4Layers_GELU_AdamW_LR0001_DropoutIncreased_CosineScheduler_30Epochs_StratifiedDataSplit20Test80Train"
BATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_EPOCHS = 30

class CNN_Exp4(nn.Module):
    def __init__(self, num_classes=37):
        super(CNN_Exp4, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(p=0.2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(p=0.2),          # было 0.1 → 0.2

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(p=0.3),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(p=0.3),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

def main():
    setup_directories()
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    train_loader, test_loader = get_dataloaders_improved(train_transform, test_transform, BATCH_SIZE)
    model = CNN_Exp4().to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    history = train_model_exp4(model, train_loader, test_loader, criterion, optimizer, scheduler, NUM_EPOCHS, device)

    torch.save(model.state_dict(), f'models/{EXP_NAME}.pth')
    plot_metrics(history, EXP_NAME)
    evaluate_and_save(model, test_loader, device, EXP_NAME)

if __name__ == '__main__':
    main()