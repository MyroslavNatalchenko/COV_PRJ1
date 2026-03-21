import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from utils import setup_directories, get_dataloaders, train_model, evaluate_and_save, plot_metrics

EXP_NAME = "exp3_3_warstwy_bez_batchNorm_ReLu_Adam"
BATCH_SIZE = 8
LEARNING_RATE = 0.001
NUM_EPOCHS = 15

class CNN_Exp3(nn.Module):
    def __init__(self, num_classes=37):
        super(CNN_Exp3, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(p=0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(p=0.3),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(p=0.4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 16 * 16, 512),
            nn.ReLU(),
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
        transforms.RandomRotation(degrees=20),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    train_loader, test_loader = get_dataloaders(train_transform, test_transform, BATCH_SIZE)
    model = CNN_Exp3().to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = train_model(model, train_loader, test_loader, criterion, optimizer, NUM_EPOCHS, device)

    torch.save(model.state_dict(), f'models/{EXP_NAME}.pth')
    plot_metrics(history, EXP_NAME)
    evaluate_and_save(model, test_loader, device, EXP_NAME)

if __name__ == '__main__':
    main()