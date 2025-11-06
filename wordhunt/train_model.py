# list of letters that need more training:
# F (confused as E), Y (confused as V), X (confused as K)
# D, K, U

import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import os
from PIL import Image

class LetterDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        # Map letters to indices (A=0, B=1, etc.)
        self.letter_to_idx = {chr(i + 65): i for i in range(26)}
        
        # Load all images and labels
        for letter in self.letter_to_idx.keys():
            letter_dir = os.path.join(root_dir, letter)
            if os.path.exists(letter_dir):
                for img_name in os.listdir(letter_dir):
                    if img_name.endswith('.png'):
                        img_path = os.path.join(letter_dir, img_name)
                        self.samples.append((img_path, self.letter_to_idx[letter]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, img_path

class LetterCNN(nn.Module):
    def __init__(self):
        super(LetterCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 26)  # 26 letters
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(-1, 64 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return nn.functional.log_softmax(x, dim=1)

def describe_letter_issues(letter_stats, epoch, accuracy, top_n=5):
    # Filter letters that appeared in validation
    observed_letters = [idx for idx, stats in letter_stats.items() if stats['total'] > 0]
    if not observed_letters:
        print("No validation samples available to analyze letter-level performance.")
        return

    def letter_accuracy(idx):
        stats = letter_stats[idx]
        return stats['correct'] / stats['total'] if stats['total'] else 0.0

    worst_letter_idx = min(observed_letters, key=lambda idx: letter_accuracy(idx))
    worst_stats = letter_stats[worst_letter_idx]
    worst_letter = chr(worst_letter_idx + 65)
    worst_accuracy = letter_accuracy(worst_letter_idx) * 100
    errors = worst_stats['misclassified']

    print(f"\nDetailed diagnostics for epoch {epoch} (accuracy {accuracy:.2f}%):")
    print(f"Worst-performing letter: {worst_letter} (correct {worst_stats['correct']}/{worst_stats['total']} -> {worst_accuracy:.2f}% accuracy)")

    if not errors:
        print(f"All validation samples for letter {worst_letter} classified correctly.")
        return

    # Sort by confidence descending so high-confidence mistakes surface first
    errors_sorted = sorted(errors, key=lambda item: item['confidence'], reverse=True)
    print(f"Top {min(top_n, len(errors_sorted))} misclassified samples for {worst_letter}:")
    for idx, error in enumerate(errors_sorted[:top_n], start=1):
        predicted_letter = chr(error['predicted_idx'] + 65)
        confidence = error['confidence'] * 100
        print(f"  {idx}. {error['path']} -> predicted {predicted_letter} ({confidence:.1f}% confidence)")

def train_model(model, train_loader, val_loader, device, epochs=100): # 100 epochs is prolly unnecessary, 50 is enough
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    best_accuracy = 0
    best_letter_stats = None
    best_epoch = None
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        # Training phase
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for inputs, labels, _ in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix({'loss': running_loss/len(train_loader)})
        
        # Validation phase
        model.eval()
        correct = 0
        total = 0
        letter_stats = {i: {'total': 0, 'correct': 0, 'misclassified': []} for i in range(26)}
        with torch.no_grad():
            for inputs, labels, paths in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                probabilities = outputs.exp()
                _, predicted = torch.max(probabilities, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                for idx in range(labels.size(0)):
                    label_idx = labels[idx].item()
                    pred_idx = predicted[idx].item()
                    letter_stats[label_idx]['total'] += 1
                    if pred_idx == label_idx:
                        letter_stats[label_idx]['correct'] += 1
                    else:
                        letter_stats[label_idx]['misclassified'].append({
                            'path': paths[idx],
                            'predicted_idx': pred_idx,
                            'confidence': probabilities[idx, pred_idx].item()
                        })

        accuracy = 100 * correct / total
        print(f'Validation Accuracy: {accuracy:.2f}%')

        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'letter_recognition_model.pth')
            print(f'New best model saved with accuracy: {accuracy:.2f}%')
            best_letter_stats = copy.deepcopy(letter_stats)
            best_epoch = epoch + 1
            describe_letter_issues(best_letter_stats, best_epoch, best_accuracy)

    if best_letter_stats is None:
        print("No improvement recorded during training; diagnostics unavailable.")
    else:
        print(f"\nBest validation accuracy {best_accuracy:.2f}% achieved at epoch {best_epoch}.")
        describe_letter_issues(best_letter_stats, best_epoch, best_accuracy)


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Load datasets
    dataset = LetterDataset("letter_dataset", transform=transform)
    
    # Print dataset statistics
    print(f"Total samples in dataset: {len(dataset)}")
    letter_counts = {}
    for _, label in dataset.samples:
        letter = chr(label + 65)
        letter_counts[letter] = letter_counts.get(letter, 0) + 1
    print("\nSamples per letter:")
    for letter, count in sorted(letter_counts.items()):
        print(f"{letter}: {count}")
    
    # Split into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    print(f"\nTraining samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    model = LetterCNN().to(device)
    
    # Train model
    train_model(model, train_loader, val_loader, device)
    
    print("Training completed!")

if __name__ == "__main__":
    main()
