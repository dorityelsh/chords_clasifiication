import torch  # ליבה של PyTorch
import torch.nn as nn  # מודלים ושכבות
import torch.optim as optim  # אופטימייזרים
import torch.nn.functional as F  # פונקציות אקטיבציה ואובדן 
from torch.utils.data import DataLoader, Dataset , random_split # להכנת דאטה
import numpy as np  # טיפול במערכים
from main import main  # הפונקציה הראשית שמבצעת את כל התהליך

df = main()
print("✅ DataFrame loaded with shape:", df.shape)

class ChordDataset(Dataset):
    def __init__(self, df):
        # כאן אנחנו שומרים את הפיצ׳רים והאקורדים
        self.X = torch.tensor(df.drop('chord', axis=1).values, dtype=torch.float32)
        self.y = torch.tensor(df['chord'].astype('category').cat.codes.values, dtype=torch.long)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        # כאן אפשר להוסיף preprocessing אם רוצים
        return self.X[idx], self.y[idx]
    

def create_loaders(df, batch_size=32, val_split=0.2):
    dataset = ChordDataset(df)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader

train_loader, val_loader = create_loaders(df)
class ChordMLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)
    
input_size = df.shape[1] - 1  # בלי עמודת 'chord'
num_classes = df['chord'].nunique()

model = ChordMLP(input_size, num_classes)