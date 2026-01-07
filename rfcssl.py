import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score
import random

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

C_age = np.load('C_age.npy')
C_edu = np.load('C_edu.npy')
C_sex = np.load('C_sex.npy')
X_GM = np.load('X_GM.npy')
X_SNP = np.load('X_SNP.npy')
Y_dis = np.load('Y_dis.npy')

le = LabelEncoder()
C_sex = le.fit_transform(C_sex)

X_ssl = np.hstack((X_GM, X_SNP))


def augment(x):
    x_aug = x + np.random.normal(0, 0.01, size=x.shape)
    mask = np.random.rand(*x.shape) < 0.1
    x_aug[mask] = 0
    return x_aug


class FeatureDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        x1 = augment(x)
        x2 = augment(x)
        return torch.tensor(x1, dtype=torch.float32), torch.tensor(x2, dtype=torch.float32)


class SimCLRModel(nn.Module):
    def __init__(self, input_dim, projection_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.projection = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, projection_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projection(h)
        return F.normalize(z, dim=1), h


def nt_xent_loss(z1, z2, temperature=0.5):
    z = torch.cat([z1, z2], dim=0)
    sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temperature
    batch_size = z1.size(0)
    labels = torch.arange(batch_size, device=z.device)
    labels = torch.cat([labels, labels], dim=0)
    mask = torch.eye(batch_size * 2, device=z.device).bool()
    sim.masked_fill_(mask, -1e9)
    return F.cross_entropy(sim, labels)


dataset = FeatureDataset(X_ssl)
loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimCLRModel(input_dim=X_ssl.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print("\n[SimCLR Pretraining]")
for epoch in range(50):
    model.train()
    total_loss = 0
    for x1, x2 in loader:
        x1, x2 = x1.to(device), x2.to(device)
        z1, _ = model(x1)
        z2, _ = model(x2)
        loss = nt_xent_loss(z1, z2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(loader):.4f}")

model.eval()
with torch.no_grad():
    X_tensor = torch.tensor(X_ssl, dtype=torch.float32).to(device)
    _, latent_features = model(X_tensor)
    latent_features = latent_features.cpu().numpy()

X = np.column_stack((C_age, C_edu, C_sex, X_GM, X_SNP, latent_features))
Y = np.argmax(Y_dis, axis=1)

comparison_tasks = {
    "CN vs AD": [0, 3],
    "CN vs MCI": [0, 1, 2],
    "sMCI vs pMCI": [1, 2],
    "CN vs MCI vs AD": [0, 1, 2, 3],
    "CN vs sMCI vs pMCI vs AD": [0, 1, 2, 3]
}

for task_name, classes in comparison_tasks.items():
    print(f"\n[Task: {task_name}]")
    indices = np.isin(Y, classes)
    X_task = X[indices]
    Y_task = Y[indices]

    class_mapping = {label: idx for idx, label in enumerate(sorted(classes))}
    Y_task = np.array([class_mapping[y] for y in Y_task])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, aucs = [], []

    for train_idx, test_idx in skf.split(X_task, Y_task):
        X_train, X_test = X_task[train_idx], X_task[test_idx]
        Y_train, Y_test = Y_task[train_idx], Y_task[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            class_weight='balanced',
            random_state=42
        )
        clf.fit(X_train, Y_train)
        Y_pred = clf.predict(X_test)
        Y_prob = clf.predict_proba(X_test)

        accs.append(accuracy_score(Y_test, Y_pred))
        if len(classes) > 2:
            aucs.append(roc_auc_score(Y_test, Y_prob, multi_class='ovr'))
        else:
            aucs.append(roc_auc_score(Y_test, Y_prob[:, 1]))

    print(f"Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    if len(classes) > 2:
        print(f"mAUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    else:
        print(f"AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
