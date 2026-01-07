import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
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

class FeatureDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        x1 = x + np.random.normal(0, 0.01, size=x.shape)
        x2 = x + np.random.normal(0, 0.01, size=x.shape)
        return torch.tensor(x1, dtype=torch.float32), torch.tensor(x2, dtype=torch.float32)

class SimCLRModel(nn.Module):
    def __init__(self, input_dim, projection_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.projection = nn.Sequential(
            nn.ReLU(),
            nn.Linear(128, projection_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projection(h)
        return F.normalize(z, dim=1)

def nt_xent_loss(z1, z2, temperature=0.5):
    z = torch.cat([z1, z2], dim=0)
    sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
    sim /= temperature
    batch_size = z1.size(0)
    labels = torch.arange(batch_size, device=z.device)
    labels = torch.cat([labels, labels], dim=0)
    mask = torch.eye(batch_size * 2, device=z.device).bool()
    sim.masked_fill_(mask, -1e9)
    loss = F.cross_entropy(sim, labels)
    return loss

X_ssl = np.hstack((X_GM, X_SNP))
ssl_dataset = FeatureDataset(X_ssl)
ssl_loader = DataLoader(ssl_dataset, batch_size=128, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimCLRModel(input_dim=X_ssl.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print("\n[SimCLR Pretraining]")
for epoch in range(20):
    model.train()
    total_loss = 0
    for x1, x2 in ssl_loader:
        x1, x2 = x1.to(device), x2.to(device)
        z1, z2 = model(x1), model(x2)
        loss = nt_xent_loss(z1, z2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(ssl_loader):.4f}")

model.eval()
with torch.no_grad():
    latent_features = model.encoder(torch.tensor(X_ssl, dtype=torch.float32).to(device)).cpu().numpy()

X = np.column_stack((C_age, C_edu, C_sex, latent_features))
Y = np.argmax(Y_dis, axis=1)

comparison_tasks = {
    "CN vs AD": [0, 3],
    "CN vs MCI": [0, 1, 2],
    "sMCI vs pMCI": [1, 2],
    "CN vs MCI vs AD": [0, 1, 2, 3],
    "CN vs sMCI vs pMCI vs AD": [0, 1, 2, 3]
}

for task_name, classes in comparison_tasks.items():
    print(f"\nTask: {task_name}")
    indices = np.isin(Y, classes)
    X_task = X[indices]
    Y_task = Y[indices]

    if task_name == "CN vs MCI":
        Y_task = np.where(Y_task == 0, 0, 1)
    elif task_name == "CN vs MCI vs AD":
        Y_task = np.where(np.isin(Y_task, [1, 2]), 1, Y_task)  # MCI 병합
        class_map = {0: 0, 1: 1, 3: 2}
        Y_task = np.vectorize(class_map.get)(Y_task)
    elif task_name == "CN vs sMCI vs pMCI vs AD":
        class_map = {label: i for i, label in enumerate(classes)}  # 4-class 유지
        Y_task = np.vectorize(class_map.get)(Y_task)
    else:
        class_map = {label: i for i, label in enumerate(classes)}
        Y_task = np.vectorize(class_map.get)(Y_task)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    accuracies, aucs = [], []

    for train_idx, test_idx in skf.split(X_task, Y_task):
        X_train, X_test = X_task[train_idx], X_task[test_idx]
        Y_train, Y_test = Y_task[train_idx], Y_task[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model_svm = SVC(kernel='linear', probability=True, C=1.0, random_state=SEED)
        model_svm.fit(X_train, Y_train)
        Y_pred = model_svm.predict(X_test)
        Y_prob = model_svm.predict_proba(X_test)

        accuracies.append(accuracy_score(Y_test, Y_pred))
        if len(np.unique(Y_task)) > 2:
            aucs.append(roc_auc_score(Y_test, Y_prob, multi_class='ovr'))
        else:
            aucs.append(roc_auc_score(Y_test, Y_prob[:, 1]))

    print(f'Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}')
    if len(np.unique(Y_task)) > 2:
        print(f'mAUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}')
    else:
        print(f'AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}')