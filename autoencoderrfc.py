import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

C_age = np.load('C_age.npy')
C_edu = np.load('C_edu.npy')
C_sex = np.load('C_sex.npy', allow_pickle=True)
X_GM = np.load('X_GM.npy')
Y_dis = np.load('Y_dis.npy')
X_SNP = np.load('X_SNP.npy')

le = LabelEncoder()
C_sex = le.fit_transform(C_sex)
X = np.column_stack((C_age, C_edu, C_sex, X_GM, X_SNP))
Y = np.argmax(Y_dis, axis=1)


class SimCLR_Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, projection_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )

    def forward(self, x):
        return self.encoder(x)

def augment(x):
    noise = torch.randn_like(x) * 0.05
    return x + noise


class ContrastiveDataset(Dataset):
    def __init__(self, X):
        self.X = torch.tensor(X, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        return augment(x), augment(x)


def simclr_loss(z1, z2, temperature=0.5):
    batch_size = z1.size(0)
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    z = torch.cat([z1, z2], dim=0)
    sim_matrix = torch.matmul(z, z.T)
    mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)
    sim_matrix = sim_matrix / temperature
    sim_matrix.masked_fill_(mask, -1e9)
    targets = torch.cat([torch.arange(batch_size) for _ in range(2)]).to(z.device)
    return F.cross_entropy(sim_matrix, targets)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    if task_name == "CN vs MCI":
        Y_task = np.where(Y_task == 0, 0, 1)
        classes = [0, 1]
    elif task_name == "CN vs MCI vs AD":
        Y_task = np.where(Y_task == 2, 1, Y_task)
        classes = sorted(np.unique(Y_task))

    class_mapping = {label: idx for idx, label in enumerate(sorted(classes))}
    Y_task = np.array([class_mapping[y] for y in Y_task])

    accs, aucs = [], []

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, test_idx in skf.split(X_task, Y_task):
        X_train, X_test = X_task[train_idx], X_task[test_idx]
        Y_train, Y_test = Y_task[train_idx], Y_task[test_idx]

        scaler = StandardScaler()
        X_train_std = scaler.fit_transform(X_train)
        X_test_std = scaler.transform(X_test)

        encoder = SimCLR_Encoder(input_dim=X_train_std.shape[1]).to(device)
        optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)
        contrastive_dataset = ContrastiveDataset(X_train_std)
        contrastive_loader = DataLoader(contrastive_dataset, batch_size=256, shuffle=True)

        encoder.train()
        for epoch in range(10): 
            for x1, x2 in contrastive_loader:
                x1, x2 = x1.to(device), x2.to(device)
                z1, z2 = encoder(x1), encoder(x2)
                loss = simclr_loss(z1, z2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        encoder.eval()
        with torch.no_grad():
            X_train_embed = encoder(torch.tensor(X_train_std, dtype=torch.float32).to(device)).cpu().numpy()
            X_test_embed = encoder(torch.tensor(X_test_std, dtype=torch.float32).to(device)).cpu().numpy()

        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            class_weight='balanced',
            random_state=42
        )
        clf.fit(X_train_embed, Y_train)
        Y_pred = clf.predict(X_test_embed)
        Y_prob = clf.predict_proba(X_test_embed)

        accs.append(accuracy_score(Y_test, Y_pred))
        if len(np.unique(Y_task)) > 2:
            aucs.append(roc_auc_score(Y_test, Y_prob, multi_class='ovr'))
        else:
            aucs.append(roc_auc_score(Y_test, Y_prob[:, 1]))

    print(f"Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    if len(np.unique(Y_task)) > 2:
        print(f"mAUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    else:
        print(f"AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
