import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from torch.utils.data import DataLoader, TensorDataset

C_age = np.load('C_age.npy')
C_edu = np.load('C_edu.npy')
C_sex = np.load('C_sex.npy')
X_GM = np.load('X_GM.npy')
X_SNP = np.load('X_SNP.npy')
Y_dis = np.load('Y_dis.npy')

le = LabelEncoder()
C_sex = le.fit_transform(C_sex)
X_input = np.hstack((X_GM, X_SNP)) 
Y = np.argmax(Y_dis, axis=1)

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_tensor = torch.tensor(X_input, dtype=torch.float32).to(device)
dataset = TensorDataset(X_tensor)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

model = Autoencoder(input_dim=X_input.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

print("\n[Autoencoder Pretraining]")
model.train()
for epoch in range(20):
    total_loss = 0
    for batch in loader:
        x = batch[0]
        x_recon = model(x)
        loss = loss_fn(x_recon, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(loader):.4f}")

model.eval()
with torch.no_grad():
    latent_features = model.encoder(X_tensor).cpu().numpy()

X_total = np.column_stack((C_age, C_edu, C_sex, latent_features))

comparison_tasks = {
    "CN vs AD": [0, 3],
    "CN vs MCI": [0, 1, 2],
    "sMCI vs pMCI": [1, 2],
    "CN vs MCI vs AD": [0, 1, 2, 3],
    "CN vs sMCI vs pMCI vs AD": [0, 1, 2, 3]
}

print("\n[Autoencoder Latent Feature → GPC Evaluation]")
for task_name, classes in comparison_tasks.items():
    print(f"\nTask: {task_name}")
    indices = np.isin(Y, classes)
    X_task = X_total[indices]
    Y_task = Y[indices]

    if task_name == "CN vs MCI":
        Y_task = np.where(Y_task == 0, 0, 1)
    elif task_name == "CN vs MCI vs AD":
        Y_task = np.where(np.isin(Y_task, [1, 2]), 1, Y_task)
        class_map = {0: 0, 1: 1, 3: 2}
        Y_task = np.vectorize(class_map.get)(Y_task)
    else:
        class_map = {label: i for i, label in enumerate(classes)}
        Y_task = np.vectorize(class_map.get)(Y_task)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, aucs = [], []

    for train_idx, test_idx in skf.split(X_task, Y_task):
        X_train, X_test = X_task[train_idx], X_task[test_idx]
        Y_train, Y_test = Y_task[train_idx], Y_task[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        kernel = C(1.0) * RBF()
        model_gpc = GaussianProcessClassifier(kernel=kernel, random_state=42)
        model_gpc.fit(X_train, Y_train)
        Y_pred = model_gpc.predict(X_test)
        Y_prob = model_gpc.predict_proba(X_test)

        accs.append(accuracy_score(Y_test, Y_pred))
        if len(np.unique(Y_task)) > 2:
            aucs.append(roc_auc_score(Y_test, Y_prob, multi_class='ovr'))
        else:
            aucs.append(roc_auc_score(Y_test, Y_prob[:, 1]))

    print(f'Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}')
    if len(np.unique(Y_task)) > 2:
        print(f'mAUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}')
    else:
        print(f'AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}')
