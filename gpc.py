import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score

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
        Y_task = np.where(np.isin(Y_task, [1, 2]), 1, Y_task)
        class_map = {0: 0, 1: 1, 3: 2}
        Y_task = np.vectorize(class_map.get)(Y_task)

    elif task_name == "CN vs sMCI vs pMCI vs AD":
        class_map = {label: i for i, label in enumerate(classes)}
        Y_task = np.vectorize(class_map.get)(Y_task)

    else:
        class_map = {label: i for i, label in enumerate(classes)}
        Y_task = np.vectorize(class_map.get)(Y_task)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []
    aucs = []

    for train_idx, test_idx in skf.split(X_task, Y_task):
        X_train, X_test = X_task[train_idx], X_task[test_idx]
        Y_train, Y_test = Y_task[train_idx], Y_task[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        gpc = GaussianProcessClassifier(kernel=RBF(), random_state=42)
        gpc.fit(X_train, Y_train)

        Y_pred = gpc.predict(X_test)
        Y_prob = gpc.predict_proba(X_test)

        accuracies.append(accuracy_score(Y_test, Y_pred))

        if len(np.unique(Y_task)) > 2:
            auc = roc_auc_score(Y_test, Y_prob, multi_class='ovr')
        else:
            auc = roc_auc_score(Y_test, Y_prob[:, 1])
        aucs.append(auc)

    print(f'Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}')
    if len(np.unique(Y_task)) > 2:
        print(f'mAUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}')
    else:
        print(f'AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}')
