import torch
import matplotlib.pyplot as plt
from NeuralNetworkBC import MonReseau,test
from sklearn.metrics import roc_curve, roc_auc_score


model=MonReseau()
model.load_state_dict(torch.load("model.pth"))
model.eval()
torch.manual_seed(42)

with torch.no_grad():
    logit=model(test.tensors[0])
    proba=torch.sigmoid(logit).squeeze(1)
    pred=(proba>=0.5).float()

vrai_vals=test.tensors[1]

vrai_pos=(pred==1) & (vrai_vals==1)
faux_pos=(pred==1) & (vrai_vals==0)
vrai_neg=(pred==0) & (vrai_vals==0)
faux_neg=(pred==0) & (vrai_vals==1)

n_vp=vrai_pos.sum().item()
n_fp=faux_pos.sum().item()
n_vn=vrai_neg.sum().item()
n_fn=faux_neg.sum().item()

sensitivity=n_vp/(n_vp+n_fn)
specificity=n_vn/(n_vn+n_fp)
accuracy = (pred == vrai_vals).float().mean().item() * 100

print(f"Vrais négatifs : {n_vn}")
print(f"Vrais positifs : {n_vp}")
print(f"Faux positifs : {n_fp}")
print(f"Faux négatifs : {n_fn}")
print(f"Accuracy : {accuracy:.2f}%")
print(f"Sensitivity : {sensitivity:.2f}%")
print(f"Specificity : {specificity:.2f}%")

idx=torch.arange(len(test))

plt.figure(figsize=(12,6))

plt.scatter(
    idx[vrai_neg.squeeze()],
    vrai_vals[vrai_neg].flatten(),
    label=f"Vrais négatifs (n={n_vn})"
)

plt.scatter(
    idx[vrai_pos.squeeze()],
    vrai_vals[vrai_pos].flatten(),
    label=f"Vrais positifs (n={n_vp})"
)

plt.scatter(
    idx[faux_pos.squeeze()],
    vrai_vals[faux_pos].flatten(),
    label=f"Faux positifs (n={n_fp})"
)

plt.scatter(
    idx[faux_neg.squeeze()],
    vrai_vals[faux_neg].flatten(),
    label=f"Faux négatifs (n={n_fn})"
)

plt.yticks([0,1], ["Bénin (0)", "Malin (1)"])
plt.xlabel("Individu")
plt.ylabel("Classe réelle")
plt.title(f"Prédictions du modèle — Accuracy : {accuracy:.2f}%")
plt.legend()
plt.grid(True)

plt.show()

plt.figure(figsize=(12,6))

plt.scatter(
    idx[vrai_pos],
    proba[vrai_pos],
    label=f"Vrai positifs (n={n_vp})"
)

plt.scatter(
    idx[faux_pos],
    proba[faux_pos],
    label=f"Faux positifs (n={n_fp})"
)

plt.scatter(
    idx[vrai_neg],
    proba[vrai_neg],
    label=f"Vrai négatifs (n={n_vn})"
)

plt.scatter(
    idx[faux_neg],
    proba[faux_neg],
    label=f"Faux négatifs (n={n_fn})"
)

plt.axhline(
    0.5,
    linestyle="--",
    label="Seuil de décision (0.5)"
)

plt.xlabel("Individu")
plt.ylabel("Probabilité prédite d'être malin")
plt.title(f"Probabilités prédites — Accuracy : {accuracy:.2f}%")
plt.ylim(0,1)
plt.legend()
plt.grid(True)

plt.show()

fpr, tpr, seuils=roc_curve(vrai_vals.numpy(),proba.numpy())
auc=roc_auc_score(vrai_vals.numpy(),proba.numpy())

plt.figure(figsize=(7, 7))

plt.plot(
    fpr,
    tpr,
    label=f"ROC (AUC = {auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Classifieur aléatoire"
)

plt.xlabel("Taux de faux positifs (FPR)")
plt.ylabel("Taux de vrais positifs (TPR)")
plt.title("Courbe ROC")
plt.legend()
plt.grid(True)

plt.show()

print(f"AUC = {auc:.3f}")