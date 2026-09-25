# CNN "from scratch" vs Transfert Learning — Cats vs Dogs

## Objectif du devoir 

Comparer un modèle **CNN entraîné from scratch** et un modèle en **transfert d'apprentissage**
(ResNet18 pré-entraîné sur ImageNet) sur le jeu de données Cats vs Dogs, et évaluer l'impact
du transfer learning sur la convergence, les métriques finales et le nombre de paramètres
entraînés.

Le dépôt contient deux notebooks :

- [`Part 7 - Loading Image Data (exo)_v2.ipynb`](<Part 7 - Loading Image Data (exo)_v2.ipynb>) :
  l'exercice guidé de chargement d'images (`ImageFolder` / `DataLoader`), inchangé.
- [`Devoir - CNN from scratch vs Transfer Learning.ipynb`](<Devoir - CNN from scratch vs Transfer Learning.ipynb>) :
  notebook **autonome** contenant tout le code du devoir — les deux expériences, le suivi des
  métriques, l'entraînement, l'évaluation et les comparaisons. C'est celui à exécuter et à
  rendre.

## Environnement

Le notebook est conçu pour tourner sur **Google Colab avec un runtime GPU**
(`Exécution > Modifier le type d'exécution > GPU`) — les cellules `drive.mount` /
`%cd` en tête de notebook montent le Drive contenant ce dossier.

Pour reproduire en local (CPU ou GPU) :

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

## Organisation des données

Le jeu de données [Cats vs Dogs]
Il n'est **pas** poussé sur GitHub (voir `.gitignore`).
Après téléchargement, dézippez-le à la racine du projet de façon à obtenir :

```
Cat_Dog_data/
├─ train/
│  ├─ cat/
│  └─ dog/
└─ test/
   ├─ cat/
   └─ dog/
```

Sur Colab, placez le dossier `Cat_Dog_data/` dans le même répertoire Drive que le notebook
(chemin monté par les cellules `drive.mount` / `%cd` en tête de notebook), ou décommentez la
cellule `#!unzip Cat_Dog_data.zip` si vous y déposez l'archive.

## Commandes pour entraîner

On commence par :

**Expérience A — CNN from scratch** (`CNNFromScratch`, 4 blocs Conv-BatchNorm-ReLU-MaxPool +
tête Dropout) :
- Recherche rapide d'optimiseur : Adam (`lr=1e-3`) vs SGD (`lr=1e-2`, `momentum=0.9`), 3 époques
  chacun (cellule "Recherche du meilleur optimiseur").
- Entraînement final : 10 époques avec l'optimiseur retenu + `CosineAnnealingLR`.
- Dropout `p=0.4` dans la tête dense, BatchNorm après chaque convolution.

**Expérience B — Transfer learning** (`build_transfer_model`, ResNet18 pré-entraîné) :
- Couches gelées sauf `layer4` (fine-tuning léger) + nouvelle tête `Dropout(0.4) -> Linear(2)`.
- Optimiseur Adam (`lr=1e-4`) sur les seuls paramètres entraînables, 10 époques,
  `CosineAnnealingLR`.

Les hyperparamètres (`EPOCHS_*`, `BATCH_SIZE`, `lr`, `dropout`) sont regroupés en tête des
cellules correspondantes et peuvent être ajustés directement.

Le suivi (loss/accuracy/précision/rappel, train et val, par époque) est journalisé dans
TensorBoard sous `runs/<run_name>/` :

```bash
tensorboard --logdir runs
```

## Commandes pour évaluer / recharger le modèle

Le meilleur modèle de chaque expérience (meilleure accuracy de validation) est sauvegardé
automatiquement pendant l'entraînement dans :

- `checkpoints/cnn_from_scratch_best.pth`
- `checkpoints/transfer_resnet18_best.pth`

La section **"6. Évaluation finale sur le jeu de test"** du notebook recharge ces checkpoints
dans des modèles neufs (`evaluate_on_test`) et calcule les métriques finales + matrices de
confusion sur le jeu de test, jamais vu pendant l'entraînement.

## Reproductibilité

Une graine fixe (`SEED = 42`) est utilisée pour `random`, `numpy`, `torch` (CPU et CUDA) ainsi
que pour le split train/val, avec `cudnn.deterministic = True`.

## Résultats

A compléter après exécution sur Colab (voir le tableau récapitulatif produit par la section
7 du notebook, et les courbes de la section 5) :_

| Expérience | Test loss | Test accuracy | Test précision | Test rappel |
|---|---|---|---|---|
| CNN from scratch | | | | |
| Transfer learning (ResNet18) | | | | |

**Analyse (2-3 paragraphes) :** _à rédiger après avoir observé les courbes et le tableau —
comparer la vitesse de convergence, les métriques finales, et relier ces résultats au nombre
de paramètres entraînables de chaque modèle._

## Limites & pistes d'amélioration

_À compléter :_ temps de calcul, taille du jeu de validation, sensibilité au learning rate,
fine-tuning de couches supplémentaires, augmentation plus poussée, autres backbones
(MobileNet, EfficientNet), etc.

## Remise

- Vidéo de 5 à 10 minutes expliquant le travail.
- Lien du dépôt envoyé à diallomous@gmail.com avant vendredi 25 septembre 23h00 (Africa/Dakar).
