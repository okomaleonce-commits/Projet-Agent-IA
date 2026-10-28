# Heretic — suppression automatique de censure des LLM

[Heretic](https://github.com/p-e-w/heretic) (`heretic-llm`) est un outil open-source
(licence AGPL-3.0) qui retire l'« alignement de sécurité » des modèles de langage
transformer locaux par **ablation directionnelle** (« abliteration », Arditi et al. 2024),
couplée à une optimisation automatique de paramètres (Optuna/TPE). Il minimise
conjointement le nombre de refus et la divergence KL par rapport au modèle d'origine,
pour conserver un maximum des capacités du modèle.

> Outil **annexe** à ce dépôt : il n'a aucun lien avec le moteur de pricing
> `betting_engine`. Il est documenté ici parce qu'il a été installé dans
> l'environnement de travail.

## Installation

```bash
./scripts/install-heretic.sh                 # version épinglée (1.4.0)
HERETIC_VERSION=latest ./scripts/install-heretic.sh   # dernière version
```

Le script installe Heretic dans un environnement isolé via `uv tool install`
(repli sur `pip --user` si `uv` est absent). La commande `heretic` devient
disponible dans le PATH (`~/.local/bin`).

Installation manuelle équivalente :

```bash
uv tool install heretic-llm==1.4.0
# ou
pip install -U heretic-llm
```

## Prérequis

- Python 3.10+ et **PyTorch 2.2+** adapté au matériel (tiré automatiquement ;
  certains modèles, ex. gpt-oss MXFP4, exigent PyTorch 2.6+).
- Un **GPU** avec assez de VRAM est fortement recommandé. Décensurer un modèle 4B
  prend ~20-30 min sur une RTX 3090. Sur CPU, le traitement d'un vrai modèle est
  très lent voire impraticable.
- `--quantization BNB_4BIT` (bitsandbytes) réduit fortement la VRAM nécessaire.

## Utilisation

```bash
# Décensurer un modèle HuggingFace
heretic Qwen/Qwen3-4B-Instruct-2507

# Toutes les options
heretic --help

# Évaluer un modèle déjà abliteré (refus + divergence KL)
heretic --model google/gemma-3-12b-it --evaluate-model p-e-w/gemma-3-12b-it-heretic
```

En fin de run, Heretic propose de sauvegarder le modèle, l'uploader sur HuggingFace,
discuter avec pour tester, et/ou lancer des benchmarks.

## Avertissement

Heretic **retire les garde-fous de sécurité** des modèles. À réserver à un usage
de recherche légitime, sous votre responsabilité et dans le respect des licences
des modèles traités.
