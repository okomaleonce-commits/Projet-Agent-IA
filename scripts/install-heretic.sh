#!/usr/bin/env bash
#
# Installe Heretic (https://github.com/p-e-w/heretic), un outil d'« abliteration »
# qui retire automatiquement l'alignement de sécurité des LLM transformer locaux.
#
# Outil annexe au projet (recherche / expérimentation), installé dans un
# environnement isolé via `uv tool`. Repli sur `pip --user` si uv est absent.
#
# Usage :
#   ./scripts/install-heretic.sh              # version épinglée ci-dessous
#   HERETIC_VERSION=latest ./scripts/install-heretic.sh   # dernière version
#
set -euo pipefail

# Version épinglée pour la reproductibilité (voir `uv tool list`).
HERETIC_VERSION="${HERETIC_VERSION:-1.4.0}"

if [ "${HERETIC_VERSION}" = "latest" ]; then
    SPEC="heretic-llm"
else
    SPEC="heretic-llm==${HERETIC_VERSION}"
fi

echo ">> Installation de ${SPEC}"

if command -v uv >/dev/null 2>&1; then
    echo ">> uv détecté — installation en outil isolé (uv tool install)"
    uv tool install --force "${SPEC}"
else
    echo ">> uv absent — repli sur pip (installation utilisateur)"
    python3 -m pip install --user -U "${SPEC}"
fi

echo
echo ">> Terminé. Vérification :"
heretic --help >/dev/null 2>&1 && echo "   heretic OK : $(command -v heretic)" \
    || echo "   ATTENTION : la commande 'heretic' n'est pas dans le PATH (ajoutez ~/.local/bin)."

cat <<'EOF'

Prérequis d'exécution :
  - Python 3.10+ et PyTorch 2.2+ adapté au matériel (tiré automatiquement).
  - Un GPU avec assez de VRAM est fortement recommandé ; sur CPU le traitement
    d'un vrai modèle est très lent. L'option `--quantization BNB_4BIT` réduit
    la VRAM nécessaire.

Exemples :
  heretic Qwen/Qwen3-4B-Instruct-2507
  heretic --help
EOF
