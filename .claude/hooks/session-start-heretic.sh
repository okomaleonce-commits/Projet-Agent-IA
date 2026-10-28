#!/usr/bin/env bash
#
# SessionStart hook : s'assure que Heretic (heretic-llm) est disponible.
#
# Conçu pour les sessions cloud, dont le conteneur est éphémère (dépôt recloné
# à chaque démarrage). Le hook est :
#   - idempotent : no-op immédiat si la commande `heretic` est déjà présente ;
#   - non bloquant : l'installation (tire PyTorch, plusieurs Go) est lancée en
#     arrière-plan pour ne pas retarder le démarrage de la session.
#
# La sortie de ce script est injectée dans le contexte de la session ; on la
# garde donc courte.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INSTALLER="${REPO_DIR}/scripts/install-heretic.sh"
LOG_DIR="${REPO_DIR}/.claude/hooks"
LOG_FILE="${LOG_DIR}/heretic-install.log"

# Rendre ~/.local/bin visible pour la détection ci-dessous.
export PATH="${HOME}/.local/bin:${PATH}"

if command -v heretic >/dev/null 2>&1; then
    echo "Heretic déjà installé ($(command -v heretic))."
    exit 0
fi

if [ ! -x "${INSTALLER}" ]; then
    echo "Heretic : installeur introuvable (${INSTALLER}) — étape ignorée."
    exit 0
fi

# Lancement détaché : la session démarre sans attendre la fin de l'install.
nohup bash "${INSTALLER}" >"${LOG_FILE}" 2>&1 &
echo "Heretic : installation lancée en arrière-plan (log : ${LOG_FILE})."
echo "La commande 'heretic' sera disponible dans quelques minutes."
exit 0
