"""
Milan Capital — Créer / mettre à jour un compte du dashboard.

Usage:
    python -m src.dashboard.create_user
"""
from __future__ import annotations

import getpass
import json
import pathlib

import bcrypt

USERS_PATH = pathlib.Path(__file__).parent / "users.json"


def main() -> None:
    username = input("Identifiant: ").strip()
    password = getpass.getpass("Mot de passe: ")
    if not username or not password:
        print("❌ Identifiant et mot de passe requis.")
        return

    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    users = json.loads(USERS_PATH.read_text()) if USERS_PATH.exists() else []
    users = [u for u in users if u["username"] != username]
    users.append({"username": username, "password_hash": hashed})
    USERS_PATH.write_text(json.dumps(users, indent=2))

    print(f"✅ Compte '{username}' créé/mis à jour dans {USERS_PATH}")
    print("ℹ️  Pour le déploiement Railway, copie le contenu de ce fichier")
    print("   dans la variable d'env MILAN_USERS_JSON (railway variables set).")


if __name__ == "__main__":
    main()
