

def test_le_track_record_ne_contient_jamais_le_portefeuille_de_repli():
    """
    Régression du 2026-08-19.

    Quand IBKR est injoignable, le runner fabrique un PortfolioSnapshot de
    100 000 $ vide pour pouvoir continuer son analyse. Cette valeur n'a jamais
    décrit aucun compte — mais elle était écrite dans le track record signé,
    aux côtés des vraies séances.

    Résultat sur le fichier de production : 1 021 514 $ le 14 août, puis
    100 000 $ les 17 et 18 — une chute de −90 % qui n'a jamais eu lieu, signée
    et présentée comme un relevé de performance. Trois lignes au total, dont
    une remontant au 15 juillet.

    Le garde-fou correspondant existait déjà pour le coupe-circuit, dont le
    commentaire dit mot pour mot que ce repli produirait « a spurious 90%+
    drawdown ». Il n'avait pas été appliqué au track record.
    """
    from pathlib import Path
    import csv

    p = Path("logs/live_track_record.csv")
    if not p.exists():
        import pytest
        pytest.skip("pas de track record local")

    faux = []
    with p.open() as f:
        for row in csv.DictReader(f):
            try:
                if abs(float(row["netliq"]) - 100_000.0) < 0.01:
                    faux.append(row["date"])
            except (ValueError, KeyError, TypeError):
                continue

    assert not faux, (
        f"{len(faux)} séance(s) au portefeuille de repli dans le track record "
        f"signé : {faux}. Une séance sans courtier n'est pas une séance."
    )
