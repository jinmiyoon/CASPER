def ac(cfe: float, feh: float) -> float:
    return cfe + feh + 8.43


def cfe(ac: float, feh: float) -> float:
    return ac - 8.43 - feh
