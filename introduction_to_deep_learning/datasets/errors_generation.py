import pandas as pd
import numpy as np
import random


def generate_errors(df: pd.DataFrame, p: float = 0.1):
    """
    generates a copy of the dataframe with introduced errors at the given probability p for each cell
    """
    columns = ['type_contrat', 'formation_selective',
               'concour', 'nombre_candidats', 'taux_admission', 'taux_femmes',
               'taux_boursiers', 'taux_meme_academie', 'taux_meme_etablissement',
               'taux_bac_technologique', 'taux_bac_pro', 'taux_mention_assez_bien',
               'taux_mention_bien', 'taux_mention_tres_bien',
               'taux_mention_tres_bien_felicitations']
    copy = df[columns].copy()
    # type contrat
    distribution = {"Public": 0.794505,
                    "Privé sous contrat d'association": 0.155495,
                    "Privé enseignement supérieur": 0.044176,
                    "Privé hors contrat": 0.005824}
    strings = np.array(list(distribution.keys()))
    thresholds = np.cumsum(np.array(list(distribution.values()))[:-1])
    indexes = np.sum(np.random.rand(len(df), 1) > thresholds[None, :], axis=1)
    copy["type_contrat"] = np.where(np.random(len(df)) < p, strings[indexes], copy.type_contrat)
    # formation selective
    copy["format_selective"] = np.where(np.random(len(df)) < p, np.random(len(df)) < 0.77967, copy.format_selective)
    # concour
    copy["concour"] = np.where(np.random(len(df)) < p, np.random(len(df)) < 0.043736, copy.concour)
    # nombre candidats
    x = np.random.uniform(0, 1, len(df))
    n = np.round(np.clip(np.exp(5.5+x+np.arctanh(2*(x-0.5))), 4, 18190)).astype(int)
    copy["nombre_candidats"] = np.where(np.random(len(df)) < p, n, copy.nombre_candidats)
    # taux admission
    copy["taux_admission"] = np.where(np.random(len(df)) < p, np.random.uniform(0, 1, len(df)), copy.taux_admission)
    # taux femmes
    x = np.random.uniform(0, 1, len(df))
    n = np.maximum(0, (x - 0.08)/(1 - 0.08))
    copy["taux_femmes"] = np.where(np.random(len(df)) < p, np.random.uniform(0, 1, len(df)), copy.taux_femmes)
    # taux boursier
    tb = np.clip(np.exp(-1.1 - 0.5*x + np.arctanh(2.1*(x-0.55))), 0, 1)
    copy["taux_boursiers"] = np.where(np.random(len(df)) < p, tb, copy.taux_boursiers)
    # taux meme academie
    x = np.random.uniform(0, 1, len(df))
    tma = np.minimum((1.15*np.maximum(0, x-0.016))**0.5, 1.)
    copy["taux_meme_academie"] = np.where(np.random(len(df)) < p, tma, copy.taux_meme_academie)
    # taux meme etablissement
    x = np.random.uniform(0, 1, len(df))
    tme =  np.maximum(0, 25*x**3 - 52.5*x**2 + 37.5*x -9)
    copy["taux_meme_etablissement"] = np.where(np.random(len(df)) < p, tme, copy.taux_meme_etablissement)




def dataframe_to_tensor():
    pass


def metric():
    pass


if __name__ == "__main__":
    import pathlib
    path = pathlib.Path(__file__).parent
    df = pd.read_csv(path / "train.csv")
    generate_errors(df)