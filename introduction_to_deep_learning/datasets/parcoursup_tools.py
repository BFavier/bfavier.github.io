import pandas as pd
import numpy as np
import torch

columns = ['type_contrat', 'formation_selective',
           'concour', 'nombre_candidats', 'taux_admission', 'taux_femmes',
           'taux_boursiers', 'taux_meme_academie', 'taux_meme_etablissement',
           'taux_bac_technologique', 'taux_bac_pro', 'taux_mention_assez_bien',
           'taux_mention_bien', 'taux_mention_tres_bien',
           'taux_mention_tres_bien_felicitations']


def generate_errors(df: pd.DataFrame, p: float = 0.1):
    """
    generates a copy of the dataframe with introduced errors at the given probability p for each cell
    """
    copy = df[columns].copy()
    # type contrat
    distribution = {"Public": 0.794505,
                    "Privé sous contrat d'association": 0.155495,
                    "Privé enseignement supérieur": 0.044176,
                    "Privé hors contrat": 0.005824}
    strings = np.array(list(distribution.keys()))
    thresholds = np.cumsum(np.array(list(distribution.values()))[:-1])
    indexes = np.sum(np.random.rand(len(df), 1) > thresholds[None, :], axis=1)
    copy["type_contrat"] = np.where(np.random.rand(len(df)) < p, strings[indexes], copy.type_contrat)
    # formation selective
    copy["formation_selective"] = np.where(np.random.rand(len(df)) < p, np.random.rand(len(df)) < 0.77967, copy.formation_selective)
    # concour
    copy["concour"] = np.where(np.random.rand(len(df)) < p, np.random.rand(len(df)) < 0.043736, copy.concour)
    # nombre candidats
    x = np.random.rand(len(df))
    n = np.round(np.clip(np.exp(5.5+x+np.arctanh(2*(x-0.5))), 4, 18190)).astype(int)
    copy["nombre_candidats"] = np.where(np.random.rand(len(df)) < p, n, copy.nombre_candidats)
    # taux admission
    copy["taux_admission"] = np.where(np.random.rand(len(df)) < p, np.random.rand(len(df)), copy.taux_admission)
    # taux femmes
    x = np.random.rand(len(df))
    n = np.maximum(0, (x - 0.08)/(1 - 0.08))
    copy["taux_femmes"] = np.where(np.random.rand(len(df)) < p, np.random.rand(len(df)), copy.taux_femmes)
    # taux boursier
    tb = np.clip(np.exp(-1.1 - 0.5*x + np.arctanh(np.clip(2.1*(x-0.55), -0.9999, 0.9999))), 0, 1)
    copy["taux_boursiers"] = np.where(np.random.rand(len(df)) < p, tb, copy.taux_boursiers)
    # taux meme academie
    x = np.random.rand(len(df))
    tma = np.minimum((1.15*np.maximum(0, x-0.016))**0.5, 1.)
    copy["taux_meme_academie"] = np.where(np.random.rand(len(df)) < p, tma, copy.taux_meme_academie)
    # taux meme etablissement
    x = np.random.rand(len(df))
    tme =  np.maximum(0, 25*x**3 - 52.5*x**2 + 37.5*x -9)
    copy["taux_meme_etablissement"] = np.where(np.random.rand(len(df)) < p, tme, copy.taux_meme_etablissement)
    # types de bac/types mention
    params = {'taux_bac_technologique': [0.2, 0.87],
              'taux_bac_pro': [0.347, 0.876],
              'taux_mention_assez_bien': [0.046, 0.876],
              'taux_mention_bien': [0.1, 0.85],
              'taux_mention_tres_bien': [0.415, 0.862],
              'taux_mention_tres_bien_felicitations': [0.851, 0.923]}
    for all_taux in (['taux_bac_technologique', 'taux_bac_pro'],
                     ['taux_mention_assez_bien',
                      'taux_mention_bien', 'taux_mention_tres_bien',
                      'taux_mention_tres_bien_felicitations']):
        for taux in all_taux:
            margin = 1 - df[[c for c in all_taux if c != taux]].sum(axis=1)
            x = np.random.rand(len(df))
            a, b = params[taux]
            t = np.clip((x - a)/(b - a), 0, 1)
            copy[taux] = np.where(np.random.rand(len(df)) < p, t*margin, copy[taux])
    return copy


def dataframe_to_tensor(df: pd.DataFrame):
    """
    converts a dataframe to a tensor of floats of shape (N, D)
    """
    df = df.copy()
    df["nombre_candidats"] = df["nombre_candidats"]/1413
    df["type_contrat"] = [0 if tc.startswith("Public") else 1 for tc in df["type_contrat"]]
    return df[columns].astype(float).to_numpy()


def metric(y_pred: torch.Tensor, y_target: torch.Tensor):
    pass



if __name__ == "__main__":
    import pathlib
    import IPython
    path = pathlib.Path(__file__).parent
    df = pd.read_csv(path / "train.csv")
    errs = generate_errors(df)
    t = dataframe_to_tensor(df)
    IPython.embed()
