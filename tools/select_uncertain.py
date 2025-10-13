import os, glob, statistics, shutil
from pathlib import Path

# Répertoires (relatifs à ce script)
SCRIPT_DIR = Path(__file__).resolve().parent
PRED_DIR = SCRIPT_DIR / r'al_runs' / r'predict_pool' / r'labels'
IMG_SRC  = SCRIPT_DIR / r'pool_newspaper'
OUT_DIR  = SCRIPT_DIR / r'to_annotate_active'
OUT_DIR.mkdir(exist_ok=True, parents=True)

# Paramètres
N = int(os.environ.get("SELECT_N", 40))  # nombre d'images à copier
TH_LO = float(os.environ.get("TH_LO", 0.2))  # borne inf pour "incertain"
TH_HI = float(os.environ.get("TH_HI", 0.4))  # borne sup pour "incertain"
INCLUDE_NO_DET = os.environ.get("INCLUDE_NO_DET", "0") == "1"  # inclure fichiers sans détection (score=0)

print(f"[INFO] PRED_DIR: {PRED_DIR}")
print(f"[INFO] IMG_SRC : {IMG_SRC}")
print(f"[INFO] OUT_DIR : {OUT_DIR}")
if not PRED_DIR.exists():
    raise SystemExit(f"[ERREUR] Dossier des labels prédits introuvable: {PRED_DIR}\n"
                     f"→ As-tu bien lancé l'inférence avec project=al_runs name=predict_pool ?")

# Indexer toutes les images du pool (jpg + png, récursif) pour retrouver le chemin réel par 'stem'
pool_map = {}
for ext in ("*.jpg", "*.png"):
    for p in IMG_SRC.rglob(ext):
        stem = p.stem.lower()
        # priorité au .jpg si conflit
        if stem not in pool_map or p.suffix.lower() == ".jpg":
            pool_map[stem] = p

print(f"[INFO] Images détectées dans le pool: {len(pool_map)}")

# Lire les fichiers de labels prédits et calculer la moyenne des confs
candidats = []   # (score_moyen, path_image)
hors_plage = []  # items dont la moyenne n'est pas 0 et pas dans [TH_LO, TH_HI]
no_det = []      # fichiers sans détection (score=0)

label_files = sorted(glob.glob(str(PRED_DIR / "*.txt")))
print(f"[INFO] Fichiers de labels prédits trouvés: {len(label_files)}")

for lbl in label_files:
    with open(lbl, "r", encoding="utf-8", errors="ignore") as f:
        confs = []
        for line in f:
            sp = line.split()
            # format attendu: class cx cy w h conf  -> len>=6 et conf = dernier champ
            if len(sp) >= 6:
                try:
                    confs.append(float(sp[-1]))
                except ValueError:
                    pass

    if confs:
        score = statistics.mean(confs)
    else:
        score = 0.0  # pas de détection

    stem = Path(lbl).stem.lower()
    img_path = pool_map.get(stem)

    if not img_path:
        # Peut arriver si l'image d'origine n'est plus dans le pool
        continue

    if score == 0.0:
        if INCLUDE_NO_DET:
            no_det.append((score, img_path))
        # sinon on ignore ces cas (trop durs)
    elif TH_LO <= score <= TH_HI:
        # zone d'incertitude souhaitée
        candidats.append((score, img_path))
    else:
        hors_plage.append((score, img_path))

# Trier par score croissant (plus bas = plus incertain)
candidats.sort(key=lambda x: x[0])

# Si on n'a pas assez de candidats entre 0.2 et 0.4,
# on complète avec les plus faibles hors_plage (mais >0 si possible)
if len(candidats) < N:
    # on prend d'abord les scores >0 mais <TH_LO (modérément difficiles)
    below = [(s, p) for (s, p) in hors_plage if 0.0 < s < TH_LO]
    below.sort(key=lambda x: x[0])
    supplement = N - len(candidats)
    candidats.extend(below[:supplement])

# Si toujours pas assez, et si activé, on ajoute les no_det=0.0 (très durs)
if len(candidats) < N and INCLUDE_NO_DET and no_det:
    no_det.sort(key=lambda x: x[0])  # c'est tout à 0.0, mais pour homogénéité
    supplement = N - len(candidats)
    candidats.extend(no_det[:supplement])

# Tronquer à N
candidats = candidats[:N]

# Copier
copied = 0
for score, img in candidats:
    dst = OUT_DIR / img.name
    try:
        shutil.copy2(img, dst)
        copied += 1
    except Exception as e:
        print(f"[WARN] Copie échouée pour {img}: {e}")

print(f"[RESUME] Sélectionnés (zone {TH_LO}-{TH_HI}) : {len(candidats)} demandés = {N}")
print(f"[RESUME] Images copiées vers {OUT_DIR}: {copied}")
print("[TIP] Tu peux ajuster N/TH_LO/TH_HI en variables d'environnement, ex.:")
print('     $env:SELECT_N=60; $env:TH_LO=0.15; $env:TH_HI=0.35; python select_uncertain.py')
