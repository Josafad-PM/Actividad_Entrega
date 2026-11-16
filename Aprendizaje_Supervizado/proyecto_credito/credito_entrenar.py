# -*- coding: utf-8 -*-
"""
Aprobar o rechazar créditos según el historial del solicitante
Actividad 3.2.1 · Aprendizaje automático supervisado · KNN, SVM y Random Forest

DATOS: no se pudo descargar un dataset real, por lo que este script GENERA un dataset
SINTÉTICO de 1,000 solicitudes (estructura inspirada en German Credit, UCI). Las
relaciones entre variables las define el propio generador: los resultados NO
representan el desempeño con datos reales.
Para usar datos reales:  python credito_entrenar.py --csv mis_datos.csv
(el CSV debe tener las columnas de COLUMNAS_ENTRADA y la columna 'impago': 1 = impago).

Salidas (carpeta actual): datos_credito_sintetico.csv, modelo_credito.json (lo usa la
página web), prueba_validacion.json, tabla_comparativa.csv, modelo.pkl y figuras/*.png
"""
import json
import sys
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score,
                             fbeta_score, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold, cross_val_predict,
                                     cross_val_score, train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
SEED = 42
OK, MAL = "#2E8B6A", "#C0392B"          # verde = cumple, rojo = impago
COLUMNAS_ENTRADA = ["edad", "ingreso_mensual", "monto_credito", "plazo_meses", "antiguedad_empleo",
                    "historial_crediticio", "creditos_previos", "retrasos_previos", "ahorros",
                    "vivienda_propia", "dependientes"]
FEATURES = COLUMNAS_ENTRADA + ["cuota_ingreso"]
ETIQUETAS = {"edad": "Edad", "ingreso_mensual": "Ingreso mensual", "monto_credito": "Monto solicitado",
             "plazo_meses": "Plazo", "antiguedad_empleo": "Antigüedad en el empleo",
             "historial_crediticio": "Historial crediticio", "creditos_previos": "Créditos previos",
             "retrasos_previos": "Pagos atrasados (2 años)", "ahorros": "Ahorros",
             "vivienda_propia": "Vivienda propia", "dependientes": "Dependientes económicos",
             "cuota_ingreso": "Cuota / ingreso"}


# ---------------------------------------------------------------- datos
def generar_datos(n=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    edad = np.clip(rng.normal(37, 11, n), 21, 70).round()
    ingreso = np.clip(rng.lognormal(np.log(16000), 0.5, n), 5000, 120000).round(-2)
    plazo = rng.choice([6, 12, 18, 24, 36, 48, 60], n, p=[.08, .22, .12, .28, .15, .08, .07])
    monto = np.clip(rng.uniform(0.05, 0.55, n) * ingreso * plazo, 3000, 400000).round(-2)
    antig = np.clip((edad - 20) * rng.uniform(0, 0.8, n), 0, 35).round(1)
    hist = rng.choice([0, 1, 2], n, p=[.2, .3, .5])                 # 0 malo, 1 regular, 2 bueno
    prev = np.clip(rng.poisson(1.5, n), 0, 6)
    retr = np.clip(rng.poisson(np.where(hist == 0, 3.0, np.where(hist == 1, 1.2, 0.3))), 0, 8)
    ahorros = (np.clip(rng.lognormal(np.log(15000), 1.0, n), 0, 250000) * rng.binomial(1, .85, n)).round(-2)
    vivienda = rng.binomial(1, np.where(edad > 40, 0.6, 0.3))
    dep = np.clip(rng.poisson(1.1, n), 0, 5)
    cuota = monto / plazo / ingreso
    base = (4.0 * cuota + 1.0 * (hist == 0) - 0.8 * (hist == 2) + 0.30 * retr
            - 0.5 * np.log1p(ahorros / 5000) - 0.04 * antig - 0.35 * vivienda + 0.12 * dep
            + 0.15 * (plazo / 12) + rng.normal(0, 0.9, n))
    lo, hi = -10.0, 10.0                                            # calibra a ~30 % de impago
    for _ in range(60):
        b = (lo + hi) / 2
        lo, hi = (b, hi) if (1 / (1 + np.exp(-(base + b)))).mean() < 0.30 else (lo, b)
    impago = (rng.random(n) < 1 / (1 + np.exp(-(base + b)))).astype(int)
    df = pd.DataFrame({"edad": edad, "ingreso_mensual": ingreso, "monto_credito": monto,
                       "plazo_meses": plazo, "antiguedad_empleo": antig, "historial_crediticio": hist,
                       "creditos_previos": prev, "retrasos_previos": retr, "ahorros": ahorros,
                       "vivienda_propia": vivienda, "dependientes": dep, "impago": impago})
    df.loc[rng.random(n) < 0.06, "ahorros"] = np.nan                # valores faltantes
    return df


def agregar_cuota(df):
    df = df.copy()
    df["cuota_ingreso"] = df["monto_credito"] / df["plazo_meses"] / df["ingreso_mensual"]
    return df


if "--csv" in sys.argv:
    df = pd.read_csv(sys.argv[sys.argv.index("--csv") + 1])
    ORIGEN = "real (CSV del equipo)"
else:
    df = generar_datos()
    df.to_csv("datos_credito_sintetico.csv", index=False)
    ORIGEN = "sintético"
df = agregar_cuota(df)
print(f"Datos {ORIGEN}: {len(df)} registros | impago: {df['impago'].mean():.1%} | "
      f"nulos: {int(df[FEATURES].isnull().sum().sum())} | duplicados: {int(df.duplicated().sum())}")

# ---------------------------------------------------------------- EDA
eda_g = df.groupby("historial_crediticio")["impago"].agg(["count", "mean"])
retr_bins = df["retrasos_previos"].clip(upper=3).astype(int)
eda_r = df.groupby(retr_bins)["impago"].agg(["count", "mean"])
bins = np.arange(0, 0.65, 0.05)
cu = df["cuota_ingreso"].clip(upper=0.6 - 1e-9)
h_ok = np.histogram(cu[df.impago == 0], bins)[0].tolist()
h_mal = np.histogram(cu[df.impago == 1], bins)[0].tolist()
corr = df[FEATURES + ["impago"]].corr()["impago"].drop("impago").sort_values()

fig, ax = plt.subplots(figsize=(4.2, 3.5))
sns.countplot(x=df["impago"].map({0: "Cumple", 1: "Impago"}), order=["Cumple", "Impago"],
              hue=df["impago"].map({0: "Cumple", 1: "Impago"}), hue_order=["Cumple", "Impago"],
              palette=[OK, MAL], legend=False, ax=ax)
ax.set_title("Distribución de la variable objetivo"); ax.set_xlabel(""); ax.set_ylabel("Solicitudes")
fig.tight_layout(); fig.savefig("figuras/01_distribucion_target.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(5, 3.5))
ax.bar(["Malo", "Regular", "Bueno"], eda_g["mean"] * 100, color=["#C0392B", "#D9A441", "#2E8B6A"])
for i, v in enumerate(eda_g["mean"] * 100):
    ax.text(i, v + 1, f"{v:.0f} %", ha="center")
ax.set_ylim(0, eda_g["mean"].max() * 100 * 1.15); ax.set_ylabel("Tasa de impago (%)"); ax.set_title("Impago según el historial crediticio")
fig.tight_layout(); fig.savefig("figuras/02_impago_por_historial.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 3.5))
x = bins[:-1] + 0.025
ax.bar(x - 0.011, h_ok, width=0.022, color=OK, label="Cumple")
ax.bar(x + 0.011, h_mal, width=0.022, color=MAL, label="Impago")
ax.set_xlabel("Cuota mensual / ingreso mensual"); ax.set_ylabel("Solicitudes")
ax.set_title("Carga de la cuota según el resultado"); ax.legend()
fig.tight_layout(); fig.savefig("figuras/03_hist_cuota_ingreso.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 4.2))
ax.barh([ETIQUETAS[c] for c in corr.index], corr.values, color=[MAL if v > 0 else OK for v in corr.values])
ax.axvline(0, color="#555", lw=0.8); ax.set_xlabel("Correlación con el impago")
ax.set_title("Variables más asociadas al impago")
fig.tight_layout(); fig.savefig("figuras/04_correlacion.png", dpi=150); plt.close(fig)

# ---------------------------------------------------------------- división
X, y = df[FEATURES], df["impago"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
print(f"Entrenamiento: {len(X_tr)} | Prueba: {len(X_te)}")


def pipe_de(clf):
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("clf", clf)])


class SVMPlatt:
    """SVM con calibración de Platt propia (probabilidad = sigmoide(a·f + b)); es reproducible en JS."""
    def __init__(self, pipe, a, b): self.pipe, self.a, self.b = pipe, a, b
    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-(self.a * self.pipe.decision_function(X) + self.b)))
        return np.column_stack([1 - p, p])


CONFIG = {
    "KNN": (KNeighborsClassifier(), {"clf__n_neighbors": [5, 9, 15, 21, 31], "clf__weights": ["uniform", "distance"]}),
    "SVM": (SVC(kernel="rbf"), {"clf__C": [0.1, 1, 10], "clf__gamma": [0.01, 0.03, 0.1]}),
    "Random Forest": (RandomForestClassifier(random_state=SEED),
                      {"clf__n_estimators": [100, 150], "clf__max_depth": [4, 6, 8], "clf__min_samples_leaf": [3, 5, 10]}),
}
UMBRALES = np.arange(0.05, 0.96, 0.01)
COSTO_FN = 3          # aprobar a un mal pagador cuesta 3 veces más que rechazar a un buen cliente


def costo(y_real, pred):
    fn = int(((pred == 0) & (y_real == 1)).sum()); fp = int(((pred == 1) & (y_real == 0)).sum())
    return (COSTO_FN * fn + fp) / len(y_real)

modelos, metricas, rocs = {}, {}, {}
for nombre, (clf, rejilla) in CONFIG.items():
    gs = GridSearchCV(pipe_de(clf), rejilla, cv=cv, scoring="roc_auc", n_jobs=-1).fit(X_tr, y_tr)
    pipe = gs.best_estimator_
    if nombre == "SVM":
        f_oof = cross_val_predict(pipe, X_tr, y_tr, cv=cv, method="decision_function")
        pl = LogisticRegression(C=1e6).fit(f_oof.reshape(-1, 1), y_tr)
        modelo = SVMPlatt(pipe, float(pl.coef_[0, 0]), float(pl.intercept_[0]))
        p_oof = modelo.predict_proba  # noqa
        p_oof = 1 / (1 + np.exp(-(modelo.a * f_oof + modelo.b)))
    else:
        modelo = pipe
        p_oof = cross_val_predict(pipe, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
    # umbral de decisión: minimiza el costo esperado (validación cruzada sobre entrenamiento)
    umbral = float(UMBRALES[np.argmin([costo(y_tr.values, (p_oof >= t).astype(int)) for t in UMBRALES])])
    proba = modelo.predict_proba(X_te)[:, 1]
    pred = (proba >= umbral).astype(int)
    p_tr = (modelo.predict_proba(X_tr)[:, 1] >= umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    fpr, tpr, _ = roc_curve(y_te, proba)
    idx = np.unique(np.linspace(0, len(fpr) - 1, 60).astype(int))
    auc_cv = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
    modelos[nombre] = modelo
    rocs[nombre] = (fpr, tpr, roc_auc_score(y_te, proba))
    metricas[nombre] = {
        "accuracy": accuracy_score(y_te, pred), "precision": precision_score(y_te, pred),
        "recall": recall_score(y_te, pred), "f1": f1_score(y_te, pred),
        "costo": costo(y_te.values, pred), "auc": roc_auc_score(y_te, proba), "umbral": umbral,
        "brecha": accuracy_score(y_tr, p_tr) - accuracy_score(y_te, pred),
        "auc_cv": [float(auc_cv.mean()), float(auc_cv.std())],
        "hiperparametros": {k.replace("clf__", ""): (v if isinstance(v, str) else float(v) if isinstance(v, float) else int(v))
                            for k, v in gs.best_params_.items()},
        "cm": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "roc": {"fpr": fpr[idx].round(4).tolist(), "tpr": tpr[idx].round(4).tolist()},
    }
    m = metricas[nombre]
    print(f"{nombre:14s} umbral={umbral:.2f} acc={m['accuracy']:.3f} prec={m['precision']:.3f} "
          f"rec={m['recall']:.3f} F1={m['f1']:.3f} costo={m['costo']:.3f} AUC={m['auc']:.3f} "
          f"AUC-CV={auc_cv.mean():.3f}±{auc_cv.std():.3f} {m['hiperparametros']}")

ganador = min(metricas, key=lambda k: (round(metricas[k]["costo"], 4), -metricas[k]["auc"]))
base_aprobar_todos = costo(y_te.values, np.zeros(len(y_te), dtype=int))
base_rechazar_todos = costo(y_te.values, np.ones(len(y_te), dtype=int))
print(f"Costo por solicitud: aprobar a todos = {base_aprobar_todos:.3f} | rechazar a todos = {base_rechazar_todos:.3f}")
print("MODELO GANADOR:", ganador)

tabla = pd.DataFrame([{"Modelo": k, **{kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in m.items()
                                       if kk in ("umbral", "accuracy", "precision", "recall", "f1", "costo", "auc", "brecha")},
                       "AUC CV (media ± desv.)": f"{m['auc_cv'][0]:.3f} ± {m['auc_cv'][1]:.3f}"}
                      for k, m in metricas.items()])
tabla.to_csv("tabla_comparativa.csv", index=False)

# análisis de umbrales del ganador
pg = modelos[ganador].predict_proba(X_te)[:, 1]
tabla_umbral = []
for t in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6]:
    pr = (pg >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, pr).ravel()
    tabla_umbral.append({"umbral": t, "precision": float(precision_score(y_te, pr, zero_division=0)),
                         "recall": float(recall_score(y_te, pr)), "rechazados": float(pr.mean()),
                         "malos_aprobados": int(fn), "buenos_rechazados": int(fp), "costo": costo(y_te.values, pr)})

# figuras de resultados
fig, ax = plt.subplots(figsize=(5.5, 5))
for k, (f, t, a) in rocs.items():
    ax.plot(f, t, label=f"{k} (AUC = {a:.2f})")
ax.plot([0, 1], [0, 1], "k--", lw=0.8); ax.set_xlabel("Tasa de falsos positivos"); ax.set_ylabel("Tasa de verdaderos positivos")
ax.set_title("Curvas ROC (conjunto de prueba)"); ax.legend(loc="lower right")
fig.tight_layout(); fig.savefig("figuras/05_curvas_roc.png", dpi=150); plt.close(fig)
for k in modelos:
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    ConfusionMatrixDisplay(np.array(metricas[k]["cm"]), display_labels=["Cumple", "Impago"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Matriz de confusión — {k}"); fig.tight_layout()
    fig.savefig(f"figuras/cm_{k.replace(' ', '_')}.png", dpi=150); plt.close(fig)
joblib.dump(modelos[ganador], "modelo.pkl")

# ---------------------------------------------------------------- exportación para la página web
imp = modelos["KNN"].named_steps["imp"]
sc = modelos["KNN"].named_steps["sc"]
Xs_tr = modelos["KNN"][:-1].transform(X_tr)
knn = modelos["KNN"].named_steps["clf"]
svm_pipe = modelos["SVM"].pipe
svc = svm_pipe.named_steps["clf"]
rf = modelos["Random Forest"].named_steps["clf"]
# la imputación y el escalado son los mismos para los tres modelos (se ajustan solo con train)
r = lambda a, d=4: np.round(np.asarray(a, dtype=float), d).tolist()
assert np.allclose(svm_pipe.named_steps["sc"].mean_, sc.mean_)
arboles = []
for est in rf.estimators_:
    t = est.tree_
    v = t.value[:, 0, :]
    p1 = v[:, 1] / v.sum(axis=1)
    arboles.append({"l": t.children_left.tolist(), "r": t.children_right.tolist(),
                    "f": t.feature.tolist(), "t": r(np.where(t.feature >= 0, t.threshold, 0), 8),
                    "v": r(np.where(t.feature < 0, p1, 0), 4)})
entradas = [
    {"id": "edad", "min": 18, "max": 75, "step": 1, "def": 38},
    {"id": "ingreso_mensual", "min": 3000, "max": 150000, "step": 500, "def": 18000},
    {"id": "monto_credito", "min": 3000, "max": 400000, "step": 1000, "def": 60000},
    {"id": "plazo_meses", "min": 6, "max": 60, "step": 6, "def": 24},
    {"id": "antiguedad_empleo", "min": 0, "max": 40, "step": 0.5, "def": 5},
    {"id": "historial_crediticio", "tipo": "select", "opciones": ["Malo", "Regular", "Bueno"], "def": 2},
    {"id": "creditos_previos", "min": 0, "max": 10, "step": 1, "def": 1},
    {"id": "retrasos_previos", "min": 0, "max": 12, "step": 1, "def": 0},
    {"id": "ahorros", "min": 0, "max": 300000, "step": 1000, "def": 15000},
    {"id": "vivienda_propia", "tipo": "select", "opciones": ["No", "Sí"], "def": 0},
    {"id": "dependientes", "min": 0, "max": 8, "step": 1, "def": 1},
]
for e in entradas:
    e["etiqueta"] = ETIQUETAS[e["id"]]
ejemplos = {
    "Perfil sólido": dict(edad=42, ingreso_mensual=32000, monto_credito=60000, plazo_meses=24, antiguedad_empleo=12,
                          historial_crediticio=2, creditos_previos=2, retrasos_previos=0, ahorros=45000, vivienda_propia=1, dependientes=1),
    "Perfil de riesgo": dict(edad=24, ingreso_mensual=9000, monto_credito=90000, plazo_meses=24, antiguedad_empleo=0.5,
                             historial_crediticio=0, creditos_previos=1, retrasos_previos=5, ahorros=0, vivienda_propia=0, dependientes=3),
    "Caso intermedio": dict(edad=31, ingreso_mensual=15000, monto_credito=70000, plazo_meses=24, antiguedad_empleo=4,
                            historial_crediticio=1, creditos_previos=1, retrasos_previos=1, ahorros=8000, vivienda_propia=0, dependientes=2),
}
export = {
    "meta": {"origen": ORIGEN, "n": len(df), "n_train": len(X_tr), "n_test": len(X_te), "seed": SEED,
             "tasa_impago": float(df["impago"].mean()), "costo_fn": COSTO_FN,
             "costo_aprobar_todos": base_aprobar_todos, "costo_rechazar_todos": base_rechazar_todos},
    "features": FEATURES, "etiquetas": ETIQUETAS, "entradas": entradas, "ejemplos": ejemplos,
    "imputer": r(imp.statistics_, 10), "scaler": {"mean": r(sc.mean_, 12), "scale": r(sc.scale_, 12)},
    "modelos": {
        "KNN": {"k": int(knn.n_neighbors), "weights": knn.weights, "X": r(Xs_tr, 4), "y": y_tr.astype(int).tolist()},
        "SVM": {"gamma": float(svc._gamma), "sv": r(svc.support_vectors_, 4), "coef": r(svc.dual_coef_[0], 6),
                "b": float(svc.intercept_[0]), "platt": [modelos["SVM"].a, modelos["SVM"].b]},
        "Random Forest": {"trees": arboles},
    },
    "metricas": metricas, "ganador": ganador, "umbrales": tabla_umbral,
    "eda": {"historial": {"n": eda_g["count"].tolist(), "tasa": r(eda_g["mean"], 4)},
            "retrasos": {"n": eda_r["count"].tolist(), "tasa": r(eda_r["mean"], 4)},
            "hist_cuota": {"bins": r(bins, 2), "cumple": h_ok, "impago": h_mal},
            "correlacion": {ETIQUETAS[k]: round(float(v), 3) for k, v in corr.items()},
            "stats": {ETIQUETAS[c]: [float(df[c].min()), float(df[c].mean()), float(df[c].max())] for c in FEATURES}},
}
json.dump(export, open("modelo_credito.json", "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))

# conjunto de prueba para validar que la implementación en JavaScript coincide con Python
raw_te = X_te.copy()
json.dump({"X": [[None if pd.isna(v) else float(v) for v in row] for row in raw_te[COLUMNAS_ENTRADA].values],
           "proba": {k: modelos[k].predict_proba(X_te)[:, 1].round(6).tolist() for k in modelos}},
          open("prueba_validacion.json", "w"))
import os
print(f"modelo_credito.json: {os.path.getsize('modelo_credito.json') / 1024:.0f} KB")


# ---------------------------------------------------------------- uso del modelo
def evaluar_solicitud(s: dict, umbral=None):
    fila = agregar_cuota(pd.DataFrame([s]))[FEATURES]
    p = float(modelos[ganador].predict_proba(fila)[0, 1])
    u = metricas[ganador]["umbral"] if umbral is None else umbral
    return {"prob_impago": round(p, 3), "decision": "RECHAZAR" if p >= u else "APROBAR"}


for nombre, caso in ejemplos.items():
    print(f"{nombre:16s} -> {evaluar_solicitud(caso)}")
