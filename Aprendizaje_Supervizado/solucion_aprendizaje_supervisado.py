# -*- coding: utf-8 -*-
"""
Actividad 3.2.1 · Aprendizaje automático supervisado
Clasificador de tumores de mama (benigno / maligno) con KNN, SVM y Random Forest.

Dataset : Breast Cancer Wisconsin (Diagnostic), UCI. Se carga desde scikit-learn,
          por lo que NO requiere descargas ni archivos externos.
Ejecutar: python solucion_aprendizaje_supervisado.py
          (en Google Colab basta con pegar el código en una celda)
Salidas : carpeta ./salida con figuras, tabla_comparativa.csv y modelo.pkl
"""
import os
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")                       # permite ejecutar sin pantalla
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
                             classification_report, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
SEED = 42                                   # semilla fija: resultados reproducibles
SALIDA = "salida"
os.makedirs(SALIDA, exist_ok=True)
COLORES = ["#4c9f70", "#c0392b"]            # verde = benigno, rojo = maligno
NOMBRES = {0: "Benigno", 1: "Maligno"}


# ======================================================================
# ETAPA 1. PROBLEMA Y DATOS
# ======================================================================
print("=" * 70, "\nETAPA 1. Problema y datos\n" + "=" * 70)
datos = load_breast_cancer(as_frame=True)
df = datos.frame.copy()
df["target"] = 1 - df["target"]             # en el original 0 = maligno; aquí 1 = maligno
FEATURES = list(datos.feature_names)

print(f"Registros: {df.shape[0]} | Variables de entrada: {len(FEATURES)}")
print("Tipo de tarea: clasificación binaria (1 = maligno, 0 = benigno)")
print(f"Valores nulos: {int(df.isnull().sum().sum())} | Duplicados: {int(df.duplicated().sum())}")
print(df["target"].map(NOMBRES).value_counts().to_string(), "\n")


# ======================================================================
# ETAPA 2. ANÁLISIS EXPLORATORIO Y PREPARACIÓN
# ======================================================================
print("=" * 70, "\nETAPA 2. EDA y preparación\n" + "=" * 70)
etiqueta = df["target"].map(NOMBRES)
medias = [f for f in FEATURES if f.startswith("mean ")]

# 2.1 Distribución del objetivo
plt.figure(figsize=(4, 3.5))
sns.countplot(x=etiqueta, order=list(NOMBRES.values()), hue=etiqueta,
              hue_order=list(NOMBRES.values()), palette=COLORES, legend=False)
plt.title("Distribución de la variable objetivo"); plt.xlabel("")
plt.tight_layout(); plt.savefig(f"{SALIDA}/01_distribucion_target.png", dpi=150); plt.close()

# 2.2 Histogramas por clase
g = sns.FacetGrid(df[medias[:6] + ["target"]].melt(id_vars="target"), col="variable",
                  col_wrap=3, sharex=False, sharey=False, height=2.6)
g.map_dataframe(sns.histplot, x="value", hue="target", palette=COLORES, bins=25)
g.savefig(f"{SALIDA}/02_histogramas.png", dpi=150); plt.close("all")

# 2.3 Valores atípicos
plt.figure(figsize=(7, 5))
sns.boxplot(data=df[medias[:6]].apply(lambda s: (s - s.mean()) / s.std()), orient="h")
plt.title("Valores atípicos (variables estandarizadas)")
plt.tight_layout(); plt.savefig(f"{SALIDA}/03_boxplots_outliers.png", dpi=150); plt.close()

# 2.4 Correlaciones
plt.figure(figsize=(8, 6.5))
sns.heatmap(df[medias + ["target"]].corr(), cmap="coolwarm", center=0, annot=True,
            fmt=".1f", annot_kws={"size": 6})
plt.title("Mapa de correlación"); plt.tight_layout()
plt.savefig(f"{SALIDA}/04_correlacion.png", dpi=150); plt.close()

# 2.5 División estratificada 80/20. El escalado va DENTRO del Pipeline (más abajo),
#     así se ajusta solo con datos de entrenamiento y no hay fuga de información.
X, y = df[FEATURES], df["target"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
print(f"Entrenamiento: {len(X_tr)} | Prueba: {len(X_te)} (estratificado, semilla {SEED})")
print("Sin nulos ni duplicados: no se requiere imputación. Se conservan los outliers (mediciones reales).\n")


# ======================================================================
# ETAPA 3. ENTRENAMIENTO Y COMPARACIÓN DE 3 MODELOS
# ======================================================================
print("=" * 70, "\nETAPA 3. Modelos y métricas\n" + "=" * 70)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
CONFIG = {
    "KNN": (KNeighborsClassifier(),
            {"clf__n_neighbors": [3, 5, 7, 9, 11], "clf__weights": ["uniform", "distance"]}),
    "SVM": (SVC(probability=True, random_state=SEED),
            {"clf__C": [0.1, 1, 10, 100], "clf__gamma": ["scale", 0.01, 0.001],
             "clf__kernel": ["rbf", "linear"]}),
    "Random Forest": (RandomForestClassifier(random_state=SEED),
                      {"clf__n_estimators": [100, 300], "clf__max_depth": [None, 5, 10],
                       "clf__min_samples_leaf": [1, 3]}),
}

filas, mejores = [], {}
fig_roc, ax_roc = plt.subplots(figsize=(5.5, 5))
for nombre, (clasificador, rejilla) in CONFIG.items():
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clasificador)])

    acc_base = accuracy_score(y_te, pipe.fit(X_tr, y_tr).predict(X_te))   # línea base
    gs = GridSearchCV(pipe, rejilla, cv=cv, scoring="recall", n_jobs=-1).fit(X_tr, y_tr)
    modelo = gs.best_estimator_                                            # optimizado
    mejores[nombre] = modelo

    pred = modelo.predict(X_te)
    proba = modelo.predict_proba(X_te)[:, 1]
    cv_f1 = cross_val_score(modelo, X, y, cv=cv, scoring="f1")             # estabilidad
    cv_rec = cross_val_score(modelo, X, y, cv=cv, scoring="recall")

    filas.append({
        "Modelo": nombre,
        "Accuracy (base)": acc_base,
        "Accuracy": accuracy_score(y_te, pred),
        "Precision": precision_score(y_te, pred),
        "Recall": recall_score(y_te, pred),
        "F1": f1_score(y_te, pred),
        "AUC-ROC": roc_auc_score(y_te, proba),
        "Acc. train": accuracy_score(y_tr, modelo.predict(X_tr)),
        "Recall CV (media ± desv.)": f"{cv_rec.mean():.3f} ± {cv_rec.std():.3f}",
        "F1 CV (media ± desv.)": f"{cv_f1.mean():.3f} ± {cv_f1.std():.3f}",
        "Hiperparámetros": {k.replace("clf__", ""): v for k, v in gs.best_params_.items()},
    })

    print(f"--- {nombre} --- mejores hiperparámetros: {filas[-1]['Hiperparámetros']}")
    print(classification_report(y_te, pred, target_names=list(NOMBRES.values()), digits=3))

    ConfusionMatrixDisplay.from_predictions(y_te, pred, display_labels=list(NOMBRES.values()),
                                            cmap="Blues")
    plt.title(f"Matriz de confusión — {nombre}"); plt.tight_layout()
    plt.savefig(f"{SALIDA}/cm_{nombre.replace(' ', '_')}.png", dpi=150); plt.close()
    RocCurveDisplay.from_predictions(y_te, proba, name=nombre, ax=ax_roc)

ax_roc.plot([0, 1], [0, 1], "k--", lw=0.8)
ax_roc.set_title("Curvas ROC (conjunto de prueba)")
fig_roc.tight_layout(); fig_roc.savefig(f"{SALIDA}/05_curvas_roc.png", dpi=150); plt.close(fig_roc)

tabla = pd.DataFrame(filas)
tabla["Brecha train-test"] = tabla["Acc. train"] - tabla["Accuracy"]       # sobreajuste
cols_num = ["Accuracy (base)", "Accuracy", "Precision", "Recall", "F1", "AUC-ROC",
            "Acc. train", "Brecha train-test"]
tabla[cols_num] = tabla[cols_num].round(4)
tabla.to_csv(f"{SALIDA}/tabla_comparativa.csv", index=False)

print("TABLA COMPARATIVA (conjunto de prueba)")
print(tabla.drop(columns="Hiperparámetros").to_string(index=False), "\n")

# Modelo ganador: mayor Recall (un falso negativo es lo más costoso); desempate por F1 y AUC
ganador = tabla.sort_values(["Recall", "F1", "AUC-ROC"], ascending=False).iloc[0]["Modelo"]
print(f"MODELO GANADOR: {ganador}")
print("Criterio: mayor recall en la clase maligna; desempate por F1 y AUC-ROC.")
print("Nota: con 42 casos malignos en prueba, diferencias de 1 caso no son concluyentes;")
print("      por eso se reporta también la validación cruzada (columnas 'CV').\n")
joblib.dump(mejores[ganador], f"{SALIDA}/modelo.pkl")


# ======================================================================
# ETAPA 4. PREDICCIÓN CON EL MODELO GUARDADO
# (para la interfaz web ver app.py con Streamlit; aquí se muestra el uso)
# ======================================================================
def predecir(valores: dict, ruta_modelo: str = f"{SALIDA}/modelo.pkl") -> dict:
    """Recibe un dict {nombre_variable: valor} con las 30 mediciones y devuelve la clase."""
    modelo = joblib.load(ruta_modelo)
    entrada = pd.DataFrame([valores])[FEATURES]
    p_maligno = float(modelo.predict_proba(entrada)[0, 1])
    return {"clase": NOMBRES[int(p_maligno >= 0.5)], "prob_maligno": round(p_maligno, 4)}


print("=" * 70, "\nETAPA 4. Ejemplo de predicción\n" + "=" * 70)
for etiqueta_real in (0, 1):
    caso = X[y == etiqueta_real].iloc[0].to_dict()
    print(f"Caso real {NOMBRES[etiqueta_real]:8s} -> {predecir(caso)}")
print(f"\nArchivos generados en ./{SALIDA}/")
