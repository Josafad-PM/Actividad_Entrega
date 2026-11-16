// Motor de predicción en JavaScript: reproduce los tres modelos entrenados en Python.
// Entrada: el JSON exportado por credito_entrenar.py. Sin dependencias.
(function (root) {
  const COLUMNAS = ["edad", "ingreso_mensual", "monto_credito", "plazo_meses", "antiguedad_empleo",
    "historial_crediticio", "creditos_previos", "retrasos_previos", "ahorros", "vivienda_propia", "dependientes"];

  // 1) valores crudos -> vector escalado (imputa faltantes con la mediana y calcula cuota/ingreso)
  function preparar(M, s) {
    return M.features.map((nombre, i) => {
      let v = nombre === "cuota_ingreso" ? s.monto_credito / s.plazo_meses / s.ingreso_mensual : s[nombre];
      if (v === null || v === undefined || Number.isNaN(v)) v = M.imputer[i];
      return (v - M.scaler.mean[i]) / M.scaler.scale[i];
    });
  }

  function knn(m, x) {
    const d = m.X.map((fila, j) => {
      let s = 0;
      for (let i = 0; i < x.length; i++) { const t = fila[i] - x[i]; s += t * t; }
      return [Math.sqrt(s), m.y[j]];
    }).sort((a, b) => a[0] - b[0]).slice(0, m.k);
    if (m.weights === "uniform") return d.reduce((a, [, y]) => a + y, 0) / m.k;
    const cero = d.filter(([dist]) => dist === 0);
    const lista = cero.length ? cero.map(([, y]) => [1, y]) : d.map(([dist, y]) => [1 / dist, y]);
    const sw = lista.reduce((a, [w]) => a + w, 0);
    return lista.reduce((a, [w, y]) => a + w * y, 0) / sw;
  }

  function svm(m, x) {
    let f = m.b;
    for (let j = 0; j < m.sv.length; j++) {
      let s = 0; const sv = m.sv[j];
      for (let i = 0; i < x.length; i++) { const t = sv[i] - x[i]; s += t * t; }
      f += m.coef[j] * Math.exp(-m.gamma * s);
    }
    return 1 / (1 + Math.exp(-(m.platt[0] * f + m.platt[1])));
  }

  function bosque(m, x0) {
    const x = x0.map(Math.fround);        // scikit-learn compara los árboles en float32
    let suma = 0;
    for (const t of m.trees) {
      let n = 0;
      while (t.l[n] !== -1) n = x[t.f[n]] <= t.t[n] ? t.l[n] : t.r[n];
      suma += t.v[n];
    }
    return suma / m.trees.length;
  }

  // 2) probabilidad de impago de cada modelo
  function predecir(M, solicitud) {
    const x = preparar(M, solicitud);
    return { "KNN": knn(M.modelos["KNN"], x), "SVM": svm(M.modelos["SVM"], x),
             "Random Forest": bosque(M.modelos["Random Forest"], x) };
  }

  const api = { COLUMNAS, preparar, predecir };
  if (typeof module !== "undefined") module.exports = api; else root.Predictor = api;
})(typeof window !== "undefined" ? window : globalThis);
