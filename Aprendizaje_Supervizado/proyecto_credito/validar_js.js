const P = require("./predictor.js");
const M = require("./modelo_credito.json");
const V = require("./prueba_validacion.json");
for (const k of Object.keys(M.modelos)) {
  let maxd = 0, dif = 0;
  V.X.forEach((fila, i) => {
    const s = Object.fromEntries(P.COLUMNAS.map((c, j) => [c, fila[j]]));
    const p = P.predecir(M, s)[k];
    maxd = Math.max(maxd, Math.abs(p - V.proba[k][i]));
    const u = M.metricas[k].umbral;
    if ((p >= u) !== (V.proba[k][i] >= u)) dif++;
  });
  console.log(`${k.padEnd(14)} dif. máxima de probabilidad = ${maxd.toExponential(2)} | decisiones distintas: ${dif}/${V.X.length}`);
}
