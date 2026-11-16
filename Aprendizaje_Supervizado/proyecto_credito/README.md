# Aprobación de créditos con aprendizaje supervisado

Actividad 3.2.1 · Fundamentos de Inteligencia Artificial · Universidad Politécnica de Atlautla

Clasifica solicitudes de crédito (impago / cumple) con KNN, SVM y Random Forest, y las evalúa en una página web.

> **Los datos son sintéticos** (generados por `credito_entrenar.py`, estructura inspirada en Statlog German Credit, UCI).
> Los resultados no representan el desempeño con clientes reales. Para usar datos reales: `python credito_entrenar.py --csv mis_datos.csv`
> (columnas: edad, ingreso_mensual, monto_credito, plazo_meses, antiguedad_empleo, historial_crediticio, creditos_previos,
> retrasos_previos, ahorros, vivienda_propia, dependientes e `impago` con 1 = impago).

## Archivos
| Archivo | Para qué sirve |
|---|---|
| `credito_entrenar.py` | Genera los datos, hace EDA, entrena y compara los 3 modelos, elige umbral por costo y exporta `modelo_credito.json` |
| `predictor.js` | Reproduce los 3 modelos en JavaScript (sin dependencias) |
| `pagina.html`, `construir_web.py` | Plantilla de la página y script que la une con el JSON y el predictor en `index.html` |
| `index.html` | Página web autocontenida: ábrela con doble clic o publícala en GitHub Pages |
| `validar_js.js` | Compara JavaScript con Python en las 200 solicitudes de prueba |
| `datos_credito_sintetico.csv`, `tabla_comparativa.csv`, `figuras/` | Datos, resultados y gráficas |

## Uso
```bash
pip install -r requirements.txt
python credito_entrenar.py     # entrena y exporta modelo_credito.json
node validar_js.js             # comprueba que JavaScript coincide con Python
python construir_web.py        # regenera index.html
```

## Resultados (prueba de 200 solicitudes, umbral por costo 3:1)
| Modelo | Recall | Precision | AUC-ROC | Costo/solic. |
|---|---|---|---|---|
| KNN | 0.750 | 0.395 | 0.736 | 0.570 |
| **SVM (ganador)** | **0.867** | 0.416 | **0.782** | **0.485** |
| Random Forest | 0.783 | 0.427 | 0.758 | 0.510 |

Referencias de costo: aprobar a todos 0.900; rechazar a todos 0.700.
La relación de costos (aprobar a un mal pagador cuesta 3 veces más que rechazar a un buen cliente) es un supuesto académico.
