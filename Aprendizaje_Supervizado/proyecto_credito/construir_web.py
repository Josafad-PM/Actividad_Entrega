"""Une pagina.html + predictor.js + modelo_credito.json en una sola página autocontenida."""
import json
pagina = open("pagina.html", encoding="utf-8").read()
js = open("predictor.js", encoding="utf-8").read()
modelo = open("modelo_credito.json", encoding="utf-8").read()
assert "</script" not in modelo
cuerpo = pagina.replace("__MODELO_JSON__", modelo).replace("__PREDICTOR_JS__", js)
open("pagina_publicar.html", "w", encoding="utf-8").write(cuerpo)           # versión para publicar (sin <html>)
open("index.html", "w", encoding="utf-8").write(
    '<!doctype html>\n<html lang="es">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n</head>\n<body>\n'
    + cuerpo + "\n</body>\n</html>\n")
print("ok", len(cuerpo) // 1024, "KB")
