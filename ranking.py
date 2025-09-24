
import logging

logger = logging.getLogger("ranking_debug")
logging.basicConfig(level=logging.DEBUG)

def analizar_top3(top50, factores):
    logger.debug("Iniciando análisis de top3")
    if not factores:
        logger.warning("No hay factores configurados para seleccionar top3")
        return [], "sin_top3_factors"

    candidatos = []
    for item in top50:
        try:
            puntaje = sum(item.get(f, 0) for f in factores)
            candidatos.append((puntaje, item))
        except Exception as e:
            logger.error(f"Error calculando puntaje para {item.get('code')}: {e}")
    candidatos.sort(reverse=True, key=lambda x: x[0])

    top3 = [c[1] for c in candidatos[:3]]
    if not top3:
        logger.warning("No se logró seleccionar ningún top3")
    else:
        logger.debug(f"Top3 seleccionado: {[i.get('code') for i in top3]}")
    return top3, None
