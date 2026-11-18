# Detector de guantes de seguridad con Haar Cascade y OpenCV

**AVPF-3 – Visión Computacional aplicada a Ingeniería Civil (UTP)**
**Autora:** Elizabeth Asteria Nuñez Chunga

**Video demostrativo:** https://drive.google.com/file/d/1LjWgjH2CuHMgAYG1RiVK12LKAWSIwrT0/view?usp=sharing

Detector en tiempo real de **guantes de seguridad** (EPP de protección de manos en obra) con Python, OpenCV y un
clasificador **Haar Cascade** entrenado con **Cascade Trainer GUI** a partir de un dataset propio.

## Contenido del repositorio

```
.
├── APF3_ElizabethNunez.py          # Detector en tiempo real (webcam / DroidCam) + modo evaluación
├── cascade.xml                     # Modelo entrenado (20 etapas, ventana 50 x 20)
├── captura/
│   ├── Detector_Guantes_p.py       # Captura de positivas (recorte 50 x 20)
│   └── Detector_Guantes_n.py       # Captura de negativas (con modo automático)
├── p/                              # Dataset positivo (169 imágenes)   <- copiar aquí si el tamaño lo permite
├── n/                              # Dataset negativo (305 imágenes)   <- copiar aquí si el tamaño lo permite
├── evidencia_dataset/              # Montaje de las positivas, pos.lst y neg.lst
├── resultados/                     # Capturas de resultados
├── Informe_Tecnico_APF3_ElizabethNunez.pdf
└── requirements.txt
```

## Requisitos
- Python 3.x y `pip install -r requirements.txt`
- Cámara: app **DroidCam** en el celular (misma red WiFi que la PC) o webcam de la PC.
- Usa una carpeta de proyecto con ruta simple (sin tildes ni paréntesis), por ejemplo `C:\APF3_Guantes`.

## Uso
1. En `APF3_ElizabethNunez.py`, cambia `IP_CELULAR` por la IP que muestra DroidCam (si no conecta, usa la webcam).
2. Ejecuta: `python APF3_ElizabethNunez.py`
3. Teclas: `R` girar · `F` pantalla completa · `+`/`-` sensibilidad (`minNeighbors`) ·
   `C`/`P`/`N` anotar acierto / falso positivo / falso negativo · `G` guardar captura · `ESC` salir.
   Al salir se guarda un resumen en `resultados/resumen_evaluacion.txt`.

## Cómo se construyó el modelo
1. **Positivas:** `captura/Detector_Guantes_p.py` guarda recortes de 50 × 20 px con el guante.
2. **Negativas:** `captura/Detector_Guantes_n.py` guarda cuadros completos (320 px de ancho) sin el guante.
3. **Muestras:** las positivas se empaquetan en `pos_samples.vec`.
4. **Entrenamiento** en Cascade Trainer GUI → `cascade.xml`.

| Parámetro | Valor |
|---|---|
| Características | HAAR (BASIC) |
| Ventana | 50 × 20 px |
| Etapas | 20 |
| Boosting | GAB (Gentle AdaBoost) |
| minHitRate / maxFalseAlarm | 0.995 / 0.5 |
| weightTrimRate / maxDepth / maxWeakCount | 0.95 / 1 / 100 |
| Clasificadores débiles | 104 en total |

## Parámetros de detección
`scaleFactor = 1.05` · `minNeighbors = 3` · `minSize = (50, 20)` · ancho de proceso 960 px · hasta 2 detecciones por cuadro.

## Resultados (resumen)
- 153 de 169 muestras positivas de entrenamiento (90.5 %) atraviesan las 20 etapas del modelo.
- En tiempo real con DroidCam: 28–42 FPS y detección correcta en las capturas de `resultados/`.
- El rectángulo resulta ~1.6–2.4 veces más ancho que el guante (los recortes de entrenamiento incluyen fondo).

El análisis completo está en el informe técnico en PDF.

## Limitaciones y mejoras
- Dataset con poca variedad de fondos, distancias e iluminación y sin guantes puestos en la mano.
- Conviene volver a capturar positivas con recuadro más ajustado y más variedad, y ampliar las negativas con manos sin guantes.
- Falta un conjunto de prueba independiente para medir precisión y sensibilidad con más datos.

## Aplicación en Ingeniería Civil
Verificación automática de EPP (protección de manos) en obra, registro de cumplimiento para seguridad y salud en el
trabajo, y base para ampliar a casco, chaleco o lentes.
