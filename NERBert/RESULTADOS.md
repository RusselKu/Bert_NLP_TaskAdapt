# Resultados NER - Jonathan

Se seleccionó full por su F1 de validación (95.11%). La diferencia observada respecto a partial fue 3.40 puntos porcentuales. La selección se registró antes de evaluar test. Se entrenó una sola semilla por método; no se realizaron pruebas de significancia estadística. Aunque la diferencia supera 3 puntos en esta ejecución, una sola semilla no caracteriza su variabilidad.

```csv
method,trainable_parameters,validation_f1,test_f1,test_precision,test_recall,test_accuracy,training_seconds,peak_vram_gib
partial,14182665,0.9171186440677965,0.8878828229027962,0.8903329179277194,0.8854461756373938,0.9774738882308603,115.62909130001208,0.7636408805847168
full,107726601,0.9511290866194809,0.9153532849300513,0.9155154091392136,0.9151912181303116,0.9828362226768601,358.5377167999977,2.4198708534240723
```

Ver results/ para el historial y la configuración exacta.
