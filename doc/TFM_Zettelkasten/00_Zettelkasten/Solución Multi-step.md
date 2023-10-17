Fecha: 2023-10-17 17:54  
Status: #Idea

# Solución Multi-step

La solución multi-step consiste en dividir la predicción en dos pasos:

1. Determinar la [[Región de interés|región de interés]] por medio de [[Métrica Intersection Over Unit|la métrica IoU]] en las clases: *tumor/no tumor*.
2. Determinar si la(s) región(es) de interés es de la clase *benigna/maligna*.

De esta forma la salida del primer paso es la entrada del segundo y se puede evaluar tal que:

```pseudocode
output1, output2 = Model(image)
loss1 = 1 - IoU(output1, groud_truth)
if loss1 < threshold:
	loss2 = loss1*crossentropy_loss(output2, value)
else:
	loss2 = None
```

Con lo que se deberá obtener un valor umbral para la pérdida asociada a la región de interés para que la segunda pérdida tenga efecto.

## Ventajas

1. Se puede introducir como salida de cualquier modelo.
2. Se trata de una forma de entrenamiento que puede ser beneficiosa dado el [[Sesgo de los datos|sesgo]].

## Desventajas

1. Se deben extraer dos salidas del modelo y manejar bien el paso de información de una a otra.
2. La implementación de la función de pérdida no es trivial.
3. El entrenamiento puede alargarse más tiempo que de forma convencional.

---
