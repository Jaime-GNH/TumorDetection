Fecha: 2023-11-06 20:22  
Status: #Idea

# Renormalización y percolación

Se puede hallar el [[Punto crítico de percolación]] de las imágenes a través del valor de la media del ratio de píxeles positivos y negativos de las mismas.

$$
\begin{equation}
Ratio = \frac{positives}{negatives}
\end{equation}
$$

^eq-Ratio-Positivos-Negativos



Para distintas dimensiones se realizará lo siguiente para cada imagen:

1. Redimensionamiento a imagen cuadrada
2. Cálculo del [[Renormalización y percolación#^eq-Ratio-Positivos-Negativos|ratio inicial]].
3. División de la imagen en $N$ sub-imágenes de misma dimensión.
4. Redimensionamiento al tamaño original.
5. Cálculo del [[Renormalización y percolación#^eq-Ratio-Positivos-Negativos|ratio]] de cada una.
6. Media de los ratios obtenidos.

El punto crítico de percolación lo dará aquella dimensión que mantenga el ratio a lo largo de muchas iteraciones.

Este punto crítico define el tamaño del [[Kernel de enlace]] a utilizar para la imagen. El que se aproxime más a todas las imágenes será aquel que se utilice.
 