Fecha: 2023-11-11 18:39  
Status: #Idea

# Solución Reinforcement Learning con Hipergrafos

Sea una imagen definida como un conjunto de [[Hipernodos|hipernodos]] conectados por [[Hiperenlace|hiperenlaces]] donde los hipernodos son conjuntos de píxeles, parches de imágenes, y los hiperenlaces definen la distancia entre ellos se puede implementar una [[Graph Neural Network|red neuronal basada en hipergrafos]] para cumplir el objetivo dado realizando pasos de aproximación tales que:

1. Conversión de imagen a hipergrafo con una dimensión de hipernodo (tamaño del parche) y unión de todos con todos a distancia 1 [[Kernel de enlace#Kernel en estrella. *star*|tipo estrella]].
2. Aplicación de una GNN de clasificación de hipernodos clasificadora *{$0, 1$}*. Será positivo si el hipernodo contiene la clase definida entre: tumor, benigno o maligno.
3. Cálculo de la pérdida, $loss_{N}$.
4. La imagen se recalcula como una máscara de 0's más el valor de los píxeles que se encuentren en el hipergrafo.

Se puede definir el tamaño de los parches por paso o por dimensión de la imagen.




Se puede definir la condición de terminación como un número máximo de aproximaciones efectuadas o por la homogeneización de la imagen, *todo es 0*. Este caso sería la desaparición del hiper grafo.

A las pérdidas de cada nivel se le asocia una función de pérdida conocida y la red entrena con una recompensa definida por la [[Ecuación de Bellman]].

La aplicación de [[Aprendizaje por Refuerzo]] permite disminuir el tamaño del problema y de la GNN.
