Fecha: 2023-10-17 17:47  
Status: #Idea

# Solución Graph Neural Network

Sea una imagen definida como un conjunto de [[Nodo|nodos]] conectados por [[Enlace|enlaces]] donde los nodos son píxeles y los enlaces definen la distancia entre ellos se puede implementar una [[Graph Neural Network|red neuronal basada en grafos]] para cumplir el objetivo dado[^1].

## Ventajas 

1. Una imagen es considerada un grafo homogéneo por lo que se simplifica el uso de las [[Graph Neural Network|GNNs]].
2. No importa la dimensión de entrada, una [[Graph Neural Network|GNN]] es independiente del número de nodos.
3. Aumenta la percepción espacial del modelo a través de las conexiones entre los enlaces.

## Desventajas

1. El coste computacional es alto.
2. Es compleja de implementar.

---
## Referencias 

[^1]: 