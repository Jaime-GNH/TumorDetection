Fecha: 2023-11-07 21:10  
Status: #Teoría

# Tamaño de grafo

El tamaño de los grafos impide realizar la [[Backward propagation|propagación hacia atrás]] del [[Entrenamiento|entrenamiento]]. 

Se considera la introducción de un `NeighborLoader`[^1] que extraiga patches de cada imagen. Para ello es necesario conocer el [[Punto crítico de percolación]]. Se utilizará para extraer de cada imagen/[[Grafo]] inicial un lote de entrenamiento completo.

---
## Referencias

[^1]: [Torch_geomeric NeighborLoader](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/loader/neighbor_loader.html#NeighborLoader)
