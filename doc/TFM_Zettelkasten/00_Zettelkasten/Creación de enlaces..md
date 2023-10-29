Fecha: 2023-10-24 19:42  
Status: #Idea

# Creación de enlaces.

Los enlaces se crean a partir de reglas lógicas entre las posiciones de los [[Nodo|nodos]] y las [[Dilatación de convoluciones|dilataciones]] impuestas.

![[Enlaces dilatacion 11.png]]

Se deben eliminar los auto-enlaces y transformar el grafo a no dirigido, pues no tiene dirección privilegiada, y eliminar enlaces duplicados:
```python
Graph.remove_self_loops()
Graph.to_undirected()
Graph.coalesce()
```
