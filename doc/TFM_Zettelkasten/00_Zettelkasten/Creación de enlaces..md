Fecha: 2023-10-24 19:42  
Status: #Idea

# Creación de enlaces.

Utilizar la función [torch_geometric.utils.grid](https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html#torch_geometric.utils.grid) para crear kernels de [[Enlace|enlaces]].

Un [[Kernel de enlace|kernel de enlace]] puede ser:

```python
k = (3,3)
(row, col), _ = grid(height=k[0], width=k[1])
```

de forma que cada elemento 0, 1, 2, 3, 4, 5, 6, 7, 8 se conecta con el 0, 1, 2, 3, 4, 5, 6, 7, 8 siendo `row` y `col` los tensores `src`, `dst` que representan los enlaces [[edge_index]] del sub-grafo centrado en 0. Este kernel se puede aplicar a cada nodo del grafo.

Se deben eliminar los auto-enlaces y transformar el grafo a no dirigido, pues no tiene dirección privilegiada, y eliminar enlaces duplicados:
```python
Graph.remove_self_loops()
Graph.to_undirected()
Graph.coalesce()
```

En términos de [[Dilatación de convoluciones|dilatación de convoluciones]] este caso introduciría una dilatación `d=1`. 

Para añadir dilataciones de orden $N$  se puede extraer un kernel mayor, por ejemplo `k = (7,7)`, y tomar reducir los tensores tal que:

```python
k = (7,7)
d = 2
(row, col), _ = grid(height=k[0], width=k[1])
row, col = row[::d], col[::d]
```

Con lo que el snippet final quedaría como:

```python
edge_index = torch.empty()
dilations = [1, 3, 5]
for d in dilations:
	(row, col) = torch_geometric.utils.grid(3+d, 3+d)
	graph.edge_index = torch.cat(
						(graph.edge_index, 
						 torch.stack([row[::d], col[::d]]),
						dim = 1
						)
	graph.remove_self_loops()
	graph.to_undirected()
	graph.coalesce()	
```

==No comprobado. Es un borrador==
