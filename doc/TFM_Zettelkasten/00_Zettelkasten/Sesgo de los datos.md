Fecha: 2023-10-17 18:03  
Status: #Teoría

# Sesgo de los datos

Los datos a utilizar[^1] contienen el siguiente sesgo insalvable:

>[!cite] Las imágenes se distribuyen de forma que hay tres tipos de máscara:
>- Máscaras de la clase 1
>- Máscaras de la clase 1 y 2
>- Máscaras de la clase 1 y 3

Siendo las clases:
- Clase 1: Normal.
- Clase 2: Tumor benigno
- Clase 3: Tumor maligno

Con lo que un modelo no será capaz (o no debería serlo) de obtener máscaras con las clases 1, 2 y 3 ni únicamente con las clases 2 y 3.

---
## Referencias

[^1]: W. Al-Dhabyani, M. Gomaa, H. Khaled, y A. Fahmy, «Dataset of breast ultrasound images», _Data Brief_, vol. 28, p. 104863, feb. 2020, doi: [10.1016/j.dib.2019.104863](https://doi.org/10.1016/j.dib.2019.104863).