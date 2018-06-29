# align
Align a given object in two pictures and ajust the exposure of the pictures.
1. The object might be partial covered or blocked by other shadow/objects.
2. The exposure and contrast are different in most cases.

A direct idea: calculate the offset of the two pictures by calculating the correlation, but how to guarantee the correlation will reach the maximum when the blocked object aligned with the complete object exactly?
Another idea: compute the SIFT (or other) feature of the object and align two pictures. Now the results are not very good. Maybe other features should be used. 
