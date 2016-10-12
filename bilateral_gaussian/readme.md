


## TO Do

- implementiere reverse blur, sodass backpropagation geht

Wie?
--> mache neuen Op als Gradient und teile so viel Code mit altem CustomOp wie möglich

- Tensorflow-Hashtable anstatt eigene Hash table verwenden

Probleme:
- Lookup Table von TF hat keine grow() Methode (die wir aber brauchen)
- Lookup Table von TF hat keine lookupOffset Methode (Was macht diese genau??)
- Lookup Table ist eher im Sinne eines Op implementiert. Weiß nicht so recht, was ich damit anfangen soll...
- Warum wird splatting, blurring und slicing 2x durchlaufen??
