## Eigen Reductions

All reduction operations take a single parameter of type
```<TensorType>::Dimensions``` which can always be specified as an array of
ints.  These are called the "reduction dimensions."  The values are the indices
of the dimensions of the input tensor over which the reduction is done.  The
parameter can have at most as many element as the rank of the input tensor;
each element must be less than the tensor rank, as it indicates one of the
dimensions to reduce.

Each dimension of the input tensor should occur at most once in the reduction
dimensions as the implementation does not remove duplicates.

The order of the values in the reduction dimensions does not affect the
results, but the code may execute faster if you list the dimensions in
increasing order.



## Genauer Ablauf einer reduction

Bei der Deklaration einer Tensor wird eine Instanz der Klasse TensorBase (in TensorBase.h) instantiiert. Diese Klasse besitzt die Methode sum, welche die Reduction (hier also Reduktion durch Summation der Elemente) in Gang setzt. 
Um die Reduction durchzuführen, wird zuerst eine Instanz des structs TensorEvaluator instantiiert (Implementierung in TensorReduction.h). Dann wird die Methode bool evalSubExprsIfNeeded(...) von TensorEvaluator aufgerufen, wo versucht wird, eine für GPU optimierte Implementierung der Reduction durchzuführen.
Die Hauptarbeit wird in der Methode CoeffReturnType coeff(index) des structs TensorEvaluator erledigt, welche direkt nachfolgend auf evalSubExprsIfNeeded(...) aufgerufen wird. 
Einfach gesprochen wird coeff(index) für jeden Eintrag des fertigen reduzierten Tensors einmal aufgerufen. In coeff(index) wird dann ein generischer Befehl zur Reduction entlang der jeweiligen Dimensionen ausgeführt. Dies geschieht mithilfe des structs GenericDimReducer, welches in TensorReduction.h implementiert ist. Dieses Struct besitzt die generische Methode void reduce(...), in der wiederum eine Methode reduce(...) aufgerufen wird. Hier wird dann je nachdem, welche Art der Reduction gewählt wurde (Product, Mean, Sum, Max, ...) die konkrete (und weniger generische) Funktion void reduce(...) vom struct ProductReducer, MeanReducer,  bzw. SumReducer aufgerufen. Am Beispiel des struct SumReducer wird in TensorFunctors.h Zeile 105 die Summation durchgeführt (  (*accum) += t; ) . Die ganzen anderen spezifischen Reducer Implementierungen sind ebenfalls der Datei TensorFunctors.h.

DimInitializer: Initialisiert die Dimensionen des Output-Tensors




## Kompilieren der Eigen CUDA

- Wann wird CUDA version von Reducer ausgeführt? Welche Dimensionen müssen reduziert werden? scheinbar letzte paar dimensionen
  in Liste

