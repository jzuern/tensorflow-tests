

# Notes

- im Makefile immer gcc-4.8 als compiler angeben (Kompabilität)
- habe Grafikkarte GT 755M
- cuda compute capability 3.0

- Musste bei bazel makefile änderungen vornehmen ( siehe hier: https://github.com/tensorflow/tensorflow/issues/1066#issuecomment-200580370)
 

## TensorFLow usage


### To build with GPU support:
bazel build -c opt --config=cuda //tensorflow/tools/pip_package:build_pip_package (run this every time build rules, c++ file or python file change)


### Startup of tensorflow
- run in bash in order to let TF find cuda libraries:
	export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64"
	export CUDA_HOME=/usr/local/cuda



## Versions

cuda sdk: 	7.5
cudnn: 		4.0.7
tensorflow: 	1.0
python: 	2.7.12



## Paths

tensorflow	/home/jzuern/tf_installation/tensorflow



eigenv3         /home/jzuern/tf_installation/tensorflow/bazel-tensorflow/external/eigen_archive

eigen_tensor	/home/jzuern/tf_installation/tensorflow/bazel-tensorflow/external/eigen_archive/unsupported/Eigen/CXX11/src/Tensor
cuda: 		/usr/local/cuda-7.5

cudnn: 		/usr/local/cuda-7.5

python: 	/usr/bin/python

bazel: 		/home/jzuern/bin/bazel


# Definitions

## CUDA stream object

A stream is a queue of device work (kernel launches, memory copies can be launched from within a stream). Host places work in queue und continues immediately (asynchronous). Declaration of stream handle: cudaStream_t stream; cudaStreamCreate(&stream); (allocates a stream).
Placing work into stream: 4th launch parameter: kernel <<<blocks,threads,smem,stream>>>();
Placed in some API calls as well: cudaMemcpyAsync(dst,src,size,dir,stream);


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





