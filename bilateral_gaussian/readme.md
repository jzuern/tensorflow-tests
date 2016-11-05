# Environment Setup

Vor TF-Skript Aufruf ausführen:

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64"
export CUDA_HOME=/usr/local/cuda


# TO Do


## Warum wird splatting, blurring und slicing 2x durchlaufen??

--> das Originalbild hat Auflösung 800x1200. Wenn ich Auflösung auf unter ca. 750x... reduziere, wird es
nur 1x durchlaufen. Vielleicht gibt es einen Buffer overflow oder so?

Ideen: 
- Update tensorflow to v0.11, 
- use Valgrind to check for memory leaks etc., 
- types nochmal checken

## CUDA Implementierung von bilateral

- Implementierung nachvollziehen... done
- Implementierung als CustomOp Kernel ...done


Weiter:
- Ersetzen von Inhalt in cuda_op_kernel.cc /.cu.cc mit bilateral Zeug


Erkenntnis: 
- Kann alles mit bazel kompilieren
