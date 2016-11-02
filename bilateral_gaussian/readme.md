


# TO Do


## Warum wird splatting, blurring und slicing 2x durchlaufen??

--> das Originalbild hat Auflösung 800x1200. Wenn ich Auflösung auf unter ca. 750x... reduziere, wird es
nur 1x durchlaufen. Vielleicht gibt es einen Buffer overflow oder so?

Ideen: 
- Update tensorflow to v0.11, 
- use Valgrind to check for memory leaks etc., 
- types nochmal checken
- ...?

## DA Implementierung von bilateral

0. kompiliere CUDA Implementierung ohne Fehler
1. Implementierung nachvollziehen
2. Implementierung als KERNEL konvertieren

## gitlab Account zugriff wie?




