# Einarbeitung

## Lesen allgemein
- Profiling von CPU, GPU verstehen
	Verwende NVIDIA Visual Profiler?

## Installieren

- CUDA toolkit installieren (done)
- gcc version 4.8 installieren(done)
- CuDNN installieren (done)
- TensorFlow gitHub fork and clone (done)
- TensorFlow installing from sources (done)
- CuDNN in TensorFlow einbinden mit configure script in tensorflow root (done)
- TensorFlow build with bazel (done)


## Gespräch 29.07.

- Tensorflow GPU implementierungen von Operatoren zu wirr, daher schauen wir erstmal Eigen Implementierungen an.


## Gespräch 09.08.

- Info: packet() funktionen: Nutze Architektur von CPU Register, sodass 4 Vektor-Einträge
	  gleichzeitig bearbeitet werden. Am Ende wird wieder zusammengefasst
- neue Aufgabe: Cuda für Eigen zum Kompilieren und Laufen bringen
- Mail schicken mit Zusammenfassung, Output der Programms


## Gespräch 16.08.

- Aufgabe: GPUassert invalid memory (...) beheben (für gewisse Fälle geht es jetzt, dank größerem Buffer)
- Aufgabe: segment_reduction_ops.cc nachvollziehen (done)
- Aufgabe: Ausführungskette bei unsorted_segment_sum verstehen (gdb / print out)



##

- Eigentliche Hiwi Aufgabe: implementierung von slice operationen auf tensor, die bisher in /home/jzuern/Dropbox/develop/hiwi_mrt/tensorflow/tensorflow/core/kernels/segment_reduction_ops.cc sind in Eigen-Syntax überführen (Eigen implementierung als .sliced_sum(...) zusätzlich zu .sum(...) Methode
