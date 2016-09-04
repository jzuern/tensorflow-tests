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



## Gespräch 29.08.

- was ist nun neuer Plan wegen bereits implementierten unsorted_segment_sort.cc GPU CUDA?
- Habe unsorted_segment_sum.cc und unsorted_segment_sum_gpu.cu.cc nachvollzogen
- Kompilieren der Unit Tests mit bazel
- Problem: unit test von unsorted_segment_sum scheint nicht implementiert zu sein (im Gegesatz zu bspw. scatter_op_test.cc u.a.


## Gespräch Mail ab 30.08.
Segment_sum ist bereits implementiert worden, daher Themawechsel


3 Mögliche Themen neue Themen
- sparse_softmax_cross_entropy_with_logits implementieren
- bilineares Skalieren --> Bereits implementiert worden
- 3D gaussian Blur (schwer)



