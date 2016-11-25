# Code Strukturierung:




# TODO:


- MirroredArray struct nicht mehr verwenden (erspart hin-und Her kopieren der Tensor Einträge)
- Bug der Bildgröße-> Segfault finden
- Ole Code in gitlab repository Link schicken



# Erkenntnis:
- Übergebe mit input.data() an Kernel, sodass ich mit Pointern auf Memory arbeiten kann!
(Sollte dies auch in die CPU implementation übernehmen)

- Wenn ich einen Kernel als CPU-device kompatibel definiere, kann ich in Op-Methode void Compute(...) auf die Daten des Input Tensors ganz normal zugreifen. Wenn ich aber den selben Kernel als GPU-device kompatibel definiere, kommt segmentation fault, wenn ich auf Daten des Input Tensors zugreifen möchte. Daher kann ich nur auf Input zugreifen, wenn ich innerhalb von CUDA Kernel call bin!


