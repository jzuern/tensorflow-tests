


# To Do

- implementiere allocate_temp für hash table variablen und Matrix
- implementiere allocate_persistent für 1. Kernel call


# Probleme

- Wie allokiere ich diese Matrix (struct MatrixEntry) als Tensor?



# Aktuelle Version wieder ausrollen:

- cudaFree wie früher



- Idee: Variable values in  createHashTable(int capacity) kann filter Funktion bereits übergeben werden aus OpKernel, und dann ist auch kein cudaMalloc von values nötig!

- setZeroKernel unnötig? --> cudamemset stattdessen?
