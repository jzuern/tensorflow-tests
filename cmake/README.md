- Aufgabe Ole:
Bisher laden die CMake-Skripte von TensorFlow alle Abhängigkeiten herunter und bauen diese statisch in die libtensorflow.so. Das Ziel wäre jetzt, bei gesetztem USE_SYSTEM_LIBRARIES-Flag die Abhängigkeiten nicht mehr herunterzuladen. Stattdessen sollte mit find_package nach den dynamischen Systembibliotheken gesucht werden und gegen diese gelinkt werden.
Damit wird die libtensorflow.so deutlich kleiner (momentan ist sie über 270 MB groß) und profitiert vor allem von Bug- und Security-Fixes in den Systembibliotheken.


# TODO:
- cmakelists.txt aufräumen, ole schicken und danach pull request starten
- eigen kann nicht verlinkt werden, da es ja header only ist

# zlib,gif,png,jpeg, protobuf: cmake-eigenes find_package() Skript
#eigen3, boringssl, grpc, jsoncpp: online ein FindXY.cmake Skript gefunden
#gemmlowp, farmhash, highwayhash: nichts online gefunden, egal. Baue immer separat

- Mache libtensorflow.so --> tf_libtensorflow.cmake mit add_library(... SHARED) und mache alle .cmake Sachen außer "example", "python","tests","tutorials"


# Wissen:
- statische Library (.a): All the code relating to the library is in this file, and it is directly linked into the program at compile time. A program using a static library takes copies of the code that it uses from the static library and makes it part of the program (shipping with all the library it needs - one large executable)

- dynamische Library (.so): All the code relating to the library is in this file, and it is referenced by programs using it at run-time. A program using a shared library only makes reference to the code that it uses in the shared library. (shipping with only my own code, must make sure that .so files are present on target system)

- cmake build directory muss immer gelöscht werden wenn fundamentale änderungen an CMakeLists.txt erfolgen (da sonst tmp files nicht gelöscht werden)
  - rm CMakeCache.txt
  - rm -rf CMakeFiles



