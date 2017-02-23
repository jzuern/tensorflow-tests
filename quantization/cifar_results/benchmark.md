
# Benchmark of inception-v3 Tensorflow model for the ImageNet dataset

# images:
- book.jpg
![](https://gitlab.mrt.uni-karlsruhe.de/zuern/tf-ops/raw/master/quantization/book.jpg)

- car.jpg
![](https://gitlab.mrt.uni-karlsruhe.de/zuern/tf-ops/raw/master/quantization/car.jpg)

- desk.jpg
![](https://gitlab.mrt.uni-karlsruhe.de/zuern/tf-ops/raw/master/quantization/desk.jpg)

- house.png
![](https://gitlab.mrt.uni-karlsruhe.de/zuern/tf-ops/raw/master/quantization/house.png)

- notebook.jpg
![](https://gitlab.mrt.uni-karlsruhe.de/zuern/tf-ops/raw/master/quantization/notebook.jpg)



|                        | book.jpg                                                                                                                      | car.jpg                                                                                                                          | desk.jpg                                                                                                                     | house.png                                                                                                                      | notebook.jpg                                                                                                                          |
|:----------------------:|-------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------|
|       unmodified       | tray (766): 0.118142 <br>lampshade (814): 0.0965843 <br>envelope (879): 0.0548038<br>  binder (835): 0.02828<br> table  lamp (304): 0.0229261<br> | sports car (274): 0.531175<br> beach wagon (266): 0.149122<br> grille (725): 0.123515<br> car wheel (563): 0.065441<br> racer (273): 0.0100775<br>   | desk (313): 0.727403<br> file (305): 0.221826<br> chiffonier (303): 0.00620727<br> bookcase (300): 0.00299016<br> wardrobe (317): 0.00243635<br> | boathouse (689): 0.195809<br> birdhouse (839): 0.0914381<br> barn (683): 0.0628866<br> beacon (733): 0.0481074<br> solar dish (577): 0.0197614<br> | notebook (552): 0.638669<br> laptop (228): 0.0588798<br> desktop computer (550): 0.0397173<br> modem (764): 0.0222256<br> space bar (858): 0.0189627<br>  |
| --mode=eightbit        | tray (766): 0.149952<br> lampshade (814): 0.0386165<br> envelope (879): 0.0352772<br> wardrobe (317): 0.0342296<br> shoji (832): 0.0187302<br>    | beach wagon (266): 0.325936<br> sports car (274): 0.261816<br> grille (725): 0.126147<br> car wheel (563): 0.0653836<br> racer (273): 0.00815976<br> | desk (313): 0.691759<br> file (305): 0.231175 chiffonier<br> (303): 0.0126318<br> bookcase (300): 0.00862773<br> wardrobe (317): 0.00589286 <br> | boathouse (689): 0.158759<br> birdhouse (839): 0.0943576<br> barn (683): 0.0418517<br> beacon (733): 0.036747<br> solar dish (577): 0.0248744<br>  | notebook (552): 0.686832<br> laptop (228): 0.0499712<br> desktop computer (550): 0.0351428<br> space bar (858): 0.0228549<br> screen (510): 0.016073<br>  |
| --mode=weights         | tray (766): 0.135119<br> lampshade (814): 0.0808479<br> envelope (879): 0.0401146<br> binder (835): 0.0378881<br> ashcan (752): 0.0224833<br>     | sports car (274): 0.526249<br> grille (725): 0.143194<br> beach wagon (266): 0.122421<br> car wheel (563): 0.0712892<br> racer (273): 0.00550965<br> | desk (313): 0.758875<br> file (305): 0.198252<br> chiffonier (303): 0.00566149<br> bookcase (300): 0.00247311<br> wardrobe (317): 0.00186279<br> | boathouse (689): 0.248064<br> birdhouse (839): 0.0723223<br> barn (683): 0.0540484<br> beacon (733): 0.0382707<br> solar dish (577): 0.0196698<br> | notebook (552): 0.602104<br> laptop (228): 0.0664463<br> desktop computer (550): 0.0359998<br> space bar (858): 0.0285657<br> screen (510): 0.0232851<br> |
| --mode=weights_rounded | tray (766): 0.186322<br> lampshade (814): 0.0892759<br> envelope (879): 0.0409567<br> binder (835): 0.0331128<br> table lamp (304): 0.0227132<br> | sports car (274): 0.53236<br> beach wagon (266): 0.126007<br> grille (725): 0.119366<br> car wheel (563): 0.0935807<br> racer (273): 0.00793687<br>  | desk (313): 0.759107<br> file (305): 0.189873<br> chiffonier (303): 0.00814134<br> bookcase (300): 0.00374687<br> wardrobe (317): 0.00286039<br> | boathouse (689): 0.317449<br> birdhouse (839): 0.0814318<br> barn (683): 0.0490688<br> beacon (733): 0.0367627<br> solar dish (577): 0.0206877<br> | notebook (552): 0.591893<br> laptop (228): 0.0671224<br> desktop computer (550): 0.0482873<br> modem (764): 0.0211112<br> space bar (858): 0.0205525<br>  |



