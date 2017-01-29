
# Benchmark of inception-v3 Tensorflow model for the ImageNet dataset

# images:
- book.jpg
- car.jpg
- desk.jpg
- house.jpg
- notebook.jpg


|                        | book.jpg                                                                                                                      | car.jpg | desk | house.jpg | notebook.jpg |
|:----------------------:|-------------------------------------------------------------------------------------------------------------------------------|---------|------|-----------|--------------|
|       unmodified       | tray (766): 0.118142 <br>lampshade (814): 0.0965843 envelope (879): 0.0548038  binder (835): 0.02828 table  lamp (304): 0.0229261 |         |      |           |              |
| --mode=eightbit        |                                                                                                                               |         |      |           |              |
| --mode=weights         |                                                                                                                               |         |      |           |              |
| --mode=weights_rounded |                                                                                                                               |         |      |           |              |

#unmodified:
# book
 tray (766): 0.118142
 lampshade (814): 0.0965843
 envelope (879): 0.0548038
 binder (835): 0.02828
 table lamp (304): 0.0229261

# car
 sports car (274): 0.531175
 beach wagon (266): 0.149122
 grille (725): 0.123515
 car wheel (563): 0.065441
 racer (273): 0.0100775

# desk
 desk (313): 0.727403
 file (305): 0.221826
 chiffonier (303): 0.00620727
 bookcase (300): 0.00299016
 wardrobe (317): 0.00243635

# house
 boathouse (689): 0.195809
 birdhouse (839): 0.0914381
 barn (683): 0.0628866
 beacon (733): 0.0481074
 solar dish (577): 0.0197614

# notebook
 notebook (552): 0.638669
 laptop (228): 0.0588798
 desktop computer (550): 0.0397173
 modem (764): 0.0222256
 space bar (858): 0.0189627


#eightbit:
# book
 tray (766): 0.149952
 lampshade (814): 0.0386165
 envelope (879): 0.0352772
 wardrobe (317): 0.0342296
 shoji (832): 0.0187302


# car
 beach wagon (266): 0.325936
 sports car (274): 0.261816
 grille (725): 0.126147
 car wheel (563): 0.0653836
 racer (273): 0.00815976


# desk
 desk (313): 0.691759
 file (305): 0.231175
 chiffonier (303): 0.0126318
 bookcase (300): 0.00862773
 wardrobe (317): 0.00589286


# house
 boathouse (689): 0.158759
 birdhouse (839): 0.0943576
 barn (683): 0.0418517
 beacon (733): 0.036747
 solar dish (577): 0.0248744


# notebook
 notebook (552): 0.686832
 laptop (228): 0.0499712
 desktop computer (550): 0.0351428
 space bar (858): 0.0228549
 screen (510): 0.016073


#weights:
# book
 tray (766): 0.135119
 lampshade (814): 0.0808479
 envelope (879): 0.0401146
 binder (835): 0.0378881
 ashcan (752): 0.0224833


# car
 sports car (274): 0.526249
 grille (725): 0.143194
 beach wagon (266): 0.122421
 car wheel (563): 0.0712892
 racer (273): 0.00550965


# desk
 desk (313): 0.758875
 file (305): 0.198252
 chiffonier (303): 0.00566149
 bookcase (300): 0.00247311
 wardrobe (317): 0.00186279


# house
 boathouse (689): 0.248064
 birdhouse (839): 0.0723223
 barn (683): 0.0540484
 beacon (733): 0.0382707
 solar dish (577): 0.0196698


# notebook
 notebook (552): 0.602104
 laptop (228): 0.0664463
 desktop computer (550): 0.0359998
 space bar (858): 0.0285657
 screen (510): 0.0232851


#weights_rounded:
# book
 tray (766): 0.186322
 lampshade (814): 0.0892759
 envelope (879): 0.0409567
 binder (835): 0.0331128
 table lamp (304): 0.0227132


# car
 sports car (274): 0.53236
 beach wagon (266): 0.126007
 grille (725): 0.119366
 car wheel (563): 0.0935807
 racer (273): 0.00793687


# desk
 desk (313): 0.759107
 file (305): 0.189873
 chiffonier (303): 0.00814134
 bookcase (300): 0.00374687
 wardrobe (317): 0.00286039


# house
 boathouse (689): 0.317449
 birdhouse (839): 0.0814318
 barn (683): 0.0490688
 beacon (733): 0.0367627
 solar dish (577): 0.0206877


# notebook
 notebook (552): 0.591893
 laptop (228): 0.0671224
 desktop computer (550): 0.0482873
 modem (764): 0.0211112
 space bar (858): 0.0205525
