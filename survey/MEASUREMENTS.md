All latency columns in us; nearest-rank percentiles.
Budget: 3000.0 us (ASSUMED K3 deadline; GLM window unmeasured).
| hop | p50 | p90 | p99 | max | samples | p99 budget |
|---|---:|---:|---:|---:|---:|---|
| nuc2->nuc3/ethernet/k=0 | 182.8 | 198.3 | 339.6 | 613.7 | 1000 | — |
| nuc2->nuc3/ethernet/k=1 | 261.3 | 369.5 | 569.2 | 1668.2 | 1000 | — |
| nuc2->nuc3/ethernet/k=2 | 315.1 | 421.2 | 549.2 | 2119.1 | 1000 | — |
| nuc2->nuc3/ethernet/k=4 | 404.3 | 530.2 | 923.0 | 1951.6 | 1000 | — |
| nuc2->nuc3/ethernet/k=8 | 635.6 | 826.7 | 1038.9 | 1678.4 | 1000 | — |
| nuc2->nuc3/thunderbolt0/k=0 | 183.7 | 214.1 | 355.0 | 791.4 | 1000 | — |
| nuc2->nuc3/thunderbolt0/k=1 | 270.4 | 399.2 | 565.2 | 2440.8 | 1000 | — |
| nuc2->nuc3/thunderbolt0/k=2 | 391.2 | 516.8 | 690.4 | 2261.4 | 1000 | — |
| nuc2->nuc3/thunderbolt0/k=4 | 415.4 | 673.7 | 848.3 | 3114.8 | 1000 | — |
| nuc2->nuc3/thunderbolt0/k=8 | 624.9 | 865.2 | 1343.6 | 2453.8 | 1000 | — |
| nuc2/exact=0/resident/loopback/k=1 | 594.5 | 837.5 | 1117.2 | 2674.9 | 1000 | — |
| nuc2/exact=0/resident/loopback/k=2 | 917.5 | 975.7 | 1254.1 | 2734.0 | 1000 | — |
| nuc2/exact=0/resident/loopback/k=4 | 1861.4 | 2324.0 | 2687.6 | 5320.9 | 1000 | — |
| nuc2/exact=0/resident/loopback/k=8 | 3288.1 | 3845.2 | 4455.3 | 6973.7 | 1000 | — |
| nuc2/exact=1/resident/loopback/k=1 | 11887.0 | 12319.7 | 13326.7 | 14810.5 | 627 | — |
| nuc2/exact=1/resident/loopback/k=2 | 19568.8 | 20214.5 | 20973.2 | 21484.6 | 378 | — |
| nuc2/exact=1/resident/loopback/k=4 | 38099.3 | 39114.6 | 42834.7 | 44033.5 | 189 | — |
| nuc2/exact=1/resident/loopback/k=8 | 74850.1 | 76915.6 | 81946.7 | 81946.7 | 92 | — |
| nuc3->nuc2/ethernet/k=0 | 292.4 | 440.6 | 567.1 | 1580.7 | 1000 | — |
| nuc3->nuc2/ethernet/k=1 | 273.2 | 383.8 | 542.9 | 2236.7 | 1000 | — |
| nuc3->nuc2/ethernet/k=2 | 390.7 | 539.1 | 650.3 | 1201.2 | 1000 | — |
| nuc3->nuc2/ethernet/k=4 | 443.1 | 632.6 | 861.5 | 1635.2 | 1000 | — |
| nuc3->nuc2/ethernet/k=8 | 556.0 | 803.4 | 1040.1 | 1319.6 | 1000 | — |
| nuc3->nuc2/thunderbolt0/k=0 | 194.6 | 261.6 | 405.8 | 1142.8 | 1000 | — |
| nuc3->nuc2/thunderbolt0/k=1 | 272.8 | 379.5 | 533.4 | 1013.4 | 1000 | — |
| nuc3->nuc2/thunderbolt0/k=2 | 370.5 | 463.9 | 783.5 | 1522.1 | 1000 | — |
| nuc3->nuc2/thunderbolt0/k=4 | 407.3 | 523.5 | 942.2 | 1547.9 | 1000 | — |
| nuc3->nuc2/thunderbolt0/k=8 | 552.3 | 692.0 | 1042.5 | 1396.9 | 1000 | — |
| nuc3/exact=0/resident/loopback/k=1 | 556.6 | 736.3 | 1358.2 | 4208.7 | 1000 | — |
| nuc3/exact=0/resident/loopback/k=2 | 912.4 | 1020.4 | 2044.3 | 4729.5 | 1000 | — |
| nuc3/exact=0/resident/loopback/k=4 | 1823.6 | 2267.7 | 4180.0 | 6894.8 | 1000 | — |
| nuc3/exact=0/resident/loopback/k=8 | 4029.8 | 4561.9 | 6161.2 | 7057.3 | 1000 | — |
| nuc3/exact=1/resident/loopback/k=1 | 11450.5 | 12580.9 | 14025.8 | 15594.1 | 648 | — |
| nuc3/exact=1/resident/loopback/k=2 | 19371.9 | 20843.7 | 23334.0 | 24555.7 | 378 | — |
| nuc3/exact=1/resident/loopback/k=4 | 42264.6 | 44315.3 | 47662.1 | 48870.5 | 171 | — |
| nuc3/exact=1/resident/loopback/k=8 | 82084.9 | 88339.7 | 93247.8 | 93247.8 | 83 | — |
| spark->nuc2/ethernet/k=0 | 103.9 | 108.0 | 205.6 | 1128.3 | 1000 | — |
| spark->nuc2/ethernet/k=1 | 297.3 | 314.6 | 393.5 | 815.8 | 1000 | — |
| spark->nuc2/ethernet/k=2 | 389.5 | 410.4 | 575.0 | 3446.8 | 1000 | — |
| spark->nuc2/ethernet/k=4 | 574.0 | 664.7 | 911.3 | 3088.2 | 1000 | — |
| spark->nuc2/ethernet/k=8 | 903.1 | 980.4 | 1137.1 | 1596.3 | 1000 | — |
| spark->nuc2/exact=0/cold/prototype/k=1 | 7913.0 | 8762.8 | 9671.2 | 11652.2 | 483 | EXCEEDS; p99 weak; fast math only; prototype |
| spark->nuc2/exact=0/cold/prototype/k=2 | 14337.8 | 15599.8 | 16796.0 | 17471.3 | 259 | EXCEEDS; p99 weak; fast math only; prototype |
| spark->nuc2/exact=0/cold/prototype/k=3 | 21060.6 | 22512.4 | 24513.2 | 24815.5 | 170 | EXCEEDS; p99 weak; fast math only; prototype |
| spark->nuc2/exact=0/cold/prototype/k=4 | 27737.7 | 30029.6 | 32340.5 | 33521.9 | 127 | EXCEEDS; p99 weak; fast math only; prototype |
| spark->nuc2/exact=0/cold/prototype/k=5 | 35147.6 | 37152.0 | 38634.4 | 38634.4 | 99 | EXCEEDS; p99 weak; fast math only; prototype |
| spark->nuc2/exact=0/cold/prototype/k=6 | 42435.6 | 44232.1 | 50481.5 | 50481.5 | 81 | EXCEEDS; p99 weak; fast math only; prototype |
| spark->nuc2/exact=0/cold/prototype/k=7 | 49357.5 | 51291.6 | 56293.1 | 56293.1 | 68 | EXCEEDS; p99 weak; fast math only; prototype |
| spark->nuc2/exact=0/cold/prototype/k=8 | 56299.6 | 59333.1 | 71527.8 | 71527.8 | 58 | EXCEEDS; p99 weak; fast math only; prototype |
| spark->nuc2/exact=0/lru/prototype/k=1 | 898.1 | 960.0 | 1229.7 | 1690.4 | 1000 | fits; fast math only; prototype |
| spark->nuc2/exact=0/lru/prototype/k=2 | 1173.3 | 1280.5 | 1429.8 | 1660.1 | 1000 | fits; fast math only; prototype |
| spark->nuc2/exact=0/lru/prototype/k=3 | 1497.5 | 1604.6 | 2155.2 | 2668.8 | 1000 | fits; fast math only; prototype |
| spark->nuc2/exact=0/lru/prototype/k=4 | 2060.8 | 2515.1 | 2938.4 | 5323.4 | 1000 | fits; fast math only; prototype |
| spark->nuc2/exact=0/lru/prototype/k=5 | 2742.7 | 3027.9 | 3455.0 | 3986.5 | 860 | EXCEEDS; p99 weak; fast math only; prototype |
| spark->nuc2/exact=0/lru/prototype/k=6 | 2990.2 | 3325.1 | 3717.1 | 4152.6 | 757 | EXCEEDS; p99 weak; fast math only; prototype |
| spark->nuc2/exact=0/lru/prototype/k=7 | 3377.9 | 3688.9 | 3967.0 | 6213.7 | 661 | EXCEEDS; p99 weak; fast math only; prototype |
| spark->nuc2/exact=0/lru/prototype/k=8 | 3529.5 | 4162.1 | 5238.6 | 6289.1 | 606 | EXCEEDS; p99 weak; fast math only; prototype |
| spark->nuc2/exact=0/resident/k=1 | 917.2 | 1044.4 | 2022.5 | 4037.1 | 1000 | fits; fast math only |
| spark->nuc2/exact=0/resident/k=2 | 1223.4 | 1582.2 | 1752.2 | 1950.7 | 1000 | fits; fast math only |
| spark->nuc2/exact=0/resident/k=3 | 1543.0 | 1910.8 | 2449.1 | 2714.2 | 1000 | fits; fast math only |
| spark->nuc2/exact=0/resident/k=4 | 2156.2 | 2605.6 | 3155.2 | 7930.9 | 1000 | EXCEEDS; fast math only |
| spark->nuc2/exact=0/resident/k=5 | 2621.7 | 2974.6 | 3384.7 | 3789.4 | 931 | EXCEEDS; p99 weak; fast math only |
| spark->nuc2/exact=0/resident/k=6 | 3052.9 | 3408.3 | 3906.6 | 4525.2 | 883 | EXCEEDS; p99 weak; fast math only |
| spark->nuc2/exact=0/resident/k=7 | 3437.2 | 3949.8 | 4626.6 | 5118.7 | 659 | EXCEEDS; p99 weak; fast math only |
| spark->nuc2/exact=0/resident/k=8 | 3493.2 | 4087.8 | 4613.3 | 4888.2 | 600 | EXCEEDS; p99 weak; fast math only |
| spark->nuc2/exact=1/cold/prototype/k=1 | 18559.5 | 19243.6 | 19976.5 | 20012.3 | 206 | EXCEEDS; p99 weak; prototype |
| spark->nuc2/exact=1/cold/prototype/k=2 | 32686.6 | 33889.8 | 35147.9 | 36711.5 | 110 | EXCEEDS; p99 weak; prototype |
| spark->nuc2/exact=1/cold/prototype/k=3 | 47889.9 | 49293.8 | 50565.6 | 50565.6 | 72 | EXCEEDS; p99 weak; prototype |
| spark->nuc2/exact=1/cold/prototype/k=4 | 63707.5 | 65452.3 | 66408.9 | 66408.9 | 52 | EXCEEDS; p99 weak; prototype |
| spark->nuc2/exact=1/cold/prototype/k=5 | 79854.1 | 81858.8 | 82047.9 | 82047.9 | 39 | EXCEEDS; p99 weak; prototype |
| spark->nuc2/exact=1/cold/prototype/k=6 | 91923.7 | 94526.7 | 96067.6 | 96067.6 | 33 | EXCEEDS; p99 weak; prototype |
| spark->nuc2/exact=1/cold/prototype/k=7 | 107354.6 | 109995.3 | 110337.9 | 110337.9 | 27 | EXCEEDS; p99 weak; prototype |
| spark->nuc2/exact=1/cold/prototype/k=8 | 123749.2 | 129359.2 | 135500.7 | 135500.7 | 22 | EXCEEDS; p99 weak; prototype |
| spark->nuc2/exact=1/lru/prototype/k=1 | 11014.8 | 11708.9 | 11971.3 | 12309.3 | 340 | EXCEEDS; p99 weak; prototype |
| spark->nuc2/exact=1/lru/prototype/k=2 | 18977.3 | 19743.8 | 21043.8 | 22843.7 | 193 | EXCEEDS; p99 weak; prototype |
| spark->nuc2/exact=1/lru/prototype/k=3 | 27188.1 | 27778.5 | 29314.5 | 29739.6 | 133 | EXCEEDS; p99 weak; prototype |
| spark->nuc2/exact=1/lru/prototype/k=4 | 37026.0 | 38070.4 | 40401.2 | 40401.2 | 95 | EXCEEDS; p99 weak; prototype |
| spark->nuc2/exact=1/lru/prototype/k=5 | 46544.5 | 47482.1 | 49562.5 | 49562.5 | 74 | EXCEEDS; p99 weak; prototype |
| spark->nuc2/exact=1/lru/prototype/k=6 | 54024.5 | 55697.9 | 59950.8 | 59950.8 | 62 | EXCEEDS; p99 weak; prototype |
| spark->nuc2/exact=1/lru/prototype/k=7 | 62423.9 | 65355.5 | 67759.5 | 67759.5 | 53 | EXCEEDS; p99 weak; prototype |
| spark->nuc2/exact=1/lru/prototype/k=8 | 74614.0 | 79402.4 | 81894.5 | 81894.5 | 42 | EXCEEDS; p99 weak; prototype |
| spark->nuc2/exact=1/resident/k=1 | 11447.9 | 11939.4 | 12720.1 | 13455.6 | 332 | EXCEEDS; p99 weak |
| spark->nuc2/exact=1/resident/k=2 | 19138.7 | 20086.3 | 21232.0 | 21241.7 | 191 | EXCEEDS; p99 weak |
| spark->nuc2/exact=1/resident/k=3 | 27478.7 | 28959.8 | 30380.3 | 30713.6 | 130 | EXCEEDS; p99 weak |
| spark->nuc2/exact=1/resident/k=4 | 36885.5 | 38106.4 | 43676.0 | 43676.0 | 95 | EXCEEDS; p99 weak |
| spark->nuc2/exact=1/resident/k=5 | 46776.0 | 48436.6 | 49701.3 | 49701.3 | 73 | EXCEEDS; p99 weak |
| spark->nuc2/exact=1/resident/k=6 | 52863.3 | 55831.8 | 58660.8 | 58660.8 | 63 | EXCEEDS; p99 weak |
| spark->nuc2/exact=1/resident/k=7 | 61307.9 | 63771.8 | 67779.8 | 67779.8 | 53 | EXCEEDS; p99 weak |
| spark->nuc2/exact=1/resident/k=8 | 72077.8 | 74897.8 | 77490.3 | 77490.3 | 44 | EXCEEDS; p99 weak |
| spark->nuc3/ethernet/k=0 | 103.6 | 207.5 | 274.9 | 640.8 | 1000 | — |
| spark->nuc3/ethernet/k=1 | 273.7 | 296.5 | 669.0 | 1434.6 | 1000 | — |
| spark->nuc3/ethernet/k=2 | 374.1 | 399.5 | 631.5 | 1708.2 | 1000 | — |
| spark->nuc3/ethernet/k=4 | 579.3 | 625.5 | 729.2 | 1456.9 | 1000 | — |
| spark->nuc3/ethernet/k=8 | 955.6 | 1024.3 | 1115.0 | 1858.4 | 1000 | — |
| spark->nuc3/exact=0/cold/prototype/k=1 | 7204.8 | 7919.4 | 8646.7 | 9593.9 | 521 | EXCEEDS; p99 weak; fast math only; prototype |
| spark->nuc3/exact=0/cold/prototype/k=2 | 13112.1 | 13827.7 | 15048.5 | 15791.1 | 277 | EXCEEDS; p99 weak; fast math only; prototype |
| spark->nuc3/exact=0/cold/prototype/k=3 | 19114.8 | 19981.2 | 23997.8 | 36495.3 | 187 | EXCEEDS; p99 weak; fast math only; prototype |
| spark->nuc3/exact=0/cold/prototype/k=4 | 25014.3 | 25928.6 | 27417.6 | 28562.7 | 141 | EXCEEDS; p99 weak; fast math only; prototype |
| spark->nuc3/exact=0/cold/prototype/k=5 | 31072.5 | 32010.2 | 33639.3 | 33692.2 | 112 | EXCEEDS; p99 weak; fast math only; prototype |
| spark->nuc3/exact=0/cold/prototype/k=6 | 36087.1 | 36897.9 | 39829.4 | 39829.4 | 95 | EXCEEDS; p99 weak; fast math only; prototype |
| spark->nuc3/exact=0/cold/prototype/k=7 | 42662.1 | 43834.5 | 45987.8 | 45987.8 | 79 | EXCEEDS; p99 weak; fast math only; prototype |
| spark->nuc3/exact=0/cold/prototype/k=8 | 48711.4 | 49829.5 | 53689.7 | 53689.7 | 70 | EXCEEDS; p99 weak; fast math only; prototype |
| spark->nuc3/exact=0/lru/prototype/k=1 | 748.2 | 799.3 | 1094.1 | 1454.6 | 1000 | fits; fast math only; prototype |
| spark->nuc3/exact=0/lru/prototype/k=2 | 1147.2 | 1181.0 | 1297.7 | 1405.3 | 1000 | fits; fast math only; prototype |
| spark->nuc3/exact=0/lru/prototype/k=3 | 1791.5 | 1941.2 | 2831.9 | 3018.2 | 1000 | fits; fast math only; prototype |
| spark->nuc3/exact=0/lru/prototype/k=4 | 2103.3 | 2423.4 | 3094.5 | 4746.1 | 1000 | EXCEEDS; fast math only; prototype |
| spark->nuc3/exact=0/lru/prototype/k=5 | 2861.1 | 3205.4 | 4216.5 | 5446.6 | 864 | EXCEEDS; p99 weak; fast math only; prototype |
| spark->nuc3/exact=0/lru/prototype/k=6 | 3321.7 | 3777.0 | 4923.2 | 6147.3 | 748 | EXCEEDS; p99 weak; fast math only; prototype |
| spark->nuc3/exact=0/lru/prototype/k=7 | 3558.7 | 4014.7 | 4545.0 | 5453.4 | 681 | EXCEEDS; p99 weak; fast math only; prototype |
| spark->nuc3/exact=0/lru/prototype/k=8 | 4057.8 | 4391.5 | 5025.4 | 6305.0 | 672 | EXCEEDS; p99 weak; fast math only; prototype |
| spark->nuc3/exact=0/resident/k=1 | 746.9 | 788.7 | 1020.9 | 1339.5 | 1000 | fits; fast math only |
| spark->nuc3/exact=0/resident/k=2 | 1144.5 | 1196.5 | 1328.0 | 1492.6 | 1000 | fits; fast math only |
| spark->nuc3/exact=0/resident/k=3 | 1510.7 | 1819.8 | 2401.4 | 5977.9 | 1000 | fits; fast math only |
| spark->nuc3/exact=0/resident/k=4 | 2127.6 | 2782.0 | 3309.0 | 3535.5 | 1000 | EXCEEDS; fast math only |
| spark->nuc3/exact=0/resident/k=5 | 2952.5 | 3289.8 | 4528.5 | 6445.9 | 812 | EXCEEDS; p99 weak; fast math only |
| spark->nuc3/exact=0/resident/k=6 | 3453.6 | 3881.1 | 4678.1 | 6343.4 | 735 | EXCEEDS; p99 weak; fast math only |
| spark->nuc3/exact=0/resident/k=7 | 3707.8 | 4175.6 | 4969.6 | 7361.8 | 665 | EXCEEDS; p99 weak; fast math only |
| spark->nuc3/exact=0/resident/k=8 | 4108.9 | 4529.1 | 6075.9 | 7235.7 | 580 | EXCEEDS; p99 weak; fast math only |
| spark->nuc3/exact=1/cold/prototype/k=1 | 17794.6 | 18782.7 | 19949.5 | 20380.4 | 214 | EXCEEDS; p99 weak; prototype |
| spark->nuc3/exact=1/cold/prototype/k=2 | 31954.7 | 33510.1 | 44399.9 | 44608.4 | 112 | EXCEEDS; p99 weak; prototype |
| spark->nuc3/exact=1/cold/prototype/k=3 | 47712.7 | 49802.5 | 51090.2 | 51090.2 | 72 | EXCEEDS; p99 weak; prototype |
| spark->nuc3/exact=1/cold/prototype/k=4 | 64886.6 | 67140.0 | 70963.4 | 70963.4 | 51 | EXCEEDS; p99 weak; prototype |
| spark->nuc3/exact=1/cold/prototype/k=5 | 81083.7 | 83777.9 | 84807.9 | 84807.9 | 39 | EXCEEDS; p99 weak; prototype |
| spark->nuc3/exact=1/cold/prototype/k=6 | 93680.7 | 98161.6 | 112081.6 | 112081.6 | 32 | EXCEEDS; p99 weak; prototype |
| spark->nuc3/exact=1/cold/prototype/k=7 | 110125.1 | 112818.1 | 114632.2 | 114632.2 | 26 | EXCEEDS; p99 weak; prototype |
| spark->nuc3/exact=1/cold/prototype/k=8 | 128570.2 | 132087.0 | 133137.8 | 133137.8 | 21 | EXCEEDS; p99 weak; prototype |
| spark->nuc3/exact=1/lru/prototype/k=1 | 11817.6 | 12391.5 | 13040.2 | 13307.1 | 322 | EXCEEDS; p99 weak; prototype |
| spark->nuc3/exact=1/lru/prototype/k=2 | 20253.4 | 20944.8 | 21906.1 | 23055.9 | 184 | EXCEEDS; p99 weak; prototype |
| spark->nuc3/exact=1/lru/prototype/k=3 | 30995.1 | 32132.2 | 33891.5 | 34402.6 | 115 | EXCEEDS; p99 weak; prototype |
| spark->nuc3/exact=1/lru/prototype/k=4 | 42915.8 | 44507.8 | 47404.9 | 47404.9 | 80 | EXCEEDS; p99 weak; prototype |
| spark->nuc3/exact=1/lru/prototype/k=5 | 54475.0 | 56224.1 | 60109.6 | 60109.6 | 61 | EXCEEDS; p99 weak; prototype |
| spark->nuc3/exact=1/lru/prototype/k=6 | 61273.5 | 62892.7 | 65239.1 | 65239.1 | 54 | EXCEEDS; p99 weak; prototype |
| spark->nuc3/exact=1/lru/prototype/k=7 | 72901.5 | 76183.7 | 77149.3 | 77149.3 | 43 | EXCEEDS; p99 weak; prototype |
| spark->nuc3/exact=1/lru/prototype/k=8 | 84480.1 | 88271.0 | 91103.8 | 91103.8 | 36 | EXCEEDS; p99 weak; prototype |
| spark->nuc3/exact=1/resident/k=1 | 11610.0 | 12156.3 | 12809.1 | 13565.8 | 326 | EXCEEDS; p99 weak |
| spark->nuc3/exact=1/resident/k=2 | 20746.0 | 21768.0 | 23687.7 | 23784.8 | 177 | EXCEEDS; p99 weak |
| spark->nuc3/exact=1/resident/k=3 | 31155.1 | 32362.3 | 34047.9 | 34089.5 | 115 | EXCEEDS; p99 weak |
| spark->nuc3/exact=1/resident/k=4 | 42799.4 | 43756.1 | 45717.0 | 45717.0 | 81 | EXCEEDS; p99 weak |
| spark->nuc3/exact=1/resident/k=5 | 54643.7 | 57009.8 | 75980.8 | 75980.8 | 61 | EXCEEDS; p99 weak |
| spark->nuc3/exact=1/resident/k=6 | 60273.6 | 62426.8 | 66437.3 | 66437.3 | 55 | EXCEEDS; p99 weak |
| spark->nuc3/exact=1/resident/k=7 | 70201.5 | 72749.4 | 76150.9 | 76150.9 | 45 | EXCEEDS; p99 weak |
| spark->nuc3/exact=1/resident/k=8 | 84088.8 | 85973.6 | 89340.8 | 89340.8 | 37 | EXCEEDS; p99 weak |
Bulk spark->nuc2/ethernet/bulk: 2.301 Gbit/s (one stream, one transfer; excluded from latency percentiles).
Bulk nuc3->nuc2/ethernet/bulk: 10.130 Gbit/s (one stream, one transfer; excluded from latency percentiles).
Bulk spark->nuc3/ethernet/bulk: 2.305 Gbit/s (one stream, one transfer; excluded from latency percentiles).
Bulk nuc2->nuc3/ethernet/bulk: 10.115 Gbit/s (one stream, one transfer; excluded from latency percentiles).
Bulk nuc2->nuc3/thunderbolt0/bulk: 12.132 Gbit/s (one stream, one transfer; excluded from latency percentiles).
Bulk nuc3->nuc2/thunderbolt0/bulk: 13.273 Gbit/s (one stream, one transfer; excluded from latency percentiles).
Errors: 76. Raw files remain authoritative; incomplete cases are never imputed.
NOTE: Below 1,000 samples: tail estimate is weak; rerun that section with a higher time cap. At 1,000 samples only ~10 observations cover the top 1%.
NOTE: CPU comparison absent; g53c is bench-only, no GLM CPU server exists in this checkout.
NOTE: Missing NVMe record distributions (inspect *-nvme-build.log).
NOTE: Non-JSON diagnostics present; inspect raw files.
NOTE: Spark server log absent: actual layer window and lookup wait unknown.
NOTE: cpu/server: unavailable: g53c has bench/check only; k3cs is K3 math
NOTE: nuc2-igpu.jsonl: nuc2/g53c/exact=0/bench/k=1: rc=1 
NOTE: nuc2-igpu.jsonl: nuc2/g53c/exact=0/bench/k=2: rc=1 
NOTE: nuc2-igpu.jsonl: nuc2/g53c/exact=0/bench/k=4: rc=1 
NOTE: nuc2-igpu.jsonl: nuc2/g53c/exact=0/bench/k=8: rc=1 
NOTE: nuc2-igpu.jsonl: nuc2/g53x/exact=0/bench/k=1: rc=-6 
NOTE: nuc2-igpu.jsonl: nuc2/g53x/exact=0/bench/k=2: rc=-6 
NOTE: nuc2-igpu.jsonl: nuc2/g53x/exact=0/bench/k=4: rc=-6 
NOTE: nuc2-igpu.jsonl: nuc2/g53x/exact=0/bench/k=8: rc=-6 
NOTE: nuc2-igpu.jsonl: nuc2/g53x/exact=1/bench/k=1: rc=-6 
NOTE: nuc2-igpu.jsonl: nuc2/g53x/exact=1/bench/k=2: rc=-6 
NOTE: nuc2-igpu.jsonl: nuc2/g53x/exact=1/bench/k=4: rc=-6 
NOTE: nuc2-igpu.jsonl: nuc2/g53x/exact=1/bench/k=8: rc=-6 
NOTE: nuc2-nvme.jsonl: nuc2/error: rc=127 ['/home/xero/glm53-survey/20261030T074347386706Z/glm_readbench', '/srv/glm-5.3/store/glm53-q2-v2.k3xs', 1048576, 1, 1, 1000, 4.0]
NOTE: nuc2-nvme.jsonl: nuc2/error: rc=127 ['/home/xero/glm53-survey/20261030T074347386706Z/glm_readbench', '/srv/glm-5.3/store/glm53-q2-v2.k3xs', 1048576, 1, 8, 1000, 4.0]
NOTE: nuc2-nvme.jsonl: nuc2/error: rc=127 ['/home/xero/glm53-survey/20261030T074347386706Z/glm_readbench', '/srv/glm-5.3/store/glm53-q2-v2.k3xs', 1048576, 16, 1, 1000, 4.0]
NOTE: nuc2-nvme.jsonl: nuc2/error: rc=127 ['/home/xero/glm53-survey/20261030T074347386706Z/glm_readbench', '/srv/glm-5.3/store/glm53-q2-v2.k3xs', 1048576, 16, 8, 1000, 4.0]
NOTE: nuc2-nvme.jsonl: nuc2/error: rc=127 ['/home/xero/glm53-survey/20261030T074347386706Z/glm_readbench', '/srv/glm-5.3/store/glm53-q2-v2.k3xs', 1048576, 4, 1, 1000, 4.0]
NOTE: nuc2-nvme.jsonl: nuc2/error: rc=127 ['/home/xero/glm53-survey/20261030T074347386706Z/glm_readbench', '/srv/glm-5.3/store/glm53-q2-v2.k3xs', 1048576, 4, 8, 1000, 4.0]
NOTE: nuc2-nvme.jsonl: nuc2/error: rc=127 ['/home/xero/glm53-survey/20261030T074347386706Z/glm_readbench', '/srv/glm-5.3/store/glm53-q2-v2.k3xs', 1048576, 8, 1, 1000, 4.0]
NOTE: nuc2-nvme.jsonl: nuc2/error: rc=127 ['/home/xero/glm53-survey/20261030T074347386706Z/glm_readbench', '/srv/glm-5.3/store/glm53-q2-v2.k3xs', 1048576, 8, 8, 1000, 4.0]
NOTE: nuc2-nvme.jsonl: nuc2/error: rc=127 ['/home/xero/glm53-survey/20261030T074347386706Z/glm_readbench', '/srv/glm-5.3/store/glm53-q2-v2.k3xs', 12091392, 1, 1, 1000, 4.0]
NOTE: nuc2-nvme.jsonl: nuc2/error: rc=127 ['/home/xero/glm53-survey/20261030T074347386706Z/glm_readbench', '/srv/glm-5.3/store/glm53-q2-v2.k3xs', 12091392, 1, 8, 1000, 4.0]
NOTE: nuc2-nvme.jsonl: nuc2/error: rc=127 ['/home/xero/glm53-survey/20261030T074347386706Z/glm_readbench', '/srv/glm-5.3/store/glm53-q2-v2.k3xs', 4194304, 1, 1, 1000, 4.0]
NOTE: nuc2-nvme.jsonl: nuc2/error: rc=127 ['/home/xero/glm53-survey/20261030T074347386706Z/glm_readbench', '/srv/glm-5.3/store/glm53-q2-v2.k3xs', 4194304, 1, 8, 1000, 4.0]
NOTE: nuc2-nvme.jsonl: nuc2/error: rc=127 ['/home/xero/glm53-survey/20261030T074347386706Z/glm_readbench', '/srv/glm-5.3/store/glm53-q2-v2.k3xs', 4194304, 16, 1, 1000, 4.0]
NOTE: nuc2-nvme.jsonl: nuc2/error: rc=127 ['/home/xero/glm53-survey/20261030T074347386706Z/glm_readbench', '/srv/glm-5.3/store/glm53-q2-v2.k3xs', 4194304, 16, 8, 1000, 4.0]
NOTE: nuc2-nvme.jsonl: nuc2/error: rc=127 ['/home/xero/glm53-survey/20261030T074347386706Z/glm_readbench', '/srv/glm-5.3/store/glm53-q2-v2.k3xs', 4194304, 4, 1, 1000, 4.0]
NOTE: nuc2-nvme.jsonl: nuc2/error: rc=127 ['/home/xero/glm53-survey/20261030T074347386706Z/glm_readbench', '/srv/glm-5.3/store/glm53-q2-v2.k3xs', 4194304, 4, 8, 1000, 4.0]
NOTE: nuc2-nvme.jsonl: nuc2/error: rc=127 ['/home/xero/glm53-survey/20261030T074347386706Z/glm_readbench', '/srv/glm-5.3/store/glm53-q2-v2.k3xs', 4194304, 8, 1, 1000, 4.0]
NOTE: nuc2-nvme.jsonl: nuc2/error: rc=127 ['/home/xero/glm53-survey/20261030T074347386706Z/glm_readbench', '/srv/glm-5.3/store/glm53-q2-v2.k3xs', 4194304, 8, 8, 1000, 4.0]
NOTE: nuc2-nvme.jsonl: nuc2/error: rc=127 ['/home/xero/glm53-survey/20261030T074347386706Z/glm_readbench', '/srv/glm-5.3/store/glm53-q2-v2.k3xs', 524288, 1, 1, 1000, 4.0]
NOTE: nuc2-nvme.jsonl: nuc2/error: rc=127 ['/home/xero/glm53-survey/20261030T074347386706Z/glm_readbench', '/srv/glm-5.3/store/glm53-q2-v2.k3xs', 524288, 1, 8, 1000, 4.0]
NOTE: nuc2-nvme.jsonl: nuc2/error: rc=127 ['/home/xero/glm53-survey/20261030T074347386706Z/glm_readbench', '/srv/glm-5.3/store/glm53-q2-v2.k3xs', 524288, 16, 1, 1000, 4.0]
NOTE: nuc2-nvme.jsonl: nuc2/error: rc=127 ['/home/xero/glm53-survey/20261030T074347386706Z/glm_readbench', '/srv/glm-5.3/store/glm53-q2-v2.k3xs', 524288, 16, 8, 1000, 4.0]
NOTE: nuc2-nvme.jsonl: nuc2/error: rc=127 ['/home/xero/glm53-survey/20261030T074347386706Z/glm_readbench', '/srv/glm-5.3/store/glm53-q2-v2.k3xs', 524288, 4, 1, 1000, 4.0]
NOTE: nuc2-nvme.jsonl: nuc2/error: rc=127 ['/home/xero/glm53-survey/20261030T074347386706Z/glm_readbench', '/srv/glm-5.3/store/glm53-q2-v2.k3xs', 524288, 4, 8, 1000, 4.0]
NOTE: nuc2-nvme.jsonl: nuc2/error: rc=127 ['/home/xero/glm53-survey/20261030T074347386706Z/glm_readbench', '/srv/glm-5.3/store/glm53-q2-v2.k3xs', 524288, 8, 1, 1000, 4.0]
NOTE: nuc2-nvme.jsonl: nuc2/error: rc=127 ['/home/xero/glm53-survey/20261030T074347386706Z/glm_readbench', '/srv/glm-5.3/store/glm53-q2-v2.k3xs', 524288, 8, 8, 1000, 4.0]
NOTE: nuc2/error: remote failed
NOTE: nuc2/exact=1/resident/loopback/k=1: case time cap; p99 may be undersampled
NOTE: nuc2/exact=1/resident/loopback/k=2: case time cap; p99 may be undersampled
NOTE: nuc2/exact=1/resident/loopback/k=4: case time cap; p99 may be undersampled
NOTE: nuc2/exact=1/resident/loopback/k=8: case time cap; p99 may be undersampled
NOTE: nuc2/g53c/exact=0/bench/k=1: failed
NOTE: nuc2/g53c/exact=0/bench/k=2: failed
NOTE: nuc2/g53c/exact=0/bench/k=4: failed
NOTE: nuc2/g53c/exact=0/bench/k=8: failed
NOTE: nuc2/g53x/exact=0/bench/k=1: failed
NOTE: nuc2/g53x/exact=0/bench/k=2: failed
NOTE: nuc2/g53x/exact=0/bench/k=4: failed
NOTE: nuc2/g53x/exact=0/bench/k=8: failed
NOTE: nuc2/g53x/exact=1/bench/k=1: failed
NOTE: nuc2/g53x/exact=1/bench/k=2: failed
NOTE: nuc2/g53x/exact=1/bench/k=4: failed
NOTE: nuc2/g53x/exact=1/bench/k=8: failed
NOTE: nuc3-igpu.jsonl: nuc3/g53c/exact=0/bench/k=1: rc=1 
NOTE: nuc3-igpu.jsonl: nuc3/g53c/exact=0/bench/k=2: rc=1 
NOTE: nuc3-igpu.jsonl: nuc3/g53c/exact=0/bench/k=4: rc=1 
NOTE: nuc3-igpu.jsonl: nuc3/g53c/exact=0/bench/k=8: rc=1 
NOTE: nuc3-igpu.jsonl: nuc3/g53x/exact=0/bench/k=1: rc=-6 
NOTE: nuc3-igpu.jsonl: nuc3/g53x/exact=0/bench/k=2: rc=-6 
NOTE: nuc3-igpu.jsonl: nuc3/g53x/exact=0/bench/k=4: rc=-6 
NOTE: nuc3-igpu.jsonl: nuc3/g53x/exact=0/bench/k=8: rc=-6 
NOTE: nuc3-igpu.jsonl: nuc3/g53x/exact=1/bench/k=1: rc=-6 
NOTE: nuc3-igpu.jsonl: nuc3/g53x/exact=1/bench/k=2: rc=-6 
NOTE: nuc3-igpu.jsonl: nuc3/g53x/exact=1/bench/k=4: rc=-6 
NOTE: nuc3-igpu.jsonl: nuc3/g53x/exact=1/bench/k=8: rc=-6 
NOTE: nuc3-nvme.jsonl: nuc3/error: rc=127 ['/home/xero/glm53-survey/20261030T074347386706Z/glm_readbench', '/srv/glm-5.3/store/glm53-q2-v2.k3xs', 1048576, 1, 1, 1000, 4.0]
NOTE: nuc3-nvme.jsonl: nuc3/error: rc=127 ['/home/xero/glm53-survey/20261030T074347386706Z/glm_readbench', '/srv/glm-5.3/store/glm53-q2-v2.k3xs', 1048576, 1, 8, 1000, 4.0]
NOTE: nuc3-nvme.jsonl: nuc3/error: rc=127 ['/home/xero/glm53-survey/20261030T074347386706Z/glm_readbench', '/srv/glm-5.3/store/glm53-q2-v2.k3xs', 1048576, 16, 1, 1000, 4.0]
NOTE: nuc3-nvme.jsonl: nuc3/error: rc=127 ['/home/xero/glm53-survey/20261030T074347386706Z/glm_readbench', '/srv/glm-5.3/store/glm53-q2-v2.k3xs', 1048576, 16, 8, 1000, 4.0]
NOTE: nuc3-nvme.jsonl: nuc3/error: rc=127 ['/home/xero/glm53-survey/20261030T074347386706Z/glm_readbench', '/srv/glm-5.3/store/glm53-q2-v2.k3xs', 1048576, 4, 1, 1000, 4.0]
NOTE: nuc3-nvme.jsonl: nuc3/error: rc=127 ['/home/xero/glm53-survey/20261030T074347386706Z/glm_readbench', '/srv/glm-5.3/store/glm53-q2-v2.k3xs', 1048576, 4, 8, 1000, 4.0]
NOTE: nuc3-nvme.jsonl: nuc3/error: rc=127 ['/home/xero/glm53-survey/20261030T074347386706Z/glm_readbench', '/srv/glm-5.3/store/glm53-q2-v2.k3xs', 1048576, 8, 1, 1000, 4.0]
NOTE: nuc3-nvme.jsonl: nuc3/error: rc=127 ['/home/xero/glm53-survey/20261030T074347386706Z/glm_readbench', '/srv/glm-5.3/store/glm53-q2-v2.k3xs', 1048576, 8, 8, 1000, 4.0]
NOTE: nuc3-nvme.jsonl: nuc3/error: rc=127 ['/home/xero/glm53-survey/20261030T074347386706Z/glm_readbench', '/srv/glm-5.3/store/glm53-q2-v2.k3xs', 12091392, 1, 1, 1000, 4.0]
NOTE: nuc3-nvme.jsonl: nuc3/error: rc=127 ['/home/xero/glm53-survey/20261030T074347386706Z/glm_readbench', '/srv/glm-5.3/store/glm53-q2-v2.k3xs', 12091392, 1, 8, 1000, 4.0]
NOTE: nuc3-nvme.jsonl: nuc3/error: rc=127 ['/home/xero/glm53-survey/20261030T074347386706Z/glm_readbench', '/srv/glm-5.3/store/glm53-q2-v2.k3xs', 4194304, 1, 1, 1000, 4.0]
NOTE: nuc3-nvme.jsonl: nuc3/error: rc=127 ['/home/xero/glm53-survey/20261030T074347386706Z/glm_readbench', '/srv/glm-5.3/store/glm53-q2-v2.k3xs', 4194304, 1, 8, 1000, 4.0]
NOTE: nuc3-nvme.jsonl: nuc3/error: rc=127 ['/home/xero/glm53-survey/20261030T074347386706Z/glm_readbench', '/srv/glm-5.3/store/glm53-q2-v2.k3xs', 4194304, 16, 1, 1000, 4.0]
NOTE: nuc3-nvme.jsonl: nuc3/error: rc=127 ['/home/xero/glm53-survey/20261030T074347386706Z/glm_readbench', '/srv/glm-5.3/store/glm53-q2-v2.k3xs', 4194304, 16, 8, 1000, 4.0]
NOTE: nuc3-nvme.jsonl: nuc3/error: rc=127 ['/home/xero/glm53-survey/20261030T074347386706Z/glm_readbench', '/srv/glm-5.3/store/glm53-q2-v2.k3xs', 4194304, 4, 1, 1000, 4.0]
NOTE: nuc3-nvme.jsonl: nuc3/error: rc=127 ['/home/xero/glm53-survey/20261030T074347386706Z/glm_readbench', '/srv/glm-5.3/store/glm53-q2-v2.k3xs', 4194304, 4, 8, 1000, 4.0]
NOTE: nuc3-nvme.jsonl: nuc3/error: rc=127 ['/home/xero/glm53-survey/20261030T074347386706Z/glm_readbench', '/srv/glm-5.3/store/glm53-q2-v2.k3xs', 4194304, 8, 1, 1000, 4.0]
NOTE: nuc3-nvme.jsonl: nuc3/error: rc=127 ['/home/xero/glm53-survey/20261030T074347386706Z/glm_readbench', '/srv/glm-5.3/store/glm53-q2-v2.k3xs', 4194304, 8, 8, 1000, 4.0]
NOTE: nuc3-nvme.jsonl: nuc3/error: rc=127 ['/home/xero/glm53-survey/20261030T074347386706Z/glm_readbench', '/srv/glm-5.3/store/glm53-q2-v2.k3xs', 524288, 1, 1, 1000, 4.0]
NOTE: nuc3-nvme.jsonl: nuc3/error: rc=127 ['/home/xero/glm53-survey/20261030T074347386706Z/glm_readbench', '/srv/glm-5.3/store/glm53-q2-v2.k3xs', 524288, 1, 8, 1000, 4.0]
NOTE: nuc3-nvme.jsonl: nuc3/error: rc=127 ['/home/xero/glm53-survey/20261030T074347386706Z/glm_readbench', '/srv/glm-5.3/store/glm53-q2-v2.k3xs', 524288, 16, 1, 1000, 4.0]
NOTE: nuc3-nvme.jsonl: nuc3/error: rc=127 ['/home/xero/glm53-survey/20261030T074347386706Z/glm_readbench', '/srv/glm-5.3/store/glm53-q2-v2.k3xs', 524288, 16, 8, 1000, 4.0]
NOTE: nuc3-nvme.jsonl: nuc3/error: rc=127 ['/home/xero/glm53-survey/20261030T074347386706Z/glm_readbench', '/srv/glm-5.3/store/glm53-q2-v2.k3xs', 524288, 4, 1, 1000, 4.0]
NOTE: nuc3-nvme.jsonl: nuc3/error: rc=127 ['/home/xero/glm53-survey/20261030T074347386706Z/glm_readbench', '/srv/glm-5.3/store/glm53-q2-v2.k3xs', 524288, 4, 8, 1000, 4.0]
NOTE: nuc3-nvme.jsonl: nuc3/error: rc=127 ['/home/xero/glm53-survey/20261030T074347386706Z/glm_readbench', '/srv/glm-5.3/store/glm53-q2-v2.k3xs', 524288, 8, 1, 1000, 4.0]
NOTE: nuc3-nvme.jsonl: nuc3/error: rc=127 ['/home/xero/glm53-survey/20261030T074347386706Z/glm_readbench', '/srv/glm-5.3/store/glm53-q2-v2.k3xs', 524288, 8, 8, 1000, 4.0]
NOTE: nuc3/error: remote failed
NOTE: nuc3/exact=1/resident/loopback/k=1: case time cap; p99 may be undersampled
NOTE: nuc3/exact=1/resident/loopback/k=2: case time cap; p99 may be undersampled
NOTE: nuc3/exact=1/resident/loopback/k=4: case time cap; p99 may be undersampled
NOTE: nuc3/exact=1/resident/loopback/k=8: case time cap; p99 may be undersampled
NOTE: nuc3/g53c/exact=0/bench/k=1: failed
NOTE: nuc3/g53c/exact=0/bench/k=2: failed
NOTE: nuc3/g53c/exact=0/bench/k=4: failed
NOTE: nuc3/g53c/exact=0/bench/k=8: failed
NOTE: nuc3/g53x/exact=0/bench/k=1: failed
NOTE: nuc3/g53x/exact=0/bench/k=2: failed
NOTE: nuc3/g53x/exact=0/bench/k=4: failed
NOTE: nuc3/g53x/exact=0/bench/k=8: failed
NOTE: nuc3/g53x/exact=1/bench/k=1: failed
NOTE: nuc3/g53x/exact=1/bench/k=2: failed
NOTE: nuc3/g53x/exact=1/bench/k=4: failed
NOTE: nuc3/g53x/exact=1/bench/k=8: failed
NOTE: session: READY
NOTE: spark->nuc2/exact=0/cold/prototype/k=1: case time cap; p99 may be undersampled
NOTE: spark->nuc2/exact=0/cold/prototype/k=2: case time cap; p99 may be undersampled
NOTE: spark->nuc2/exact=0/cold/prototype/k=3: case time cap; p99 may be undersampled
NOTE: spark->nuc2/exact=0/cold/prototype/k=4: case time cap; p99 may be undersampled
NOTE: spark->nuc2/exact=0/cold/prototype/k=5: case time cap; p99 may be undersampled
NOTE: spark->nuc2/exact=0/cold/prototype/k=6: case time cap; p99 may be undersampled
NOTE: spark->nuc2/exact=0/cold/prototype/k=7: case time cap; p99 may be undersampled
NOTE: spark->nuc2/exact=0/cold/prototype/k=8: case time cap; p99 may be undersampled
NOTE: spark->nuc2/exact=0/lru/prototype/k=5: case time cap; p99 may be undersampled
NOTE: spark->nuc2/exact=0/lru/prototype/k=6: case time cap; p99 may be undersampled
NOTE: spark->nuc2/exact=0/lru/prototype/k=7: case time cap; p99 may be undersampled
NOTE: spark->nuc2/exact=0/lru/prototype/k=8: case time cap; p99 may be undersampled
NOTE: spark->nuc2/exact=0/resident/k=5: case time cap; p99 may be undersampled
NOTE: spark->nuc2/exact=0/resident/k=6: case time cap; p99 may be undersampled
NOTE: spark->nuc2/exact=0/resident/k=7: case time cap; p99 may be undersampled
NOTE: spark->nuc2/exact=0/resident/k=8: case time cap; p99 may be undersampled
NOTE: spark->nuc2/exact=1/cold/prototype/k=1: case time cap; p99 may be undersampled
NOTE: spark->nuc2/exact=1/cold/prototype/k=2: case time cap; p99 may be undersampled
NOTE: spark->nuc2/exact=1/cold/prototype/k=3: case time cap; p99 may be undersampled
NOTE: spark->nuc2/exact=1/cold/prototype/k=4: case time cap; p99 may be undersampled
NOTE: spark->nuc2/exact=1/cold/prototype/k=5: case time cap; p99 may be undersampled
NOTE: spark->nuc2/exact=1/cold/prototype/k=6: case time cap; p99 may be undersampled
NOTE: spark->nuc2/exact=1/cold/prototype/k=7: case time cap; p99 may be undersampled
NOTE: spark->nuc2/exact=1/cold/prototype/k=8: case time cap; p99 may be undersampled
NOTE: spark->nuc2/exact=1/lru/prototype/k=1: case time cap; p99 may be undersampled
NOTE: spark->nuc2/exact=1/lru/prototype/k=2: case time cap; p99 may be undersampled
NOTE: spark->nuc2/exact=1/lru/prototype/k=3: case time cap; p99 may be undersampled
NOTE: spark->nuc2/exact=1/lru/prototype/k=4: case time cap; p99 may be undersampled
NOTE: spark->nuc2/exact=1/lru/prototype/k=5: case time cap; p99 may be undersampled
NOTE: spark->nuc2/exact=1/lru/prototype/k=6: case time cap; p99 may be undersampled
NOTE: spark->nuc2/exact=1/lru/prototype/k=7: case time cap; p99 may be undersampled
NOTE: spark->nuc2/exact=1/lru/prototype/k=8: case time cap; p99 may be undersampled
NOTE: spark->nuc2/exact=1/resident/k=1: case time cap; p99 may be undersampled
NOTE: spark->nuc2/exact=1/resident/k=2: case time cap; p99 may be undersampled
NOTE: spark->nuc2/exact=1/resident/k=3: case time cap; p99 may be undersampled
NOTE: spark->nuc2/exact=1/resident/k=4: case time cap; p99 may be undersampled
NOTE: spark->nuc2/exact=1/resident/k=5: case time cap; p99 may be undersampled
NOTE: spark->nuc2/exact=1/resident/k=6: case time cap; p99 may be undersampled
NOTE: spark->nuc2/exact=1/resident/k=7: case time cap; p99 may be undersampled
NOTE: spark->nuc2/exact=1/resident/k=8: case time cap; p99 may be undersampled
NOTE: spark->nuc3/exact=0/cold/prototype/k=1: case time cap; p99 may be undersampled
NOTE: spark->nuc3/exact=0/cold/prototype/k=2: case time cap; p99 may be undersampled
NOTE: spark->nuc3/exact=0/cold/prototype/k=3: case time cap; p99 may be undersampled
NOTE: spark->nuc3/exact=0/cold/prototype/k=4: case time cap; p99 may be undersampled
NOTE: spark->nuc3/exact=0/cold/prototype/k=5: case time cap; p99 may be undersampled
NOTE: spark->nuc3/exact=0/cold/prototype/k=6: case time cap; p99 may be undersampled
NOTE: spark->nuc3/exact=0/cold/prototype/k=7: case time cap; p99 may be undersampled
NOTE: spark->nuc3/exact=0/cold/prototype/k=8: case time cap; p99 may be undersampled
NOTE: spark->nuc3/exact=0/lru/prototype/k=5: case time cap; p99 may be undersampled
NOTE: spark->nuc3/exact=0/lru/prototype/k=6: case time cap; p99 may be undersampled
NOTE: spark->nuc3/exact=0/lru/prototype/k=7: case time cap; p99 may be undersampled
NOTE: spark->nuc3/exact=0/lru/prototype/k=8: case time cap; p99 may be undersampled
NOTE: spark->nuc3/exact=0/resident/k=5: case time cap; p99 may be undersampled
NOTE: spark->nuc3/exact=0/resident/k=6: case time cap; p99 may be undersampled
NOTE: spark->nuc3/exact=0/resident/k=7: case time cap; p99 may be undersampled
NOTE: spark->nuc3/exact=0/resident/k=8: case time cap; p99 may be undersampled
NOTE: spark->nuc3/exact=1/cold/prototype/k=1: case time cap; p99 may be undersampled
NOTE: spark->nuc3/exact=1/cold/prototype/k=2: case time cap; p99 may be undersampled
NOTE: spark->nuc3/exact=1/cold/prototype/k=3: case time cap; p99 may be undersampled
NOTE: spark->nuc3/exact=1/cold/prototype/k=4: case time cap; p99 may be undersampled
NOTE: spark->nuc3/exact=1/cold/prototype/k=5: case time cap; p99 may be undersampled
NOTE: spark->nuc3/exact=1/cold/prototype/k=6: case time cap; p99 may be undersampled
NOTE: spark->nuc3/exact=1/cold/prototype/k=7: case time cap; p99 may be undersampled
NOTE: spark->nuc3/exact=1/cold/prototype/k=8: case time cap; p99 may be undersampled
NOTE: spark->nuc3/exact=1/lru/prototype/k=1: case time cap; p99 may be undersampled
NOTE: spark->nuc3/exact=1/lru/prototype/k=2: case time cap; p99 may be undersampled
NOTE: spark->nuc3/exact=1/lru/prototype/k=3: case time cap; p99 may be undersampled
NOTE: spark->nuc3/exact=1/lru/prototype/k=4: case time cap; p99 may be undersampled
NOTE: spark->nuc3/exact=1/lru/prototype/k=5: case time cap; p99 may be undersampled
NOTE: spark->nuc3/exact=1/lru/prototype/k=6: case time cap; p99 may be undersampled
NOTE: spark->nuc3/exact=1/lru/prototype/k=7: case time cap; p99 may be undersampled
NOTE: spark->nuc3/exact=1/lru/prototype/k=8: case time cap; p99 may be undersampled
NOTE: spark->nuc3/exact=1/resident/k=1: case time cap; p99 may be undersampled
NOTE: spark->nuc3/exact=1/resident/k=2: case time cap; p99 may be undersampled
NOTE: spark->nuc3/exact=1/resident/k=3: case time cap; p99 may be undersampled
NOTE: spark->nuc3/exact=1/resident/k=4: case time cap; p99 may be undersampled
NOTE: spark->nuc3/exact=1/resident/k=5: case time cap; p99 may be undersampled
NOTE: spark->nuc3/exact=1/resident/k=6: case time cap; p99 may be undersampled
NOTE: spark->nuc3/exact=1/resident/k=7: case time cap; p99 may be undersampled
NOTE: spark->nuc3/exact=1/resident/k=8: case time cap; p99 may be undersampled
