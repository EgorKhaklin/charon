| transform | lr (x stability limit) | init | median error vs true w | mean error | median residual after 6000 steps |
|---|---|---|---|---|---|
| w (no wormhole) | 1 | 0.01 | 0.901 | 0.897 | 2e-17 |
| w|w| | 0.3 | 0.1 | 0.639 | 0.625 | 3e-04 |
| w|w| o w|w| (2 deep) | 1 | 0.0001 | 0.104 | 0.168 | 2e-02 |
| w|w| o w|w| o w|w| (3 deep) | 0.3 | 0.01 | 0.950 | 0.910 | 6e-01 |
| 3tanh(w/3) o w|w| | 0.3 | 0.1 | 0.645 | 0.632 | 7e-05 |
| sinh o w|w| | 0.3 | 0.1 | 0.621 | 0.605 | 5e-03 |
| w|w| o sinh | 0.3 | 0.1 | 0.609 | 0.593 | 6e-03 |

| problem | run | steps to residual 1e-8 | error vs true w | L1 norm | true L1 norm |
|---|---|---|---|---|---|
| 0 | init 0.1 | 21200 | 0.649 | 11.83 | 7.81 |
| 0 | init 0.0001 | over 400000 | 0.028 | 7.88 | 7.81 |
| 0 | init 0.1, jump to u*v = 1e-4 every 200 steps | 36800 | 0.648 | 11.58 | 7.81 |
| 0 | init 0.1, jump to u*v = 1e-8 every 200 steps | 36800 | 0.648 | 11.58 | 7.81 |
| 1 | init 0.1 | 26200 | 0.429 | 12.54 | 8.14 |
| 1 | init 0.0001 | over 400000 | 0.002 | 8.15 | 8.14 |
| 1 | init 0.1, jump to u*v = 1e-4 every 200 steps | 43800 | 0.404 | 12.13 | 8.14 |
| 1 | init 0.1, jump to u*v = 1e-8 every 200 steps | 43800 | 0.404 | 12.13 | 8.14 |
| 2 | init 0.1 | 27500 | 0.528 | 12.52 | 8.34 |
| 2 | init 0.0001 | over 400000 | 0.005 | 8.36 | 8.34 |
| 2 | init 0.1, jump to u*v = 1e-4 every 200 steps | 46100 | 0.519 | 12.34 | 8.34 |
| 2 | init 0.1, jump to u*v = 1e-8 every 200 steps | 46200 | 0.519 | 12.34 | 8.34 |

| wormhole | problem | predicted fit residual | predicted error vs true w | GD vs predicted, lr 0.25 | GD vs predicted, lr 0.025 |
|---|---|---|---|---|---|
| u^2 - v^2 metric, a = 0.01 | 0 | 9e-14 | 0.5850 | 2.4e-02 | 2.6e-03 |
| u^2 - v^2 metric, a = 0.01 | 1 | 4e-15 | 0.3229 | 2.5e-02 | 2.6e-03 |
| u^2 - v^2 metric, a = 0.01 | 2 | 2e-13 | 0.4323 | 2.9e-02 | 3.1e-03 |
| u^2 - v^2 metric, a = 0.1 | 0 | 4e-13 | 0.8081 | 1.8e-02 | 1.8e-03 |
| u^2 - v^2 metric, a = 0.1 | 1 | 2e-13 | 0.7054 | 2.2e-02 | 2.3e-03 |
| u^2 - v^2 metric, a = 0.1 | 2 | 2e-13 | 0.7469 | 2.1e-02 | 2.2e-03 |
| log wormhole, f = 1e-3, tau = 3e-3 | 0 | 5e-15 | 0.0000 | 1.5e-04 | 1.3e-04 |
| log wormhole, f = 1e-3, tau = 3e-3 | 1 | 8e-15 | 0.0000 | 2.9e-05 | 2.8e-05 |
| log wormhole, f = 1e-3, tau = 3e-3 | 2 | 1e-13 | 0.0000 | 4.3e-05 | 3.4e-05 |

| nonzero sizes | nonzeros | wormhole | wormhole exact (of 100) | L1 exact | wormhole only | L1 only |
|---|---|---|---|---|---|---|
| all 1.5 | 6 | log wormhole (f 1e-4, tau 1e-3) | 91 | 98 | 0 | 7 |
| all 1.5 | 6 | u^2 - v^2 metric (a 1e-3) | 0 | 98 | 0 | 98 |
| all 1.5 | 8 | log wormhole (f 1e-4, tau 1e-3) | 67 | 81 | 0 | 14 |
| all 1.5 | 8 | u^2 - v^2 metric (a 1e-3) | 0 | 81 | 0 | 81 |
| all 1.5 | 10 | log wormhole (f 1e-4, tau 1e-3) | 35 | 53 | 0 | 18 |
| all 1.5 | 10 | u^2 - v^2 metric (a 1e-3) | 0 | 53 | 0 | 53 |
| all 1.5 | 12 | log wormhole (f 1e-4, tau 1e-3) | 6 | 13 | 0 | 7 |
| all 1.5 | 12 | u^2 - v^2 metric (a 1e-3) | 0 | 13 | 0 | 13 |
| uniform 1 to 2 | 6 | log wormhole (f 1e-4, tau 1e-3) | 92 | 98 | 0 | 6 |
| uniform 1 to 2 | 6 | u^2 - v^2 metric (a 1e-3) | 0 | 98 | 0 | 98 |
| uniform 1 to 2 | 8 | log wormhole (f 1e-4, tau 1e-3) | 65 | 81 | 0 | 16 |
| uniform 1 to 2 | 8 | u^2 - v^2 metric (a 1e-3) | 0 | 81 | 0 | 81 |
| uniform 1 to 2 | 10 | log wormhole (f 1e-4, tau 1e-3) | 34 | 53 | 0 | 19 |
| uniform 1 to 2 | 10 | u^2 - v^2 metric (a 1e-3) | 0 | 53 | 0 | 53 |
| uniform 1 to 2 | 12 | log wormhole (f 1e-4, tau 1e-3) | 6 | 13 | 0 | 7 |
| uniform 1 to 2 | 12 | u^2 - v^2 metric (a 1e-3) | 0 | 13 | 0 | 13 |
