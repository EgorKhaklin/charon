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
