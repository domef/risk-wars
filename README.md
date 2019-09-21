# RisiKo-probabilities-calculator

#### Description

This is a probabilities calculator that, given the armies size of the attacker and the defender, will calculate the **probability of winning the battle** and the **expected losses of the attacker** using _Markov chain_. At the end the _heatmap_ of the two computed values will be plotted.
 
#### Usage
 
Program usage:
```
optional arguments:
 -h, --help   show this help message and exit
 -b, --boost  Boost CPU computation with PyTorch
 -p, --plot   Plot the heatmaps
 ```
 
#### Dependencies

Required libraries:
 - numpy
 - matplotlib (optional): used with _--plot_
 - pytorch (optional): used with _--boost_
