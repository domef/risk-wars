# RisiKo-probabilities-calculator

#### Description

This is a probabilities calculator that, given the armies size of the attacker and the defender, will calculate the **probability of winning the battle** and the **expected losses of the attacker** using _Markov chain_. Using command line arguments the CPU computation can be boosted and the heatmaps of the two values can be plotted.
 
#### Usage
 
Program usage:
```
optional arguments:
 -h, --help   show this help message and exit
 -b, --boost  boost CPU computation
 -p, --plot   plot the heatmaps
 ```
 
#### Dependencies

Required libraries:
 - numpy
 - matplotlib (optional): used with _--plot_
 - pytorch (optional): used with _--boost_
