# RisiKo-probabilities-calculator

#### Description

This is a probabilities calculator that, given the armies size of the attacker and the defender, will calculate the **probability of winning the battle** and the **expected losses of the attacker** using a _Markov stochastic process_. Use command line arguments to plot the ptobability heatmap.

The project is based on Markov Chains for the RISK Board Game Revisited

@article{10.2307/3219306,
 ISSN = {0025570X, 19300980},
 URL = {http://www.jstor.org/stable/3219306},
 author = {Jason A. Osborne},
 journal = {Mathematics Magazine},
 number = {2},
 pages = {129--135},
 publisher = {Mathematical Association of America},
 title = {Markov Chains for the RISK Board Game Revisited},
 volume = {76},
 year = {2003}
}
 
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
