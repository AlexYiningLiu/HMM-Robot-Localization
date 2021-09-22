# HMM-Robot-Localization
Course project that implements the Viterbi and Forward-Backward algorithms for tracking a robot's location on a grid. 

File ```rover.py``` contains functions for generating the initial distribution, the transition probabilities given a current hidden state, and the
observation probabilities given a current hidden state.

Files ```test.txt``` and ```test_missing.txt``` contain data. the first three columns correspond to the hidden states, and the last two columns correspond to the observations.

A visualization tool is provided by ```graphics.py```, which can be turned on by setting ```enable_graphics``` to True in ```inference.py```.

Run ```inference.py``` to observe results. 
