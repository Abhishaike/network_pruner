This is Keras code that implements weight and neuron pruning algorithms for a toy network for an MNIST dataset. 

It will fit a network on the MNIST dataset and apply both pruning techniques to a fitted version of the network. It will plot the accuracy of these techniques on the same plot for three different percentile intervals for pruning: [0, 25, 50, 60, 70, 80, 90, 95, 97, 99], at every 1%, and at every .1%. It will save each of these plots underneath 'Normal', '0_100_1', and '0_100_.1', respectively. Basically everything about this is modular, from the algorithms to the model architecture init, so there's a lot of freedom here. 

To run this, first clone it and run ```pip install -r requirements.txt``` in bash (in a virtual environment if you want) to get all of the requirments. I'm using tensorflow-gpu on my end, but this repo should work for running with vanilla tensorflow as well. 

Then, simply run ```python3 training_testing.py```. This will fit the model and evaulate it on the above percentile ranges, and save the created plots. 

For a more detailed writeup on conclusions from this, read here: https://docs.google.com/document/d/1FnW1PAhQvCZPj3hRIJPhYBc2kc-U4Fx9gvRq0ABDYuU/edit?usp=sharing

