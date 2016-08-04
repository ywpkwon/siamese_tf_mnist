# Implementing Siamese network using Tensorflow with MNIST example

<p align="center"> !
[result](https://github.com/ywpkwon/siamese_tf_mnist/result.png) 
</p>

I have been interested in Siamese network. To my understanding, it is one way of dealing with weakly supervised problems. Its beauty lies in its simple scheme. It seems Siamese networks ( and Triplet network) have been popularly used in many applications such as face similarity and image matching .

Here, I implement a simple Siamese example. It embeds hand-written digits into 2D space. A loss function controls the embedding to be closer for guys in the same class and further for guys in the different classes.

I tried to keep codes simple, including

* `run.py` : nothing but a wrapper for running.
* `inference.py` :  architecture and loss definition. you can modify as you want.
* `visualize.py` : for visualizing result.

You can run simply

```bash
$ python run.py
...
step 34740: loss 0.120
step 34750: loss 0.179
step 34760: loss 0.113
step 34770: loss 0.078
...
```
This will download and extract MNIST dataset (once downloaded, it will skip downloading next time). The result will look like the image above.

It will save the final embedding in `embed.txt`. So, if you just want to visualize it again, then run simply

```bash
$ python visualize.py
```

For visualization and network architecture design, I refer [here](http://andersbll.github.io/deeppy-website/examples/siamese_mnist.html). 

