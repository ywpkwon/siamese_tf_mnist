# Implementing Siamese network using Tensorflow with MNIST example

I have been interested in Siamese network. To my understading, it is one way of dealing with weakly supuervised problems. Its beauty lies in its simple scheme. It seems Siamese networks ( and Triplet network) have been popularly used in many applications such as face similarity and image matching .

Here, I implement a simple Siamese example. It embeds hand-written digits into 2D space. In other words, it embeds 28$$$\times$$$28 image (a data point in 794D) into a point in 2D, i.e., $$$ x\in \mathbb{R}^{794} \rightarrow y\in \mathbb{R}^2 $$$. A loss function controls the embedding to be closer for guys in the same class and futher for guys in the different classes.

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
This will download and extract MNIST dataset (once downloaded, it will skip downloading next time).

Result looks like this.

![here](https://github.com/ywpkwon/siamese_tf_mnist/result.png)

It will save the final result in `embed.txt`. If you just want to visualize it again, then run simply

```bash
$ python visualize.py
```

For visualizion and network architecture design, I refered [here](http://andersbll.github.io/deeppy-website/examples/siamese_mnist.html). 

