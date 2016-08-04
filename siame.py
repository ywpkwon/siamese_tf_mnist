class siamese:

    # Create model
    self.x1 = tf.placeholder(tf.float32, [None, 784])
    self.x2 = tf.placeholder(tf.float32, [None, 784])

    with tf.variable_scope("siamese") as scope:
        o1, weights1 = network(self.x1)
        scope.reuse_variables()
        o2, weights2 = network(self.x2)

    # Create loss
    self.y_ = tf.placeholder(tf.float32, [None])
    self.loss = loss_with_step(o1, o2, y_)

    def network(self, x):
        weights = []
        fc1, w1 = fc_layer(x, 1024, "fc1")
        ac1 = tf.nn.relu(fc1)
        fc2, w2 = fc_layer(ac1, 1024, "fc2")
        ac2 = tf.nn.relu(fc2)
        fc3, w3 = fc_layer(ac2, 2, "fc3")
        weights += w1+w2+w3
        return fc3, weights

    def fc_layer(self, bottom, n_weight, name):
        assert len(bottom.get_shape()) == 2
        n_prev_weight = bottom.get_shape()[1]
        initer = tf.truncated_normal_initializer(stddev=0.01)
        W = tf.get_variable(name+'W', dtype=tf.float32, shape=[n_prev_weight, n_weight], initializer=initer)
        b = tf.get_variable(name+'b', dtype=tf.float32, initializer=tf.constant(0.01, shape=[n_weight], dtype=tf.float32))
        fc = tf.nn.bias_add(tf.matmul(bottom, W), b)
        return fc, [W, b]

    def loss_with_spring(self, o1, o2, y_):
        margin = 5.0
        labels_t = y_
        labels_f = tf.sub(1.0, y_, name="1-yi")          # labels_ = !labels;
        eucd2 = tf.pow(tf.sub(o1, o2), 2)
        eucd2 = tf.reduce_sum(eucd2, 1)
        eucd = tf.sqrt(eucd2+1e-6, name="eucd")
        C = tf.constant(margin, name="C")
        # yi*||CNN(p1i)-CNN(p2i)||^2 + (1-yi)*max(0, C-||CNN(p1i)-CNN(p2i)||^2)
        pos = tf.mul(labels_t, eucd2, name="yi_x_eucd2")
        # neg = tf.mul(labels_f, tf.sub(0.0,eucd2), name="yi_x_eucd2")
        # neg = tf.mul(labels_f, tf.maximum(0.0, tf.sub(C,eucd2)), name="Nyi_x_C-eucd_xx_2")
        neg = tf.mul(labels_f, tf.pow(tf.maximum(tf.sub(C, eucd), 0), 2), name="Nyi_x_C-eucd_xx_2")
        losses = tf.add(pos, neg, name="losses")
        loss = tf.reduce_mean(losses, name="loss")
        return loss

    def loss_with_step(self, o1, o2, y_):
        margin = 5.0
        labels_t = y_
        labels_f = tf.sub(1.0, y_, name="1-yi")          # labels_ = !labels;
        eucd2 = tf.pow(tf.sub(o1, o2), 2)
        eucd2 = tf.reduce_sum(eucd2, 1)
        eucd = tf.sqrt(eucd2+1e-6, name="eucd")
        C = tf.constant(margin, name="C")
        pos = tf.mul(labels_t, eucd, name="y_x_eucd")
        neg = tf.mul(labels_f, tf.maximum(0.0, tf.sub(C, eucd)), name="Ny_C-eucd")
        losses = tf.add(pos, neg, name="losses")
        loss = tf.reduce_mean(losses, name="loss")
        return loss