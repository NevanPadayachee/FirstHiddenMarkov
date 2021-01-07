import tensorflow_probability as tfp
import tensorflow as tf

#makes a shortcut for later on
tfd = tfp.distributions
#Refers to point 2 above
initial_distribution = tfd.Categorical(probs=[0.8, 0.2])
#refers to point 3, 4
transitional_distribution = tfd.Categorical(probs=[[0.7, 0.3], [0.2, 0.8]])
#refers to point 5, the loc argument represents the mean and the scale is the standard deviation
observation_distribution = tfd.Normal(loc=[0., 15.], scale=[5., 10.])

#num_Steps is the number of days we want to predict for
model = tfd.HiddenMarkovModel(initial_distribution=initial_distribution,
                              transition_distribution=transitional_distribution,
                              observation_distribution=observation_distribution,
                              num_steps=7)

#due to the way tensorflow works on a lower level we need to evaluate part of the graph
#from within a session to see the value of this tensor
mean=model.mean()

with tf.compat.v1.Session() as sess:
    print(mean.numpy())

