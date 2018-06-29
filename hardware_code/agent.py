"""An agent that can restore and run a policy learned by PPO."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import normalize
import utility
import numpy as np


class SimplePPOPolicy(object):
  """A simple PPO policy that is independent to the PPO infrastructure.

  This class restores the policy network from a tensorflow checkpoint that was
  learned from PPO training. 
  """

  def __init__(self, sess, network, policy_layers, value_layers,
               checkpoint):
    self.sess = sess
    observation_size = 28
    action_size = 8
    self.observation_placeholder = tf.placeholder(
        tf.float32, [None, observation_size], name="Input")
    self._observ_filter = normalize.StreamingNormalize(
        self.observation_placeholder[0],
        center=True,
        scale=True,
        clip=5,
        name="normalize_observ")
    self._restore_policy(
        network,
        policy_layers=policy_layers,
        value_layers=value_layers,
        action_size=action_size,
        checkpoint=checkpoint)

  def _restore_policy(self, network, policy_layers, value_layers, action_size,
                      checkpoint):
    """Restore the PPO policy from a TensorFlow checkpoint.

    Args:
      network: The neural network definition.
      policy_layers: A tuple specify the number of layers and number of neurons
        of each layer for the policy network.
      value_layers: A tuple specify the number of layers and number of neurons
        of each layer for the value network.
      action_size: The dimension of the action space.
      checkpoint: The checkpoint path.
    """
    observ = self._observ_filter.transform(self.observation_placeholder)
    with tf.variable_scope("network/rnn"):
      self.network = network(
          policy_layers=policy_layers,
          value_layers=value_layers,
          action_size=action_size)

    with tf.variable_scope("temporary"):
      self.last_state = tf.Variable(
          self.network.zero_state(1, tf.float32), False)
      self.sess.run(self.last_state.initializer)

    with tf.variable_scope("network"):
      (mean_action, _, _), new_state = tf.nn.dynamic_rnn(
          self.network,
          observ[:, None],
          tf.ones(1),
          self.last_state,
          tf.float32,
          swap_memory=True)
      self.mean_action = mean_action
      self.update_state = self.last_state.assign(new_state)

    saver = utility.define_saver(exclude=(r"temporary/.*",))
    saver.restore(self.sess, checkpoint)

  def get_action(self, observation):
    normalized_observation = self._normalize_observ(observation)
    normalized_action, _ = self.sess.run(
        [self.mean_action, self.update_state],
        feed_dict={self.observation_placeholder: normalized_observation})
    action = self._denormalize_action(normalized_action)
    return action[:, 0]

  def _denormalize_action(self, action):
    min_ = np.array([-1., -1., -1., -1., -1., -1., -1., -1.])
    max_ = np.array([1., 1., 1., 1., 1., 1., 1., 1.])
    action = (action + 1) / 2 * (max_ - min_) + min_
    return action

  def _normalize_observ(self, observ):
    min_ = np.array([  -1.5807964,   -1.5807964,   -1.5807964,   -1.5807964,   -1.5807964,
   -1.5807964,   -1.5807964,   -1.5807964, -167.72488,   -167.72488,
 -167.72488,   -167.72488,   -167.72488,   -167.72488,   -167.72488,
 -167.72488,     -5.71,        -5.71,        -5.71,        -5.71,
   -5.71,        -5.71,        -5.71,        -5.71,        -1.01,
   -1.01,        -1.01,        -1.01     ])
    max_ = np.array([  1.5807964,   1.5807964,   1.5807964,   1.5807964,   1.5807964,   1.5807964,
   1.5807964,   1.5807964, 167.72488,   167.72488,   167.72488,   167.72488,
 167.72488,   167.72488,   167.72488,   167.72488,     5.71,        5.71,
   5.71,        5.71,        5.71,        5.71,        5.71,        5.71,
   1.01,        1.01,        1.01,        1.01     ])
    observ = 2 * (observ - min_) / (max_ - min_) - 1
    return observ
