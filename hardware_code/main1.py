from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import tensorflow as tf
import utility
import agent as ppo_agent
import numpy as np

import paho.mqtt.client as mqtt
import logging
import random
log_filename1="test1.log"
log_filename2="test2.log"


MQTT_SERVER = "localhost" #IP of the broker/server
MQTT_PATH_1 = "test_channel_1"
MQTT_PATH_2 = "test_channel_2"

flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS
flags.DEFINE_string("logdir", "/home/abhimanyu/mqTT/hardware_code/weights_file",
                        "The directory that contains checkpoint and config.")
flags.DEFINE_string("checkpoint", "model.ckpt-8800020", "The checkpoint file path.")

class RLAlgorithm(object):
  """docstring for RLAlgorithm"""
  def __init__(self):
    config = utility.load_config(FLAGS.logdir)
    policy_layers = config.policy_layers
    value_layers = config.value_layers
    network = config.network

    self.sess = tf.Session()
    self.agent = ppo_agent.SimplePPOPolicy(
        self.sess,
        network,
        policy_layers=policy_layers,
        value_layers=value_layers,
        checkpoint=os.path.join(FLAGS.logdir, FLAGS.checkpoint))

  def predict_action(self, observation):
    return self.agent.get_action([observation]) #returns action

ai = RLAlgorithm()

#observation = np.random.rand(28) #Initializing observation
count=1

# on_message is a callback function which is triggered when connection is made to the server/broker
def on_connect(client_1, userdata, flags, rc):
    print("Client_1 Connected with result code "+str(rc))
    client_1.subscribe(MQTT_PATH_1)

# on_message is a callback function which is triggered when message is recieved by the subscriber
def on_message(client_1, userdata, msg):
    
    global count
    s=msg.payload.decode("utf-8")
    s_l=eval(s)
    observation=np.asarray(s_l)
    #print("observation==> ", observation)

    action = ai.predict_action(observation)
    #print("action==> ", action)

    action_l=list(action[0])
    client_2.publish(MQTT_PATH_2,str(action_l))
    print(count)
    count+=1

client_2 = mqtt.Client()    # Publishes through path2
client_1 = mqtt.Client()    #Subscribes through path1
client_1.on_connect = on_connect
client_1.on_message = on_message
client_1.connect(MQTT_SERVER, 1883, 60) #Connecting to the server/broker
client_2.connect(MQTT_SERVER, 1883, 60) #Connecting to the server/broker
 
# Main function recieves encoder values from Raspi and publishes joint angle value to the raspi

def main(): 
  client_1.loop_forever()

if __name__ == "__main__":
  main()
