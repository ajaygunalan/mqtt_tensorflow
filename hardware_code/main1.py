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


MQTT_SERVER = "192.168.43.206" #IP of the broker/server
MQTT_PATH_1 = "test_channel_1"
MQTT_PATH_2 = "test_channel_2"

flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS
flags.DEFINE_string("logdir", "/home/abhimanyu/mqTT/hardware_code/weights_file",
                    "The directory that contains checkpoint and config.")
flags.DEFINE_string("checkpoint", "model.ckpt-8800020", "The checkpoint file path.")

observation = np.random.rand(28) #Initializing observation
count=0
# on_message is a callback function which is triggered when connection is made to the server/broker
def on_connect(client_1, userdata, flags, rc):
    print("Client_1 Connected with result code "+str(rc))
    client_1.subscribe(MQTT_PATH_1)
# on_message is a callback function which is triggered when message is recieved by the subscriber
def on_message(client_1, userdata, msg):
    global observation
    global count
    s=msg.payload.decode("utf-8")
    s_l=eval(s)
    observation=np.asarray(s_l)
    print (observation)
    print(count)
    count=count+1

client_2 = mqtt.Client()    # Publishes through path2
client_1 = mqtt.Client()    #Subscribes through path1
client_1.on_connect = on_connect
client_1.on_message = on_message
client_1.connect(MQTT_SERVER, 1883, 60) #Connecting to the server/broker
client_2.connect(MQTT_SERVER, 1883, 60) #Connecting to the server/broker
 
# Main function recieves encoder values from Raspi and publishes joint angle value to the raspi

def main(argv): 
  global observation
  del argv  # Unused.
  config = utility.load_config(FLAGS.logdir)
  policy_layers = config.policy_layers
  value_layers = config.value_layers
  network = config.network

  with tf.Session() as sess:
    agent = ppo_agent.SimplePPOPolicy(
        sess,
        network,
        policy_layers=policy_layers,
        value_layers=value_layers,
        checkpoint=os.path.join(FLAGS.logdir, FLAGS.checkpoint))

    while True:
      #observation = np.random.rand(28)
      action = agent.get_action([observation]) #tensorflow calculates the joint angle value
      #print("action to execute====> ", action[0])
      action_l=list(action[0])
      client_2.publish(MQTT_PATH_2,str(action_l)) #publishing the joint angles
      # This sleep is to prevent serial communication error on the real robot.
      

client_1.loop_start()
if __name__ == "__main__":
  tf.app.run(main) 
