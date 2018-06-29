import paho.mqtt.client as mqtt
import logging
import random
import numpy as np

#log_filename1="test1.log"
#log_filename2="test2.log"

count=0;   
MQTT_SERVER = "localhost" # IP of the server/broker
MQTT_PATH_1 = "test_channel_1"
MQTT_PATH_2 = "test_channel_2"

# on_message is a callback function which is triggered when connection is made to the server/broker
def on_connect(client_2, userdata, flags, rc):
    print(" Client_2 Connected with result code "+str(rc))
    client_2.subscribe(MQTT_PATH_2)

# on_message is a callback function which is triggered when message is recieved by the subscriber
def on_message(client_2, userdata, msg):
    global count
    #print(msg.payload.decode("utf-8") )
    msg1=eval(msg.payload.decode("utf-8"))
    d=np.asarray(msg1)
    print((d))
    print(count)
    count=count+1
    
client_1 = mqtt.Client() # Publishes through Path 1
client_2 = mqtt.Client() #Subscribes through Path 2
client_2.on_connect = on_connect 
client_2.on_message = on_message
client_2.connect(MQTT_SERVER, 1883, 60) #Connecting to the server/broker
client_1.connect(MQTT_SERVER, 1883, 60) #Connecting to the server/broker

client_2.loop_start() 

while True:
    observation = np.random.rand(28)
    observation_l=list(observation)
    #print(observation_l)
    client_1.publish(MQTT_PATH_1,str(observation_l)) #Publishing an array of random digits
    

