# mqtt_tensorflow
The repositry contains the lowest-level implementation of the neural network trained for the Quadruped robot, directly deployable on the hardware, with least possible dependencies and requirements.

# Requirements:
1. Python3 
  ```
  sudo apt install python3-pip
  ```
2. Tensorflow:
  ```
  sudo pip3 install tensorflow==1.5
  ```
3. Numpy: 
```sudo pip3 install numpy
```
4. Yaml: 
```sudo pip3 install ruamel.yaml
```
5. Paho MQTT client

# Usage(For tensor flow):
1. Do NOT modify the helper files: "utility.py, normalize.py and weights_file".
2. "agent.py" contains the class to initialize the neural network and predict the resulting action.
3. Running main.py as **python3 main1.py** will launch a dummy neural network that imputs dummy observations from the robot and spits out the motor joint angles to be executed. Just put the code from main.py inside the RPI code.
4. Inside "main1.py", provide proper address and name of the weights_file and model file in the logdir and checkpoint flag respectively. (An example is already given).
5. Over the time, I will share the updated weights_file presenting the best neural network learned. Just replace the weights_file folder with the latest one and provide appropriate address. 

# Usage(MQTT):
1. There is a server/broker and two clients on each side(Between Rpi and Ubuntu)
2. Install paho MQTT client for python using "pip3 install paho-mqtt". 
3. In this system we will be using One broker, two clients on each side and two different channels.
4. Define a global variable which stores the value whenever it is subscribed in the on_message function.
5. It is then passed on to the Tensorflow, which when calculates and publishes the joint angle values to Rpi.

Author: Abhik, Abhimanyu
Email Id: abhiksingla10@gmail.com, abhimanyusingh8713@gmail.com




