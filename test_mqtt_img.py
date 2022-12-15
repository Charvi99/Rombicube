# subscriber.py
import paho.mqtt.client as mqtt
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle

def on_connect(client, userdata, flags, rc):
    print(f"Connected with result code {rc}")
    # subscribe, which need to put into on_connect
    # if reconnect after losing the connection with the broker, it will continue to subscribe to the raspberry/topic topic
    client.subscribe("raspberry/topic")
    client.subscribe("raspberry/will")

# the callback function, it will be triggered when receiving messages
def on_message(client, userdata, msg):
    #print(f"{msg.topic} {msg.payload}")
   
    #data = str(msg.payload.decode('utf-8'))
    if len(msg.payload) > 3:
        img_np = np.asarray(pickle.loads(msg.payload))
        cv2.imshow("ahoj", img_np)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            exit()
    # try:
    # except:
    #     pass

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message



# set the will message, when the Raspberry Pi is powered off, or the network is interrupted abnormally, it will send the will message to other clients
client.will_set('raspberry/status', b'{"status": "Off"}')

# create connection, the three parameters are broker address, broker port number, and keep-alive time respectively
client.connect("broker.emqx.io", 1883, 60)

# set the network loop blocking, it will not actively end the program before calling disconnect() or the program crash
client.loop_forever()