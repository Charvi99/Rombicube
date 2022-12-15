# subscriber.py
import paho.mqtt.client as mqtt
import matplotlib.pyplot as plt

def on_connect(client, userdata, flags, rc):
    print(f"Connected with result code {rc}")
    # subscribe, which need to put into on_connect
    # if reconnect after losing the connection with the broker, it will continue to subscribe to the raspberry/topic topic
    client.subscribe("raspberry/kalman")
    client.subscribe("raspberry/will")

# the callback function, it will be triggered when receiving messages
def on_message(client, userdata, msg):
    #print(f"{msg.topic} {msg.payload}")
   
    # processData(msg.payload.decode("utf-8"))
    # visData()
    data = str(msg.payload.decode("utf-8"))
    content_collector.append(data)
    if msg.topic == "raspberry/will":
        with open(r'C:\Users\jakub\Desktop\data.csv', 'w') as fp:
            for item in content_collector:
                # write each item on a new line
                fp.write("%s\n" % item)
            print("data saved")
        print("end")
    else:
        processData(data)


def processData(data:str):
    data_container = data.split(",")
    i = 0
    time.append(float(data_container[0]))
    
    gyroEulerRoll.append(float(data_container[1]))
    gyroEulerPitch.append(float(data_container[2]))
    gyroEulerYaw.append(float(data_container[3]))
    
    accEulerRoll.append(float(data_container[4]))
    accEulerPitch.append(float(data_container[5]))
    accEulerYaw.append(float(data_container[6]))
    
    magEulerRoll.append(float(data_container[7]))
    magEulerPitch.append(float(data_container[8]))
    magEulerYaw.append(float(data_container[9]))

    filterEulerRoll.append(float(data_container[10]))
    filterEulerPitch.append(float(data_container[11]))
    filterEulerYaw.append(float(data_container[12]))
    #print(data_container[10])
    #visData()
    
def visData():
    
    axs[0,0].clear()
    axs[0,1].clear()
    axs[1,0].clear()
    if len(filterEulerRoll)> 100:
        # axs[0, 0].plot(time[len(filterEulerRoll)-90:len(filterEulerRoll)], accEulerRoll[len(filterEulerRoll)-90:len(filterEulerRoll)])
        axs[0, 0].plot(time[len(filterEulerRoll)-90:len(filterEulerRoll)], magEulerRoll[len(filterEulerRoll)-90:len(filterEulerRoll)])
        axs[0, 0].plot(time[len(filterEulerRoll)-90:len(filterEulerRoll)], gyroEulerRoll[len(filterEulerRoll)-90:len(filterEulerRoll)])
        axs[0, 0].plot(time[len(filterEulerRoll)-90:len(filterEulerRoll)], filterEulerRoll[len(filterEulerRoll)-90:len(filterEulerRoll)])

        # axs[0, 1].plot(time[len(filterEulerRoll)-90:len(filterEulerRoll)], accEulerPitch[len(filterEulerRoll)-90:len(filterEulerRoll)])
        axs[0, 1].plot(time[len(filterEulerRoll)-90:len(filterEulerRoll)], magEulerPitch[len(filterEulerRoll)-90:len(filterEulerRoll)])
        axs[0, 1].plot(time[len(filterEulerRoll)-90:len(filterEulerRoll)], gyroEulerPitch[len(filterEulerRoll)-90:len(filterEulerRoll)])
        axs[0, 1].plot(time[len(filterEulerRoll)-90:len(filterEulerRoll)], filterEulerPitch[len(filterEulerRoll)-90:len(filterEulerRoll)])
        
        # axs[0, 0].plot(time[len(filterEulerRoll)-90:len(filterEulerRoll)], accEulerYaw[len(filterEulerRoll)-90:len(filterEulerRoll)])
        axs[1, 0].plot(time[len(filterEulerRoll)-90:len(filterEulerRoll)], magEulerYaw[len(filterEulerRoll)-90:len(filterEulerRoll)])
        axs[1, 0].plot(time[len(filterEulerRoll)-90:len(filterEulerRoll)], gyroEulerYaw[len(filterEulerRoll)-90:len(filterEulerRoll)])
        axs[1, 0].plot(time[len(filterEulerRoll)-90:len(filterEulerRoll)], filterEulerYaw[len(filterEulerRoll)-90:len(filterEulerRoll)])

    else:
        # axs[0, 0].plot(time, accEulerRoll)
        axs[0, 0].plot(time, magEulerRoll)
        axs[0, 0].plot(time, gyroEulerRoll)
        axs[0, 0].plot(time, filterEulerRoll)


        # axs[0, 1].plot(time, accEulerPitch)
        axs[0, 1].plot(time, magEulerPitch)
        axs[0, 1].plot(time, gyroEulerPitch)
        axs[0, 1].plot(time, filterEulerPitch)


        # axs[1, 0].plot(time, accEulerYaw)
        axs[1, 0].plot(time, magEulerYaw)
        axs[1, 0].plot(time, gyroEulerYaw)
        axs[1, 0].plot(time, filterEulerYaw)
        
    plt.draw()
    plt.pause(0.001)
    



global time
global gyroEulerRoll
global gyroEulerPitch
global gyroEulerYaw
global accEulerRoll
global accEulerPitch
global accEulerYaw
global magEulerRoll
global magEulerPitch
global magEulerYaw
global filterEulerRoll 
global filterEulerPitch 
global filterEulerYaw

time = []
gyroEulerRoll = []
gyroEulerPitch = []
gyroEulerYaw = []
accEulerRoll = []
accEulerPitch = []
accEulerYaw = []
magEulerRoll = []
magEulerPitch = []
magEulerYaw = []
filterEulerRoll =  []
filterEulerPitch =  []
filterEulerYaw =  []

#processData(b'0.029352903366088867,0.0,0.0,0.0,-1.2021186741873744,9.568190635644923,0.0,0.0,0.0,349.00311639595213,-0.10985981804426981,0.870648006527504,0.3483573843860107'.decode("utf-8"))
# visData()
# processData("0.11593770980834961,-0.04516090334415437,-0.00892083260774611,-0.015146280190944664,-8.296179748693207,1.9081729085618704,0.0,0.0,0.0,347.86844242638654,-0.8038020897838547,0.9556999285959299,0.6926494545662575")
# visData()
# processData("0.14423751831054688,-0.058392151637666104,-0.008814436567354584,-0.0227004343746191,23.57079130256312,11.91220364366369,0.0,0.0,0.0,-7.000719905670536,1.0709159202585445,1.8031678199404162,0.6798405144653483")
# visData()

global axs
plt.ion()
fig, axs = plt.subplots(2, 2)
axs[0, 0].set_title('ROLL')
axs[0, 1].set_title('PITCH')
axs[1, 0].set_title('YAW')

    
content_collector = []
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message



# set the will message, when the Raspberry Pi is powered off, or the network is interrupted abnormally, it will send the will message to other clients
client.will_set('raspberry/status', b'{"status": "Off"}')

# create connection, the three parameters are broker address, broker port number, and keep-alive time respectively
client.connect("broker.emqx.io", 1883, 60)

# set the network loop blocking, it will not actively end the program before calling disconnect() or the program crash
client.loop_forever()