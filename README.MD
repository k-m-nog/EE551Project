# Dogerizer


Thank you to Alvin Wan for a great and very educational tutorial on using openCV to detect faces and adding masks: 

   https://hackmd.io/@alvinwan/SkOZwiT-M?type=view#Step-1-%E2%80%94-Creating-Our-Project-and-Installing-Dependencies

An application to turn all people in a photo into dogs. 
This project uses python3, openCV, pytorch, numpy, MatPlotLib and other libraries 

### QuickStart

1. Install all Requirements

        
        pip3 install -r requirements.txt

2. Create a static folder

        mkdir ./assets

3. Place pretained model in assets directory
        

        wget -O assets/haarcascade_frontalface_default.xml https://github.com/opencv/opencv/raw/master/data/haarcascades/haarcascade_frontalface_default.xml


4. Run app 

        python3 flaskDogerizer.py
