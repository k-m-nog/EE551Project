# Dogerizer


Used the following https://hackmd.io/@alvinwan/SkOZwiT-M?type=view#Step-1-%E2%80%94-Creating-Our-Project-and-Installing-Dependencies

An application to turn all people in a photo into dogs. 
This project uses python3, openCV, pytorch, numpy, MatPlotLib and other libraries 

### QuickStart

1. Install all Requirements

        
        pip3 install -r requirements.txt

2. Place pretained model in assets directory
        

        wget -O assets/haarcascade_frontalface_default.xml https://github.com/opencv/opencv/raw/master/data/haarcascades/haarcascade_frontalface_default.xml


3. Run app 

        python3 flaskDogerizer.py