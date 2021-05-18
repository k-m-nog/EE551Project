import cv2
import numpy as np
from flask import Flask, request, render_template, flash , redirect, url_for
from werkzeug.utils import secure_filename
import os
  
# Flask constructor
app = Flask(__name__) 


def get_faces(img):
    
    cascade = cv2.CascadeClassifier("assets/haarcascade_frontalface_default.xml")
    # Convert to black-and-white -- classifier was trained on b&w
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blackwhite = cv2.equalizeHist(gray)
    # get positions of all faces
    rects = cascade.detectMultiScale(
        blackwhite, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE)

    return rects


def apply_mask(face: np.array,mask: np.array):
    mask_h, mask_w, _=mask.shape
    face_h, face_w, _= face.shape
    factor=min(face_h/mask_h, face_w/mask_w)

    new_mask_w =int(factor * mask_w)
    new_mask_h =int(factor * mask_h)
    resized_mask=cv2.resize(mask, (new_mask_w,new_mask_h))

    non_white_pixels= (resized_mask<250).all(axis=2)

    offset_h = int((face_h - new_mask_h)/2)
    offset_w = int((face_w - new_mask_w)/2)

    face_with_mask=face.copy()
    face_with_mask[offset_h: offset_h+new_mask_h, offset_w: offset_w+new_mask_w][non_white_pixels] = resized_mask[non_white_pixels]

    return face_with_mask


def mask(img_path, outputPath):
    #face=cv2.imread('assets/child.png')
    
    image = cv2.imread(img_path)
    
    faces= get_faces(image)
    mask= cv2.imread('assets/dog.png')

    image_h, image_w, _ = image.shape

    for x,y,w,h in faces:
        y0, y1 = int(y - 0.25*h), int(y + 0.75*h)
        x0, x1 = x, x + w

        # Dont add mask if the cropped frame would be out-of-bounds
        if x0 < 0 or y0 < 0 or x1 > image_w or y1 > image_h:
            continue

        image[y0: y1, x0: x1]= apply_mask(image[y0:y1, x0:x1], mask)
   
    cv2.imwrite(outputPath, image)




@app.route('/', methods=["GET","POST"])
def prompt():
    # Repeatedly Ask the user who theyd like to see with a dog filter
    if request.method == "POST":
       # getting input from HTML form
       file = request.files["file"]
       # getting input with name = lname in HTML form 
       filename = secure_filename(file.filename)
       file.save(os.path.join("./static", filename))
       return redirect(url_for('uploaded_file',
                                    filename=filename))

       #return "Your name is "+name 
    return render_template("page.html")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    
    #filename = "./static/" + filename
    # decide on output filename
    split_path = filename.split('/')
    split_name = split_path[-1].split('.')
    name = split_name[-2]
    ext = split_name[-1]
    #outputFile=('./static/'+ name +'_withMask.' + ext)
    outputFile=( name +'_withMask.' + ext)


    # MASKERIZE
    mask("./static/" + filename, './static/'+ outputFile)
    print(filename, " ", outputFile)
    return render_template("result.html", inIMG=filename, outIMG=outputFile)


if __name__ == '__main__':
    app.run()