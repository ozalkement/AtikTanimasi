import sys
import types 
import importlib
if 'imp' not in sys.modules:
    fake_imp=types.ModuleType('imp')
    fake_imp.reload=importlib.reload
    fake_imp.find_module=importlib.util.find_spec
    sys.modules['imp']=fake_imp
import tensorflow as tf
from picamera2 import Picamera2
import numpy as np
from PIL import Image
import cv2
import time
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input, decode_predictions
from tensorflow.keras.utils import img_to_array

picam2=Picamera2()

config=picam2.create_still_configuration(main={"size":(1280,720),"format":"RGB888"})
picam2.configure(config)
picam2.start()

tflite_model_path='/home/ozal/copayiklama/CopSanal1/atik_classifier_model_AUG.tflite'
interpreter=tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

input_details=interpreter.get_input_details()
output_details=interpreter.get_output_details()

class_labels = {
    0: 'cardboard',  # C harfi
    1: 'glass',      # G harfi
    2: 'metal',      # M harfi
    3: 'paper',      # P harfi (Pa...)
    4: 'plastic',    # P harfi (Pl...)
}

confidence_threshold=65.0


#model=MobileNetV3Large(weights='imagenet', input_shape=(224,224,3))
try:
    while True:
        frame=picam2.capture_array()
        if frame.shape[2]!=3:
            frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            
        img=Image.fromarray(frame)
        img=img.resize((224,224))
        img_array=img_to_array(img).astype(np.float32)
        img_array=np.expand_dims(img_array,axis=0)
        
        interpreter.set_tensor(input_details[0]['index'],img_array)
        
        interpreter.invoke()
        
        prediction=interpreter.get_tensor(output_details[0]['index'])[0]
        
        predicted_class=np.argmax(prediction)
        predicted_label=class_labels.get(predicted_class,"Unknown")
        predicted_probablity=prediction[predicted_class]*100
        
        if predicted_probablity<confidence_threshold:
            text="No classfication"
        else:
            text=f"predicted: {predicted_label} ({predicted_probablity:.2f}%)"        
        
            
        display_frame=cv2.putText(frame.copy(),text,(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        cv2.imshow("Kamera: ",display_frame)
            
        if cv2.waitKey(1) & 0XFF==ord('q'):
            break
            
        time.sleep(0.1)   
            
except KeyboardInterrupt:
        print("Gerçek zamanlı çıkarım kullanıcı tarafından durduruldu")
finally:
    picam2.stop()
    cv2.destroyAllWindows()