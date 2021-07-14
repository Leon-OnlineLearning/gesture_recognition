from tensorflow import keras
from EDA.HandDetector import handDetector

class Singleton_model:
   __model = None
   __detector = None
   @staticmethod 
   def getInstance():
      if Singleton_model.__model == None:
         Singleton_model()
      return Singleton_model.__model, Singleton_model.__detector

   
   def __init__(self, min_detection_confidence= .9, min_tracking_confidence=.9):
      if Singleton_model.__model != None:
         raise Exception("This class is a singleton!")
      else:
         Singleton_model.__model = keras.models.load_model('../Gesture recognition_Sign Language_mobilenet_v6')
         Singleton_model.__detector= handDetector(min_detection_confidence, min_tracking_confidence)