from tensorflow import keras
from EDA.HandDetector import handDetector
from EDA.HandPose import handPose
import pickle

class Singleton_model:
   __model = None
   __detector = None
   __pose_detection = None

   @staticmethod 
   def getInstance():
      if Singleton_model.__model == None:
         Singleton_model()
      return Singleton_model.__model, Singleton_model.__detector,Singleton_model.__pose_detection

   
   def __init__(self, min_detection_confidence= .8, min_tracking_confidence=.8):
      if Singleton_model.__model != None:
         raise Exception("This class is a singleton!")
      else:
         Singleton_model.__model = pickle.load(open('../finalized_model.sav', 'rb'))
         Singleton_model.__detector = handDetector(min_detection_confidence, min_tracking_confidence)
         Singleton_model.__pose_detection = handPose(min_detection_confidence, min_tracking_confidence)