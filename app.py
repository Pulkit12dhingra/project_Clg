# import libraries 
# deactivate CUDA 
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# GUI libraries
from tkinter import *
from tkinter import filedialog

# image processing
from PIL import Image, ImageDraw
import cv2

# load models
import tensorflow as tf
import numpy as np

class gui():

	# constructor
	def __init__(self):
		self.root=Tk()
		self.root.title("Paint Application")
		self.root.geometry("800x800")
		self.width = 500
		self.height = 350
		self.white = (255,255,255)
		self.black = (0,0,0)
		self.SIZE = 256
		self.model_female=tf.keras.models.load_model('./models/female.h5')
		self.model_male=tf.keras.models.load_model('./models/model_male.h5')
		self.bg = PhotoImage(file = "images/male.png")
		self.wn=Canvas(self.root, width=1000, height=1000, bg='white')
		self.wn.create_image( 200, 200, image = self.bg, anchor = "nw")
		self.image1 = Image.new("RGB", (self.width, self.height), 
			self.white)
		self.draw = ImageDraw.Draw(self.image1)
		self.btn = None
		self.btn1 = None
		self.btn2 = None
		self.btn3 = None

	def paint(self,event):
	    
	    # get x1, y1, x2, y2 co-ordinates
	    x1, y1 = (event.x-3), (event.y-3)
	    x2, y2 = (event.x+3), (event.y+3)
	    color = "black"

	    # display the mouse movement inside canvas
	    self.wn.create_oval(x1, y1, x2, y2, fill=color, outline=color)
	    self.draw.ellipse([x1, y1, x2, y2], self.black)

	# female prediction
	def get_pred_fem(self):

		image=cv2.imread("images/my_drawing.jpg")
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = cv2.resize(image, (self.SIZE, self.SIZE))

		# normalizing image 
		image = image.astype('float32') / 255.0
		
		#predicted =np.clip(model.predict(test_sketch_image[i].reshape(1,SIZE,SIZE,3)),0.0,1.0).reshape(SIZE,SIZE,3)
		predicted =np.clip(self.model_female.predict(image.reshape(1,
			self.SIZE,self.SIZE,3)),0.0,1.0).reshape(self.SIZE,self.SIZE,3)
		predicted = cv2.cvtColor(predicted, cv2.COLOR_RGB2BGR)
		return predicted

	# Get the pred
	def female(self):
		filename = "images/my_drawing.jpg"
		self.image1.save(filename)
		cv2.imshow("window",self.get_pred_fem())
		return True

	def open_custom(self):
		file_path = filedialog.askopenfilename()
		self.image1=cv2.imread(file_path)
		filename = "my_drawing.jpg"
		cv2.imwrite(filename,self.image1)
		cv2.imshow("window",self.get_pred_male())
		return True

	# clear the screen 
	def clr(self):
		self.wn.delete()
		return True

	# male prediction
	def get_pred_male(self):

		image = cv2.imread("images/my_drawing.jpg")
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = cv2.resize(image, (self.SIZE, self.SIZE))

		# normalizing image 
		image = image.astype('float32') / 255.0
		
		#predicted =np.clip(model.predict(test_sketch_image[i].reshape(1,SIZE,SIZE,3)),0.0,1.0).reshape(SIZE,SIZE,3)
		predicted =np.clip(self.model_male.predict(image.reshape(1,
			self.SIZE,self.SIZE,3)),0.0,1.0).reshape(self.SIZE,self.SIZE,3)
		predicted = cv2.cvtColor(predicted, cv2.COLOR_RGB2BGR)
		return predicted

	# Get the pred
	def male(self):
		filename = "images/my_drawing.jpg"
		self.image1.save(filename)
		cv2.imshow("window",self.get_pred_male())
		return True

	def output(self):
	
		# bind mouse event with canvas(wn)
		self.wn.bind('<B1-Motion>', self.paint)
		self.wn.pack()
		self.btn = Button(self.root, text = 'Female', bd = '5',
                          command = self.female)
		self.btn1 = Button(self.root, text = 'male', bd = '5',
                          command = self.male)
		self.btn2 = Button(self.root, text = 'clear', bd = '5',
                          command = self.clr)
		self.btn3 = Button(self.root, text = 'Open_file', bd = '5',
                          command = self.open_custom)

		button1_canvas = self.wn.create_window( 0, 10, 
                                       anchor = "nw",
                                       window = self.btn)
		button2_canvas = self.wn.create_window( 80, 10,
                                       anchor = "nw",
                                       window = self.btn1)
		button3_canvas = self.wn.create_window( 160, 10,
		                                       anchor = "nw",
		                                       window = self.btn2)
		button4_canvas = self.wn.create_window( 240, 10,
		                                       anchor = "nw",
		                                       window = self.btn3)
		self.root.mainloop()

if __name__=="__main__":
	obj=gui()
	obj.output()