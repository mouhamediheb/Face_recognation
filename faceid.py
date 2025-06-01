# Import kivy dependencies first 
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

# Import kivy UX components
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label

# Import other kivy stuff
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.logger import Logger 

# Import other dependencies 
import cv2
import tensorflow as tf
from layers import L1Dist
import os 
import numpy as np


class CamApp(App):
    def build(self):
        # Main layout components
        self.web_cam = Image(size_hint=(1, .8))
        self.button = Button(text="verify", on_press=self.verify, size_hint=(1, .1))
        self.verification_label = Label(text="Verification Uninitiated", size_hint=(1, .1))

        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)

        # Load tensorflow/keras model
        from tensorflow.keras.models import load_model
        self.model = load_model('siamese_model.h5', custom_objects={'L1Dist': L1Dist})

        # Setup video capture device
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0 / 33.0)
        return layout

    def update(self, *args):
        ret, frame = self.capture.read()
        if not ret:
            return

        # Crop frame to 250×250 region (optional)
        frame = frame[120:120 + 250, 200:200 + 250, :]

        # Flip horizontally and convert to texture
        buf = cv2.flip(frame, 0).tobytes()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture

    def load_and_preprocess_image(self, image_path):
        # load and convert image to 100×100
        byte_img = tf.io.read_file(image_path)
        img = tf.io.decode_jpeg(byte_img)
        img = tf.image.resize(img, (100, 100))
        img = img / 255.0
        return img

    def verify(self, *args):
        detection_threshold = 0.5
        verification_threshold = 0.5

        SAVE_DIR = os.path.join('real_time_prediction', 'input_images')
        os.makedirs(SAVE_DIR, exist_ok=True)
        SAVE_PATH = os.path.join(SAVE_DIR, 'input_image.jpg')

        ret, frame = self.capture.read()
        if not ret:
            return

        frame = frame[120:120 + 250, 200:200 + 250, :]
        cv2.imwrite(SAVE_PATH, frame)

        input_img = self.load_and_preprocess_image(SAVE_PATH)
        results = []

        for image in os.listdir(os.path.join('real_time_prediction', 'verification_images')):
            val_path = os.path.join('real_time_prediction', 'verification_images', image)
            validation_img = self.load_and_preprocess_image(val_path)

            # predict expects a list of two 4D tensors: [anchor_batch, validation_batch]
            result = self.model.predict([
                np.expand_dims(input_img, axis=0),
                np.expand_dims(validation_img, axis=0)
            ])
            results.append(result)

        detection = np.sum(np.array(results) > detection_threshold)
        verification = detection / len(os.listdir(os.path.join('real_time_prediction', 'verification_images')))
        verified = verification > verification_threshold

        # Update the label text:
        self.verification_label.text = 'Verified' if verified else 'Unverified'
        #Log out details
        Logger.info(results)
        Logger.info(np.sum(np.array(results))>0.5)
        
        return results, verified


if __name__ == '__main__':
    CamApp().run()