# -*- coding: utf-8 -*-
"""
Created on Sun May 12 16:13:43 2019

@author: S.Primakov
"""

import sys
#import random
sys.setrecursionlimit(10000)

from cv2 import imread

import shutil
from pathlib import Path

from kivy.app import App
from kivy.lang import Builder
from kivy.core.window import Window
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.checkbox import CheckBox
from kivy.uix.bubble import Bubble
from kivy.uix.button import Button
from kivy.properties import StringProperty,ObjectProperty
from kivy.properties import ListProperty
from kivy.uix.image import Image
from kivy.uix.progressbar import ProgressBar
from kivy.uix.textinput import TextInput
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.popup import Popup
from kivy.factory import Factory
from kivy.uix.label import Label
#import SimpleITK as sitk
import matplotlib.pyplot as plt
import os,re
#import threading
from kivy.uix.slider import Slider
#from sklearn.utils import shuffle
import numpy as np
import gc
import pandas as pd
from kivy.graphics import Color, Rectangle
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
#import telegram
#import re

from kivy.config import Config
from kivy.core.window import Window

os.environ['KIVY_GL_BACKEND']='angle_sdl2'

Config.set('graphics', 'width', '1024')
Config.set('graphics', 'height', '768')

# Create both screens. Please note the root.manager.current: this is how
# you can control the ScreenManager from kv. Each screen has by default a
# property manager that gives you the instance of the ScreenManager used.

#BoxLayout:
#        Button:
#            text: 'Start Test'
#            on_press: root.manager.current = 'comp_scr'
#        Button:
#            text: 'Quit application'
#            on_press: root.close_app()




Builder.load_string("""
<Main_Screen>:
    BoxLayout:
        canvas.before:
            Color:
                rgba: 27/255.0, 70/255., 132/255., 1
            Rectangle:
                pos: self.pos
                size: self.size
        orientation: 'horizontal'

        Label:
            size_hint: (0.3, 1)
            canvas.before:
                Color:
                    rgba: 1, 1, 1, 1
                Rectangle:
                    pos: self.pos
                    size: self.size
        BoxLayout:
            orientation: 'vertical'
            canvas.before:
                Color:
                    rgba: 1, 1, 1, 1
                Rectangle:
                    pos: self.pos
                    size: self.size
            Label:
                size_hint: (1, 0.2)
                canvas.before:
                    Color:
                        rgba: 1, 1, 1, 1
                    Rectangle:
                        pos: self.pos
                        size: self.size

            Image:
                size_hint: (1, None)
                source: './imgs/D_lab_logo.png'
                canvas.before:
                    Color:
                        rgba: 1, 1, 1, 1
                    Rectangle:
                        pos: self.pos
                        size: self.size

            Label:
                font_size: 25
                halign: 'center'
                valign: 'top'
                multiline: True
                size_hint: (1, None)
                height: 2*self.texture_size[1]
                text: '[color=#000000]Interpolation method comparison experiment[/color]'
                text_size: self.width, None
                markup: True
                canvas.before:
                    Color:
                        rgba: 1, 1, 1, 1
                    Rectangle:
                        pos: self.pos
                        size: self.size

            Bubble:
                canvas.before:
                    Color:
                        rgba: 1, 1, 1, 1
                    Rectangle:
                        pos: self.pos
                        size: self.size
                background_color: (2/255.0, 54/255., 129/255., 0.8)
                orientation: 'vertical'
                show_arrow: False
                size_hint: (1, 0.15)
                pos_hint: {'center_x': .5, 'y': .1}

                BubbleButton:
                    text: 'Start Test'
                    on_release: root.manager.current = 'descr_scr'

                BubbleButton:
                    text: 'Quit application'
                    on_release: root.close_app()
            Label:
                size_hint: (1, 0.2)
                canvas.before:
                    Color:
                        rgba: 1, 1, 1, 1
                    Rectangle:
                        pos: self.pos
                        size: self.size
        
        Label:
            size_hint: (0.3, 1)
            canvas.before:
                Color:
                    rgba: 1, 1, 1, 1
                Rectangle:
                    pos: self.pos
                    size: self.size

<Comparison_Screen>:

    RstDocument:
        #text: root.descr_text2
        source: 'test2.rst'
        canvas.before:
            Color:
                rgba: (1,1,1,0.8)
            Rectangle:
                pos: self.x, self.y 
                size: self.width, self.height
        canvas:
            Color:
                rgba: (1,1,1,0.1)
            Rectangle:
                pos: self.x, self.y 
                size: self.width, self.height
        
    BoxLayout:
        orientstion:'horizontal' 
        size_hint: (1,0.15)
        Button:
            id: btn_back_to_menu
            size_hint: (1,1)
            font_size: 25
            text: 'Back'
            on_release: root.manager.current = 'descr_scr'
            
        Button:
            id: btn_to_comp_screen
            size_hint: (1,1)
            font_size: 25
            text: 'Continue'
            on_release: root.manager.current = 'test_scr'

<Description_Screen>:
    md_check_box:md
    rad_check_box:rad
    radonc_check_box:radonc
    cs_check_box:cs
    stud_check_box:stud
    oth_check_box:oth
    u_name: user_name
    u_email: user_email
    ack_check_box: acknowledge

    BoxLayout:
        orientation: 'vertical'
        RstDocument:
            #text: root.descr_text
            source: 'test.rst'
            canvas.before:
                Color:
                    rgba: (1,1,1,0.8)
                Rectangle:
                    pos: self.x, self.y
                    size: self.width, self.height
            canvas:
                Color:
                    rgba: (1,1,1,0.1)
                Rectangle:
                    pos: self.x, self.y
                    size: self.width, self.height
        BoxLayout:
            size_hint: (1,0.32)
            orientation: 'horizontal'
            padding: 5
            canvas.before:
                Color:
                    rgba: 243/255, 243/255, 243/255, 1
                Rectangle:
                    pos: self.pos
                    size: self.size

            GridLayout:
                size_hint: (0.5,1)
                cols: 2
                Label:
                    text: ''
                Label:
                    text: '[b][color=#000000]Training[/color][/b]'
                    markup: True
                    text_size: self.size
                    halign: 'right'
                    valign: 'middle'
                CheckBox:
                    id: md
                    text: 'Medical Doctor'
                    group: 'spec'
                    color: 0,0,0,1
                Label:
                    text: '[color=#000000]Medical Doctor[/color]'
                    markup: True
                    text_size: self.size
                    halign: 'left'
                    valign: 'middle'
                CheckBox:
                    id: rad
                    text: 'Radiologist'
                    group: 'spec'
                    color: 0,0,0,1
                Label:
                    text: '[color=#000000]Radiologist[/color]'
                    markup: True
                    text_size: self.size
                    halign: 'left'
                    valign: 'middle'
                CheckBox:
                    id: radonc
                    text: 'Radiation oncologist'
                    group: 'spec'
                    color: 0,0,0,1
                Label:
                    text: '[color=#000000]Radiation oncologist[/color]'
                    markup: True
                    multiline: True
                    text_size: self.size
                    halign: 'left'
                    valign: 'middle'
            GridLayout:
                size_hint: (0.5,1)
                cols: 2
                Label:
                    text: ''
                Label:
                    text: ''
                CheckBox:
                    id: cs
                    group: 'spec'
                    text: 'Computer scientist'
                    color: 0,0,0,1
                Label:
                    text: '[color=#000000]Computer scientist[/color]'
                    markup: True
                    multiline: True
                    text_size: self.size
                    halign: 'left'
                    valign: 'middle'
                CheckBox:
                    id: stud
                    text: 'Student'
                    group: 'spec'
                    color: 0,0,0,1
                Label:
                    text: '[color=#000000]Student[/color]'
                    markup: True
                    text_size: self.size
                    halign: 'left'
                    valign: 'middle'
                CheckBox:
                    id: oth
                    text: 'Other'
                    group: 'spec'
                    color: 0,0,0,1
                Label:
                    text: '[color=#000000]Other[/color]'
                    markup: True
                    text_size: self.size
                    halign: 'left'
                    valign: 'middle'
                    
            GridLayout:
                size_hint: (0.5,1)
                cols: 2
                Label:
                    text: ''
                Label:
                    text: ''

            BoxLayout:
                size_hint: (0.3,1)
                orientation: 'vertical'
                padding: 2
                Label:
                    text: '[color=#000000]Name[/color]'
                    markup: True
                    halign: 'left'
                TextInput:
                    height: 300
                    id: user_name
                    text: ''
                Label:
                    text: '[color=#000000]E-mail[/color]'
                    markup: True
                    halign: 'left'
                TextInput:
                    size_hint: (1,1)
                    id: user_email
                    text: ''


            GridLayout:
                size_hint: (0.5,1)
                cols: 2
                Label:
                    text: ''
                Label:
                    text: ''

            BoxLayout:
                size_hint: (0.3,1)
                orientation: 'vertical'
                padding: 2
                Label:
                    text: ''
                    markup: True
                    halign: 'left'
                Label:
                    text: '[color=#000000]Acknowledgment in Article?[/color]'
                    markup: True
                    halign: 'left'
                CheckBox:
                    id: acknowledge
                    text: 'Article Acknowledgement'
                    color: 0,0,0,1
                Label:
                    text: ''
                    markup: True
                    halign: 'left'
                    
            GridLayout:
                size_hint: (0.5,1)
                cols: 2
                Label:
                    text: ''
                Label:
                    text: ''

        BoxLayout:
            orientstion:'horizontal'
            size_hint: (1,0.15)
            Button:
                id: btn_back_to_menu
                size_hint: (1,1)
                font_size: 25
                text: 'Back to menu'
                on_release: root.manager.current = 'main_scr'

            Button:
                id: btn_to_comp_screen
                size_hint: (1,1)
                font_size: 25
                text: 'Continue'
                on_release: root.manager.current = 'comp_scr'

<PopupBox>:
    pop_up_text: _pop_up_text
    progress_bar: pb
    size_hint: .5, .5
    auto_dismiss: True
    title: 'Status:'

    BoxLayout:
        orientation: "vertical"
        Label:
            id: _pop_up_text
            text: ''
        ProgressBar:
            id:pb
            max: 1000

<PopupBox2>:
    pop_up_text: _pop_up_text

    size_hint: .65, .65
    auto_dismiss: True
    title: 'Results screen'
#    back_to_menu: back_to_menu
    quit_app:quit_app

    BoxLayout:
        orientation: "vertical"
        padding: 20
        Label:
            id: _pop_up_text
            text: ''
            valign: 'top'
            halign: 'left'
#        Button:
#            id: back_to_menu
#            size_hint_y: 0.5
#            height: 40
#            font_size: 20
#            text: 'Back to menu'
        Button:
            id: quit_app
            size_hint_y: 0.2
            height: 40
            font_size: 20
            text: 'Quit application'


""")

# Declare both screens
class Main_Screen(Screen):

    def close_app(self,*args):
        Main_App.get_running_app().stop()
        Window.close()
        gc.collect()

class Comparison_Screen(Screen):
    descr_text2 = StringProperty('Text here')
    label_text = StringProperty('To start drag folder with patients to the screen above')
    image_bg = StringProperty('./imgs/main.png')

class Description_Screen(Screen):
    descr_text = StringProperty('Text here')
    #image_bg = StringProperty('main.png')

class MyLabel(Label):
    def on_size(self, *args):
        self.canvas.before.clear()
        with self.canvas.before:
            Color(1, 1, 1, 1)
            Rectangle(pos=self.pos, size=self.size)

class MySlider(Slider):
    def on_size(self, *args):
        self.canvas.before.clear()
        with self.canvas.before:
            Color(1, 1, 1, 1)
            Rectangle(pos=self.pos, size=self.size)

class MyBoxLayout(BoxLayout):
    def on_size(self, *args):
        self.canvas.before.clear()
        with self.canvas.before:
            Color(1, 1, 1, 1)
            Rectangle(pos=self.pos, size=self.size)

class Test_Screen(Screen):
    count=0
    img_len=0
    counts = StringProperty(str(count))
    plt_images = FigureCanvasKivyAgg(plt.gcf())

    button_left = Button(text="[b]Left[/b]",markup=True,font_size = 25)
    button_right = Button(text="[b]Right[/b]",markup=True,font_size = 25)

    label1 = MyLabel(text= '',halign= 'center',font_size = 20, valign= 'middle',size_hint =(1,0.1),markup=True)
    comment_box = TextInput(hint_text='Please leave your comments here...',multiline = True, size_hint =(0.5,1))
    WW = MySlider(min=50, max=2000, value=1500,size_hint =(0.4,0.6),pos_hint={'center_x': 0.35, 'center_y': 0.6},cursor_height =22,step=10)
    WL = MySlider(min=-1000, max=2000, value=-600,size_hint =(0.4,0.6),pos_hint={'center_x': 0.35, 'center_y': 0.5},cursor_height =22,step=10)

    ww_label = MyLabel(text= '[color=#000000]WW: [/color]',halign= 'left',valign= 'middle',font_size = 12,size_hint =(0.2,1), markup=True)
    ww_label.bind(size=ww_label.setter('text_size'))
    wl_label = MyLabel(text= '[color=#000000]WL: [/color]',halign= 'left', valign= 'middle',font_size = 12, size_hint =(0.2,1),markup=True)
    wl_label.bind(size=wl_label.setter('text_size'))
    sl_label = MyLabel(text= '[color=#000000][b]Image window settings:[/b][/color]',halign= 'left',font_size = 16,valign= 'middle',size_hint =(1,1), markup=True)
    sl_label.bind(size=sl_label.setter('text_size'))
    
    def update_images(self,img0, img1, img2, dr_left):
    
        print(re)

        #Update text input text
        self.comment_box.text = ''
        if re.search('left',dr_left):
            
            plt.clf()
            plt.title('Ground Truth')
            plt.imshow(np.hstack((img1,
                                  #np.ones((img1.shape[0], int(img1.shape[1]/5)))*np.max(img1.flatten()),
                                  img0,
                                  img2)),cmap='bone')

            #plt.contour(np.hstack((msk,np.zeros((img.shape[0],int(img.shape[1]/5))),msk2)),colors='r')
            plt.axis('off')
            plt.tight_layout(pad=2)
            self.plt_images.draw()
            
        else:
            plt.clf()
            plt.title('Ground Truth')
            plt.imshow(np.hstack((img1,
                                  #np.ones((img1.shape[0],int(img1.shape[1]/5)))*np.max(img1.flatten()),
                                  img0,
                                  img2)),cmap='bone')
            #plt.contour(np.hstack((msk2,np.zeros((img.shape[0],int(img.shape[1]/5))),msk)),colors='r')
            plt.axis('off')
            plt.tight_layout(pad=2)
            self.plt_images.draw()

    def on_pre_enter(self):
        self.label1.text = '[color=#000000][b]Please choose preferable contour\nFinished: %s out of %d[/b][/color]'%(self.counts,self.img_len)
        box = BoxLayout(orientation='vertical',size_hint = (1, 1))
        plot_layout = BoxLayout(orientation = 'horizontal',size_hint =(1,0.7))
        plot_layout.add_widget(self.plt_images)
        features_layout = MyBoxLayout(orientation = 'horizontal',size_hint =(1,0.13),padding =[40,2])
        settings_layout = MyBoxLayout(orientation = 'vertical',size_hint =(0.5,1),padding =[40,2])
        settings_layout.add_widget(self.sl_label)
        ww_layout = MyBoxLayout(orientation = 'horizontal',size_hint =(1,1))
        ww_layout.add_widget(self.ww_label)
        ww_layout.add_widget(self.WW)
        ww_layout.add_widget(MyLabel(text= '',size_hint =(0.2,1)))
        wl_layout = MyBoxLayout(orientation = 'horizontal',size_hint =(1,1))
        wl_layout.add_widget(self.wl_label)
        wl_layout.add_widget(self.WL)
        wl_layout.add_widget(MyLabel(text= '',size_hint =(0.2,1)))
        settings_layout.add_widget(ww_layout)
        settings_layout.add_widget(wl_layout)
        features_layout.add_widget(self.comment_box)
        features_layout.add_widget(settings_layout)
        button_layout = BoxLayout(orientation='horizontal',size_hint =(1,0.1))
        button_layout.add_widget(self.button_left)
        button_layout.add_widget(self.button_right)

        box.add_widget(self.label1)
        box.add_widget(plot_layout)
        box.add_widget(features_layout)
        box.add_widget(button_layout)
        self.add_widget(box)


class PopupBox(Popup):     #https://stackoverflow.com/questions/30595908/building-a-simple-progress-bar-or-loading-animation-in-kivy
    pop_up_text = ObjectProperty()
    progress_bar = ObjectProperty()
    def on_pre_enter(self):
        self.progress_bar.value = 1
        self.auto_dismiss = False
    def update_pop_up_text(self, p_message):
        self.pop_up_text.text = p_message
    def update_progress(self, value):
        self.progress_bar.value = value

class PopupBox2(Popup):     #https://stackoverflow.com/questions/30595908/building-a-simple-progress-bar-or-loading-animation-in-kivy
    pop_up_text = ObjectProperty()
    def update_pop_up_text(self, p_message):
        self.pop_up_text.text = p_message
        self.pop_up_text.markup = True
        #self.pop_up_text.size_hint = (None,None)
        self.pop_up_text.bind(size=self.pop_up_text.setter('text_size'))

class ImageButton(ButtonBehavior, Image):
    pass


'''
class TelegramCallback():
    config = {'token': '750823998:AAGKz1rqN1BGjvsJE2IYzcKBYUXRSWHT5Bo',
              'telegram_id':220715886 }# chat_id -1001357772126,config = {'token': '750823998:AAGKz1rqN1BGjvsJE2IYzcKBYUXRSWHT5Bo','telegram_id':220715886 }# chat_id -1001357772126,
    def __init__(self,*args):
        #super(TelegramCallback, self).__init__()
        self.user_id = self.config['telegram_id']
        self.bot = telegram.Bot(self.config['token'])
        self.max_val=0

    def send_message(self, text):
        try:
            self.bot.send_message(chat_id=self.user_id, text=text)
        except Exception as e:
            print('Message did not send. Error: {}.'.format(e))
'''

class Main_App(App):
    title = 'Slice interpolation experiment 1.0'
    Patient_dict={}
    pat_images = []
    Img_ready = False
    test_finished = False
    DL_score =0
    Dr_score =0

    dr_tag = ''
    Test_active= True
    Active_imgs = None
    columns_list =['Participant','E-mail','Speciality','Acknowledgment', 'Comment','Dr_score','Dl_score','Chosen','Not_Chosen']
    Main_stats = pd.DataFrame(data=None,columns = columns_list )
    Description_text = ''
    User_data = ['','','','']

    def build(self):
        #print('\t\t ... building ...')
        self.results_popup = PopupBox2()
        self.sm = ScreenManager()
        self.main_screen = Main_Screen(name='main_scr')
        self.descr_screen = Description_Screen(name='descr_scr')
        self.comp_screen = Comparison_Screen(name='comp_scr')
        self.test_screen = Test_Screen(name='test_scr')
        self.sm.add_widget(self.main_screen)
        self.sm.add_widget(self.descr_screen)
        self.sm.add_widget(self.comp_screen)
        self.sm.add_widget(self.test_screen)

        #Binds
        Window.bind(on_dropfile=self.get_images)
        #self.comp_screen.image_on_bg.bind(on_press=self._open_test_scr)
        self.test_screen.bind(on_enter = self.on_enter_test_scr)
        self.test_screen.button_left.bind(on_release = self._left_button_pr)
        self.test_screen.button_right.bind(on_release = self._right_button_pr)
        self.test_screen.WL.bind(value = self.update_window_params)
        self.test_screen.WW.bind(value = self.update_window_params)
        self.descr_screen.md_check_box.bind(active=self.on_checkbox_active)
        self.descr_screen.rad_check_box.bind(active=self.on_checkbox_active)
        self.descr_screen.radonc_check_box.bind(active=self.on_checkbox_active)
        self.descr_screen.cs_check_box.bind(active=self.on_checkbox_active)
        self.descr_screen.stud_check_box.bind(active=self.on_checkbox_active)
        self.descr_screen.oth_check_box.bind(active=self.on_checkbox_active)
        self.descr_screen.u_name.bind(text = self.on_user_name)
        self.descr_screen.u_email.bind(text = self.on_user_email )
        self.descr_screen.ack_check_box.bind(active = self.on_checkbox_active2)
        self.comp_screen.bind(on_enter = self.on_enter_comp_screen)
        
        #syn image variables
        self.nn_score = 0
        self.ws_score = 0
        self.ln_score = 0
        self.mp_score = 0
        self.bs_score = 0
        self.syn_list = []
        #self.syn_types = ['BSpline', 'CosineWindowedSinc', 'Linear', 'NearestNeighbor']
        #self.syn_types = ['BSpline', 'NearestNeighbor']
        self.syn_types = ['NearestNeighbor']
        #self.syn_types = ['BSpline']

        #self.results_popup.back_to_menu.bind(on_release = self._open_main_scr)
        self.results_popup.quit_app.bind(on_release = self.main_screen.close_app)
        self.read_descr_text()
        #self.tg_callback = TelegramCallback()
        
        f = open("../path.txt", "r")
        path = f.read()
        f.close()
        self.get_images(path)

        return self.sm

    def on_enter_comp_screen(self,*kwargs):
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__))[:-len(os.path.dirname(os.path.abspath(__file__)).split('\\')[-1])],'test_data')
        try:
            self._on_file_drop(Window, data_path)
        except:
            print('Drag folder manually')

    def read_descr_text(self):
        pass
        #print(self.Description_text)
        #self.descr_screen.descr_text = self.Description_text#"Thank you for agreeing to participate in this experiment!\n---------------\nFor the record, we would like to register your name, e-mail (in case we want you to redo the test to look at the reproducibility of your evaluation or if we would like to include you as coauthor or put you in the acknowledgements. It is [b]OK[/b] if you prefer to stay anonymous. To make further sub analysis we also would like to specify your training: [b]Medical Doctor, Radiologist, Radiation oncologist, Computer scientist, Student, Other[/b] specify.\n\nWe are going to show you two images per screen: One segmented manually by doctors and another one segmented automatically by a software. We would like you to indicate which contour is, according to you, the most accurate. For example, a contour encompassing some normal tissues around the tumors should be seen as inaccurate, same for a contour missing part of the tumor. It could be that you will find that both contours are not correct. In that case, we would like you to indicate the less bad one and leave a comment. We do not ask you to evaluate the 'cosmetic' of the contour. Some contours can look unusual but it doesn’t mean that they are not accurate. If you want to add comments please write it in the dedicated window.\n\nIn the end of the experiment file  'experiment_results.xlsx' will be generated, please send this file to [b]w.rogers@maastrichtuniversity.nl[/b]"

    def shuffle_imgs(self):
        #self.pat_images = shuffle(self.pat_images)
        pass

    def get_images(self, file_path):
        self.test_screen.img_len = 12
        
        try:
            path = file_path.decode("utf-8")
        except:
            path = file_path

        files = os.listdir(path)
        #print('\t\t ... getting {} images ...'.format(len(files)))
        
        self.pat_images = []
        self.ground_truths = []
        self.Img_ready = False

        ids = np.unique([file[:16] for file in files]).squeeze()
        
        #np.random.shuffle(ids)

        for n, id in enumerate(ids):
            if n > self.test_screen.img_len-1: break
            #print(id)
            #print([file for file in files if file.find(id) > -1])
            id_files = [file for file in files if file.find(id) > -1]
            #print([file for file in id_files if file.find('PixelCNN') > -1])
            tru_file = [file for file in id_files if file.find('PixelCNN') > -1][0]
            #img1 = np.load(os.path.join(file_path, tru_file))
            img1 = imread(os.path.join(file_path, tru_file), 0)
            #print('TRU:', img1.min(), img1.max())
            #img1 += 1
            img1 = img1 / 255
            img1 = img1 * 4095
            img1 = img1 - 1024
            #print('TRU:', img1.min(), img1.max())
            
            #print(id)
            #print([file for file in files if file.find(id) > -1])
            id_files = [file for file in files if file.find(id) > -1]
            #print([file for file in id_files if file.find('tru') > -1])
            tru_file = [file for file in id_files if file.find('one') > -1][0]
            #img0 = np.load(os.path.join(file_path, gtru_file))
            img0 = imread(os.path.join(file_path, tru_file), 0)
            #print('TRU:', img1.min(), img1.max())
            #img0 += 1
            img0 = img0 / 255
            img0 = img0 * 4095
            img0 = img0 - 1024
            self.ground_truths.append(img0)
            #print('TRU:', img0.min(), img0.max())

            syn = np.random.choice(self.syn_types)
            self.syn_list.append(syn)
            #print("SYN:", syn)
            syn_file = [file for file in id_files if file.find(syn) > -1][0]
            path = os.path.join(file_path, syn_file)
            #print('SYN PATH:', path)
            img2 = imread(os.path.join(file_path, syn_file), 0)
            #print('TRU:', img2.min(), img2.max())
            #img2 += 1
            img2 = img2 / 255
            img2 = img2 * 4095
            img2 = img2 - 1024
            #print('TRU:', img2.min(), img2.max())

            self.pat_images.append([img1, img2])
            #print()

    def on_checkbox_active(self,checkbox, value):
        #print('CHECKTEST 1:', checkbox, checkbox.text, value)
        if value:
            #print('The checkbox', checkbox.text, 'is active')
            self.User_data[2] = checkbox.text
            
    def on_checkbox_active2(self,checkbox, value):
        #print('CHECKTEST 2:', checkbox, checkbox.text, value)
        if value:
            #print('The checkbox', checkbox.text, 'is active')
            self.User_data[3] = checkbox.text

    def on_user_name(self,*args):
        name = self.descr_screen.u_name.text
        #print('The name is', name)
        self.User_data[0] = name

    def on_user_email(self,*args):
        name = self.descr_screen.u_email.text
        #print('The name is', name)
        self.User_data[1] = name

    def on_enter_test_scr(self,*args):
        self.update_test_screen()

    def update_test_screen(self,*args):
        #print('\t\t ... Update test screen ...')
        #print('There are {} images in the stack'.format(len(self.pat_images)))
        self.shuffle_imgs()
        img = self.pat_images.pop()
        #print('There are {} images in the stack after pop'.format(len(self.pat_images)))
        
        '''
        print(len(img))
        plt.figure(figsize=(16, 16))
        plt.subplot(1,2,1)
        plt.imshow(img[0])
        plt.subplot(1,2,2)
        plt.imshow(img[1])
        plt.show()
        '''

        self.Active_GT = self.ground_truths.pop()

        if np.random.randint(10)>4:
            self.dr_tag = 'left'
            self.Active_imgs = (img[0], img[1])

        else:
            self.dr_tag = 'right'
            self.Active_imgs = (img[1], img[0])

        #print('--------------------------------------------------->', self.dr_tag)

        self.update_window_params()
        #self.test_screen.update_images(img,dr,dl,self.dr_tag)

    def update_window_params(self,*args):
        WW_val = self.test_screen.WW.value
        WL_val = self.test_screen.WL.value
        self.test_screen.ww_label.text = '[color=#000000]WW: %s[/color]'%str(WW_val)
        self.test_screen.wl_label.text = '[color=#000000]WL: %s[/color]'%str(WL_val)

        up_lim,low_lim = WL_val+WW_val/2,WL_val-WW_val/2
        if low_lim< -1000:
            low_lim = -1000

        img = self.Active_imgs[0]
        new_img = img.copy()
        new_img[np.where(new_img<low_lim)] = low_lim
        new_img[np.where(new_img>up_lim)] = up_lim

        img = self.Active_imgs[1]
        new_img2 = img.copy()
        new_img2[np.where(new_img2<low_lim)] = low_lim
        new_img2[np.where(new_img2>up_lim)] = up_lim

        org_img = self.Active_GT.copy()
        org_img[np.where(org_img<low_lim)] = low_lim
        org_img[np.where(org_img>up_lim)] = up_lim

        self.test_screen.update_images(org_img, new_img, new_img2, self.dr_tag)

    def _left_button_pr(self,*args):
        if re.search('left',self.dr_tag) and self.Test_active :
            #print('Correct, its Doctor')
            self.Dr_score+=1
            syn = self.syn_list.pop()
            self._update_stats([1,0, 'PixelCNN', syn])

        elif re.search('right',self.dr_tag) and self.Test_active :
            #print('Nope its a deep learning!')
            self.DL_score+=1
            syn = self.syn_list.pop()
            self._update_stats([0,1,syn, 'PixelCNN'])
        else:
            pass

        #print('\n--------------')
        #print(self.dr_tag)
        #print(self.Dr_score)
        #print(self.DL_score)
        #print('--------------')

        if len(self.pat_images)==0:
            self.open_pop_up2()
            self.Test_active =False
            if not self.test_finished:
                writer = pd.ExcelWriter("experiment_results.xlsx")
                self.Main_stats.to_excel(writer,'Sheet1')
                writer.save()
                os.replace("experiment_results.xlsx", "../experiment_results.xlsx")
                #tg_thread = threading.Thread(target=self.tg_callback.send_message('%s has finished the test with results: Doctor: %d, Deep learning: %d'%(self.User_data[0],self.Dr_score,self.DL_score)))
                #tg_thread.start()
            self.test_finished = True

        else:
            self.test_screen.count+=1
            self.test_screen.label1.text='[color=#000000][b]Please choose preferable image\nFinished: %s out of %d[/b][/color]'%(self.test_screen.count,self.test_screen.img_len)
            self.update_test_screen()

    def _right_button_pr(self,*args):
        if re.search('right',self.dr_tag) and self.Test_active :
            #print('Correct, its Doctor')
            self.Dr_score+=1
            syn = self.syn_list.pop()
            self._update_stats([1,0,'PixelCNN', syn])
        elif re.search('left',self.dr_tag) and self.Test_active :
            #print('Nope its a deep learning!')
            self.DL_score+=1
            syn = self.syn_list.pop()
            self._update_stats([0,1,syn, 'PixelCNN'])
        else:
            pass

        #print('\n--------------')
        #print(self.dr_tag)
        #print(self.Dr_score)
        #print(self.DL_score)
        #print('--------------')

        if len(self.pat_images)==0:
            self.open_pop_up2()
            self.Test_active =False
            if not self.test_finished:
                writer = pd.ExcelWriter('experiment_results.xlsx')
                self.Main_stats.to_excel(writer,'Sheet1')
                writer.save()
                os.replace("experiment_results.xlsx", "../experiment_results.xlsx")
                #tg_thread = threading.Thread(target=self.tg_callback.send_message('%s has finished the test with results: Doctor: %d, Deep learning: %d'%(self.User_data[0],self.Dr_score,self.DL_score)))
                #tg_thread.start()
            self.test_finished = True
        else:
            self.test_screen.count+=1
            self.test_screen.label1.text='[color=#000000][b]Please choose the better quality image\nFinished: %s out of %d[/b][/color]'%(self.test_screen.count,self.test_screen.img_len)
            self.update_test_screen()

    def _update_stats(self,button_score):
        if self.test_screen.comment_box.text == '':
            comments ='No comments'
        else:
            comments = self.test_screen.comment_box.text

        #print('Stats:', type(self.User_data[0]),type(self.User_data[1]),type(self.User_data[2]),type(comments),type(button_score[0]),type(button_score[1]))
        #print('Stats:', self.User_data[0],self.User_data[1],self.User_data[2],comments,self.dr_tag,button_score[0],button_score[1])

        series = pd.Series([self.User_data[0],self.User_data[1],self.User_data[2], self.User_data[3],comments,button_score[0],button_score[1],button_score[2], button_score[3]],index = self.columns_list)

        #print(series)

        self.Main_stats = self.Main_stats.append(series,ignore_index=True)

    def open_pop_up2(self,*args):
        self.results_popup.open()
        if self.Dr_score>self.DL_score:
            dr_color = '[color=#00f662]'
            dl_color = '[color=#f60000]'
        elif self.Dr_score<self.DL_score:
            dr_color = '[color=#f60000]'
            dl_color = '[color=#00f662]'
        else:
            dr_color = '[color=#00f662]'
            dl_color = '[color=#00f662]'

        self.results_popup.update_pop_up_text('[b]Congratulations you are finished![/b]\nHere are the results:\n\n    [size=25]%s%d[/color][/size]      Times you have chosen PixelCNN images.\n    [size=25]%s%d[/color][/size]      Times you have chosen other interpolation methods.\n\nFile with results have been generated: %s\nPlease send it to: [b]w.rogers@maastrichtuniversity.nl[/b]'%(dr_color,self.Dr_score,dl_color,self.DL_score,str(os.path.dirname(os.path.abspath(__file__))+'\experiment_results.xlsx')))

    def _open_test_scr(self,obj):
        if self.Img_ready:
            self.DL_score =0
            self.Dr_score =0
            self.sm.current = 'test_scr'

    def _open_main_scr(self,obj):
        self.results_popup.dismiss()
        self.comp_screen.image_bg = './imgs/main.png'
        self.sm.current = 'main_scr'
        self.Img_ready = False

    def show_popup(self):
        self.pop_up = Factory.PopupBox()
        self.pop_up.update_pop_up_text('Running some task...')
        self.pop_up.open()


if __name__ == '__main__':
    Window.size = (1280, 800)
    Main_App().run()