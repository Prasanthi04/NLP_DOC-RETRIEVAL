# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 15:06:21 2017

@author: prasa
"""

import os, zipfile
import shutil

dir_name = "C:/Drive/FALL2017/NLP/Project/data_weneed"
extension = ".zip"

os.chdir(dir_name) # change directory from working dir to dir with files

for item in os.listdir(dir_name): # loop through items in dir
    if item.endswith(extension): # check for ".zip" extension
        file_name = os.path.abspath(item) # get full path of files
        zip_ref = zipfile.ZipFile(file_name) # create zipfile object
        zip_ref.extractall(dir_name) # extract file to dir
        zip_ref.close() # close file
        os.remove(file_name) # delete zipped file


TargetFolder = "C:/Drive/FALL2017/NLP/Project/data_all_weneed"
for root, dirs, files in os.walk((os.path.normpath(dir_name)), topdown=False):
        for name in files:
            if name.endswith('.html'):
                print("Found")
                SourceFolder = os.path.join(root,name)
                shutil.copy2(SourceFolder, TargetFolder) #copies csv to new folder