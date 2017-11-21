#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 16:20:29 2017

@author: zyx
"""

import os
import shutil

import sys
defaultencoding = 'utf-8'
if sys.getdefaultencoding() != defaultencoding:
    reload(sys)
    sys.setdefaultencoding(defaultencoding)
    

if __name__ == "__main__":

    root = 'crop'
    list = os.listdir(root)
    count = 0
    for root1, persons, files1 in os.walk(root):
        for person in persons:
            print ('%s'%(person))
            person_path = os.path.join(root, person)
            for root2, dir2, imgs in os.walk(person_path):
                img_index = 0
                for img in imgs:
                    os.rename('%s//%s'%(person_path,img),'%s//%s_%04d_0.png'%(person_path, person, img_index))
                    img_index += 1
            # os.rename('%s' % (person_path), '%s//%07d' % (root, count))
            count += 1
