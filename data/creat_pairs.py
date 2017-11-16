#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 20:08:43 2017

@author: zyx
"""

import os
import sys
import random  
#defaultencoding = 'utf-8'
#if sys.getdefaultencoding() != defaultencoding:
#    reload(sys)
#sys.setdefaultencoding(defaultencoding)

if __name__ == '__main__':

    root = '/home/ubuntu/HD4T/data/200person/org'
    namelist = os.listdir(root)
    match_pairs_num = 3000
    mismatch_pairs_num = 3000
    
    f = open('test.txt','w')
    
    for i in range(match_pairs_num):
        nameindex = random.randint(1, len(namelist) - 1)
        name = namelist[nameindex]
        name_path = os.path.join(root,name)
        imglist = os.listdir(name_path)
        imgindex1 = random.randint(1, len(imglist) - 1)
        imgindex2 = random.randint(1, len(imglist) - 1)
        while imgindex1 == imgindex2:
            imgindex2 = random.randint(1, len(imglist) - 1)
        f.write('%s\t%d\t%d\n'%(name,imgindex1,imgindex2))

    for i in range(mismatch_pairs_num):
        nameindex1 = random.randint(1, len(namelist) - 1)
        name1 = namelist[nameindex1]
        name_path1 = os.path.join(root,name1)
        imglist1 = os.listdir(name_path1)
        imgindex1 = random.randint(1, len(imglist1) - 1)

        nameindex2 = random.randint(1, len(namelist) - 1)
        name2 = namelist[nameindex2]
        name_path2 = os.path.join(root, name2)
        imglist2 = os.listdir(name_path2)
        imgindex2 = random.randint(1, len(imglist2) - 1)

        f.write('%s\t%d\t%s\t%d\n'%(name1,imgindex1,name2,imgindex2))
    
    
    f.close()
