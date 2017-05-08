# -*- coding: utf-8 -*-
import zipfile
import os


def ziplib():
    """
    将线上运行代码打包
    :return:
    """
    dir = os.path.dirname(__file__)  # this should point to your packages directory
    libpath1 = os.path.join(dir, '../../jd')

    # set it as save file path
    zippath = './jd.zip'  # some random filename in writable directory
    zf = zipfile.PyZipFile(zippath, mode='w')

    try:
        zf.debug = 3  # making it verbose, good for debugging
        zf.writepy(libpath1)
        return zippath  # return path to generated zip archive
    finally:
        zf.close()


ziplib()
