import os

def mkdir(fn,opt=0):
    if opt == 1 :# until the last /
        fn = fn[:fn.rfind('/')]
    if not os.path.exists(fn):
        if opt==2:
            os.makedirs(fn)
        else:
            os.mkdir(fn)



