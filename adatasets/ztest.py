# pylint: disable=C0321,C0103,C0301,E1305,E1121,C0302,C0330,C0111,W0613,W0611,R1705
# -*- coding: utf-8 -*-
import os, sys, time, datetime,inspect


##################################################################################################
def test1():
   from utilmy import (os_makedirs, Session, global_verbosity, os_system  
                       
                      )
 

   os_makedirs('ztmp/ztmp2/myfile.txt')
   os_makedirs('ztmp/ztmp3/ztmp4')
   os.system("ls ztmp")


   print('verbosity', global_verbosity(__file__, "../confi.json", 40,))


   sess = Session("ztmp/session")
   sess.save('mysess', globals(), '01')
   os.system("ls ztmp/session")


   res = os_system( f" ls . ",  doprint=True) 
   print(res)


   print("success")


if __name__ == "__main__":
    import fire
    fire.Fire()





