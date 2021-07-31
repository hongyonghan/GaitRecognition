import os
import shutil
# os.chdir("..")   #修改
def DelFiles(delpath):
    delList = []
    delDir = delpath
    delList = os.listdir(delDir)
    # delList=os.makedirs(delDir)
    for f in delList:
      filePath = os.path.join( delDir, f )
      if os.path.isfile(filePath):
        os.remove(filePath)
        print(filePath + " was removed!")
      elif os.path.isdir(filePath):
        shutil.rmtree(filePath,True)
        print("Directory: " + filePath +" was removed!")