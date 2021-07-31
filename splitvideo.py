import os
import numpy as np
import cv2
import json
from testdb import dbmain
import shutil

class VideoToFrames:

    print("Video To Frames Ready to run!")
    def draw_person(self,image, persont):
        x, y, w, h = persont
        cv2.rectangle(image, (x, y), (x + w, y + h), (255), 2)
    def noise(self,img):
     #定义结构元素
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
            #开闭运算，先开运算去除背景噪声，再继续闭运算填充目标内的孔洞
            opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
            closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
            #cv2.imshow("closed", closed)
            return closed
    def getid(self,name):
                #读取姓名id
        print("in splitvideo")
        # f = open('directory/user_info.json', 'r', encoding='utf-8')
        # res=f.read()  #读文件
        sql="SELECT ID, Name  from person"
        l=dbmain(sql)
        self.data=l
        #f.close()
        print(list(self.data.keys()),list(self.data.values()))
                # global list
        if name in list(self.data.values()):
            id=int(list(l.keys())[list(l.values()).index(name)])
        else:
            id=len(list(self.data.keys()))
        return id

    def copy_dir(self,src_path, target_path):
        if os.path.isdir(src_path) and os.path.isdir(target_path):
            filelist_src = os.listdir(src_path)
            for file in filelist_src:
                path = os.path.join(os.path.abspath(src_path), file)
                if os.path.isdir(path):
                    path1 = os.path.join(os.path.abspath(target_path), file)
                    if not os.path.exists(path1):
                        os.mkdir(path1)
                    copy_dir(path, path1)
                else:
                    with open(path, 'rb') as read_stream:
                        contents = read_stream.read()
                        path1 = os.path.join(target_path, file)
                        with open(path1, 'wb') as write_stream:
                            write_stream.write(contents)
            return True

        else:
            return False
    def run(self,path,name,mode,frames,set):
        filename = os.path.basename(path)  ##获取文件名
        catalog = path.split("-")[1] + "-" + path.split("-")[2]  # 获取中间目录名nm-01
        angle = path.split('-')[3].split(".")[0]  # 获取角度

        id=self.getid(name)
        print("id=",id)
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        if mode=="knn":
            fgbg = cv2.createBackgroundSubtractorKNN(history = frames,detectShadows=1)
        else:
            fgbg = cv2.createBackgroundSubtractorMOG2(history=frames , detectShadows = 1)
        #count = 1
        video = cv2.VideoCapture(path)
        #total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        #video.set(cv2.CAP_PROP_POS_FRAMES, 2)
        print(video.isOpened())
        framerate = video.get(cv2.CAP_PROP_FPS)
        print(framerate)
        try :
            id2="{0:03}".format(id)
            # id3 = "{0:02}".format(id)
            # print("id",id)
            # print("id2",id2)
            # print("id3",id3)
            os.makedirs("dataset/test_gallery/"+str(id2) )

        except:
            print("diractory maked")
        list = os.listdir("dataset/test_gallery/"+id2)
        seq = len(list)+1
        try:
            # os.makedirs("dataset/test_gallery/"+str(id2)+"/nm-{0:02}".format(seq)+"/090" )
            os.makedirs("dataset/test_gallery/" + str(id2) + "/" + catalog.format(seq) + "/" + angle)
        except:
            print("/nm diractory maked")
        i=1
        mark=0
        markf=0
        mark5=0
        images=None
        while (video.isOpened()):
            #frameId = video.get(1)
            success, image = video.read()

            if success!=1:
                break
            image=cv2.resize(image,(400,300), interpolation = cv2.INTER_AREA)
            image_copy = image.copy()
            fgmask = fgbg.apply(image_copy)
             # 对原始帧进行腐蚀膨胀去噪
            th = cv2.threshold(fgmask, 244, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (set[0], set[1])), iterations=2)
            th = cv2.erode(thresh, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (set[2], set[2])), iterations=2)
            if image is not None:


                #frame = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)#转为灰度值图
                rects, weights = hog.detectMultiScale(image)
                print("person:",len(rects))

                print(i)
                mark5=mark5+1
                if mark5>60:
                     mark5=1
                     mark=0
                     markf=0

                if len(rects)>0:
                    mark=mark+1
                    #print(mark)
                    if mark>10:
                        images=1
                else:
                    markf=markf+1
                    if markf>30:
                        images=None
                if( images is not None ):
                    #cv2.imshow("BGR2GRAY", frame)
                    #cv2.imshow("fgmask", fgmask)
                    #ret, thresh = cv2.threshold(fgmask,244,255,cv2.THRESH_BINARY)#转为二值图
                    #image=thresh
                    #image= self.noise(thresh)
                    contours, _ = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                    #cv2.imshow("thresh", thresh)
                    cimg = np.zeros_like(th)
                    #cv2.imshow("thresh", thresh)
                    print("contours:",len(contours))
                    for j in range(len(contours)):
                            #x,y,w,h = cv2.boundingRect(contours[i]) #绘制矩形
                            #cimg = cv2.rectangle(cimg,(x,y),(x+w,y+h),(255,255,255),2)
                            cv2.drawContours(cimg, contours, j, color=255, thickness=-1)
                            #cimg = self.noise(cimg)
                            #cv2.imshow("drawContours", cimg)
                    if i >1:#跳过前n张
                        # filename = "dataset/test_gallery/{0:03}".format(id)+"/nm-{0:02}".format(seq) +"/090"+ "/{0:03}-nm-".format(id)+"{0:02}-".format(seq)+"090-"+ "{0:03}".format(i-4) + ".png"
                        filename = "dataset/test_gallery/{0:03}".format(id) + "/" + catalog + "/" + angle + "/{0:03}-".format(id) + catalog + "-" + angle + "-" + "{0:03}".format(i - 4) + ".png"
                        print(filename)
                        cv2.imwrite(filename,cimg)
            i=i+1
            if (success != True):
                break
            if i>999:
                break
        video.release()
        sql="INSERT INTO person (ID,Name) \
                   VALUES( '{}', '{}')".format(str(id),str(name))
        dbmain(sql)
        #写入姓名id保存为json文件
        # self.data["user1"].update({str(id):str(name)})
        # fw = open('directory/user_info.json', 'w', encoding='utf-8')
        # json.dump(self.data,fw,ensure_ascii=False,indent=4)
        # fw.close()
        #删除空白帧
        # listd=os.listdir("dataset/test_gallery/"+str(id2)+"/nm-{0:02}".format(seq)+"/090")
        listd = os.listdir("dataset/test_gallery/" + str(id2) + "/" + catalog + "/" + angle)
        print(listd)
        # for l in listd[-1:]:#删除后50帧
        #     os.remove("vsil/"+str(id2)+"/nm-{0:03}/".format(seq)+l+"/090")
        #     print("deleting",l)
        # for lh in listd[0:3]:
        #     os.remove("vsil/"+str(id2)+"/nm-{0:03}/".format(seq)+lh)
        #     print("deleting",lh)
        # print('done')
        #count+=1



        print("done")
if __name__ == '__main__':
    f=VideoToFrames()
    path=r"video/001-nm-01-090.avi"
    name="S001"
    f.run(path,name,"knn",50,[2,2,2])
    # src_path = "dataset/test_probe"
    # target_path = "dataset/test_gallery"
    # f.copy_dir(src_path, target_path)


    print("done")

