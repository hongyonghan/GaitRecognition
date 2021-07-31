from flask import Flask, render_template, request, Response, redirect, url_for, flash
from werkzeug import secure_filename
import os
import sys
import utils.videoSender as video_sender
import gait as gait
import utils.frameDivider as frame_divider
import core.extract_embeddings as embeddings_extractor
import core.train_model as classfier
import core.recognize_video as recognizer
from flask_toastr import Toastr
import io
import pandas as pd
import threading
from trainsplitvideo import TrainVideoToFrames
from testsplitvideo import TestVideoToFrames
from werkzeug.utils import secure_filename
from copy_dir import copy_dir
from webtest import modeltest
from webtrain import modeltrain
from testpretreatment import TestPreImage
from trainpretreatment import TrainPreImage
from webtest import modeltest
from delfile import DelFiles
app = Flask(__name__)
toastr = Toastr(app)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

UPLOAD_FOLDER = 'facedataset'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

@app.route('/')
def index():
    return render_template('home.html')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
global empName
@app.route('/upload/',methods = ['GET','POST'])
def upload_file():
    if request.method =='POST':
        empName = request.form['name']
        fileList = request.files.getlist('file[]')
        print("empname", empName)
        print("fileList", fileList)

        vidFile = request.files['file']
        vidFileName = 'video.mp4'
        vidFile.save(vidFileName)
        vidFile.stream.seek(0)
        saveLocation = os.path.sep.join([app.config['UPLOAD_FOLDER'], empName]) #面部数据相对路径
        # print("saveLocation",saveLocation) #测试使用
        frame_divider.vid_to_images(vidFileName, saveLocation, 15)  #facedata


        trainfile = request.files['file']

        basepath = os.path.dirname(__file__)  # 当前文件所在路径
        print("basepath",basepath)

        upload_path = os.path.join(basepath, 'video/',secure_filename(trainfile.filename))  # 注意：没有的文件夹一定要先创建，不然会提示没有该路径
        print("upload_path",upload_path)  #得到视频的全路径
        trainfile.save(upload_path)  #将视频保存到video文件夹下
        videosplit = TrainVideoToFrames()  # 视频切割
        videosplit.run(upload_path, empName, "knn",50,[2,2,2])  # 视频切割模块

        if fileList and len(fileList) != 1:
            print(fileList, len(fileList))
            if not (os.path.exists(saveLocation)):
                os.mkdir(saveLocation)
            for f in fileList:
                filename = secure_filename(f.filename.split('/')[1])
                f.save(os.path.join(saveLocation, filename))

        flash(u'New employee was successfully registered!', 'success')  #提示切割完成
        return redirect(url_for('index'))

    return "render_template('file_upload.html')"

@app.route('/extraction/',methods = ['GET','POST'])
def extraction():
    if request.method == 'POST':
        confidence = request.form['range']
        num_of_embeddings = embeddings_extractor.extract_face_embeddings(confidence)

        DelFiles("dataset/test_gallery_pre")  # 删除已经训练的图像
        TrainPreImage() #对数据进行预处理。
        # copy_dir("dataset/test_gallery_pre","dataset/train") #拷贝文件到训练文件夹中

        print("图片预处理完成")


        flash(u'{} facial embeddings were extracted!'.format(num_of_embeddings), 'success')

        # return render_template('home.html', number=0)
        return render_template('home.html', number=str(num_of_embeddings))

@app.route('/train/',methods = ['GET','POST'])
def train():
    if request.method == 'POST':
        message = classfier.train_classifier()
        flash(u'The classifier was successfully trained!', 'success')
        return render_template('home.html', message=message)

@app.route('/surveillance/',methods = ['GET','POST'])
def surveillance():
    if request.method =='POST':
        confidence = request.form['range1']
        testfile = request.files['file1']
        testpath = os.path.dirname(__file__)  # 当前文件所在路径
        print("当前路径为",os.path.abspath("."))  #测试使用
        # os.chdir("../")
        # print("当前路径为", os.path.abspath("."))
        os.chdir("F:\pythonProject\gaaa\GaitRecognition")   #设置项目的绝对路径。
        os.getcwd()
        sys.path.append("F:\pythonProject\gaaa\GaitRecognition")

        DelFiles("dataset/test_probe")  # 删除原始图像
        test_upload_path = os.path.join(testpath, 'video/',
                                        secure_filename(testfile.filename))  # 注意：没有的文件夹一定要先创建，不然会提示没有该路径
        print("upload_path", test_upload_path)  # 得到视频的全路径
        # vidFile.save(upload_path)  # 将视频保存到video文件夹下
        revideosplit = TestVideoToFrames()  # 视频切割
        revideosplit.run(test_upload_path, "XXX", "knn", 50, [2, 2, 2])  # 视频切割模块
        DelFiles("dataset/test_probe_pre")  # 删除预处理的文件
        TestPreImage()  # 图像预处理
        # copy_dir("dataset/test_probe_pre", "dataset/test_gallery_pre")  ## 复制文件的函数
        DelFiles("work/partition")  # 删除生成的文件


        vidFile = request.files['file1']
        filename = 'surveillance.mp4'
        vidFile.save(filename)
        vidFile.stream.seek(0)
        myfile = 'surveillance.mp4'

        t1 = threading.Thread(target=recognizer.recognize, args=(confidence, myfile))
        # t2 = threading.Thread(target=video_sender.send_surveillance_video, args=[myfile])
        # t2 = threading.Thread(target=gait.gait_reply)
        # gait.gait_reply()
        t1.daemon = True
        # t2.daemon = True
        t1.start()
        # t2.start()
        return render_template('video.html')

@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(recognizer.generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

@app.route("/current_identification/")
def current_identification():
	# return the response generated along with the specific media
	# type (mime type)
    if request.headers.get('accept') == 'text/event-stream':
        return Response(recognizer.current_identification(), mimetype ='text/event-stream')

@app.route("/all_count/")
def all_count():
	# return the response generated along with the specific media
	# type (mime type)
    if request.headers.get('accept') == 'text/event-stream':
        return Response(recognizer.all_count(), mimetype ='text/event-stream')

@app.route("/gait_reply/")
def gait_reply():
    if request.headers.get('accept') == 'text/event-stream':
        return Response(gait.gait_reply(), mimetype ='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True, threaded=True)