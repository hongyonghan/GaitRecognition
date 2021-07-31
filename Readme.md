本步态识别系统主要基于GaitSet模型进行实现。在尝试部署本系统之前，建立理解GaitSet模型的网络结构、训练和推理方法。

**系统的实现效果如视频所示：**

[演示视频](https://www.bilibili.com/video/BV12f4y1G7V5/)

**由于模型较大，部分模型文件存储在百度云盘。**

[链接](https://pan.baidu.com/s/1N-kopPL-vr5GROWb7bqH4w )提取码：33mb 

## 具体部署过程

#### 1.下载代码

#### 2.安装requirements.txt

#### 3.下载百度网盘的work文件夹到GaitRecognition文件夹下并进行解压，并将里面的openface_nn4.small2.v1.t7文件移动到GaitRecognition文件夹下。

#### 4.设置app文件中项目的运行路径。

```python
os.chdir("F:\pythonProject\GaitRecognition")   #设置项目的绝对路径。
os.getcwd()
sys.path.append("F:\pythonProject\GaitRecognition")
```

#### 5.运行app.py文件