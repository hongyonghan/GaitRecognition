本步态识别系统主要基于GaitSet模型进行实现。在尝试部署本系统之前，建立理解GaitSet模型的网络结构、训练和推理方法。

**系统的实现效果如视频所示：**

[演示视频](https://www.bilibili.com/video/BV12f4y1G7V5/)

**由于模型较大，部分模型文件存储在百度云盘。**

[链接](https://pan.baidu.com/s/1N-kopPL-vr5GROWb7bqH4w )提取码：33mb 

## 具体部署过程

#### 1.下载代码

#### 2.安装requirements.txt
- 当时的环境大多数都在requirement.txt和requirement.txt当中。毕业之后服务器的环境就没了。如果环境中有些版本不对，可以尝试使用requirements717.txt中的相关版本。这个是一个学弟在22年跑出来的环境的配置。

#### 3.下载百度网盘的work文件夹到GaitRecognition文件夹下并进行解压，并将里面的openface_nn4.small2.v1.t7文件移动到GaitRecognition文件夹下。

#### 4.设置app文件中项目的运行路径。

```python
os.chdir("F:\pythonProject\GaitRecognition")   #设置项目的绝对路径。
os.getcwd()
sys.path.append("F:\pythonProject\GaitRecognition")
```
我当时只是设置了app文件中surveillance函数设置了运行路径就可以运行了。后来学弟在部署的时候他的电脑环境下需要设置app文件下upload_file()、extraction()、train()、surveillance()，为了系统更加稳定，推荐在部署的时候也添加上这些函数的运行路径。

#### 5.运行app.py文件


#### 如果部署上出现问题可以参考下面的参考链接：
- https://blog.csdn.net/Leon_____/article/details/122862976?spm=1001.2014.3001.5506
- https://blog.csdn.net/qq_39237205/article/details/124141716?spm=1001.2014.3001.5506
- https://blog.csdn.net/qq_39237205/article/details/124199534?spm=1001.2014.3001.5506
- https://blog.csdn.net/qq_39237205/article/details/124135045?spm=1001.2014.3001.5506
- https://blog.csdn.net/weixin_46694417/article/details/118496986?spm=1001.2014.3001.5506
- https://blog.csdn.net/qq_36731217/article/details/118491537?spm=1001.2014.3001.5506
- https://blog.csdn.net/qq_21464351/article/details/109546421?spm=1001.2014.3001.5506
#### 大家部署成功后欢迎大家把自己的部署出现的问题及解决方案提交commit或者发布issue。
