

## this is useless file ,byhanhongyong
import os
path=r"F:\BaiduNetdiskDownload\DatasetB-1\DatasetB-1\video\001-bg-01-000.avi"
filename = os.path.basename(path)
print(filename)
# print(path.split("-"))
catalog = filename.split("-")[1]+"-"+filename.split("-")[2]
print(catalog)
angle = filename.split('-')[3].split(".")[0]
print(angle)


# print(os.path.basename(path))