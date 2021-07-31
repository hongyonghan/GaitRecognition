import os
folder_name = "F:/pythonProject/Overseer-Engine/dataset/train_pre/014/nm-01/180"

#获取文件夹的文件名
file_names = os.listdir(folder_name)
print(file_names)
# print(file_names[1].split("-")[1:])
# print("126"+"-"+"-".join(file_names[1].split("-")[1:]))
os.chdir(folder_name)
for name in file_names:
    os.rename(name, "126"+"-"+"-".join(name.split("-")[1:]))
# filename = os.path.basename(folder_name)
# print(filename)
# # print(path.split("-"))
# catalog = filename.split("-")[1]+"-"+filename.split("-")[2]
# print(catalog)
# angle = filename.split('-')[3].split(".")[0]
# print(angle)

