# import socket
# import io
#
# SERVER_IP = '10.10.24.129'
# reply_gait = ""
#
# def send_train_video(empName, filename):
#     soc = socket.socket()
#     soc.connect((SERVER_IP,8080))
#
#     print('waiting for connection...')
#
#     soc.send(empName.encode())
#     reply = ''
#     reply = soc.recv(1024).decode()
#     print ("reply: "+reply)
#
#     if(reply == "OK"):
#         with soc:
#             # filename = input('enter filename to send: ')
#             # with open(filename, 'rb') as file:
#             #     sendfile = file.read()
#             bin_file = io.BytesIO(filename.read())
#             soc.sendall(bin_file.read())
#             print('file sent')
#
# def send_surveillance_video(filename):
#     global reply_gait
#     reply_gait = ''
#     soc = socket.socket()
#     soc.connect((SERVER_IP,8081))
#
#     print('waiting for connection...')
#
#     with soc:
#         # filename = vidFile
#         with open(filename, 'rb') as file:
#             bin_file = io.BytesIO(file.read())
#             # sendfile = file.read()
#             soc.send(str(file.tell()).encode())
#             soc.sendall(bin_file.read())
#             print('file sent')
#             reply = ''
#             reply = soc.recv(1024).decode()
#             reply_gait = reply
#             print ("reply: "+reply)
