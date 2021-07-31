
from webtest import modeltest
import sys

def gait_reply():
    global reply_gait
    try:
        reply_gait = modeltest()
        print('dataaa')
        print(reply_gait)
    finally:
        # sys.exit(1) #程序异常退出。
        return "data: " + reply_gait + "\n\n"
    # return "data: " + reply_gait + "\n\n"