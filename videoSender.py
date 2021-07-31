
from webtest import modeltest

def gait_reply():
    reply_gait = modeltest()

    yield "data: " + reply_gait + "\n\n"