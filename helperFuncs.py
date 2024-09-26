from datetime import datetime

def currentTime():
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]