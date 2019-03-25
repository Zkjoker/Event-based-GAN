import cv2
import numpy as np

class Eventtxt2pic():
    def __init__(self,root):
        self.eventfile = root+'/events.txt'
        self.savefile=root
        self.shape=(180,240)
        self.emptyImage=np.full(self.shape,128,dtype='uint8')
        self.timeinteval=0.0128
        self.fill_frame=self.fill_frame2  ##确定堆叠方式
    def fill_frame1(self,emptyImage,col, row, polarity):
        '以叠加方式堆叠event帧'
        if ((polarity == 1) and (emptyImage[col][row] > 0)):
            emptyImage[col][row] -= 127
        elif ((polarity == 0) and (emptyImage[col][row] < 255)):
            emptyImage[col][row] += 127
    def fill_frame2(self,emptyImage, col, row, polarity):
        '以最后出现的形式堆叠event帧'
        if (polarity == 1):
            emptyImage[col][row] = 255
        elif (polarity == 0):
            emptyImage[col][row] = 0
    def start_stack(self):
        filenamepre = 'event_'
        piccount=0
        pointnum=1
        with open(self.eventfile) as f:
            while True:
                line = f.readline()
                if not line: break
                event = line.split()
                timestamp = float(event[0])
                row = int(event[1])
                col = int(event[2])
                polar = int(event[3])
                if (timestamp < self.timeinteval * (piccount + 1)):
                    self.fill_frame(self.emptyImage, col, row, polar)
                    print("正在填充第{}张图像的第{}个event像素点！".format(piccount, pointnum))
                    pointnum = pointnum + 1
                else:
                    filename = filenamepre + str(piccount).zfill(8)
                    cv2.imwrite((self.savefile+("/recovery/{}.png".format(filename))), self.emptyImage)
                    print("Program has recovered {} frames!".format(piccount))
                    piccount = piccount + 1
                    pointnum = 1
                    self.emptyImage = np.full(self.shape, 128, dtype='uint8')
                    self.fill_frame(self.emptyImage, col, row, polar)
                    print("正在填充第{}张图像的第{}个event像素点！".format(piccount, pointnum))
                    pointnum = pointnum + 1
            filename = filenamepre + str(piccount).zfill(8)
            cv2.imwrite((self.savefile + ("/recovery/{}.png".format(filename))), self.emptyImage)
            print("Program has recovered {} Event-frames!".format(piccount))
        print("Finish recovering!")

if __name__=='__main__':
    eventfile='../data/urban'
    txt2pic=Eventtxt2pic(eventfile)
    txt2pic.start_stack()