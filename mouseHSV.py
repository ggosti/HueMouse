#
# To run type: 
#	python mouseHSV.py --input video.mov --algo KNN


import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np
import glob
import pandas as pd

import argparse

def nothing(x):
    pass

# median blur and resize
def medianBlurWResize(original,blurPar,scale_percent,x0,x1,y0,y1):
    blur = cv.medianBlur(original, blurPar)#5)
    #scale_percent = 20 # percent of original size
    down_width = int(blur.shape[1] * scale_percent / 100)
    down_height = int(blur.shape[0] * scale_percent / 100)
    down_points = (down_width, down_height)
    frame = cv.resize(blur, down_points, interpolation= cv.INTER_LINEAR)
    if not x1 == None: frame= frame[y0:y1,x0:x1] #crop
    return frame

# find and remove small connected components
def removeSmallCC(fgMask,min_size):    
    #find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv.connectedComponentsWithStats(fgMask, connectivity=8)
    #connectedComponentswithStats yields every seperated component with information on each of them, such as size
    #the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
    sizes = stats[1:, -1]; nb_components = nb_components - 1
    #your answer image
    fgMask = np.zeros((fgMask.shape),dtype=np.uint8)
    #for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            fgMask[output == i + 1] = 255
    return fgMask

parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                              OpenCV. You can process both videos and images.')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='vtest.avi')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='KNN')
parser.add_argument('--rotate', help='rotate.', action=argparse.BooleanOptionalAction, default=False)
args = parser.parse_args()

if args.rotate == True:
	print('rotate',args.rotate)
#if args.algo == 'MOG2':
#    backSub = cv.createBackgroundSubtractorMOG2()
#else:
#    backSub = cv.createBackgroundSubtractorKNN()

capture = cv.VideoCapture(args.input)
if not capture.isOpened():
    print('Unable to open: ' + args.input)
    exit(0)

hname = args.input.split('.')[0]
directory='./'+hname
if not os.path.exists(directory):
    os.makedirs(directory)



th0=255 #high0,high1,high2=50,5,150
th1=80 #high0,high1,high2=50,5,150
th2=150 #high0,high1,high2=50,5,150
th12=189
x0,y0,x1,y1=50,20,325,190
blurPar = 5
scale_percent = 20 # percent of original size

if os.path.isfile(directory+'/pars.csv'):
    df = pd.read_csv(directory+'/pars.csv')
    print(df) 
    print(df.columns) 
    print(df['0'])
    x0,y0,x1,y1=int(df['0'][0]),int(df['0'][1]),int(df['0'][2]),int(df['0'][3])
    th0,th1,th12=int(df['0'][4]),int(df['0'][5]),int(df['0'][6])
    blurPar=int(df['0'][8])
    scale_percent=int(df['0'][9])

#---------------------------------------------------
#change thresholds
#-----------------------------------------------------


cv.namedWindow('image')
cv.namedWindow('th')
    
cv.createTrackbar('th0','th',th0,255,nothing)  
cv.createTrackbar('th1','th',th1,255,nothing) 
cv.createTrackbar('th2','th',th2,255,nothing)   
cv.createTrackbar('th12','th',th12,255,nothing) 
cv.createTrackbar('x0','th',x0,255,nothing)  
cv.createTrackbar('y0','th',y0,255,nothing)      
cv.createTrackbar('x1','th',x1,400,nothing)  
cv.createTrackbar('y1','th',y1,250,nothing)  

while True:
    ret, original = capture.read()
    if original is None:
        break
    if args.rotate:
        original=cv.rotate(original, cv.ROTATE_90_CLOCKWISE)
    
    
    #-------------------------------------------------------------
    #median blur and resize for faster processing and denoising
    #-------------------------------------------------------------
    frame = medianBlurWResize(original,blurPar,scale_percent,x0,x1,y0,y1)
    
    #-------------------------------------------------------------   
    # tresholds HSV
    #-------------------------------------------------------------
    
    # get current positions of four trackbars
    th0 = cv.getTrackbarPos('th0','th')
    th1 = cv.getTrackbarPos('th1','th')
    th2 = cv.getTrackbarPos('th2','th')
    th12 = cv.getTrackbarPos('th12','th')
    x0 = cv.getTrackbarPos('x0','th')
    y0 = cv.getTrackbarPos('y0','th')
    x1 = cv.getTrackbarPos('x1','th')
    y1 = cv.getTrackbarPos('y1','th')
    
    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    #low0 = high0 #low0,low1,low2=70,70,70
    th = 255*np.uint8( (hsv_frame[:,:,0]< th0)* (hsv_frame[:,:,1]< th1)* (hsv_frame[:,:,2]> th2)*  ((hsv_frame[:,:,1] + hsv_frame[:,:,2]) > th12) ) # *(> high2) )
    #print(th0)
    
    cv.imshow('image',frame)
    cv.imshow('th',th)
    k = cv.waitKey(1) & 0xFF
    if k == 27:
        break

print('bkg rem algo',args.algo)
print('th0',th0)
print('th1',th1)
print('th12',th12)
print('x0',x0)
print('y0',y0)
print('x1',x1)
print('y1',y1)
df = pd.DataFrame([x0,y0,x1,y1,th0,th1,th12,args.algo,blurPar,scale_percent], index = ['x0', 'y0', 'x1',  'y1','th0','th1','th12','bkg rem algo','blurPar','scale_percent'])
df.to_csv(directory+'/pars.csv', index=True)

ts = []
xs = []
ys = []
tssmart = []
xssmart = []
yssmart = []
tsfg = []  
xsfg = []
ysfg = []
#trajFrame = np.zeros((1,1),np.uint8)

capture.set(cv.CAP_PROP_POS_FRAMES, 0)

while True:
    ret, original = capture.read()
    if original is None:
        break
    if args.rotate:
        original=cv.rotate(original, cv.ROTATE_90_CLOCKWISE)
   
    
    #-------------------------------------------------------------
    #median blur and resize for faster processing and denoising
    #-------------------------------------------------------------
    frame = medianBlurWResize(original,blurPar,scale_percent,x0,x1,y0,y1)

    #fgMask = backSub.apply(frame,learningRate=0.0030)
    #fgMask[fgMask<255]=0
 
    #-------------------------------------------------------------   
    # remove small connected components from fgMask
    #-------------------------------------------------------------
    min_size = 12  # minimum size of particles we want to keep (number of pixels)
    #fgMask = removeSmallCC(fgMask,min_size)

    #-------------------------------------------------------------   
    # tresholds HSV
    #-------------------------------------------------------------
    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    
    #low0 = high0 #low0,low1,low2=70,70,70
    th = 255*np.uint8( (hsv_frame[:,:,0]< th0)* (hsv_frame[:,:,1]< th1)* (hsv_frame[:,:,2]> th2)* ((hsv_frame[:,:,1] + hsv_frame[:,:,2]) > th12) ) # *(> high2) )

    
    #-------------------------------------------------------------   
    # remove small connected components from th
    #-------------------------------------------------------------
    min_size = 30  # minimum size of particles we want to keep (number of pixels)
    th = removeSmallCC(th,min_size)
        
    #thEye = 255*np.uint8((hsv_frame[:,:,0]< high0) )#*(frame[:,:,1]< low1)*(frame[:,:,2]<low2) )
    
    
    #-------------------------------------------------------------   
    # compute centroids and plot stuff
    #-------------------------------------------------------------
    
    frameAug = frame.copy()
    #smart = np.uint8((th>0))#*(fgMask>0))
    r,c,_ = frame.shape
    yy,xx=np.mgrid[0:r,0:c]
    
    if np.sum(th)>0:
        cy = int(np.sum(yy*th)/np.sum(th))
        cx = int(np.sum(xx*th)/np.sum(th))
    th = cv.cvtColor(th,cv.COLOR_GRAY2RGB)
    
    
    #if np.sum(smart)>0:
    #    cysmart = int(np.sum(yy*smart)/np.sum(smart))
    #    cxsmart = int(np.sum(xx*smart)/np.sum(smart))
    #smart = cv.cvtColor(255*smart,cv.COLOR_GRAY2RGB)
    
    #if np.sum(fgMask)>0:
    #    cyfg = int(np.sum(yy*fgMask[:,:])/np.sum(fgMask[:,:]))
    #    cxfg = int(np.sum(xx*fgMask[:,:])/np.sum(fgMask[:,:]))
    ##print(fgMask.max())
    #fgMask = cv.cvtColor(np.uint8(fgMask),cv.COLOR_GRAY2RGB)

    if np.sum(th)>0:
        #trajFrame[int(cy),int(cx)] = 1
        #frameAug[int(cy),int(cx),:] = [0,0,255]
        cv.circle(frameAug,(cx,cy), 5, (0,255,255), -1)
        #frame[trajFrame] = [255,0,0]
        #smartEye[int(cy),int(cx),:] = [0,0,255]
        #fgMask[int(cyfg),int(cxfg),:] = [0,255,0]
        cv.circle(th,(cx,cy), 5, (0,255,255), -1)
        time = capture.get(cv.CAP_PROP_POS_MSEC)
        if time > 0:
            xs.append(cx)
            ys.append(cy)
            ts.append(time)
        else:
            print('time error',time)    
    #if np.sum(smart)>0:
    #    #trajFrame[int(cy),int(cx)] = 1
    #    #frameAug[int(cy),int(cx),:] = [0,0,255]
    #    cv.circle(frameAug,(cxsmart,cysmart), 5, (0,0,255), -1)
    #    #frame[trajFrame] = [255,0,0]
    #    #smartEye[int(cy),int(cx),:] = [0,0,255]
    #    #fgMask[int(cyfg),int(cxfg),:] = [0,255,0]
    #    cv.circle(smart,(cxsmart,cysmart), 5, (0,0,255), -1)
    #    xssmart.append(cxsmart)
    #    yssmart.append(cysmart)
    #    tssmart.append(capture.get(cv.CAP_PROP_POS_MSEC))
    #if np.sum(fgMask)>0:
    #    cv.circle(frameAug,(cxfg,cyfg), 5, (0,255,0), -1)
    #    cv.circle(fgMask,(cxfg,cyfg), 5, (0,255,0), -1)
    #    xsfg.append(cxfg)
    #    ysfg.append(cyfg)
    #    tsfg.append(capture.get(cv.CAP_PROP_POS_MSEC))
    
    cv.rectangle(frameAug, (10, 2), (100,20), (255,255,255), -1)
    cv.putText(frameAug, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
        
    
    cv.imshow('Frame', frameAug)
    #cv.imshow('FG Mask', fgMask)
    cv.imshow('th', th)
    #cv.imshow('thEye', thEye)
    #cv.imshow('th and fg', smart)
    
    if (int(capture.get(cv.CAP_PROP_POS_FRAMES))%1000)==1  :
        print('frame',frame.shape,frame.dtype)
        cv.imwrite(directory+'/frame'+str(capture.get(cv.CAP_PROP_POS_FRAMES))+'.png', frame)
        data0= frame[::4,::4,0].flatten()
        data1= frame[::4,::4,1].flatten()
        data2= frame[::4,::4,2].flatten()
        hsv_data = cv.cvtColor(frame[::4,::4], cv.COLOR_BGR2HSV)
        #data=frame[mask]
        f,axs=plt.subplots(3,3)
        axs[0,0].scatter(data0,data1,s=1)
        axs[0,1].scatter(data0,data2,s=1)
        axs[0,2].scatter(data1,data2,s=1)
        axs[1,0].hist(data0,bins=100)
        axs[1,1].hist(data1,bins=100)
        axs[1,2].hist(data2,bins=100)
        #HSV
        hsv_data0= hsv_data[:,:,0].flatten()
        hsv_data1= hsv_data[:,:,1].flatten()
        hsv_data2= hsv_data[:,:,2].flatten()
        axs[2,0].scatter(hsv_data0,hsv_data1,s=1)
        axs[2,0].axvline(x=th0,c='r')
        axs[2,0].axhline(y=th1,c='g')
        axs[2,1].scatter(hsv_data0,hsv_data2,s=1)
        axs[2,1].axvline(x=th0,c='r')
        axs[2,1].axhline(y=th2,c='b')
        axs[2,2].scatter(hsv_data1,hsv_data2,s=1)
        axs[2,2].axvline(x=th1,c='g')
        axs[2,2].axhline(y=th2,c='b')
        axs[2,2].plot([0,255],[255,0])
        if os.path.isfile(directory+'/mask'+str(capture.get(cv.CAP_PROP_POS_FRAMES))+'.png'):
            try:
                mask = (cv.imread(directory+'/mask'+str(capture.get(cv.CAP_PROP_POS_FRAMES))+'.png')[::4,::4,0]>0)
                print('mask',mask.shape,mask.dtype)
                dataMasked0= frame[::4,::4,0][mask]
                dataMasked1= frame[::4,::4,1][mask]
                dataMasked2= frame[::4,::4,2][mask]
                axs[0,0].scatter(dataMasked0,dataMasked1,s=1)
                axs[1,0].hist(dataMasked0,bins=100)
                axs[0,1].scatter(dataMasked0,dataMasked2,s=1)
                axs[1,1].hist(dataMasked1,bins=100)
                axs[1,2].hist(dataMasked2,bins=100)
                hsv_dataMasked0= hsv_data[:,:,0][mask].flatten()
                hsv_dataMasked1= hsv_data[:,:,1][mask].flatten()
                hsv_dataMasked2= hsv_data[:,:,2][mask].flatten()
                axs[2,0].scatter(hsv_dataMasked0,hsv_dataMasked1,s=1)
                axs[2,1].scatter(hsv_dataMasked0,hsv_dataMasked2,s=1)
                axs[2,2].scatter(hsv_dataMasked1,hsv_dataMasked2,s=1)
            except:
                print("mask problem")
        f.savefig(directory+'/data'+str(capture.get(cv.CAP_PROP_POS_FRAMES))+'.png')   # save the figure to file
        plt.close(f)    # close the figure window
    
    
    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break


f,ax=plt.subplots(1,1)
xs = np.array(xs)
ys = -np.array(ys)
ts = np.array(ts)
ax.plot(xs,ys)

print(ts)
#f1,ax=plt.subplots(1,1)
#xsfg = np.array(xsfg)
#ysfg = -np.array(ysfg)
#ax.plot(xsfg,ysfg)

f2 = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(xsfg, ysfg, tsfg, 'b')
ax.plot3D(xs,ys, ts, 'r')




df = pd.DataFrame(list(zip(ts, xs,ys)), columns = ['times (msec)', 'x (px)', 'y (px)'])
df.to_csv(directory+'/trajTh.csv', index=False)
#df = pd.DataFrame(list(zip(tssmart, xssmart,yssmart)), columns = ['times (msec)', 'x (px)', 'y (px)'])
#df.to_csv(directory+'/trajsmart.csv', index=False)
#df = pd.DataFrame(list(zip(tsfg, xsfg,ysfg)), columns = ['times (msec)', 'x (px)', 'y (px)'])
#df.to_csv(directory+'/traj'+args.algo+'.csv', index=False)

#smoothing

#w=40
#xsfgC=np.convolve(xsfg, np.ones(w)/w, mode='valid')
#ysfgC=np.convolve(ysfg, np.ones(w)/w, mode='valid')
#ax.plot(xsfgC,ysfgC)
#ax.set_aspect('equal')
#margin = (len(xsfg)-len(xsfgC))/2
#tsfgc=tsfg[margin:-margin]
#df = pd.DataFrame(list(zip(xsfgC, ysfgC,tsfgc)), columns = ['times (msec)', 'x (px)', 'y (px)'])
#df.to_csv('traj'+args.algo+'smooth.csv', index=False)
#f3 = plt.figure()
#ax = plt.axes(projection='3d')
#ax.plot3D(xsfgC, ysfgC, tsfgc, 'b')


f.savefig(directory+'/traj.png')
#f1.savefig(directory+'/trajFG.png')
f2.savefig(directory+'/traj3d.png')
#f3.savefig(directory'/trajFG3dsmooth.png')plt.show()
plt.show()
