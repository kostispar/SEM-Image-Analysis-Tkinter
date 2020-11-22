from tkinter import *
import webbrowser
from tkinter import filedialog
from PIL import ImageTk, Image
import matplotlib.pyplot as plt
import cv2
import time
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import tkinter.font as font
import skimage
import skimage.feature
from skimage.viewer import ImageViewer
from matplotlib import cm
import math
from matplotlib.ticker import FuncFormatter
from functools import partial

def click_estimate_density():
    newWindow = Toplevel(root)
    newWindow.geometry('600x600+800+100')
    newWindow.iconbitmap("e:\cnts_sem\icon3.ico")
    newWindow.attributes("-topmost", True)

    class GUI(Frame):

        def __init__(self, master=newWindow):
            Frame.__init__(self, master)
            w, h = root.winfo_screenwidth(), root.winfo_screenheight()
            newWindow.geometry("%dx%d+0+0" % (w, h))
            newWindow.minsize(width=w, height=h)
            newWindow.maxsize(width=w, height=h)
            self.pack()

            self.file = Button(self, text='Browse', command=self.choose)

            self.image_original = PhotoImage()

            self.label_original = Label(self,image=self.image_original)
            self.image = PhotoImage()
            self.label = Label(self,image=self.image)
            self.scale = Button(self, text="Adjust", command=self.adjust)
            self.surface_label = Label(self, text="Total Surface of CNTs: ",  cursor ="dot", font = 12,fg = "black", relief = RAISED)
            self.horizontal = Scale(self, from_=0, to=100, orient=HORIZONTAL)
            self.l4 = Label(self, text="Threshold")
            self.vertical = Scale(self, from_=0, to=1000, orient=HORIZONTAL)
            self.l5 = Label(self, text="High Threshold")
            self.sigma_ = Scale(self, from_=0, to=100, orient=HORIZONTAL)
            self.l6 = Label(self, text="Sigma")
            self.surface_label.pack(side= "bottom" )
            self.path =""
            self.file.pack()
            self.scale.pack()
            self.label.pack(side="right",expand = True)
            self.label_original.pack(side="left",expand = True)
            self.horizontal.pack()
            self.l4.pack()
            self.vertical.pack()
            self.l5.pack()
            self.sigma_.pack()
            self.l6.pack()


        def choose(self):
            ifile = filedialog.askopenfilename(parent = newWindow, initialdir="/", title="Select a file",
                                                        filetypes=(("png files", "*.png"), ("all files", "*.*")))
            path = ifile
            self.path = path
            self.image2 = PhotoImage(file=path)
            self.label.configure(image=self.image2)
            self.label.image=self.image2
            self.label_original.configure(image=self.image2)
            self.label_original.image = self.image2

        def adjust(self):
            value1 =self.horizontal.get()
            value2 = self.vertical.get()
            value3 = self.sigma_.get()
            im = cv2.imread(self.path, 0)
            blur = skimage.filters.gaussian(im, sigma=value3/10)
            mask = blur < value1/100
            a =mask.astype(int)
            a[a == 1] = 255
            #volume estimation
            unique, counts = np.unique(a, return_counts=True)
            dictionary = (dict(zip(unique, counts)))
            whites = dictionary.get(255)
            blacks = dictionary.get(0)
            surface_per = (blacks/(blacks+whites))
            #####
            my_image  = Image.fromarray(np.uint8(a))
            self.image2 = ImageTk.PhotoImage(my_image)
            self.label.configure(image=self.image2)
            self.label.image = self.image2
            self.surface_label.config(text="Total Surface of CNTs: " + str(round(surface_per*100,2)) + " %" )

        def adjust1(self):
            value1 = self.horizontal.get()
            value2 = self.vertical.get()
            value3 = self.sigma_.get()
            im = cv2.imread(self.path, 0)
            edges = skimage.feature.canny(
                image=im,
                sigma=float(value3),
                low_threshold=value1 / 10,
                high_threshold=value2 / 10,
            )

            a = ((edges.astype(int)))
            print("------------------------------")
            a[a == 1] = 255
            my_image = Image.fromarray(np.uint8(a))
            self.image2 = ImageTk.PhotoImage(my_image)
            self.label.configure(image=self.image2)
            self.label.image = self.image2

    app = GUI()
    app.mainloop()
    root.destroy()

def click_estimate_density2():
    global my_image
    newWindow = Toplevel(root)
    newWindow.geometry('600x600+800+100')
    newWindow.iconbitmap("e:\cnts_sem\icon3.ico")
    newWindow.attributes("-topmost", True)
    selector = Label(newWindow, text="Original Image").pack()

    newWindow.filename = filedialog.askopenfilename(initialdir="/", title="Select a file",
                                                    filetypes=(("png files", "*.png"), ("all files", "*.*")))



    def update():
        sel = "Horizontal Scale Value = " + str(v1.get())
        l1.config(text = sel, font =("Courier", 14))

        edges = cv2.Canny(image,v1.get(),200)

        im = Image.fromarray(np.uint8(cm.gist_earth(edges)*255))
        my_image = ImageTk.PhotoImage(im)
        newWindow = Toplevel(root)
        newWindow.geometry('600x600+800+100')
        newWindow.iconbitmap("..\cnts_sem\icon3.ico")
        newWindow.attributes("-topmost", True)
        selector = Label(newWindow, text="XXXX").pack()
        my_image_label = Label(newWindow,image=my_image).pack()
        newWindow.pack()
    a = 100
    v1 = DoubleVar()
    image = skimage.io.imread(fname=newWindow.filename, as_gray=False)
    edges = cv2.Canny(image, a,200)
    im = Image.fromarray(np.uint8(cm.gist_earth(edges)*255))
    my_image = ImageTk.PhotoImage(im)
    my_image_label = Label(newWindow,image=my_image).pack()

    s1 = Scale(newWindow, variable = v1,
           from_ = 1, to = 100,
           orient = HORIZONTAL)
    l3 = Label(newWindow, text = "Horizontal Scaler")

    b1 = Button(newWindow, text ="Adjust",
            command = update,
            bg = "yellow")

    s1.pack(anchor = CENTER)
    l3.pack()
    b1.pack(anchor = CENTER)
    l1 = Label(newWindow)
    l1.pack()


def CountFrequency(my_list, my_list_pre, org_path="", pre_path = "",nn = False):
    # Creating an empty dictionary
    freq = {}
    for item in my_list:
        if (item in freq):
            freq[item] += 1
        else:
            freq[item] = 1

    #mean, variance
    mean = np.mean(my_list)
    var = np.var(my_list)
    median = np.median(my_list)
    #mode = np.mod(my_list)
    std = np.std(my_list)
    #percentiles dictionary

    lst_cnt = lst_all = lst_1st_left = lst_1st_right = lst_2std_left = lst_2std_right = 0
    for key, value in freq.items():
        #print("% d : % d" % (key, value))
        lst_all += value
        if key > 40:
            lst_cnt += value
        if mean-std <= key <= mean:
            lst_1st_left += value
        if mean <= key <= mean+std:
            lst_1st_right += value
        if mean-2*std <= key <= mean-std:
            lst_2std_left += value
        if mean+std <= key <= mean+2*std:
            lst_2std_right += value
    lst_meanstd_left = lst_1st_left/lst_all
    lst_meanstd_right = lst_1st_right / lst_all
    lst_mean2std_left = lst_2std_left / lst_all
    lst_mean2std_right = lst_2std_right / lst_all
    cnt_pixel_vol = lst_cnt / lst_all
    pos = np.arange(len(freq.keys()))
    width = 1.0  # gives histogram aspect to the bar diagram
    if nn == False:
        plt.figure(figsize=(30, 30))
        plt.title("Distribution of CNTs per layer")
        plt.xlabel("Height (grayscale value)")
        plt.ylabel("Count")
        plt.bar(freq.keys(), freq.values(), width, color="g")
        x = [40]
        y = range(0,1000)
        plt.plot(x*len(y), y,'-b',label = 'hard margin')
        plt.plot([mean] * len(y), y, color = 'grey', label='mean')
        plt.fill_between([mean - 2*std, mean + 2*std], [mean, 0], [1000, 1000], linestyle="--",
                         color='k', alpha=0.25, label='mean - 2*std')
        plt.fill_between([mean - std, mean + std], [mean, 0], [1000, 1000], linestyle="--",
                         color='k', alpha=0.25, label='mean - std')
        plt.text(mean-std/2, len(y)/2,str(round(100*lst_meanstd_left,2)) + ' %')
        plt.text(mean + std / 2, len(y) / 2, str(round(100*lst_meanstd_right,2)) + ' %')
        plt.text(mean - 3*std / 2  , len(y) / 2, str(round(100 * lst_mean2std_left, 2)) + ' %')
        plt.text(mean + 3*std / 2, len(y) / 2, str(round(100 * lst_mean2std_right, 2)) + ' %')
        plt.legend()
        plt.show()
    if nn == True:
        x = np.arange(2)
        y = np.array([lst_cnt,lst_all-lst_cnt])
        plt.subplot(221)
        plt.xticks(np.arange(1,3, 1))
        plt.bar(x,y)
        plt.xticks(x, ('CNTs', 'Black'))
        plt.text(-0.2, lst_cnt - 10000,weight='bold',s='Pixel Density \n of CNTs: ' + str(round(lst_cnt/lst_all,2)*100) + ' %')
        plt.text(0.8, lst_all - lst_cnt - 10000,weight='bold', s='Pixel Density \n of black: ' + str(round((lst_all - lst_cnt) / lst_all, 2)*100) + ' %')
        plt.ylabel('No of Pixels')
        plt.title("Image Analysis")

        plt.subplot(222)
        freq_pre = {}
        for item in my_list_pre:
            if (item in freq_pre):
                freq_pre[item] += 1
            else:
                freq_pre[item] = 1
        lst_all_pre = 0
        lst_cnt_pre = 0
        for key, value in freq_pre.items():
            # print("% d : % d" % (key, value))
            lst_all_pre += value
            if key == 29:
                lst_cnt_pre += value
        x_pre = np.arange(2)
        y_pre = np.array([lst_cnt_pre, lst_all_pre - lst_cnt_pre])

        plt.xticks(np.arange(1, 3, 1))
        plt.bar(x_pre, y_pre)
        plt.xticks(x_pre, ('CNTs', 'Black'))
        plt.text(-0.2, lst_cnt_pre - 10000, weight='bold',
                 s='Predicted Density \n of CNTs: ' + str(round(lst_cnt_pre / lst_all_pre, 2) * 100) + ' %')
        plt.text(0.8, lst_all_pre - lst_cnt_pre - 10000, weight='bold',
                 s='Predicted Density \n of black: ' + str(round((lst_all_pre - lst_cnt_pre) / lst_all_pre, 2) * 100) + ' %')
        plt.ylabel('No of Pixels')
        plt.title("Deep Learning")
        plt.subplot(223)
        img_pre = cv2.imread(pre_path,  cv2.IMREAD_GRAYSCALE)
        img_org = cv2.imread(org_path,  cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_org)
        plt.title("Original SEM image")
        plt.subplot(224)
        plt.imshow(img_pre)
        plt.title("Predictions")
        plt.show()


def open_file_browser(window):
    filepath = filedialog.askopenfilename(initialdir="/", title="Select a file",
                               filetypes=(("png files", "*.png"), ("all files", "*.*")))



def visitwebpage():
    webbrowser.open("http://innovation-res.eu/about/")

def click_recon():
    global my_image
    newWindow = Toplevel(root)
    newWindow.geometry('600x600+800+100')
    newWindow.iconbitmap("e:\cnts_sem\icon3.ico")
    newWindow.attributes("-topmost", True)
    selector = Label(newWindow, text = "Select a file").pack()
    newWindow.filename = filedialog.askopenfilename(initialdir="/", title="Select a file",
                                          filetypes=(("png files", "*.png"), ("all files", "*.*")))
    my_image = ImageTk.PhotoImage(Image.open(newWindow.filename))
    my_image_label = Label(newWindow,image=my_image).pack()
    im = cv2.imread(newWindow.filename,0)
    x = np.linspace(0, len(im[0]), len(im[0]))
    y = np.linspace(0, len(im), len(im))

    X, Y = np.meshgrid(x, y)
    Z = im

    fig = plt.figure()

    ax = plt.axes(projection='3d')
    plt.title("3D reconstruction based on brightness")
    ax.contour3D(Y, X, Z, 50, cmap='binary')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()


def click_estimate_frequency():
    global my_image
    newWindow = Toplevel(root)
    newWindow.geometry('600x600+800+100')
    newWindow.iconbitmap("e:\cnts_sem\icon3.ico")
    newWindow.attributes("-topmost", True)
    selector = Label(newWindow, text = "Select a file").pack()
    newWindow.filename = filedialog.askopenfilename(initialdir="/", title="Select a file",
                                          filetypes=(("png files", "*.png"), ("all files", "*.*")))
    my_image = ImageTk.PhotoImage(Image.open(newWindow.filename))
    my_image_label = Label(newWindow,image=my_image).pack()
    im = cv2.imread(newWindow.filename,0)
    x = np.linspace(0, len(im[0]), len(im[0]))
    y = np.linspace(0, len(im), len(im))

    X, Y = np.meshgrid(x, y)
    Z = im
    a = Z.tolist()
    b = []
    for list in a:
        for element in list:
            b.append(element)

    CountFrequency(b,[],org_path="",pre_path="", nn=False)

def metadata_removal(im):
    '''
    removes the bottom part with metadata
    :return: new numpy array of image
    '''
    row_to_delete = 0
    for j in range(len(im)):
        if np.all(im[j] == 0):
            row_to_delete = j
            break
    im = np.delete(im, range(row_to_delete,len(im)),axis = 0)
    return(im)


def eight_bit(im):
    '''
    convert to 8 bit
    :param im:
    :return:
    '''
    image = im.astype(np.uint8)
    return(image)


def grayscale(im):
    '''
    convert to grayscale
    :param im:
    :return:
    '''
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return gray


def histogram_equalization(im):
    '''
    histogram equalization of image
    :param im:
    :return:
    '''
    im = cv2.equalizeHist(im)
    return (im)

def median_filtering(im):
    '''
    median filtering of image
    :param im:
    :return:
    '''
    median = cv2.medianBlur(im, 5)
    return (median)

def otsu_method(im):
    '''
    apply otsu method for binarization
    :param im:
    :return:
    '''
    otsu = cv2.threshold(im,0,255,cv2.THRESH_OTSU)
    otsu_np = otsu[1]
    return (otsu_np)


def add_border(im):
    '''
    add a black frame around the image
    :param im:
    :return:
    '''
    border = cv2.copyMakeBorder(im, 10, 10, 10, 10, cv2.BORDER_CONSTANT)
    return(border)

def reversing(im):
    '''
    reverse black and white pixels
    :param im:
    :return:
    '''
    im = cv2.bitwise_not(im)
    return (im)

def opening_operation(im,kernel):
    '''
    opening operation
    :param im:
    :return:
    '''
    opening = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel, iterations=3)
    return(opening)

def closing_operation(im,kernel):
    '''
    closing operation
    :param im:
    :param kernel:
    :return:
    '''
    im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel, iterations=3)
    return(im)

def plot(w,h,columns,rows,images,titles):
    fig = plt.figure(figsize=(8, 8))
    titles = ['original', 'grayscale', 'histogram equalized', 'median filtered', 'otsu method', 'reverse and border',
              'opening operation', 'closing operation']

    for i in range(1, columns * rows):
        fig.add_subplot(rows, columns, i)
        plt.title(titles[i - 1])
        plt.imshow(images[i - 1], cmap='gray')
        plt.xticks([])
        plt.yticks([])
    plt.show()

def Average(lst):
    return sum(lst) / len(lst)



def click_estimate_diameter():
    newWindow = Toplevel(root)
    newWindow.geometry('600x600+800+100')
    newWindow.iconbitmap("e:\cnts_sem\icon3.ico")
    newWindow.attributes("-topmost", True)
    ifile = filedialog.askopenfilename(initialdir="/", title="Select a file",
                                       filetypes=(("png files", "*.png"), ("all files", "*.*")))
    path = ifile
    im = cv2.imread(path)
    #print(len(im), len(im[0]))
    boxes1 = []
    def on_mouse_0(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            #print('Start Mouse Position: ' + str(x) + ', ' + str(y))
            sbox = [x, y]
            boxes1.append(sbox)
            cv2.circle(im, (x, y), radius=0, color=(0, 0, 255), thickness=5)
            #print(boxes)
    cv2.startWindowThread()
    cv2.namedWindow('Select scale', cv2.WND_PROP_FULLSCREEN)

    cv2.setMouseCallback('Select scale', on_mouse_0, 0)
    while True:
        cv2.imshow('Select scale', im)
        if cv2.waitKey(10) == 27:
            break
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    scale = math.sqrt( ((boxes1[0][0]-boxes1[1][0])**2)+((boxes1[0][1]-boxes1[1][1])**2) )
    print(scale)
    result = 0

    boxes = []
    def on_mouse(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            #print('Start Mouse Position: ' + str(x) + ', ' + str(y))
            sbox = [x, y]
            boxes.append(sbox)
            cv2.circle(im, (x, y), radius=0, color=(0, 0, 255), thickness=5)
            #print(boxes)
    cv2.startWindowThread()
    cv2.namedWindow('Select minimum diameter')

    cv2.setMouseCallback('Select minimum diameter', on_mouse, 0)
    while True:
        cv2.imshow('Select minimum diameter', im)
        if cv2.waitKey(10) == 27:
            break
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    min_distance = math.sqrt( ((boxes[0][0]-boxes[1][0])**2)+((boxes[0][1]-boxes[1][1])**2) )
    print(min_distance)

    selector_s = Label(newWindow, text="Deleting irrelevant parts...", font='12')
    selector_s.pack()
    newWindow.update()
    im = metadata_removal(im)
    im = eight_bit(im)
    gray = grayscale(im)
    equ = histogram_equalization(gray)
    median = median_filtering(equ)
    otsu = otsu_method(median)
    border = add_border(otsu)
    reverse = reversing(border)
    opening = opening_operation(reverse, np.ones((6, 5), np.uint8))
    im = closing_operation(opening, np.ones((6, 5), np.uint8))

    # list of all images
    t0 = time.time()
    selector = Label(newWindow, text="> Performing detections...",font = '12')
    selector.pack()
    newWindow.update()
    print('Starting estimations...')
    total_x_0 = np.empty((len(im) ,len(im[0])))
    total_x_225 = np.empty((len(im) ,len(im[0])))
    total_x_45 = np.empty((len(im) ,len(im[0])))
    total_y_0 = np.empty((len(im) ,len(im[0])))
    total_y_225 = np.empty((len(im), len(im[0])))
    total_y_45 = np.empty((len(im), len(im[0])))
    indexes_0, indexes_225, indexes_45, indexes, indexes_0_y,  indexes_45_y,  indexes_225_y, indexes_y = [],[],[],[],[],[],[],[]
    total = np.zeros((len(im) ,len(im[0])))
    total_y = np.zeros((len(im) ,len(im[0])))
    lst_all_diameters, lst_all_diameters_x, lst_all_diameters_y = [],[],[]
    for y in range(len(im)):
            for x in range(len(im[y])):
                if im[y][x]==0:
                    # X-SCAN#
                    if im[y][x -1]==255:
                        # x0 scan
                        start_0_x = [x,y]
                        end_0_x = [x,y]
                        count_0_x = 1
                        for k in range(x + 1 ,len(im[y])):
                            if im[y][k] == 0:
                                end_0_x = [k,y]
                                count_0_x += 1
                            else:
                                total_x_0[y][x] = count_0_x * 500 / scale
                                indexes_0.append([start_0_x,end_0_x])
                                break
                        # x225 scan
                        count_225_x = 1
                        start_225_x = [x,y]
                        end_225_x = [x,y]
                        for k in range(x + 1 ,len(im[y])):
                            if im[y+count_225_x][k] == 0:
                                end_225_x = [k, y+count_225_x]
                                count_225_x += 1
                            else:
                                total_x_225[y][x] = count_225_x * 1.4141 * 500 / scale
                                indexes_225.append([start_225_x, end_225_x])
                                break
                        # x45 scan
                        count_45_x = 1
                        start_45_x = [x, y]
                        end_45_x = [x, y]
                        for k in range(x + 1, len(im[y])):
                            if im[y - count_45_x][k] == 0:
                                end_45_x = [k, y - count_45_x]
                                count_45_x += 1
                            else:
                                total_x_45[y][x] = count_45_x * 1.4141 * 500 / scale
                                indexes_45.append([start_45_x, end_45_x])
                                break

                        # find minimum on X
                        if ( total_x_225[y][x] <= total_x_0[y][x] ) and ( total_x_225[y][x] <= total_x_45[y][x] ) and ( total_x_225[y][x] > min_distance ):
                            total[y][x] = total_x_225[y][x]
                            indexes.append([start_225_x, end_225_x])
                            lst_all_diameters_x.append(total_x_225[y][x])
                            lst_all_diameters.append(total_x_225[y][x])
                        elif ( total_x_0[y][x] <= total_x_225[y][x] ) and ( total_x_0[y][x] <= total_x_45[y][x] ) and ( total_x_0[y][x] > min_distance ):
                            indexes.append([start_0_x, end_0_x])
                            total[y][x] = total_x_0[y][x]
                            lst_all_diameters_x.append(total_x_0[y][x])
                            lst_all_diameters.append(total_x_0[y][x])
                        elif (total_x_45[y][x] <= total_x_225[y][x]) and (total_x_45[y][x] <= total_x_0[y][x]) and ( total_x_45[y][x] > min_distance):
                            indexes.append([start_45_x, end_45_x])
                            total[y][x] = total_x_45[y][x]
                            lst_all_diameters_x.append(total_x_45[y][x])
                            lst_all_diameters.append(total_x_45[y][x])
                    #Y-SCAN#
                    if im[y-1][x] == 255:
                        # y0 scan
                        start_0_y = [x,y]
                        end_0_y = [x,y]
                        count_0_y = 1
                        for k in range(y + 1 ,len(im)):
                            if im[k][x] == 0:
                                end_0_y = [x,k]
                                count_0_y += 1
                            else:
                                total_y_0[y][x] = count_0_y * 500 / scale
                                indexes_0_y.append([start_0_y,end_0_y])
                                break
                        # y225 scan
                        count_225_y = 1
                        start_225_y = [x, y]
                        end_225_y = [x, y]
                        for k in range(y + 1,len(im)):
                            if im[k][x - count_225_y] == 0:
                                end_225_y = [x - count_225_y, k]
                                count_225_y += 1
                            else:
                                total_y_225[y][x] = count_225_y * 1.4141 * 500 / scale
                                indexes_225_y.append([start_225_y, end_225_y])
                                break
                        # y45 scan
                        count_45_y = 1
                        start_45_y = [x, y]
                        end_45_y = [x, y]
                        for k in range(y + 1,len(im)):
                            if im[k][x + count_45_y] == 0:
                                end_45_y = [x + count_45_y, k]
                                count_45_y += 1
                            else:
                                total_y_45[y][x] = count_45_y * 1.4141 * 500 / scale
                                indexes_45_y.append([start_45_y, end_45_y])
                                break
                        # find minimum on Y
                        if ( total_y_225[y][x] <= total_y_0[y][x] ) and ( total_y_225[y][x] <= total_y_45[y][x] ) and (total_y_225[y][x] > min_distance):
                            total_y[y][x] = total_y_225[y][x]
                            indexes_y.append([start_225_y, end_225_y])
                            lst_all_diameters_y.append(total_y_225[y][x])
                            lst_all_diameters.append(total_y_225[y][x])
                        elif ( total_y_0[y][x] <= total_y_225[y][x] ) and ( total_y_0[y][x] <= total_y_45[y][x] ) and (total_y_0[y][x] > min_distance):
                            indexes_y.append([start_0_y, end_0_y])
                            total[y][x] = total_y_0[y][x]
                            lst_all_diameters_y.append(total_y_0[y][x])
                            lst_all_diameters.append(total_y_0[y][x])
                        elif  ( total_y_45[y][x] <= total_y_225[y][x] ) and ( total_y_45[y][x] <= total_y_0[y][x] ) and (total_y_45[y][x] > min_distance):
                            indexes_y.append([start_45_y, end_45_y])
                            total_y[y][x] = total_y_45[y][x]
                            lst_all_diameters_y.append(total_y_45[y][x])
                            lst_all_diameters.append(total_y_45[y][x])
    t0_1 = time.time()
    total_det = t0_1 - t0
    selector0 = Label(newWindow, text="Done...", font='12').pack()
    newWindow.update()
    selector1 =Label(newWindow, text="Time needed for " + str(len(lst_all_diameters)) + " detections: " + str(round(total_det,2)) + " sec \n", font='12').pack()
    newWindow.update()
    selector2 = Label(newWindow, text="> Calculating diameter...",font = '12').pack()
    t1 = time.time()
    newWindow.update()
    avg = Average(lst_all_diameters)
    avg_x =  Average(lst_all_diameters_x)
    avg_y = Average(lst_all_diameters_y)
    print('The average diameter in x axis is: ', avg_x)
    print('The average diameter in y axis is: ', avg_y)
    print('The average diameter in both axes is: ', avg)
    t2 = time.time()
    total_calc = t2 - t1
    selector3 = Label(newWindow, text="Average cnt diameter is: " + str(round(avg,2)) + "nm", font='12').pack()
    newWindow.update()
    selector4 = Label(newWindow, text="Time needed for diameter calculation: " + str(round(total_calc,10)) + " sec \n", font='12').pack()
    newWindow.update()

    t3 = time.time()
    selector6 = Label(newWindow, text="> Plotting... ",                      font='12').pack()
    newWindow.update()

    print('Starting visualizations...')
    #X AXIS visualization#
    plt.imshow(im, cmap='gray')
    for item in indexes:
        x = [item[0][0],item[1][0]]
        y = [item[0][1],item[1][1]]
        plt.plot(x, y)
        plt.xticks([])
        plt.yticks([])
    plt.title("Detections with Image Processing \n Average diameter in nanometer: " + str(round(avg_x,2)))
    plt.draw()
    plt.pause(0.001)
    input("Press enter to continue...")
    plt.close()
    # Y AXIS visualization#
    plt.imshow(im, cmap='gray')
    for item in indexes_y:
        x = [item[0][0],item[1][0]]
        y = [item[0][1],item[1][1]]
        plt.plot(x, y)
        plt.xticks([])
        plt.yticks([])
    plt.title("Detections with Image Processing \n Average diameter in nanometer: " + str(round(avg_y,2)))
    plt.draw()
    plt.pause(0.001)
    input("Press enter to continue...")
    plt.close()
    plot_hist(lst_all_diameters)



def plot_hist(data):
    plt.hist(data,
            bins=25,
            rwidth=1)
    plt.plot([Average(data), Average(data)], [max(data) * 5, 0], label="average")
    percent_formatter = partial(to_percent,
                                n=len(data))
    formatter = FuncFormatter(percent_formatter)
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.xticks(range(0, 250, 20))
    plt.xlabel("nm")
    data_np = np.asarray(data)
    plt.plot([np.percentile(data_np, 25), np.percentile(data_np, 25)], [max(data) * 5, 0], label="Q25", c='black')
    plt.plot([np.percentile(data_np, 75), np.percentile(data_np, 75)], [max(data) * 5, 0], label="Q75", c='indigo')
    data_after_lower = [x for x in data if x > np.percentile(data_np, 25)]
    data_final = [x for x in data_after_lower if x < np.percentile(data_np, 75)]
    plt.title("Distribution of diameters \n Average value: " + str(
        round(Average(data), 2)) + " nm  \n Average value (after Q25/Q75 normalization): " +
    str(round(Average(data_final), 2)) )
    plt.legend()
    plt.show()

def to_percent(y, position, n):
    s = str(round(100 * y / n, 2))

    if plt.rcParams['text.usetex']:
        return s + r'$\%$'

    return s + '%'

def click_estimate_density3():
    global my_image
    newWindow = Toplevel(root)
    newWindow.geometry('600x600+800+100')
    newWindow.iconbitmap("e:\cnts_sem\icon3.ico")
    newWindow.attributes("-topmost", True)
    selector = Label(newWindow, text="Select a file").pack()
    newWindow.filename = filedialog.askopenfilename(initialdir="/", title="Select a file",
                                                    filetypes=(("png files", "*.png"), ("all files", "*.*")))
    my_image = ImageTk.PhotoImage(Image.open(newWindow.filename))
    my_image_label = Label(newWindow, image=my_image).pack()
    im = cv2.imread(newWindow.filename, 0)
    x = np.linspace(0, len(im[0]), len(im[0]))
    y = np.linspace(0, len(im), len(im))

    X, Y = np.meshgrid(x, y)
    Z = im
    a = Z.tolist()
    b = []
    for list in a:
        for element in list:
            b.append(element)

    im_pre = cv2.imread(r"C:\Users\kostis\Desktop\to_clear\Classified image1.tif", 0)
    x_pre = np.linspace(0, len(im_pre[0]), len(im_pre[0]))
    y_pre = np.linspace(0, len(im_pre), len(im_pre))
    X_pre, Y_pre = np.meshgrid(x_pre, y_pre)
    Z_pre = im_pre
    a_pre = Z_pre.tolist()
    b_pre = []
    for list in a_pre:
        for element in list:
            b_pre.append(element)
    CountFrequency(b,b_pre,org_path=newWindow.filename,pre_path=r"C:\Users\kostis\Desktop\to_clear\Classified image1.tif",nn=TRUE)


def click_estimate_volume():
    return

def click_run_full_analysis():
    return
###############################################set general front end attributes#######################################
root = Tk()

w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
root.title("SEM CNT Image Analysis")
root.iconbitmap(r"C:\Users\kostis\PycharmProjects\ProjectX\cnt_software\icon3.ico")


#######################################################################################################################

############################################################# MENU BAR ################################################
menubar = Menu(root)
filemenu = Menu(menubar, tearoff=0)
filemenu.add_command(label="Exit", command=root.quit)
menubar.add_cascade(label="Options", menu=filemenu)

helpmenu = Menu(menubar, tearoff=0)
helpmenu.add_command(label="Help Index", command=click_run_full_analysis)
helpmenu.add_command(label="About...", command=visitwebpage)
menubar.add_cascade(label="Help", menu=helpmenu)

root.config(menu=menubar)
#####################################################################################################################

myFont = font.Font(size=12 ,family='Helvetica')
myFont2 = font.Font(size=16 , weight="bold",family='Helvetica')
menuLabel = Label(root, text = 'Main Menu', font=(None, 15), padx = 750, pady = 100, fg = 'black')
menuLabel.place(height=80, width=205, x = w/2 - 130, y =0)
button_reconstruct = Button(root, text = 'Reconstruct Topology',padx = 40, pady = 20, fg = 'black', bg = 'red',command = click_recon)
button_reconstruct.place(height=80, width=250, x = w/2 - 300, y =100)
button_reconstruct['font'] = myFont
button_estimate_density = Button(root, text = 'Estimate CNT Density', padx = 42, pady = 20, fg = 'black', bg = 'red',command = click_estimate_density)
button_estimate_density.place(height=80, width=250, x = w/2 - 300, y =200)
button_estimate_density['font'] = myFont
button_estimate_volume = Button(root, text = 'Estimate CNT Volume', padx = 41, pady = 20, fg = 'black', bg = 'red',command = click_estimate_volume)
button_estimate_volume.place(height=80, width=250, x = w/2 , y =100)
button_estimate_volume['font'] = myFont
button_estimate_frequency = Button(root, text = 'Estimate CNT depth Distribution', padx = 35, pady = 20, fg = 'black', bg = 'red',command = click_estimate_frequency)
button_estimate_frequency.place(height=80, width=550, x = w/2 - 300 , y =300)
button_estimate_frequency['font'] = myFont
button_estimate_diam = Button(root, text = 'Estimate CNT diameter', padx = 38, pady = 20, fg = 'black', bg = 'red',command = click_estimate_diameter)
button_estimate_diam.place(height=80, width=250, x = w/2 , y =200)
button_estimate_diam['font'] = myFont
button_run_full = Button(root, text = 'Run full single analysis', padx = 56, pady = 30,fg = 'black', bg = 'red',command = click_run_full_analysis)
button_run_full['font'] = myFont2
button_run_full.place(height=100, width=550, x = w/2 - 300, y =450)
button_run_batch_analysis = Button(root, text = 'Run full batch analysis', padx = 56, pady = 30,fg = 'black', bg = 'red',command = click_run_full_analysis)
button_run_batch_analysis['font'] = myFont2
button_run_batch_analysis.place(height=100, width=550, x = w/2 - 300, y =550)
button_documentation = Button(root, text = 'Documentation', padx = 56, pady = 30,fg = 'black', command = click_run_full_analysis)
button_documentation['font'] = font.Font(size=12 , family='Helvetica')
button_documentation.place(height=50, width=250, x = 1200, y =650)
button_saved = Button(root, text = 'Saved files', padx = 56, pady = 30,fg = 'black', command = click_run_full_analysis)
button_saved['font'] = font.Font(size=12 , family='Helvetica')
button_saved.place(height=50, width=250, x = 1200, y =600)
button_exit = Button(root, text = 'Exit', padx = 56, pady = 30,fg = 'black', command = root.quit)
button_exit['font'] = font.Font(size=12 , family='Helvetica')
button_exit.place(height=50, width=250, x = 1200, y =700)
button_about = Button(root, text = 'About', padx = 56, pady = 30,fg = 'black', command = visitwebpage)
button_about['font'] = font.Font(size=12 , family='Helvetica')
button_about.place(height=50, width=250, x = 50, y =700)




#main loop
root.mainloop()