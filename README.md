# SEM-Image-Analysis-Tkinter
This repo is a Tkinter software implementation of SEM image analysis for Carbon Nanotubes.

* Run `sem.py` to start the software. It will open up the Main Menu:
![Main Menu](https://github.com/kostispar/SEM-Image-Analysis-Tkinter/blob/main/data/image.PNG)

* Choose `Estimate CNT diameter` to run a custom algorithm for detecting CNTs and estimating statistics on the diameter of the CNTs in the selected image.
  You will first have to select the scale bar of the Image through the UI. Finally, you will be asked to also select the lowest diameter you want to detect. This is optional, but   helps for outlier removal.
* First, the algorithm should detect edges and diameters. The detection part is quick and you'll be able to see the progress:<br/>
![progress bar](https://github.com/kostispar/SEM-Image-Analysis-Tkinter/blob/main/data/image2.PNG)
<br/>

* Ploting the detection can take some time:<br/>
![ploting](https://github.com/kostispar/SEM-Image-Analysis-Tkinter/blob/main/data/image3.PNG)

* The final statistical report gives you information about average value, distribution of diameters etc. It can be used for reporting the quality of your CNTs:<br/>
![Statistical Report](https://github.com/kostispar/SEM-Image-Analysis-Tkinter/blob/main/data/image4.png)

* Choose `Reconstruct Topology` to see a 3D reconstruction of your SEM image based on pixel brightness:<br/>
![Topology](https://github.com/kostispar/SEM-Image-Analysis-Tkinter/blob/main/data/image5.PNG)

* Choose `Estimate Density` to play with different image processing thresholds and compute respective density of CNTs in your image:<br/>
![Density](https://github.com/kostispar/SEM-Image-Analysis-Tkinter/blob/main/data/image6.PNG)

* Choose `Estimate CNT Depth Distribution` to see a statistical report on the distribution of pixel brightness values:<br/>
![Density](https://github.com/kostispar/SEM-Image-Analysis-Tkinter/blob/main/data/image7.png)
<br/>

* `Run Full Single Analyis` will go through all above steps for one image<br/>
* `Run Full Batch Analysis` will go through all above steps for a directory of images and will average results<br/>
