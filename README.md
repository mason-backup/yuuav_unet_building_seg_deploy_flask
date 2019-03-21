### README
![Image text](https://raw.githubusercontent.com/MeisonP/yuuav_building_seg_deploy_flask/master/Screenshot%202019-01-08%20at%205.43.29%20PM.png)

1.
maek sure the version of Tensorflow are >= 1.11.0

2.
the name test images should not contain "_color"，
and the suffix ".png"or ".jpg" must contain in the file name  
And the numbers of images must be even number, can't not be odd

3.
the python scripts are modified specifically from U-Net-tensorflow precject,
and it is quite different to the original one, just for this demo use.

4.
How to run the demo:
	put your test images into folder "images" # without numbers limitation
	run the shell script #sh demo.sh
	the results will be stored in folder "results"

5. 
the predict method is restoring from checkpoint.
and must contains three file: **.meta, **.index, ***.data
