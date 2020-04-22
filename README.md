# Yolo-object-detection



For predictiong and loccalize multiple object in a single image, There are many algorithm exist like R-CNN, Fast R-CNN, Faster R-CNN and YOLO. 


Yolo is real time object detection algorithm which is faster than any other algorithm like R-CNN, Fast R-CNN, Faster R-CNN. 

R-CNN at first select the region of object from image and predict by CNN and localize each region.Apply region proposal algorithm to select region. sometimes detect noisy region.
Fast R-CNN apply Cnn to the full image and select the region of object from feature map.
Faster R-CNN use region proposal network before fully connected layer and discard non object. 
Yolo divide the full image into grid to avoid sliding windows. Each grid gives vector which is input of cnn along with the image. use non max suppression to keep only one bounding box for each object according to maximum probability. Use anchor boxes to allow multiple box for one grid cell. Intersection over union(IOU) measure the accuracy of localization.
