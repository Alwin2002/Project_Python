OCR is one of the most important and popular problems that computer vision has tried to solve for a long period.

Several of them are classical computer vision approach while others are deep learning.
We divided the whole process into 3 parts 
First is Page Segmentation
Second is Line Segmentation
Third is Word Recognition
As the project is divided into 3 parts we have created three training models for each stage

Page Segmentation
Here the text is considered as one class and the background as the other class. The area within the image which contains text has to be classified as 1 while the background area has 0

Link to the dataset https://drive.google.com/drive/mobile/folders/1yNAXvtesLjl0tejAacHTGOD5sdtQ4Q_t/1gB2Ex5hDiuOSNIy13tnlDNHs_xJkWvHV/1Seh5GYm7ASeSGP3n70BULHrI0P9wvJfW?sort=13&direction=a


This dataset contains few images and it's mask 
A mask which is for the entire line as shown above

I have used the CNN UNet for doing image segmentation. I decided to use UNet because of its structure. In UNet, the image is first downsampled and then upsampled to its original size


The contour indexes can be used to get the bounding boxes coordinates for the lines which would help to crop the text regions from the page.


Further these cropped images is passed on to line Segmentation model to further process

Line Segmentation

We have used same techniques as given above
and the same CNN Unet Architecture to further segment the line images into words
Before feeding these image lines to model they are padded to 512 *512 size because this accepts only image of this size . Furthermore we have added few functions which can reduce the noise of these images 

Link to the dataset
https://drive.google.com/drive/mobile/folders/1GU310cvc3xg_dTaTayTpKg8g3CiKXsLC?sort=13&direction=a
These images are further cropped into words and passed onto CRNN model for word recognition


 



Word recognition
Here we will be training a CRNN model which takes a 128x32 image as input. Since the aim of this experiment was just to figure out the OCR training pipeline and the recognition was to be performed on clear PDF images
We are using a string module to get the list of characters. We have defined two utility functions find_dominant_color and preprocess_img. The find_dominant_color function is used to find the background color in the image. Since in any image the number of pixels of background will always be more than the text color pixels hence the pixels having the greatest occurrence in the image are the pixels related to the background and hence the color of these pixels is the background color. We have created this function so that we can pad our input image to 128x32 and avoid resizing as resizing have effects on accuracy. The preprocess_img function performs the actual resizing operation. The encode_to_labels function converts the text to indexes as neural networks processes numbers and not text.

At last we are printing the recognized text as list of strings to the console
