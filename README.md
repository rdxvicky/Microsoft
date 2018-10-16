<h1> Improving Data Quality in Medical Imaging 
HOSTED BY MICROSOFT</h1>


Improving Data Quality in Medical Imaging
According to a 2012 report, the annual cost of medical errors in the United States in 2008 alone was $19.5 billion.
Some of these errors are very hard to prevent without massive overhauls of operational processes and organizational culture. 
But some errors could be easily flagged for human review using artificial intelligence.
In this challenge, you will use standard AI tools to predict a patient's body orientation in CT scans so that discrepancies in manually entered data can be automatically identified and corrected.

<b>Background</b>
When a patient has a CT scan taken, a special device uses X-rays to take measurements from a variety of angles which are then computationally reconstructed into a 3D matrix of intensity values. Each layer of the matrix shows one very thin "slice" of the patient's body. This data is saved in an industry-standard format known as DICOM, which saves the image matrix in a set binary format and then wraps this data with a huge variety of metadata tags.

Some of these fields (e.g. hardware manufacturer, device serial number, voltage) are usually correct because they are automatically read from hardware and software settings

The problem is that many important fields must be added manually by the technician and are therefore subject to human error factors like confusion, fatigue, loss of situational awareness, and simple typos.

A doctor scrutinizing image data will usually be able to detect incorrect metadata, but in an era when more and more diagnoses are being carried out by computers it is becoming increasingly important that patient record data is as accurate as possible.

This is where AI comes in. Using your skills, we want to improve the error checking for one single but incredibly important value: a field known as Image Orientation (Patient) which indicates the 3D orientation of the patient's body in the image.


My job is to:

Train a model using the images in train and the labels train_labels.csv
Predict orientation labels for the images in test for which you don't know the true orientations.
The files are named with the conventions {id}.png. The {id} in the filename matches the id column in the training labels for the training data and in the submission format for the test data. These images have been scaled so that they are 64px by 64px.


The orientation labels you are predicting have the following meaning when the image is viewed on an upright, vertical surface like a computer screen:

0: Spine at bottom, patient facing up.
1: Spine at right, patient facing left.
2: Spine at top, patient facing down.
3: Spine at left, patient facing right.
