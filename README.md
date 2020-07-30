

# Image Pixelization with Differential Privacy



Implementation of the  Differentially Private Pixelization (*Pix*) method from  from the [Paper](https://link.springer.com/chapter/10.1007/978-3-319-95729-6_10).

## Files

📦dp-pix  
 ┣ 📂notebooks   
 ┃ ┣ 📜Demo.ipynb ................................ # demo results on Faces (AT&T), miniImageNet and Omniglot dataset  
 ┃ ┣ 📜Pixelate Performance.ipynb ....... # comparing speed of pixelation of methods in pixelate.py  
 ┃ ┣ 📜Ways to Pixelate.ipynb .............. # demo of methods in resize.py & pixelate.py  
 ┃ ┗ 📜Non-Private Pixelation.ipynb ....... # demo of np_pixel.py i.e, image pixelation with gaussian noise
 ┣ 📂src  
 ┃ ┣ 📜dataset.py .............. # dataset classes for loading images    
 ┃ ┣ 📜dp_pixel.py ............ # source code for image pixelization with differential privacy  
 ┃ ┣ 📜image_util.py ......... # helper methods  
 ┃ ┣ 📜noise.py ................. # methods for adding laplace or gaussian noise  
 ┃ ┣ 📜np_pixel.py ............ # source code for image pixelization with gaussian noise
 ┃ ┣ 📜pixelate.py ............. # pixelation implemented using skimage, PyTorch and manually   
 ┃ ┣ 📜resize.py ................ # pad and crop methods  
 ┃ ┣ 📜scale.py ................ # methods for down sampling an image to a given size  
 ┃ ┗ 📜timer.py ..................  # to measure performance  
 ┗ 📜README.md



## Datasets

Datasets are available at the following links:

- [Faces (AT&T)](https://git-disl.github.io/GTDLBench/datasets/att_face_dataset/)
- [Omniglot](https://github.com/brendenlake/omniglot/tree/master/python)
- [miniImageNet](https://github.com/yaoyao-liu/mini-imagenet-tools) or [Direct Download](https://drive.google.com/uc?id=0B3Irx3uQNoBMQ1FlNXJsZUdYWEE)



## Cite

    @inproceedings{10.1007/978-3-319-95729-6_10,
        title        = {Image Pixelization with Differential Privacy},
        author       = {Fan, Liyue},
        year         = 2018,
        booktitle    = {Data and Applications Security and Privacy XXXII},
        publisher    = {Springer International Publishing},
        address      = {Cham},
        pages        = {148--162},
        isbn         = {978-3-319-95729-6},
        editor       = {Kerschbaum, Florian and Paraboschi, Stefano},
    }

