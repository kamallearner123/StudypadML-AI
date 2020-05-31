This repository is to track the learning phase and modify the order of learning process to help others.

Inputs:
Main reference from Vijay's github
The below are my steps to learn ML/AI:
A blog about "From software engineer to ML/AI expert": https://blog.insightdatascience.com/preparing-for-the-transition-to-applied-ai-d41e48403447
Take a free and short course from TensorFlow wiht Udacity: https://classroom.udacity.com/courses/ud187
Take another free and crash course to get more familiar with terminology along with TensorFlow: https://developers.google.com/machine-learning/crash-course/
Computer Vision: A complete course on CV from Stanford University (16 classes) https://www.youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv
NLP: A Complete cpurse on NLP from Stanford university https://www.youtube.com/playlist?list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6
Learn MAths for ML: https://www.youtube.com/playlist?list=PLmAuaUS7wSOP-iTNDivR0ANKuTUhEzMe4
GPUs: 7.1) Know why should we use GPUs in higher computations: https://research.nvidia.com/sites/default/files/pubs/2007-02_How-GPUs-Work/04085637.pdf 7.2) Know how to work with GPU using CUDA (NVIDIA GPU driver library): http://lorenabarba.com/gpuatbu/Program_files/Cruz_gpuComputing09.pdf
Outcome:
Few experiments on RaspberryPi and coral board or NVIDIA Jetson Nano Developer Kit.
Start implementing algorithms using C++ and provide Python library (using Cython)
Present papers.
[Inputs from experts]:
The below comments are collected from Chomba Bupe:

Yes the best approach to learning such complex fields is always to challenge yourself with practical projects. I was actually learning from books such as the one referenced in the details to this question plus journals from David Lowe[1] and many more from the author of that book while building my computer vision (CV) system from ground-up.

I learnt about computer vision by finishing an actual CV system. It was not easy though, it took 4 years for me to do that. I had to overcome a lot of negativity from toxic friends and my own insecurities in order to gather the courage to believe in my abilities to learn and build working novel solutions to really hard problems. I am glad I gave myself the chance, now I know a great deal about CV and ML thanks to Google search, Stackoverflow and many top universities providing many sources to read from freely online.

The start was actually hard but I started small and gradually increased the challenges. I started implementing image processing techniques like convolutional operations for filters like Gaussian, Binomial and box filters. I also spent some time reading about neuroscience especially the so called brodmann area 17 or primary visual cortex in primates. I named the project, “project area 17” after brodmann area 17.

I learned about retinal ganglion, simple, complex and hypercomplex cells. There is also the so called two-streams hypothesis of the visual cortex which comprises of the dorsal and ventral streams, I read about all that as well. I spent a great deal of time trying to make sense of all of that.

It took me 2 years to reach a point where I understood the stuff. A picture started forming in my head about how I could build my own system. At that point I had already gone through a lot of literature including artificial neural networks (ANN). I was also constantly implementing the basic concepts in CV all the time.

Afterwards I decided to develop a core library for building my own computer vision system. I named it biologically motivated object recognition (BIMOR) because the ideas were somehow motivated by what I came to understand about the neuroscience theories of vision. Developing the library was a challenging task because at the time my coding skills were below par. I had to step up by improving my coding skills, I chose to use C++ because of efficiency as I had to code low-level high-performance operations from scratch as well.

After I finished the library I went on to start working on an automatic panorama stitching engine called Bimostitch (BIMOR + stitch) built on top of BIMOR. Again that was a very challenging task. The aim was to develop a stitching algorithm such that given a set of unordered images (more than 1) output a panorama(s). The system has to figure out which images are matching and then recover the focal lengths and 3D rotations from the matching key points. I managed to finish that project and ported the code to Android via the NDK (native developer kit) to build the app called Bimostitch, a state of the art panorama stitching engine.

I built novel sub-systems in the process such as:

A simple but highly robust feature detector. I don't use SIFT, SURF nor FAST features in BIMOR. A simple but effective descriptor for BIMOR called M3oh. I again don't use SIFT nor SURF descriptors because they are patented. A new fast scalable matching algorithm for high dimensional vectors called the FastMatcher. I developed this one just recently and deployed it to Bimostitch panorama stitcher. It does not again match FAST key points, the name merely means it is extremely fast at finding k nearest neighbors given a query vector. The above approaches use very little machine learning (ML), only clustering algorithms such k-means or reciprocal nearest neighbor hierarchical clustering are mostly used there. So to introduce ML to my vision systems I decided to start building a 3D vision system that learns from examples, that system I call integrated recognition and inference system (IRIS) for 3D object detection. IRIS is an end-to-end deep learning system designed with the goal of learning to localize objects in a 3D scene. I have been working on IRIS for about a year now and it has undergone many design and implementation changes so far.

So those projects above have stretched my understanding of the ML and CV fields in particular. Working on them has increased my knowledge dramatically and that is how I learnt CV and ML.

So the best resources are just available online just start working on something interesting like a side project and your mind will be curious to learn so that you can solve the tasks at hand.

The prerequisites to learning CV are:

Maths:
Linear algebra
Probability and statistics
Numerical optimization
Calculus
Image processing: or generally digital signal processing.
Digital filters: Gaussian blur
Image transformations: rotations, translations, scaling and many more.
Scale space theorem for natural stimuli processing. Basically implemented using image pyramids, the so called multi-scale approach.
Machine learning (ML):
A very important area that is changing many AI sub-fields. You need to know about:

Convolutional neural networks (CNN).
Fully connected neural networks.
Support vector machines (SVM)
Recurrent neural networks (RNN) such as long-short-term-memory (LSTM) or gated recurrent unit (GRU) networks.
Generative adversarial networks (GAN): For many tasks like image-to-image translation. Image enhancements, art style transfer and so on.
Autoencoders: For noise reduction and image compression.
Programming:
C/C++: You can use OpenCV for CV implementations or build your own from scratch using C/C++.
Java: OpenCV is also available in Java.
Python: You can again use OpenCV in Python language. Probably this is a better choice as Python is a very high-level easy to learn language.
