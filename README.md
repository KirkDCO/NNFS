# NNFS
Neural Networks From Scratch

**Background**

In the summer of 2018, I completed the Deep Learning Specialization from Coursera.  The courses and the code projects were very good and provided me with an understanding of neural networks and their workings.  But, that understanding felt a bit superficial.  While I did well in the courses and was easily able to complete the code projects, I felt that I didn't have enough of an understanding of the theory.  I began reading Neural Networks for Pattern Recognition by Chris Bishop, which is an excellent book despite being somewhat dated.  The presentation of neural networks from a statistcal learning theory perspective was incredibly helpful.  But I still felt I needed to know more.  That's where this project started.

**Goal**

My intention with this project was to write a fully functioning neural network architecture from base principles without ising an existing framework.  I wanted to implement various activation functions (linearly, sigmoidal, tanh, ReLU, leaky ReLU, and softmax) along with the code necessary for back propagation.  (I did use R's built in tanh function, however.)

**What This Is**

I have succeeded in my goal and was able to develop code necessary to create arbitraily deep and tall neural networks with the activation functions listed above.  All of that is contained in NNFS.R which provides the functions necessary for this general purpose library.  The Examples.R file contains a variety of experiments I used to test the code while developing it.  In the end, I'm quite happy with the result and have learned a tremendous amount about the intricacies of neural networks.

**What This Isn't**

*Fast* 
Actually, it runs quite well but it has in no way been optimized for speed.

*A framework*
This was a learning project and while it provides a reusable code base, it certainly isn't a general purpose neural networks framework.

*Tidy*
Since my intention was more mathematical and theorhetical, I relied on base R.

