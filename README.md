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

*A  general purpose framework*

This was a learning project and while it provides a reusable code base, it certainly isn't a general purpose neural networks framework.

*Tidy*

Since my intention was more mathematical and theoretical, I relied on base R.

*Done*

There are a few things left to do here (e.g., batch normalization of weights, implement dropout), and I'm sure more will come up over time.  For now, I've accomplished my goal and will work on this as I have time.


## LICENSE

Copyright 2018 Robert Kirk DeLisle

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
