# [APP5] GPU programming (2.5 ECTS)

This course offers a practical introduction to GPU programming, focusing on the principles and techniques behind parallel computation using the CUDA programming model. Students will explore how modern GPUs execute thousands of threads concurrently, how memory hierarchies affect performance, and how to design efficient kernels for tasks such as matrix computations, scientific computing, and machine-learning-related operations. Through a blend of lectures, hands-on examples, and targeted programming assignments, the course builds core skills in parallel algorithm design and performance optimization, preparing students to create high-performance GPU-accelerated applications and use existing optimized GPU libraries.

# Course organization

We will have six sessions of 4 hours each (including mini-exams, c.f. below). The first session is an introduction to the GPU architecture and CUDA programming model. The rest will be a mix of course (around 1h) and lab session (around 2h30-3h) on that subject. You might not be able to finish all exercises during the lab session (which is completely normal); you should make sure that you can understand and finish all basic exercises (first two or three) during the session so that you feel more confident to finish the rest on your own.

Please put your name, e-mail address and master's track here to register for the course:

https://cirrus.universite-paris-saclay.fr/s/HG3YiP4qr3ryxSK

## Session 1: Introduction to GPU architecture and CUDA programming
## Session 2: Introduction to CUDA programming (cont.)
## Session 3: 1D and 2D GPU kernels, matrix multiplication
## Session 4: GPU memory challenges, memory coalescence, matrix multiplication and stencil computations using shared memory
## Session 5: Reduction on GPUs, thread divergence, memory bank conflits, matrix transposition.
## Session 6: GPU streams, using CUDA libraries for efficient GPU applications

# Evaluation

## End-of-session mini-exams
At the end of weeks 3, 4, 5, 6 there will be mini-exams of 20-25 minutes, giving 5/20 points each. Types of exercises that might appear on these mini-exams are:

**Debugging:** Given a complete code with multiple bugs, the goal is to identify each bug (without correcting them), say what the problem is in a single sentence, and point out the corresponding line(s) in the code.

**Output:** Given a complete code, the goal is to find out what the program could potentially display. Indeed, this requires a thorough understanding of how the code works. There might be multiple possibilities, in which case you should indicate all of them.

**Coding:** A squeleton code or a function signature will be given for a specific task, and you will be expected to provide a parallel/optimized code that performs the intended task.

**Course concepts:** Small questions requiring interpretations from the concepts learned in the class. Example: How come GPUs are able to make very fast context switch between warps in a block?

**Code deciphering:** Given a complete GPU code (with anonymized variable names), understand and explain what the code actually computes.

You can bring **four** A4 sheets to the mini-exam (both sides); no other material is allowed. No electronics will be allowed; computers, smartphones, tablets etc. must be put in your backpack.

The mini-exam of week n will cover the material up to (including) week n - 1 (e.g., first mini-exam will be on the material of first two weeks).

Lab sessions will not be graded; so please do not use AI tools for programming assignments -- you will get a fish for the day but will not learn how to fish for later.

# Technical tips

Text editor: If you do not have a "preferred" text editor for development already, I recommend Visual Studio Code: https://code.visualstudio.com . You can follow the official tutorial for the basics: https://www.youtube.com/watch?v=B-s71n0dHUk&list=PLj6YeMhvp2S5UgiQnBfvD7XgOMKs3O_G6

You will need a NVCC compiler for this course. You can use the Godbolt compiler to run your code: cuda.godbolt.org
