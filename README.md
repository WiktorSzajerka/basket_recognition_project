# 1. The goal
The goal of the project is to detect made basketball shots from the recording of a basketball game. The game is recorded from a single camera on the side of the court
# 2. Solution
Made basket detection is splited into two parts
## 2.1 Basket detection
To detect the basket and the ball, the ready-made Yolov11s model was used, which was
trained on a collection of 1,255 photos from basketball games prepared by me with
marked baskets and balls.
## 2.2 Detecting whether the ball went into the basket
This task is performed by a CNN network designed by myself using the
Pytorch library. This network takes a 32x32 photo as input and classifies whether
the ball went into the basket or not.

