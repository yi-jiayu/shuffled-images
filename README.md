# shuffled-images
Square jigsaw puzzle solver based on [1]. Generates solutions for the [Huawei
Honorcup Marathon 2](https://codeforces.com/contest/1235/problem/A1)

## Prerequisites

- Python 3.6+

## Quickstart

1. Create a new virtual environment and install dependencies:

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Dataset can be downloaded from https://yadi.sk/d/BmnFhkaD1Vy4vA
3. Extract the data into `data/` in the project root
4. To run the solver on an image cut into 8x8 parts and scrambled:

```
$ python main.py data/data_train/64/1623.png 8 8
1623.png
50 8 43 45 25 57 52 35 11 36 56 63 40 41 5 12 0 21 1 27 13 46 33 17 31 34 58 60 15 59 14 38 54 23 26 49 18 37 55 16 32 42 47 6 3 53 7 2 62 61 30 39 29 44 48 24 22 9 10 28 20 51 19 4
1623.png,5.762885261,0.9553571428571429,1
```

The first two lines of the output are printed to stdout and are in the contest
submission format, while the 3rd line was actually printed to stderr and
contains some statistics about the solving process in the format `basename,time
taken,score,iterations`.

5. To view the unscrambled image, you can pass the tile indices (second line of
   the output above) to `rearrange_image.py` via standard input. The rest of
   the arguments are the input image, number of rows and columns and output
   file.

```
$ pbpaste | python rearrange_image.py data/data_train/64/1623.png 8 8 out.png
```

[1]: D. Pomeranz, M. Shemesh, and O. Ben-Shahar, [A fully automated greedy square jigsaw puzzle
solver](https://www.cs.bgu.ac.il/~ben-shahar/Publications/2011-Pomeranz_Shemesh_and_Ben_Shahar-A_Fully_Automated_Greedy_Square_Jigsaw_Puzzle_Solver.pdf),
In the Proceedings of the IEEE International Conference on Computer Vision and Pattern Recognition
(CVPR), Colorado Springs, USA, June, pp. 9-16, 2011. See also the [project
page](http://www.cs.bgu.ac.il/~icvl/projects/project-jigsaw.html) for more demos, info, and code
