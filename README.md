# shuffled-images
Square jigsaw puzzle solver based on [1]. Generates solutions for the [Huawei
Honorcup Marathon 2](https://codeforces.com/contest/1235/problem/A1)

## Generating solutions

```
$ for f in data/data_train/64/*; do echo $f; python main.py $f 8 8 >> A1.txt; done^C
```

[1]: D. Pomeranz, M. Shemesh, and O. Ben-Shahar, [A fully automated greedy square jigsaw puzzle
solver](https://www.cs.bgu.ac.il/~ben-shahar/Publications/2011-Pomeranz_Shemesh_and_Ben_Shahar-A_Fully_Automated_Greedy_Square_Jigsaw_Puzzle_Solver.pdf),
In the Proceedings of the IEEE International Conference on Computer Vision and Pattern Recognition
(CVPR), Colorado Springs, USA, June, pp. 9-16, 2011. See also the [project
page](http://www.cs.bgu.ac.il/~icvl/projects/project-jigsaw.html) for more demos, info, and code
