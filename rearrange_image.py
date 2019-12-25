import sys
from skimage import io
import numpy as np

in_file, nrows, ncols, out_file = sys.argv[1:5]
nrows, ncols = int(nrows), int(ncols)
image = io.imread(in_file)
squares = np.array([np.hsplit(r, ncols) for r in np.split(image, nrows)])
square_shape = squares[0][0].shape
permutation = [int(n) for n in input().split()]
rearranged_squares = squares[[n // nrows for n in permutation], [n % nrows for n in permutation]].reshape((nrows, ncols) + square_shape)
rearranged_image = np.vstack([np.hstack(row) for row in rearranged_squares])
io.imsave(out_file, rearranged_image)
