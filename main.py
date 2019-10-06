import numpy as np
import itertools
from skimage import color, io

P = 0.3
Q = 0.0625

LEFT = 0
RIGHT = 1
UP = 2
DOWN = 3


def load_image(path):
    return color.rgb2lab(io.imread(path))


def split_into_squares(image, nrows, ncols):
    return list(itertools.chain.from_iterable(np.hsplit(r, ncols) for r in np.split(image, nrows)))


def calculate_dissimilarity(x_i, x_j, relation):
    nrows, ncols, _ = x_i.shape

    if relation == LEFT:
        return calculate_dissimilarity(x_j, x_i, RIGHT)
    elif relation == RIGHT:
        return np.sum(
            np.power(np.power(np.abs((2 * x_i[:, ncols - 1] - x_i[:, ncols - 2]) - x_j[:, 0]), P) +
                     np.power(np.abs((2 * x_j[:, 0] - x_j[:, 1]) - x_i[:, ncols - 1]), P), Q / P))
    elif relation == UP:
        return calculate_dissimilarity(x_j, x_i, DOWN)
    elif relation == DOWN:
        return np.sum(
            np.power(np.power(np.abs((2 * x_i[nrows - 1] - x_i[nrows - 2]) - x_j[0]), P) +
                     np.power(np.abs((2 * x_j[0] - x_j[1]) - x_i[nrows - 1]), P), Q / P))
    else:
        raise TypeError(f'invalid relation: {relation}')


def build_dissimilarity_matrix(squares):
    dissimilarity_matrix = np.empty((4, len(squares), len(squares)))
    for relation in range(4):
        for i, x_i in enumerate(squares):
            for j, x_j in enumerate(squares):
                if i == j:
                    continue
                dissimilarity_matrix[relation][i][j] = calculate_dissimilarity(x_i, x_j, relation)
    return dissimilarity_matrix


def calculate_compatibility(dissimilarity_matrix, i, j, relation):
    return np.exp(-dissimilarity_matrix[relation][i][j] /
                  np.percentile(np.delete(dissimilarity_matrix[relation][i], i), 25))


def build_compatibility_matrix(dissimilarity_matrix):
    _, order, _ = dissimilarity_matrix.shape
    compatibility_matrix = np.empty((4, order, order))
    for relation in range(4):
        for i in range(order):
            for j in range(order):
                if i == j:
                    continue
                compatibility_matrix[relation][i][j] = calculate_compatibility(dissimilarity_matrix, i, j, relation)
    return compatibility_matrix


def calculate_best_neighbours(compatibility_matrix):
    _, order, _ = compatibility_matrix.shape
    best_neighbours = np.zeros((4, order), dtype=int)
    for relation in range(4):
        for i in range(order):
            best_neighbours[relation][i] = np.argmax(compatibility_matrix[relation][i])
    return best_neighbours


def opposite_relation(relation):
    if relation == 0 or relation == 2:
        return relation + 1
    else:
        return relation - 1


def find_best_estimated_seed(best_neighbours):
    _, order = best_neighbours.shape
    num_best_buddies = np.zeros(order, dtype=int)
    for relation in range(4):
        for i in range(order):
            buddy = best_neighbours[relation][i]
            opposite = opposite_relation(relation)
            if best_neighbours[opposite][buddy] == i:
                num_best_buddies[i] += 1
    return np.argmax(num_best_buddies)


if __name__ == '__main__':
    image = load_image('data/data_train/64/1200.png')
    squares = split_into_squares(image, 8, 8)
    dissimilarity_matrix = build_dissimilarity_matrix(squares)
    compatibility_matrix = build_compatibility_matrix(dissimilarity_matrix)
    best_neighbours = calculate_best_neighbours(compatibility_matrix)
    print(best_neighbours[RIGHT][0])
    print(best_neighbours[LEFT][3])

