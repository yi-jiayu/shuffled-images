import numpy as np
import itertools
from skimage import color, io

P = 0.3
Q = 0.0625

LEFT = 0
RIGHT = 1
UP = 2
DOWN = 3


def calculate_dissimilarity(x_i, x_j, relation):
    nrows, ncols = x_i.shape[0], x_i.shape[1]

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
    for i, x_i in enumerate(squares):
        for j, x_j in enumerate(squares):
            for relation in range(4):
                if i == j:
                    continue
                elif i < j:
                    dissimilarity_matrix[relation][i][j] = calculate_dissimilarity(x_i, x_j, relation)
                else:
                    dissimilarity_matrix[relation][i][j] = dissimilarity_matrix[opposite_relation(relation)][j][i]
    return dissimilarity_matrix


def calculate_compatibility(dissimilarity_matrix, i, j, relation):
    percentile = np.percentile(np.delete(dissimilarity_matrix[relation][i], i), 25)
    if percentile == 0:
        percentile = 2.220446049250313e-16
    return np.exp(-dissimilarity_matrix[relation][i][j] /
                  percentile)


def build_compatibility_matrix(squares):
    dissimilarity_matrix = build_dissimilarity_matrix(squares)
    _, order, _ = dissimilarity_matrix.shape
    compatibility_matrix = np.empty((4, order, order))
    for i in range(order):
        for j in range(order):
            for relation in range(4):
                if i == j:
                    continue
                elif i < j:
                    compatibility_matrix[relation][i][j] = calculate_compatibility(dissimilarity_matrix, i, j, relation)
                else:
                    compatibility_matrix[relation][i][j] = compatibility_matrix[opposite_relation(relation)][j][i]
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


def adjacent(i, j):
    yield i - 1, j
    yield i + 1, j
    yield i, j - 1
    yield i, j + 1


def is_in_puzzle(puzzle, i, j):
    nrows, ncols = puzzle.shape
    return 0 <= i < nrows and 0 <= j < ncols


def is_occupied_slot(puzzle, i, j):
    return is_in_puzzle(puzzle, i, j) and puzzle[i][j] >= 0


def find_candidate_slots(puzzle):
    slots = {}
    nrows, ncols = puzzle.shape
    for i in range(nrows):
        for j in range(ncols):
            if puzzle[i][j] >= 0:
                for x, y in adjacent(i, j):
                    # skip if we've already added this slot
                    if (x, y) in slots:
                        continue
                    # skip if (x, y) is already occupied
                    if is_occupied_slot(puzzle, x, y):
                        continue
                    # count the number of occupied slots around (x, y)
                    slots[(x, y)] = sum(1 if is_occupied_slot(puzzle, p, q) else 0 for p, q in adjacent(x, y))
    max_neighbours = max(slots.values())
    return set(slot for slot, num_neighbours in slots.items() if num_neighbours == max_neighbours)


def best_buddies(best_neighbours, relation, i, j):
    return best_neighbours[relation][i] == j and best_neighbours[opposite_relation(relation)][j] == i


# part fits in slot if it is best buddies with all the occupied neighbours of that slot
def does_part_fit_in_slot(puzzle, best_neighbours, slot, part):
    nrows, ncols = puzzle.shape
    i, j = slot

    for relation in range(4):
        x, y = related_coords(relation, i, j)
        if is_occupied_slot(puzzle, x, y):
            if not best_buddies(best_neighbours, relation, part, puzzle[x][y]):
                return False
    return True


def related_coords(relation, i, j):
    if relation == RIGHT:
        return i, j + 1
    elif relation == LEFT:
        return i, j - 1
    elif relation == UP:
        return i - 1, j
    elif relation == DOWN:
        return i + 1, j
    else:
        raise ValueError(f'invalid relation: {relation}')


def average_compatibility_with_slot(puzzle, compatibility_matrix, slot, part):
    i, j = slot
    total_compatibility = 0
    num_neighbours = 0

    for relation in range(4):
        x, y = related_coords(relation, i, j)
        if is_occupied_slot(puzzle, x, y):
            total_compatibility += compatibility_matrix[relation][part][puzzle[x][y]]
            num_neighbours += 1

    return total_compatibility / num_neighbours


class SlotAssignError(Exception):
    pass


def try_assign(puzzle, slot, part, unallocated_parts):
    nrows, ncols = puzzle.shape
    # can always assign if slot is within puzzle
    i, j = slot
    if not is_in_puzzle(puzzle, i, j):
        # check if opposite edge is empty and roll puzzle
        # slot is above top edge - check if bottom edge is empty
        if i < 0:
            if not np.all(puzzle[-1] == -1):
                raise SlotAssignError
            puzzle = np.roll(puzzle, 1, 0)
            i += 1
        # slot is below bottom edge - check if top edge is empty
        elif i >= nrows:
            if not np.all(puzzle[0] == -1):
                raise SlotAssignError
            puzzle = np.roll(puzzle, -1, 0)
            i -= 1
        # slot is to the left of puzzle - check if right edge is empty
        elif j < 0:
            if not np.all(puzzle[:, -1] == -1):
                raise SlotAssignError
            puzzle = np.roll(puzzle, 1, 1)
            j += 1
            # slot is to the right side of puzzle - check if left edge is empty
        elif j >= ncols:
            if not np.all(puzzle[:, 0] == - 1):
                raise SlotAssignError
            puzzle = np.roll(puzzle, -1, 1)
            j -= 1
        else:
            raise ValueError('invalid slot')

    # update unallocated parts
    unallocated_parts.remove(part)
    # update puzzle
    puzzle[i][j] = part
    return puzzle, unallocated_parts


def initialise_solution(squares, nrows, ncols):
    # initialise compatibility matrix
    compatibility_matrix = build_compatibility_matrix(squares)
    solution = np.full((nrows, ncols), -1)
    unallocated_parts = set(range(nrows * ncols))

    # calculate best seed and place in centre of solution
    best_estimated_seed = find_best_estimated_seed(calculate_best_neighbours(compatibility_matrix))
    solution[nrows // 2][ncols // 2] = best_estimated_seed
    unallocated_parts.remove(best_estimated_seed)

    return solution, compatibility_matrix, unallocated_parts


def place_remaining_parts(puzzle, compatibility_matrix, unallocated_parts):
    best_neighbours = calculate_best_neighbours(compatibility_matrix)
    candidate_slots = find_candidate_slots(puzzle)

    while True:
        matches = [(slot, part) for slot in candidate_slots for part in unallocated_parts if
                   does_part_fit_in_slot(puzzle, best_neighbours, slot, part)]
        if len(matches) == 1:
            slot, part = matches.pop()
        else:
            average_compatibilities = [
                (average_compatibility_with_slot(puzzle, compatibility_matrix, slot, part), (slot, part))
                for slot in candidate_slots for part in unallocated_parts]
            best = max(average_compatibilities, key=lambda x: x[0])
            slot, part = best[1]

        try:
            puzzle, unallocated_parts = try_assign(puzzle, slot, part, unallocated_parts)
            return puzzle, unallocated_parts
        except SlotAssignError:
            candidate_slots.remove(slot)
            if not candidate_slots:
                raise ValueError('no more slots')


def solve(image_path, nrows, ncols):
    # load image
    image = io.imread(image_path)

    #  convert to LAB colour space if RGB
    if image.ndim > 2:
        image = color.rgb2lab(image)

    # split into squares
    squares = list(
        itertools.chain.from_iterable(np.hsplit(r, ncols) for r in np.split(image, nrows)))

    # initialise solution
    solution, compatibility_matrix, unallocated_parts = initialise_solution(squares, nrows, ncols)

    # assign remaining parts
    while unallocated_parts:
        solution, unallocated_parts = place_remaining_parts(solution, compatibility_matrix, unallocated_parts)

    # return solution as a 1-D array
    return np.ravel(solution)


if __name__ == '__main__':
    import sys
    import os.path
    import time

    start_time = time.monotonic()
    image_path, nrows, ncols = sys.argv[1:4]
    basename = os.path.basename(image_path)

    solution = solve(image_path, int(nrows), int(ncols))
    print(basename)
    print(' '.join(str(x) for x in solution))
    print(f'{basename},{time.monotonic() - start_time}', file=sys.stderr)
