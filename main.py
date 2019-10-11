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


def calculate_compatibility(dissimilarity_matrix, percentiles, i, j, relation):
    percentile = percentiles[relation][i]
    if percentile == 0:
        percentile = 2.220446049250313e-16
    return np.exp(-dissimilarity_matrix[relation][i][j] /
                  percentile)


def build_compatibility_matrix(squares):
    dissimilarity_matrix = build_dissimilarity_matrix(squares)
    _, order, _ = dissimilarity_matrix.shape

    # precalculate percentiles
    percentiles = np.empty((4, order))
    for i in range(order):
        for relation in range(4):
            percentiles[relation][i] = np.percentile(np.delete(dissimilarity_matrix[relation][i], i), 25)

    compatibility_matrix = np.empty((4, order, order))
    for i in range(order):
        for j in range(order):
            for relation in range(4):
                if i == j:
                    continue
                elif i < j:
                    compatibility_matrix[relation][i][j] = calculate_compatibility(dissimilarity_matrix, percentiles, i,
                                                                                   j, relation)
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


def is_in_grid(grid, i, j):
    nrows, ncols = grid.shape
    return 0 <= i < nrows and 0 <= j < ncols


def is_occupied_slot(puzzle, i, j):
    return is_in_grid(puzzle, i, j) and puzzle[i][j] >= 0


def find_candidate_slots(puzzle):
    slots = {}
    for i, j in np.argwhere(puzzle != -1):
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
    if not is_in_grid(puzzle, i, j):
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
    best_neighbours = calculate_best_neighbours(compatibility_matrix)
    # seed = find_best_estimated_seed(best_neighbours)
    seed = np.random.randint(nrows * ncols)
    solution[nrows // 2][ncols // 2] = seed
    unallocated_parts.remove(seed)

    return solution, compatibility_matrix, best_neighbours, unallocated_parts


def place_remaining_parts(puzzle, compatibility_matrix, best_neighbours, unallocated_parts):
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
    solution, compatibility_matrix, best_neighbours, unallocated_parts = initialise_solution(squares, nrows, ncols)

    best_score = -1
    best_solution = None
    iterations = 0
    while True:
        # placer
        solution = placer(solution, compatibility_matrix, best_neighbours, unallocated_parts)
        score = calculate_best_buddies_metric(solution, best_neighbours)
        if score <= best_score:
            break

        iterations += 1
        best_score = score
        best_solution = solution

        # segmenter
        segments = segment(solution, best_neighbours)
        masked = mask_largest_segment(solution, segments)

        # shifter
        solution, unallocated_parts = center_occupied_in_grid(masked)

    # return solution as a 1-D array
    return np.ravel(best_solution), best_score, iterations


def placer(solution, compatibility_matrix, best_neighbours, unallocated_parts):
    while unallocated_parts:
        solution, unallocated_parts = place_remaining_parts(solution, compatibility_matrix, best_neighbours,
                                                            unallocated_parts)
    return solution


def calculate_best_buddies_metric(puzzle, best_neighbours):
    nrows, ncols = puzzle.shape
    num_edges = (nrows - 1) * ncols + (ncols - 1) * nrows
    num_best_buddies = 0

    # left/right
    for i in range(nrows):
        for j in range(ncols - 1):
            if best_buddies(best_neighbours, RIGHT, puzzle[i][j], puzzle[i][j + 1]):
                num_best_buddies += 1
    # up/down
    for i in range(nrows - 1):
        for j in range(ncols):
            if best_buddies(best_neighbours, DOWN, puzzle[i][j], puzzle[i + 1][j]):
                num_best_buddies += 1

    return num_best_buddies / num_edges


def is_part_in_segment(puzzle, best_neighbours, segments, segment, i, j):
    for relation in range(4):
        x, y = related_coords(relation, i, j)
        if is_in_grid(segments, x, y) and segments[x][y] == segment:
            if not best_buddies(best_neighbours, relation, puzzle[i][j], puzzle[x][y]):
                return False
    return True


def segment(puzzle, best_neighbours):
    nrows, ncols = puzzle.shape
    segments = np.zeros((nrows, ncols), dtype=int)
    segment_counter = 1

    while True:
        unassigned_coords = np.argwhere(segments == 0)
        if unassigned_coords.size == 0:
            break

        stack = [unassigned_coords[np.random.choice(range(len(unassigned_coords)))]]
        while stack:
            i, j = stack.pop()
            segments[i][j] = segment_counter
            for x, y in adjacent(i, j):
                if is_in_grid(segments, x, y):
                    if segments[x][y] == 0 and \
                            is_part_in_segment(puzzle, best_neighbours, segments, segment_counter, x, y):
                        stack.append((x, y))
        segment_counter += 1
    return segments


def largest_segment_index(segments):
    segment_indices, counts = np.unique(segments, return_counts=True)
    return segment_indices[np.argmax(counts)]


def mask_largest_segment(puzzle, segments):
    return np.where(segments == largest_segment_index(segments), puzzle, -1)


def center_occupied_in_grid(puzzle):
    nrows, ncols = puzzle.shape
    rows, cols = np.nonzero(puzzle + 1)
    delta_y = nrows // 2 - (max(rows) + min(rows)) // 2 - 1
    delta_x = ncols // 2 - (max(cols) + min(cols)) // 2 - 1
    puzzle = np.roll(puzzle, delta_y, 0)
    puzzle = np.roll(puzzle, delta_x, 1)
    allocated_parts = set(puzzle[puzzle != -1])
    unallocated_parts = set(range(nrows * ncols)) - allocated_parts
    return puzzle, unallocated_parts


if __name__ == '__main__':
    import sys
    import os.path
    import time

    start_time = time.monotonic()
    image_path, nrows, ncols = sys.argv[1:4]
    basename = os.path.basename(image_path)

    attempts = []
    for i in range(10):
        attempts.append(solve(image_path, int(nrows), int(ncols)))

    best_attempt = max(attempts, key=lambda x: x[1])
    solution, score, iterations = best_attempt
    print(basename)
    print(' '.join(str(x) for x in solution))
    print(f'{basename},{time.monotonic() - start_time},{score},{iterations}', file=sys.stderr)
