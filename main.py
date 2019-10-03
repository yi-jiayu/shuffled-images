Relation = int
Left: Relation = 0
Right: Relation = 1
Up: Relation = 2
Down: Relation = 3


def compatibility(x_i, x_j, r: Relation) -> float:
    """Returns the likelihood that part x_i is placed to the R side of x_j."""
    pass


def opposite_relation(r: Relation) -> Relation:
    if r == Left:
        return Right
    elif r == Right:
        return Left
    elif r == Up:
        return Down
    elif r == Down:
        return Up
    raise TypeError


def best_buddies(x_i, x_j, r: Relation, parts) -> bool:
    """Returns a boolean result indicating whether part x_i and x_j are "best buddies"."""
    r_1, r_2 = r, opposite_relation(r)
    return all(compatibility(x_i, x_j, r_1) >= compatibility(x_i, x_k, r_1) for x_k in parts) and all(
        compatibility(x_j, x_i, r_2) >= compatibility(x_j, x_p, r_2) for x_p in parts)
