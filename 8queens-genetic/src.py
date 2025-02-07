import random

MAX_GENERATIONS = 6000
POPULATION_SIZE = 45
MUTATION_CHANCE = 0.3
BOARD_LEN = 8


def generate_random_locations():
    # board contains 'BOARD_LEN' ints denoting columns, and the
    # index within the board list itself a row
    return [random.randint(0, BOARD_LEN - 1) for _ in range(BOARD_LEN)]


def generate_initial_population():
    return [generate_random_locations() for _ in range(POPULATION_SIZE)]


def print_board(board):
    print("\n", board, "\n")
    for row in board:
        print(row * ". ", end="")
        print("Q ", end="")
        print((BOARD_LEN - row - 1) * ". ")


def evaluate_badness(board):
    badness = 0

    # row 'x'
    for i in range(BOARD_LEN):
        # row 'x + 1'
        for j in range(i + 1, BOARD_LEN):

            # note: no need to check the rows because we know
            # we are only generating one queen per row

            if (
                # if two columns are equal
                board[i] == board[j]
                or
                # the diff. b/w rows is equal to the diff. b/w columns
                # i.e a queen lies in another's diagonal
                abs(board[i] - board[j]) == (j - i)
            ):
                badness += 1
    return badness


# Random distribution, but the worse a particular board is,
# the lower its odds of selection should be
def select_parents(population, badnesses):
    total = sum(badnesses)
    probabilities = [((total - b) / total) for b in badnesses]
    parents = random.choices(population=population, weights=probabilities, k=2)
    return parents


def crossover(parent1, parent2):
    idx = random.randint(0, BOARD_LEN - 1)
    return parent1[0:idx] + parent2[idx:BOARD_LEN]


def mutate(board):
    if random.random() < MUTATION_CHANCE:

        # NOTE: this approach is much, much, much more efficient
        # than swapping two randomly generated indexes

        # Perhaps because trivially swapping just cycles b/w shitty options?

        idx = random.randint(0, BOARD_LEN - 1)
        board[idx] = random.randint(0, BOARD_LEN - 1)
    return board


def genetic_algorithm(population):
    for generation in range(MAX_GENERATIONS):
        badnesses = [evaluate_badness(board) for board in population]

        best_score = min(badnesses)
        best_index = badnesses.index(best_score)

        # if we solved it, return
        if best_score == 0:
            return (generation, population[best_index])

        # keep best sample
        new_population = [population[best_index]]

        # Generate new population
        while len(new_population) < POPULATION_SIZE:
            parent1, parent2 = select_parents(population, badnesses)
            child = mutate(crossover(parent1, parent2))
            new_population.append(child)

        # update pop for next generation
        population = new_population
    return (MAX_GENERATIONS, [])


def main():
    pop = generate_initial_population()
    generation_count, board = genetic_algorithm(pop)

    if not board:
        print("Solution could not be found in", MAX_GENERATIONS, "generations")
    else:
        print("Solution found in", generation_count, "generations.")
        print_board(board)


if __name__ == "__main__":
    main()
