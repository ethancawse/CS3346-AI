import csv
import mdp
import pygame



def loadGrid(path):  # loads file from csv
    grid = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            newRow = []
            for cell in row:
                if cell == "##":
                    newRow.append("##")
                else:
                    newRow.append(float(cell))
            grid.append(newRow)
    return grid


# load data grid
data = loadGrid("test.csv")

# mdp values
V, policy = mdp.value_iteration(data)
arrows = mdp.arrows_from_policy(policy)


pygame.init()

screen = pygame.display.set_mode((640, 480))
pygame.display.set_caption("MDP Optimal Path Visualizer")

# load images from current directory
buildingImg = pygame.image.load("building.png").convert_alpha()
roadImg = pygame.image.load("road.png").convert_alpha()
goalImg = pygame.image.load("goal.png").convert_alpha()


# scale images to fit cells
def scaleImages(cellWidth, cellHeight):
    buildingScaled = pygame.transform.scale(buildingImg, (int(cellWidth), int(cellHeight)))
    roadScaled = pygame.transform.scale(roadImg, (int(cellWidth), int(cellHeight)))
    goalScaled = pygame.transform.scale(goalImg, (int(cellWidth), int(cellHeight)))
    return buildingScaled, roadScaled, goalScaled


font = pygame.font.SysFont(None, 28)


def displayGrid(grid, arrows, agentPos=None):
    rows = len(grid)
    cols = len(grid[0])
    cellHeight = screen.get_height() / rows
    cellWidth = screen.get_width() / cols

    # scale images to the sizes of the cells
    buildingScaled, roadScaled, goalScaled = scaleImages(cellWidth, cellHeight)

    for i in range(rows):
        for j in range(cols):
            cell = grid[i][j]

            # draw images based on the cell
            if cell == "##":
                # wall cell
                screen.blit(buildingScaled, (j * cellWidth, i * cellHeight))
            else:
                # check if the cell is a goal
                if mdp.is_terminal(cell) and cell > 0:
                    # goal
                    screen.blit(goalScaled, (j * cellWidth, i * cellHeight))
                else:
                    # normal path
                    screen.blit(roadScaled, (j * cellWidth, i * cellHeight))

            pygame.draw.rect(screen, (0, 0, 0), (j * cellWidth, i * cellHeight, cellWidth, cellHeight), 1)



            if cell != "##" and not mdp.is_terminal(cell):
                arrow = arrows[i][j]
                txt = font.render(arrow, True, (0, 0, 0))
                screen.blit(txt, (j * cellWidth + cellWidth / 3, i * cellHeight + cellHeight / 3))


    if agentPos:
        ai, aj = agentPos
        x = aj * cellWidth + cellWidth / 2
        y = ai * cellHeight + cellHeight / 2
        pygame.draw.circle(screen, (0, 0, 255), (int(x), int(y)), int(min(cellWidth, cellHeight) / 3))


def stepAgent(grid, policy, i, j):
    action = policy[i][j]
    if action in mdp.ACTIONS:
        di, dj = mdp.ACTIONS[action]
        ni, nj = i + di, j + dj

        if mdp.in_bounds(grid, ni, nj) and not mdp.is_wall(grid[ni][nj]):
            return ni, nj
    return i, j


def findStart(grid):
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] != "##" and grid[i][j] == 0.0:
                return (i, j)
    return (0, 0)



agentPos = findStart(data)
clock = pygame.time.Clock()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((255, 255, 255))

    # draw grid + agent marker
    displayGrid(data, arrows, agentPos)

    pygame.display.flip()


    i, j = agentPos
    if not mdp.is_terminal(data[i][j]):
        agentPos = stepAgent(data, policy, i, j)

    clock.tick(5)  # how many steps the agent takes per second

pygame.quit()
