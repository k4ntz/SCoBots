from random import randint, choice

# Global Variables
TOGGLE_ORANGE = 0
TOGGLE_CYAN = 0
TOGGLE_PINK = 0
TOGGLE_RED = 0
NUMBER_POWER_PILLS = 4
LAST_PP_STATUS = 4
IS_INVERTED = False
LVL_NUM = 0
LIVES = 2
COL = None
LINE = None
COL_POS = [42, 67, 134]
LINE_POS = [26, 74, 122]
DOT_POS = [(0, 6), (5, 13), (12, 18)]

TIMER = 1

DOT_STATES = [
    59,
    60,
    61,
    62,
    65,
    66,
    67,
    71,
    72,
    73,
    83,
    89,
    90,
    91,
    92,
    95,
    98,
    99,
    100,
]

# Each line has the same pattern, to get the states and values for the other lines, simply add line number * 3 (counting from 0) to the first value of the tupel.
# For the first line its 59+(0*3), for the second line its 59+(1*3), for the third its 59+(2*3)... so state + ((n-1) * 3)
DOT_PATTERN = [
    (59, 64),
    (60, 128),
    (60, 32),
    (60, 8),
    (60, 2),
    (61, 1),
    (61, 4),
    (61, 16),
    (61, 64),
    (60, 64),
    (60, 16),
    (60, 4),
    (60, 1),
    (61, 2),
    (61, 8),
    (61, 32),
    (61, 128),
    (59, 16),
]

GRID1 = [
    [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
    [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0],
    [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
    [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
    [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
    [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    [1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1],
    [1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]


def make_edible(env, ghost_number, x_pos, ram_x, y_pos, ram_y):
    """A helper function to make a certain ghost edible.
    The position is changed as well to avoid glitches.
    ghost_number: integer between 1 and 4 to choose between the 4 ghosts
    ram_x: integer which defines the ram cell in which the x-position is saved
    x_pos: integer which defines the new x-position of the ghost
    ram_y: integer which defines the ram cell in which the y-position is saved
    y_pos: integer which defines the new y-position of the ghost
    """
    env.set_ram(ghost_number, 130)
    env.set_ram(ram_x, x_pos)
    env.set_ram(ram_y, y_pos)


def set_start_condition(self):
    """A helper function to set the start condition at the beginning of
    each level/ after a reset"""
    # set the ghost to "edible" and change start location to avoid glitches
    # orange ghost
    make_edible(self, 1, 120, 6, 50, 12)
    # cyan ghost
    make_edible(self, 2, 100, 7, 50, 13)
    # pink ghost
    make_edible(self, 3, 80, 8, 50, 14)
    # red ghost
    make_edible(self, 4, 60, 9, 50, 15)
    # set the timer to max number
    self.set_ram(116, 255)


def inverted_power_pill(self):
    """A helper function to make the ghost "normal" again.
    They will be able to eat Ms. Pacman for a certain amount of time."""
    i = 1
    while i < 5:
        # make ghosts "normal"
        self.set_ram(i, 0)
        i += 1
    # set timer
    self.set_ram(116, 62)


def power_pill_is_done(self):
    """A helper function to make all ghosts edible again."""
    i = 1
    while i < 5:
        # make ghosts edible
        self.set_ram(i, 130)
        i += 1
    # set timer
    self.set_ram(116, 190)


def static_ghosts(self):
    """
    static_ghosts: Manipulates the RAM cell at position 6-9 and 12-15 to fix the position of
    the ghost inside the square in the middle of the screen.
    """
    if TOGGLE_ORANGE:
        self.set_ram(6, 93)
        self.set_ram(12, 80)
    if TOGGLE_CYAN:
        self.set_ram(7, 83)
        self.set_ram(13, 80)
    if TOGGLE_PINK:
        self.set_ram(8, 93)
        self.set_ram(14, 67)
    if TOGGLE_RED:
        self.set_ram(9, 83)
        self.set_ram(15, 67)


def number_power_pills(self):
    """
    number_power_pills: Manipulates the RAM cell at position 117 (= power pills),
    62 and 95 (= edible tokens). Thus resulting in switching the specified
    number of power pills with normal edible tokens.
    is_at_start: a bool used as a switch to determine if the game was reset.
    current_lives: an integer used to store the current number of lives of Ms. Pacman
    """

    if NUMBER_POWER_PILLS == 0:  # no power pills
        self.set_ram(62, 80)
        self.set_ram(95, 80)
        self.set_ram(117, 0)
    elif NUMBER_POWER_PILLS == 1:  # 1 power pill
        self.set_ram(62, 64)
        self.set_ram(95, 80)
        self.set_ram(117, 8)
    elif NUMBER_POWER_PILLS == 2:  # two power pills
        self.set_ram(62, 0)
        self.set_ram(95, 80)
        self.set_ram(117, 40)
    elif NUMBER_POWER_PILLS == 3:  # three power pills
        self.set_ram(62, 0)
        self.set_ram(95, 64)
        self.set_ram(117, 46)


def edible_ghosts(self):
    """
    edible_ghosts: Manipulates the RAM cell at position 117 (= power pills),
    62 and 95 (= edible tokens). Thus resulting in switching all power pills with
    normal edible tokens.
    Furthermore all ghost will be made edible the entire game using RAM cell 1-4
    (= ghost status) and the timer for the edible ghost mode (RAM cell 116)
    is_at_start: a bool used as a switch to determine if the game was reset.
    current_lives: an integer used to store the current number of lives of Ms. Pacman
    current_pp_status: an integer used to store the current status of the power pills
    current_timer: an integer used to store the current timer of the edible ghost mode
    current_orange: an integer used to store the current status of the orange ghost
    current_cyan: an integer used to store the current status of the orange ghost
    current_pink: an integer used to store the current status of the pink ghost
    current_red: an integer used to store the current status of the red ghost
    """
    current_timer = self.get_ram()[116]

    current_orange = self.get_ram()[1]
    current_cyan = self.get_ram()[2]
    current_pink = self.get_ram()[3]
    current_red = self.get_ram()[4]

    # check if timer needs to be adjusted
    if current_timer < 250:
        self.set_ram(116, 255)

    # check if a ghost has been eaten and if needed make them edible again
    # change start location to avoid glitches
    if current_orange == 112:
        make_edible(self, 1, 120, 6, 50, 12)
    if current_cyan == 112:
        make_edible(self, 2, 100, 7, 50, 13)
    if current_pink == 112:
        make_edible(self, 3, 80, 8, 50, 14)
    if current_red == 112 or current_red == 0:
        make_edible(self, 4, 60, 9, 50, 15)


def inverted_ms_pacman_reset(self):
    set_start_condition(self)
    global LAST_PP_STATUS, IS_INVERTED
    LAST_PP_STATUS = 63
    IS_INVERTED = False


def inverted_ms_pacman(self):
    """
    inverted_ms_pacman: Manipulates the RAM cell at position 1-4 (= ghost status) so all
    ghost will be edible the entire game until the player eats a power pill
    (RAM 117 = power pill status). After eating a power pill the ghost will return to
    "normal" for a certain amount of time (RAM cell 116 = timer).
    is_at_start: a bool used as a switch to determine if the game was reset.
    is_inverted: a bool used as a switch to determine the current game mode.
    last_pp_status: an integer used to save the last value in the RAM cell for
    the power pills. The value is needed to determine if a power pill has been eaten.
    current_lives: an integer used to store the current number of lives of Ms. Pacman
    current_pp_status: an integer used to store the current status of the power pills
    current_timer: an integer used to store the current timer of the edible ghost mode
    current_orange: an integer used to store the current status of the orange ghost
    current_cyan: an integer used to store the current status of the orange ghost
    current_pink: an integer used to store the current status of the pink ghost
    current_red: an integer used to store the current status of the red ghost
    """
    global LAST_PP_STATUS, IS_INVERTED
    current_pp_status = self.get_ram()[117]
    current_timer = self.get_ram()[116]

    current_orange = self.get_ram()[1]
    current_cyan = self.get_ram()[2]
    current_pink = self.get_ram()[3]
    current_red = self.get_ram()[4]

    # check if timer needs to be adjusted
    if current_timer < 250 and IS_INVERTED is False:
        self.set_ram(116, 255)

    # check if a power pill has been eaten
    # a range is required because the values in the RAm cells fluctuate
    if not ((LAST_PP_STATUS - 3) < current_pp_status < (LAST_PP_STATUS + 3)):
        IS_INVERTED = True
        inverted_power_pill(self)
        LAST_PP_STATUS = current_pp_status

    # check if effect of power pill has run out
    if current_timer == 0:
        IS_INVERTED = False  # disable switch
        power_pill_is_done(self)

    # check if a ghost has been eaten and if needed make them edible again
    # change start location to avoid glitches
    if current_orange == 112 and not IS_INVERTED:
        make_edible(self, 1, 120, 6, 50, 12)
    if current_cyan == 112 and not IS_INVERTED:
        make_edible(self, 2, 100, 7, 50, 13)
    if current_pink == 112 and not IS_INVERTED:
        make_edible(self, 3, 80, 8, 50, 14)
    if current_red == 112 and not IS_INVERTED:
        make_edible(self, 4, 60, 9, 50, 15)


def change_level(self):
    """
    Changes the level according to the argument number 0-3. If not specified, selcts random level.
    """
    global LVL_NUM
    if LVL_NUM is None:
        LVL_NUM = randint(0, 3)
        print(f"Selecting Random Level {LVL_NUM}")
    self.set_ram(0, LVL_NUM)


def end_game(self):
    """
    Only spawns a small cluster of pills. Simulates the end of a game.
    """
    global DOT_STATES, DOT_PATTERN, GRID1

    for i in range(59, 101):
        self.set_ram(i, 0)
    self.set_ram(117, 0)
    self.set_ram(119, 139)
    self.set_ram(123, LIVES)
    line = choice(range(0, 14))
    dot = choice(range(0, 18))
    if not GRID1[line][dot]:
        while not GRID1[line][dot]:
            dot = choice(range(0, 18))

    dots = [(line, dot)]
    i = 0

    while len(dots) < 15:
        for move in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            pos1 = dots[i][0] + move[0]
            pos2 = dots[i][1] + move[1]
            if (
                (0 <= pos1 < 14 and 0 <= pos2 < 18)
                and GRID1[pos1][pos2]
                and ((pos1, pos2) not in dots)
            ):
                dots.append((pos1, pos2))
        i += 1

    for d in range(15):
        ram = self.get_ram()
        pill = DOT_PATTERN[dots[d][1]][0] + 3 * dots[d][0]
        value = ram[pill] + DOT_PATTERN[dots[d][1]][1]
        self.set_ram(pill, value)


def check_reset(self):
    ram = self.get_ram()
    if ram[119] == 154:
        self.reset()


def maze_man(self):
    """
    Changes the gameplay. Ghosts are removed and all but one pill have been removed.
    Points now represent a timer.
    The goal of the game is to collect the pill in the maze before the time limit runs out.
    After a pill has been collected, time will be added and a new pill will spawn.
    If the player collects 20 pills, the game goes into the next level.
    """
    # makes red ghost invisible (is glitchy)
    self.set_ram(47, 0)
    # ram = self.get_ram()

    # # check if pill was collected
    # collected = True

    # # checks the pill states, set collected to false if there is one remaining on the map
    # for i in range(59, 101):
    #     if ram[i] != 0:
    #         collected = False

    # # if no pill on the map
    # if collected and ram[39] > 70:
    #     # increase the timer (inplace of score)
    #     add = ram[120] + 32
    #     if add <= 144:
    #         self.set_ram(120, add)
    #     else:
    #         self.set_ram(120, 144)

    #     # ramdomly pick a new pill spot/state
    #     global DOT_STATES
    #     state = choice(DOT_STATES)
    #     bit = choice([0, 2])
    #     self.set_ram(state, 16<<bit)#1<<bit)

    # # change or reset level on success/failure
    # global LVL_NUM, TIMER, LIVES
    # if ram[39] == 69:
    #     if ram[0] == 0 and ram[119] == 154:
    #             LVL_NUM = 1
    #             if LIVES < 3:
    #                 LIVES+=1
    #             self.reset()
    #     elif ram[0] == 1 and ram[119] == 150:
    #             LVL_NUM = 2
    #             if LIVES < 3:
    #                 LIVES+=1
    #             self.reset()
    #     elif ram[0] == 2 and ram[119] == 158:
    #             LVL_NUM = 3
    #             if LIVES < 3:
    #                 LIVES+=1
    #             self.reset()
    #     elif ram[0] == 3 and ram[119] == 154:
    #             LVL_NUM = 0
    #             if LIVES < 3:
    #                 LIVES+=1
    #             self.reset()

    # # Use ram state and TIMER variable as tick
    # if ram[39] == 255 and TIMER == 0:
    #     # resets the whole game if no lives remaining
    #     if ram[120] < 16 and ram[123] <= 0:
    #         LVL_NUM = 0
    #         LIVES = 2
    #         self.reset()
    #     # decrease lives if
    #     elif ram[120] < 16:
    #         LIVES-=1
    #         self.reset()
    #     # lower timer
    #     else:
    #         pass
    #         # self.set_ram(120, ram[120]-16)

    # # increase timer
    # TIMER = (TIMER+1)%150

    ram = self.get_ram()

    # check if pill was collected
    collected = True

    # checks the pill states, set collected to false if there is one remaining on the map
    for i in range(59, 101):
        if ram[i] != 0:
            collected = False

    # if no pill on the map
    if collected and ram[39] > 70:
        # increase the timer (inplace of score)
        add = ram[120] + 32
        if add <= 144:
            self.set_ram(120, add)
        else:
            self.set_ram(120, 144)

        # better choice alogorithm from endgame
        line = choice(range(0, 14))
        dot = choice(range(0, 18))
        if not GRID1[line][dot]:
            while not GRID1[line][dot]:
                dot = choice(range(0, 18))
        pill = DOT_PATTERN[dot][0] + 3*line
        value = ram[pill] + DOT_PATTERN[dot][1]
        self.set_ram(pill, value)

    # change or reset level on success/failure
    global LVL_NUM, TIMER, LIVES
    if ram[39] == 69:
        if ram[0] == 0 and ram[119] == 154:
            LVL_NUM = 1
            if LIVES < 3:
                LIVES += 1
            self.reset()
        elif ram[0] == 1 and ram[119] == 150:
            LVL_NUM = 2
            if LIVES < 3:
                LIVES += 1
            self.reset()
        elif ram[0] == 2 and ram[119] == 158:
            LVL_NUM = 3
            if LIVES < 3:
                LIVES += 1
            self.reset()
        elif ram[0] == 3 and ram[119] == 154:
            LVL_NUM = 0
            if LIVES < 3:
                LIVES += 1
            self.reset()

    # Use ram state and TIMER variable as tick
    if ram[39] == 255 and TIMER == 0:
        # resets the whole game if no lives remaining
        if ram[120] < 16 and ram[123] <= 0:
            LVL_NUM = 0
            LIVES = 2
            self.reset()
        # decrease lives if
        elif ram[120] < 16:
            LIVES -= 1
            self.reset()
        # lower timer
        else:
            # pass
            self.set_ram(120, ram[120]-16)

    # increase timer
    TIMER = (TIMER + 1) % 150


def maze_man_reset(self):
    for i in range(59, 101):
        self.set_ram(i, 0)
    global TIMER, LIVES, LINE
    TIMER = 1
    if LINE is not None:
        global COL, COL_POS, LINE_POS
        self.set_ram(10, COL_POS[COL])
        self.set_ram(16, LINE_POS[LINE])
    self.set_ram(19, 0)
    self.set_ram(117, 0)
    self.set_ram(119, 134)
    self.set_ram(120, 144)
    self.set_ram(123, LIVES)


def mini_maze_man(self):
    ram = self.get_ram()

    # check if pill was collected
    collected = True

    # checks the pill states, set collected to false if there is one remaining on the map
    for i in range(59, 101):
        if ram[i] != 0:
            collected = False

    # if no pill on the map
    global COL, LINE
    if collected and ram[39] > 70:
        # increase the timer (inplace of score)
        add = ram[120] + 16
        if add <= 144:
            self.set_ram(120, add)
        else:
            self.set_ram(121, ram[121]+1)
            self.set_ram(120, 0)

        # better choice alogorithm from endgame
        global  DOT_POS
        line = choice(range(LINE*5, ((LINE+1)*5)-1))
        dot = choice(range(DOT_POS[COL][0], DOT_POS[COL][1]))
        if not GRID1[line][dot]:
            while not GRID1[line][dot]:
                dot = choice(range(DOT_POS[COL][0], DOT_POS[COL][1]))
        pill = DOT_PATTERN[dot][0] + 3*line
        value = ram[pill] + DOT_PATTERN[dot][1]
        self.set_ram(pill, value)

    # change or reset level on success/failure
    global LVL_NUM, TIMER, LIVES
    if ram[39] == 69:
        if ram[0] == 0 and ram[119] == 154:
            LVL_NUM = 1
            if LIVES < 3:
                LIVES += 1
            self.reset()
        elif ram[0] == 1 and ram[119] == 150:
            LVL_NUM = 2
            if LIVES < 3:
                LIVES += 1
            self.reset()
        elif ram[0] == 2 and ram[119] == 158:
            LVL_NUM = 3
            if LIVES < 3:
                LIVES += 1
            self.reset()
        elif ram[0] == 3 and ram[119] == 154:
            LVL_NUM = 0
            if LIVES < 3:
                LIVES += 1
            self.reset()

    # Use ram state and TIMER variable as tick
    x, y = ram[10] - 13, ram[16]+1
    boundry_x = [0, 60, 100, 160]
    boundry_y = [0, 50, 120, 170]
    if ram[39] == 255 and TIMER == 0:
        # resets the whole game if no lives remaining
        if ram[120] < 16 and ram[121] == 0 and ram[123] <= 0:
            LVL_NUM = 0
            LIVES = 2
            self.reset()
        # decrease lives if 
        elif ram[120] < 16 and ram[121] == 0:
            LIVES-=1
            self.reset()
        # lower timer
        elif not (boundry_x[COL] < x < boundry_x[COL+1] and boundry_y[LINE] < y < boundry_y[LINE+1]):
            if ram[120]-32 < 1:
                if ram[121] > 0:
                    self.set_ram(121, ram[121]-1)
                    if ram[120]:
                        self.set_ram(120, 144)
                    else:
                        self.set_ram(120, 128)
                # Full reset
                elif ram[123] <= 0:
                        LVL_NUM = 0
                        LIVES = 2
                        self.reset()
                # decrease lives
                else:
                    LIVES-=1
                    self.reset()
            else:
                self.set_ram(120, ram[120]-32)
        else:
            # pass
            if ram[120] == 0:
                self.set_ram(121, ram[121]-1)
                self.set_ram(120, 144)
            else:
                self.set_ram(120, ram[120]-16)
            

    # increase timer
    TIMER = (TIMER+1) % 150
    #remove ghosts
    self.set_ram(47, 0)


def _modif_funcs(env, modifs):
    global TOGGLE_CYAN, TOGGLE_PINK, TOGGLE_ORANGE, TOGGLE_RED

    if "edible_ghosts" in modifs and "inverted" in modifs:
        raise ValueError(
            'The modification "ghosts_edible" is unnecessary when playing in inverted mode'
        )
    for mod in modifs:
        if mod == "caged_ghosts":
            TOGGLE_CYAN = True
            TOGGLE_ORANGE = True
            TOGGLE_RED = True
            TOGGLE_PINK = True
            env.step_modifs.append(static_ghosts)
        elif mod == "disable_orange":
            TOGGLE_ORANGE = True
            env.step_modifs.append(static_ghosts)
        elif mod == "disable_red":
            TOGGLE_RED = True
            env.step_modifs.append(static_ghosts)
        elif mod == "disable_cyan":
            TOGGLE_CYAN = True
            env.step_modifs.append(static_ghosts)
        elif mod == "disable_pink":
            TOGGLE_PINK = True
            env.step_modifs.append(static_ghosts)
        elif mod.startswith("power"):
            mod_n = int(mod[-1])
            if mod_n < 0 or mod_n > 4:
                raise ValueError(
                    "Invalid Number of Power Pills, choose number 0-4")
            global NUMBER_POWER_PILLS
            NUMBER_POWER_PILLS = mod_n
            env.reset_modifs.append(number_power_pills)
        elif mod == "edible_ghosts":
            env.step_modifs.append(edible_ghosts)
        elif mod == "inverted":
            env.step_modifs.append(inverted_ms_pacman)
            env.reset_modifs.append(inverted_ms_pacman_reset)
        elif "change_level" in mod:
            if mod[-1].isdigit():
                global LVL_NUM
                LVL_NUM = int(mod[-1])
                assert LVL_NUM < 4, "Invalid Level Number (0, 1, 2 or 3)"
            env.reset_modifs.append(change_level)
        elif mod == "end_game":
            env.reset_modifs.append(end_game)
            env.step_modifs.append(check_reset)
        elif mod == "maze_man":
            env.reset_modifs.append(change_level)
            TOGGLE_CYAN = True
            TOGGLE_ORANGE = True
            TOGGLE_RED = True
            TOGGLE_PINK = True
            env.step_modifs.append(static_ghosts)
            env.step_modifs.append(maze_man)
            env.reset_modifs.append(maze_man_reset)
        elif mod.startswith("mini_maze_man"):
            global COL, LINE
            if mod[-1].isdigit() and int(mod[-1]) < 3:
                COL = int(mod[-1])
            else:
                COL = 2
            if mod[-2].isdigit() and int(mod[-2]) < 3:
                LINE = int(mod[-2])
            else:
                LINE = 2

            TOGGLE_CYAN = True
            TOGGLE_ORANGE = True
            TOGGLE_RED = True
            TOGGLE_PINK = True
            env.step_modifs.append(static_ghosts)
            env.step_modifs.append(mini_maze_man)
            env.reset_modifs.append(maze_man_reset)
