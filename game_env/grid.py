from configs import *
import numpy as np
from .player import *
import random

class Grid(object):
    """ Represent the playing grid of the game"""

    def __init__(self, game_map=None):
        """
        Create a new game grid.

        :param game_map: a list of strings representing the grid object (one string each row).
        """

        self.game_map = game_map

        # 2D numpy array
        self._cells = None

        self._beam_cells = set()  # Represents the beam cells in the grid
        self.beam_area = set()  # Represent the beam cells including the player but not shown in the grid
        self._empty_cells = set()
        self._map_to_cell_type = {
            'P': CellType.PLAYER,
            'f': CellType.PLAYER_FRONT,
            '#': CellType.WALL,
            'A': CellType.APPLE,
            '.': CellType.EMPTY,
            '-': CellType.BEAM,
            'O': CellType.OPPONENT,
            'G': CellType.AGENT,
        }
        self._cell_type_to_map = {
            cell_type: symbol
            for symbol, cell_type in self._map_to_cell_type.items()
        }

    def __getitem__(self, point):
        x, y = point
        return self._cells[y, x]

    def __setitem__(self, point, cell_type):
        """ Update the type of cell at the given point. """
        x, y = point
        self._cells[y, x] = cell_type

        # Do some internal bookkeeping to not rely on random selection of blank cells.
        if cell_type == CellType.BEAM:
            self._beam_cells.add(point)
        else:
            if point in self._beam_cells:
                self._beam_cells.remove(point)

        if cell_type == CellType.EMPTY:
            self._empty_cells.add(point)
        else:
            if point in self._empty_cells:
                self._empty_cells.remove(point)

    def __str__(self):
        return '\n'.join(
            ''.join(self._cell_type_to_map[cell] for cell in row)
            for row in self._cells
        )

    @property
    def width(self):
        return len(self.game_map[0])

    @property
    def height(self):
        return len(self.game_map)

    def create_grid(self):
        """
        Create the 2D numpy cells from the game map
        """
        self._cells = np.array([
            [self._map_to_cell_type[symbol] for symbol in line]
            for line in self.game_map
        ])

        self._empty_cells = {
            Point(x, y)
            for y in range(self.height)
            for x in range(self.width)
            if self[(x, y)] == CellType.EMPTY
        }

    def get_grid(self):
        """
        :return: The 2d grid 
        """
        return np.copy(self._cells)

    def get_beam_area(self):
        return self.beam_area.copy()

    def find_player(self):
        """ 
        Find the snake's head on the field. 
        
        :return: the location list of all players
        """
        point_list = []
        for y in range(self.height):
            for x in range(self.width):
                if self[(x, y)] == CellType.PLAYER:
                    point_list.append(Point(x, y))
        return point_list

    def place_player(self, player: Player):
        self[player.position] = CellType.PLAYER
        self.update_front_of_player(player)

    def clear_player(self, player: Player):
        self[player.position] = CellType.BEAM
        # self[player.position] = CellType.EMPTY
        if self[player.current_front] == CellType.PLAYER_FRONT:
            self[player.current_front] = CellType.EMPTY

    def remove_player(self, player: Player):
        if self[player.current_front] == CellType.PLAYER_FRONT:
            self[player.current_front] = CellType.EMPTY

    def update_front_of_player(self, player):
        front_position = player.current_front
        if self[front_position] not in [CellType.APPLE, CellType.WALL, CellType.BEAM, CellType.PLAYER]:
            self[front_position] = CellType.PLAYER_FRONT

    def place_apples(self, apple_list):
        for apple in apple_list:
            if not apple.is_collected:
                if self[apple.position] not in [CellType.PLAYER, CellType.WALL]:
                    self[apple.position] = CellType.APPLE

    def add_beam_area(self, x, y):
        self.beam_area.add(Point(x, y))
        if self[x, y] in (CellType.PLAYER, CellType.EMPTY, CellType.PLAYER_FRONT):
            self[x, y] = CellType.BEAM

    def empty_beam_area(self, x, y):
        if self[x, y] == CellType.BEAM:
            self.beam_area.remove(Point(x, y))
            self[x, y] = CellType.EMPTY

    def place_beam_area(self, player: Player):
        """
            Place the beam area according to the player's direction
        :param player: The player in the grid
        """
        if player.direction == PlayerDirection.NORTH:
            for yy in np.arange(player.position.y):
                self.add_beam_area(player.position.x, yy)
        elif player.direction == PlayerDirection.SOUTH:
            for yy in np.arange(player.position.y+1, self.height):
                self.add_beam_area(player.position.x, yy)
        elif player.direction == PlayerDirection.EAST:
            for xx in np.arange(player.position.x+1, self.width):
                self.add_beam_area(xx, player.position.y)
        else:
            for xx in np.arange(player.position.x):
                self.add_beam_area(xx, player.position.y)

    def clear_beam(self, player):
        if player.direction == PlayerDirection.NORTH:
            for yy in np.arange(player.position.y):
                self.empty_beam_area(player.position.x, yy)
        elif player.direction == PlayerDirection.SOUTH:
            for yy in np.arange(player.position.y+1, self.height):
                self.empty_beam_area(player.position.x, yy)
        elif player.direction == PlayerDirection.EAST:
            for xx in np.arange(player.position.x+1, self.width):
                self.empty_beam_area(xx, player.position.y)
        else:
            for xx in np.arange(player.position.x):
                self.empty_beam_area(xx, player.position.y)

    def clear_beam_area(self):
        """
            Clear the stored beam area
        """
        # To avoid python error of "set changed size during iteration"
        self.beam_area.clear()
        point_list = []
        for point in self._beam_cells:
            point_list.append(point)
        for point in point_list:
            self[point] = CellType.EMPTY

    def is_in_beam_area(self, position):
        if position in self.beam_area:
            return True
        else:
            return False

    def copy_cells(self):
        """        
        :return: Return the numpy array of the 2d cells
        """
        return np.copy(self._cells)

    def generate_random_player_position(self, num_player):
        return random.sample(self._empty_cells, num_player)




