from .environment import *

class EnvironmentGathering(EnvironmentBase):

    def __init__(self, config, n_tag, n_apple):
        super(EnvironmentGathering, self).__init__(config, n_tag)
        self.apple_list = []
        self.N_APPLE = n_apple


    def new_episode(self):
        """Reset the environment and begin a new episode"""
        self.player_list = []
        self.apple_list = []
        self.time_watch.reset()
        self.grid.create_grid()
        # self.stats.reset()
        for point in self.grid.find_player():
            self.player_list.append(Player(point))
            self.player_list[-1].initial_position = point

        idx = 1
        for player in self.player_list:
            self.grid.place_player(player)
            player.new_episode()
            player.idx = idx
            idx += 1
        self.generate_apples()
        self.grid.place_apples(self.apple_list)
        # self.current_action = None
        self.is_game_over = False
        self.get_observation()



    def generate_apples(self, size=3, start=np.array([4,14])):
        """
        add apples of the diamond shape with given size
        :param size: the size of diamond
        :param start: the starting point of the diamond in 
                        the left-bottom corner
        :return: the added apple
        """
        l = size * 2 - 1
        top = start + l - 1
        for idx in range(size - 1):
            for i in range(idx * 2 + 1):
                y = top[0] - idx
                x = start[1] + size - 1 - idx + i
                self.apple_list.append(Apple(Point(x, y)))
        for idx in range(size - 1, -1, -1):
            for i in range(idx * 2 + 1):
                y = start[0] + idx
                x = start[1] + size - 1 - idx + i
                self.apple_list.append(Apple(Point(x, y)))    

    def update_grid(self, player):
        """
        In this method, we assume the next position/direction of the player
        is valid and the player and apples will be placed on the grid
        """

        # Clear the cell for the front of the player
        if self.grid[player.current_front] == CellType.PLAYER_FRONT:
            self.grid[player.current_front] = CellType.EMPTY

        # Clear the cell for the current position of the player
        self.grid[player.position] = CellType.EMPTY

        # Move the player
        player.move()

        # Place the apples
        self.grid.place_apples(self.apple_list)

        # Place the player in the new position
        self.grid.place_player(player)



    def move(self, step):
        """
        In this method, the player is moved to the next position it should be 
        Any reward and beam detection is happened here
        """
        self.grid.clear_beam_area()
        self.update_front_of_players()
        self.respawn_apples(step)
        self.respawn_player(step)
        for player in self.player_list:
            player.reward = self.rewards['timestep']
            if not player.is_tagged:
                self.check_next_position(player)
                self.update_grid(player)
                self.collect_apple(player, step)
        self.check_if_using_beam(step)
        self.get_observation()

    def respawn_apples(self, step):
        """
        the valid apple will be respawn in this method
        """
        for apple in self.apple_list:
            if apple.is_collected:
                if step - apple.collected_time \
                            >= self.N_APPLE:
                    apple.respawn()

    def collect_apple(self, player, step):
        """
        check if the player is about to collect any apple
        """
        # A flag for the player collecting any apple
        eaten_any_apple = False
        for apple in self.apple_list:
            if not apple.is_collected and apple.position == player.position and not player.is_tagged:
                apple.get_collected(step)
                player.apple_eaten += 1
                if not DQNSetting.NOISY:
                    # Without noise
                    player.reward = self.rewards['ate_fruit']
                else:
                    # With noise
                    sample = random.random()
                    if sample > DQNSetting.P_NOISY:
                        # positive reward
                        player.reward = 2.0 / (1 - DQNSetting.P_NOISY)
                    else:
                        # Negative reward
                        player.reward = -1.0 / DQNSetting.P_NOISY
                eaten_any_apple = True
                break
        if not eaten_any_apple:
            player.reward = 0
