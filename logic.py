import csv
import os
import random
from datetime import datetime

class Logger:
    def __init__(self, file_name='game_log.csv'):
        self.file_name = file_name
        self.fields = ['timestamp', 'player1', 'player2', 'winner', 'moves', 'first_move_type']

        if not os.path.isfile(self.file_name):
            with open(self.file_name, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.fields)
                writer.writeheader()

    def log_game(self, player1, player2, winner, moves, first_move_type):
        with open(self.file_name, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fields)
            writer.writerow({
                'timestamp': datetime.now(),
                'player1': player1,
                'player2': player2,
                'winner': winner,
                'moves': moves,
                'first_move_type': first_move_type,
            })

class Player:
    def __init__(self, symbol):
        self.symbol = symbol

    def move(self, board):
        pass

class HumanPlayer(Player):
    def move(self, board):
        while True:
            try:
                move = input(f"{self.symbol}'s turn. Enter your move (row,col): ")
                row, col = map(int, move.split(","))
                if 0 <= row < 3 and 0 <= col < 3 and board[row][col] is None:
                    return (row, col)
                else:
                    print("Invalid move. Try again.")
            except ValueError:
                print("Invalid input. Enter the move as row,col. Example: 1,2")

class BotPlayer(Player):
    def move(self, board):
        while True:
            row, col = random.randint(0, 2), random.randint(0, 2)
            if board[row][col] is None:
                return (row, col)

class Game:
    def __init__(self, player1, player2):
        self.board = self.make_empty_board()
        self.current_player = player1
        self.other_player = player2
        self.logger = Logger()
        self.move_count = 0
        self.first_move_type = None

    def make_empty_board(self):
        return [[None, None, None] for _ in range(3)]

    def get_winner(self):
        board = self.board
        for i in range(3):
            if board[i][0] == board[i][1] == board[i][2] and board[i][0] is not None:
                return board[i][0]
            if board[0][i] == board[1][i] == board[2][i] and board[0][i] is not None:
                return board[0][i]
        if board[0][0] == board[1][1] == board[2][2] and board[0][0] is not None:
            return board[0][0]
        if board[0][2] == board[1][1] == board[2][0] and board[0][2] is not None:
            return board[0][2]
        return None

    def is_draw(self):
        return all(cell is not None for row in self.board for cell in row) and self.get_winner() is None

    def display_board(self):
        for row in self.board:
            print("|".join([cell if cell is not None else " " for cell in row]))
            print("-" * 5)

    def play_turn(self, player):
        row, col = player.move(self.board)
        self.board[row][col] = player.symbol
        if self.move_count == 0: 
            if (row, col) in [(0, 0), (0, 2), (2, 0), (2, 2)]:
                self.first_move_type = 'corner'
            elif (row, col) == (1, 1):
                self.first_move_type = 'center'
            else:
                self.first_move_type = 'edge'

    def switch_player(self):
        self.current_player, self.other_player = self.other_player, self.current_player

    def play(self):
        winner = None
        while winner is None and not self.is_draw():
            self.display_board()
            self.play_turn(self.current_player)
            self.move_count += 1
            winner = self.get_winner()
            if winner is None and not self.is_draw():
                self.switch
