import unittest
from tictactoe import Player, HumanPlayer, BotPlayer, Game

class TestTicTacToe(unittest.TestCase):
    
    def test_empty_board(self):
        game = Game(Player("X"), Player("O"))
        self.assertEqual(game.make_empty_board(), [[None, None, None], [None, None, None], [None, None, None]])
    
    def test_bot_move(self):
        bot = BotPlayer("O")
        board = [[None, None, None], [None, None, None], [None, None, None]]
        row, col = bot.move(board)
        self.assertIn(row, [0, 1, 2])
        self.assertIn(col, [0, 1, 2])
        self.assertIsNone(board[row][col])

    def test_winner_rows(self):
        player1 = Player("X")
        player2 = Player("O")
        game = Game(player1, player2)
        game.board = [["X", "X", "X"], [None, None, None], [None, None, None]]
        self.assertEqual(game.get_winner(), "X")

    def test_winner_columns(self):
        player1 = Player("X")
        player2 = Player("O")
        game = Game(player1, player2)
        game.board = [["X", None, None], ["X", None, None], ["X", None, None]]
        self.assertEqual(game.get_winner(), "X")

    def test_winner_diagonals(self):
        player1 = Player("X")
        player2 = Player("O")
        game = Game(player1, player2)
        game.board = [["X", None, None], [None, "X", None], [None, None, "X"]]
        self.assertEqual(game.get_winner(), "X")

    def test_no_winner(self):
        player1 = Player("X")
        player2 = Player("O")
        game = Game(player1, player2)
        game.board = [["X", "O", "X"], ["O", "X", "O"], ["O", "X", "O"]]
        self.assertIsNone(game.get_winner())

if __name__ == '__main__':
    unittest.main()
