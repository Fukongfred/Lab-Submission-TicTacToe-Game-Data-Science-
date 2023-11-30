from logic import HumanPlayer, BotPlayer, Game
import pandas as pd

def main():
    mode = input("Choose game mode (1: single player, 2: two players): ")
    player1 = HumanPlayer("X")
    player2 = BotPlayer("O") if mode == "1" else HumanPlayer("O")
    game = Game(player1, player2)
    game.play()

    # Optional: Post-game analysis
    analyze_game_log()

def analyze_game_log():
    try:
        log_df = pd.read_csv('game_log.csv')
        log_df['win'] = log_df['winner'] == log_df['player1']
        log_df['win'] = log_df['win'].astype(int) 
        log_df['moves'] = pd.to_numeric(log_df['moves'])

        players = pd.unique(log_df[['player1', 'player2']].values.ravel('K'))
        win_loss_draw = pd.DataFrame(index=players, columns=['Wins', 'Losses', 'Draws']).fillna(0)

        for player in players:
            win_loss_draw.loc[player, 'Wins'] = log_df[log_df['winner'] == player].shape[0]
            win_loss_draw.loc[player, 'Losses'] = log_df[(log_df['player1'] == player) | (log_df['player2'] == player)].shape[0] - win_loss_draw.loc[player, 'Wins']
            win_loss_draw.loc[player, 'Draws'] = log_df[(log_df['player1'] == player) & (log_df['player2'] != player) & (log_df['winner'] == 'D')].shape[0] + log_df[(log_df['player2'] == player) & (log_df['player1'] != player) & (log_df['winner'] == 'D')].shape[0]

        print(win_loss_draw)

        log_df['timestamp'] = pd.to_datetime(log_df['timestamp'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        log_df.set_index('timestamp', inplace=True)
        monthly_trends = log_df.resample('M').apply(lambda x: x['winner'].value_counts()).fillna(0)
        print("Monthly Trends:\n", monthly_trends)
    except Exception as e:
        print("An error occurred while analyzing the game log:", e)

if __name__ == '__main__':
    main()
