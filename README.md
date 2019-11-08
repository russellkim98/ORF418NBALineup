# ORF418NBALineup
Final Project for ORF418, a new statistical model for the efficiency of an NBA Lineup - using RL methods.

Sports analytics has taken off rapidly in the past few years, and special attention is being paid to the NBA. The fast paced game of basketball lends itself well to statistical analysis, and I propose a new model of NBA lineups. Typically speaking, the world of NBA analytics has generally focused on what the best combination of 5 players is. For example, the statistic ”Offensive Rating” of a lineup attempts to judge how offensively successful a lineup of 5 players is. However, as we see with a team like the Cleveland Cavaliers, there could be a situation in which the starting 5 players are exceptional players, but then the bench consists of inconsistent players. Therefore, I propose a new model that measures the overall ”goodness” of a 10 player subset out of 15. The top five bench players are a massive contribution to any team, so this may give a better idea of who the best 10 players on a team are. Since this is a subset selection online learning problem, I will simulate an entire season.