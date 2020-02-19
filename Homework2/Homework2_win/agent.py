import random, re, datetime


class Agent(object):
    def __init__(self, game):
        self.game = game

    def getAction(self, state):
        raise Exception("Not implemented yet")


class RandomAgent(Agent):
    def getAction(self, state):
        legal_actions = self.game.actions(state)
        self.action = random.choice(legal_actions)


class SimpleGreedyAgent(Agent):
    # a one-step-lookahead greedy agent that returns action with max vertical advance
    def getAction(self, state):
        legal_actions = self.game.actions(state)

        self.action = random.choice(legal_actions)

        player = self.game.player(state)
        if player == 1:
            max_vertical_advance_one_step = max([action[0][0] - action[1][0] for action in legal_actions])
            max_actions = [action for action in legal_actions if
                           action[0][0] - action[1][0] == max_vertical_advance_one_step]
        else:
            max_vertical_advance_one_step = max([action[1][0] - action[0][0] for action in legal_actions])
            max_actions = [action for action in legal_actions if
                           action[1][0] - action[0][0] == max_vertical_advance_one_step]
        self.action = random.choice(max_actions)


class TeamNameMinimaxAgent(Agent):

    def __init__(self, game, depth=3, evalFunction=None, timing=False):
        self.game = game
        # here the definition of depth is number of layers to visit in the game tree
        self.depth = depth
        self.timing = timing
        if evalFunction is None:
            self.evaluationFunction = self.naiveEvaluationFunction
        else:
            self.evaluationFunction = evalFunction
        if timing:
            self.total_exec_time = datetime.timedelta(0)
            self.num_exec = 0

    def getAction(self, state):
        if self.timing:
            start = datetime.datetime.now()
        legal_actions = self.game.actions(state)
        if not legal_actions:  # stuck
            return None
        player = self.game.player(state)
        alpha = -float('inf')
        beta = float('inf')
        score = [self.alphabeta(self.game.succ(state, action), self.depth - 1, alpha, beta, 3 - player) for action in
                 legal_actions]
        if player == 1:
            max_score = max(score)
            max_actions = [legal_actions[index] for index in range(len(score)) if score[index] == max_score]
            action = random.choice(max_actions)
        else:
            min_score = min(score)
            min_actions = [legal_actions[index] for index in range(len(score)) if score[index] == min_score]
            action = random.choice(min_actions)

        if self.timing:
            end = datetime.datetime.now()
            self.total_exec_time += end - start
            self.num_exec += 1

        return action

    def alphabeta(self, state, depth, alpha, beta, player):
        if self.game.isEnd(state):
            # return self.game.utility(state)
            return self.evaluationFunction(state)
        elif depth == 0:
            return self.evaluationFunction(state)

        if player == 1:
            v = float('-inf')
            legal_actions = self.game.actions(state)
            for action in legal_actions:
                succ = self.game.succ(state, action)
                v = max(v, self.alphabeta(succ, depth - 1, alpha, beta, 3 - player))
                alpha = max(alpha, v)
                if beta <= alpha:
                    # print 'pruned'
                    break
            return v
        else:
            v = float('inf')
            legal_actions = self.game.actions(state)
            for action in legal_actions:
                succ = self.game.succ(state, action)
                v = min(v, self.alphabeta(succ, depth - 1, alpha, beta, 3 - player))
                beta = min(beta, v)
                if beta <= alpha:
                    # print 'pruned'
                    break
            return v

    def naiveEvaluationFunction(self, currentGameState):
        size = self.game.size
        board = currentGameState[1]
        # The Evaluation function considers the sum of distances of each piece to the corner
        # With smaller distance, it is closer to win and the score is higher
        # For Player 1, it is the distance to (1,1), which is just (row - 1)
        player_piece_pos_list1 = board.getPlayerPiecePositions(1)
        p1score = sum([pos[0] - 1 for pos in player_piece_pos_list1])
        # For Player 2, it is the distance to (2*size-1,1)
        player_piece_pos_list2 = board.getPlayerPiecePositions(2)
        p2score = sum([2 * size - 1 - pos[0] for pos in player_piece_pos_list2])
        return p2score - p1score

        ### END CODE HERE ###



