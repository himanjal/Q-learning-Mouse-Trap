import sys
import os

import random
import numpy as np
from time import sleep

# Board variables
START = 0
CHEESE = 1
TRAP = 2
BLOCK = 3
GOAL = 4
MOUSE = 5
NUMMOVES = 200000

# Reward constants
CHEESE_REWARD = 6
TRAP_REWARD = -5
GOAL_REWARD = 5

# Move action variables
UP = (-1, 0)
DN = (1, 0)
RT = (0, 1)
LF = (0, -1)
MOVES = [UP, DN, RT, LF]

class Square:
	up = 0
	down = 0
	left = 0
	right = 0

	def __repr__(self):
		return 'SQR(%s, %s, %s, %s)' % (self.up, self.right, self.down, self.left)

def clear():
    os.system('cls' if os.name=='nt' else 'clear')

def generate_board1():
	board =  [[0,3,0,4,0],
	         [0,1,2,2,0],
	         [0,0,3,1,0],
	         [2,0,0,2,0],
	         [3,1,0,0,0]]
	return (board, len(board), len(board[0]))

def generate_board2():
	board =  [[0,3,1,3,2,0,3,1,3,2],
	          [0,3,0,0,0,0,0,0,2,0],
	          [0,3,0,2,0,0,0,3,1,0],
	          [0,0,0,2,0,0,3,1,3,2],
	          [1,3,2,3,2,0,3,0,0,0],
	          [0,0,0,0,0,2,0,2,2,0],
			  [0,1,0,2,0,2,0,2,2,0],
			  [3,2,0,1,0,0,0,0,2,0],
			  [2,0,2,2,0,2,3,4,0,3],
			  [3,1,0,0,0,3,1,0,0,0]]

	return (board, len(board), len(board[0]))

def generate_board3():
	board =  [[0,0,0,0,0],
	          [0,1,0,2,0],
	          [3,2,0,1,0],
	          [2,0,2,2,0],
	          [3,1,0,0,4]]
	return (board, len(board), len(board[0]))

def generate_board():
  """ Used to generate a random board with cheese, mouse traps, blocks, and exit door
  
  Return: 
	   a board made of 2D array, and the height and width
  """
  width = random.randint(2,30)
  #height = random.randint(2,10)
  height = width
  board = [[0 for x in range(width)] for y in range(height)]
  
  #assign random nos from (0,3) to all places in board  
  for w in range(height):
	for h in range(width):
		x = random.randint(0, 3)
		if((x==3 or x == 2) and random.randint(0,100) > 70):
			x = 0
		board[w][h] = x 
	  
  #marking origin as safe position
  board[0][0] = 0

  #assign one random place in board as finish Goal
  board[random.randint(1,height-1)][random.randint(1,width-1)] = 4

  #if no moves allowed initially then generate new board
  if board[1][0] == 2 or board[1][0] == 3:
	if board[0][1] == 2 or board[0][1] == 3:
	  board, height, width = generate_board()
	  
  return board, height, width

def q_init(h, w):
	"""Creates a Q with Square all set to 0
	Args:
		h (int): Height of the Q we want to generate
		w (int): Width of the Q

	Returns:
		An HxW matrix of Squares
	"""
	q = []
	for x in range(h):
		q.append([])
		for y in range(w):
			q[x].append(Square())
	return q
   
def valid_moves(mouse):
	"""Used to find valid moves for a mouse on a given board.

	Return:
		An array of possible moves.
	"""
	width = len(BOARD[0])
	height = len(BOARD)
	valid = []
	
	#check what moves don't go over the edge or hit a wall
	for mv in MOVES:
		x = mouse[0] + mv[0]
		y = mouse[1] + mv[1]
		if 0 <= x <= width - 1 and 0 <= y <= height - 1:
			if BOARD[x][y] != BLOCK:
				valid.append(mv)

	return valid

def get_reward(state, action):
	#TODO find reward from Q or Board?
	"""Finds the reward for the from the current position making the given move
	Args:
		state (int tuple): the current position
		action (int tuple): the move to make

	Returns:
		the reward value form the board
	"""
	newState = state
	newState = (newState[0] + action[0] , newState[1] + action[1])
	rewardValue = 0

	currentPossition = BOARD[ newState[0] ][ newState[1] ]

	if currentPossition == CHEESE:
		rewardValue = CHEESE_REWARD
	elif currentPossition == TRAP:
		rewardValue = TRAP_REWARD	
	elif currentPossition == GOAL:
		rewardValue = GOAL_REWARD
	else: # No reward given
		rewardValue = 0

	return rewardValue

def update_q(state, action):
	"""Q(state, action) = R(state, action) + Gamma * Max[Q(next state, all actions)]

	Args:
		state (int tuple): The current position
		action (int tuple): The move we want to make
	"""
	GAMMA = 0.5
	#What are all possible squares we could get to and what move gets us there 
	#((destination_state tuple), (move tuple))
	possible = []
	"""
	for m in valid_moves(state):
		next_sqr = (state[0] + m[0], state[1] + m[1])
		possible.append((next_sqr, m))
	"""
	next_state = (state[0] + action[0], state[1] + action[1])

	for m in valid_moves(next_state):
		if m == UP:
			possible.append(Q[next_state[0]][next_state[1]].up)
		elif m == DN:
			possible.append(Q[next_state[0]][next_state[1]].down)
		elif m == RT:
			possible.append(Q[next_state[0]][next_state[1]].right)
		elif m == LF:
			possible.append(Q[next_state[0]][next_state[1]].left)

	MAX = max(possible)

	if MAX == 0:
		while True:
			MAX = possible[random.randint(0,len(possible)-1)]
			if(MAX == 0):
				break

	#Best move
	#TODO Max[Q(next state, all actions)] what is this
	#MAX = max(map(lambda s: get_reward(state, s[1]), possible))
	#Reward
	R = get_reward(state, action)
	new_q = R + GAMMA * MAX

	if action == UP:
		Q[state[0]][state[1]].up = new_q
	elif action == DN:
		Q[state[0]][state[1]].down = new_q
	elif action == RT:
		Q[state[0]][state[1]].right = new_q
	elif action == LF:
		Q[state[0]][state[1]].left = new_q

def symbolConvert(val):
	"""
	Converts board variables into graphic symbols
	"""
	if val == BLOCK:
		return ' '
	elif val == CHEESE:
		return '+'
	elif val == TRAP:
		return 'x'
	elif val == START:
		return '.'
	elif val == GOAL:
		return 'G'
	elif val == MOUSE:
		return 'M'
	else:
		return val

def printLegend():
	"""
	Print the legend
	"""
	print "MAP LEGEND"
	print " '", symbolConvert(MOUSE), "': mouse"
	print " '", symbolConvert(BLOCK), "': no possible paths (barrier)"
	print " '", symbolConvert(CHEESE), "': cheese"
	print " '", symbolConvert(TRAP), "': trap"
	print " '", symbolConvert(START), "': normal path"
	print " '", symbolConvert(GOAL), "': goal!!!"
	print

def print_move(move, mouse):
	if move == UP:
		print "UP"
	elif move == DN:
		print "DOWN"
	elif move == RT:
		print "RIGHT"
	elif move == LF:
		print "LEFT"

def printBoard(board, mouse):
	"""
	Print the board
	"""
	width = len(board[0])
	height = len(board)
	print
	print '  ',
	for w in xrange(-1, width):
		print '-',
	print

	for h in xrange(0, height):
		print '  |',
		for w in xrange(0, width):
			val = board[h][w]
			if backupBoard[h][w] == CHEESE:
				val = CHEESE
			if h == mouse[0] and w == mouse[1]:
				val = MOUSE

			print symbolConvert(val),

			
		print '|',
		print

	print '  ',
	for w in xrange(-1, width):
		print '-',
	print
	print

def get_updated_mouse(mouse, move):
	""" Returns the coordinates for a mouse after performing the move.
	"""
	next_mouse = (mouse[0] + move[0], mouse[1] + move[1])
	return next_mouse

def find_best_move(mouse):
	#very small chance of making a random possible move to avoid local maxima
	if random.random() < 0.01:
		return random.choice(valid_moves(mouse))
	possible = []
	for mv in valid_moves(mouse):
		if mv == UP:
			possible.append(Q[mouse[0]][mouse[1]].up)
		elif mv == DN:
			possible.append(Q[mouse[0]][mouse[1]].down)
		elif mv == RT:
			possible.append(Q[mouse[0]][mouse[1]].right)
		elif mv == LF:
			possible.append(Q[mouse[0]][mouse[1]].left)
	#best attainable q value
	qval = max(possible)
	moves = []
	validMoves = valid_moves(mouse)
	#If all moves are equal choose a random one
	if qval == 0:
		if(Q[mouse[0]][mouse[1]].up == 0) and UP in validMoves:
			moves.append(UP)
		if Q[mouse[0]][mouse[1]].down == 0 and DN in validMoves:
			moves.append(DN)
		if Q[mouse[0]][mouse[1]].right == 0 and RT in validMoves:
			moves.append(RT)
		if Q[mouse[0]][mouse[1]].left == 0 and LF in validMoves:
			moves.append(LF)
		return moves[random.randint(0,len(moves)-1)]
	#return best move
	if qval == Q[mouse[0]][mouse[1]].up and UP in validMoves:
		return UP
	elif qval == Q[mouse[0]][mouse[1]].down and DN in validMoves:
		return DN
	elif qval == Q[mouse[0]][mouse[1]].right and RT in validMoves:
		return RT
	elif qval == Q[mouse[0]][mouse[1]].left and LF in validMoves:
		return LF

def printAll(next_move, mouse, board):
	clear()
	printBoard(board, mouse)
	print_move(next_move, mouse)
	print
	printLegend()
	print


def learn(board, mouse):
	"""Trains our mouse and runs the program."""
	i = 0
	moves = []

	starting_board = board
	solvable = False
	while i < NUM_LESSONS:
		if (board[mouse[0]][mouse[1]] == TRAP):
			# print "Died after %d moves " % len(moves)
			moves = []
			mouse = (0, 0)
			board = starting_board
		elif(board[mouse[0]][mouse[1]] ==  GOAL):
			#reset variables for next lesson
			solvable = True
			# print "Found solution in %d moves " % len(moves)
			moves = []
			mouse = (0, 0)
			board = starting_board
		else:
			next_move = find_best_move(mouse)
			moves.append(next_move)
			update_q(mouse, next_move)
			#eat cheese after reaching it until next lesson
			if board[mouse[0]][mouse[1]] == CHEESE:
				BOARD[mouse[0]][mouse[1]] = START
			mouse = get_updated_mouse(mouse, next_move)
		i += 1

	mouse = (0, 0)
	moves = []
	alive = solvable
	while alive:
		alive = board[mouse[0]][mouse[1]] != GOAL and board[mouse[0]][mouse[1]] != TRAP
		print "square = " + str(board[mouse[0]][mouse[1]])
		next_move = find_best_move(mouse)
		printAll(next_move, mouse, board)
		sleep(.35)
		moves.append(next_move)
		update_q(mouse, next_move)
		if backupBoard[mouse[0]][mouse[1]] == CHEESE:
			backupBoard[mouse[0]][mouse[1]] = START
		mouse = get_updated_mouse(mouse, next_move)
	if not solvable:
                #printAll(next_move, mouse, board)
		#print "   No solution found"
		clear()
		print
		print
		return 1
	return 0

result = 1
while(result == 1):
	#The random board for our program
	BOARD, H, W = generate_board()
	backupBoard = [[0 for x in range(W)] for y in range(H)]
	for x in range(len(BOARD)):
		for y in range(len(BOARD[0])):
			backupBoard[x][y] = BOARD[x][y]

	# initialize our Q matrix
	Q = q_init(H, W)
	clear()
	printBoard(BOARD, (0,0))
	print
	#Learn the board
	NUM_LESSONS = 100000
	result = learn(BOARD, (0, 0))







