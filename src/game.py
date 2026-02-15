# src/game.py

def init_game():
    """Инициализация доски: 9 лунок по 9 камней у каждого, 0 в казанах."""
    return [[9]*9, [9]*9], [0,0], [None,None]

def next_pos(side, index):
    """Определяет следующую лунку при ходе."""
    if index == 8: 
        return 1 - side, 0
    else: 
        return side, index + 1

def can_make_tuz(player, pit, tuz):
    """Проверка правил создания Туздыка."""
    opponent = 1 - player
    # Нельзя ставить туздык, если он уже есть
    if tuz[player] is not None: return False
    # Нельзя ставить в 9-ю лунку
    if pit == 8: return False
    # Нельзя ставить в туздык соперника (симметричный)
    if tuz[opponent] is not None and tuz[opponent] == pit: return False
    return True

def make_move(player, pit, board, kazans, tuz):
    """Выполняет ход и обновляет состояние доски."""
    stones = board[player][pit]
    if stones == 0: return False
    
    if stones == 1:
        board[player][pit] = 0
        stones_to_distribute = 1
        side, index = next_pos(player, pit)
    else:
        board[player][pit] = 1
        stones_to_distribute = stones - 1
        side, index = next_pos(player, pit)

    last_side, last_index = side, index
    while stones_to_distribute > 0:
        # Если попали в туздык
        if (side == 1 and tuz[0] == index): 
            kazans[0] += 1
        elif (side == 0 and tuz[1] == index): 
            kazans[1] += 1
        else: 
            board[side][index] += 1
        
        stones_to_distribute -= 1
        if stones_to_distribute > 0:
            side, index = next_pos(side, index)
            last_side, last_index = side, index

    opponent = 1 - player
    
    # Если последний камень попал в туздык
    if (last_side == 1 and tuz[0] == last_index) or \
       (last_side == 0 and tuz[1] == last_index):
        return True

    # Правила взятия камней и объявления туздыка
    if last_side == opponent:
        count = board[last_side][last_index]
        if count % 2 == 0:
            kazans[player] += count
            board[last_side][last_index] = 0
        elif count == 3:
            if can_make_tuz(player, last_index, tuz):
                tuz[player] = last_index
                kazans[player] += 3
                board[last_side][last_index] = 0
    return True

def game_over(board, kazans):
    """Проверка окончания игры."""
    if kazans[0] >= 82 or kazans[1] >= 82: return True
    
    # Атсыз қалу (остался без коня/ходов)
    if all(x==0 for x in board[0]):
        kazans[1] += sum(board[1])
        board[1] = [0]*9
        return True
    if all(x==0 for x in board[1]):
        kazans[0] += sum(board[0])
        board[0] = [0]*9
        return True
    return False

def print_board(board, kazans, tuz):
    """Красивый вывод доски в консоль."""
    row1_visual = list(reversed(board[1]))
    tuz_str = ["-", "-"]
    if tuz[0] is not None: tuz_str[0] = str(tuz[0]+1)
    if tuz[1] is not None: tuz_str[1] = str(tuz[1]+1)
    
    print("\n" + "░"*50)
    print(f" P1 (OPPONENT)| Казан: {kazans[1]:<3} | Туздык: {tuz_str[1]}")
    print(f" Лунки (9-1)  : {row1_visual}")
    print("-" * 50)
    print(f" Лунки (1-9)  : {board[0]}")
    print(f" P0 (AI MODEL)| Казан: {kazans[0]:<3} | Туздык: {tuz_str[0]}")
    print("░"*50)