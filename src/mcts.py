# src/mcts.py
import torch
import numpy as np
import math
from src.game import make_move, game_over, next_pos

class MCTSNode:
    def __init__(self, parent=None, prior=0):
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0
        self.prior = prior

    def value(self):
        if self.visit_count == 0: return 0
        return self.value_sum / self.visit_count

def encode_board(player, board, kazans, tuz, device):
    """Превращает доску в тензор для нейросети."""
    opp = 1 - player
    full_board = board[player] + board[opp]
    
    tuz_map = np.zeros(18, dtype=np.float32)
    if tuz[player] is not None and 0 <= tuz[player] < 18: 
        tuz_map[tuz[player]] = 1.0
    if tuz[opp] is not None and 0 <= tuz[opp] < 18: 
        tuz_map[tuz[opp]] = -1.0
            
    x = torch.tensor(np.stack([full_board, tuz_map], axis=0), dtype=torch.float32).unsqueeze(0).to(device) / 20.0
    k = torch.tensor([kazans[player], kazans[opp]], dtype=torch.float32).unsqueeze(0).to(device) / 81.0
    return x, k

class MCTS:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def search(self, board, kazans, tuz, player, simulations=800):
        """Запускает симуляции Монте-Карло для поиска лучшего хода."""
        root = MCTSNode()
        self._expand(root, board, kazans, tuz, player)

        for _ in range(simulations):
            node = root
            # Копируем состояние для симуляции
            s_board = [r[:] for r in board]
            s_kazans = kazans[:]
            s_tuz = tuz[:]
            s_player = player
            
            # 1. Selection
            path = [node]
            while len(node.children) > 0:
                move, node = self._select_child(node)
                make_move(s_player, move, s_board, s_kazans, s_tuz)
                s_player = 1 - s_player
                path.append(node)

            # 2. Expansion & Evaluation
            value = 0
            if not game_over(s_board, s_kazans):
                value = self._expand(node, s_board, s_kazans, s_tuz, s_player)
            else:
                if s_kazans[s_player] > s_kazans[1-s_player]: value = 1.0
                elif s_kazans[s_player] < s_kazans[1-s_player]: value = -1.0
                else: value = 0.0

            # 3. Backpropagation
            for node in reversed(path):
                node.value_sum += value
                node.visit_count += 1
                value = -value

        # Возвращаем вероятности посещения
        counts = {m: node.visit_count for m, node in root.children.items()}
        total = sum(counts.values())
        return {m: c / total for m, c in counts.items()}

    def _select_child(self, node):
        best_score = -float('inf')
        best_move = -1
        best_child = None
        
        for move, child in node.children.items():
            # Формула UCB (PUCT)
            u = 1.0 * child.prior * math.sqrt(node.visit_count) / (1 + child.visit_count)
            # Value храним для того кто ходил, поэтому берем минус для текущего
            score = -child.value() + u
            
            if score > best_score:
                best_score = score
                best_move = move
                best_child = child
        return best_move, best_child

    def _expand(self, node, board, kazans, tuz, player):
        x, k = encode_board(player, board, kazans, tuz, self.device)
        with torch.no_grad():
            logits, value = self.model(x, k)
        
        valid_moves = [i for i in range(9) if board[player][i] > 0]
        
        # Маскировка невозможных ходов
        mask = torch.tensor([-float('inf')] * 9).to(self.device)
        mask[valid_moves] = 0
        probs = torch.softmax(logits[0] + mask, dim=0).cpu().numpy()
        
        for move in valid_moves:
            node.children[move] = MCTSNode(parent=node, prior=probs[move])
            
        return value.item()