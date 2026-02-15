# train.py
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import time
import random
import copy
from collections import deque

# Импорт наших модулей
from src.game import init_game, make_move, game_over
from src.model import ToguzZeroResNet
from src.mcts import MCTS, encode_board

# --- ГИПЕРПАРАМЕТРЫ ОБУЧЕНИЯ ---
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-4
MCTS_SIMULATIONS = 400   # 400 - хороший баланс скорости и ума
GAMES_PER_LOOP = 20      # Играем 20 партий, потом учимся
EPOCHS_PER_LOOP = 5      # Прогоняем данные 5 раз через сеть
MEMORY_SIZE = 10000      # Помним последние 10k ходов
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "models/toguz_best.pth"
CHECKPOINT_PATH = "models/toguz_checkpoint.pth"

def self_play(mcts):
    """Генерация одной партии игры модели самой с собой."""
    board, kazans, tuz = init_game()
    player = 0
    history = [] # Сохраняем: (board, kazans, tuz, player, probs)
    
    moves_count = 0
    while not game_over(board, kazans) and moves_count < 250:
        # Запускаем MCTS для поиска лучшего распределения вероятностей
        mcts_probs = mcts.search(board, kazans, tuz, player, simulations=MCTS_SIMULATIONS)
        
        # Сохраняем состояние
        history.append([
            copy.deepcopy(board), 
            copy.deepcopy(kazans), 
            copy.deepcopy(tuz), 
            player, 
            mcts_probs
        ])
        
        # Первые 30 ходов добавляем случайность (Exploration), 
        # чтобы модель видела разные дебюты. Дальше - строго (Exploitation).
        if moves_count < 30:
            move = np.random.choice(range(9), p=list(mcts_probs.values()))
        else:
            move = max(mcts_probs, key=mcts_probs.get)
            
        make_move(player, move, board, kazans, tuz)
        player = 1 - player
        moves_count += 1
        
    # Определение результата (1 - выиграл P0, -1 - выиграл P1, 0 - ничья)
    if kazans[0] > kazans[1]: result = 1.0
    elif kazans[1] > kazans[0]: result = -1.0
    else: result = 0.0
    
    # Формируем данные для обучения
    processed_data = []
    for h in history:
        h_board, h_kazans, h_tuz, h_player, h_probs = h
        
        # Value Target: Результат игры с точки зрения текущего игрока
        # Если выиграл P0 (res=1), то для P0 target=1, для P1 target=-1
        value_target = result if h_player == 0 else -result
        
        # Превращаем словарь probs в список из 9 чисел
        policy_target = np.zeros(9, dtype=np.float32)
        for move, prob in h_probs.items():
            policy_target[move] = prob
            
        processed_data.append({
            'board': h_board, 
            'kazans': h_kazans, 
            'tuz': h_tuz, 
            'player': h_player,
            'policy_target': policy_target,
            'value_target': value_target
        })
        
    return processed_data

def train_step(model, optimizer, data_buffer):
    """Один шаг обновления весов нейросети."""
    model.train()
    random.shuffle(data_buffer)
    
    total_loss = 0
    batches = 0
    
    for i in range(0, len(data_buffer), BATCH_SIZE):
        batch = data_buffer[i : i + BATCH_SIZE]
        if len(batch) < BATCH_SIZE // 2: break 
        
        x_list, k_list, p_target_list, v_target_list = [], [], [], []
        
        for item in batch:
            # Подготовка тензоров
            x, k = encode_board(item['player'], item['board'], item['kazans'], item['tuz'], DEVICE)
            x_list.append(x)
            k_list.append(k)
            p_target_list.append(torch.tensor(item['policy_target'], dtype=torch.float32))
            v_target_list.append(torch.tensor([item['value_target']], dtype=torch.float32))
            
        x_batch = torch.cat(x_list).to(DEVICE)
        k_batch = torch.cat(k_list).to(DEVICE)
        p_target = torch.stack(p_target_list).to(DEVICE)
        v_target = torch.stack(v_target_list).to(DEVICE)
        
        optimizer.zero_grad()
        p_pred, v_pred = model(x_batch, k_batch)
        
        # 1. Value Loss (MSE): Насколько мы ошиблись в предсказании победителя?
        loss_v = F.mse_loss(v_pred, v_target)
        
        # 2. Policy Loss (CrossEntropy): Насколько наши вероятности отличаются от MCTS?
        log_probs = F.log_softmax(p_pred, dim=1)
        loss_p = -torch.mean(torch.sum(p_target * log_probs, dim=1))
        
        # Сумма ошибок
        loss = loss_v + loss_p
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        batches += 1
        
    return total_loss / (batches + 1e-8)

def run_training():
    print(f"🚀 Запуск обучения ToguzZero (Device: {DEVICE})")
    
    # Создаем папку models, если нет
    os.makedirs("models", exist_ok=True)
    
    model = ToguzZeroResNet(num_res_blocks=5).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Пытаемся загрузить существующую модель
    start_loop = 1
    if os.path.exists(MODEL_PATH):
        print(f"🔄 Загрузка модели: {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    else:
        print("🆕 Создание новой модели с нуля.")

    mcts = MCTS(model, DEVICE)
    replay_buffer = deque(maxlen=MEMORY_SIZE)
    
    print(f"\nПараметры: MCTS_Sims={MCTS_SIMULATIONS}, Games_Per_Loop={GAMES_PER_LOOP}")
    
    try:
        loop = 0
        while True:
            loop += 1
            start_time = time.time()
            
            # 1. Self-Play
            print(f"[Loop {loop}] Генерация партий...", end=" ", flush=True)
            model.eval()
            new_data = []
            for _ in range(GAMES_PER_LOOP):
                game_data = self_play(mcts)
                new_data.extend(game_data)
                print(".", end="", flush=True)
            
            replay_buffer.extend(new_data)
            print(f" Done. Buffer: {len(replay_buffer)}")
            
            # 2. Training
            if len(replay_buffer) >= BATCH_SIZE:
                print(f"[Loop {loop}] Обучение...", end=" ")
                avg_loss = 0
                for _ in range(EPOCHS_PER_LOOP):
                    loss = train_step(model, optimizer, list(replay_buffer))
                    avg_loss += loss
                
                print(f"Loss: {avg_loss/EPOCHS_PER_LOOP:.4f}")
                
                # Сохраняем модель
                torch.save(model.state_dict(), MODEL_PATH)
                # Можно сохранять и контрольные точки
                # torch.save(model.state_dict(), f"models/toguz_loop_{loop}.pth")
            
            print(f"Время цикла: {time.time() - start_time:.1f} сек.\n")
            
    except KeyboardInterrupt:
        print("\n🛑 Обучение остановлено пользователем.")
        print("Модель сохранена.")
        torch.save(model.state_dict(), MODEL_PATH)

if __name__ == "__main__":
    run_training()