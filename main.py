# main.py
import torch
import sys
import os
from src.game import init_game, make_move, game_over, print_board
from src.model import ToguzZeroResNet
from src.mcts import MCTS

# Настройки
MODEL_PATH = "models/toguz_best.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SIMULATIONS = 800  # Сложность: 400 (Быстро), 800 (КМС), 1600 (Мастер)

def get_human_move(player, board):
    """Безопасный ввод хода для игрока."""
    valid_moves = [i for i in range(9) if board[player][i] > 0]
    while True:
        try:
            user_input = input(f"👤 Ваш ход { [x+1 for x in valid_moves] } (q - выход): ")
            if user_input.lower() in ['q', 'exit', 'quit']:
                print("Выход из игры.")
                sys.exit(0)
            
            move = int(user_input) - 1
            if move in valid_moves:
                return move
            print("❌ Невозможный ход.")
        except ValueError:
            print("❌ Введите число.")
        except KeyboardInterrupt:
            print("\nИгра прервана.")
            sys.exit(0)

def main():
    print(f"🚀 Запуск ToguzZero AI на устройстве: {DEVICE}")
    
    # 1. Загрузка модели
    model = ToguzZeroResNet(num_res_blocks=5).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            model.eval()
            print(f"✅ Модель успешно загружена: {MODEL_PATH}")
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            return
    else:
        print(f"❌ Файл модели не найден: {MODEL_PATH}")
        print("Пожалуйста, поместите файл .pth в папку models/")
        return

    # 2. Инициализация MCTS
    mcts = MCTS(model, DEVICE)
    board, kazans, tuz = init_game()

    # 3. Выбор стороны
    print("\nДобро пожаловать в Тогызкумалак!")
    while True:
        choice = input("Вы играете Белыми (1) или Черными (0)? (q - выход): ")
        if choice in ['1', '0']:
            human_side = 0 if choice == '1' else 1
            break
        if choice.lower() == 'q': return

    ai_side = 1 - human_side
    player = 0
    print_board(board, kazans, tuz)

    # 4. Игровой цикл
    while not game_over(board, kazans):
        if player == human_side:
            move = get_human_move(player, board)
        else:
            print(f"🤖 AI думает ({SIMULATIONS} вариантов)...", end="", flush=True)
            probs = mcts.search(board, kazans, tuz, player, simulations=SIMULATIONS)
            best_move = max(probs, key=probs.get)
            print(f" Ход: {best_move+1} (Уверенность: {probs[best_move]*100:.1f}%)")
            move = best_move

        make_move(player, move, board, kazans, tuz)
        print_board(board, kazans, tuz)
        player = 1 - player

    # 5. Итоги
    print("===== ИГРА ОКОНЧЕНА =====")
    print(f"СЧЕТ: ЧЕЛОВЕК {kazans[human_side]} - {kazans[ai_side]} AI")
    if kazans[ai_side] > kazans[human_side]:
        print("🤖 ПОБЕДА ИСКУССТВЕННОГО ИНТЕЛЛЕКТА!")
    elif kazans[human_side] > kazans[ai_side]:
        print("👤 ПОБЕДА ЧЕЛОВЕКА!")
    else:
        print("🤝 НИЧЬЯ!")

if __name__ == "__main__":
    main()