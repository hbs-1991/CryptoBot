#!/usr/bin/env python
"""
Скрипт для запуска симуляции арбитражных сделок.
Обертка над main.py для удобного запуска из командной строки.
"""

import os
import sys
import argparse
import subprocess

def main():
    """
    Парсит аргументы командной строки и запускает симуляцию
    с заданными параметрами.
    """
    # Создаем парсер аргументов командной строки
    parser = argparse.ArgumentParser(description='Crypto Arbitrage Bot Simulation Runner')
    
    parser.add_argument(
        '--duration', 
        type=int, 
        default=300,
        help='Продолжительность симуляции в секундах (по умолчанию: 300)'
    )
    
    parser.add_argument(
        '--debug', 
        action='store_true',
        help='Включить режим отладки с подробным логированием'
    )
    
    parser.add_argument(
        '--live', 
        action='store_true',
        help='Запустить в реальном режиме торговли (ВНИМАНИЕ: использует реальные средства)'
    )
    
    args = parser.parse_args()
    
    # Формируем команду для запуска main.py
    cmd = [sys.executable, 'main.py']
    
    if args.live:
        cmd.append('--live')
    else:
        cmd.append('--simulation')
    
    cmd.append('--duration')
    cmd.append(str(args.duration))
    
    if args.debug:
        cmd.append('--debug')
    
    # Выводим информацию о запуске
    mode = "LIVE TRADING" if args.live else "SIMULATION"
    print(f"Starting Crypto Arbitrage Bot in {mode} mode for {args.duration} seconds")
    print(f"Debug mode: {'ON' if args.debug else 'OFF'}")
    
    if args.live:
        # Запрашиваем подтверждение для реального режима
        confirm = input("WARNING: You are about to run in LIVE trading mode which uses real funds.\n"
                       "Are you sure you want to continue? (yes/no): ")
        if confirm.lower() not in ['yes', 'y']:
            print("Aborted.")
            return 1
    
    # Запускаем процесс
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Error running simulation: {str(e)}")
        return e.returncode
    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
        return 130  # Стандартный код возврата для SIGINT
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
