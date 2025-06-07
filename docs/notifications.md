# Система уведомлений

## Обзор

Система уведомлений в проекте `crypto_bot` предназначена для информирования пользователя о важных событиях, таких как арбитражные возможности, выполненные сделки, ошибки и общий статус системы. В настоящее время реализована поддержка уведомлений через Telegram.

## Особенности

- **Поддержка Telegram**: отправка уведомлений в заданный чат через Telegram Bot API
- **Различные типы уведомлений**: простые сообщения, предупреждения, информация о сделках, арбитражные возможности, статус системы
- **Уровни важности**: фильтрация сообщений по уровню важности (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- **Очередь сообщений**: асинхронная отправка сообщений для предотвращения блокировок
- **Задержка между уведомлениями**: предотвращение спама однотипными сообщениями
- **Расширяемость**: возможность добавления новых типов нотификаторов (почта, SMS и т.д.)

## Настройка Telegram-бота

Для использования уведомлений через Telegram, необходимо:

1. Создать нового бота в Telegram через [@BotFather](https://t.me/BotFather)
2. Получить API токен для созданного бота
3. Узнать ID вашего чата/группы, куда будут отправляться уведомления
4. Добавить полученные данные в конфигурационный файл `.env`:

```
TELEGRAM_BOT_TOKEN=ваш_токен_бота
TELEGRAM_CHAT_ID=ваш_id_чата
```

## Использование

### Инициализация

```python
from src.notifier import NotificationManager, TelegramNotifier

# Создаем менеджер уведомлений
notification_manager = NotificationManager()

# Можно также создать менеджер с настройками
notification_manager = NotificationManager(
    min_level=NotificationLevel.INFO,  # минимальный уровень важности
    cooldown_seconds=5.0,  # задержка между однотипными уведомлениями
    enabled_types=None  # None = все типы включены
)

# При необходимости можно добавить дополнительные нотификаторы
telegram_notifier = TelegramNotifier(token="другой_токен", chat_id="другой_чат_id")
notification_manager.add_notifier(telegram_notifier)
```

### Отправка простых сообщений

```python
# Отправка простого текстового сообщения
await notification_manager.send_message("Это простое сообщение")

# Отправка сообщения с уровнем важности
await notification_manager.send_alert(
    title="Важное сообщение",
    message="Детали сообщения",
    level=NotificationLevel.WARNING
)
```

### Уведомления о торговых операциях

```python
await notification_manager.send_trade_info(
    exchange="binance",
    symbol="BTC/USDT",
    operation="buy",
    amount=0.05,
    price=25000.0,
    status="executed",
    details={
        "Идентификатор ордера": "12345678",
        "Комиссия": "0.00005 BTC"
    }
)
```

### Уведомления об арбитражных возможностях

```python
await notification_manager.send_arbitrage_opportunity(
    buy_exchange="kucoin",
    sell_exchange="binance",
    symbol="BTC/USDT",
    buy_price=24950.0,
    sell_price=25200.0,
    profit_percent=1.0,
    estimated_profit=25.0,
    details={
        "Объем для торговли": "0.1 BTC",
        "Комиссии учтены": "Да"
    }
)
```

### Отправка статуса системы

```python
balances = {
    "binance": {"BTC": 0.15, "USDT": 5000.0},
    "kucoin": {"BTC": 0.08, "USDT": 3000.0}
}

await notification_manager.send_system_status(
    status="running",
    balances=balances,
    active_tasks=12,
    errors_count=2,
    details={
        "Время работы": "3 часа 45 минут",
        "Выполнено операций": 24
    }
)
```

### Отправка уведомлений об ошибках

```python
try:
    # Код, который может вызвать исключение
    result = execute_operation()
except Exception as e:
    import traceback
    stacktrace = traceback.format_exc()
    
    await notification_manager.send_error(
        title="Ошибка при выполнении операции",
        error_message=str(e),
        stacktrace=stacktrace
    )
```

### Управление фильтрацией уведомлений

```python
# Устанавливаем минимальный уровень уведомлений
notification_manager.set_min_level(NotificationLevel.WARNING)

# Устанавливаем задержку между однотипными уведомлениями
notification_manager.set_cooldown(10.0)  # 10 секунд

# Отключаем определенный тип уведомлений
notification_manager.disable_notification_type(NotificationType.TRADE)

# Включаем определенный тип уведомлений
notification_manager.enable_notification_type(NotificationType.TRADE)
```

### Корректное завершение

```python
# Ждем отправки всех сообщений в очереди и закрываем соединения
await notification_manager.close()
```

## Структура сообщений

Система использует HTML-форматирование для сообщений в Telegram, что позволяет создавать структурированные и красивые уведомления:

- **Торговые операции**: содержат информацию о бирже, паре, типе операции, объеме, цене и статусе
- **Арбитражные возможности**: содержат информацию о биржах, паре, ценах, процентной и абсолютной прибыли
- **Статус системы**: содержит общий статус, данные о балансах, активных задачах и ошибках
- **Сообщения об ошибках**: содержат заголовок, сообщение об ошибке и опционально трассировку стека

## Расширение системы

Для добавления нового типа нотификатора:

1. Создайте класс, наследующий `BaseNotifier` из `src/notifier/base_notifier.py`
2. Реализуйте все абстрактные методы интерфейса
3. По необходимости добавьте метод `initialize()` для настройки соединения
4. Добавьте метод `close()` для корректного завершения работы
5. Зарегистрируйте ваш нотификатор в `NotificationManager` с помощью `add_notifier()`

## Примеры

Полные примеры использования системы уведомлений можно найти в файле `examples/notification_example.py`.
