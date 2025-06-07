# db/repository.py

"""
Репозитории для работы с данными в базе данных.
Предоставляют методы CRUD (Create, Read, Update, Delete) для работы с моделями,
используя как синхронные, так и асинхронные сессии SQLAlchemy.
"""

import datetime
from typing import TypeVar, Generic, Type, List, Optional, Dict, Any, Tuple, Sequence, Callable, Coroutine

# Используем select как основной способ построения запросов (стиль SQLAlchemy 2.0)
from sqlalchemy import select, update, delete, func, and_, or_, desc, asc, exists as sql_exists
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session, InstrumentedAttribute # Для type hint

# Импортируем базовый класс и модели
from db.models import Base, ArbitrageOpportunity, Trade, Exchange, Symbol, Balance, Order, Config, LogEntry
from db.models import ArbitrageStatus, OrderStatus, OrderType, OrderSide

# Импортируем менеджеры БД для type hinting
from db.db_manager import DatabaseManager, AsyncDatabaseManager # Замените на правильный путь, если db_manager не в корне проекта

# Утилита логирования (замените на вашу реализацию)
import logging
def get_logger(name: str) -> logging.Logger:
    # Простая заглушка, замените на вашу реальную реализацию get_logger
    logger = logging.getLogger(name)
    # Настройте handler и formatter, если нужно
    # handler = logging.StreamHandler()
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # handler.setFormatter(formatter)
    # if not logger.handlers:
    #     logger.addHandler(handler)
    # logger.setLevel(logging.INFO) # Установите нужный уровень
    return logger


# Общий тип для моделей SQLAlchemy, наследуемых от Base
T = TypeVar('T', bound=Base)

# --- Базовый Синхронный Репозиторий ---

class Repository(Generic[T]):
    """
    Базовый синхронный репозиторий для работы с моделями SQLAlchemy (стиль 2.0).
    Реализует базовые CRUD-операции для модели.

    Attributes:
        model_class: Класс модели SQLAlchemy
        db_manager: Синхронный менеджер базы данных
        logger: Экземпляр логгера
    """

    def __init__(self, model_class: Type[T], db_manager: DatabaseManager):
        """
        Инициализирует репозиторий.

        Args:
            model_class: Класс модели SQLAlchemy
            db_manager: Синхронный менеджер базы данных
        """
        self.model_class = model_class
        self.db_manager = db_manager
        # Имя логгера включает имя класса модели для ясности
        self.logger = get_logger(f"{__name__}.{self.model_class.__name__}Repository")

    def create(self, session: Session, **kwargs) -> T:
        """
        Создает новую запись в базе данных.

        Args:
            session: Сессия SQLAlchemy
            **kwargs: Атрибуты для создания модели

        Returns:
            Созданный экземпляр модели
        """
        instance = self.model_class(**kwargs)
        session.add(instance)
        session.flush()  # Получаем ID и другие defaults без коммита
        self.logger.debug(f"Created {self.model_class.__name__} instance: {instance.id}")
        return instance

    def get_by_id(self, session: Session, id_: int) -> Optional[T]:
        """
        Получает запись по её ID.

        Args:
            session: Сессия SQLAlchemy
            id_: ID записи

        Returns:
            Экземпляр модели или None, если не найден
        """
        stmt = select(self.model_class).where(id_ == self.model_class.id)
        return session.execute(stmt).scalars().first()

    def get_all(self, session: Session, offset: int = 0, limit: Optional[int] = None) -> Sequence[T]:
        """
        Получает все записи с пагинацией.

        Args:
            session: Сессия SQLAlchemy
            offset: Смещение (пропуск N записей)
            limit: Ограничение количества записей

        Returns:
            Последовательность моделей (scalars().all() возвращает Sequence)
        """
        stmt = select(self.model_class).offset(offset)
        if limit is not None:
            stmt = stmt.limit(limit)
        return session.execute(stmt).scalars().all()

    def update(self, session: Session, id_: int, **kwargs) -> Optional[T]:
        """
        Обновляет запись по её ID. Находит запись и обновляет ее атрибуты.

        Args:
            session: Сессия SQLAlchemy
            id_: ID записи
            **kwargs: Атрибуты для обновления

        Returns:
            Обновленный экземпляр модели или None, если запись не найдена
        """
        instance = self.get_by_id(session, id_)
        if instance:
            for key, value in kwargs.items():
                if hasattr(instance, key):
                    setattr(instance, key, value)
                else:
                    self.logger.warning(f"Attempted to update non-existent attribute '{key}' on {self.model_class.__name__} ID: {id_}")
            session.flush() # Применяем изменения к сессии
            self.logger.debug(f"Updated {self.model_class.__name__} instance: {instance.id}")
            return instance
        return None

    def update_by_filter(self, session: Session, filter_by: Dict[str, Any], update_values: Dict[str, Any]) -> Callable[
        [], int]:
        """
        Обновляет несколько записей по фильтру за один запрос (без загрузки объектов).

        Args:
            session: Сессия SQLAlchemy
            filter_by: Критерии фильтрации записей для обновления
            update_values: Атрибуты для обновления

        Returns:
            Количество обновленных записей
        """
        # Динамическое построение условий WHERE
        conditions = [getattr(self.model_class, k) == v for k, v in filter_by.items()]
        stmt = (
            update(self.model_class)
            .where(and_(*conditions))
            .values(**update_values)
            # synchronizes the session (makes session aware of changes) - use with caution or omit if not needed
            .execution_options(synchronize_session="fetch") # или False, или 'evaluate'
        )
        result = session.execute(stmt)
        session.flush() # Убедимся, что изменения отражены для последующих запросов в той же сессии
        self.logger.debug(f"Bulk updated {result.rowcount} instances of {self.model_class.__name__} matching {filter_by}")
        return result.rowcount

    def delete_by_id(self, session: Session, id_: int) -> bool:
        """
        Удаляет запись по её ID. Находит, затем удаляет.

        Args:
            session: Сессия SQLAlchemy
            id_: ID записи

        Returns:
            True, если запись удалена, иначе False
        """
        instance = self.get_by_id(session, id_)
        if instance:
            session.delete(instance)
            session.flush() # Применяем удаление
            self.logger.debug(f"Deleted {self.model_class.__name__} instance: {id_}")
            return True
        return False

    def delete_by_filter(self, session: Session, **kwargs) -> Callable[[], int]:
        """
        Удаляет несколько записей по фильтру за один запрос (без загрузки объектов).

        Args:
            session: Сессия SQLAlchemy
            **kwargs: Критерии фильтрации записей для удаления

        Returns:
            Количество удаленных записей
        """
        conditions = [getattr(self.model_class, k) == v for k, v in kwargs.items()]
        stmt = (
            delete(self.model_class)
            .where(and_(*conditions))
            # synchronizes the session (makes session aware of changes) - use with caution or omit if not needed
            .execution_options(synchronize_session="fetch") # или False, или 'evaluate'
        )
        result = session.execute(stmt)
        session.flush()
        self.logger.debug(f"Bulk deleted {result.rowcount} instances of {self.model_class.__name__} matching {kwargs}")
        return result.rowcount

    def count(self, session: Session, **kwargs) -> int:
        """
        Возвращает количество записей, опционально с фильтром.

        Args:
            session: Сессия SQLAlchemy
            **kwargs: Критерии фильтрации

        Returns:
            Количество записей
        """
        stmt = select(func.count(self.model_class.id))
        if kwargs:
            conditions = [getattr(self.model_class, k) == v for k, v in kwargs.items()]
            stmt = stmt.where(and_(*conditions))
        count = session.execute(stmt).scalar()
        return count or 0

    def filter_by(self, session: Session, **kwargs) -> Sequence[T]:
        """
        Фильтрует записи по заданным критериям (равенство).

        Args:
            session: Сессия SQLAlchemy
            **kwargs: Критерии фильтрации (поле=значение)

        Returns:
            Последовательность моделей, удовлетворяющих критериям
        """
        # filter_by удобен для простых равенств
        stmt = select(self.model_class).filter_by(**kwargs)
        return session.execute(stmt).scalars().all()

    def exists(self, session: Session, **kwargs) -> bool:
        """
        Проверяет существование записи по заданным критериям.

        Args:
            session: Сессия SQLAlchemy
            **kwargs: Критерии проверки

        Returns:
            True, если запись существует, иначе False
        """
        # Использование exists() более эффективно, чем count() > 0 или filter_by().first() is not None
        stmt = select(self.model_class).filter_by(**kwargs)
        exists_stmt = select(sql_exists(stmt))
        result = session.execute(exists_stmt).scalar()
        return result or False # scalar() может вернуть None

    def bulk_create(self, session: Session, items: List[Dict[str, Any]]) -> Sequence[T]:
        """
        Создает несколько записей за один вызов (эффективнее, чем по одной).

        Args:
            session: Сессия SQLAlchemy
            items: Список словарей с атрибутами для создания моделей

        Returns:
            Последовательность созданных моделей (может не содержать ID до коммита, зависит от БД)
        """
        # Note: session.add_all() is often optimized by drivers
        instances = [self.model_class(**item) for item in items]
        session.add_all(instances)
        session.flush() # Попытка получить ID и defaults
        self.logger.debug(f"Bulk created {len(instances)} instances of {self.model_class.__name__}")
        # Возвращаем добавленные экземпляры (их состояние может обновиться после flush/commit)
        return instances

    def find_with_pagination(
        self,
        session: Session,
        filters: Optional[Dict[str, Any]] = None,
        order_by: Optional[str] = None,
        order_direction: str = "asc",
        offset: int = 0,
        limit: Optional[int] = None
    ) -> Tuple[Sequence[T], int]:
        """
        Находит записи с пагинацией, фильтрацией (по равенству) и сортировкой.

        Args:
            session: Сессия SQLAlchemy
            filters: Критерии фильтрации (словарь {поле: значение})
            order_by: Имя поля для сортировки (атрибут модели)
            order_direction: Направление сортировки ("asc" или "desc")
            offset: Смещение (пропуск N записей)
            limit: Ограничение количества записей

        Returns:
            Кортеж (последовательность моделей, общее количество записей по фильтру)
        """
        filters = filters or {}

        # Запрос для получения данных
        stmt = select(self.model_class)

        # Запрос для подсчета общего количества (с теми же фильтрами)
        count_stmt = select(func.count(self.model_class.id))

        # Применяем фильтры (простое равенство через filter_by)
        if filters:
            stmt = stmt.filter_by(**filters)
            count_stmt = count_stmt.filter_by(**filters) # Важно применить те же фильтры к count

        # Получаем общее количество записей для пагинации
        total_count = session.execute(count_stmt).scalar() or 0

        # Применяем сортировку
        if order_by and hasattr(self.model_class, order_by):
            column: InstrumentedAttribute = getattr(self.model_class, order_by)
            order_func = desc if order_direction.lower() == "desc" else asc
            stmt = stmt.order_by(order_func(column))
        elif order_by:
             self.logger.warning(f"Attempted to order by non-existent attribute '{order_by}' on {self.model_class.__name__}")


        # Применяем пагинацию
        stmt = stmt.offset(offset)
        if limit is not None:
            stmt = stmt.limit(limit)

        # Выполняем основной запрос
        results = session.execute(stmt).scalars().all()

        return results, total_count


# --- Базовый Асинхронный Репозиторий ---

class AsyncRepository(Generic[T]):
    """
    Асинхронный базовый репозиторий для работы с моделями SQLAlchemy.
    Реализует базовые CRUD-операции для модели с использованием асинхронных сессий.

    Attributes:
        model_class: Класс модели SQLAlchemy
        db_manager: Асинхронный менеджер базы данных
        logger: Экземпляр логгера
    """

    def __init__(self, model_class: Type[T], db_manager: AsyncDatabaseManager):
        """
        Инициализирует асинхронный репозиторий.

        Args:
            model_class: Класс модели SQLAlchemy
            db_manager: Асинхронный менеджер базы данных
        """
        self.model_class = model_class
        self.db_manager = db_manager
        self.logger = get_logger(f"{__name__}.Async{self.model_class.__name__}Repository")

    async def create(self, session: AsyncSession, **kwargs) -> T:
        """
        Асинхронно создает новую запись в базе данных.

        Args:
            session: Асинхронная сессия SQLAlchemy
            **kwargs: Атрибуты для создания модели

        Returns:
            Созданный экземпляр модели
        """
        instance = self.model_class(**kwargs)
        session.add(instance)
        await session.flush()
        self.logger.debug(f"Created {self.model_class.__name__} instance: {instance.id}")
        return instance

    async def get_by_id(self, session: AsyncSession, id_: int) -> Optional[T]:
        """
        Асинхронно получает запись по её ID.

        Args:
            session: Асинхронная сессия SQLAlchemy
            id_: ID записи

        Returns:
            Экземпляр модели или None, если не найден
        """
        stmt = select(self.model_class).where(id_ == self.model_class.id)
        result = await session.execute(stmt)
        return result.scalars().first()

    async def get_all(self, session: AsyncSession, offset: int = 0, limit: Optional[int] = None) -> Sequence[T]:
        """
        Асинхронно получает все записи с пагинацией.

        Args:
            session: Асинхронная сессия SQLAlchemy
            offset: Смещение (пропуск N записей)
            limit: Ограничение количества записей

        Returns:
            Последовательность моделей
        """
        stmt = select(self.model_class).offset(offset)
        if limit is not None:
            stmt = stmt.limit(limit)
        result = await session.execute(stmt)
        return result.scalars().all()

    async def update(self, session: AsyncSession, id_: int, **kwargs) -> Optional[T]:
        """
        Асинхронно обновляет запись по её ID. Находит и обновляет атрибуты.

        Args:
            session: Асинхронная сессия SQLAlchemy
            id_: ID записи
            **kwargs: Атрибуты для обновления

        Returns:
            Обновленный экземпляр модели или None, если не найдена
        """
        instance = await self.get_by_id(session, id_)
        if instance:
            for key, value in kwargs.items():
                if hasattr(instance, key):
                    setattr(instance, key, value)
                else:
                    self.logger.warning(f"Attempted to update non-existent attribute '{key}' on {self.model_class.__name__} ID: {id_}")
            await session.flush()
            self.logger.debug(f"Updated {self.model_class.__name__} instance: {instance.id}")
            return instance
        return None

    async def update_by_filter(self, session: AsyncSession, filter_by: Dict[str, Any], update_values: Dict[str, Any]) -> \
    Callable[[], int]:
        """
        Асинхронно обновляет несколько записей по фильтру за один запрос.

        Args:
            session: Асинхронная сессия SQLAlchemy
            filter_by: Критерии фильтрации записей для обновления
            update_values: Атрибуты для обновления

        Returns:
            Количество обновленных записей
        """
        conditions = [getattr(self.model_class, k) == v for k, v in filter_by.items()]
        stmt = (
            update(self.model_class)
            .where(and_(*conditions))
            .values(**update_values)
            .execution_options(synchronize_session="fetch") # Важно для async тоже, если нужна синхронизация
        )
        result = await session.execute(stmt)
        await session.flush()
        self.logger.debug(f"Bulk updated {result.rowcount} instances of {self.model_class.__name__} matching {filter_by}")
        return result.rowcount

    async def delete_by_id(self, session: AsyncSession, id_: int) -> bool:
        """
        Асинхронно удаляет запись по её ID. Находит, затем удаляет.

        Args:
            session: Асинхронная сессия SQLAlchemy
            id_: ID записи

        Returns:
            True, если запись удалена, иначе False
        """
        instance = await self.get_by_id(session, id_)
        if instance:
            await session.delete(instance)
            await session.flush()
            self.logger.debug(f"Deleted {self.model_class.__name__} instance: {id_}")
            return True
        return False

    async def delete_by_filter(self, session: AsyncSession, **kwargs) -> Callable[[], int]:
        """
        Асинхронно удаляет несколько записей по фильтру за один запрос.

        Args:
            session: Асинхронная сессия SQLAlchemy
            **kwargs: Критерии фильтрации записей для удаления

        Returns:
            Количество удаленных записей
        """
        conditions = [getattr(self.model_class, k) == v for k, v in kwargs.items()]
        stmt = (
            delete(self.model_class)
            .where(and_(*conditions))
            .execution_options(synchronize_session="fetch")
        )
        result = await session.execute(stmt)
        await session.flush()
        self.logger.debug(f"Bulk deleted {result.rowcount} instances of {self.model_class.__name__} matching {kwargs}")
        return result.rowcount

    async def count(self, session: AsyncSession, **kwargs) -> int:
        """
        Асинхронно возвращает количество записей, опционально с фильтром.

        Args:
            session: Асинхронная сессия SQLAlchemy
            **kwargs: Критерии фильтрации

        Returns:
            Количество записей
        """
        stmt = select(func.count(self.model_class.id))
        if kwargs:
            conditions = [getattr(self.model_class, k) == v for k, v in kwargs.items()]
            stmt = stmt.where(and_(*conditions))
        result = await session.execute(stmt)
        count = result.scalar()
        return count or 0

    async def filter_by(self, session: AsyncSession, **kwargs) -> Sequence[T]:
        """
        Асинхронно фильтрует записи по заданным критериям (равенство).

        Args:
            session: Асинхронная сессия SQLAlchemy
            **kwargs: Критерии фильтрации

        Returns:
            Последовательность моделей, удовлетворяющих критериям
        """
        stmt = select(self.model_class).filter_by(**kwargs)
        result = await session.execute(stmt)
        return result.scalars().all()

    async def exists(self, session: AsyncSession, **kwargs) -> bool:
        """
        Асинхронно проверяет существование записи по заданным критериям.

        Args:
            session: Асинхронная сессия SQLAlchemy
            **kwargs: Критерии проверки

        Returns:
            True, если запись существует, иначе False
        """
        stmt = select(self.model_class).filter_by(**kwargs)
        exists_stmt = select(sql_exists(stmt))
        result = await session.execute(exists_stmt)
        exists = result.scalar()
        return exists or False

    async def bulk_create(self, session: AsyncSession, items: List[Dict[str, Any]]) -> Sequence[T]:
        """
        Асинхронно создает несколько записей за один вызов.

        Args:
            session: Асинхронная сессия SQLAlchemy
            items: Список словарей с атрибутами для создания моделей

        Returns:
            Последовательность созданных моделей
        """
        instances = [self.model_class(**item) for item in items]
        session.add_all(instances)
        await session.flush()
        self.logger.debug(f"Bulk created {len(instances)} instances of {self.model_class.__name__}")
        return instances

    async def find_with_pagination(
        self,
        session: AsyncSession,
        filters: Optional[Dict[str, Any]] = None,
        order_by: Optional[str] = None,
        order_direction: str = "asc",
        offset: int = 0,
        limit: Optional[int] = None
    ) -> Tuple[Sequence[T], int]:
        """
        Асинхронно находит записи с пагинацией, фильтрацией и сортировкой.

        Args:
            session: Асинхронная сессия SQLAlchemy
            filters: Критерии фильтрации (словарь {поле: значение})
            order_by: Имя поля для сортировки
            order_direction: Направление сортировки ("asc" или "desc")
            offset: Смещение (пропуск N записей)
            limit: Ограничение количества записей

        Returns:
            Кортеж (последовательность моделей, общее количество записей по фильтру)
        """
        filters = filters or {}

        # Запрос для получения данных
        stmt = select(self.model_class)
        # Запрос для подсчета
        count_stmt = select(func.count(self.model_class.id))

        # Применяем фильтры
        if filters:
            stmt = stmt.filter_by(**filters)
            count_stmt = count_stmt.filter_by(**filters)

        # Получаем общее количество
        count_result = await session.execute(count_stmt)
        total_count = count_result.scalar() or 0

        # Применяем сортировку
        if order_by and hasattr(self.model_class, order_by):
            column: InstrumentedAttribute = getattr(self.model_class, order_by)
            order_func = desc if order_direction.lower() == "desc" else asc
            stmt = stmt.order_by(order_func(column))
        elif order_by:
             self.logger.warning(f"Attempted to order by non-existent attribute '{order_by}' on {self.model_class.__name__}")

        # Применяем пагинацию
        stmt = stmt.offset(offset)
        if limit is not None:
            stmt = stmt.limit(limit)

        # Выполняем основной запрос
        result = await session.execute(stmt)
        results = result.scalars().all()

        return results, total_count

# ------------- Специализированные Репозитории -------------
# Каждый репозиторий наследуется от Repository или AsyncRepository
# и может добавлять специфичные методы запросов.

# --- Синхронные ---

class ExchangeRepository(Repository[Exchange]):
    def __init__(self, db_manager: DatabaseManager):
        super().__init__(Exchange, db_manager)

    def get_by_name(self, session: Session, name: str) -> Optional[Exchange]:
        return session.execute(
            select(self.model_class).where(name == self.model_class.name)
        ).scalars().first()

    def get_active(self, session: Session) -> Sequence[Exchange]:
        return self.filter_by(session, is_active=True)


class SymbolRepository(Repository[Symbol]):
    def __init__(self, db_manager: DatabaseManager):
        super().__init__(Symbol, db_manager)

    def get_by_name(self, session: Session, name: str) -> Optional[Symbol]:
         return session.execute(
            select(self.model_class).where(name == self.model_class.name)
        ).scalars().first()

    def get_active(self, session: Session) -> Sequence[Symbol]:
        return self.filter_by(session, is_active=True)

    def find_by_assets(self, session: Session, base_asset: str, quote_asset: str) -> Optional[Symbol]:
        return session.execute(
            select(self.model_class).where(
                and_(self.model_class.base_asset == base_asset, self.model_class.quote_asset == quote_asset)
            )
        ).scalars().first()


class BalanceRepository(Repository[Balance]):
    def __init__(self, db_manager: DatabaseManager):
        super().__init__(Balance, db_manager)

    def get_by_exchange_and_asset(self, session: Session, exchange_id: int, asset: str) -> Optional[Balance]:
        # 1. Выполняем filter_by, получаем Sequence
        results: Sequence[Balance] = self.filter_by(session, exchange_id=exchange_id, asset=asset)
        # 2. Проверяем, есть ли элементы в Sequence, и возвращаем первый, если есть, иначе None
        return results[0] if results else None

    def get_all_by_exchange(self, session: Session, exchange_id: int) -> Sequence[Balance]:
        return self.filter_by(session, exchange_id=exchange_id)

    def get_non_zero(self, session: Session, exchange_id: Optional[int] = None) -> Sequence[Balance]:
        stmt = select(self.model_class).where(
            or_(self.model_class.free > 0, self.model_class.locked > 0) # Сравнение с 0 для Decimal/Numeric может потребовать внимания
        )
        if exchange_id is not None:
            stmt = stmt.where(exchange_id == self.model_class.exchange_id)
        return session.execute(stmt).scalars().all()


class ArbitrageOpportunityRepository(Repository[ArbitrageOpportunity]):
    def __init__(self, db_manager: DatabaseManager):
        super().__init__(ArbitrageOpportunity, db_manager)

    def get_by_status(self, session: Session, status: ArbitrageStatus) -> Sequence[ArbitrageOpportunity]:
        return self.filter_by(session, status=status)

    def get_recent(self, session: Session, hours: int = 24) -> Sequence[ArbitrageOpportunity]:
        time_threshold = datetime.datetime.now(datetime.UTC) - datetime.timedelta(hours=hours)
        stmt = select(self.model_class).where(
            self.model_class.detected_at >= time_threshold
        ).order_by(desc(self.model_class.detected_at))
        return session.execute(stmt).scalars().all()

    def get_profitable(self, session: Session, min_profit_percentage: float = 0.5) -> Sequence[ArbitrageOpportunity]:
        # Убедитесь, что тип min_profit_percentage совместим с типом столбца (Decimal/Numeric)
        stmt = select(self.model_class).where(
            self.model_class.profit_percentage >= min_profit_percentage
        ).order_by(desc(self.model_class.profit_percentage))
        return session.execute(stmt).scalars().all()


class TradeRepository(Repository[Trade]):
    def __init__(self, db_manager: DatabaseManager):
        super().__init__(Trade, db_manager)

    def get_by_opportunity_id(self, session: Session, opportunity_id: int) -> Sequence[Trade]:
        return self.filter_by(session, opportunity_id=opportunity_id)

    def get_simulated(self, session: Session) -> Sequence[Trade]:
        return self.filter_by(session, is_simulated=True)

    def get_real(self, session: Session) -> Sequence[Trade]:
        return self.filter_by(session, is_simulated=False)

    def get_by_status(self, session: Session, status: str) -> Sequence[Trade]: # Или TradeStatus, если вы его ввели
        return self.filter_by(session, status=status)

    def get_recent(self, session: Session, hours: int = 24) -> Sequence[Trade]:
        time_threshold = datetime.datetime.now(datetime.UTC) - datetime.timedelta(hours=hours)
        stmt = select(self.model_class).where(
            self.model_class.created_at >= time_threshold
        ).order_by(desc(self.model_class.created_at))
        return session.execute(stmt).scalars().all()

    def get_profitable(self, session: Session, min_profit_percentage: float = 0.5) -> Sequence[Trade]:
        stmt = select(self.model_class).where(
            self.model_class.profit_percentage >= min_profit_percentage
        ).order_by(desc(self.model_class.profit_percentage))
        return session.execute(stmt).scalars().all()


class OrderRepository(Repository[Order]):
    def __init__(self, db_manager: DatabaseManager):
        super().__init__(Order, db_manager)

    def get_by_trade_id(self, session: Session, trade_id: int) -> Sequence[Order]:
        return self.filter_by(session, trade_id=trade_id)

    def get_by_exchange_id(self, session: Session, exchange_id: int) -> Sequence[Order]:
        return self.filter_by(session, exchange_id=exchange_id)

    def get_by_symbol_id(self, session: Session, symbol_id: int) -> Sequence[Order]:
         return self.filter_by(session, symbol_id=symbol_id)

    def get_by_status(self, session: Session, status: OrderStatus) -> Sequence[Order]:
        return self.filter_by(session, status=status)

    def get_by_side(self, session: Session, side: OrderSide) -> Sequence[Order]:
        return self.filter_by(session, side=side)

    def get_by_type(self, session: Session, order_type: OrderType) -> Sequence[Order]:
        return self.filter_by(session, order_type=order_type)

    def get_by_exchange_order_id(self, session: Session, exchange_order_id: str) -> Optional[Order]:
         # Используем where вместо filter_by, т.к. filter_by не всегда хорошо работает с именами полей, совпадающими с именами аргументов
         stmt = select(self.model_class).where(exchange_order_id == self.model_class.exchange_order_id)
         return session.execute(stmt).scalars().first()

    def get_recent(self, session: Session, hours: int = 24) -> Sequence[Order]:
        time_threshold = datetime.datetime.now(datetime.UTC) - datetime.timedelta(hours=hours)
        stmt = select(self.model_class).where(
            self.model_class.created_at >= time_threshold
        ).order_by(desc(self.model_class.created_at))
        return session.execute(stmt).scalars().all()


class ConfigRepository(Repository[Config]):
    def __init__(self, db_manager: DatabaseManager):
        super().__init__(Config, db_manager)

    def get_by_key(self, session: Session, key: str) -> Optional[Config]:
         return session.execute(
             select(self.model_class).where(key == self.model_class.key)
         ).scalars().first()

    def get_value_by_key(self, session: Session, key: str, default: Any = None) -> Any:
        config = self.get_by_key(session, key)
        return config.value if config else default

    def set_value(self, session: Session, key: str, value: Any, description: Optional[str] = None) -> Config:
        config = self.get_by_key(session, key)
        if config:
            config.value = value
            if description is not None: # Allow clearing description
                config.description = description
            self.logger.info(f"Updating config key '{key}'")
        else:
            config = self.model_class(key=key, value=value, description=description)
            session.add(config)
            self.logger.info(f"Creating config key '{key}'")
        session.flush()
        return config


class LogEntryRepository(Repository[LogEntry]):
    def __init__(self, db_manager: DatabaseManager):
        super().__init__(LogEntry, db_manager)

    def get_by_level(self, session: Session, level: str) -> Sequence[LogEntry]:
        return self.filter_by(session, level=level)

    def get_by_module(self, session: Session, module: str) -> Sequence[LogEntry]:
        return self.filter_by(session, module=module)

    def get_recent(self, session: Session, hours: int = 24) -> Sequence[LogEntry]:
        time_threshold = datetime.datetime.now(datetime.UTC) - datetime.timedelta(hours=hours)
        stmt = select(self.model_class).where(
            self.model_class.created_at >= time_threshold
        ).order_by(desc(self.model_class.created_at))
        return session.execute(stmt).scalars().all()

    def get_errors(self, session: Session) -> Sequence[LogEntry]:
        stmt = select(self.model_class).where(
            or_(self.model_class.level == "ERROR", self.model_class.level == "CRITICAL")
        ).order_by(desc(self.model_class.created_at))
        return session.execute(stmt).scalars().all()


# --- Асинхронные ---
# Структура аналогична синхронным, но использует AsyncSession и await

class AsyncExchangeRepository(AsyncRepository[Exchange]):
    def __init__(self, db_manager: AsyncDatabaseManager):
        super().__init__(Exchange, db_manager)

    async def get_by_name(self, session: AsyncSession, name: str) -> Optional[Exchange]:
         result = await session.execute(
            select(self.model_class).where(name == self.model_class.name)
        )
         return result.scalars().first()

    async def get_active(self, session: AsyncSession) -> Sequence[Exchange]:
        return await self.filter_by(session, is_active=True)


class AsyncSymbolRepository(AsyncRepository[Symbol]):
    def __init__(self, db_manager: AsyncDatabaseManager):
        super().__init__(Symbol, db_manager)

    async def get_by_name(self, session: AsyncSession, name: str) -> Optional[Symbol]:
        result = await session.execute(
            select(self.model_class).where(name == self.model_class.name)
        )
        return result.scalars().first()

    async def get_active(self, session: AsyncSession) -> Sequence[Symbol]:
        return await self.filter_by(session, is_active=True)

    async def find_by_assets(self, session: AsyncSession, base_asset: str, quote_asset: str) -> Optional[Symbol]:
        result = await session.execute(
            select(self.model_class).where(
                and_(self.model_class.base_asset == base_asset, self.model_class.quote_asset == quote_asset)
            )
        )
        return result.scalars().first()


class AsyncBalanceRepository(AsyncRepository[Balance]):
    def __init__(self, db_manager: AsyncDatabaseManager):
        super().__init__(Balance, db_manager)

    async def get_by_exchange_and_asset(self, session: AsyncSession, exchange_id: int, asset: str) -> Optional[Balance]:
        # filter_by возвращает Sequence, берем первый элемент
        results = await self.filter_by(session, exchange_id=exchange_id, asset=asset)
        return results[0] if results else None


    async def get_all_by_exchange(self, session: AsyncSession, exchange_id: int) -> Sequence[Balance]:
        return await self.filter_by(session, exchange_id=exchange_id)

    async def get_non_zero(self, session: AsyncSession, exchange_id: Optional[int] = None) -> Sequence[Balance]:
        stmt = select(self.model_class).where(
            or_(self.model_class.free > 0, self.model_class.locked > 0)
        )
        if exchange_id is not None:
            stmt = stmt.where(exchange_id == self.model_class.exchange_id)
        result = await session.execute(stmt)
        return result.scalars().all()


class AsyncArbitrageOpportunityRepository(AsyncRepository[ArbitrageOpportunity]):
    def __init__(self, db_manager: AsyncDatabaseManager):
        super().__init__(ArbitrageOpportunity, db_manager)

    async def get_by_status(self, session: AsyncSession, status: ArbitrageStatus) -> Sequence[ArbitrageOpportunity]:
        return await self.filter_by(session, status=status)

    async def get_recent(self, session: AsyncSession, hours: int = 24) -> Sequence[ArbitrageOpportunity]:
        time_threshold = datetime.datetime.now(datetime.UTC) - datetime.timedelta(hours=hours)
        stmt = select(self.model_class).where(
            self.model_class.detected_at >= time_threshold
        ).order_by(desc(self.model_class.detected_at))
        result = await session.execute(stmt)
        return result.scalars().all()

    async def get_profitable(self, session: AsyncSession, min_profit_percentage: float = 0.5) -> Sequence[ArbitrageOpportunity]:
        stmt = select(self.model_class).where(
            self.model_class.profit_percentage >= min_profit_percentage
        ).order_by(desc(self.model_class.profit_percentage))
        result = await session.execute(stmt)
        return result.scalars().all()


class AsyncTradeRepository(AsyncRepository[Trade]):
    def __init__(self, db_manager: AsyncDatabaseManager):
        super().__init__(Trade, db_manager)

    async def get_by_opportunity_id(self, session: AsyncSession, opportunity_id: int) -> Sequence[Trade]:
        return await self.filter_by(session, opportunity_id=opportunity_id)

    async def get_simulated(self, session: AsyncSession) -> Sequence[Trade]:
        return await self.filter_by(session, is_simulated=True)

    async def get_real(self, session: AsyncSession) -> Sequence[Trade]:
        return await self.filter_by(session, is_simulated=False)

    async def get_by_status(self, session: AsyncSession, status: str) -> Sequence[Trade]: # Или TradeStatus
        return await self.filter_by(session, status=status)

    async def get_recent(self, session: AsyncSession, hours: int = 24) -> Sequence[Trade]:
        time_threshold = datetime.datetime.now(datetime.UTC) - datetime.timedelta(hours=hours)
        stmt = select(self.model_class).where(
            self.model_class.created_at >= time_threshold
        ).order_by(desc(self.model_class.created_at))
        result = await session.execute(stmt)
        return result.scalars().all()

    async def get_profitable(self, session: AsyncSession, min_profit_percentage: float = 0.5) -> Sequence[Trade]:
        stmt = select(self.model_class).where(
            self.model_class.profit_percentage >= min_profit_percentage
        ).order_by(desc(self.model_class.profit_percentage))
        result = await session.execute(stmt)
        return result.scalars().all()


class AsyncOrderRepository(AsyncRepository[Order]):
    def __init__(self, db_manager: AsyncDatabaseManager):
        super().__init__(Order, db_manager)

    async def get_by_trade_id(self, session: AsyncSession, trade_id: int) -> Sequence[Order]:
        return await self.filter_by(session, trade_id=trade_id)

    async def get_by_exchange_id(self, session: AsyncSession, exchange_id: int) -> Sequence[Order]:
        return await self.filter_by(session, exchange_id=exchange_id)

    async def get_by_symbol_id(self, session: AsyncSession, symbol_id: int) -> Sequence[Order]:
         return await self.filter_by(session, symbol_id=symbol_id)

    async def get_by_status(self, session: AsyncSession, status: OrderStatus) -> Sequence[Order]:
        return await self.filter_by(session, status=status)

    async def get_by_side(self, session: AsyncSession, side: OrderSide) -> Sequence[Order]:
        return await self.filter_by(session, side=side)

    async def get_by_type(self, session: AsyncSession, order_type: OrderType) -> Sequence[Order]:
        return await self.filter_by(session, order_type=order_type)

    async def get_by_exchange_order_id(self, session: AsyncSession, exchange_order_id: str) -> Optional[Order]:
         stmt = select(self.model_class).where(exchange_order_id == self.model_class.exchange_order_id)
         result = await session.execute(stmt)
         return result.scalars().first()

    async def get_recent(self, session: AsyncSession, hours: int = 24) -> Sequence[Order]:
        time_threshold = datetime.datetime.now(datetime.UTC) - datetime.timedelta(hours=hours)
        stmt = select(self.model_class).where(
            self.model_class.created_at >= time_threshold
        ).order_by(desc(self.model_class.created_at))
        result = await session.execute(stmt)
        return result.scalars().all()


class AsyncConfigRepository(AsyncRepository[Config]):
    def __init__(self, db_manager: AsyncDatabaseManager):
        super().__init__(Config, db_manager)

    async def get_by_key(self, session: AsyncSession, key: str) -> Optional[Config]:
         result = await session.execute(
             select(self.model_class).where(key == self.model_class.key)
         )
         return result.scalars().first()

    async def get_value_by_key(self, session: AsyncSession, key: str, default: Any = None) -> Any:
        config = await self.get_by_key(session, key)
        return config.value if config else default

    async def set_value(self, session: AsyncSession, key: str, value: Any, description: Optional[str] = None) -> Config:
        config = await self.get_by_key(session, key)
        if config:
            config.value = value
            if description is not None:
                config.description = description
            self.logger.info(f"Updating config key '{key}'")
        else:
            config = self.model_class(key=key, value=value, description=description)
            session.add(config)
            self.logger.info(f"Creating config key '{key}'")
        await session.flush()
        return config


class AsyncLogEntryRepository(AsyncRepository[LogEntry]):
    def __init__(self, db_manager: AsyncDatabaseManager):
        super().__init__(LogEntry, db_manager)

    async def get_by_level(self, session: AsyncSession, level: str) -> Sequence[LogEntry]:
        return await self.filter_by(session, level=level)

    async def get_by_module(self, session: AsyncSession, module: str) -> Sequence[LogEntry]:
        return await self.filter_by(session, module=module)

    async def get_recent(self, session: AsyncSession, hours: int = 24) -> Sequence[LogEntry]:
        time_threshold = datetime.datetime.now(datetime.UTC) - datetime.timedelta(hours=hours)
        stmt = select(self.model_class).where(
            self.model_class.created_at >= time_threshold
        ).order_by(desc(self.model_class.created_at))
        result = await session.execute(stmt)
        return result.scalars().all()

    async def get_errors(self, session: AsyncSession) -> Sequence[LogEntry]:
        stmt = select(self.model_class).where(
            or_(self.model_class.level == "ERROR", self.model_class.level == "CRITICAL")
        ).order_by(desc(self.model_class.created_at))
        result = await session.execute(stmt)
        return result.scalars().all()