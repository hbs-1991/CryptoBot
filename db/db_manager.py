"""
Менеджер базы данных для создания соединений и сессий SQLAlchemy.
"""

import asyncio
import os
import logging
from typing import Optional, Any, List, Dict, Generator, AsyncGenerator
from contextlib import contextmanager, asynccontextmanager

from sqlalchemy import create_engine, event, exc
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import text

from db.models import Base
from config.settings import Settings
from src.utils import get_logger

# Получаем настройки
settings = Settings()


def _set_sqlite_pragma(dbapi_connection: Any, connection_record: Any) -> None:
    """
    Устанавливает PRAGMA для SQLite для включения внешних ключей.

    Args:
        dbapi_connection: Соединение DBAPI
        connection_record: Запись соединения
    """
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


class DatabaseManager:
    """
    Менеджер базы данных для работы с SQLAlchemy.
    Обеспечивает создание и управление соединениями и сессиями.
    """
    
    def __init__(
        self,
        db_url: Optional[str] = None,
        echo: bool = False,
        pool_size: int = 5,
        max_overflow: int = 10,
        pool_timeout: int = 30,
        pool_recycle: int = 1800,
        connect_args: Optional[Dict[str, Any]] = None
    ):
        """
        Инициализирует менеджер базы данных.
        
        Args:
            db_url: URL для подключения к базе данных
            echo: Включает подробный вывод SQL-запросов в логи
            pool_size: Размер пула соединений
            max_overflow: Максимальное количество дополнительных соединений
            pool_timeout: Таймаут ожидания доступного соединения в секундах
            pool_recycle: Время в секундах, после которого соединение переустанавливается
            connect_args: Дополнительные аргументы для соединения
        """
        self.logger = get_logger(__name__)
        self.db_url = db_url or settings.DATABASE_URL
        self.echo = echo
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_timeout = pool_timeout
        self.pool_recycle = pool_recycle
        self.connect_args = connect_args or {}
        
        # Проверяем, существует ли директория для SQLite
        if self.db_url.startswith('sqlite:///'):
            db_path = self.db_url.replace('sqlite:///', '')
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Для SQLite добавляем параметр check_same_thread=False
        if self.db_url.startswith('sqlite'):
            self.connect_args.setdefault('check_same_thread', False)
        
        # Инициализация движка SQLAlchemy
        self.engine = create_engine(
            self.db_url,
            echo=self.echo,
            pool_size=self.pool_size,
            max_overflow=self.max_overflow,
            pool_timeout=self.pool_timeout,
            pool_recycle=self.pool_recycle,
            connect_args=self.connect_args
        )
        
        # Для SQLite включаем поддержку внешних ключей
        if self.db_url.startswith('sqlite'):
            event.listen(self.engine, 'connect', _set_sqlite_pragma)
        
        # Создание сессии
        self.Session = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """
        Создает сессию SQLAlchemy как контекстный менеджер.
        
        Yields:
            Сессия SQLAlchemy
        """
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            self.logger.error(f"Ошибка при работе с сессией БД: {str(e)}")
            raise
        finally:
            session.close()
    
    def create_tables(self) -> None:
        """
        Создает все таблицы в базе данных на основе моделей.
        """
        self.logger.info("Создание таблиц в базе данных...")
        Base.metadata.create_all(self.engine)
        self.logger.info("Таблицы успешно созданы")
    
    def drop_tables(self) -> None:
        """
        Удаляет все таблицы из базы данных.
        """
        self.logger.warning("Удаление всех таблиц из базы данных...")
        Base.metadata.drop_all(self.engine)
        self.logger.warning("Все таблицы удалены")

    # Пример для синхронного метода
    def execute_raw_sql(self, sql: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        params = params or {}
        try:
            with self.get_session() as session:
                result = session.execute(text(sql), params)
                if result.returns_rows:
                    # fetchall() может быть большим, рассмотреть потоковую обработку для больших результатов
                    return [dict(row._mapping) for row in
                            result.fetchall()]  # Использовать _mapping для доступа к словарю
                return []
        except exc.SQLAlchemyError as e:
            self.logger.error(f"Ошибка при выполнении raw SQL: {sql} с параметрами {params}. Ошибка: {str(e)}")
            raise  # Перебрасываем ошибку дальше для отката сессии
    
    def get_table_names(self) -> List[str]:
        """
        Возвращает список имен таблиц в базе данных.
        
        Returns:
            Список имен таблиц
        """
        return Base.metadata.tables.keys()
    
    def check_connection(self) -> bool:
        """
        Проверяет соединение с базой данных.
        
        Returns:
            True, если соединение успешно, иначе False
        """
        try:
            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            return True
        except exc.SQLAlchemyError as e:
            self.logger.error(f"Ошибка соединения с базой данных: {str(e)}")
            return False
    
    def get_database_info(self) -> Dict[str, Any]:
        """
        Возвращает информацию о базе данных.
        
        Returns:
            Словарь с информацией о БД
        """
        table_names = self.get_table_names()
        table_rows_count = {}
        
        with self.get_session() as session:
            for table_name in table_names:
                try:
                    result = session.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                    table_rows_count[table_name] = result.scalar()
                except Exception as e:
                    self.logger.error(f"Ошибка при подсчете строк в таблице {table_name}: {str(e)}")
                    table_rows_count[table_name] = "Error"
        
        return {
            "url": self.db_url.split("@")[-1] if "@" in self.db_url else self.db_url,  # не показываем пароль
            "tables": table_names,
            "rows_count": table_rows_count,
            "engine": str(self.engine.dialect.name),
            "is_connected": self.check_connection()
        }
    
    def close(self) -> None:
        """
        Закрывает соединения с базой данных.
        """
        self.logger.info("Закрытие соединений с базой данных...")
        self.engine.dispose()
        self.logger.info("Соединения с базой данных закрыты")


def _set_sqlite_pragma(dbapi_connection: Any, connection_record: Any) -> None:
    """
    Устанавливает PRAGMA для SQLite для включения внешних ключей.

    Args:
        dbapi_connection: Соединение DBAPI
        connection_record: Запись соединения
    """
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


class AsyncDatabaseManager:
    """
    Асинхронный менеджер базы данных для работы с SQLAlchemy.
    Обеспечивает создание и управление асинхронными соединениями и сессиями.
    """
    
    def __init__(
        self,
        db_url: Optional[str] = None,
        echo: bool = False,
        pool_size: int = 5,
        max_overflow: int = 10,
        pool_timeout: int = 30,
        pool_recycle: int = 1800,
        connect_args: Optional[Dict[str, Any]] = None
    ):
        """
        Инициализирует менеджер базы данных.
        
        Args:
            db_url: URL для подключения к базе данных
            echo: Включает подробный вывод SQL-запросов в логи
            pool_size: Размер пула соединений
            max_overflow: Максимальное количество дополнительных соединений
            pool_timeout: Таймаут ожидания доступного соединения в секундах
            pool_recycle: Время в секундах, после которого соединение переустанавливается
            connect_args: Дополнительные аргументы для соединения
        """
        self.logger = get_logger(__name__)
        self._db_url = db_url or settings.DATABASE_URL
        
        # SQLite не поддерживает асинхронные операции напрямую,
        # поэтому для SQLite используем синхронное соединение в отдельном потоке
        if self._db_url.startswith('sqlite'):
            # Для асинхронной работы с SQLite используем aiosqlite
            self._async_db_url = self._db_url.replace('sqlite', 'sqlite+aiosqlite')
        else:
            # Для других БД используем асинхронные драйверы
            # Для PostgreSQL: sqlite+asyncpg, MySQL: mysql+aiomysql и т.д.
            self._async_db_url = self._db_url
        
        self.echo = echo
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_timeout = pool_timeout
        self.pool_recycle = pool_recycle
        self.connect_args = connect_args or {}
        
        # Проверяем, существует ли директория для SQLite
        if self._db_url.startswith('sqlite:///'):
            db_path = self._db_url.replace('sqlite:///', '')
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Инициализация асинхронного движка SQLAlchemy
        self.async_engine = create_async_engine(
            self._async_db_url,
            echo=self.echo,
            pool_size=self.pool_size,
            max_overflow=self.max_overflow,
            pool_timeout=self.pool_timeout,
            pool_recycle=self.pool_recycle,
            connect_args=self.connect_args
        )
        
        # Создание асинхронной сессии
        self.AsyncSession = async_sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.async_engine,
            expire_on_commit=False
        )
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Создает асинхронную сессию SQLAlchemy как контекстный менеджер.
        
        Yields:
            Асинхронная сессия SQLAlchemy
        """
        session = self.AsyncSession()
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            self.logger.error(f"Ошибка при работе с асинхронной сессией БД: {str(e)}")
            raise
        finally:
            await session.close()

    # Пример исправления для create_tables в AsyncDatabaseManager
    async def create_tables(self) -> None:
        """
        Асинхронно создает все таблицы в базе данных на основе моделей.
        """
        self.logger.info("Асинхронное создание таблиц в базе данных...")
        try:
            async with self.async_engine.begin() as conn:
                # Для SQLite включаем поддержку внешних ключей (если это необходимо для асинхронного драйвера)
                # Примечание: aiosqlite по умолчанию работает с foreign_keys=ON,
                # но можно добавить проверку или явное включение при необходимости
                # await conn.run_sync(self._set_sqlite_pragma_sync) # Пример, если нужна прагма

                await conn.run_sync(Base.metadata.create_all)
            self.logger.info("Таблицы успешно созданы")
        except Exception as e:
            self.logger.error(f"Ошибка при асинхронном создании таблиц: {str(e)}")
            raise

    # Аналогично для drop_tables:
    async def drop_tables(self) -> None:
        """
        Асинхронно удаляет все таблицы из базы данных.
        """
        self.logger.warning("Асинхронное удаление всех таблиц из базы данных...")
        try:
            async with self.async_engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
            self.logger.warning("Все таблицы удалены")
        except Exception as e:
            self.logger.error(f"Ошибка при асинхронном удалении таблиц: {str(e)}")
            raise

    # Аналогично для асинхронного метода
    async def execute_raw_sql(self, sql: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        params = params or {}
        try:
            async with self.get_session() as session:
                result = await session.execute(text(sql), params)
                if result.returns_rows:
                    # fetchall() может быть большим, рассмотреть потоковую обработку для больших результатов
                    return [dict(row._mapping) for row in result.fetchall()]  # Использовать _mapping
                return []
        except exc.SQLAlchemyError as e:
            self.logger.error(f"Ошибка при выполнении async raw SQL: {sql} с параметрами {params}. Ошибка: {str(e)}")
            raise

    @staticmethod
    async def get_table_names() -> List[str]:
        """
        Асинхронно возвращает список имен таблиц в базе данных.

        Returns:
            Список имен таблиц
        """
        return list(Base.metadata.tables.keys())

    async def check_connection(self) -> bool:
        """
        Асинхронно проверяет соединение с базой данных.
        
        Returns:
            True, если соединение успешно, иначе False
        """
        try:
            async with self.async_engine.connect() as connection:
                await connection.execute(text("SELECT 1"))
            return True
        except exc.SQLAlchemyError as e:
            self.logger.error(f"Ошибка соединения с базой данных: {str(e)}")
            return False
    
    async def get_database_info(self) -> Dict[str, Any]:
        """
        Асинхронно возвращает информацию о базе данных.
        
        Returns:
            Словарь с информацией о БД
        """
        table_names = await self.get_table_names()
        table_rows_count = {}
        
        async with self.get_session() as session:
            for table_name in table_names:
                try:
                    result = await session.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                    count = result.scalar()
                    table_rows_count[table_name] = count
                except Exception as e:
                    self.logger.error(f"Ошибка при подсчете строк в таблице {table_name}: {str(e)}")
                    table_rows_count[table_name] = "Error"
        
        connection_status = await self.check_connection()
        
        return {
            "url": self._async_db_url.split("@")[-1] if "@" in self._async_db_url else self._async_db_url,
            "tables": table_names,
            "rows_count": table_rows_count,
            "engine": str(self.async_engine.dialect.name),
            "is_connected": connection_status
        }
    
    async def close(self) -> None:
        """
        Асинхронно закрывает соединения с базой данных.
        """
        self.logger.info("Закрытие асинхронных соединений с базой данных...")
        await self.async_engine.dispose()
        self.logger.info("Асинхронные соединения с базой данных закрыты")
