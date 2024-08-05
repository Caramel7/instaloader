import base64
import hashlib
import json
import os
from contextlib import contextmanager
from datetime import datetime, timedelta
from lzma import LZMAError
from typing import Any, Callable, Dict, Iterable, Iterator, NamedTuple, Optional, Tuple, TypeVar, Generic

from .exceptions import AbortDownloadException, InvalidArgumentException, QueryReturnedBadRequestException
from .instaloadercontext import InstaloaderContext

T = TypeVar('T')

class FrozenNodeIterator(NamedTuple):
    query_hash: str
    query_variables: Dict[str, Any]
    query_referer: Optional[str]
    context_username: Optional[str]
    total_index: int
    best_before: Optional[float]
    remaining_data: Optional[Dict[str, Any]]
    first_node: Optional[Dict[str, Any]]

class NodeIterator(Generic[T], Iterator[T]):
    _GRAPHQL_PAGE_LENGTH = 12
    _SHELF_LIFE = timedelta(days=29)

    def __init__(
        self,
        context: InstaloaderContext,
        query_hash: str,
        edge_extractor: Callable[[Dict[str, Any]], Dict[str, Any]],
        node_wrapper: Callable[[Dict[str, Any]], T],
        query_variables: Optional[Dict[str, Any]] = None,
        query_referer: Optional[str] = None,
        first_data: Optional[Dict[str, Any]] = None,
        is_first: Optional[Callable[[T, Optional[T]], bool]] = None
    ):
        self._context = context
        self._query_hash = query_hash
        self._edge_extractor = edge_extractor
        self._node_wrapper = node_wrapper
        self._query_variables = query_variables or {}
        self._query_referer = query_referer
        self._page_index = 0
        self._total_index = 0
        self._data = first_data or self._query()
        self._best_before = datetime.now() + self._SHELF_LIFE
        self._first_node: Optional[Dict[str, Any]] = None
        self._is_first = is_first

    def _query(self, after: Optional[str] = None) -> Dict[str, Any]:
        pagination_variables = {'first': self._GRAPHQL_PAGE_LENGTH, 'after': after} if after else {'first': self._GRAPHQL_PAGE_LENGTH}
        try:
            data = self._edge_extractor(
                self._context.graphql_query(
                    self._query_hash,
                    {**self._query_variables, **pagination_variables},
                    self._query_referer
                )
            )
            self._best_before = datetime.now() + self._SHELF_LIFE
            return data
        except QueryReturnedBadRequestException:
            return self._handle_bad_request(after)

    def _handle_bad_request(self, after: Optional[str]) -> Dict[str, Any]:
        new_page_length = self._GRAPHQL_PAGE_LENGTH // 2
        if new_page_length >= 12:
            self._GRAPHQL_PAGE_LENGTH = new_page_length
            self._context.error("HTTP Error 400 (Bad Request) on GraphQL Query. Retrying with shorter page length.", repeat_at_end=False)
            return self._query(after)
        raise

    def __next__(self) -> T:
        if self._page_index < len(self._data['edges']):
            return self._process_next_item()
        if self._data.get('page_info', {}).get('has_next_page'):
            return self._fetch_next_page()
        raise StopIteration()

    def _process_next_item(self) -> T:
        node = self._data['edges'][self._page_index]['node']
        self._page_index += 1
        self._total_index += 1
        item = self._node_wrapper(node)
        self._update_first_node(item, node)
        return item

    def _update_first_node(self, item: T, node: Dict[str, Any]) -> None:
        if self._is_first:
            if self._is_first(item, self.first_item):
                self._first_node = node
        elif self._first_node is None:
            self._first_node = node

    def _fetch_next_page(self) -> T:
        query_response = self._query(self._data['page_info']['end_cursor'])
        if self._data['edges'] != query_response['edges'] and query_response['edges']:
            self._page_index = 0
            self._data = query_response
            return self.__next__()
        raise StopIteration()

    @property
    def count(self) -> Optional[int]:
        return self._data.get('count') if self._data is not None else None

    @property
    def total_index(self) -> int:
        return self._total_index

    @property
    def magic(self) -> str:
        magic_hash = hashlib.blake2b(digest_size=6)
        magic_hash.update(json.dumps([self._query_hash, self._query_variables, self._query_referer, self._context.username]).encode())
        return base64.urlsafe_b64encode(magic_hash.digest()).decode()

    @property
    def first_item(self) -> Optional[T]:
        return self._node_wrapper(self._first_node) if self._first_node is not None else None

    @staticmethod
    def page_length() -> int:
        return NodeIterator._GRAPHQL_PAGE_LENGTH

    def freeze(self) -> FrozenNodeIterator:
        remaining_data = None
        if self._data is not None:
            remaining_data = {**self._data, 'edges': self._data['edges'][max(self._page_index - 1, 0):]}
        return FrozenNodeIterator(
            query_hash=self._query_hash,
            query_variables=self._query_variables,
            query_referer=self._query_referer,
            context_username=self._context.username,
            total_index=max(self.total_index - 1, 0),
            best_before=self._best_before.timestamp() if self._best_before else None,
            remaining_data=remaining_data,
            first_node=self._first_node,
        )

    def thaw(self, frozen: FrozenNodeIterator) -> None:
        self._validate_thaw(frozen)
        self._total_index = frozen.total_index
        self._best_before = datetime.fromtimestamp(frozen.best_before)
        self._data = frozen.remaining_data
        self._first_node = frozen.first_node

    def _validate_thaw(self, frozen: FrozenNodeIterator) -> None:
        if self._total_index or self._page_index:
            raise InvalidArgumentException("thaw() called on already-used iterator.")
        if (self._query_hash != frozen.query_hash or
            self._query_variables != frozen.query_variables or
            self._query_referer != frozen.query_referer or
            self._context.username != frozen.context_username):
            raise InvalidArgumentException("Mismatching resume information.")
        if not frozen.best_before:
            raise InvalidArgumentException("\"best before\" date missing.")
        if frozen.remaining_data is None:
            raise InvalidArgumentException("\"remaining_data\" missing.")

@contextmanager
def resumable_iteration(
    context: InstaloaderContext,
    iterator: Iterable,
    load: Callable[[InstaloaderContext, str], Any],
    save: Callable[[FrozenNodeIterator, str], None],
    format_path: Callable[[str], str],
    check_bbd: bool = True,
    enabled: bool = True
) -> Iterator[Tuple[bool, int]]:
    if not enabled or not isinstance(iterator, NodeIterator):
        yield False, 0
        return

    resume_file_path = format_path(iterator.magic)
    is_resuming, start_index = load_resume_info(context, iterator, resume_file_path, load, check_bbd)

    try:
        yield is_resuming, start_index
    except (KeyboardInterrupt, AbortDownloadException):
        save_resume_info(context, iterator, resume_file_path, save)
        raise
    finally:
        if os.path.exists(resume_file_path):
            os.unlink(resume_file_path)
            context.log(f"Iteration complete, deleted resume information file {resume_file_path}.")

def load_resume_info(
    context: InstaloaderContext,
    iterator: NodeIterator,
    resume_file_path: str,
    load: Callable[[InstaloaderContext, str], Any],
    check_bbd: bool
) -> Tuple[bool, int]:
    if not os.path.isfile(resume_file_path):
        return False, 0

    try:
        fni = load(context, resume_file_path)
        if not isinstance(fni, FrozenNodeIterator):
            raise InvalidArgumentException("Invalid type.")
        if check_bbd and fni.best_before and datetime.fromtimestamp(fni.best_before) < datetime.now():
            raise InvalidArgumentException("\"Best before\" date exceeded.")
        iterator.thaw(fni)
        context.log(f"Resuming from {resume_file_path}.")
        return True, iterator.total_index
    except (InvalidArgumentException, LZMAError, json.decoder.JSONDecodeError, EOFError) as exc:
        context.error(f"Warning: Not resuming from {resume_file_path}: {exc}")
        return False, 0

def save_resume_info(
    context: InstaloaderContext,
    iterator: NodeIterator,
    resume_file_path: str,
    save: Callable[[FrozenNodeIterator, str], None]
) -> None:
    os.makedirs(os.path.dirname(resume_file_path), exist_ok=True)
    save(iterator.freeze(), resume_file_path)
    context.log(f"\nSaved resume information to {resume_file_path}.")
