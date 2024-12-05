from collections.abc import Mapping
from typing import TypedDict, Any, TypeVar


class Stuff(TypedDict):
    a: dict[int, Any]
    b: str


def foo(s: Mapping) -> None:
    print(s["a"])
    s["a"]["5"] = 10


T = TypeVar("T", bound=Mapping)


class Bar[T]:
    s: T

    def baz(self):
        print(self.s["a"])
        self.s["a"]["5"] = 10


s = Stuff(a={}, b="a")

foo(s)
