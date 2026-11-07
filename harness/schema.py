"""Еталонний контракт відповіді. Саме цю схему використовує перевірка (check.py і фінальний прогін).

Не змінюйте цей файл: фінальний прогін використовує власну копію.
"""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

CATEGORIES = ("delivery", "payment", "returns", "technical", "product", "other")
PRIORITIES = ("low", "medium", "high", "urgent")


class TriageResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    category: Literal["delivery", "payment", "returns", "technical", "product", "other"]
    priority: Literal["low", "medium", "high", "urgent"]
    needs_human: bool
    language: str = Field(pattern=r"^[a-z]{2}$")
    summary: str = Field(min_length=1, max_length=200)


def validate_output(obj):
    """Повертає (ok, error_text). Приймає dict (або pydantic-модель, яку перетворює на dict)."""
    if isinstance(obj, BaseModel):
        obj = obj.model_dump()
    if not isinstance(obj, dict):
        return False, f"очікувався dict, отримано {type(obj).__name__}"
    try:
        TriageResult.model_validate(obj)
    except Exception as e:  # pydantic.ValidationError
        msg = str(e).splitlines()
        return False, " | ".join(msg[:4])[:300]
    return True, None
