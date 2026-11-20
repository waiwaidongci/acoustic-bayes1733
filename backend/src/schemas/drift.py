import math

from pydantic import BaseModel, Field, field_validator


class ShiftInput(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    order: int = Field(ge=1, le=1000000)
    references: list[str] = Field(min_length=1, max_length=100000)


class PlanInput(BaseModel):
    source_id: int
    note: str = Field(default='', max_length=2000)
    shifts: list[ShiftInput] = Field(min_length=1, max_length=100000)


class JobInput(BaseModel):
    run_id: int
    label: str = Field(min_length=1, max_length=120)
    penalty: float
    plan_id: int | None = None

    @field_validator('penalty')
    @classmethod
    def penalty_must_be_finite_and_non_negative(cls, value):
        if not math.isfinite(value) or value < 0:
            raise ValueError('罚分必须是非负有限数值')
        return value
