from __future__ import annotations

import typing as t
import zlib
from io import StringIO
from pathlib import Path

import pandas as pd
from sqlglot import exp
from sqlglot.dialects.dialect import UNESCAPED_SEQUENCES
from sqlglot.optimizer.normalize_identifiers import normalize_identifiers

from sqlmesh.core.model.common import parse_bool
from sqlmesh.utils.pandas import columns_to_types_from_df
from sqlmesh.utils.pydantic import PydanticModel, field_validator


class CsvSettings(PydanticModel):
    """Settings for CSV seeds."""

    delimiter: t.Optional[str] = None
    quotechar: t.Optional[str] = None
    doublequote: t.Optional[bool] = None
    escapechar: t.Optional[str] = None
    skipinitialspace: t.Optional[bool] = None
    lineterminator: t.Optional[str] = None
    encoding: t.Optional[str] = None

    @field_validator("doublequote", "skipinitialspace", mode="before")
    @classmethod
    def _bool_validator(cls, v: t.Any) -> t.Optional[bool]:
        if v is None:
            return v
        return parse_bool(v)

    @field_validator(
        "delimiter", "quotechar", "escapechar", "lineterminator", "encoding", mode="before"
    )
    @classmethod
    def _str_validator(cls, v: t.Any) -> t.Optional[str]:
        if v is None or not isinstance(v, exp.Expression):
            return v

        # SQLGlot parses escape sequences like \t as \\t for dialects that don't treat \ as
        # an escape character, so we map them back to the corresponding escaped sequence
        v = v.this
        return UNESCAPED_SEQUENCES.get(v, v)


class CsvSeedReader:
    def __init__(self, content: str, dialect: str, settings: CsvSettings):
        self.content = content
        self.dialect = dialect
        self.settings = settings
        self._df: t.Optional[pd.DataFrame] = None
        self._expected_columns: t.Optional[t.Dict[str, str]] = None

    @property
    def columns_to_types(self) -> t.Dict[str, exp.DataType]:
        return columns_to_types_from_df(self._get_df())

    @property
    def column_hashes(self) -> t.Dict[str, str]:
        df = self._get_df()
        return {
            column_name: str(zlib.crc32(df[column_name].to_json().encode("utf-8")))
            for column_name in df.columns
        }

    def read(self, batch_size: t.Optional[int] = None) -> t.Generator[pd.DataFrame, None, None]:
        df = self._get_df()

        batch_size = batch_size or df.size
        batch_start = 0
        while batch_start < df.shape[0]:
            yield df.iloc[batch_start : batch_start + batch_size, :]
            batch_start += batch_size

    def set_expected_columns(self, expected_columns: t.Dict[str, str]) -> None:
        """Set the expected column mapping from CSV names to model column names.
        
        This is used to handle case-sensitive quoted identifiers in the model definition.
        
        Args:
            expected_columns: A dict mapping CSV column names to expected model column names
        """
        self._expected_columns = expected_columns
        # Reset the cached DataFrame so it gets re-created with new column mappings
        self._df = None

    def _get_df(self) -> pd.DataFrame:
        if self._df is None:
            self._df = pd.read_csv(
                StringIO(self.content),
                index_col=False,
                on_bad_lines="error",
                low_memory=False,
                **{k: v for k, v in self.settings.dict().items() if v is not None},
            )
            
            # Create column name mapping
            if self._expected_columns:
                # Use the provided mapping
                rename_map = self._expected_columns
            else:
                # Default behavior: normalize all columns
                rename_map = {
                    col: normalize_identifiers(col, dialect=self.dialect).name
                    for col in self._df.columns
                }
            
            self._df = self._df.rename(columns=rename_map)

        return self._df


class Seed(PydanticModel):
    """Represents content of a seed.

    Presently only CSV format is supported.
    """

    content: str

    def reader(self, dialect: str = "", settings: t.Optional[CsvSettings] = None) -> CsvSeedReader:
        return CsvSeedReader(self.content, dialect, settings or CsvSettings())


def create_seed(path: str | Path) -> Seed:
    with open(Path(path), "r", encoding="utf-8") as fd:
        return Seed(content=fd.read())
