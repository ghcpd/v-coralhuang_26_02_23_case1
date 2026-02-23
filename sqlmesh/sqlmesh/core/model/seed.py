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




# Helper utilities ----------------------------------------------------------

def _seed_column_mapping(
    original_columns: t.List[str],
    normalized_columns: t.List[str],
    dialect: str,
    model_columns: t.Optional[t.Set[str]] = None,
) -> t.Dict[str, str]:
    """Return a mapping of *normalized* column names to their desired final names.

    The CSV reader initially normalizes every header using unquoted SQL identifier
    rules (e.g. lowercasing for Postgres).  This routine takes the original header
    value and the already-normalized value and decides how to rename it based on
    the set of columns that the model declared.

    Rules:
      * If the model explicitly declares a column that matches the quoted form of
        the original header, prefer that exact casing (quoted identifiers are
        case-sensitive in Postgres).
      * Otherwise, fall back to the unquoted normalization (this lowercases for
        Postgres and is the behaviour prior to this bug fix).  Undeclared columns
        are treated the same way as unquoted model columns.

    The returned dictionary can be fed directly to ``DataFrame.rename`` or used
    to rewrite column-hash keys.
    """
    mapping: t.Dict[str, str] = {}

    # make a set for faster membership tests
    model_cols = set(model_columns) if model_columns is not None else None

    for orig, norm in zip(original_columns, normalized_columns):
        # for Postgres, quoting preserves case; use normalize_identifiers on a
        # quoted string to compute that form.
        quoted_norm = normalize_identifiers(f'"{orig}"', dialect=dialect).name
        unquoted_norm = norm

        if model_cols is not None and quoted_norm in model_cols:
            # user declared this column with explicit quotes; keep case exactly
            mapping[norm] = quoted_norm
        else:
            # either the column is unquoted in the model or undeclared; use the
            # unquoted normalization which matches previous behaviour
            mapping[norm] = unquoted_norm
    return mapping


class CsvSeedReader:
    def __init__(self, content: str, dialect: str, settings:CsvSettings):
        self.content = content
        self.dialect = dialect
        self.settings = settings
        # raw dataframe read from CSV; headers may be normalized below
        self._df: t.Optional[pd.DataFrame] = None
        # keep the original CSV header names so we can later reconcile
        # against model declarations (for case-sensitive, quoted names)
        self._original_columns: t.Optional[t.List[str]] = None

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

    def _get_df(self) -> pd.DataFrame:
        if self._df is None:
            # read the file but keep headers exactly as they appear
            df = pd.read_csv(
                StringIO(self.content),
                index_col=False,
                on_bad_lines="error",
                low_memory=False,
                **{k: v for k, v in self.settings.dict().items() if v is not None},
            )
            # stash the original header names for later use
            self._original_columns = list(df.columns)
            # normalize identifiers using unquoted semantics so downstream
            # logic that doesn't know about quoted names continues to work as
            # before (lowercase for postgres, etc).  We'll perform a second,
            # context-aware pass in the model layer when rendering so that
            # quoted columns can be restored.
            df = df.rename(
                columns={
                    col: normalize_identifiers(col, dialect=self.dialect).name
                    for col in df.columns
                },
            )
            self._df = df

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
