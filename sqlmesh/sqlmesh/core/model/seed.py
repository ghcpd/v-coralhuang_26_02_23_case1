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
    def __init__(
        self,
        content: str,
        dialect: str,
        settings: CsvSettings,
        declared_columns: t.Optional[t.Iterable[str]] = None,
    ):
        """
        Csv seed reader that can optionally align CSV headers with declared columns.

        For dialects like Postgres, quoted identifiers are case-sensitive. The CSV headers
        are not quoted, so we need to preserve the header casing for any quoted columns
        declared in the model while still normalizing unquoted identifiers. To achieve
        this we accept the declared column names (as provided by the model definition)
        and use them to drive the renaming logic.
        """

        self.content = content
        self.dialect = dialect
        self.settings = settings
        # Preserve the raw declared column names (i.e., as specified in columns(...)).
        # We store them as a set for quick lookup. Empty set means "no declared columns".
        self._declared_columns: t.Set[str] = set(declared_columns or [])
        self._df: t.Optional[pd.DataFrame] = None

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

    def _normalized_declared_lookup(self) -> t.Dict[str, str]:
        """
        Builds a map of normalized declared column names -> declared names.

        If a declared column was quoted (e.g., "CamelCase"), the normalized name produced by
        sqlglot will still be "CamelCase". For unquoted identifiers the normalized name will
        be folded according to the dialect (lower-case for Postgres).
        """

        declared_lookup: t.Dict[str, str] = {}
        for declared in self._declared_columns:
            normalized = normalize_identifiers(declared, dialect=self.dialect).name
            declared_lookup[normalized] = declared
        return declared_lookup

    def _build_column_mapping(self, original_columns: t.Iterable[str]) -> t.Dict[str, str]:
        """
        Given original CSV headers, decide their final DataFrame column names.

        - Default: normalize headers using sqlglot (current behavior)
        - If a normalized header matches a declared column, rename to the declared column
          name to preserve casing for quoted identifiers (Postgres case sensitivity)
        """

        declared_lookup = self._normalized_declared_lookup()
        mapping: t.Dict[str, str] = {}

        for col in original_columns:
            normalized = normalize_identifiers(col, dialect=self.dialect).name
            # If the normalized header matches a declared column, use the declared
            # casing (preserves quoted identifiers). Otherwise keep the normalized form.
            mapping[col] = declared_lookup.get(normalized, normalized)

        return mapping

    def _get_df(self) -> pd.DataFrame:
        if self._df is None:
            self._df = pd.read_csv(
                StringIO(self.content),
                index_col=False,
                on_bad_lines="error",
                low_memory=False,
                **{k: v for k, v in self.settings.dict().items() if v is not None},
            )

            # IMPORTANT: we normalize headers, but if a normalized header matches a declared
            # column, we rename to the declared column name to preserve casing for quoted
            # identifiers (e.g., Postgres).
            mapping = self._build_column_mapping(self._df.columns)
            self._df = self._df.rename(columns=mapping)

        return self._df


class Seed(PydanticModel):
    """Represents content of a seed.

    Presently only CSV format is supported.
    """

    content: str

    def reader(
        self,
        dialect: str = "",
        settings: t.Optional[CsvSettings] = None,
        declared_columns: t.Optional[t.Iterable[str]] = None,
    ) -> CsvSeedReader:
        return CsvSeedReader(
            self.content,
            dialect,
            settings or CsvSettings(),
            declared_columns=declared_columns,
        )


def create_seed(path: str | Path) -> Seed:
    with open(Path(path), "r", encoding="utf-8") as fd:
        return Seed(content=fd.read())
