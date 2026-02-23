import pandas as pd
import pytest
from pathlib import Path
from sqlglot import exp

from sqlmesh.core import dialect as d
from sqlmesh.core.model import load_sql_based_model
from sqlmesh.core.model.kind import SeedKind


def test_seed_case_sensitive_columns_postgres(tmp_path):
    """Test seed with case-sensitive quoted columns in Postgres."""
    model_csv_path = (tmp_path / "model.csv").absolute()
    
    with open(model_csv_path, "w", encoding="utf-8") as fd:
        fd.write(
            """camelCaseId,camelCaseBool,camelCaseString,normalisedCaseDate,camelCaseTimestamp
1,false,Alice,2022-01-01,2022-01-01 00:00:00
2,true,Bob,2022-01-02,2022-01-02 12:30:00
"""
        )
    expressions = d.parse(
        f"""
        MODEL (
            name db.seed,
            dialect postgres,
            kind SEED (
              path '{str(model_csv_path)}',
            ),
            columns (
              "camelCaseId" int,
              "camelCaseBool" boolean,
              "camelCaseString" text,
              "camelCaseTimestamp" timestamp
            )
        );
        """
    )

    model = load_sql_based_model(
        expressions,
        path=Path("./test_model.sql"),
        dialect="postgres"
    )

    assert isinstance(model.kind, SeedKind)
    assert model.seed is not None
    assert len(model.seed.content) > 0
    
    assert model.columns_to_types == {
        "camelCaseId": exp.DataType.build("int"),
        "camelCaseBool": exp.DataType.build("boolean"),
        "camelCaseString": exp.DataType.build("text"),
        "camelCaseTimestamp": exp.DataType.build("TIMESTAMP"),
    }
    
    df = next(model.render(context=None))
    
    assert "camelCaseId" in df.columns
    assert "camelCaseBool" in df.columns
    assert "camelCaseString" in df.columns
    assert "camelCaseTimestamp" in df.columns
    
    # Use pandas API type checkers for version compatibility
    assert pd.api.types.is_integer_dtype(df["camelCaseId"])
    assert pd.api.types.is_bool_dtype(df["camelCaseBool"])
    assert pd.api.types.is_string_dtype(df["camelCaseString"])
    assert pd.api.types.is_datetime64_any_dtype(df["camelCaseTimestamp"])
    
    assert df["camelCaseId"].iloc[0] == 1
    assert df["camelCaseBool"].iloc[0] == False
    assert df["camelCaseString"].iloc[0] == "Alice"
    assert df["camelCaseTimestamp"].iloc[0] == pd.Timestamp("2022-01-01 00:00:00")
    
    assert df["camelCaseId"].iloc[1] == 2
    assert df["camelCaseBool"].iloc[1] == True
    assert df["camelCaseString"].iloc[1] == "Bob"
    assert df["camelCaseTimestamp"].iloc[1] == pd.Timestamp("2022-01-02 12:30:00")
    
    assert "normalisedcasedate" in df.columns
    assert df["normalisedcasedate"].iloc[0] == "2022-01-01"

    # column_hashes are computed on the final, aligned names as well
    hashes = model.column_hashes
    assert "camelCaseId" in hashes
    assert "camelCaseTimestamp" in hashes
    assert "normalisedcasedate" in hashes


def test_seed_case_sensitive_partial_columns(tmp_path):
    """Test declared and undeclared columns with case sensitivity."""
    model_csv_path = (tmp_path / "partial.csv").absolute()
    
    with open(model_csv_path, "w", encoding="utf-8") as fd:
        fd.write(
            """DeclaredColumn,UndeclaredColumn,AnotherDeclared
1,value1,text1
2,value2,text2
"""
        )

    expressions = d.parse(
        f"""
        MODEL (
            name db.partial_seed,
            dialect postgres,
            kind SEED (
              path '{str(model_csv_path)}',
            ),
            columns (
              "DeclaredColumn" int,
              "AnotherDeclared" text
            )
        );
        """
    )

    model = load_sql_based_model(
        expressions,
        path=Path("./partial_test.sql"),
        dialect="postgres"
    )

    df = next(model.render(context=None))
    
    assert "DeclaredColumn" in df.columns
    assert "AnotherDeclared" in df.columns
    
    assert "undeclaredcolumn" in df.columns
    assert "UndeclaredColumn" not in df.columns
    
    assert df["DeclaredColumn"].iloc[0] == 1
    assert df["AnotherDeclared"].iloc[0] == "text1"
    assert df["undeclaredcolumn"].iloc[0] == "value1"


def test_seed_case_sensitive_mixed_quotes(tmp_path):
    """Test unquoted columns in model definition are normalized to match CSV."""
    model_csv_path = (tmp_path / "mixed.csv").absolute()
    
    with open(model_csv_path, "w", encoding="utf-8") as fd:
        fd.write(
            """MixedCase,lowercase,UPPERCASE
1,2,3
"""
        )

    expressions = d.parse(
        f"""
        MODEL (
            name db.mixed_seed,
            dialect postgres,
            kind SEED (
              path '{str(model_csv_path)}',
            ),
            columns (
              "MixedCase" int,
              lowercase int,
              UPPERCASE int
            )
        );
        """
    )

    model = load_sql_based_model(
        expressions,
        path=Path("./mixed_test.sql"),
        dialect="postgres"
    )

    df = next(model.render(context=None))
    
    assert "MixedCase" in df.columns
    
    assert "lowercase" in df.columns
    assert "uppercase" in df.columns
    
    assert df["MixedCase"].iloc[0] == 1
    assert df["lowercase"].iloc[0] == 2
    assert df["uppercase"].iloc[0] == 3
