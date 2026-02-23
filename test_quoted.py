from sqlglot import exp, parse
from sqlglot.optimizer.normalize_identifiers import normalize_identifiers

# Test quoted identifiers
col = exp.column('"camelCaseId"')
print(f'col={col}')
print(f'col.name={col.name}')
print(f'col.this={col.this}')
if hasattr(col.this, 'quoted'):
    print(f'col.this.quoted={col.this.quoted}')
print()

# Test what normalize_identifiers does
normalized = normalize_identifiers(col, dialect='postgres')
print(f'normalized={normalized}')
print(f'normalized.name={normalized.name}')
if hasattr(normalized.this, 'quoted'):
    print(f'normalized.this.quoted={normalized.this.quoted}')
print()

# Test unquoted
col2 = exp.column('camelCaseId')
print(f'col2={col2}')
if hasattr(col2.this, 'quoted'):
    print(f'col2.this.quoted={col2.this.quoted}')
normalized2 = normalize_identifiers(col2, dialect='postgres')
print(f'normalized2={normalized2}')
print(f'normalized2.name={normalized2.name}')
print()

# Test ColumnDef
col_def = exp.ColumnDef(this=exp.Identifier(this='camelCaseId', quoted=True), kind=exp.DataType.build('INT'))
print(f'col_def={col_def}')
print(f'col_def.name={col_def.name}')
print(f'col_def.this={col_def.this}')
print(f'col_def.this.quoted={col_def.this.quoted}')

normalized_col_def = normalize_identifiers(col_def, dialect='postgres')
print(f'normalized_col_def={normalized_col_def}')
print(f'normalized_col_def.this.quoted={normalized_col_def.this.quoted}')
