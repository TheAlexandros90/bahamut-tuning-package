# Publish checklist

## 1. Install build tooling

```bash
pip install -U build twine pytest
```

## 2. Run tests

```bash
pytest
```

## 3. Build the distributions

```bash
python -m build
```

## 4. Validate artifacts

```bash
twine check dist/*
```

## 5. Upload to TestPyPI first

```bash
twine upload --repository testpypi dist/*
```

## 6. Upload to PyPI

```bash
twine upload dist/*
```
