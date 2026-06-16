"""Valida que todas as dependências estão instaladas corretamente."""

import importlib
import sys

REQUIRED = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("deap", "deap"),
    ("transformers", "transformers"),
    ("torch", "torch"),
    ("dotenv", "python-dotenv"),
    ("folium", "folium"),
    ("geopy", "geopy"),
    ("networkx", "networkx"),
    ("plotly", "plotly"),
    ("pytest", "pytest"),
]

ok = True
for module, package in REQUIRED:
    try:
        lib = importlib.import_module(module)
        version = getattr(lib, "__version__", "n/a")
        print(f"  {package:<20} {version}")
    except ImportError:
        print(f"  {package:<20} NAO ENCONTRADO")
        ok = False

if ok:
    print("\nAmbiente OK — todas as dependencias instaladas.")
else:
    print("\nERRO — execute: pip install -r requirements.txt")
    sys.exit(1)
