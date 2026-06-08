"""Validate a new Excel file uploaded by the user before pipeline processing."""
from pathlib import Path

import pandas as pd

from config import (
    CRITICAL_COLUMNS,
    REQUIRED_COLUMNS,
    VALID_PLANTS,
    REQUIRE_ALL_PLANTS,
    MAX_NULL_RATIO,
)


class ValidationError(Exception):
    """Raised when the uploaded file fails validation."""
    pass


def validate_new_data(file_path: str | Path) -> dict:
    """
    Validate a new Excel file before feeding it into the pipeline.

    Returns:
        {
            "valid": bool,
            "message": str,
            "rows": int,
            "plants": list[int],
            "null_rows_removed": int,
        }
    """
    path = Path(file_path)

    # 1. File exists
    if not path.exists():
        raise ValidationError(f"Archivo no encontrado: {path}")

    # 2. Read Excel
    try:
        df = pd.read_excel(path)
    except Exception as e:
        raise ValidationError(f"No se pudo leer el Excel: {e}")

    if df.empty:
        raise ValidationError("El archivo está vacío.")

    # 3. Check required columns exist
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValidationError(
            f"Columnas faltantes: {missing_cols}. "
            f"Requeridas: {REQUIRED_COLUMNS}"
        )

    # 4. Check parseable datetime in start_time
    try:
        df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
    except Exception as e:
        raise ValidationError(f"start_time no es parseable como fecha: {e}")

    unparsable = df["start_time"].isna().sum()
    if unparsable > 0:
        raise ValidationError(
            f"{unparsable} filas tienen start_time inválido."
        )

    # 5. Check plant codes are valid
    invalid_plants = df[~df["ship_plant_code"].isin(VALID_PLANTS)]
    if not invalid_plants.empty:
        bad = invalid_plants["ship_plant_code"].unique().tolist()
        raise ValidationError(
            f"Códigos de planta inválidos: {bad}. "
            f"Válidos: {VALID_PLANTS}"
        )

    # 6. Check volume >= 0
    negative_vol = (df["u_Volumen"] < 0).sum()
    if negative_vol > 0:
        raise ValidationError(
            f"{negative_vol} filas tienen u_Volumen negativo."
        )

    # 7. Check nulls in critical columns
    null_mask = df[CRITICAL_COLUMNS].isna().any(axis=1)
    null_count = null_mask.sum()
    null_ratio = null_count / len(df)

    if null_ratio > MAX_NULL_RATIO:
        raise ValidationError(
            f"Demasiados nulls en columnas críticas: "
            f"{null_count}/{len(df)} ({null_ratio:.1%}). "
            f"Máximo permitido: {MAX_NULL_RATIO:.0%}."
        )

    # Remove null rows and report
    df_clean = df[~null_mask].copy()
    removed = null_count

    # 8. Check all plants present
    present_plants = sorted(df_clean["ship_plant_code"].unique())
    if REQUIRE_ALL_PLANTS:
        missing_plants = [p for p in VALID_PLANTS if p not in present_plants]
        if missing_plants:
            raise ValidationError(
                f"Faltan plantas en el dataset: {missing_plants}. "
                f"Requeridas: {VALID_PLANTS}"
            )

    return {
        "valid": True,
        "message": "Validación exitosa.",
        "rows": len(df_clean),
        "plants": present_plants,
        "null_rows_removed": removed,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        result = validate_new_data(sys.argv[1])
        print(result)
    else:
        print("Uso: python validate.py <archivo.xlsx>")
