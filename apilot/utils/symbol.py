from apilot.core.constant import Exchange


def split_symbol(symbol: str) -> tuple[str, str]:
    """Split trading symbol into base symbol and exchange string"""
    parts = symbol.split(".")
    if len(parts) != 2:
        raise ValueError(f"Invalid symbol format: {symbol}")
    return parts[0], parts[1]


def get_exchange(symbol: str) -> Exchange:
    """Get exchange object from full trading symbol"""
    _, exchange_str = split_symbol(symbol)
    return Exchange(exchange_str)


def get_base_symbol(symbol: str) -> str:
    """Get base symbol from full trading symbol"""
    base_symbol, _ = split_symbol(symbol)
    return base_symbol
